use crate::{
    poly::{BoxMultilinearPolyOwned, DenseMultilinearPoly, MultilinearPoly},
    transcript::{TranscriptRead, TranscriptWrite},
    util::{
        arithmetic::{horner, ExtensionField, Field},
        Itertools,
    },
    Error,
};
use rayon::prelude::*;
use std::{convert::Infallible, fmt::Debug, marker::PhantomData};

pub mod eq_f;
pub mod generic;
pub mod quadratic;

pub fn prove_sum_check<F, E, P, PE>(
    g: &impl SumCheckFunction<F, E>,
    claim: E,
    polys: impl IntoIterator<Item = SumCheckPoly<F, E, P, PE>>,
    transcript: &mut (impl TranscriptWrite<F, E> + ?Sized),
) -> Result<(E, Vec<E>, Vec<E>), Error>
where
    F: Field,
    E: ExtensionField<F>,
    P: MultilinearPoly<F, E>,
    PE: MultilinearPoly<E, E>,
{
    let num_vars = g.num_vars();
    assert!(num_vars > 0);

    let degree = g.degree();
    assert!(degree >= 1);

    let mut claim = claim;
    let mut polys = Polys::new(num_vars, polys);
    let mut r = Vec::with_capacity(num_vars);
    for round in 0..num_vars {
        let sum = if round == 0 {
            g.compute_sum(round, claim, &polys.borrowed)
        } else {
            g.compute_sum(round, claim, &polys.owned())
        };

        g.write_sum(round, &sum, transcript)?;
        assert_eq!(sum.len(), degree + 1);

        let r_i = transcript.squeeze_challenge();

        claim = horner(&sum, &r_i);
        polys.fix_var(&r_i);
        r.push(r_i);
    }

    Ok((claim, r, polys.into_evals()))
}

pub fn verify_sum_check<F: Field, E: ExtensionField<F>>(
    g: &impl SumCheckFunction<F, E>,
    claim: E,
    transcript: &mut (impl TranscriptRead<F, E> + ?Sized),
) -> Result<(E, Vec<E>), Error> {
    let num_vars = g.num_vars();
    assert!(num_vars > 0);

    let degree = g.degree();
    assert!(degree >= 1);

    let mut claim = claim;
    let mut r = Vec::with_capacity(num_vars);
    for round in 0..num_vars {
        let sum = g.read_sum(round, claim, transcript)?;
        assert_eq!(sum.len(), degree + 1);

        let r_i = transcript.squeeze_challenge();

        claim = horner(&sum, &r_i);
        r.push(r_i);
    }

    Ok((claim, r))
}

pub fn err_unmatched_evaluation() -> Error {
    Error::InvalidSumCheck("Unmatched evaluation from SumCheck subclaim".to_string())
}

pub trait SumCheckFunction<F, E>: Debug + Send + Sync {
    fn num_vars(&self) -> usize;

    fn degree(&self) -> usize;

    fn compute_sum(
        &self,
        round: usize,
        claim: E,
        polys: &[SumCheckPoly<F, E, impl MultilinearPoly<F, E>, impl MultilinearPoly<E, E>>],
    ) -> Vec<E>;

    fn write_sum(
        &self,
        round: usize,
        sum: &[E],
        transcript: &mut (impl TranscriptWrite<F, E> + ?Sized),
    ) -> Result<(), Error>;

    fn read_sum(
        &self,
        round: usize,
        claim: E,
        transcript: &mut (impl TranscriptRead<F, E> + ?Sized),
    ) -> Result<Vec<E>, Error>;
}

pub enum SumCheckPoly<F, E, P, PE> {
    Base(P),
    Extension(PE),
    Unreachable(Infallible, PhantomData<(F, E)>),
}

#[allow(clippy::len_without_is_empty)]
impl<F, E, P, PE> SumCheckPoly<F, E, P, PE>
where
    P: MultilinearPoly<F, E>,
    PE: MultilinearPoly<E, E>,
{
    pub fn num_vars(&self) -> usize {
        op_sum_check_poly!(self, |a| a.num_vars())
    }

    pub fn len(&self) -> usize {
        op_sum_check_poly!(self, |a| a.len())
    }

    pub fn fix_var(&self, x_i: &E) -> BoxMultilinearPolyOwned<'static, E> {
        op_sum_check_poly!(self, |a| a.fix_var(x_i))
    }

    pub fn op<T>(
        a: &Self,
        b: &Self,
        bb: &impl Fn(&P, &P) -> T,
        be: &impl Fn(&PE, &P) -> T,
        ee: &impl Fn(&PE, &PE) -> T,
    ) -> T {
        match (a, b) {
            (Self::Base(a), Self::Base(b)) => bb(a, b),
            (Self::Base(b), Self::Extension(a)) | (Self::Extension(a), Self::Base(b)) => be(a, b),
            (Self::Extension(a), Self::Extension(b)) => ee(a, b),
            _ => unreachable!(),
        }
    }
}

impl<F, E, PE> SumCheckPoly<F, E, DenseMultilinearPoly<F, Vec<F>>, PE>
where
    PE: MultilinearPoly<E, E>,
{
    pub fn exts(polys: impl IntoIterator<Item = PE>) -> Vec<Self> {
        polys.into_iter().map(Self::Extension).collect()
    }
}

impl<F, E, P> SumCheckPoly<F, E, P, DenseMultilinearPoly<E, Vec<E>>>
where
    P: MultilinearPoly<F, E>,
{
    pub fn bases(polys: impl IntoIterator<Item = P>) -> Vec<Self> {
        polys.into_iter().map(Self::Base).collect()
    }
}

#[macro_export]
macro_rules! op_sum_check_polys {
    (|$a:ident, $b:ident| $op:expr, |$bb_out:ident| $op_bb_out:expr) => {
        SumCheckPoly::op(
            $a,
            $b,
            &|a, b| {
                let $a = a;
                let $b = b;
                let $bb_out = $op;
                $op_bb_out
            },
            &|a, b| {
                let $a = a;
                let $b = b;
                $op
            },
            &|a, b| {
                let $a = a;
                let $b = b;
                $op
            },
        )
    };
}

#[macro_export]
macro_rules! op_sum_check_poly {
    ($a:ident, |$tmp_a:ident| $op:expr, |$b_out:ident| $op_b_out:expr) => {
        match $a {
            SumCheckPoly::Base(a) => {
                let $tmp_a = a;
                let $b_out = $op;
                $op_b_out
            }
            SumCheckPoly::Extension(a) => {
                let $tmp_a = a;
                $op
            }
            _ => unreachable!(),
        }
    };
    ($a:ident, |$tmp_a:ident| $op:expr) => {
        op_sum_check_poly!($a, |$tmp_a| $op, |out| out)
    };
    (|$a:ident| $op:expr, |$b_out:ident| $op_b_out:expr) => {
        op_sum_check_poly!($a, |$a| $op, |$b_out| $op_b_out)
    };
    (|$a:ident| $op:expr) => {
        op_sum_check_poly!(|$a| $op, |out| out)
    };
}

pub use {op_sum_check_poly, op_sum_check_polys};

struct Polys<F, E, P, PE> {
    borrowed: Vec<SumCheckPoly<F, E, P, PE>>,
    owned: Vec<BoxMultilinearPolyOwned<'static, E>>,
    _marker: PhantomData<F>,
}

impl<F, E, P, PE> Polys<F, E, P, PE>
where
    F: Field,
    E: ExtensionField<F>,
    P: MultilinearPoly<F, E>,
    PE: MultilinearPoly<E, E>,
{
    fn new(num_vars: usize, polys: impl IntoIterator<Item = SumCheckPoly<F, E, P, PE>>) -> Self {
        assert!(num_vars > 0);

        let polys = polys.into_iter().collect_vec();
        assert!(!polys.is_empty());
        assert!(!polys.iter().any(|poly| poly.num_vars() != num_vars));

        Self {
            borrowed: polys,
            owned: Vec::new(),
            _marker: PhantomData,
        }
    }

    fn owned(&self) -> Vec<SumCheckPoly<F, E, P, &BoxMultilinearPolyOwned<'static, E>>> {
        self.owned.iter().map(SumCheckPoly::Extension).collect()
    }

    fn fix_var(&mut self, r_i: &E) {
        if self.owned.is_empty() {
            self.owned = self
                .borrowed
                .par_iter()
                .map(|poly| poly.fix_var(r_i))
                .collect();
        } else {
            self.owned
                .par_iter_mut()
                .for_each(|poly| poly.fix_var_in_place(r_i));
        }
    }

    fn into_evals(self) -> Vec<E> {
        self.owned.into_iter().map(|poly| poly[0]).collect()
    }
}
