use crate::{
    poly::{BoxMultilinearPoly, BoxMultilinearPolyOwned, MultilinearPoly, MultilinearPolyExt},
    transcript::{TranscriptRead, TranscriptWrite},
    util::{
        arithmetic::{horner, ExtensionField, Field},
        Itertools,
    },
    Error,
};
use rayon::prelude::*;
use std::{convert::Infallible, fmt::Debug, marker::PhantomData, mem};

pub mod eq_f;
pub mod generic;
pub mod quadratic;

pub fn prove_sum_check<F, E, P, PE>(
    g: &impl SumCheckFunction<F, E>,
    claim: E,
    polys: impl IntoIterator<Item = SumCheckPoly<F, E, P, PE>>,
    transcript: &mut dyn TranscriptWrite<F, E>,
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
            g.compute_round_poly(round, claim, &polys.borrowed)
        } else {
            g.compute_round_poly(round, claim, &polys.owned())
        };

        g.write_round_poly(round, &sum, transcript)?;
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
    transcript: &mut dyn TranscriptRead<F, E>,
) -> Result<(E, Vec<E>), Error> {
    let num_vars = g.num_vars();
    assert!(num_vars > 0);

    let degree = g.degree();
    assert!(degree >= 1);

    let mut claim = claim;
    let mut r = Vec::with_capacity(num_vars);
    for round in 0..num_vars {
        let sum = g.read_round_poly(round, claim, transcript)?;
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

#[auto_impl::auto_impl(&, Box)]
pub trait SumCheckFunction<F, E>: Debug + Send + Sync {
    fn num_vars(&self) -> usize;

    fn degree(&self) -> usize;

    fn evaluate(&self, evals: &[E]) -> E;

    #[cfg(any(test, feature = "sanity-check"))]
    fn compute_sum(&self, round: usize, polys: &[BoxSumCheckPoly<F, E>]) -> E;

    fn compute_round_poly(&self, round: usize, claim: E, polys: &[BoxSumCheckPoly<F, E>])
        -> Vec<E>;

    fn write_round_poly(
        &self,
        round: usize,
        sum: &[E],
        transcript: &mut dyn TranscriptWrite<F, E>,
    ) -> Result<(), Error>;

    fn read_round_poly(
        &self,
        round: usize,
        claim: E,
        transcript: &mut dyn TranscriptRead<F, E>,
    ) -> Result<Vec<E>, Error>;
}

pub trait SumCheckFunctionExt<F, E>: SumCheckFunction<F, E> {
    fn boxed<'a>(self) -> Box<dyn SumCheckFunction<F, E> + 'a>
    where
        Self: 'a + Sized,
    {
        Box::new(self)
    }
}

impl<F, E, N: SumCheckFunction<F, E>> SumCheckFunctionExt<F, E> for N {
    fn boxed<'a>(self) -> Box<dyn SumCheckFunction<F, E> + 'a>
    where
        Self: 'a + Sized,
    {
        Box::new(self)
    }
}

type BoxSumCheckPoly<'a, F, E> =
    SumCheckPoly<F, E, BoxMultilinearPoly<'a, F, E>, BoxMultilinearPoly<'a, E>>;

#[derive(Clone, Debug)]
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

    pub fn as_ref(&self) -> SumCheckPoly<F, E, &P, &PE> {
        match self {
            Self::Base(poly) => SumCheckPoly::Base(poly),
            Self::Extension(poly) => SumCheckPoly::Extension(poly),
            _ => unreachable!(),
        }
    }

    pub fn boxed<'a>(self) -> BoxSumCheckPoly<'a, F, E>
    where
        Self: 'a,
    {
        match self {
            Self::Base(poly) => SumCheckPoly::Base(poly.boxed()),
            Self::Extension(poly) => SumCheckPoly::Extension(poly.boxed()),
            _ => unreachable!(),
        }
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

impl<'a, F, E, PE: MultilinearPoly<E, E>> SumCheckPoly<F, E, BoxMultilinearPoly<'a, F, E>, PE> {
    pub fn exts(polys: impl IntoIterator<Item = PE>) -> Vec<Self> {
        polys.into_iter().map(Self::Extension).collect()
    }
}

impl<'a, F, E, P: MultilinearPoly<F, E>> SumCheckPoly<F, E, P, BoxMultilinearPoly<'a, E>> {
    pub fn bases(polys: impl IntoIterator<Item = P>) -> Vec<Self> {
        polys.into_iter().map(Self::Base).collect()
    }
}

#[macro_export]
macro_rules! op_sum_check_polys {
    (|$a:ident, $b:ident| $op:expr, |$bb_out:ident| $op_bb_out:expr) => {
        $crate::sum_check::SumCheckPoly::op(
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
    (|$a:ident, $b:ident| $op:expr) => {
        op_sum_check_polys!(|$a, $b| $op, |out| out)
    };
}

#[macro_export]
macro_rules! op_sum_check_poly {
    ($a:ident, |$tmp_a:ident| $op:expr, |$b_out:ident| $op_b_out:expr) => {
        match $a {
            $crate::sum_check::SumCheckPoly::Base(a) => {
                let $tmp_a = a;
                let $b_out = $op;
                $op_b_out
            }
            $crate::sum_check::SumCheckPoly::Extension(a) => {
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

#[derive(Debug)]
struct Polys<'a, F, E> {
    borrowed: Vec<BoxSumCheckPoly<'a, F, E>>,
    owned: Vec<BoxMultilinearPolyOwned<'static, E>>,
    _marker: PhantomData<F>,
}

impl<'a, F, E> Polys<'a, F, E>
where
    F: Field,
    E: ExtensionField<F>,
{
    fn new<P, PE>(
        num_vars: usize,
        polys: impl IntoIterator<Item = SumCheckPoly<F, E, P, PE>>,
    ) -> Self
    where
        P: 'a + MultilinearPoly<F, E>,
        PE: 'a + MultilinearPoly<E, E>,
    {
        assert!(num_vars > 0);

        let polys = polys.into_iter().map(SumCheckPoly::boxed).collect_vec();
        assert!(!polys.is_empty());
        assert!(!polys.iter().any(|poly| poly.num_vars() != num_vars));

        Self {
            borrowed: polys,
            owned: Vec::new(),
            _marker: PhantomData,
        }
    }

    fn owned(&self) -> Vec<BoxSumCheckPoly<F, E>> {
        self.owned
            .iter()
            .map(MultilinearPolyExt::boxed)
            .map(SumCheckPoly::Extension)
            .collect()
    }

    fn fix_var(&mut self, r_i: &E) {
        if self.owned.is_empty() {
            self.owned = mem::take(&mut self.borrowed)
                .into_par_iter()
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

#[cfg(test)]
pub(super) mod test {
    use crate::{
        izip_eq,
        sum_check::{prove_sum_check, verify_sum_check, BoxSumCheckPoly, SumCheckFunction},
        transcript::StdRngTranscript,
        util::{
            arithmetic::{ExtensionField, PrimeField},
            dev::seeded_std_rng,
            Itertools,
        },
    };
    use rand::rngs::StdRng;

    pub(super) type TestData<F, E, G> = (G, Vec<BoxSumCheckPoly<'static, F, E>>);

    pub(super) fn run_sum_check<F: PrimeField, E: ExtensionField<F>, G: SumCheckFunction<F, E>>(
        f: impl Fn(usize, &mut StdRng) -> TestData<F, E, G>,
    ) {
        let mut rng = seeded_std_rng();
        for num_vars in 1..10 {
            let (g, polys) = f(num_vars, &mut rng);
            let claim = g.compute_sum(0, &polys);

            let (proof, evals) = {
                let polys = polys.iter().map(|poly| poly.as_ref().boxed());
                let mut transcript = StdRngTranscript::default();
                let (_, _, evals) = prove_sum_check(&g, claim, polys, &mut transcript).unwrap();
                (transcript.into_proof(), evals)
            };

            let x = {
                let mut transcript = StdRngTranscript::from_proof(&proof);
                let (sub_claim, x) = verify_sum_check(&g, claim, &mut transcript).unwrap();
                assert_eq!(g.evaluate(&evals), sub_claim);
                x
            };

            izip_eq!(polys, evals).for_each(|(poly, eval)| {
                assert_eq!(op_sum_check_poly!(|poly| poly.evaluate(&x)), eval)
            });
        }
    }
}
