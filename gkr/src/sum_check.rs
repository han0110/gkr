use crate::{
    poly::{BoxMultilinearPolyOwned, MultilinearPoly},
    transcript::{TranscriptRead, TranscriptWrite},
    util::{
        arithmetic::{horner, Field},
        Itertools,
    },
    Error,
};
use rayon::prelude::*;
use std::fmt::Debug;

pub mod generic;
pub mod quadratic;

pub fn prove_sum_check<'a, F, P>(
    g: &impl SumCheckFunction<F>,
    num_vars: usize,
    claim: F,
    polys: impl IntoIterator<Item = &'a P>,
    transcript: &mut (impl TranscriptWrite<F> + ?Sized),
) -> Result<(F, Vec<F>, Vec<F>), Error>
where
    F: Field,
    P: 'a + MultilinearPoly<F> + ?Sized,
{
    let degree = g.degree();
    assert!(degree >= 2);

    let mut claim = claim;
    let mut polys = Polys::new(num_vars, polys);
    let mut r = Vec::with_capacity(num_vars);
    for round in 0..num_vars {
        let sum = if round == 0 {
            g.compute_sum(round, claim, &polys.borrowed)
        } else {
            g.compute_sum(round, claim, &polys.owned())
        };

        g.write_sum(&sum, transcript)?;
        assert_eq!(sum.len(), degree + 1);

        let r_i = transcript.squeeze_challenge();

        claim = horner(&sum, &r_i);
        polys.fix_var(&r_i);
        r.push(r_i);
    }

    Ok((claim, r, polys.into_evals()))
}

pub fn verify_sum_check<F: Field>(
    g: &impl SumCheckFunction<F>,
    num_vars: usize,
    claim: F,
    transcript: &mut (impl TranscriptRead<F> + ?Sized),
) -> Result<(F, Vec<F>), Error> {
    assert!(num_vars > 0);

    let degree = g.degree();
    assert!(degree >= 2);

    let mut claim = claim;
    let mut r = Vec::with_capacity(num_vars);
    for _ in 0..num_vars {
        let sum = g.read_sum(claim, transcript)?;
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

pub trait SumCheckFunction<F>: Debug {
    fn degree(&self) -> usize;

    fn compute_sum(
        &self,
        round: usize,
        claim: F,
        polys: &[&(impl MultilinearPoly<F> + ?Sized)],
    ) -> Vec<F>;

    fn write_sum(
        &self,
        sum: &[F],
        transcript: &mut (impl TranscriptWrite<F> + ?Sized),
    ) -> Result<(), Error>;

    fn read_sum(
        &self,
        claim: F,
        transcript: &mut (impl TranscriptRead<F> + ?Sized),
    ) -> Result<Vec<F>, Error>;
}

struct Polys<'a, F, P: ?Sized> {
    borrowed: Vec<&'a P>,
    owned: Vec<BoxMultilinearPolyOwned<'static, F>>,
}

impl<'a, F, P> Polys<'a, F, P>
where
    F: Field,
    P: 'a + MultilinearPoly<F> + ?Sized,
{
    fn new(num_vars: usize, polys: impl IntoIterator<Item = &'a P>) -> Self {
        assert!(num_vars > 0);

        let polys = polys.into_iter().collect_vec();
        assert!(!polys.is_empty());
        assert!(!polys.iter().any(|poly| poly.num_vars() != num_vars));

        Self {
            borrowed: polys,
            owned: Vec::new(),
        }
    }

    fn owned(&self) -> Vec<&(impl MultilinearPoly<F> + ?Sized)> {
        self.owned.iter().collect()
    }

    fn fix_var(&mut self, r_i: &F) {
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

    fn into_evals(self) -> Vec<F> {
        self.owned.into_iter().map(|poly| poly[0]).collect()
    }
}
