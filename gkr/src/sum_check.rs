use crate::{
    poly::MultilinearPoly,
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

pub fn prove_sum_check<F: Field>(
    g: &impl SumCheckFunction<F>,
    claim: F,
    polys: impl IntoIterator<Item = MultilinearPoly<F>>,
    transcript: &mut (impl TranscriptWrite<F> + ?Sized),
) -> Result<(F, Vec<F>, Vec<F>), Error> {
    let mut polys = polys.into_iter().collect_vec();
    assert!(!polys.is_empty());

    let num_vars = polys[0].num_vars();
    assert!(num_vars > 0);
    assert!(!polys.iter().any(|poly| poly.num_vars() != num_vars));

    let degree = g.degree();
    assert!(degree >= 2);

    let mut claim = claim;
    let mut r = Vec::with_capacity(num_vars);
    for _ in 0..num_vars {
        let sum = g.compute_sum(claim, &polys);
        g.write_sum(&sum, transcript)?;
        assert_eq!(sum.len(), degree + 1);

        let r_i = transcript.squeeze_challenge();

        claim = horner(&sum, &r_i);
        polys.par_iter_mut().for_each(|poly| poly.fix_var(&r_i));
        r.push(r_i);
    }

    let evals = polys.into_iter().map(|poly| poly[0]).collect_vec();
    Ok((claim, r, evals))
}

pub fn verify_sum_check<F: Field>(
    g: &impl SumCheckFunction<F>,
    claim: F,
    num_vars: usize,
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

    fn compute_sum(&self, claim: F, polys: &[MultilinearPoly<F>]) -> Vec<F>;

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