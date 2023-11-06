use crate::{
    poly::MultilinearPoly,
    transcript::{TranscriptRead, TranscriptWrite},
    util::{div_ceil, horner, izip, izip_eq, num_threads, parallelize_iter, Field, Itertools},
    Error,
};
use std::{array, borrow::Cow};

pub fn prove_sum_check<'a, F: Field, H: Function<F>>(
    claim: F,
    polys: impl IntoIterator<Item = Cow<'a, MultilinearPoly<F>>>,
    transcript: &mut impl TranscriptWrite<F>,
) -> Result<(F, Vec<F>, Vec<F>), Error> {
    let mut polys = polys.into_iter().map(Cow::into_owned).collect_vec();
    assert!(!polys.is_empty());

    let num_vars = polys[0].num_vars();
    assert!(num_vars > 0);
    assert!(!polys.iter().any(|poly| poly.num_vars() != num_vars));

    let degree = H::degree();
    assert!(degree >= 2);

    let mut claim = claim;
    let mut r = Vec::with_capacity(num_vars);
    for _ in 0..num_vars {
        let sum = H::compute_sum(claim, &polys);
        H::write_sum(&sum, transcript)?;
        assert_eq!(sum.len(), degree + 1);

        let r_i = transcript.squeeze_challenge();

        claim = horner(&sum, &r_i);
        polys.iter_mut().for_each(|poly| poly.fix_var(&r_i));
        r.push(r_i);
    }

    let evals = polys.into_iter().map(|poly| poly[0]).collect_vec();
    Ok((claim, r, evals))
}

pub fn verify_sum_check<F: Field, H: Function<F>>(
    claim: F,
    num_vars: usize,
    transcript: &mut impl TranscriptRead<F>,
) -> Result<(F, Vec<F>), Error> {
    assert!(num_vars > 0);

    let degree = H::degree();
    assert!(degree >= 2);

    let mut claim = claim;
    let mut r = Vec::with_capacity(num_vars);
    for _ in 0..num_vars {
        let sum = H::read_sum(claim, transcript)?;
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

pub trait Function<F: Field> {
    fn degree() -> usize;

    fn compute_sum(claim: F, polys: &[MultilinearPoly<F>]) -> Vec<F>;

    fn write_sum(sum: &[F], transcript: &mut impl TranscriptWrite<F>) -> Result<(), Error>;

    fn read_sum(claim: F, transcript: &mut impl TranscriptRead<F>) -> Result<Vec<F>, Error>;
}

pub struct Quadratic;

impl<F: Field> Function<F> for Quadratic {
    fn degree() -> usize {
        2
    }

    fn compute_sum(claim: F, polys: &[MultilinearPoly<F>]) -> Vec<F> {
        assert_eq!(polys.len(), 3);

        if cfg!(feature = "sanity-check") {
            let polys = polys.iter().map(|poly| poly.as_slice()).collect_vec();
            assert_eq!(
                claim,
                F::sum(izip_eq!(polys[0], polys[1], polys[2]).map(|(a, b, c)| *a + *b * c))
            )
        }

        let num_threads = num_threads().min(polys[0].len() >> 1);
        let chunk_size = div_ceil(polys[0].len() >> 1, num_threads);

        let mut partials = vec![[F::ZERO; 3]; num_threads];
        parallelize_iter(
            izip!(&mut partials, (0..).step_by(chunk_size << 1)),
            |(partial, start)| {
                let [a, b, c] = array::from_fn(|idx| polys[idx][start..].iter());
                izip!(a.step_by(2), b.tuples(), c.tuples())
                    .take(chunk_size)
                    .for_each(|(a_lo, (b_lo, b_hi), (c_lo, c_hi))| {
                        partial[0] += *a_lo + *b_lo * c_lo;
                        partial[2] += (*b_hi - b_lo) * (*c_hi - c_lo);
                    });
            },
        );

        let mut sum = [F::ZERO; 3];
        partials.iter().for_each(|partial| {
            sum[0] += partial[0];
            sum[2] += partial[2];
        });
        sum[1] = claim - sum[0].double() - sum[2];
        sum.to_vec()
    }

    fn write_sum(sum: &[F], transcript: &mut impl TranscriptWrite<F>) -> Result<(), Error> {
        transcript.write_felt(&sum[0])?;
        transcript.write_felt(&sum[2])?;
        Ok(())
    }

    fn read_sum(claim: F, transcript: &mut impl TranscriptRead<F>) -> Result<Vec<F>, Error> {
        let mut sum = [F::ZERO; 3];
        sum[0] = transcript.read_felt()?;
        sum[2] = transcript.read_felt()?;
        sum[1] = claim - sum[0].double() - sum[2];
        Ok(sum.to_vec())
    }
}
