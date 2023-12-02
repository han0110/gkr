use crate::{
    poly::MultilinearPoly,
    sum_check::SumCheckFunction,
    transcript::{TranscriptRead, TranscriptWrite},
    util::{arithmetic::Field, collection::AdditiveArray, izip_par},
    Error,
};
use rayon::prelude::*;

#[derive(Debug)]
pub struct Quadratic;

impl<F: Field> SumCheckFunction<F> for Quadratic {
    fn degree(&self) -> usize {
        2
    }

    fn compute_sum(&self, claim: F, polys: &[MultilinearPoly<F>]) -> Vec<F> {
        assert_eq!(polys.len() % 2, 0);

        if cfg!(feature = "sanity-check") {
            assert_eq!(
                polys
                    .par_chunks(2)
                    .flat_map(|polys| izip_par!(&polys[0][..], &polys[1][..]).map(|(a, b)| *a * b))
                    .sum::<F>(),
                claim
            )
        }

        let AdditiveArray([coeff_0, coeff_2]) = polys
            .par_chunks(2)
            .flat_map(|polys| {
                izip_par!(&polys[0][..], &polys[0][1..], &polys[1][..], &polys[1][1..])
                    .step_by(2)
                    .fold_with(AdditiveArray::default(), |mut coeffs, values| {
                        let (a_lo, a_hi, b_lo, b_hi) = values;
                        coeffs[0] += *a_lo * b_lo;
                        coeffs[1] += (*a_hi - a_lo) * (*b_hi - b_lo);
                        coeffs
                    })
            })
            .sum();

        vec![coeff_0, claim - coeff_0.double() - coeff_2, coeff_2]
    }

    fn write_sum(
        &self,
        sum: &[F],
        transcript: &mut (impl TranscriptWrite<F> + ?Sized),
    ) -> Result<(), Error> {
        transcript.write_felt(&sum[0])?;
        transcript.write_felt(&sum[2])?;
        Ok(())
    }

    fn read_sum(
        &self,
        claim: F,
        transcript: &mut (impl TranscriptRead<F> + ?Sized),
    ) -> Result<Vec<F>, Error> {
        let mut sum = vec![F::ZERO; 3];
        sum[0] = transcript.read_felt()?;
        sum[2] = transcript.read_felt()?;
        sum[1] = claim - sum[0].double() - sum[2];
        Ok(sum)
    }
}
