use crate::{
    poly::MultilinearPoly,
    sum_check::SumCheckFunction,
    transcript::{TranscriptRead, TranscriptWrite},
    util::{arithmetic::Field, collection::AdditiveArray, izip_par},
    Error,
};
use rayon::prelude::*;

#[derive(Debug)]
pub struct Quadratic {
    num_vars: usize,
}

impl Quadratic {
    pub fn new(num_vars: usize) -> Self {
        Self { num_vars }
    }
}

impl<F: Field> SumCheckFunction<F> for Quadratic {
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn degree(&self) -> usize {
        2
    }

    fn compute_sum(
        &self,
        _: usize,
        claim: F,
        polys: &[&(impl MultilinearPoly<F> + ?Sized)],
    ) -> Vec<F> {
        assert_eq!(polys.len() % 2, 0);
        let (a, b) = polys.split_at(polys.len() >> 1);

        if cfg!(feature = "sanity-check") {
            assert_eq!(
                izip_par!(a, b)
                    .flat_map(|(a, b)| (0..a.len()).into_par_iter().map(|idx| a[idx] * b[idx]))
                    .sum::<F>(),
                claim
            )
        }

        let AdditiveArray([coeff_0, coeff_2]) = izip_par!(a, b)
            .flat_map(|(a, b)| {
                (0..a.len())
                    .into_par_iter()
                    .step_by(2)
                    .with_min_len(64)
                    .map(|idx| {
                        let coeff_0 = a[idx] * b[idx];
                        let coeff_2 = (a[idx + 1] - a[idx]) * (b[idx + 1] - b[idx]);
                        AdditiveArray([coeff_0, coeff_2])
                    })
            })
            .sum();

        vec![coeff_0, claim - coeff_0.double() - coeff_2, coeff_2]
    }

    fn write_sum(
        &self,
        _: usize,
        sum: &[F],
        transcript: &mut (impl TranscriptWrite<F> + ?Sized),
    ) -> Result<(), Error> {
        transcript.write_felt(&sum[0])?;
        transcript.write_felt(&sum[2])?;
        Ok(())
    }

    fn read_sum(
        &self,
        _: usize,
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
