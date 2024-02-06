use crate::{
    poly::MultilinearPoly,
    sum_check::SumCheckFunction,
    transcript::{TranscriptRead, TranscriptWrite},
    util::{
        arithmetic::{BatchInvert, Field},
        Itertools,
    },
    Error,
};
use rayon::prelude::*;
use std::mem;

#[derive(Debug)]
pub struct EqF<F> {
    r: Vec<F>,
    r_inv: Vec<F>,
    subsets: Vec<Vec<F>>,
}

impl<F: Field> EqF<F> {
    pub fn new(r: &[F], is_proving: bool) -> Self {
        let subsets = is_proving.then(|| eq_subsets(r)).unwrap_or_default();
        let r_inv = {
            let mut r_inv = r.to_vec();
            r_inv.batch_invert();
            r_inv
        };
        Self {
            r: r.to_vec(),
            r_inv,
            subsets,
        }
    }
}

impl<F: Field> SumCheckFunction<F> for EqF<F> {
    fn num_vars(&self) -> usize {
        self.r.len()
    }

    fn degree(&self) -> usize {
        1
    }

    fn compute_sum(
        &self,
        round: usize,
        claim: F,
        polys: &[&(impl MultilinearPoly<F> + ?Sized)],
    ) -> Vec<F> {
        assert_eq!(polys.len(), 1);

        let r_i = self.r[round];
        let r_i_inv = self.r_inv[round];
        let subset_i = &self.subsets[round];
        let f = polys[0];

        if cfg!(feature = "sanity-check") {
            let one_minus_r_i = F::ONE - r_i;
            assert_eq!(
                (0..f.len())
                    .into_par_iter()
                    .map(|idx| {
                        f[idx] * subset_i[idx >> 1] * if idx & 1 == 0 { one_minus_r_i } else { r_i }
                    })
                    .sum::<F>(),
                claim
            )
        }

        let eval_0 = (0..f.len())
            .into_par_iter()
            .step_by(2)
            .with_min_len(64)
            .map(|idx| f[idx] * subset_i[idx >> 1])
            .sum();
        let eval_1 = (claim - (F::ONE - r_i) * eval_0) * r_i_inv;

        vec![eval_0, eval_1 - eval_0]
    }

    fn write_sum(
        &self,
        _: usize,
        sum: &[F],
        transcript: &mut (impl TranscriptWrite<F> + ?Sized),
    ) -> Result<(), Error> {
        transcript.write_felt(&sum[0])?;
        Ok(())
    }

    fn read_sum(
        &self,
        round: usize,
        claim: F,
        transcript: &mut (impl TranscriptRead<F> + ?Sized),
    ) -> Result<Vec<F>, Error> {
        let r_i = self.r[round];
        let r_i_inv = self.r_inv[round];
        let mut sums = vec![F::ZERO; 2];
        sums[0] = transcript.read_felt()?;
        sums[1] = (claim - (F::ONE - r_i) * sums[0]) * r_i_inv - sums[0];
        Ok(sums)
    }
}

fn eq_subsets<F: Field>(r: &[F]) -> Vec<Vec<F>> {
    let mut subsets = r
        .iter()
        .enumerate()
        .rev()
        .scan(vec![F::ONE], |subset, (idx, r_i)| {
            if idx == 0 {
                mem::take(subset)
            } else {
                let one_minus_r_i = F::ONE - r_i;
                mem::replace(
                    subset,
                    (0..subset.len() << 1)
                        .into_par_iter()
                        .with_min_len(64)
                        .map(|idx| {
                            subset[idx >> 1] * if idx & 1 == 0 { one_minus_r_i } else { *r_i }
                        })
                        .collect(),
                )
            }
            .into()
        })
        .collect_vec();
    subsets.reverse();
    subsets
}

#[cfg(test)]
mod test {
    use crate::{
        poly::{DenseMultilinearPoly, MultilinearPoly},
        sum_check::{eq_f::EqF, prove_sum_check, verify_sum_check},
        transcript::StdRngTranscript,
        util::dev::{rand_vec, seeded_std_rng},
    };
    use halo2_curves::bn256::Fr;

    #[test]
    fn eq_f() {
        let mut rng = seeded_std_rng();
        for num_vars in 1..10 {
            let f = DenseMultilinearPoly::new(rand_vec(1 << num_vars, &mut rng));
            let r = rand_vec::<Fr>(num_vars, &mut rng);
            let claim = f.evaluate(&r);

            let proof = {
                let g = EqF::new(&r, true);
                let mut transcript = StdRngTranscript::<Vec<_>>::default();
                prove_sum_check(&g, claim, [&f], &mut transcript).unwrap();
                transcript.into_proof()
            };

            let (sub_claim, r_prime) = {
                let g = EqF::new(&r, false);
                let mut transcript = StdRngTranscript::from_proof(&proof);
                verify_sum_check(&g, claim, &mut transcript).unwrap()
            };

            assert_eq!(sub_claim, f.evaluate(&r_prime))
        }
    }
}
