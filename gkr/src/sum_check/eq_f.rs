//! Application of section 3.2 of [Some Improvements for the PIOP for ZeroCheck] on qudratic
//! SumCheck when one of multiplicand is eq polynomial.
//!
//! [Some Improvements for the PIOP for ZeroCheck]: https://eprint.iacr.org/2024/108

use crate::{
    poly::MultilinearPoly,
    sum_check::{op_sum_check_poly, SumCheckFunction, SumCheckPoly},
    transcript::{TranscriptRead, TranscriptWrite},
    util::{
        arithmetic::{BatchInvert, ExtensionField, Field},
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

impl<F: Field, E: ExtensionField<F>> SumCheckFunction<F, E> for EqF<E> {
    fn num_vars(&self) -> usize {
        self.r.len()
    }

    fn degree(&self) -> usize {
        1
    }

    fn compute_sum(
        &self,
        round: usize,
        claim: E,
        polys: &[SumCheckPoly<F, E, impl MultilinearPoly<F, E>, impl MultilinearPoly<E, E>>],
    ) -> Vec<E> {
        assert_eq!(polys.len(), 1);

        let r_i = self.r[round];
        let r_i_inv = self.r_inv[round];
        let subset_i = &self.subsets[round];
        let f = &polys[0];

        if cfg!(feature = "sanity-check") {
            let one_minus_r_i = E::ONE - r_i;
            assert_eq!(
                op_sum_check_poly!(|f| {
                    (0..f.len())
                        .into_par_iter()
                        .map(|idx| {
                            subset_i[idx >> 1]
                                * f[idx]
                                * if idx & 1 == 0 { one_minus_r_i } else { r_i }
                        })
                        .sum::<E>()
                }),
                claim
            )
        }

        let eval_0 = op_sum_check_poly!(|f| {
            (0..f.len())
                .into_par_iter()
                .step_by(2)
                .with_min_len(64)
                .map(|idx| subset_i[idx >> 1] * f[idx])
                .sum()
        });
        let eval_1 = (claim - (E::ONE - r_i) * eval_0) * r_i_inv;

        vec![eval_0, eval_1 - eval_0]
    }

    fn write_sum(
        &self,
        _: usize,
        sum: &[E],
        transcript: &mut (impl TranscriptWrite<F, E> + ?Sized),
    ) -> Result<(), Error> {
        transcript.write_felt_ext(&sum[0])?;
        Ok(())
    }

    fn read_sum(
        &self,
        round: usize,
        claim: E,
        transcript: &mut (impl TranscriptRead<F, E> + ?Sized),
    ) -> Result<Vec<E>, Error> {
        let r_i = self.r[round];
        let r_i_inv = self.r_inv[round];
        let mut sums = vec![E::ZERO; 2];
        sums[0] = transcript.read_felt_ext()?;
        sums[1] = (claim - (E::ONE - r_i) * sums[0]) * r_i_inv - sums[0];
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
        poly::{box_dense_poly, MultilinearPoly},
        sum_check::{eq_f::EqF, prove_sum_check, verify_sum_check, SumCheckPoly},
        transcript::StdRngTranscript,
        util::dev::{rand_vec, seeded_std_rng},
    };
    use goldilocks::{Goldilocks, GoldilocksExt2};

    #[test]
    fn eq_f() {
        let mut rng = seeded_std_rng();
        for num_vars in 1..10 {
            let f = box_dense_poly(rand_vec::<Goldilocks>(1 << num_vars, &mut rng));
            let r = rand_vec::<GoldilocksExt2>(num_vars, &mut rng);
            let claim = f.evaluate(&r);

            let proof = {
                let g = EqF::new(&r, true);
                let mut transcript = StdRngTranscript::default();
                prove_sum_check(&g, claim, SumCheckPoly::bases([&f]), &mut transcript).unwrap();
                transcript.into_proof()
            };

            let (sub_claim, r_prime) = {
                let g = EqF::new(&r, false);
                let mut transcript = StdRngTranscript::from_proof(&proof);
                verify_sum_check::<Goldilocks, _>(&g, claim, &mut transcript).unwrap()
            };

            assert_eq!(sub_claim, f.evaluate(&r_prime))
        }
    }
}
