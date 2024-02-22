//! Application of section 3.2 of [Some Improvements for the PIOP for ZeroCheck] on qudratic
//! SumCheck when one of multiplicand is eq polynomial.
//!
//! [Some Improvements for the PIOP for ZeroCheck]: https://eprint.iacr.org/2024/108

use crate::{
    sum_check::{op_sum_check_poly, BoxSumCheckPoly, SumCheckFunction},
    transcript::{TranscriptRead, TranscriptWrite},
    util::{
        arithmetic::{BatchInvert, ExtensionField, Field},
        chain, Itertools,
    },
    Error,
};
use rayon::prelude::*;
use std::mem;

#[derive(Clone, Debug)]
pub struct EqF<F> {
    r: Vec<F>,
    r_inv: Vec<F>,
    one_minus_r_inv: Vec<F>,
    subsets: Vec<Vec<F>>,
}

impl<F: Field> EqF<F> {
    pub fn new(r: &[F], is_proving: bool) -> Self {
        let subsets = is_proving.then(|| eq_subsets(r)).unwrap_or_default();
        let mut r_inv = r.to_vec();
        let mut one_minus_r_inv = r.iter().map(|r_i| F::ONE - r_i).collect_vec();
        chain![&mut r_inv, &mut one_minus_r_inv].batch_invert();
        Self {
            r: r.to_vec(),
            r_inv,
            one_minus_r_inv,
            subsets,
        }
    }

    pub fn r_i(&self, round: usize) -> F {
        self.r[round]
    }

    pub fn r_i_inv(&self, round: usize) -> F {
        self.r_inv[round]
    }
    pub fn one_minus_r_i_inv(&self, round: usize) -> F {
        self.one_minus_r_inv[round]
    }

    pub fn subset_i(&self, round: usize) -> &[F] {
        &self.subsets[round]
    }

    pub fn eval_0(&self, round: usize, claim: F, eval_1: F) -> F {
        (claim - self.r_i(round) * eval_1) * self.one_minus_r_i_inv(round)
    }

    pub fn eval_1(&self, round: usize, claim: F, eval_0: F) -> F {
        (claim - (F::ONE - self.r_i(round)) * eval_0) * self.r_i_inv(round)
    }
}

impl<F: Field, E: ExtensionField<F>> SumCheckFunction<F, E> for EqF<E> {
    fn num_vars(&self) -> usize {
        self.r.len()
    }

    fn degree(&self) -> usize {
        1
    }

    fn evaluate(&self, evals: &[E]) -> E {
        evals[0]
    }

    #[cfg(any(test, feature = "sanity-check"))]
    fn compute_sum(&self, round: usize, polys: &[BoxSumCheckPoly<F, E>]) -> E {
        assert_eq!(polys.len(), 1);

        let r_i = self.r_i(round);
        let subset_i = self.subset_i(round);
        let f = &polys[0];

        op_sum_check_poly!(|f| (0..f.len())
            .into_par_iter()
            .map(|idx| {
                let scalar = subset_i[idx >> 1] * if idx & 1 == 0 { E::ONE - r_i } else { r_i };
                scalar * f[idx]
            })
            .sum::<E>())
    }

    fn compute_round_poly(
        &self,
        round: usize,
        claim: E,
        polys: &[BoxSumCheckPoly<F, E>],
    ) -> Vec<E> {
        assert_eq!(polys.len(), 1);

        #[cfg(feature = "sanity-check")]
        assert_eq!(self.compute_sum(round, polys), claim);

        let subset_i = self.subset_i(round);
        let f = &polys[0];

        let eval_0 = op_sum_check_poly!(|f| {
            (0..f.len())
                .into_par_iter()
                .step_by(2)
                .with_min_len(64)
                .map(|idx| subset_i[idx >> 1] * f[idx])
                .sum()
        });
        let eval_1 = self.eval_1(round, claim, eval_0);

        vec![eval_0, eval_1 - eval_0]
    }

    fn write_round_poly(
        &self,
        _: usize,
        sum: &[E],
        transcript: &mut dyn TranscriptWrite<F, E>,
    ) -> Result<(), Error> {
        transcript.write_felt_ext(&sum[0])?;
        Ok(())
    }

    fn read_round_poly(
        &self,
        round: usize,
        claim: E,
        transcript: &mut dyn TranscriptRead<F, E>,
    ) -> Result<Vec<E>, Error> {
        let mut sum = vec![E::ZERO; 2];
        sum[0] = transcript.read_felt_ext()?;
        sum[1] = self.eval_1(round, claim, sum[0]) - sum[0];
        Ok(sum)
    }
}

fn eq_subsets<F: Field>(r: &[F]) -> Vec<Vec<F>> {
    let mut subsets = r
        .iter()
        .enumerate()
        .rev()
        .scan(vec![F::ONE], |subset_i, (idx, r_i)| {
            if idx == 0 {
                mem::take(subset_i)
            } else {
                let one_minus_r_i = F::ONE - r_i;
                mem::replace(
                    subset_i,
                    (0..subset_i.len() << 1)
                        .into_par_iter()
                        .with_min_len(64)
                        .map(|idx| {
                            subset_i[idx >> 1] * if idx & 1 == 0 { one_minus_r_i } else { *r_i }
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
        poly::box_dense_poly,
        sum_check::{eq_f::EqF, test::run_sum_check, SumCheckPoly},
        util::dev::rand_vec,
    };
    use goldilocks::{Goldilocks, GoldilocksExt2};

    #[test]
    fn eq_f() {
        run_sum_check::<Goldilocks, GoldilocksExt2, _>(|num_vars, mut rng| {
            let r = rand_vec(num_vars, &mut rng);
            let g = EqF::new(&r, true);
            let f = box_dense_poly(rand_vec(1 << num_vars, &mut rng));
            (g, vec![SumCheckPoly::Base(f)])
        });
    }
}
