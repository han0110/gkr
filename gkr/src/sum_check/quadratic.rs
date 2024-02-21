use crate::{
    poly::MultilinearPoly,
    sum_check::{eq_f::EqF, op_sum_check_polys, SumCheckFunction, SumCheckPoly},
    transcript::{TranscriptRead, TranscriptWrite},
    util::{
        arithmetic::{ExtensionField, Field},
        collection::AdditiveArray,
    },
    Error,
};
use rayon::prelude::*;

#[derive(Debug)]
pub struct Quadratic<F> {
    num_vars: usize,
    pairs: Vec<(Option<F>, usize, usize)>,
    eq: Option<EqF<F>>,
}

impl<F: Field, E: ExtensionField<F>> SumCheckFunction<F, E> for Quadratic<E> {
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn degree(&self) -> usize {
        2
    }

    fn evaluate(&self, evals: &[E]) -> E {
        self.pairs
            .iter()
            .map(|(scalar, f, g)| {
                let eval = evals[*f] * evals[*g];
                scalar.map(|scalar| scalar * eval).unwrap_or(eval)
            })
            .sum()
    }

    #[cfg(any(test, feature = "sanity-check"))]
    fn compute_sum(
        &self,
        round: usize,
        polys: &[SumCheckPoly<F, E, impl MultilinearPoly<F, E>, impl MultilinearPoly<E, E>>],
    ) -> E {
        if let Some(eq) = self.eq() {
            let r_i = eq.r_i(round);
            let subset_i = eq.subset_i(round);
            self.pairs
                .par_iter()
                .map(|(scalar, f, g)| {
                    let (f, g) = (&polys[*f], &polys[*g]);
                    let sum = op_sum_check_polys!(|f, g| {
                        (0..f.len())
                            .into_par_iter()
                            .map(|b| {
                                let eval = f[b] * g[b];
                                let eq_eval =
                                    subset_i[b >> 1] * if b & 1 == 0 { E::ONE - r_i } else { r_i };
                                eq_eval * eval
                            })
                            .sum::<E>()
                    });
                    scalar.map(|scalar| scalar * sum).unwrap_or(sum)
                })
                .sum()
        } else {
            self.pairs
                .par_iter()
                .map(|(scalar, f, g)| {
                    let (f, g) = (&polys[*f], &polys[*g]);
                    let sum = op_sum_check_polys!(
                        |f, g| {
                            (0..f.len())
                                .into_par_iter()
                                .map(|idx| f[idx] * g[idx])
                                .sum()
                        },
                        |sum| E::from(sum)
                    );
                    scalar.map(|scalar| scalar * sum).unwrap_or(sum)
                })
                .sum()
        }
    }

    fn compute_round_poly(
        &self,
        round: usize,
        claim: E,
        polys: &[SumCheckPoly<F, E, impl MultilinearPoly<F, E>, impl MultilinearPoly<E, E>>],
    ) -> Vec<E> {
        #[cfg(feature = "sanity-check")]
        assert_eq!(self.compute_sum(round, polys), claim);

        let AdditiveArray([coeff_0, coeff_2]) = if let Some(eq) = self.eq() {
            let subset_i = eq.subset_i(round);
            self.pairs
                .par_iter()
                .map(|(scalar, f, g)| {
                    let (f, g) = (&polys[*f], &polys[*g]);
                    let sum = op_sum_check_polys!(|f, g| (0..f.len())
                        .into_par_iter()
                        .step_by(2)
                        .with_min_len(64)
                        .map(|b| {
                            let eq_eval = subset_i[b >> 1];
                            let coeff_0 = f[b] * g[b];
                            let coeff_2 = (f[b + 1] - f[b]) * (g[b + 1] - g[b]);
                            AdditiveArray([eq_eval * coeff_0, eq_eval * coeff_2])
                        })
                        .sum::<AdditiveArray<_, 2>>());
                    scalar
                        .map(|scalar| AdditiveArray(sum.0.map(|sum| sum * scalar)))
                        .unwrap_or(sum)
                })
                .sum()
        } else {
            self.pairs
                .par_iter()
                .map(|(scalar, f, g)| {
                    let (f, g) = (&polys[*f], &polys[*g]);
                    let sum = op_sum_check_polys!(
                        |f, g| (0..f.len())
                            .into_par_iter()
                            .step_by(2)
                            .with_min_len(64)
                            .map(|b| {
                                let coeff_0 = f[b] * g[b];
                                let coeff_2 = (f[b + 1] - f[b]) * (g[b + 1] - g[b]);
                                AdditiveArray([coeff_0, coeff_2])
                            })
                            .sum::<AdditiveArray<_, 2>>(),
                        |sum| AdditiveArray(sum.0.map(E::from))
                    );
                    scalar
                        .map(|scalar| AdditiveArray(sum.0.map(|sum| sum * scalar)))
                        .unwrap_or(sum)
                })
                .sum()
        };

        let eval_1 = if let Some(eq) = self.eq() {
            eq.eval_1(round, claim, coeff_0)
        } else {
            claim - coeff_0
        };
        vec![coeff_0, eval_1 - coeff_0 - coeff_2, coeff_2]
    }

    fn write_round_poly(
        &self,
        _: usize,
        sum: &[E],
        transcript: &mut (impl TranscriptWrite<F, E> + ?Sized),
    ) -> Result<(), Error> {
        transcript.write_felt_ext(&sum[0])?;
        transcript.write_felt_ext(&sum[2])?;
        Ok(())
    }

    fn read_round_poly(
        &self,
        round: usize,
        claim: E,
        transcript: &mut (impl TranscriptRead<F, E> + ?Sized),
    ) -> Result<Vec<E>, Error> {
        let mut sum = vec![E::ZERO; 3];
        sum[0] = transcript.read_felt_ext()?;
        sum[2] = transcript.read_felt_ext()?;
        let eval_1 = if let Some(eq) = self.eq() {
            eq.eval_1(round, claim, sum[0])
        } else {
            claim - sum[0]
        };
        sum[1] = eval_1 - sum[0] - sum[2];
        Ok(sum)
    }
}

impl<F: Field> Quadratic<F> {
    pub fn new(num_vars: usize, pairs: Vec<(Option<F>, usize, usize)>) -> Self {
        let pairs = pairs
            .into_iter()
            .filter_map(|(scalar, f, g)| match scalar {
                Some(scalar) if scalar == F::ZERO => unreachable!(),
                Some(scalar) if scalar == F::ONE => Some((None, f, g)),
                _ => Some((scalar, f, g)),
            })
            .collect();
        Self {
            num_vars,
            pairs,
            eq: None,
        }
    }

    pub fn mul_by_eq(mut self, r_eq: &[F], is_proving: bool) -> Self {
        self.eq = Some(EqF::new(r_eq, is_proving));
        self
    }

    pub fn eq(&self) -> Option<&EqF<F>> {
        self.eq.as_ref()
    }
}

#[cfg(test)]
mod test {
    use crate::{
        poly::box_dense_poly,
        sum_check::{quadratic::Quadratic, test::run_sum_check, SumCheckPoly},
        util::{arithmetic::Field, dev::rand_vec},
    };
    use goldilocks::{Goldilocks, GoldilocksExt2};
    use std::iter;

    #[test]
    fn quadratic() {
        run_sum_check::<Goldilocks, GoldilocksExt2, _>(|num_vars, mut rng| {
            let scalar = Field::random(&mut rng);
            let g = Quadratic::new(num_vars, vec![(Some(scalar), 0, 1)]);
            let polys = iter::repeat_with(|| rand_vec(1 << num_vars, &mut rng))
                .map(box_dense_poly)
                .take(2);
            (g, polys.map(SumCheckPoly::Base).collect())
        });
    }

    #[test]
    fn quadratic_mul_by_eq() {
        run_sum_check::<Goldilocks, GoldilocksExt2, _>(|num_vars, mut rng| {
            let scalar = Field::random(&mut rng);
            let r = rand_vec(num_vars, &mut rng);
            let g = Quadratic::new(num_vars, vec![(Some(scalar), 0, 1)]).mul_by_eq(&r, true);
            let polys = iter::repeat_with(|| rand_vec(1 << num_vars, &mut rng))
                .map(box_dense_poly)
                .take(2);
            (g, polys.map(SumCheckPoly::Base).collect())
        });
    }
}
