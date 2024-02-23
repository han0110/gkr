use crate::{
    chain_par, op_sum_check_poly,
    sum_check::{eq_f::EqF, op_sum_check_polys, BoxSumCheckPoly, SumCheckFunction},
    transcript::{TranscriptRead, TranscriptWrite},
    util::{
        arithmetic::{powers, ExtensionField, Field},
        chain,
        collection::AdditiveArray,
        expression::Expression,
        Itertools,
    },
    Error,
};
use rayon::prelude::*;
use std::{borrow::Borrow, collections::BTreeMap, ops::Mul};

#[derive(Debug)]
pub struct Quadratic<E> {
    num_vars: usize,
    d_0: Option<E>,
    d_1: Vec<(Option<E>, usize)>,
    d_2: Vec<(Option<E>, usize, usize)>,
    eq: Option<EqF<E>>,
}

impl<F: Field, E: ExtensionField<F>> SumCheckFunction<F, E> for Quadratic<E> {
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn degree(&self) -> usize {
        2
    }

    fn evaluate(&self, evals: &[E]) -> E {
        chain![
            self.d_1.iter().map(|(scalar, f)| (*scalar, evals[*f])),
            self.d_2
                .iter()
                .map(|(scalar, f, g)| (*scalar, evals[*f] * evals[*g])),
        ]
        .map(maybe_scale)
        .sum::<E>()
            + self.d_0.unwrap_or(E::ZERO)
    }

    #[cfg(any(test, feature = "sanity-check"))]
    fn compute_sum(&self, round: usize, polys: &[BoxSumCheckPoly<F, E>]) -> E {
        let sum = if let Some(eq) = self.eq() {
            let r_i = eq.r_i(round);
            let subset_i = eq.subset_i(round);
            chain_par![
                self.d_1.par_iter().map(|(scalar, f)| {
                    let sum = eq.compute_sum(round, std::slice::from_ref(&polys[*f]));
                    (scalar, sum)
                }),
                self.d_2.par_iter().map(|(scalar, f, g)| {
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
                            .sum()
                    });
                    (scalar, sum)
                })
            ]
            .map(maybe_scale)
            .sum::<E>()
        } else {
            chain_par![
                self.d_1.par_iter().map(|(scalar, f)| {
                    let f = &polys[*f];
                    let sum = op_sum_check_poly!(
                        |f| (0..f.len()).into_par_iter().map(|b| f[b]).sum(),
                        |sum| E::from(sum)
                    );
                    (scalar, sum)
                }),
                self.d_2.par_iter().map(|(scalar, f, g)| {
                    let (f, g) = (&polys[*f], &polys[*g]);
                    let sum = op_sum_check_polys!(
                        |f, g| (0..f.len()).into_par_iter().map(|b| f[b] * g[b]).sum(),
                        |sum| E::from(sum)
                    );
                    (scalar, sum)
                })
            ]
            .map(maybe_scale)
            .sum()
        };
        sum + self.d_0_sum(round)
    }

    fn compute_round_poly(
        &self,
        round: usize,
        claim: E,
        polys: &[BoxSumCheckPoly<F, E>],
    ) -> Vec<E> {
        #[cfg(feature = "sanity-check")]
        assert_eq!(self.compute_sum(round, polys), claim);

        let AdditiveArray([coeff_0, coeff_2]) = if let Some(eq) = self.eq() {
            let subset_i = eq.subset_i(round);
            chain_par![
                self.d_1.par_iter().map(|(scalar, f)| {
                    let f = &polys[*f];
                    let coeff_0 = op_sum_check_poly!(|f| {
                        (0..f.len())
                            .into_par_iter()
                            .step_by(2)
                            .with_min_len(64)
                            .map(|b| subset_i[b >> 1] * f[b])
                            .sum()
                    });
                    (scalar, AdditiveArray([coeff_0, E::ZERO]))
                }),
                self.d_2.par_iter().map(|(scalar, f, g)| {
                    let (f, g) = (&polys[*f], &polys[*g]);
                    let sum = op_sum_check_polys!(|f, g| {
                        (0..f.len())
                            .into_par_iter()
                            .step_by(2)
                            .with_min_len(64)
                            .map(|b| {
                                let eq_eval = subset_i[b >> 1];
                                let coeff_0 = f[b] * g[b];
                                let coeff_2 = (f[b + 1] - f[b]) * (g[b + 1] - g[b]);
                                AdditiveArray([eq_eval * coeff_0, eq_eval * coeff_2])
                            })
                            .sum()
                    });
                    (scalar, sum)
                })
            ]
            .map(maybe_scale)
            .sum()
        } else {
            chain_par![
                self.d_1.par_iter().map(|(scalar, f)| {
                    let f = &polys[*f];
                    let coeff_0 = op_sum_check_poly!(
                        |f| (0..f.len())
                            .into_par_iter()
                            .step_by(2)
                            .with_min_len(64)
                            .map(|b| f[b])
                            .sum(),
                        |coeff_0| E::from(coeff_0)
                    );
                    (scalar, AdditiveArray([coeff_0, E::ZERO]))
                }),
                self.d_2.par_iter().map(|(scalar, f, g)| {
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
                    (scalar, sum)
                })
            ]
            .map(maybe_scale)
            .sum()
        };
        let coeff_0 = coeff_0 + self.d_0_round_sum(round);

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
        transcript: &mut dyn TranscriptWrite<F, E>,
    ) -> Result<(), Error> {
        transcript.write_felt_ext(&sum[0])?;
        transcript.write_felt_ext(&sum[2])?;
        Ok(())
    }

    fn read_round_poly(
        &self,
        round: usize,
        claim: E,
        transcript: &mut dyn TranscriptRead<F, E>,
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

impl<E: Field> Quadratic<E> {
    pub fn new(num_vars: usize, pairs: Vec<(E, usize, usize)>) -> Self {
        let d_2 = pairs
            .into_iter()
            .map(|(scalar, f, g)| {
                assert_ne!(scalar, E::ZERO);
                if scalar == E::ONE {
                    (None, f, g)
                } else {
                    (Some(scalar), f, g)
                }
            })
            .collect();
        Self {
            num_vars,
            d_0: None,
            d_1: Vec::new(),
            d_2,
            eq: None,
        }
    }

    pub fn parse(num_vars: usize, expression: &Expression<E, usize>) -> Self {
        assert!(expression.degree() <= 2);

        let pairs = expression.evaluate(
            &|constant| {
                (constant != E::ZERO)
                    .then(|| vec![(constant, None, None)])
                    .unwrap_or_default()
            },
            &|poly| vec![(E::ONE, Some(poly), None)],
            &|mut pairs| {
                pairs.iter_mut().for_each(|pair| pair.0 = -pair.0);
                pairs
            },
            &|a, b| [a, b].concat(),
            &|a, b| {
                a.iter()
                    .cartesian_product(b.iter())
                    .map(|(a, b)| {
                        let mut inputs = chain![a.1, a.2, b.1, b.2].sorted();
                        (a.0 * b.0, inputs.next(), inputs.next())
                    })
                    .collect()
            },
        );

        let (d_0, d_1, d_2) = pairs
            .into_iter()
            .fold(BTreeMap::new(), |mut map, pair| {
                map.entry((pair.1, pair.2))
                    .and_modify(|scalar| *scalar += pair.0)
                    .or_insert(pair.0);
                map
            })
            .into_iter()
            .fold(
                (E::ZERO, Vec::new(), Vec::new()),
                |(mut d_0, mut d_1, mut d_2), (inputs, scalar)| {
                    if scalar != E::ZERO {
                        match inputs {
                            (None, None) => d_0 += scalar,
                            (Some(f), g) => {
                                let scalar = (scalar != E::ONE).then_some(scalar);
                                if let Some(g) = g {
                                    d_2.push((scalar, f, g))
                                } else {
                                    d_1.push((scalar, f))
                                }
                            }
                            _ => unreachable!(),
                        }
                    }
                    (d_0, d_1, d_2)
                },
            );

        Self {
            num_vars,
            d_0: (d_0 != E::ZERO).then_some(d_0),
            d_1,
            d_2,
            eq: None,
        }
    }

    pub fn mul_by_eq(mut self, r_eq: &[E], is_proving: bool) -> Self {
        self.eq = Some(EqF::new(r_eq, is_proving));
        self
    }

    pub fn eq(&self) -> Option<&EqF<E>> {
        self.eq.as_ref()
    }

    fn d_0_sum(&self, round: usize) -> E {
        match (self.d_0, self.eq()) {
            (Some(d_0), Some(_)) => d_0,
            (Some(d_0), None) => d_0 * powers(E::ONE.double()).nth(self.num_vars - round).unwrap(),
            _ => E::ZERO,
        }
    }

    fn d_0_round_sum(&self, round: usize) -> E {
        self.d_0_sum(round + 1)
    }
}

fn maybe_scale<F, T>((scalar, value): (impl Borrow<Option<F>>, T)) -> T
where
    F: Copy,
    T: Copy + Mul<F, Output = T>,
{
    scalar
        .borrow()
        .map(|scalar| value * scalar)
        .unwrap_or(value)
}

#[cfg(test)]
mod test {
    use crate::{
        poly::box_dense_poly,
        sum_check::{quadratic::Quadratic, test::run_sum_check, SumCheckPoly},
        util::{arithmetic::Field, dev::rand_vec, expression::Expression},
    };
    use goldilocks::{Goldilocks, GoldilocksExt2};
    use std::iter;

    #[test]
    fn d_2() {
        run_sum_check::<Goldilocks, GoldilocksExt2, _>(|num_vars, mut rng| {
            let scalar = Field::random(&mut rng);
            let g = Quadratic::new(num_vars, vec![(scalar, 0, 1)]);
            let polys = iter::repeat_with(|| rand_vec(1 << num_vars, &mut rng))
                .map(box_dense_poly)
                .take(2);
            (g, polys.map(SumCheckPoly::Base).collect())
        });
    }

    #[test]
    fn d_2_mul_by_eq() {
        run_sum_check::<Goldilocks, GoldilocksExt2, _>(|num_vars, mut rng| {
            let scalar = Field::random(&mut rng);
            let r = rand_vec(num_vars, &mut rng);
            let g = Quadratic::new(num_vars, vec![(scalar, 0, 1)]).mul_by_eq(&r, true);
            let polys = iter::repeat_with(|| rand_vec(1 << num_vars, &mut rng))
                .map(box_dense_poly)
                .take(2);
            (g, polys.map(SumCheckPoly::Base).collect())
        });
    }

    #[test]
    fn parse_d_0() {
        run_sum_check::<Goldilocks, GoldilocksExt2, _>(|num_vars, rng| {
            let d_0 = Field::random(rng);
            let g = Quadratic::parse(num_vars, &Expression::Constant(d_0));
            assert_eq!(g.d_0, Some(d_0));
            assert!(g.d_1.is_empty());
            assert!(g.d_2.is_empty());
            (g, Vec::new())
        });
    }

    #[test]
    fn parse_d_0_mul_by_eq() {
        run_sum_check::<Goldilocks, GoldilocksExt2, _>(|num_vars, mut rng| {
            let d_0 = Field::random(&mut rng);
            let r = rand_vec(num_vars, &mut rng);
            let g = Quadratic::parse(num_vars, &Expression::Constant(d_0)).mul_by_eq(&r, true);
            assert_eq!(g.d_0, Some(d_0));
            assert!(g.d_1.is_empty());
            assert!(g.d_2.is_empty());
            (g, Vec::new())
        });
    }

    #[test]
    fn parse_d_1() {
        run_sum_check::<Goldilocks, GoldilocksExt2, _>(|num_vars, mut rng| {
            let scalar = Field::random(&mut rng);
            let expr = Expression::Data(0) * Expression::Constant(scalar);
            let g = Quadratic::parse(num_vars, &expr);
            assert_eq!(g.d_1, [(Some(scalar), 0)]);
            assert!(g.d_0.is_none());
            assert!(g.d_2.is_empty());
            let f = box_dense_poly(rand_vec(1 << num_vars, &mut rng));
            (g, vec![SumCheckPoly::Base(f)])
        });
    }

    #[test]
    fn parse_d_1_mul_by_eq() {
        run_sum_check::<Goldilocks, GoldilocksExt2, _>(|num_vars, mut rng| {
            let scalar = Field::random(&mut rng);
            let expr = Expression::Data(0) * Expression::Constant(scalar);
            let r = rand_vec(num_vars, &mut rng);
            let g = Quadratic::parse(num_vars, &expr).mul_by_eq(&r, true);
            assert_eq!(g.d_1, [(Some(scalar), 0)]);
            assert!(g.d_0.is_none());
            assert!(g.d_2.is_empty());
            let f = box_dense_poly(rand_vec(1 << num_vars, &mut rng));
            (g, vec![SumCheckPoly::Base(f)])
        });
    }
}
