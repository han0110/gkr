use crate::{
    circuit::node::{CombinedEvalClaim, EvalClaim, Node},
    izip_par,
    poly::{box_dense_poly, eq_eval, eq_poly, merge, BoxMultilinearPoly, MultilinearPoly},
    sum_check::{
        err_unmatched_evaluation, generic::Generic, prove_sum_check, verify_sum_check, SumCheckPoly,
    },
    transcript::{TranscriptRead, TranscriptWrite},
    util::{
        arithmetic::{inner_product, powers, ExtensionField, Field, ParallelBatchInvert},
        chain,
        expression::Expression,
        izip, Itertools,
    },
    Error,
};
use rayon::prelude::*;
use std::{array, iter, ops::Index};

#[derive(Clone, Debug)]
pub struct LogUpNode {
    log2_t_size: usize,
    log2_f_size: usize,
    num_fs: usize,
}

impl LogUpNode {
    pub fn new(log2_t_size: usize, log2_f_size: usize, num_fs: usize) -> Self {
        assert_ne!(num_fs, 0);

        Self {
            log2_t_size,
            log2_f_size,
            num_fs,
        }
    }
}

impl<F: Field, E: ExtensionField<F>> Node<F, E> for LogUpNode {
    fn is_input(&self) -> bool {
        false
    }

    fn log2_input_size(&self) -> usize {
        self.log2_t_size.max(self.log2_f_size)
    }

    fn log2_output_size(&self) -> usize {
        0
    }

    fn evaluate(&self, _: Vec<&BoxMultilinearPoly<F, E>>) -> BoxMultilinearPoly<'static, F, E> {
        box_dense_poly([F::ZERO])
    }

    fn prove_claim_reduction(
        &self,
        _: CombinedEvalClaim<E>,
        inputs: Vec<&BoxMultilinearPoly<F, E>>,
        transcript: &mut dyn TranscriptWrite<F, E>,
    ) -> Result<Vec<Vec<EvalClaim<E>>>, Error> {
        assert_eq!(inputs.len(), self.num_fs + 2);

        let (m, inputs) = inputs.split_first().unwrap();
        let (t, fs) = inputs.split_first().unwrap();
        assert_eq!(t.num_vars(), m.num_vars());
        assert!(!fs.iter().any(|f| f.num_vars() != fs[0].num_vars()));

        let gamma = transcript.squeeze_challenge();

        if cfg!(feature = "sanity-check") {
            let mut lhs = poly_gamma(*t, gamma);
            lhs.par_batch_invert();
            izip_par!(0..lhs.len(), &mut lhs).for_each(|(b, t)| *t *= m[b]);
            let mut rhs = Vec::from_par_iter(fs.par_iter().flat_map(|f| poly_gamma(*f, gamma)));
            rhs.par_batch_invert();
            assert_eq!(lhs.par_iter().sum::<E>(), rhs.par_iter().sum::<E>());
        }

        let m_t_fsums = fractional_sums(Some(m), t, gamma);
        let f_fsums = fs
            .iter()
            .map(|f| fractional_sums(None, f, gamma))
            .collect_vec();

        let m_t_sum_check_polys = |log2_size: usize| {
            if log2_size + 1 == self.log2_t_size {
                let len = 1 << self.log2_t_size;
                let mid = len >> 1;
                let [[m_l, m_r], [t_l, t_r]] = [m, t].map(|p| {
                    p.as_dense()
                        .map(|p| [&p[..mid], &p[mid..]].map(box_dense_poly::<F, E, _>))
                        .unwrap_or_else(|| {
                            [(0..mid), (mid..len)]
                                .map(|range| range.into_par_iter().map(|b| p[b]).collect())
                                .map(box_dense_poly::<F, E, Vec<_>>)
                        })
                });
                [m_l, m_r, t_l, t_r].map(SumCheckPoly::Base)
            } else {
                let (numer, denom) = m_t_fsums.iter().nth_back(log2_size + 1).unwrap();
                let mid = numer.len() >> 1;
                let [(n_l, n_r), (d_l, d_r)] = [numer, denom].map(|value| value.split_at(mid));
                [n_l, n_r, d_l, d_r]
                    .map(|value| box_dense_poly::<E, E, _>(value))
                    .map(SumCheckPoly::<F, E, _, _>::Extension)
            }
        };
        let f_sum_check_polys = |log2_size: usize| {
            if log2_size + 1 == self.log2_f_size {
                let len = 1 << self.log2_f_size;
                let mid = len >> 1;
                fs.iter()
                    .flat_map(|f| {
                        f.as_dense()
                            .map(|f| [&f[..mid], &f[mid..]].map(box_dense_poly::<F, E, _>))
                            .unwrap_or_else(|| {
                                [(0..mid), (mid..len)]
                                    .map(|range| range.into_par_iter().map(|b| f[b]).collect())
                                    .map(box_dense_poly::<F, E, Vec<_>>)
                            })
                    })
                    .map(SumCheckPoly::Base)
                    .collect_vec()
            } else {
                f_fsums
                    .iter()
                    .flat_map(|polys| {
                        let (numer, denom) = polys.iter().nth_back(log2_size + 1).unwrap();
                        let mid = numer.len() >> 1;
                        let [(n_l, n_r), (d_l, d_r)] =
                            [numer, denom].map(|value| value.split_at(mid));
                        [n_l, n_r, d_l, d_r]
                            .map(|value| box_dense_poly::<E, E, _>(value))
                            .map(SumCheckPoly::<F, E, _, _>::Extension)
                    })
                    .collect_vec()
            }
        };

        if self.log2_t_size == 0 {
            transcript.write_felt(&m[0])?;
            transcript.write_felt(&t[0])?;
        } else {
            let (numer, denom) = m_t_fsums.last().unwrap();
            transcript.write_felt_ext(&numer[0])?;
            transcript.write_felt_ext(&denom[0])?;
        }
        for (f, f_fsums) in izip!(fs, &f_fsums) {
            if self.log2_f_size == 0 {
                transcript.write_felt(&f[0])?;
            } else {
                let (numer, denom) = f_fsums.last().unwrap();
                transcript.write_felt_ext(&numer[0])?;
                transcript.write_felt_ext(&denom[0])?;
            }
        }

        let mut m_t_claims = vec![E::from_base(m[0]), E::from_base(t[0])];
        let mut f_claims = fs.iter().map(|f| E::from_base(f[0])).collect_vec();
        let mut r_m_t = Vec::new();
        let mut r_f = Vec::new();
        let mut r = Vec::new();
        for log2_size in 0..self.log2_t_size.max(self.log2_f_size) {
            let is_proving_m_t = log2_size < self.log2_t_size;
            let is_proving_m_t_initial = log2_size + 1 == self.log2_t_size;
            let is_proving_f = log2_size < self.log2_f_size;
            let is_proving_f_initial = log2_size + 1 == self.log2_f_size;

            let (r_prime, m_t_evals, f_evals) = if log2_size == 0 {
                let m_t_evals = if is_proving_m_t_initial {
                    let m_t_fsum_polys = m_t_sum_check_polys(log2_size);
                    let m_t_evals = m_t_fsum_polys
                        .iter()
                        .map(|poly| match poly {
                            SumCheckPoly::Base(poly) => poly[0],
                            _ => unreachable!(),
                        })
                        .collect_vec();
                    transcript.write_felts(&m_t_evals)?;
                    m_t_evals.into_iter().map(E::from_base).collect_vec()
                } else if is_proving_m_t {
                    let m_t_fsum_polys = m_t_sum_check_polys(log2_size);
                    let m_t_evals = m_t_fsum_polys
                        .iter()
                        .map(|poly| match poly {
                            SumCheckPoly::Extension(poly) => poly[0],
                            _ => unreachable!(),
                        })
                        .collect_vec();
                    transcript.write_felt_exts(&m_t_evals)?;
                    m_t_evals
                } else {
                    Vec::new()
                };
                let f_evals = if is_proving_f_initial {
                    let f_fsum_polys = f_sum_check_polys(log2_size);
                    let f_evals = f_fsum_polys
                        .iter()
                        .map(|poly| match poly {
                            SumCheckPoly::Base(poly) => poly[0],
                            _ => unreachable!(),
                        })
                        .collect_vec();
                    transcript.write_felts(&f_evals)?;
                    f_evals.into_iter().map(E::from_base).collect_vec()
                } else if is_proving_f {
                    let f_fsum_polys = f_sum_check_polys(log2_size);
                    let f_evals = f_fsum_polys
                        .iter()
                        .map(|poly| match poly {
                            SumCheckPoly::Extension(poly) => poly[0],
                            _ => unreachable!(),
                        })
                        .collect_vec();
                    transcript.write_felt_exts(&f_evals)?;
                    f_evals
                } else {
                    Vec::new()
                };
                (vec![], m_t_evals, f_evals)
            } else {
                let (g, claim, polys) = {
                    let mut expressions = Vec::new();
                    let mut claims = Vec::new();
                    let mut polys = Vec::new();
                    if is_proving_m_t_initial {
                        let gamma = &Expression::constant(gamma);
                        let [m_l, m_r, t_l, t_r] = &array::from_fn(Expression::poly);
                        expressions.extend([
                            m_l * (t_r + gamma) + m_r * (t_l + gamma),
                            (t_l + gamma) * (t_r + gamma),
                        ]);
                        claims.extend(m_t_claims.clone());
                        polys.extend(m_t_sum_check_polys(log2_size));
                    } else if is_proving_m_t {
                        let [n_l, n_r, d_l, d_r] = &array::from_fn(Expression::poly);
                        expressions.extend([n_l * d_r + n_r * d_l, d_l * d_r]);
                        claims.extend(m_t_claims.clone());
                        polys.extend(m_t_sum_check_polys(log2_size));
                    }
                    if is_proving_f_initial {
                        let gamma = &Expression::constant(gamma);
                        expressions.extend((polys.len()..).step_by(2).take(self.num_fs).flat_map(
                            |offset| {
                                let [f_l, f_r] =
                                    &array::from_fn(|idx| Expression::poly(offset + idx));
                                [(f_r + gamma) + (f_l + gamma), (f_l + gamma) * (f_r + gamma)]
                            },
                        ));
                        claims.extend(f_claims.clone());
                        polys.extend(f_sum_check_polys(log2_size));
                    } else if is_proving_f {
                        expressions.extend((polys.len()..).step_by(4).take(self.num_fs).flat_map(
                            |offset| {
                                let [n_l, n_r, d_l, d_r] =
                                    &array::from_fn(|idx| Expression::poly(offset + idx));
                                [n_l * d_r + n_r * d_l, d_l * d_r]
                            },
                        ));
                        claims.extend(f_claims.clone());
                        polys.extend(f_sum_check_polys(log2_size));
                    }

                    let alpha = transcript.squeeze_challenge();
                    let eq = Expression::poly(polys.len());
                    let expression = eq * Expression::distribute_powers(expressions, alpha);
                    let g = Generic::new(log2_size, &expression);

                    let claim = inner_product::<E, E>(&claims, powers(alpha).take(claims.len()));

                    polys.push(SumCheckPoly::Extension(box_dense_poly(eq_poly(&r, E::ONE))));

                    (g, claim, polys)
                };

                let (_, r_prime, evals) = prove_sum_check(&g, claim, polys, transcript)?;

                let evals = &mut evals.into_iter();
                let m_t_evals = is_proving_m_t
                    .then(|| evals.take(4).collect_vec())
                    .unwrap_or_default();
                let f_evals = is_proving_f
                    .then(|| evals.take(evals.len() - 1).collect_vec())
                    .unwrap_or_default();

                transcript.write_felt_exts(&m_t_evals)?;
                transcript.write_felt_exts(&f_evals)?;

                (r_prime, m_t_evals, f_evals)
            };

            let mu = transcript.squeeze_challenge();

            if is_proving_m_t {
                m_t_claims = merge(&m_t_evals, &mu)
            };
            if is_proving_f {
                f_claims = merge(&f_evals, &mu)
            };

            r = chain![r_prime, [mu]].collect();

            if is_proving_m_t_initial {
                r_m_t = r.clone()
            }
            if is_proving_f_initial {
                r_f = r.clone()
            }
        }

        Ok(chain![
            izip!(iter::repeat(r_m_t), m_t_claims),
            izip!(iter::repeat(r_f), f_claims),
        ]
        .map(|(r, value)| vec![EvalClaim::new(r, value)])
        .collect())
    }

    fn verify_claim_reduction(
        &self,
        _: CombinedEvalClaim<E>,
        transcript: &mut dyn TranscriptRead<F, E>,
    ) -> Result<Vec<Vec<EvalClaim<E>>>, Error> {
        let gamma = transcript.squeeze_challenge();

        let mut m_t_claims = if self.log2_t_size == 0 {
            transcript
                .read_felts(2)?
                .into_iter()
                .map(E::from_base)
                .collect_vec()
        } else {
            transcript.read_felt_exts(2)?
        };
        let mut f_claims = if self.log2_f_size == 0 {
            transcript
                .read_felts(self.num_fs)?
                .into_iter()
                .map(E::from_base)
                .collect_vec()
        } else {
            transcript.read_felt_exts(2 * self.num_fs)?
        };
        let mut r_m_t = Vec::new();
        let mut r_f = Vec::new();
        let mut r = Vec::new();
        for log2_size in 0..self.log2_t_size.max(self.log2_f_size) {
            let is_proving_m_t = log2_size < self.log2_t_size;
            let is_proving_m_t_initial = log2_size + 1 == self.log2_t_size;
            let is_proving_f = log2_size < self.log2_f_size;
            let is_proving_f_initial = log2_size + 1 == self.log2_f_size;

            let (r_prime, m_t_evals, f_evals) = if log2_size == 0 {
                let m_t_evals = if is_proving_m_t_initial {
                    transcript
                        .read_felts(4)?
                        .into_iter()
                        .map(E::from_base)
                        .collect_vec()
                } else if is_proving_m_t {
                    transcript.read_felt_exts(4)?
                } else {
                    Vec::new()
                };
                let f_evals = if is_proving_f_initial {
                    transcript
                        .read_felts(2 * self.num_fs)?
                        .into_iter()
                        .map(E::from_base)
                        .collect_vec()
                } else if is_proving_f {
                    transcript.read_felt_exts(4 * self.num_fs)?
                } else {
                    Vec::new()
                };
                (Vec::new(), m_t_evals, f_evals)
            } else {
                let (g, claim) = {
                    let mut expressions = Vec::new();
                    let mut claims = Vec::new();
                    let mut offset = 0;
                    if is_proving_m_t_initial {
                        let gamma = &Expression::constant(gamma);
                        let [m_l, m_r, t_l, t_r] = &array::from_fn(Expression::poly);
                        expressions.extend([
                            m_l * (t_r + gamma) + m_r * (t_l + gamma),
                            (t_l + gamma) * (t_r + gamma),
                        ]);
                        claims.extend(m_t_claims.clone());
                        offset += 4;
                    } else if is_proving_m_t {
                        let [n_l, n_r, d_l, d_r] = &array::from_fn(Expression::poly);
                        expressions.extend([n_l * d_r + n_r * d_l, d_l * d_r]);
                        claims.extend(m_t_claims.clone());
                        offset += 4;
                    }
                    if is_proving_f_initial {
                        let gamma = &Expression::constant(gamma);
                        expressions.extend((offset..).step_by(2).take(self.num_fs).flat_map(
                            |offset| {
                                let [f_l, f_r] =
                                    &array::from_fn(|idx| Expression::poly(offset + idx));
                                [(f_r + gamma) + (f_l + gamma), (f_l + gamma) * (f_r + gamma)]
                            },
                        ));
                        claims.extend(f_claims.clone());
                        offset += 2 * self.num_fs;
                    } else if is_proving_f {
                        expressions.extend((offset..).step_by(4).take(self.num_fs).flat_map(
                            |offset| {
                                let [n_l, n_r, d_l, d_r] =
                                    &array::from_fn(|idx| Expression::poly(offset + idx));
                                [n_l * d_r + n_r * d_l, d_l * d_r]
                            },
                        ));
                        claims.extend(f_claims.clone());
                        offset += 4 * self.num_fs;
                    }

                    let alpha = transcript.squeeze_challenge();
                    let eq = Expression::poly(offset);
                    let expression = eq * Expression::distribute_powers(expressions, alpha);
                    let g = Generic::new(log2_size, &expression);

                    let claim = inner_product::<E, E>(&claims, powers(alpha).take(claims.len()));

                    (g, claim)
                };

                let (sub_claim, r_prime) = verify_sum_check(&g, claim, transcript)?;

                let m_t_evals = if is_proving_m_t {
                    transcript.read_felt_exts(4)?
                } else {
                    Vec::new()
                };
                let f_evals = if is_proving_f_initial {
                    transcript.read_felt_exts(2 * self.num_fs)?
                } else if is_proving_f {
                    transcript.read_felt_exts(4 * self.num_fs)?
                } else {
                    Vec::new()
                };

                let final_eval = {
                    let eq_eval = eq_eval([r.as_slice(), r_prime.as_slice()]);
                    let evals = chain![&m_t_evals, &f_evals, [&eq_eval]].collect_vec();
                    g.expression().evaluate(
                        &|constant| constant,
                        &|poly| *evals[poly],
                        &|value| -value,
                        &|a, b| a + b,
                        &|a, b| a * b,
                    )
                };
                if sub_claim != final_eval {
                    return Err(err_unmatched_evaluation());
                }

                (r_prime, m_t_evals, f_evals)
            };

            let mu = transcript.squeeze_challenge();

            if is_proving_m_t {
                m_t_claims = merge(&m_t_evals, &mu)
            };
            if is_proving_f {
                f_claims = merge(&f_evals, &mu)
            };

            r = chain![r_prime, [mu]].collect();

            if is_proving_m_t_initial {
                r_m_t = r.clone()
            }
            if is_proving_f_initial {
                r_f = r.clone()
            }
        }

        Ok(chain![
            izip!(iter::repeat(r_m_t), m_t_claims),
            izip!(iter::repeat(r_f), f_claims),
        ]
        .map(|(r, value)| vec![EvalClaim::new(r, value)])
        .collect())
    }
}

fn poly_gamma<F: Field, E: ExtensionField<F>>(
    poly: impl MultilinearPoly<F, E>,
    gamma: E,
) -> Vec<E> {
    (0..1 << poly.num_vars())
        .into_par_iter()
        .map(|b| gamma + poly[b])
        .collect()
}

fn fractional_sums<F: Field, E: ExtensionField<F>>(
    numer: Option<&BoxMultilinearPoly<F, E>>,
    denom: &BoxMultilinearPoly<F, E>,
    gamma: E,
) -> Vec<(Vec<E>, Vec<E>)> {
    return iter::successors(
        (denom.len() > 1).then(|| inner(denom.len(), numer, denom, Some(gamma))),
        |(numer, denom)| (denom.len() > 1).then(|| inner(numer.len(), Some(numer), denom, None)),
    )
    .collect();

    fn inner<F: Field, E: ExtensionField<F>>(
        len: usize,
        numer: Option<&(impl Index<usize, Output = F> + Sync)>,
        denom: &(impl Index<usize, Output = F> + Sync),
        gamma: Option<E>,
    ) -> (Vec<E>, Vec<E>) {
        let mid = len >> 1;
        match (numer, gamma) {
            (Some(numer), Some(gamma)) => (0..mid)
                .into_par_iter()
                .map(|b| {
                    let n_l = numer[b];
                    let n_r = numer[mid + b];
                    let d_l = gamma + denom[b];
                    let d_r = gamma + denom[mid + b];
                    (d_r * n_l + d_l * n_r, d_l * d_r)
                })
                .unzip(),
            (Some(numer), None) => (0..mid)
                .into_par_iter()
                .map(|b| {
                    let n_l = numer[b];
                    let n_r = numer[mid + b];
                    let d_l = denom[b];
                    let d_r = denom[mid + b];
                    (E::from_base(d_r * n_l + d_l * n_r), E::from_base(d_l * d_r))
                })
                .unzip(),
            (None, Some(gamma)) => (0..mid)
                .into_par_iter()
                .map(|b| {
                    let d_l = gamma + denom[b];
                    let d_r = gamma + denom[mid + b];
                    (d_r + d_l, d_l * d_r)
                })
                .unzip(),
            (None, None) => (0..mid)
                .into_par_iter()
                .map(|b| {
                    let d_l = denom[b];
                    let d_r = denom[mid + b];
                    (E::from_base(d_r + d_l), E::from_base(d_l * d_r))
                })
                .unzip(),
        }
    }
}

#[cfg(test)]
pub mod test {
    use crate::{
        circuit::{
            node::{input::InputNode, LogUpNode},
            test::{run_circuit, TestData},
            Circuit,
        },
        poly::box_dense_poly,
        util::{
            arithmetic::{ExtensionField, Field},
            chain,
            dev::{rand_range, rand_vec},
            Itertools, RngCore,
        },
    };
    use goldilocks::{Goldilocks, GoldilocksExt2};
    use std::iter;

    #[test]
    fn single_input() {
        run_circuit::<Goldilocks, GoldilocksExt2>(circuit::<_, _, 1>);
    }

    #[test]
    fn multiple_input() {
        run_circuit::<Goldilocks, GoldilocksExt2>(circuit::<_, _, 3>);
    }

    fn circuit<F: Field, E: ExtensionField<F>, const N: usize>(
        log2_f_size: usize,
        mut rng: &mut impl RngCore,
    ) -> TestData<F, E> {
        let log2_t_size = rand_range(0..2 * log2_f_size, &mut rng);
        let circuit = {
            let mut circuit = Circuit::default();
            let m = circuit.insert(InputNode::new(log2_t_size, 1));
            let t = circuit.insert(InputNode::new(log2_t_size, 1));
            let fs = [(); N].map(|_| circuit.insert(InputNode::new(log2_f_size, 1)));
            let logup = circuit.insert(LogUpNode::new(log2_t_size, log2_f_size, N));
            chain![[m, t], fs].for_each(|from| circuit.connect(from, logup));
            circuit
        };

        let inputs = {
            let mut m = vec![F::ZERO; 1 << log2_t_size];
            let t = rand_vec(1 << log2_t_size, &mut rng);
            let fs = [(); N].map(|_| {
                iter::repeat_with(|| {
                    let idx = rand_range(0..1 << log2_t_size, &mut rng);
                    m[idx] += F::ONE;
                    t[idx]
                })
                .take(1 << log2_f_size)
                .collect_vec()
            });

            chain![[m, t], fs].map(box_dense_poly).collect_vec()
        };

        (circuit, inputs, None)
    }
}
