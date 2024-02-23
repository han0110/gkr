use crate::{
    circuit::node::{log_up::LogUpState::*, CombinedEvalClaim, EvalClaim, Node},
    izip_par,
    poly::{box_dense_poly, merge, BoxMultilinearPoly, MultilinearPoly},
    sum_check::{
        err_unmatched_evaluation, generic::Generic, prove_sum_check, quadratic::Quadratic,
        verify_sum_check, SumCheckFunction, SumCheckFunctionExt, SumCheckPoly,
    },
    transcript::{Transcript, TranscriptRead, TranscriptWrite},
    util::{
        arithmetic::{inner_product, powers, ExtensionField, Field, ParallelBatchInvert},
        chain,
        expression::Expression,
        izip, Itertools,
    },
    Error,
};
use rayon::prelude::*;
use std::{array::from_fn, cmp::Ordering::*, iter, ops::Index};

#[derive(Clone, Debug)]
pub struct LogUpNode {
    log2_t_size: usize,
    log2_f_size: usize,
    num_fs: usize,
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

        let m_t_sum_check_polys = |layer: usize| {
            if layer + 1 == self.log2_t_size {
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
                let (n, d) = &m_t_fsums[m_t_fsums.len() - layer - 2];
                let mid = n.len() >> 1;
                let [(n_l, n_r), (d_l, d_r)] = [n, d].map(|value| value.split_at(mid));
                [n_l, n_r, d_l, d_r]
                    .map(box_dense_poly::<E, E, _>)
                    .map(SumCheckPoly::<F, E, _, _>::Extension)
            }
            .into()
        };
        let f_sum_check_polys = |layer: usize| {
            if layer + 1 == self.log2_f_size {
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
                        let (n, d) = &polys[polys.len() - layer - 2];
                        let mid = n.len() >> 1;
                        let [(n_l, n_r), (d_l, d_r)] = [n, d].map(|value| value.split_at(mid));
                        [n_l, n_r, d_l, d_r].map(box_dense_poly::<E, E, _>)
                    })
                    .map(SumCheckPoly::<F, E, _, _>::Extension)
                    .collect_vec()
            }
        };

        let mut m_t_claims = if self.log2_t_size == 0 {
            transcript.write_felt(&m[0])?;
            transcript.write_felt(&t[0])?;
            vec![E::from(m[0]), E::from(t[0])]
        } else {
            let (n, d) = m_t_fsums.last().unwrap();
            transcript.write_felt_ext(&n[0])?;
            transcript.write_felt_ext(&d[0])?;
            vec![n[0], d[0]]
        };
        let mut f_claims = izip!(fs, &f_fsums)
            .map(|(f, f_fsums)| {
                if self.log2_f_size == 0 {
                    transcript.write_felt(&f[0])?;
                    Ok(vec![E::from(f[0])])
                } else {
                    let (n, d) = f_fsums.last().unwrap();
                    transcript.write_felt_ext(&n[0])?;
                    transcript.write_felt_ext(&d[0])?;
                    Ok(vec![n[0], d[0]])
                }
            })
            .flatten_ok()
            .try_collect::<_, Vec<_>, _>()?;
        let mut r_m_t = Vec::new();
        let mut r_f = Vec::new();
        for layer in 0..self.log2_t_size.max(self.log2_f_size) {
            let [m_t_state, f_state] = self.log_up_state(layer);

            let (r_prime, m_t_evals, f_evals) = if layer == 0 {
                let m_t_evals = match m_t_state {
                    Interm => {
                        let (n, d) = &m_t_fsums[m_t_fsums.len() - 2];
                        let m_t_evals = vec![n[0], n[1], d[0], d[1]];
                        transcript.write_felt_exts(&m_t_evals)?;
                        m_t_evals
                    }
                    Initial => {
                        let m_t_evals = vec![m[0], m[1], t[0], t[1]];
                        transcript.write_felts(&m_t_evals)?;
                        m_t_evals.into_iter().map_into().collect()
                    }
                    Finished => Vec::new(),
                };
                let f_evals = match f_state {
                    Interm => {
                        let fs = f_fsums.iter().map(|f_fsums| &f_fsums[f_fsums.len() - 2]);
                        let f_evals = fs.flat_map(|(n, d)| [n[0], n[1], d[0], d[1]]).collect_vec();
                        transcript.write_felt_exts(&f_evals)?;
                        f_evals
                    }
                    Initial => {
                        let f_evals = fs.iter().flat_map(|f| [f[0], f[1]]).collect_vec();
                        transcript.write_felts(&f_evals)?;
                        f_evals.into_iter().map_into().collect()
                    }
                    Finished => Vec::new(),
                };
                (vec![], m_t_evals, f_evals)
            } else {
                let (g, claim) = self.sum_check_relation::<_, _, true>(
                    gamma,
                    layer,
                    (&m_t_claims, &r_m_t),
                    (&f_claims, &r_f),
                    transcript,
                );
                let polys = chain![
                    m_t_state.is_proving().then(|| m_t_sum_check_polys(layer)),
                    f_state.is_proving().then(|| f_sum_check_polys(layer)),
                ]
                .flatten();

                let (_, r_prime, mut evals) = prove_sum_check(&g, claim, polys, transcript)?;

                let m_t_evals = m_t_state
                    .is_proving()
                    .then(|| evals.drain(..4).collect_vec())
                    .unwrap_or_default();
                let f_evals = f_state
                    .is_proving()
                    .then(|| evals.drain(..).collect_vec())
                    .unwrap_or_default();

                transcript.write_felt_exts(&m_t_evals)?;
                transcript.write_felt_exts(&f_evals)?;

                (r_prime, m_t_evals, f_evals)
            };

            let mu = transcript.squeeze_challenge();

            if m_t_state.is_proving() {
                m_t_claims = merge(&m_t_evals, &mu);
                r_m_t = chain![r_prime.clone(), [mu]].collect();
            };
            if f_state.is_proving() {
                f_claims = merge(&f_evals, &mu);
                r_f = chain![r_prime.clone(), [mu]].collect();
            };
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

        let (mut m_t_claims, mut f_claims) = self.read_final_claims(gamma, transcript)?;
        let mut r_m_t = Vec::new();
        let mut r_f = Vec::new();
        for layer in 0..self.log2_t_size.max(self.log2_f_size) {
            let [m_t_state, f_state] = self.log_up_state(layer);

            let (r_prime, m_t_evals, f_evals) = if layer == 0 {
                let m_t_evals = match m_t_state {
                    Interm => transcript.read_felt_exts(4)?,
                    Initial => transcript.read_felts_as_exts(4)?,
                    Finished => Vec::new(),
                };
                let f_evals = match f_state {
                    Interm => transcript.read_felt_exts(4 * self.num_fs)?,
                    Initial => transcript.read_felts_as_exts(2 * self.num_fs)?,
                    Finished => Vec::new(),
                };
                (Vec::new(), m_t_evals, f_evals)
            } else {
                let (g, claim) = self.sum_check_relation::<_, _, false>(
                    gamma,
                    layer,
                    (&m_t_claims, &r_m_t),
                    (&f_claims, &r_f),
                    transcript,
                );

                let (sub_claim, r_prime) = verify_sum_check(&g, claim, transcript)?;

                let m_t_evals = match m_t_state {
                    Interm | Initial => transcript.read_felt_exts(4)?,
                    Finished => Vec::new(),
                };
                let f_evals = match f_state {
                    Interm => transcript.read_felt_exts(4 * self.num_fs)?,
                    Initial => transcript.read_felt_exts(2 * self.num_fs)?,
                    Finished => Vec::new(),
                };

                if sub_claim != g.evaluate(&chain![&m_t_evals, &f_evals].copied().collect_vec()) {
                    return Err(err_unmatched_evaluation());
                }

                (r_prime, m_t_evals, f_evals)
            };

            let mu = transcript.squeeze_challenge();

            if m_t_state.is_proving() {
                m_t_claims = merge(&m_t_evals, &mu);
                r_m_t = chain![r_prime.clone(), [mu]].collect();
            };
            if f_state.is_proving() {
                f_claims = merge(&f_evals, &mu);
                r_f = chain![r_prime.clone(), [mu]].collect();
            };
        }

        Ok(chain![
            izip!(iter::repeat(r_m_t), m_t_claims),
            izip!(iter::repeat(r_f), f_claims),
        ]
        .map(|(r, value)| vec![EvalClaim::new(r, value)])
        .collect())
    }
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

    fn log_up_state(&self, layer: usize) -> [LogUpState; 2] {
        [self.log2_t_size, self.log2_f_size].map(|log2_size| match (layer + 1).cmp(&log2_size) {
            Less => Interm,
            Equal => Initial,
            Greater => Finished,
        })
    }

    fn read_final_claims<F: Field, E: ExtensionField<F>>(
        &self,
        gamma: E,
        transcript: &mut (impl TranscriptRead<F, E> + ?Sized),
    ) -> Result<(Vec<E>, Vec<E>), Error> {
        let m_t_claims = if self.log2_t_size == 0 {
            transcript.read_felts_as_exts(2)?
        } else {
            transcript.read_felt_exts(2)?
        };
        let f_claims = if self.log2_f_size == 0 {
            transcript.read_felts_as_exts(self.num_fs)?
        } else {
            transcript.read_felt_exts(2 * self.num_fs)?
        };

        let lhs = if self.log2_t_size == 0 {
            m_t_claims[0] * (m_t_claims[1] + gamma).invert().unwrap()
        } else {
            m_t_claims[0] * m_t_claims[1].invert().unwrap()
        };
        let rhs = if self.log2_f_size == 0 {
            f_claims.iter().map(|f| (gamma + f).invert().unwrap()).sum()
        } else {
            f_claims
                .iter()
                .tuples()
                .map(|(n, d)| *n * d.invert().unwrap())
                .sum()
        };
        (lhs == rhs)
            .then_some((m_t_claims, f_claims))
            .ok_or(Error::InvalidSumCheck(
                "Unmatched LogUp final claims".to_string(),
            ))
    }

    fn sum_check_relation<F, E, const IS_PROVING: bool>(
        &self,
        gamma: E,
        layer: usize,
        (m_t_claims, r_m_t): (&[E], &[E]),
        (f_claims, r_f): (&[E], &[E]),
        transcript: &mut (impl Transcript<F, E> + ?Sized),
    ) -> (Box<dyn SumCheckFunction<F, E>>, E)
    where
        F: Field,
        E: ExtensionField<F>,
    {
        let [m_t_state, f_state] = self.log_up_state(layer);

        let mut pairs = Vec::new();
        let mut claims = Vec::new();
        let mut offset = 0;
        let mut r = [].as_slice();
        if m_t_state.is_proving() {
            let m_t_pair = if matches!(m_t_state, Interm) {
                from_fn(Expression::poly)
            } else {
                let gamma = &Expression::constant(gamma);
                let [m_l, m_r, t_l, t_r] = from_fn(Expression::poly);
                [m_l, m_r, t_l + gamma, t_r + gamma]
            };
            pairs.push(m_t_pair);
            claims.extend(m_t_claims);
            offset += 4;
            r = r_m_t;
        }
        if f_state.is_proving() {
            let f_pairs = if matches!(f_state, Interm) {
                (offset..)
                    .step_by(4)
                    .take(self.num_fs)
                    .map(|offset| from_fn(|idx| Expression::poly(offset + idx)))
                    .collect_vec()
            } else {
                let one = &Expression::constant(E::ONE);
                let gamma = &Expression::constant(gamma);
                (offset..)
                    .step_by(2)
                    .take(self.num_fs)
                    .map(|offset| {
                        let [f_l, f_r] = from_fn(|idx| Expression::poly(offset + idx));
                        [one.clone(), one.clone(), f_l + gamma, f_r + gamma]
                    })
                    .collect_vec()
            };
            pairs.extend(f_pairs);
            claims.extend(f_claims);
            r = r_f;
        }

        let alpha = transcript.squeeze_challenge();
        let g = {
            let exprs = pairs
                .iter()
                .flat_map(|[n_l, n_r, d_l, d_r]| [n_l * d_r + n_r * d_l, d_l * d_r]);
            let expr = Expression::distribute_powers(exprs, alpha);
            if IS_PROVING {
                Quadratic::parse(layer, &expr).mul_by_eq(r, true).boxed()
            } else {
                Generic::new(layer, &expr).mul_by_eq(r, false).boxed()
            }
        };
        let claim = inner_product::<E, E>(&claims, powers(alpha).take(claims.len()));

        (g, claim)
    }
}

#[derive(Debug)]
enum LogUpState {
    Interm,
    Initial,
    Finished,
}

impl LogUpState {
    fn is_proving(&self) -> bool {
        matches!(self, Interm | Initial)
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
                    (E::from(d_r * n_l + d_l * n_r), E::from(d_l * d_r))
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
                    (E::from(d_r + d_l), E::from(d_l * d_r))
                })
                .unzip(),
        }
    }
}

#[cfg(test)]
pub mod test {
    use crate::{
        circuit::{
            node::{input::InputNode, log_up::LogUpNode},
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
        run_circuit::<Goldilocks, GoldilocksExt2>(log_up_circuit::<_, _, 1>);
    }

    #[test]
    fn multiple_input() {
        run_circuit::<Goldilocks, GoldilocksExt2>(log_up_circuit::<_, _, 3>);
    }

    fn log_up_circuit<F: Field, E: ExtensionField<F>, const N: usize>(
        log2_f_size: usize,
        mut rng: &mut impl RngCore,
    ) -> TestData<F, E> {
        let log2_t_size = rand_range(0..2 * log2_f_size, &mut rng);
        let circuit = {
            let mut circuit = Circuit::default();
            let m = circuit.insert(InputNode::new(log2_t_size, 1));
            let t = circuit.insert(InputNode::new(log2_t_size, 1));
            let fs = [(); N].map(|_| circuit.insert(InputNode::new(log2_f_size, 1)));
            let log_up = circuit.insert(LogUpNode::new(log2_t_size, log2_f_size, N));
            chain![[m, t], fs].for_each(|from| circuit.connect(from, log_up));
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
