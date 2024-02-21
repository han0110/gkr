use crate::{
    circuit::node::{CombinedEvalClaim, EvalClaim, Node},
    poly::{
        box_dense_poly, eq_eval, repeated_dense_poly, BoxMultilinearPoly, MultilinearPoly,
        PartialEqPoly,
    },
    sum_check::{
        err_unmatched_evaluation, prove_sum_check, quadratic::Quadratic, verify_sum_check,
        SumCheckPoly,
    },
    transcript::{TranscriptRead, TranscriptWrite},
    util::{
        arithmetic::{inner_product, ExtensionField, Field},
        chain,
        collection::{AdditiveVec, Hadamard},
        expression::Expression,
        izip, izip_par, Itertools,
    },
    Error,
};
use rayon::prelude::*;
use std::{
    collections::{BTreeSet, HashMap},
    iter,
    marker::PhantomData,
};

#[derive(Clone, Debug)]
pub struct VanillaNode<F, E> {
    input_arity: usize,
    log2_sub_input_size: usize,
    log2_sub_output_size: usize,
    log2_reps: usize,
    gates: Vec<VanillaGate<F>>,
    inputs: Vec<BTreeSet<usize>>,
    wirings: Vec<Vec<WiringExpression<F>>>,
    _marker: PhantomData<E>,
}

impl<F: Field, E: ExtensionField<F>> Node<F, E> for VanillaNode<F, E> {
    fn is_input(&self) -> bool {
        false
    }

    fn log2_input_size(&self) -> usize {
        self.log2_sub_input_size + self.log2_reps
    }

    fn log2_output_size(&self) -> usize {
        self.log2_sub_output_size + self.log2_reps
    }

    fn evaluate(
        &self,
        inputs: Vec<&BoxMultilinearPoly<F, E>>,
    ) -> BoxMultilinearPoly<'static, F, E> {
        assert_eq!(inputs.len(), self.input_arity);
        assert!(!inputs.iter().any(|input| input.len() != self.input_size()));

        let output = (0..self.output_size())
            .into_par_iter()
            .map(|b_g| {
                let b_x = (b_g >> self.log2_sub_output_size) << self.log2_sub_input_size;
                let gate = &self.gates[b_g % self.gates.len()];
                chain![
                    gate.d_0,
                    gate.d_1
                        .iter()
                        .map(|(s, (i_0, b_0))| maybe_mul!(s, inputs[*i_0][b_x + b_0])),
                    gate.d_2.iter().map(|(s, (i_0, b_0), (i_1, b_1))| {
                        maybe_mul!(s, inputs[*i_0][b_x + b_0] * inputs[*i_1][b_x + b_1])
                    }),
                ]
                .sum()
            })
            .collect::<Vec<_>>();
        box_dense_poly(output)
    }

    fn prove_claim_reduction(
        &self,
        claim: CombinedEvalClaim<E>,
        inputs: Vec<&BoxMultilinearPoly<F, E>>,
        transcript: &mut dyn TranscriptWrite<F, E>,
    ) -> Result<Vec<Vec<EvalClaim<E>>>, Error> {
        assert_eq!(inputs.len(), self.input_arity);

        let eq_r_gs = self.eq_r_gs(&claim.points, &claim.alphas);
        let eq_r_g_prime = self.eq_r_g_prime(&eq_r_gs);

        let mut claim = claim.value;
        let mut r_xs = Vec::new();
        let mut eq_r_xs = Vec::new();
        let mut input_r_xs = Vec::new();
        for (phase, indices) in izip!(0.., &self.inputs) {
            let (subclaim, r_x_i, evals) = {
                let g = self.sum_check_function(phase);
                let claim = claim - self.sum_check_eval(&eq_r_gs, &eq_r_xs, &input_r_xs);
                let polys = self.sum_check_polys(&inputs, &eq_r_g_prime, &eq_r_xs, &input_r_xs);
                prove_sum_check(&g, claim, polys, transcript)?
            };
            let input_r_x_is = evals.into_iter().skip(indices.len()).collect_vec();
            transcript.write_felt_exts(&input_r_x_is)?;

            claim = subclaim;
            r_xs.push(r_x_i);
            input_r_xs.push((izip!(indices.clone(), input_r_x_is)).collect());
            if phase == self.inputs.len() - 1 {
                break;
            }
            eq_r_xs.push(self.eq_r_x(&r_xs[phase], &input_r_xs[phase]));
        }

        Ok(self.input_claims(&r_xs, &input_r_xs))
    }

    fn verify_claim_reduction(
        &self,
        claim: CombinedEvalClaim<E>,
        transcript: &mut dyn TranscriptRead<F, E>,
    ) -> Result<Vec<Vec<EvalClaim<E>>>, Error> {
        let eq_r_gs = self.eq_r_gs(&claim.points, &claim.alphas);

        let mut claim = claim.value;
        let mut r_xs = Vec::new();
        let mut eq_r_xs = Vec::new();
        let mut input_r_xs = Vec::new();
        for (phase, indices) in izip!(0.., &self.inputs) {
            let (subclaim, r_x_i) = {
                let g = self.sum_check_function(phase);
                let claim = claim - self.sum_check_eval(&eq_r_gs, &eq_r_xs, &input_r_xs);
                verify_sum_check(&g, claim, transcript)?
            };
            let input_r_x_is = transcript.read_felt_exts(indices.len())?;

            claim = subclaim;
            r_xs.push(r_x_i);
            input_r_xs.push((izip!(indices.iter().cloned(), input_r_x_is)).collect());
            eq_r_xs.push(self.eq_r_x(&r_xs[phase], &input_r_xs[phase]));
        }
        if claim != self.sum_check_eval(&eq_r_gs, &eq_r_xs, &input_r_xs) {
            return Err(err_unmatched_evaluation());
        }

        Ok(self.input_claims(&r_xs, &input_r_xs))
    }
}

impl<F: Field, E: ExtensionField<F>> VanillaNode<F, E> {
    pub fn new(
        input_arity: usize,
        log2_sub_input_size: usize,
        gates: Vec<VanillaGate<F>>,
        num_reps: usize,
    ) -> Self {
        assert!(!gates.is_empty());
        let inputs = Vec::from_iter(gates.iter().flat_map(|gate| {
            chain![
                gate.d_1.iter().map(|w| w.1),
                gate.d_2.iter().flat_map(|w| [w.1, w.2])
            ]
        }));
        assert!(!inputs.iter().any(|(_, b)| *b >= 1 << log2_sub_input_size));
        assert_eq!(inputs.iter().map(|(i, _)| i).unique().count(), input_arity);
        assert!(num_reps != 0);

        let log2_sub_output_size = gates.len().next_power_of_two().ilog2() as usize;
        let gates = chain![gates, iter::repeat_with(Default::default)]
            .take(1 << log2_sub_output_size)
            .collect_vec();
        let log2_reps = num_reps.next_power_of_two().ilog2() as usize;

        let inputs = gates
            .iter()
            .flat_map(|gate| {
                chain![
                    gate.d_1.iter().map(|w| (0, w.1 .0)),
                    gate.d_2.iter().flat_map(|w| [(0, w.1 .0), (1, w.2 .0)])
                ]
            })
            .unique()
            .fold(Vec::new(), |mut inputs, (phase, i)| {
                if inputs.len() < phase + 1 {
                    inputs.resize_with(phase + 1, BTreeSet::new);
                }
                inputs[phase].insert(i);
                inputs
            });
        let wirings = wiring_expressions(input_arity, log2_sub_input_size, &gates, &inputs);

        Self {
            input_arity,
            log2_sub_input_size,
            log2_sub_output_size,
            log2_reps,
            gates,
            wirings,
            inputs,
            _marker: PhantomData,
        }
    }

    pub fn log2_sub_input_size(&self) -> usize {
        self.log2_sub_input_size
    }

    pub fn log2_sub_output_size(&self) -> usize {
        self.log2_sub_output_size
    }

    pub fn log2_reps(&self) -> usize {
        self.log2_reps
    }

    fn eq_r_gs(&self, r_gs: &[Vec<E>], alphas: &[E]) -> Vec<PartialEqPoly<E>> {
        izip_par!(r_gs, alphas)
            .map(|(r_g, alpha)| PartialEqPoly::new(r_g, self.log2_sub_output_size, *alpha))
            .collect()
    }

    fn eq_r_g_prime(&self, eq_r_gs: &[PartialEqPoly<E>]) -> BoxMultilinearPoly<'static, E> {
        box_dense_poly(eq_r_gs.par_iter().map(PartialEqPoly::expand).hada_sum())
    }

    fn eq_r_x(&self, r_x: &[E], input_r_xs: &HashMap<usize, E>) -> PartialEqPoly<E> {
        let scalar = if self.input_arity == 1 {
            *input_r_xs.values().next().unwrap()
        } else {
            E::ONE
        };
        PartialEqPoly::new(r_x, self.log2_sub_input_size, scalar)
    }

    fn sum_check_function(&self, phase: usize) -> Quadratic<E> {
        let n = self.inputs[phase].len();
        let pairs = (0..n).map(|idx| (None, idx, n + idx)).collect();
        Quadratic::new(self.log2_input_size(), pairs)
    }

    fn sum_check_polys<'a>(
        &self,
        inputs: &'a [&BoxMultilinearPoly<'a, F, E>],
        eq_r_g_prime: &BoxMultilinearPoly<E>,
        eq_r_xs: &[PartialEqPoly<E>],
        input_r_xs: &[HashMap<usize, E>],
    ) -> SumCheckPolys<'a, F, E> {
        let phase = eq_r_xs.len();
        let wirings = match phase {
            0 => self.phase_0_wiring(inputs, eq_r_g_prime),
            1 => self.phase_1_wiring(eq_r_g_prime, eq_r_xs, input_r_xs),
            _ => unreachable!(),
        };
        let inputs = self.inputs[phase]
            .iter()
            .map(|input| SumCheckPoly::Base(inputs[*input]));
        chain![wirings, inputs].collect()
    }

    fn phase_0_wiring<'a>(
        &self,
        inputs: &'a [&BoxMultilinearPoly<F, E>],
        eq_r_g_prime: &BoxMultilinearPoly<E>,
    ) -> SumCheckPolys<'a, F, E> {
        let data = chain![
            inputs.iter().copied().map(SumCheckPoly::Base),
            [SumCheckPoly::Extension(eq_r_g_prime)]
        ]
        .collect();
        self.inner_wiring(0, data)
    }

    fn phase_1_wiring<'a>(
        &self,
        eq_r_g_prime: &BoxMultilinearPoly<E>,
        eq_r_xs: &[PartialEqPoly<E>],
        input_r_xs: &[HashMap<usize, E>],
    ) -> SumCheckPolys<'a, F, E> {
        let inputs = self.inputs[0]
            .iter()
            .map(|i| repeated_dense_poly([input_r_xs[0][i]], self.log2_input_size()))
            .collect_vec();
        let eq_r_x_0 = box_dense_poly(eq_r_xs[0].expand());
        let data = SumCheckPoly::exts(chain![&inputs, [eq_r_g_prime, &eq_r_x_0]]);
        self.inner_wiring(1, data)
    }

    fn inner_wiring<'a>(
        &self,
        phase: usize,
        data: Vec<SumCheckPoly<F, E, impl MultilinearPoly<F, E>, impl MultilinearPoly<E>>>,
    ) -> SumCheckPolys<'a, F, E> {
        let sub_size = data
            .iter()
            .map(|data| {
                if data.num_vars() == self.log2_input_size() {
                    self.log2_sub_input_size()
                } else {
                    self.log2_sub_output_size()
                }
            })
            .collect_vec();
        let evaluate = |expr: &Expression<F, Wire>, rep: usize| {
            expr.evaluate_felt(&|(idx, b)| {
                let b = (rep << sub_size[idx]) + b;
                match &data[idx] {
                    SumCheckPoly::Base(poly) => E::from(poly[b]),
                    SumCheckPoly::Extension(poly) => poly[b],
                    _ => unreachable!(),
                }
            })
        };
        let wiring = |exprs: &Vec<Vec<_>>| {
            let log2_sub_input_size = self.log2_sub_input_size();
            let sub_input_mask = (1 << log2_sub_input_size) - 1;
            Vec::from_par_iter((0..self.input_size()).into_par_iter().map(|b| {
                let evaluate = move |expr| evaluate(expr, b >> log2_sub_input_size);
                let exprs = &exprs[b & sub_input_mask];
                exprs.par_iter().with_min_len(64).map(evaluate).sum()
            }))
        };
        self.wirings[phase]
            .par_iter()
            .map(wiring)
            .map(box_dense_poly)
            .map(SumCheckPoly::Extension)
            .collect()
    }

    fn sum_check_eval(
        &self,
        eq_r_gs: &[PartialEqPoly<E>],
        eq_r_xs: &[PartialEqPoly<E>],
        input_r_xs: &[HashMap<usize, E>],
    ) -> E {
        let phase = eq_r_xs.len();
        match phase {
            0 => self.phase_0_eval(eq_r_gs),
            1 => self.phase_1_eval(eq_r_gs, eq_r_xs, input_r_xs),
            2 => self.phase_2_eval(eq_r_gs, eq_r_xs, input_r_xs),
            _ => unreachable!(),
        }
    }

    fn phase_0_eval(&self, eq_r_gs: &[PartialEqPoly<E>]) -> E {
        izip_par!(0..self.gates.len(), &self.gates)
            .filter(|(_, gate)| gate.d_0.is_some())
            .map(|(b_g, gate)| E::sum(eq_r_gs.iter().map(|eq_r_g| eq_r_g[b_g])) * gate.d_0.unwrap())
            .sum()
    }

    fn phase_1_eval(
        &self,
        eq_r_gs: &[PartialEqPoly<E>],
        eq_r_xs: &[PartialEqPoly<E>],
        input_r_xs: &[HashMap<usize, E>],
    ) -> E {
        self.inner_eval(eq_r_gs, eq_r_xs, &|gate| {
            E::sum(gate.d_1.iter().map(|(s, (i_0, b_0))| {
                if self.input_arity == 1 {
                    maybe_mul!(s, eq_r_xs[0][*b_0])
                } else {
                    maybe_mul!(s, eq_r_xs[0][*b_0] * input_r_xs[0][i_0])
                }
            }))
        })
    }

    fn phase_2_eval(
        &self,
        eq_r_gs: &[PartialEqPoly<E>],
        eq_r_xs: &[PartialEqPoly<E>],
        input_r_xs: &[HashMap<usize, E>],
    ) -> E {
        self.inner_eval(eq_r_gs, eq_r_xs, &|gate| {
            E::sum(gate.d_2.iter().map(|(s, (i_0, b_0), (i_1, b_1))| {
                let common = eq_r_xs[0][*b_0] * eq_r_xs[1][*b_1];
                if self.input_arity == 1 {
                    maybe_mul!(s, common)
                } else {
                    maybe_mul!(s, common * input_r_xs[0][i_0] * input_r_xs[1][i_1])
                }
            }))
        })
    }

    fn inner_eval<T>(&self, eq_r_gs: &[PartialEqPoly<E>], eq_r_xs: &[PartialEqPoly<E>], f: &T) -> E
    where
        T: Fn(&VanillaGate<F>) -> E + Send + Sync,
    {
        let evals = izip_par!(0..self.gates.len(), &self.gates)
            .fold_with(AdditiveVec::new(eq_r_gs.len()), |mut evals, (b_g, gate)| {
                let v = f(gate);
                izip!(&mut evals[..], eq_r_gs).for_each(|(eval, eq_r_g)| *eval += eq_r_g[b_g] * v);
                evals
            })
            .reduce_with(|acc, item| acc + item)
            .unwrap();
        let eq_r_hi_evals = eq_r_gs
            .iter()
            .map(|eq_r_g| chain![[eq_r_g.r_hi()], eq_r_xs.iter().map(|eq_r_x| eq_r_x.r_hi())])
            .map(eq_eval);
        inner_product(eq_r_hi_evals, &evals[..])
    }

    fn input_claims(
        &self,
        r_xs: &[Vec<E>],
        input_r_xs: &[HashMap<usize, E>],
    ) -> Vec<Vec<EvalClaim<E>>> {
        (0..self.input_arity)
            .map(|input| {
                izip!(r_xs, input_r_xs)
                    .filter(|(_, input_r_xs)| input_r_xs.contains_key(&input))
                    .map(|(r_x_i, input_r_x_is)| {
                        EvalClaim::new(r_x_i.clone(), input_r_x_is[&input])
                    })
                    .collect()
            })
            .collect()
    }
}

type SumCheckPolys<'a, F, E> =
    Vec<SumCheckPoly<F, E, &'a BoxMultilinearPoly<'a, F, E>, BoxMultilinearPoly<'static, E, E>>>;

type WiringExpression<F> = Vec<Vec<Expression<F, Wire>>>;

fn wiring_expressions<F: Field>(
    input_arity: usize,
    log2_sub_input_size: usize,
    gates: &[VanillaGate<F>],
    inputs: &[BTreeSet<usize>],
) -> Vec<Vec<WiringExpression<F>>> {
    let sub_input_size = 1 << log2_sub_input_size;
    let input_idx = inputs
        .iter()
        .map(|indices| HashMap::<_, _>::from_iter(izip!(0.., indices).map(|(idx, i)| (*i, idx))))
        .collect_vec();
    let process = |phase: usize| {
        let mut wirings = vec![vec![Vec::new(); sub_input_size]; inputs[phase].len()];
        let mut push = |i, b: &usize, expr| wirings[input_idx[phase][i]][*b].push(expr);
        let eq_offset = match phase {
            0 => input_arity,
            1 => input_idx[0].len(),
            _ => unreachable!(),
        };
        match phase {
            0 => izip!(0.., gates).for_each(|(b_g, gate)| {
                let eq_r_g_prime = &Expression::Data((eq_offset, b_g));
                gate.d_1.iter().for_each(|(s, (i_0, b_0))| {
                    push(i_0, b_0, maybe_mul_expr!(s, eq_r_g_prime));
                });
                gate.d_2.iter().for_each(|(s, (i_0, b_0), (i_1, b_1))| {
                    let input = Expression::Data((*i_1, *b_1));
                    push(i_0, b_0, maybe_mul_expr!(s, eq_r_g_prime) * input);
                });
            }),
            1 => izip!(0.., gates).for_each(|(b_g, gate)| {
                let eq_r_g_prime = &Expression::Data((eq_offset, b_g));
                gate.d_2.iter().for_each(|(s, (i_0, b_0), (i_1, b_1))| {
                    let eq_r_x_0 = &Expression::Data((eq_offset + 1, *b_0));
                    let comm = maybe_mul_expr!(s, eq_r_g_prime * eq_r_x_0);
                    if input_arity == 1 {
                        push(i_1, b_1, comm);
                    } else {
                        push(i_1, b_1, comm * Expression::Data((input_idx[0][i_0], 0)));
                    };
                });
            }),
            _ => unreachable!(),
        }
        wirings
    };
    (0..inputs.len()).map(process).collect()
}

pub type Wire = (usize, usize);

#[derive(Clone, Debug)]
pub struct VanillaGate<F> {
    d_0: Option<F>,
    d_1: Vec<(Option<F>, Wire)>,
    d_2: Vec<(Option<F>, Wire, Wire)>,
}

impl<F> Default for VanillaGate<F> {
    fn default() -> Self {
        Self {
            d_0: None,
            d_1: Vec::new(),
            d_2: Vec::new(),
        }
    }
}

impl<F> VanillaGate<F> {
    pub fn new(
        d_0: Option<F>,
        d_1: Vec<(Option<F>, Wire)>,
        d_2: Vec<(Option<F>, Wire, Wire)>,
    ) -> Self {
        Self { d_0, d_1, d_2 }
    }

    pub fn constant(constant: F) -> Self {
        Self::new(Some(constant), Vec::new(), Vec::new())
    }

    pub fn relay(w: Wire) -> Self {
        Self::new(None, vec![(None, w)], Vec::new())
    }

    pub fn add(w_0: Wire, w_1: Wire) -> Self {
        Self::new(None, vec![(None, w_0), (None, w_1)], Vec::new())
    }

    pub fn sub(w_0: Wire, w_1: Wire) -> Self
    where
        F: Field,
    {
        Self::new(None, vec![(None, w_0), (Some(-F::ONE), w_1)], Vec::new())
    }

    pub fn mul(w_0: Wire, w_1: Wire) -> Self {
        Self::new(None, Vec::new(), vec![(None, w_0, w_1)])
    }

    pub fn sum(bs: impl IntoIterator<Item = Wire>) -> Self {
        let w_1 = bs.into_iter().map(|b| (None, b)).collect();
        Self::new(None, w_1, Vec::new())
    }

    pub fn and(w_0: Wire, w_1: Wire) -> Self {
        Self::mul(w_0, w_1)
    }

    pub fn xor(w_0: Wire, w_1: Wire) -> Self
    where
        F: Field,
    {
        let d_1 = vec![(None, w_0), (None, w_1)];
        let d_2 = vec![(Some(-F::ONE.double()), w_0, w_1)];
        Self::new(None, d_1, d_2)
    }

    pub fn xnor(w_0: Wire, w_1: Wire) -> Self
    where
        F: Field,
    {
        let d_0 = Some(F::ONE);
        let d_1 = vec![(Some(-F::ONE), w_0), (Some(-F::ONE), w_1)];
        let d_2 = vec![(Some(F::ONE.double()), w_0, w_1)];
        Self::new(d_0, d_1, d_2)
    }

    pub fn d_0(&self) -> &Option<F> {
        &self.d_0
    }

    pub fn d_1(&self) -> &[(Option<F>, Wire)] {
        &self.d_1
    }

    pub fn d_2(&self) -> &[(Option<F>, Wire, Wire)] {
        &self.d_2
    }
}

macro_rules! maybe_mul {
    ($s:expr, $item:expr) => {
        $s.map(|s| $item * s).unwrap_or_else(|| $item)
    };
}

macro_rules! maybe_mul_expr {
    ($s:expr, $item:expr) => {
        $s.map(|s| {
            assert_ne!(s, F::ZERO);
            if s == F::ONE {
                $item.clone()
            } else if s == -F::ONE {
                -$item
            } else {
                $item * Expression::Constant(s)
            }
        })
        .unwrap_or_else(|| $item.clone())
    };
}

use {maybe_mul, maybe_mul_expr};

#[cfg(test)]
pub mod test {
    use crate::{
        circuit::{
            node::{
                input::InputNode,
                vanilla::{VanillaGate, VanillaNode},
                Node,
            },
            test::{run_circuit, TestData},
            Circuit,
        },
        poly::box_dense_poly,
        util::{
            arithmetic::{ExtensionField, Field},
            chain,
            dev::{rand_bool, rand_range, rand_unique, rand_vec},
            izip, Itertools, RngCore,
        },
    };
    use goldilocks::{Goldilocks, GoldilocksExt2};
    use std::iter;

    impl<F: Field, E: ExtensionField<F>> VanillaNode<F, E> {
        fn rand(
            input_arity: usize,
            log2_input_size: usize,
            log2_output_size: usize,
            mut rng: impl RngCore,
        ) -> Self {
            let num_reps = rand_range(1..=1 << log2_input_size.min(log2_output_size), &mut rng);
            let log2_reps = num_reps.next_power_of_two().ilog2() as usize;
            let log2_sub_input_size = log2_input_size - log2_reps;
            let log2_sub_output_size = log2_output_size - log2_reps;
            let gates =
                iter::repeat_with(|| VanillaGate::rand(input_arity, log2_sub_input_size, &mut rng))
                    .take(1 << log2_sub_output_size)
                    .collect();
            VanillaNode::new(input_arity, log2_sub_input_size, gates, num_reps)
        }
    }

    impl<F: Field> VanillaGate<F> {
        fn rand(input_arity: usize, log2_sub_input_size: usize, mut rng: impl RngCore) -> Self {
            let rand_coeff = |rng: &mut _| rand_bool(rng as &mut _).then(|| F::random(rng));
            let rand_i = |rng: &mut _| rand_range(0..input_arity, rng);
            let rand_b = |rng: &mut _| rand_range(0..1 << log2_sub_input_size, rng);

            let mut gate = Self::default();
            (0..input_arity).for_each(|i_0| {
                gate.d_1
                    .push((rand_coeff(&mut rng), (i_0, rand_b(&mut rng))))
            });
            gate.d_2.push((
                rand_coeff(&mut rng),
                (0, rand_b(&mut rng)),
                (0, rand_b(&mut rng)),
            ));
            for _ in 0..rand_range(1..=32, &mut rng) {
                let s = rand_coeff(&mut rng);
                match rand_bool(&mut rng) {
                    false => gate.d_1.push((s, (rand_i(&mut rng), rand_b(&mut rng)))),
                    true => gate.d_2.push((
                        s,
                        (rand_i(&mut rng), rand_b(&mut rng)),
                        (rand_i(&mut rng), rand_b(&mut rng)),
                    )),
                }
            }
            gate.d_0 = rand_coeff(&mut rng);
            gate
        }
    }

    #[test]
    fn grand_product() {
        run_circuit::<Goldilocks, GoldilocksExt2>(grand_product_circuit);
    }

    #[test]
    fn grand_sum() {
        run_circuit::<Goldilocks, GoldilocksExt2>(grand_sum_circuit);
    }

    #[test]
    fn rand_linear() {
        run_circuit::<Goldilocks, GoldilocksExt2>(rand_linear_circuit);
    }

    #[test]
    fn rand_dag() {
        run_circuit::<Goldilocks, GoldilocksExt2>(rand_dag_circuit);
    }

    fn grand_product_circuit<F: Field, E: ExtensionField<F>>(
        log2_input_size: usize,
        rng: &mut impl RngCore,
    ) -> TestData<F, E> {
        let gates = vec![VanillaGate::mul((0, 0), (0, 1))];
        let nodes = chain![
            [InputNode::new(log2_input_size, 1).boxed()],
            (0..log2_input_size)
                .rev()
                .map(|idx| VanillaNode::new(1, 1, gates.clone(), 1 << idx))
                .map(Node::boxed)
        ]
        .collect_vec();
        let circuit = Circuit::linear(nodes);

        let input = rand_vec(1 << log2_input_size, rng);
        let succ = |input: &[_]| {
            (input.len() > 1)
                .then(|| Vec::from_iter(input.iter().tuples().map(|(lhs, rhs)| *lhs * rhs)))
        };
        let values = iter::successors(Some(input.clone()), |input| succ(input))
            .map(box_dense_poly)
            .collect();

        (circuit, vec![box_dense_poly(input)], Some(values))
    }

    fn grand_sum_circuit<F: Field, E: ExtensionField<F>>(
        log2_input_size: usize,
        rng: &mut impl RngCore,
    ) -> TestData<F, E> {
        let gates = vec![VanillaGate::add((0, 0), (0, 1))];
        let nodes = chain![
            [InputNode::new(log2_input_size, 1).boxed()],
            (0..log2_input_size)
                .rev()
                .map(|idx| VanillaNode::new(1, 1, gates.clone(), 1 << idx))
                .map(Node::boxed)
        ]
        .collect_vec();
        let circuit = Circuit::linear(nodes);

        let input = rand_vec(1 << log2_input_size, rng);
        let succ = |input: &[_]| {
            (input.len() > 1)
                .then(|| Vec::from_iter(input.iter().tuples().map(|(lhs, rhs)| *lhs + rhs)))
        };
        let values = iter::successors(Some(input.clone()), |input| succ(input))
            .map(box_dense_poly)
            .collect();

        (circuit, vec![box_dense_poly(input)], Some(values))
    }

    fn rand_linear_circuit<F: Field, E: ExtensionField<F>>(
        _: usize,
        mut rng: &mut impl RngCore,
    ) -> TestData<F, E> {
        let num_nodes = rand_range(2..=16, &mut rng);
        let log2_sizes = iter::repeat_with(|| rand_range(1..=16, &mut rng))
            .take(num_nodes)
            .collect_vec();
        let nodes = chain![
            [InputNode::new(log2_sizes[0], 1).boxed()],
            log2_sizes
                .iter()
                .tuple_windows()
                .map(|(log2_input_size, log2_output_size)| {
                    VanillaNode::rand(1, *log2_input_size, *log2_output_size, &mut rng).boxed()
                })
        ]
        .collect_vec();
        let circuit = Circuit::linear(nodes);

        let input = rand_vec(1 << log2_sizes[0], rng);

        (circuit, vec![box_dense_poly(input)], None)
    }

    fn rand_dag_circuit<F: Field, E: ExtensionField<F>>(
        _: usize,
        mut rng: &mut impl RngCore,
    ) -> TestData<F, E> {
        let num_nodes = rand_range(2..=16, &mut rng);
        let log2_size = rand_range(1..=16, &mut rng);
        let input_arities = (1..num_nodes)
            .map(|idx| rand_range(1..=idx, &mut rng))
            .collect_vec();
        let nodes = chain![
            [InputNode::new(log2_size, 1).boxed()],
            input_arities.iter().map(|input_arity| {
                VanillaNode::rand(*input_arity, log2_size, log2_size, &mut rng).boxed()
            })
        ]
        .collect_vec();
        let circuit = {
            let mut circuit = Circuit::default();
            let nodes = nodes
                .into_iter()
                .map(|node| circuit.insert(node))
                .collect_vec();
            izip!(1.., &nodes[1..], &input_arities).for_each(|(idx, to, input_arity)| {
                rand_unique(*input_arity, |rng| rand_range(0..idx, rng), &mut rng)
                    .into_iter()
                    .for_each(|from| circuit.connect(nodes[from], *to))
            });
            assert_eq!(
                circuit.indegs().skip(1).collect_vec(),
                input_arities.to_vec()
            );
            circuit
        };

        let input = rand_vec(1 << log2_size, rng);

        (circuit, vec![box_dense_poly(input)], None)
    }
}
