use crate::{
    circuit::node::{CombinedEvalClaim, EvalClaim, Node},
    poly::{eq_eval, MultilinearPoly, PartialEqPoly},
    sum_check::{
        err_unmatched_evaluation, prove_sum_check, quadratic::Quadratic, verify_sum_check,
    },
    transcript::{TranscriptRead, TranscriptWrite},
    util::{
        arithmetic::{inner_product, Field},
        chain,
        collection::{AdditiveVec, Hadamard},
        izip, izip_par, Itertools,
    },
    Error,
};
use rayon::prelude::*;
use std::{
    collections::{BTreeSet, HashMap},
    iter,
};

#[derive(Clone, Debug)]
pub struct VanillaNode<F> {
    input_arity: usize,
    log2_sub_input_size: usize,
    log2_sub_output_size: usize,
    log2_reps: usize,
    gates: Vec<VanillaGate<F>>,
    input_indices: Vec<BTreeSet<usize>>,
}

impl<F: Field> Node<F> for VanillaNode<F> {
    fn is_input(&self) -> bool {
        false
    }

    fn log2_input_size(&self) -> usize {
        self.log2_sub_input_size + self.log2_reps
    }

    fn log2_output_size(&self) -> usize {
        self.log2_sub_output_size + self.log2_reps
    }

    fn evaluate(&self, inputs: Vec<&Vec<F>>) -> Vec<F> {
        assert_eq!(inputs.len(), self.input_arity);
        assert!(!inputs.iter().any(|input| input.len() != self.input_size()));

        (0..self.output_size())
            .into_par_iter()
            .map(|b_g| {
                let b_x = (b_g >> self.log2_sub_output_size) << self.log2_sub_input_size;
                let gate = &self.gates[b_g % self.gates.len()];
                chain![
                    gate.w_0,
                    gate.w_1
                        .iter()
                        .map(|(s, (i_0, b_0))| maybe_mul!(s, inputs[*i_0][b_x + b_0])),
                    gate.w_2.iter().map(|(s, (i_0, b_0), (i_1, b_1))| {
                        maybe_mul!(s, inputs[*i_0][b_x + b_0] * inputs[*i_1][b_x + b_1])
                    }),
                ]
                .sum()
            })
            .collect()
    }

    fn prove_claim_reduction<'a>(
        &self,
        claim: CombinedEvalClaim<F>,
        inputs: Vec<&Vec<F>>,
        transcript: &mut (dyn TranscriptWrite<F> + 'a),
    ) -> Result<Vec<Vec<EvalClaim<F>>>, Error> {
        assert_eq!(inputs.len(), self.input_arity);

        let eq_r_gs = self.eq_r_gs(&claim.points, &claim.alphas);
        let eq_r_g_prime = self.eq_r_g_prime(&eq_r_gs);

        let mut claim = claim.value;
        let mut r_xs = Vec::new();
        let mut eq_r_xs = Vec::new();
        let mut input_r_xs = Vec::new();
        for (phase, indices) in izip!(0.., &self.input_indices) {
            let polys = self.sum_check_polys(&inputs, &eq_r_g_prime, &eq_r_xs, &input_r_xs);
            let (subclaim, r_x_i, evals) = {
                let claim = claim - self.sum_check_eval(&eq_r_gs, &eq_r_xs, &input_r_xs);
                prove_sum_check(&Quadratic, claim, polys, transcript)?
            };
            let input_r_x_is = evals.into_iter().skip(indices.len()).collect_vec();
            transcript.write_felts(&input_r_x_is)?;

            claim = subclaim;
            r_xs.push(r_x_i);
            input_r_xs.push((izip!(indices.iter().cloned(), input_r_x_is)).collect());
            if phase == self.input_indices.len() - 1 {
                break;
            }
            eq_r_xs.push(self.eq_r_x(&r_xs[phase], &input_r_xs[phase]));
        }

        Ok(self.input_claims(&r_xs, &input_r_xs))
    }

    fn verify_claim_reduction(
        &self,
        claim: CombinedEvalClaim<F>,
        transcript: &mut dyn TranscriptRead<F>,
    ) -> Result<Vec<Vec<EvalClaim<F>>>, Error> {
        let eq_r_gs = self.eq_r_gs(&claim.points, &claim.alphas);

        let mut claim = claim.value;
        let mut r_xs = Vec::new();
        let mut eq_r_xs = Vec::new();
        let mut input_r_xs = Vec::new();
        for (phase, indices) in izip!(0.., &self.input_indices) {
            let (subclaim, r_x_i) = {
                let claim = claim - self.sum_check_eval(&eq_r_gs, &eq_r_xs, &input_r_xs);
                verify_sum_check(&Quadratic, claim, self.log2_input_size(), transcript)?
            };
            let input_r_x_is = transcript.read_felts(indices.len())?;

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

impl<F: Field> VanillaNode<F> {
    pub fn new(
        input_arity: usize,
        log2_sub_input_size: usize,
        gates: Vec<VanillaGate<F>>,
        num_reps: usize,
    ) -> Self {
        assert!(!gates.is_empty());
        let inputs = Vec::from_iter(gates.iter().flat_map(|gate| {
            chain![
                gate.w_1.iter().map(|w| w.1),
                gate.w_2.iter().flat_map(|w| [w.1, w.2])
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

        let input_indices = gates
            .iter()
            .flat_map(|gate| {
                chain![
                    gate.w_1.iter().map(|w| (0, w.1 .0)),
                    gate.w_2.iter().flat_map(|w| [(0, w.1 .0), (1, w.2 .0)])
                ]
            })
            .unique()
            .fold(Vec::new(), |mut indices, (phase, i)| {
                indices.resize_with(phase + 1, BTreeSet::new);
                indices[phase].insert(i);
                indices
            });

        Self {
            input_arity,
            log2_sub_input_size,
            log2_sub_output_size,
            log2_reps,
            gates,
            input_indices,
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

    fn eq_r_gs(&self, r_gs: &[Vec<F>], alphas: &[F]) -> Vec<PartialEqPoly<F>> {
        izip!(r_gs, alphas)
            .map(|(r_g, alpha)| PartialEqPoly::new(r_g, self.log2_sub_output_size, *alpha))
            .collect()
    }

    fn eq_r_g_prime(&self, eq_r_gs: &[PartialEqPoly<F>]) -> Vec<F> {
        eq_r_gs.iter().map(PartialEqPoly::expand).hada_sum()
    }

    fn eq_r_x(&self, r_x: &[F], input_r_xs: &HashMap<usize, F>) -> PartialEqPoly<F> {
        let scalar = if self.input_arity == 1 {
            *input_r_xs.values().next().unwrap()
        } else {
            F::ONE
        };
        PartialEqPoly::new(r_x, self.log2_sub_input_size, scalar)
    }

    fn sum_check_polys(
        &self,
        inputs: &[&Vec<F>],
        eq_r_g_prime: &[F],
        eq_r_xs: &[PartialEqPoly<F>],
        input_r_xs: &[HashMap<usize, F>],
    ) -> Vec<MultilinearPoly<F>> {
        let phase = eq_r_xs.len();
        let wirings = match phase {
            0 => self.phase_0_wiring(inputs, eq_r_g_prime),
            1 => self.phase_1_wiring(eq_r_g_prime, eq_r_xs, input_r_xs),
            _ => unreachable!(),
        };
        let inputs = self.input_indices[phase]
            .iter()
            .map(|input| MultilinearPoly::new(inputs[*input].clone()));
        chain![wirings, inputs].collect()
    }

    fn phase_0_wiring(&self, inputs: &[&Vec<F>], eq_r_g_prime: &[F]) -> Vec<MultilinearPoly<F>> {
        self.inner_wiring(0, &|b_g, b_x, gate| {
            chain![
                gate.w_1.iter().map(move |(s, (i_0, b_0))| {
                    let value = maybe_mul!(s, eq_r_g_prime[b_g]);
                    (*i_0, b_x + b_0, value)
                }),
                gate.w_2.iter().map(move |(s, (i_0, b_0), (i_1, b_1))| {
                    let value = maybe_mul!(s, eq_r_g_prime[b_g] * inputs[*i_1][b_x + b_1]);
                    (*i_0, b_x + b_0, value)
                }),
            ]
        })
    }

    fn phase_1_wiring(
        &self,
        eq_r_g_prime: &[F],
        eq_r_xs: &[PartialEqPoly<F>],
        input_r_xs: &[HashMap<usize, F>],
    ) -> Vec<MultilinearPoly<F>> {
        let eq_r_x_0 = &eq_r_xs[0].expand();
        self.inner_wiring(1, &|b_g, b_x, gate| {
            gate.w_2.iter().map(move |(s, (i_0, b_0), (i_1, b_1))| {
                let common = eq_r_g_prime[b_g] * eq_r_x_0[b_x + b_0];
                let value = if self.input_arity == 1 {
                    maybe_mul!(s, common)
                } else {
                    maybe_mul!(s, common * input_r_xs[0][i_0])
                };
                (*i_1, b_x + b_1, value)
            })
        })
    }

    fn inner_wiring<'a, T, I>(&'a self, phase: usize, f: &'a T) -> Vec<MultilinearPoly<F>>
    where
        T: (Fn(usize, usize, &'a VanillaGate<F>) -> I) + Send + Sync,
        I: Iterator<Item = (usize, usize, F)>,
    {
        let buf = (0..self.output_size())
            .into_par_iter()
            .fold_with(vec![Vec::new(); self.input_arity], |mut buf, b_g| {
                let b_x = (b_g >> self.log2_sub_output_size) << self.log2_sub_input_size;
                let gate = &self.gates[b_g % self.gates.len()];
                f(b_g, b_x, gate).for_each(|(i, b, v)| buf[i].push((b, v)));
                buf
            })
            .reduce_with(|mut acc, item| {
                izip!(&mut acc, item).for_each(|(acc, item)| acc.extend(item));
                acc
            })
            .unwrap();
        let buf = Vec::from_iter(self.input_indices[phase].iter().map(|idx| &buf[*idx]));
        let mut wirings = vec![vec![F::ZERO; self.input_size()]; buf.len()];
        izip_par!(&mut wirings, buf).for_each(|(w, buf)| buf.iter().for_each(|(b, v)| w[*b] += v));
        wirings.into_iter().map(MultilinearPoly::new).collect()
    }

    fn sum_check_eval(
        &self,
        eq_r_gs: &[PartialEqPoly<F>],
        eq_r_xs: &[PartialEqPoly<F>],
        input_r_xs: &[HashMap<usize, F>],
    ) -> F {
        let phase = eq_r_xs.len();
        match phase {
            0 => self.phase_0_eval(eq_r_gs),
            1 => self.phase_1_eval(eq_r_gs, eq_r_xs, input_r_xs),
            2 => self.phase_2_eval(eq_r_gs, eq_r_xs, input_r_xs),
            _ => unreachable!(),
        }
    }

    fn phase_0_eval(&self, eq_r_gs: &[PartialEqPoly<F>]) -> F {
        izip_par!(0..self.gates.len(), &self.gates)
            .filter(|(_, gate)| gate.w_0.is_some())
            .map(|(b_g, gate)| gate.w_0.unwrap() * F::sum(eq_r_gs.iter().map(|eq_r_g| eq_r_g[b_g])))
            .sum::<F>()
    }

    fn phase_1_eval(
        &self,
        eq_r_gs: &[PartialEqPoly<F>],
        eq_r_xs: &[PartialEqPoly<F>],
        input_r_xs: &[HashMap<usize, F>],
    ) -> F {
        self.inner_eval(eq_r_gs, eq_r_xs, &|gate| {
            F::sum(gate.w_1.iter().map(|(s, (i_0, b_0))| {
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
        eq_r_gs: &[PartialEqPoly<F>],
        eq_r_xs: &[PartialEqPoly<F>],
        input_r_xs: &[HashMap<usize, F>],
    ) -> F {
        self.inner_eval(eq_r_gs, eq_r_xs, &|gate| {
            F::sum(gate.w_2.iter().map(|(s, (i_0, b_0), (i_1, b_1))| {
                let common = eq_r_xs[0][*b_0] * eq_r_xs[1][*b_1];
                if self.input_arity == 1 {
                    maybe_mul!(s, common)
                } else {
                    maybe_mul!(s, common * input_r_xs[0][i_0] * input_r_xs[1][i_1])
                }
            }))
        })
    }

    fn inner_eval<T>(&self, eq_r_gs: &[PartialEqPoly<F>], eq_r_xs: &[PartialEqPoly<F>], f: &T) -> F
    where
        T: Fn(&VanillaGate<F>) -> F + Send + Sync,
    {
        let evals = izip_par!(0..self.gates.len(), &self.gates)
            .fold_with(AdditiveVec::new(eq_r_gs.len()), |mut evals, (b_g, gate)| {
                let v = f(gate);
                izip!(&mut evals[..], eq_r_gs).for_each(|(eval, eq_r_g)| *eval += v * eq_r_g[b_g]);
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
        r_xs: &[Vec<F>],
        input_r_xs: &[HashMap<usize, F>],
    ) -> Vec<Vec<EvalClaim<F>>> {
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

pub type Input = (usize, usize);

#[derive(Clone, Debug)]
pub struct VanillaGate<F> {
    w_0: Option<F>,
    w_1: Vec<(Option<F>, Input)>,
    w_2: Vec<(Option<F>, Input, Input)>,
}

impl<F> Default for VanillaGate<F> {
    fn default() -> Self {
        Self {
            w_0: None,
            w_1: Vec::new(),
            w_2: Vec::new(),
        }
    }
}

impl<F> VanillaGate<F> {
    pub fn new(
        w_0: Option<F>,
        w_1: Vec<(Option<F>, Input)>,
        w_2: Vec<(Option<F>, Input, Input)>,
    ) -> Self {
        Self { w_0, w_1, w_2 }
    }

    pub fn constant(constant: F) -> Self {
        Self::new(Some(constant), Vec::new(), Vec::new())
    }

    pub fn relay(b: Input) -> Self {
        Self::new(None, vec![(None, b)], Vec::new())
    }

    pub fn add(b_0: Input, b_1: Input) -> Self {
        Self::new(None, vec![(None, b_0), (None, b_1)], Vec::new())
    }

    pub fn sub(b_0: Input, b_1: Input) -> Self
    where
        F: Field,
    {
        Self::new(None, vec![(None, b_0), (Some(-F::ONE), b_1)], Vec::new())
    }

    pub fn mul(b_0: Input, b_1: Input) -> Self {
        Self::new(None, Vec::new(), vec![(None, b_0, b_1)])
    }

    pub fn sum(bs: impl IntoIterator<Item = Input>) -> Self {
        let w_1 = bs.into_iter().map(|b| (None, b)).collect();
        Self::new(None, w_1, Vec::new())
    }

    pub fn and(b_0: Input, b_1: Input) -> Self {
        Self::mul(b_0, b_1)
    }

    pub fn xor(b_0: Input, b_1: Input) -> Self
    where
        F: Field,
    {
        let w_1 = vec![(None, b_0), (None, b_1)];
        let w_2 = vec![(Some(-F::ONE.double()), b_0, b_1)];
        Self::new(None, w_1, w_2)
    }

    pub fn xnor(b_0: Input, b_1: Input) -> Self
    where
        F: Field,
    {
        let w_0 = Some(F::ONE);
        let w_1 = vec![(Some(-F::ONE), b_0), (Some(-F::ONE), b_1)];
        let w_2 = vec![(Some(F::ONE.double()), b_0, b_1)];
        Self::new(w_0, w_1, w_2)
    }

    pub fn w_0(&self) -> &Option<F> {
        &self.w_0
    }

    pub fn w_1(&self) -> &[(Option<F>, Input)] {
        &self.w_1
    }

    pub fn w_2(&self) -> &[(Option<F>, Input, Input)] {
        &self.w_2
    }
}

macro_rules! maybe_mul {
    ($s:expr, $item:expr) => {
        $s.map(|s| $item * s).unwrap_or_else(|| $item)
    };
}

use maybe_mul;

#[cfg(test)]
pub mod test {
    use crate::{
        circuit::{
            node::{
                input::InputNode,
                vanilla::{VanillaGate, VanillaNode},
                Node,
            },
            Circuit,
        },
        test::run_gkr,
        util::{
            arithmetic::Field,
            chain, izip,
            test::{rand_bool, rand_range, rand_unique, rand_vec, seeded_std_rng},
            Itertools, RngCore,
        },
    };
    use halo2_curves::bn256::Fr;
    use std::iter;

    impl<F: Field> VanillaNode<F> {
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
                gate.w_1
                    .push((rand_coeff(&mut rng), (i_0, rand_b(&mut rng))))
            });
            gate.w_2.push((
                rand_coeff(&mut rng),
                (0, rand_b(&mut rng)),
                (0, rand_b(&mut rng)),
            ));
            for _ in 0..rand_range(1..=32, &mut rng) {
                let s = rand_coeff(&mut rng);
                match rand_bool(&mut rng) {
                    false => gate.w_1.push((s, (rand_i(&mut rng), rand_b(&mut rng)))),
                    true => gate.w_2.push((
                        s,
                        (rand_i(&mut rng), rand_b(&mut rng)),
                        (rand_i(&mut rng), rand_b(&mut rng)),
                    )),
                }
            }
            gate.w_0 = rand_coeff(&mut rng);
            gate
        }
    }

    #[test]
    fn grand_product() {
        let mut rng = seeded_std_rng();
        for log2_input_size in 1..16 {
            let (circuit, inputs, values) = grand_product_circuit::<Fr>(log2_input_size, &mut rng);
            run_gkr(&circuit, &inputs, &mut rng);
            assert_eq!(circuit.evaluate(inputs), values);
        }
    }

    #[test]
    fn grand_sum() {
        let mut rng = seeded_std_rng();
        for log2_input_size in 1..16 {
            let (circuit, inputs, values) = grand_sum_circuit::<Fr>(log2_input_size, &mut rng);
            run_gkr(&circuit, &inputs, &mut rng);
            assert_eq!(circuit.evaluate(inputs), values);
        }
    }

    #[test]
    fn rand_linear() {
        let mut rng = seeded_std_rng();
        for _ in 1..16 {
            let (circuit, inputs, _) = rand_linear_circuit::<Fr>(&mut rng);
            run_gkr(&circuit, &inputs, &mut rng);
        }
    }

    #[test]
    fn rand_dag() {
        let mut rng = seeded_std_rng();
        for _ in 1..16 {
            let (circuit, inputs, _) = rand_dag_circuit::<Fr>(&mut rng);
            run_gkr(&circuit, &inputs, &mut rng);
        }
    }

    fn grand_product_circuit<F: Field>(
        log2_input_size: usize,
        rng: impl RngCore,
    ) -> (Circuit<F>, Vec<Vec<F>>, Vec<Vec<F>>) {
        let succ = |input: &[_]| {
            (input.len() > 1)
                .then(|| Vec::from_iter(input.iter().tuples().map(|(lhs, rhs)| *lhs * rhs)))
        };

        let nodes = chain![
            [InputNode::new(log2_input_size, 1).into_boxed()],
            (0..log2_input_size)
                .rev()
                .map(|idx| VanillaNode::new(1, 1, vec![VanillaGate::mul((0, 0), (0, 1))], 1 << idx))
                .map(Node::into_boxed)
        ]
        .collect_vec();
        let circuit = Circuit::linear(nodes);
        let input = rand_vec(1 << log2_input_size, rng);
        let outputs = iter::successors(Some(input.clone()), |input| succ(input)).collect_vec();

        (circuit, vec![input], outputs)
    }

    fn grand_sum_circuit<F: Field>(
        log2_input_size: usize,
        rng: impl RngCore,
    ) -> (Circuit<F>, Vec<Vec<F>>, Vec<Vec<F>>) {
        let succ = |input: &[_]| {
            (input.len() > 1)
                .then(|| Vec::from_iter(input.iter().tuples().map(|(lhs, rhs)| *lhs + rhs)))
        };

        let nodes = chain![
            [InputNode::new(log2_input_size, 1).into_boxed()],
            (0..log2_input_size)
                .rev()
                .map(|idx| VanillaNode::new(1, 1, vec![VanillaGate::add((0, 0), (0, 1))], 1 << idx))
                .map(Node::into_boxed)
        ]
        .collect_vec();
        let circuit = Circuit::linear(nodes);
        let input = rand_vec(1 << log2_input_size, rng);
        let values = iter::successors(Some(input.clone()), |input| succ(input)).collect_vec();

        (circuit, vec![input], values)
    }

    fn rand_linear_circuit<F: Field>(
        mut rng: impl RngCore,
    ) -> (Circuit<F>, Vec<Vec<F>>, Vec<Vec<F>>) {
        let num_nodes = rand_range(2..=16, &mut rng);
        let log2_sizes = iter::repeat_with(|| rand_range(1..=16, &mut rng))
            .take(num_nodes)
            .collect_vec();
        let nodes = chain![
            [InputNode::new(log2_sizes[0], 1).into_boxed()],
            log2_sizes
                .iter()
                .tuple_windows()
                .map(|(log2_input_size, log2_output_size)| {
                    VanillaNode::rand(1, *log2_input_size, *log2_output_size, &mut rng).into_boxed()
                })
        ]
        .collect_vec();
        let circuit = Circuit::linear(nodes);
        let input = rand_vec(1 << log2_sizes[0], rng);
        let values = circuit.evaluate(vec![input.clone()]);

        (circuit, vec![input], values)
    }

    fn rand_dag_circuit<F: Field>(mut rng: impl RngCore) -> (Circuit<F>, Vec<Vec<F>>, Vec<Vec<F>>) {
        let num_nodes = rand_range(2..=16, &mut rng);
        let log2_size = rand_range(1..=16, &mut rng);
        let input_arities = (1..num_nodes)
            .map(|idx| rand_range(1..=idx, &mut rng))
            .collect_vec();
        let nodes = chain![
            [InputNode::new(log2_size, 1).into_boxed()],
            input_arities.iter().map(|input_arity| {
                VanillaNode::rand(*input_arity, log2_size, log2_size, &mut rng).into_boxed()
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
                    .for_each(|from| circuit.link(nodes[from], *to))
            });
            assert_eq!(
                circuit.indegs().skip(1).collect_vec(),
                input_arities.to_vec()
            );
            circuit
        };
        let input = rand_vec(1 << log2_size, rng);
        let values = circuit.evaluate(vec![input.clone()]);

        (circuit, vec![input], values)
    }
}
