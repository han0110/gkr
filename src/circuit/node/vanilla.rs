use crate::{
    circuit::node::{CombinedEvalClaim, EvalClaim, Node},
    poly::{eq_eval, MultilinearPoly, PartialEqPoly},
    sum_check::{
        err_unmatched_evaluation, prove_sum_check, quadratic::Quadratic, verify_sum_check,
    },
    transcript::{TranscriptRead, TranscriptWrite},
    util::{chain, hadamard_add, inner_product, izip, izip_par, AdditiveVec, Field, Itertools},
    Error,
};
use rayon::prelude::*;
use std::{borrow::Cow, collections::HashMap, iter};

#[derive(Clone, Debug)]
pub struct VanillaNode<F> {
    log2_sub_input_size: usize,
    log2_sub_output_size: usize,
    log2_reps: usize,
    gates: Vec<VanillaGate<F>>,
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
        assert_eq!(inputs.len(), 1, "Unimplemented");

        let input = inputs[0];
        assert_eq!(input.len(), self.input_size());

        (0..self.output_size())
            .into_par_iter()
            .map(|b_g| {
                let b_x = (b_g >> self.log2_sub_output_size) << self.log2_sub_input_size;
                let gate = &self.gates[b_g % self.gates.len()];
                chain![
                    gate.w_0,
                    gate.w_1
                        .iter()
                        .map(|(s, b_0)| maybe_mul!(s, input[b_x + b_0])),
                    gate.w_2
                        .iter()
                        .map(|(s, b_0, b_1)| maybe_mul!(s, input[b_x + b_0] * input[b_x + b_1])),
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
        assert_eq!(inputs.len(), 1, "Unimplemented");

        let input = MultilinearPoly::new(inputs[0].into());
        let eq_r_gs = self.eq_r_gs(&claim.points, &claim.alphas);
        let eq_r_g_prime = self.eq_r_g_prime(&eq_r_gs);
        let (sub_claim, r_x_0, input_r_x_0) = {
            let claim = claim.value - self.phase_1_eval(&eq_r_gs);
            let f = self.phase_1_poly(&input, &eq_r_g_prime);
            let polys = [Cow::Owned(f), Cow::Borrowed(&input)];
            let (subclaim, r_x_0, evals) = prove_sum_check(&Quadratic, claim, polys, transcript)?;
            transcript.write_felt(&evals[1])?;
            (subclaim, r_x_0, evals[1])
        };
        let eq_r_x_0 = self.eq_r_x(&r_x_0, input_r_x_0);
        let (r_x_1, input_r_x_1) = {
            let claim = sub_claim - self.phase_2_eval(&eq_r_gs, &eq_r_x_0);
            let f = self.phase_2_poly(&eq_r_g_prime, &eq_r_x_0);
            let polys = [f, input].map(Cow::Owned);
            let (_, r_x_1, evals) = prove_sum_check(&Quadratic, claim, polys, transcript)?;
            transcript.write_felt(&evals[1])?;
            (r_x_1, evals[1])
        };

        Ok(vec![vec![
            EvalClaim::new(r_x_0, input_r_x_0),
            EvalClaim::new(r_x_1, input_r_x_1),
        ]])
    }

    fn verify_claim_reduction(
        &self,
        claim: CombinedEvalClaim<F>,
        transcript: &mut dyn TranscriptRead<F>,
    ) -> Result<Vec<Vec<EvalClaim<F>>>, Error> {
        let num_vars = self.log2_input_size();
        let eq_r_gs = self.eq_r_gs(&claim.points, &claim.alphas);
        let (sub_claim, r_x_0, input_r_x_0) = {
            let claim = claim.value - self.phase_1_eval(&eq_r_gs);
            let (sub_claim, r_x_0) = verify_sum_check(&Quadratic, claim, num_vars, transcript)?;
            (sub_claim, r_x_0, transcript.read_felt()?)
        };
        let eq_r_x_0 = self.eq_r_x(&r_x_0, input_r_x_0);
        let (sub_claim, r_x_1, input_r_x_1) = {
            let claim = sub_claim - self.phase_2_eval(&eq_r_gs, &eq_r_x_0);
            let (sub_claim, r_x_1) = verify_sum_check(&Quadratic, claim, num_vars, transcript)?;
            (sub_claim, r_x_1, transcript.read_felt()?)
        };
        let eq_r_x_1 = self.eq_r_x(&r_x_1, input_r_x_1);
        if sub_claim != self.final_eval(&eq_r_gs, &eq_r_x_0, &eq_r_x_1) {
            return Err(err_unmatched_evaluation());
        }

        Ok(vec![vec![
            EvalClaim::new(r_x_0, input_r_x_0),
            EvalClaim::new(r_x_1, input_r_x_1),
        ]])
    }
}

impl<F: Field> VanillaNode<F> {
    pub fn new(log2_sub_input_size: usize, num_reps: usize, gates: Vec<VanillaGate<F>>) -> Self {
        assert!(!gates.is_empty());
        assert!(num_reps != 0);
        gates.iter().for_each(|gate| {
            chain![
                gate.w_1.iter().map(|(_, b_0)| b_0),
                gate.w_2.iter().flat_map(|(_, b_0, b_1)| [b_0, b_1])
            ]
            .for_each(|b| assert!(*b < 1 << log2_sub_input_size))
        });

        let log2_sub_output_size = gates.len().next_power_of_two().ilog2() as usize;
        let log2_reps = num_reps.next_power_of_two().ilog2() as usize;
        let gates = chain![gates, iter::repeat_with(Default::default)]
            .take(1 << log2_sub_output_size)
            .collect_vec();

        Self {
            log2_sub_input_size,
            log2_sub_output_size,
            log2_reps,
            gates,
        }
    }

    pub fn into_boxed(self) -> Box<dyn Node<F>> {
        Box::new(self)
    }

    pub fn log2_reps(&self) -> usize {
        self.log2_reps
    }

    fn eq_r_gs<'a>(&self, r_gs: &'a [Vec<F>], alphas: &[F]) -> Vec<PartialEqPoly<'a, F>> {
        izip!(r_gs, alphas)
            .map(|(r_g, alpha)| PartialEqPoly::new(r_g, self.log2_sub_output_size, *alpha))
            .collect()
    }

    fn eq_r_x<'a>(&self, r_x: &'a [F], input_r_x: F) -> PartialEqPoly<'a, F> {
        PartialEqPoly::new(r_x, self.log2_sub_input_size, input_r_x)
    }

    fn eq_r_g_prime(&self, eq_r_gs: &[PartialEqPoly<F>]) -> Vec<F> {
        eq_r_gs
            .iter()
            .map(PartialEqPoly::expand)
            .reduce(|acc, item| hadamard_add(acc.into(), &item))
            .unwrap()
    }

    fn phase_1_poly(&self, input: &[F], eq_r_g_prime: &[F]) -> MultilinearPoly<F> {
        self.phase_i_poly(&|buf, b_g, b_x, gate| {
            let common = eq_r_g_prime[b_g];
            gate.w_1.iter().for_each(|(s, b_0)| {
                add_or_insert!(buf, b_x + b_0, maybe_mul!(s, common));
            });
            gate.w_2.iter().for_each(|(s, b_0, b_1)| {
                add_or_insert!(buf, b_x + b_0, maybe_mul!(s, common * input[b_x + b_1]))
            });
        })
    }

    fn phase_2_poly(&self, eq_r_g_prime: &[F], eq_r_x_0: &PartialEqPoly<F>) -> MultilinearPoly<F> {
        let eq_r_x_0 = eq_r_x_0.expand();
        self.phase_i_poly(&|buf, b_g, b_x, gate| {
            let common = eq_r_g_prime[b_g];
            gate.w_2.iter().for_each(|(s, b_0, b_1)| {
                add_or_insert!(buf, b_x + b_1, maybe_mul!(s, common * eq_r_x_0[b_x + b_0]))
            });
        })
    }

    fn phase_i_poly<T>(&self, f: &T) -> MultilinearPoly<F>
    where
        T: Fn(&mut HashMap<usize, F>, usize, usize, &VanillaGate<F>) + Send + Sync,
    {
        let buf = (0..self.output_size())
            .into_par_iter()
            .fold_with(HashMap::new(), |mut buf, b_g| {
                let b_x = (b_g >> self.log2_sub_output_size) << self.log2_sub_input_size;
                let gate = &self.gates[b_g % self.gates.len()];
                f(&mut buf, b_g, b_x, gate);
                buf
            })
            .reduce_with(|mut acc, item| {
                item.iter().for_each(|(b, v)| add_or_insert!(acc, *b, *v));
                acc
            })
            .unwrap();

        let mut f = vec![F::ZERO; self.input_size()];
        buf.iter().for_each(|(b, value)| f[*b] += value);
        MultilinearPoly::new(f.into())
    }

    fn phase_1_eval(&self, eq_r_gs: &[PartialEqPoly<F>]) -> F {
        izip_par!(0..self.gates.len(), &self.gates)
            .filter(|(_, gate)| gate.w_0.is_some())
            .map(|(b_g, gate)| gate.w_0.unwrap() * F::sum(eq_r_gs.iter().map(|eq_r_g| eq_r_g[b_g])))
            .sum::<F>()
    }

    fn phase_2_eval(&self, eq_r_gs: &[PartialEqPoly<F>], eq_r_x_0: &PartialEqPoly<F>) -> F {
        self.phase_i_eval(eq_r_gs, &[eq_r_x_0], &|gate| {
            gate.w_1
                .iter()
                .map(|(s, b_0)| maybe_mul!(s, eq_r_x_0[*b_0]))
                .sum()
        })
    }

    fn final_eval(
        &self,
        eq_r_gs: &[PartialEqPoly<F>],
        eq_r_x_0: &PartialEqPoly<F>,
        eq_r_x_1: &PartialEqPoly<F>,
    ) -> F {
        self.phase_i_eval(eq_r_gs, &[eq_r_x_0, eq_r_x_1], &|gate| {
            gate.w_2
                .iter()
                .map(|(s, b_0, b_1)| maybe_mul!(s, eq_r_x_0[*b_0] * eq_r_x_1[*b_1]))
                .sum()
        })
    }

    fn phase_i_eval<T>(
        &self,
        eq_r_gs: &[PartialEqPoly<F>],
        eq_r_xs: &[&PartialEqPoly<F>],
        f: &T,
    ) -> F
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
}

#[derive(Clone, Debug)]
pub struct VanillaGate<F> {
    w_0: Option<F>,
    w_1: Vec<(Option<F>, usize)>,
    w_2: Vec<(Option<F>, usize, usize)>,
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
        w_1: Vec<(Option<F>, usize)>,
        w_2: Vec<(Option<F>, usize, usize)>,
    ) -> Self {
        Self { w_0, w_1, w_2 }
    }

    pub fn constant(constant: F) -> Self {
        Self {
            w_0: Some(constant),
            ..Default::default()
        }
    }

    pub fn add(b_0: usize, b_1: usize) -> Self {
        Self {
            w_1: vec![(None, b_0), (None, b_1)],
            ..Default::default()
        }
    }

    pub fn sub(b_0: usize, b_1: usize) -> Self
    where
        F: Field,
    {
        Self {
            w_1: vec![(None, b_0), (Some(-F::ONE), b_1)],
            ..Default::default()
        }
    }

    pub fn mul(b_0: usize, b_1: usize) -> Self {
        Self {
            w_2: vec![(None, b_0, b_1)],
            ..Default::default()
        }
    }

    pub fn sum(bs: impl IntoIterator<Item = usize>) -> Self {
        Self {
            w_1: bs.into_iter().map(|b| (None, b)).collect(),
            ..Default::default()
        }
    }

    pub fn w_0(&self) -> &Option<F> {
        &self.w_0
    }

    pub fn w_1(&self) -> &[(Option<F>, usize)] {
        &self.w_1
    }

    pub fn w_2(&self) -> &[(Option<F>, usize, usize)] {
        &self.w_2
    }
}

macro_rules! maybe_mul {
    ($s:expr, $item:expr) => {
        $s.map(|s| $item * s).unwrap_or_else(|| $item)
    };
}

macro_rules! add_or_insert {
    ($map:expr, $key:expr, $value:expr) => {{
        $map.entry($key)
            .and_modify(|acc| *acc += $value)
            .or_insert_with(|| $value);
    }};
}

use {add_or_insert, maybe_mul};

#[cfg(test)]
pub mod test {
    use crate::{
        circuit::{
            dag::DirectedAcyclicGraph,
            node::{
                input::InputNode,
                vanilla::{VanillaGate, VanillaNode},
            },
            Circuit,
        },
        test::run_gkr,
        util::{
            chain,
            test::{rand_bool, rand_range, rand_vec, seeded_std_rng},
            Field, Itertools,
        },
    };
    use halo2_curves::bn256::Fr;
    use rand::RngCore;
    use std::iter;

    impl<F: Field> VanillaGate<F> {
        pub fn rand(log2_sub_input_size: usize, mut rng: impl rand::RngCore) -> Self {
            let rand_coeff = |rng: &mut _| rand_bool(rng as &mut _).then(|| F::random(rng));
            let rand_b = |rng: &mut _| rand_range(0..1 << log2_sub_input_size, rng);

            let mut gate = Self::default();
            for _ in 0..rand_range(1..=32, &mut rng) {
                let s = rand_coeff(&mut rng);
                match rand_bool(&mut rng) {
                    false => gate.w_1.push((s, rand_b(&mut rng))),
                    true => gate.w_2.push((s, rand_b(&mut rng), rand_b(&mut rng))),
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
            assert_eq!(circuit.evaluate(inputs.clone()), values);
            run_gkr(&circuit, inputs, &mut rng);
        }
    }

    #[test]
    fn grand_sum() {
        let mut rng = seeded_std_rng();
        for log2_input_size in 1..16 {
            let (circuit, inputs, values) = grand_sum_circuit::<Fr>(log2_input_size, &mut rng);
            assert_eq!(circuit.evaluate(inputs.clone()), values);
            run_gkr(&circuit, inputs, &mut rng);
        }
    }

    #[test]
    fn rand() {
        let mut rng = seeded_std_rng();
        for _ in 1..16 {
            let (circuit, inputs, _) = rand_linear_circuit::<Fr>(&mut rng);
            run_gkr(&circuit, inputs, &mut rng);
        }
    }

    pub fn grand_product_circuit<F: Field>(
        log2_input_size: usize,
        rng: impl RngCore,
    ) -> (Circuit<F>, Vec<Vec<F>>, Vec<Vec<F>>) {
        let succ = |input: &[_]| {
            (input.len() > 1)
                .then(|| Vec::from_iter(input.iter().tuples().map(|(lhs, rhs)| *lhs * rhs)))
        };

        let nodes = chain![
            [InputNode::new(log2_input_size).into_boxed()],
            (0..log2_input_size)
                .rev()
                .map(|idx| VanillaNode::new(1, 1 << idx, vec![VanillaGate::mul(0, 1)]))
                .map(VanillaNode::into_boxed)
        ]
        .collect_vec();
        let circuit = Circuit::new(DirectedAcyclicGraph::linear(nodes));
        let input = rand_vec(1 << log2_input_size, rng);
        let outputs = iter::successors(Some(input.clone()), |input| succ(input)).collect_vec();

        (circuit, vec![input], outputs)
    }

    pub fn grand_sum_circuit<F: Field>(
        log2_input_size: usize,
        rng: impl RngCore,
    ) -> (Circuit<F>, Vec<Vec<F>>, Vec<Vec<F>>) {
        let succ = |input: &[_]| {
            (input.len() > 1)
                .then(|| Vec::from_iter(input.iter().tuples().map(|(lhs, rhs)| *lhs + rhs)))
        };

        let nodes = chain![
            [InputNode::new(log2_input_size).into_boxed()],
            (0..log2_input_size)
                .rev()
                .map(|idx| VanillaNode::new(1, 1 << idx, vec![VanillaGate::add(0, 1)]))
                .map(VanillaNode::into_boxed)
        ]
        .collect_vec();
        let circuit = Circuit::new(DirectedAcyclicGraph::linear(nodes));
        let input = rand_vec(1 << log2_input_size, rng);
        let values = iter::successors(Some(input.clone()), |input| succ(input)).collect_vec();

        (circuit, vec![input], values)
    }

    pub fn rand_linear_circuit<F: Field>(
        mut rng: impl RngCore,
    ) -> (Circuit<F>, Vec<Vec<F>>, Vec<Vec<F>>) {
        let num_nodes = rand_range(2..=16, &mut rng);
        let log2_sizes = iter::repeat_with(|| rand_range(1..=16, &mut rng))
            .take(num_nodes)
            .collect_vec();
        let nodes = chain![
            [InputNode::new(log2_sizes[0]).into_boxed()],
            log2_sizes
                .iter()
                .tuple_windows()
                .map(|(log2_input_size, log2_output_size)| {
                    let num_reps =
                        rand_range(1..=1 << log2_input_size.min(log2_output_size), &mut rng);
                    let log2_reps = num_reps.next_power_of_two().ilog2() as usize;
                    let log2_sub_input_size = log2_input_size - log2_reps;
                    let log2_sub_output_size = log2_output_size - log2_reps;
                    let gates =
                        iter::repeat_with(|| VanillaGate::rand(log2_sub_input_size, &mut rng))
                            .take(1 << log2_sub_output_size)
                            .collect_vec();
                    VanillaNode::new(log2_sub_input_size, num_reps, gates).into_boxed()
                })
        ]
        .collect_vec();
        let circuit = Circuit::new(DirectedAcyclicGraph::linear(nodes));
        let input = rand_vec(1 << log2_sizes[0], rng);
        let values = circuit.evaluate(vec![input.clone()]);

        (circuit, vec![input], values)
    }
}
