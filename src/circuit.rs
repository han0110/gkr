use crate::{
    poly::{eq_eval, eq_expand, eq_poly, MultilinearPoly},
    sum_check::SumCheckFunction,
    transcript::{TranscriptRead, TranscriptWrite},
    util::{
        chain, hadamard_add, inner_product, izip, izip_eq, izip_par, AdditiveArray, AdditiveVec,
        Field, Itertools,
    },
    Error,
};
use rayon::prelude::*;
use std::{collections::HashMap, iter, ops::Deref};

#[derive(Clone, Debug)]
pub struct Circuit<F> {
    layers: Vec<Layer<F>>,
}

impl<F: Field> Circuit<F> {
    pub fn new(layers: Vec<Layer<F>>) -> Self {
        assert!(!layers.is_empty());
        chain![layers.iter().tuple_windows()]
            .for_each(|(l_i, l_j)| assert_eq!(l_i.log2_output_len(), l_j.log2_input_len()));

        Self { layers }
    }

    pub fn layers(&self) -> &[Layer<F>] {
        &self.layers
    }

    pub fn log2_input_len(&self) -> usize {
        self.layers.first().unwrap().log2_input_len()
    }

    pub fn log2_output_len(&self) -> usize {
        self.layers.last().unwrap().log2_output_len()
    }

    pub fn outputs(&self, input: &[F]) -> Vec<Vec<F>> {
        let mut layers = self.layers.iter();
        let first = layers.next().map(|layer| layer.output(input));
        iter::successors(first, |input| Some(layers.next()?.output(input))).collect()
    }
}

#[derive(Clone, Debug)]
pub struct Layer<F> {
    log2_sub_input_len: usize,
    log2_sub_output_len: usize,
    log2_num_repetitions: usize,
    gates: Vec<Gate<F>>,
}

impl<F: Field> Layer<F> {
    pub fn new(log2_sub_input_len: usize, num_repetitions: usize, gates: Vec<Gate<F>>) -> Self {
        assert!(!gates.is_empty());
        assert!(num_repetitions != 0);
        gates.iter().for_each(|gate| {
            chain![
                gate.w_1.iter().map(|(_, b_0)| b_0),
                gate.w_2.iter().flat_map(|(_, b_0, b_1)| [b_0, b_1])
            ]
            .for_each(|b| assert!(*b < 1 << log2_sub_input_len))
        });

        let log2_sub_output_len = gates.len().next_power_of_two().ilog2() as usize;
        let log2_num_repetitions = num_repetitions.next_power_of_two().ilog2() as usize;
        let gates = chain![gates, iter::repeat_with(Default::default)]
            .take(1 << log2_sub_output_len)
            .collect_vec();

        Self {
            log2_sub_input_len,
            log2_sub_output_len,
            log2_num_repetitions,
            gates,
        }
    }

    pub fn log2_input_len(&self) -> usize {
        self.log2_sub_input_len + self.log2_num_repetitions
    }

    pub fn input_len(&self) -> usize {
        1 << self.log2_input_len()
    }

    pub fn log2_output_len(&self) -> usize {
        self.log2_sub_output_len + self.log2_num_repetitions
    }

    pub fn output_len(&self) -> usize {
        1 << self.log2_output_len()
    }

    pub fn log2_num_repetitions(&self) -> usize {
        self.log2_num_repetitions
    }

    pub fn log2_sub_input_len(&self) -> usize {
        self.log2_sub_input_len
    }

    pub fn log2_sub_output_len(&self) -> usize {
        self.log2_sub_output_len
    }

    pub fn output(&self, input: &[F]) -> Vec<F> {
        assert_eq!(input.len(), self.input_len());

        (0..self.output_len())
            .into_par_iter()
            .map(|b_g| {
                let b_off = (b_g >> self.log2_sub_output_len()) << self.log2_sub_input_len();
                let gate = &self.gates[b_g % self.gates.len()];
                chain![
                    gate.w_0,
                    gate.w_1(b_off).map(|(s, b_0)| maybe_mul!(s, input[b_0])),
                    gate.w_2(b_off)
                        .map(|(s, b_0, b_1)| maybe_mul!(s, input[b_0] * input[b_1])),
                ]
                .sum()
            })
            .collect()
    }

    pub fn eq_r_gs<'a>(&self, r_gs: &'a [Vec<F>], alphas: &[F]) -> Vec<PartialEqPoly<'a, F>> {
        izip!(r_gs, alphas)
            .map(|(r_g, alpha)| PartialEqPoly::new(r_g, self.log2_sub_output_len, *alpha))
            .collect()
    }

    pub fn eq_r_x<'a>(&self, r_x: &'a [F], input_r_x: F) -> PartialEqPoly<'a, F> {
        PartialEqPoly::new(r_x, self.log2_sub_input_len, input_r_x)
    }

    pub fn eq_r_g_prime(&self, eq_r_gs: &[PartialEqPoly<F>]) -> Vec<F> {
        eq_r_gs
            .iter()
            .map(PartialEqPoly::expand)
            .reduce(|acc, item| hadamard_add(acc.into(), &item))
            .unwrap()
    }

    pub fn phase_1_poly(&self, input: &[F], eq_r_g_prime: &[F]) -> MultilinearPoly<F> {
        self.phase_i_poly(&|buf, b_g, b_off, gate| {
            gate.w_1(b_off).for_each(|(s, b_0)| {
                add_or_insert!(buf, b_0, maybe_mul!(s, eq_r_g_prime[b_g]));
            });
            gate.w_2(b_off).for_each(|(s, b_0, b_1)| {
                add_or_insert!(buf, b_0, maybe_mul!(s, eq_r_g_prime[b_g] * input[b_1]))
            });
        })
    }

    pub fn phase_2_poly(
        &self,
        eq_r_g_prime: &[F],
        eq_r_x_0: &PartialEqPoly<F>,
    ) -> MultilinearPoly<F> {
        let eq_r_x_0 = eq_r_x_0.expand();
        self.phase_i_poly(&|buf, b_g, b_off, gate| {
            gate.w_2(b_off).for_each(|(s, b_0, b_1)| {
                add_or_insert!(buf, b_1, maybe_mul!(s, eq_r_g_prime[b_g] * eq_r_x_0[b_0]))
            });
        })
    }

    fn phase_i_poly<T>(&self, f: &T) -> MultilinearPoly<F>
    where
        T: Fn(&mut HashMap<usize, F>, usize, usize, &Gate<F>) + Send + Sync,
    {
        let buf = (0..self.output_len())
            .into_par_iter()
            .fold_with(HashMap::new(), |mut buf, b_g| {
                let b_off = (b_g >> self.log2_sub_output_len()) << self.log2_sub_input_len();
                let gate = &self.gates[b_g % self.gates.len()];
                f(&mut buf, b_g, b_off, gate);
                buf
            })
            .reduce_with(|mut acc, item| {
                item.iter().for_each(|(b, v)| add_or_insert!(acc, *b, *v));
                acc
            })
            .unwrap();

        let mut f = vec![F::ZERO; self.input_len()];
        buf.iter().for_each(|(b, value)| f[*b] += value);
        MultilinearPoly::new(f.into())
    }

    pub fn phase_1_eval(&self, eq_r_gs: &[PartialEqPoly<F>]) -> F {
        izip_par!(0..self.gates.len(), &self.gates)
            .filter(|(_, gate)| gate.w_0.is_some())
            .map(|(b_g, gate)| gate.w_0.unwrap() * F::sum(eq_r_gs.iter().map(|eq_r_g| eq_r_g[b_g])))
            .sum::<F>()
    }

    pub fn phase_2_eval(&self, eq_r_gs: &[PartialEqPoly<F>], eq_r_x_0: &PartialEqPoly<F>) -> F {
        self.phase_i_eval(eq_r_gs, &[eq_r_x_0], &|gate| {
            gate.w_1
                .iter()
                .map(|(s, b_0)| maybe_mul!(s, eq_r_x_0[*b_0]))
                .sum()
        })
    }

    pub fn final_eval(
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

    pub fn phase_i_eval<T>(
        &self,
        eq_r_gs: &[PartialEqPoly<F>],
        eq_r_xs: &[&PartialEqPoly<F>],
        f: &T,
    ) -> F
    where
        T: Fn(&Gate<F>) -> F + Send + Sync,
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
            .map(|eq_r_g| chain![[eq_r_g.r_hi], eq_r_xs.iter().map(|eq_r_x| eq_r_x.r_hi)])
            .map(eq_eval);
        inner_product(eq_r_hi_evals, &evals[..])
    }
}

pub struct PartialEqPoly<'a, F> {
    r_hi: &'a [F],
    evals: Vec<F>,
}

impl<'a, F> Deref for PartialEqPoly<'a, F> {
    type Target = Vec<F>;

    fn deref(&self) -> &Self::Target {
        &self.evals
    }
}

impl<'a, F: Field> PartialEqPoly<'a, F> {
    fn new(r: &'a [F], mid: usize, scalar: F) -> Self {
        let (r_lo, r_hi) = r.split_at(mid);
        PartialEqPoly {
            r_hi,
            evals: eq_poly(r_lo, scalar),
        }
    }

    fn expand(&self) -> Vec<F> {
        eq_expand(&self.evals, self.r_hi)
    }
}

impl<F: Field> SumCheckFunction<F> for Layer<F> {
    fn degree(&self) -> usize {
        2
    }

    fn compute_sum(&self, claim: F, polys: &[MultilinearPoly<F>]) -> Vec<F> {
        assert_eq!(polys.len(), 2);

        if cfg!(feature = "sanity-check") {
            let polys = polys.iter().map(|poly| poly.as_slice()).collect_vec();
            assert_eq!(
                claim,
                F::sum(izip_eq!(polys[0], polys[1]).map(|(a, b)| *a * b))
            )
        }

        let AdditiveArray([coeff_0, coeff_2]) =
            izip_par!(&polys[0][..], &polys[0][1..], &polys[1][..], &polys[1][1..])
                .step_by(2)
                .fold_with(AdditiveArray::default(), |mut coeffs, values| {
                    let (a_lo, a_hi, b_lo, b_hi) = values;
                    coeffs[0] += *a_lo * b_lo;
                    coeffs[1] += (*a_hi - a_lo) * (*b_hi - b_lo);
                    coeffs
                })
                .sum();

        vec![coeff_0, claim - coeff_0.double() - coeff_2, coeff_2]
    }

    fn write_sum(&self, sum: &[F], transcript: &mut impl TranscriptWrite<F>) -> Result<(), Error> {
        transcript.write_felt(&sum[0])?;
        transcript.write_felt(&sum[2])?;
        Ok(())
    }

    fn read_sum(&self, claim: F, transcript: &mut impl TranscriptRead<F>) -> Result<Vec<F>, Error> {
        let mut sum = [F::ZERO; 3];
        sum[0] = transcript.read_felt()?;
        sum[2] = transcript.read_felt()?;
        sum[1] = claim - sum[0].double() - sum[2];
        Ok(sum.to_vec())
    }
}

#[derive(Clone, Debug)]
pub struct Gate<F> {
    w_0: Option<F>,
    w_1: Vec<(Option<F>, usize)>,
    w_2: Vec<(Option<F>, usize, usize)>,
}

impl<F> Default for Gate<F> {
    fn default() -> Self {
        Self {
            w_0: None,
            w_1: Vec::new(),
            w_2: Vec::new(),
        }
    }
}

impl<F> Gate<F> {
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

    pub fn mul(b_0: usize, b_1: usize) -> Self {
        Self {
            w_2: vec![(None, b_0, b_1)],
            ..Default::default()
        }
    }

    fn w_1(&self, b_off: usize) -> impl Iterator<Item = (Option<&F>, usize)> {
        self.w_1
            .iter()
            .map(move |(s, b_0)| (s.as_ref(), b_off + b_0))
    }

    fn w_2(&self, b_off: usize) -> impl Iterator<Item = (Option<&F>, usize, usize)> {
        self.w_2
            .iter()
            .map(move |(s, b_0, b_1)| (s.as_ref(), b_off + b_0, b_off + b_1))
    }
}

#[cfg(test)]
impl<F: Field> Gate<F> {
    pub fn rand(log2_sub_input_len: usize, mut rng: impl rand::RngCore) -> Self {
        use crate::util::test::{rand_bool, rand_range};
        let rand_coeff = |rng: &mut _| rand_bool(rng as &mut _).then(|| F::random(rng));
        let rand_b = |rng: &mut _| rand_range(0..1 << log2_sub_input_len, rng);

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
        circuit::{Circuit, Gate, Layer},
        util::{
            test::{rand_range, rand_vec, seeded_std_rng},
            Field, Itertools,
        },
    };
    use halo2_curves::bn256::Fr;
    use rand::RngCore;
    use std::iter;

    #[test]
    fn grand_product() {
        let mut rng = seeded_std_rng();
        for log2_input_len in 1..16 {
            let (circuit, input, outputs) = grand_product_circuit(log2_input_len, &mut rng);
            check_circuit_output::<Fr>(&circuit, &input, &outputs);
        }
    }

    #[test]
    fn grand_sum() {
        let mut rng = seeded_std_rng();
        for log2_input_len in 1..16 {
            let (circuit, input, outputs) = grand_sum_circuit(log2_input_len, &mut rng);
            check_circuit_output::<Fr>(&circuit, &input, &outputs);
        }
    }

    fn check_circuit_output<F: Field>(circuit: &Circuit<F>, input: &[F], outputs: &[Vec<F>]) {
        assert_eq!(circuit.outputs(input), outputs);
    }

    pub fn grand_product_circuit<F: Field>(
        log2_input_len: usize,
        rng: impl RngCore,
    ) -> (Circuit<F>, Vec<F>, Vec<Vec<F>>) {
        let up = |input: &[_]| {
            (input.len() > 1)
                .then(|| Vec::from_iter(input.iter().tuples().map(|(lhs, rhs)| *lhs * rhs)))
        };

        let layers = (1..=log2_input_len)
            .rev()
            .map(|idx| Layer::new(1, 1 << (idx - 1), vec![Gate::mul(0, 1)]))
            .collect_vec();
        let circuit = Circuit::new(layers);
        let input = rand_vec(1 << log2_input_len, rng);
        let outputs = iter::successors(up(&input), |input| up(input)).collect_vec();

        (circuit, input, outputs)
    }

    pub fn grand_sum_circuit<F: Field>(
        log2_input_len: usize,
        rng: impl RngCore,
    ) -> (Circuit<F>, Vec<F>, Vec<Vec<F>>) {
        let up = |input: &[_]| {
            (input.len() > 1)
                .then(|| Vec::from_iter(input.iter().tuples().map(|(lhs, rhs)| *lhs + rhs)))
        };

        let layers = (1..=log2_input_len)
            .rev()
            .map(|idx| Layer::new(1, 1 << (idx - 1), vec![Gate::add(0, 1)]))
            .collect_vec();
        let circuit = Circuit::new(layers);
        let input = rand_vec(1 << log2_input_len, rng);
        let outputs = iter::successors(up(&input), |input| up(input)).collect_vec();

        (circuit, input, outputs)
    }

    pub fn rand_circuit<F: Field>(mut rng: impl RngCore) -> (Circuit<F>, Vec<F>, Vec<Vec<F>>) {
        let num_layers = rand_range(1..=16, &mut rng);
        let log2_input_lens = iter::repeat_with(|| rand_range(1..=16, &mut rng))
            .take(num_layers + 1)
            .collect_vec();
        let layers = log2_input_lens
            .iter()
            .tuple_windows()
            .map(|(log2_input_len, log2_output_len)| {
                let num_repetitions =
                    rand_range(1..=1 << log2_input_len.min(log2_output_len), &mut rng);
                let log2_num_repetitions = num_repetitions.next_power_of_two().ilog2() as usize;
                let log2_sub_input_len = log2_input_len - log2_num_repetitions;
                let log2_sub_output_len = log2_output_len - log2_num_repetitions;
                let gates = iter::repeat_with(|| Gate::rand(log2_sub_input_len, &mut rng))
                    .take(1 << log2_sub_output_len)
                    .collect_vec();
                Layer::new(log2_sub_input_len, num_repetitions, gates)
            })
            .collect_vec();
        let circuit = Circuit::new(layers);
        let input = rand_vec(1 << log2_input_lens[0], rng);
        let output = circuit.outputs(&input);

        (circuit, input, output)
    }
}
