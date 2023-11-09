use crate::{
    poly::{eq_eval, eq_poly, MultilinearPoly},
    sum_check::Function,
    transcript::{TranscriptRead, TranscriptWrite},
    util::{
        chain, div_ceil, izip, izip_eq, num_threads, parallelize, parallelize_iter, Field,
        Itertools,
    },
    Error,
};
use std::{array, collections::HashMap, iter};

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

        let mut output = vec![F::ZERO; self.output_len()];
        parallelize(&mut output, |(output, start)| {
            let gates = self.gates.iter().cycle().skip(start % self.gates.len());
            izip!(start.., output, gates).for_each(|(b_g, output, gate)| {
                let b_off = (b_g >> self.log2_sub_output_len()) << self.log2_sub_input_len();
                if let Some(w_0) = gate.w_0 {
                    *output += w_0
                }
                gate.w_1(b_off)
                    .for_each(|(s, b_0)| *output += maybe_mul!(s, input[b_0]));
                gate.w_2(b_off)
                    .for_each(|(s, b_0, b_1)| *output += maybe_mul!(s, input[b_0] * input[b_1]));
            });
        });
        output
    }

    pub fn eq_r_g_prime(&self, r_gs: &[Vec<F>; 2], alphas: &[F; 2]) -> Vec<F> {
        let mut eq_r_g_0 = eq_poly(&r_gs[0], alphas[0]);
        if alphas[1].is_zero_vartime() {
            eq_r_g_0
        } else {
            let eq_r_g_1 = eq_poly(&r_gs[1], alphas[1]);
            parallelize(&mut eq_r_g_0, |(eq_r_g_0, start)| {
                izip!(eq_r_g_0, &eq_r_g_1[start..]).for_each(|(lhs, rhs)| *lhs += rhs)
            });
            eq_r_g_0
        }
    }

    pub fn phase_1_polys(&self, input: &[F], eq_r_g_prime: &[F]) -> [MultilinearPoly<F>; 2] {
        self.phase_i_polys(&|buf, b_g, b_off, gate| {
            let common = eq_r_g_prime[b_g];
            if let Some(w_0) = gate.w_0 {
                add_or_insert!(buf[0], b_off, w_0 * common)
            }
            gate.w_1(b_off).for_each(|(s, b_0)| {
                add_or_insert!(buf[1], b_0, maybe_mul!(s, common));
            });
            gate.w_2(b_off).for_each(|(s, b_0, b_1)| {
                add_or_insert!(buf[1], b_0, maybe_mul!(s, common * input[b_1]))
            });
        })
    }

    pub fn phase_2_polys(
        &self,
        eq_r_g_prime: &[F],
        r_x_0: &[F],
        input_r_x_0: &F,
    ) -> [MultilinearPoly<F>; 2] {
        let eq_r_x_0 = eq_poly(r_x_0, F::ONE);
        self.phase_i_polys(&|buf, b_g, b_off, gate| {
            let common = eq_r_g_prime[b_g] * input_r_x_0;
            if let Some(w_0) = gate.w_0 {
                add_or_insert!(buf[0], b_off, w_0 * eq_r_g_prime[b_g] * eq_r_x_0[b_off])
            }
            gate.w_1(b_off).for_each(|(s, b_0)| {
                add_or_insert!(buf[0], b_off, maybe_mul!(s, common * eq_r_x_0[b_0]))
            });
            gate.w_2(b_off).for_each(|(s, b_0, b_1)| {
                add_or_insert!(buf[1], b_1, maybe_mul!(s, common * eq_r_x_0[b_0]))
            });
        })
    }

    fn phase_i_polys<T>(&self, f: &T) -> [MultilinearPoly<F>; 2]
    where
        T: Fn(&mut [HashMap<usize, F>; 2], usize, usize, &Gate<F>) + Send + Sync,
    {
        let num_threads = num_threads().min(self.output_len());
        let chunk_size = div_ceil(self.output_len(), num_threads);
        let mut buf = vec![Default::default(); num_threads];
        parallelize_iter(
            izip!(&mut buf, (0..).step_by(chunk_size)),
            |(buf, start)| {
                let b_gs = start..(start + chunk_size).min(self.output_len());
                let gates = self.gates.iter().cycle().skip(start % self.gates.len());
                izip!(b_gs, gates).for_each(|(b_g, gate)| {
                    let b_off = (b_g >> self.log2_sub_output_len()) << self.log2_sub_input_len();
                    f(buf, b_g, b_off, gate)
                })
            },
        );

        let buf = (0..2).map(|idx| buf.iter().map(move |buf| &buf[idx]));
        let mut fs = array::from_fn(|_| vec![F::ZERO; self.input_len()]);
        parallelize_iter(izip_eq!(&mut fs, buf), |(h, buf)| {
            buf.flatten().for_each(|(b, value)| h[*b] += value)
        });
        fs.map(|f| MultilinearPoly::new(f.into()))
    }

    pub fn evaluate(
        &self,
        r_gs: &[Vec<F>; 2],
        alphas: &[F; 2],
        r_xs: &[Vec<F>; 2],
        [input_r_x_0, input_r_x_1]: &[F; 2],
    ) -> F {
        let (r_g_los, r_g_his) = self.r_g_lo_hi(r_gs);
        let (r_x_los, r_x_his) = self.r_x_lo_hi(r_xs);
        let [eq_r_g_0, eq_r_g_1] = array::from_fn(|idx| eq_poly(r_g_los[idx], alphas[idx]));
        let [eq_r_x_0, eq_r_x_1] = array::from_fn(|idx| eq_poly(r_x_los[idx], F::ONE));

        let [w_0, w_1, w_2] = {
            let mut lo = [[F::ZERO; 2]; 3];
            self.gates.iter().enumerate().for_each(|(b_g, gate)| {
                let common = [
                    gate.w_0.map(|w_0| w_0 * eq_r_x_0[0] * eq_r_x_1[0]),
                    gate.w_1(0)
                        .map(|(s, b_0)| maybe_mul!(s, eq_r_x_0[b_0]))
                        .reduce(|acc, v| acc + v)
                        .map(|acc| acc * eq_r_x_1[0]),
                    gate.w_2(0)
                        .map(|(s, b_0, b_1)| maybe_mul!(s, eq_r_x_0[b_0] * eq_r_x_1[b_1]))
                        .reduce(|acc, v| acc + v),
                ];
                izip!(&mut lo, common).for_each(|(lo, common)| {
                    if let Some(common) = common {
                        lo[0] += common * eq_r_g_0[b_g];
                        (!alphas[1].is_zero_vartime()).then(|| lo[1] += common * eq_r_g_1[b_g]);
                    }
                })
            });
            let hi = r_g_his.map(|r_g_hi| eq_eval([r_g_hi, r_x_his[0], r_x_his[1]]));
            lo.map(|lo| F::sum((0..2).map(|idx| lo[idx] * hi[idx])))
        };

        w_0 + w_1 * input_r_x_0 + w_2 * input_r_x_0 * input_r_x_1
    }

    fn r_g_lo_hi<'a>(&self, r_gs: &'a [Vec<F>; 2]) -> ([&'a [F]; 2], [&'a [F]; 2]) {
        self.r_lo_hi(r_gs, self.log2_sub_output_len)
    }

    fn r_x_lo_hi<'a>(&self, r_xs: &'a [Vec<F>; 2]) -> ([&'a [F]; 2], [&'a [F]; 2]) {
        self.r_lo_hi(r_xs, self.log2_sub_input_len)
    }

    fn r_lo_hi<'a>(&self, rs: &'a [Vec<F>; 2], mid: usize) -> ([&'a [F]; 2], [&'a [F]; 2]) {
        (
            array::from_fn(|idx| &rs[idx][..mid]),
            array::from_fn(|idx| &rs[idx][mid..]),
        )
    }
}

impl<F: Field> Function<F> for Layer<F> {
    fn degree(&self) -> usize {
        2
    }

    fn compute_sum(&self, claim: F, polys: &[MultilinearPoly<F>]) -> Vec<F> {
        assert_eq!(polys.len(), 3);

        if cfg!(feature = "sanity-check") {
            let polys = polys.iter().map(|poly| poly.as_slice()).collect_vec();
            assert_eq!(
                claim,
                F::sum(izip_eq!(polys[0], polys[1], polys[2]).map(|(a, b, c)| *a + *b * c))
            )
        }

        let num_threads = num_threads().min(polys[0].len() >> 1);
        let chunk_size = div_ceil(polys[0].len() >> 1, num_threads);

        let mut partials = vec![[F::ZERO; 3]; num_threads];
        parallelize_iter(
            izip!(&mut partials, (0..).step_by(chunk_size << 1)),
            |(partial, start)| {
                let [a, b, c] = array::from_fn(|idx| polys[idx][start..].iter());
                izip!(a.step_by(2), b.tuples(), c.tuples())
                    .take(chunk_size)
                    .for_each(|(a_lo, (b_lo, b_hi), (c_lo, c_hi))| {
                        partial[0] += *a_lo + *b_lo * c_lo;
                        partial[2] += (*b_hi - b_lo) * (*c_hi - c_lo);
                    });
            },
        );

        let mut sum = [F::ZERO; 3];
        partials.iter().for_each(|partial| {
            sum[0] += partial[0];
            sum[2] += partial[2];
        });
        sum[1] = claim - sum[0].double() - sum[2];
        sum.to_vec()
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
