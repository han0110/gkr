//! Implementation of section 3 of [LXZ21] (aka zkCNN).
//!
//! [LXZ21]: https://eprint.iacr.org/2021/673

use crate::{
    poly::{eq_eval, eq_poly, evaluate, MultilinearPoly},
    sum_check::{err_unmatched_evaluation, prove_sum_check, verify_sum_check, SumCheckFunction},
    transcript::{TranscriptRead, TranscriptWrite},
    util::{
        chain, chain_par, izip, izip_eq, izip_par, AdditiveArray, Field, Itertools, PrimeField,
    },
    Error,
};
use rayon::prelude::*;
use std::{borrow::Cow, iter};

pub fn prove_fft_layer_wiring<F: PrimeField>(
    r_g: &[F],
    values: &[Vec<F>],
    r_x: &[F],
    output_r_x: F,
    transcript: &mut impl TranscriptWrite<F>,
) -> Result<(), Error> {
    assert!(!r_g.is_empty());
    assert_eq!(values.len(), r_g.len() + 1);
    assert_eq!(r_g.len(), r_x.len());

    let (output, inputs) = values.split_last().unwrap();

    if cfg!(feature = "sanity-check") {
        assert_eq!(evaluate(output, r_x), output_r_x);
    }

    let omegas = omegas(r_g.len());

    let mut claim = output_r_x;
    let mut r_x = r_x.to_vec();
    for (i, r_g_i, input) in izip!(0..r_g.len(), r_g, inputs.iter().rev()) {
        let layer = Layer::new(*r_g_i);
        let polys = {
            let eq_r_x = eq_poly(&r_x, F::ONE);
            let input = chain![input, input].copied().collect_vec();
            let omegas = omegas.iter().copied().step_by(1 << i).collect_vec();
            [eq_r_x, input, omegas].map(|f| Cow::Owned(MultilinearPoly::new(f.into())))
        };
        let (_, r_x_prime, evals) = prove_sum_check(&layer, claim, polys, transcript)?;
        if i == r_g.len() - 1 {
            assert_eq!(evals[1], F::ONE);
            break;
        }
        transcript.write_felt(&evals[1])?;
        claim = evals[1];
        r_x = r_x_prime.split_last().unwrap().1.to_vec();
    }

    Ok(())
}

pub fn verify_fft_layer_wiring<F: PrimeField>(
    r_g: &[F],
    r_x: &[F],
    output_r_x: F,
    transcript: &mut impl TranscriptRead<F>,
) -> Result<(), Error> {
    assert!(!r_g.is_empty());
    assert_eq!(r_g.len(), r_x.len());

    let mut claim = output_r_x;
    let mut r_x = r_x.to_vec();
    for (i, r_g_i) in izip!(0..r_g.len(), r_g) {
        let layer = Layer::new(*r_g_i);
        let (subclaim, r_x_prime) = verify_sum_check(&layer, claim, r_g.len() - i, transcript)?;
        let eval = if i == r_g.len() - 1 {
            F::ONE
        } else {
            transcript.read_felt()?
        };
        if subclaim != layer.evaluate(r_g.len() - i, &r_x, &r_x_prime, eval) {
            return Err(err_unmatched_evaluation());
        }
        claim = eval;
        r_x = r_x_prime.split_last().unwrap().1.to_vec();
    }

    Ok(())
}

pub fn fft_layer_init<F: PrimeField>(r_g: &[F]) -> Vec<Vec<F>> {
    let omegas = omegas(r_g.len());

    let mut bufs = Vec::with_capacity(r_g.len());
    bufs.push(vec![F::ONE]);
    for (i, r_g_i) in izip!(0..r_g.len(), r_g.iter()).rev() {
        let one_minus_r_g_i = F::ONE - r_g_i;
        let last = bufs.last().unwrap();
        let buf = izip_par!(chain_par![last, last], omegas.par_iter().step_by(1 << i))
            .map(|(last, omega)| (one_minus_r_g_i + *r_g_i * omega) * last)
            .collect();
        bufs.push(buf);
    }
    bufs
}

fn root_of_unity<F: PrimeField>(log2_m: usize) -> F {
    assert!(log2_m <= F::S as usize);
    let mut omega = F::ROOT_OF_UNITY;
    for _ in log2_m..F::S as usize {
        omega = omega.square();
    }
    omega
}

fn omegas<F: PrimeField>(log2_m: usize) -> Vec<F> {
    let root_of_unity = root_of_unity::<F>(log2_m);
    iter::successors(Some(F::ONE), |omega| Some(root_of_unity * omega))
        .take(1 << log2_m)
        .collect_vec()
}

fn omega_eval<F: PrimeField>(log2_m: usize, r_x_prime: &[F]) -> F {
    let root_of_unity = root_of_unity::<F>(log2_m);
    let omegas = iter::successors(Some(root_of_unity), move |omega| Some(omega.square()));
    izip!(r_x_prime, omegas)
        .map(|(r_x_prime_i, omega)| F::ONE - r_x_prime_i + *r_x_prime_i * omega)
        .product()
}

#[derive(Clone, Debug)]
struct Layer<F> {
    r_g_i: F,
    one_minus_r_g_i: F,
    interpolation_weights: [[F; 4]; 4],
}

impl<F: PrimeField> Layer<F> {
    fn new(r_g_i: F) -> Self {
        Self {
            r_g_i,
            one_minus_r_g_i: F::ONE - r_g_i,
            interpolation_weights: [
                [pf!(1), pf!(0), pf!(0), pf!(0)],
                [pf!(-11 / 6), pf!(3), pf!(-3 / 2), pf!(1 / 3)],
                [pf!(1), pf!(-5 / 2), pf!(2), pf!(-1 / 2)],
                [pf!(-1 / 6), pf!(1 / 2), pf!(-1 / 2), pf!(1 / 6)],
            ],
        }
    }

    fn evaluate(&self, i: usize, r_x: &[F], r_x_prime: &[F], eval: F) -> F {
        eq_eval([r_x, r_x_prime])
            * eval
            * (self.one_minus_r_g_i + self.r_g_i * omega_eval(i, r_x_prime))
    }
}

impl<F: Field> SumCheckFunction<F> for Layer<F> {
    fn degree(&self) -> usize {
        3
    }

    fn compute_sum(&self, claim: F, polys: &[MultilinearPoly<F>]) -> Vec<F> {
        assert_eq!(polys.len(), 3);

        if cfg!(feature = "sanity-check") {
            let polys = polys.iter().map(|poly| poly.as_slice()).collect_vec();
            assert_eq!(
                claim,
                izip_eq!(polys[0], polys[1], polys[2])
                    .map(|(a, b, c)| *a * b * (self.one_minus_r_g_i + self.r_g_i * c))
                    .sum()
            )
        }

        let AdditiveArray([eval_1, eval_2, eval_3]) = izip_par!(
            &polys[0][0..],
            &polys[0][1..],
            &polys[1][0..],
            &polys[1][1..],
            &polys[2][0..],
            &polys[2][1..],
        )
        .step_by(2)
        .fold_with(AdditiveArray::default(), |mut evals, values| {
            let (a_lo, a_hi, b_lo, b_hi, c_lo, c_hi) = values;
            let mut a = *a_hi;
            let mut b = *b_hi;
            let mut c = *c_hi;
            let a_diff = a - a_lo;
            let b_diff = b - b_lo;
            let c_diff = c - c_lo;
            evals[0] += a * b * (self.one_minus_r_g_i + self.r_g_i * c);
            a += a_diff;
            b += b_diff;
            c += c_diff;
            evals[1] += a * b * (self.one_minus_r_g_i + self.r_g_i * c);
            a += a_diff;
            b += b_diff;
            c += c_diff;
            evals[2] += a * b * (self.one_minus_r_g_i + self.r_g_i * c);
            evals
        })
        .sum();

        let evals = [claim - eval_1, eval_1, eval_2, eval_3];
        self.interpolation_weights
            .map(|weights| F::sum(izip!(weights, &evals).map(|(weight, eval)| weight * eval)))
            .to_vec()
    }

    fn write_sum(&self, sum: &[F], transcript: &mut impl TranscriptWrite<F>) -> Result<(), Error> {
        transcript.write_felt(&sum[0])?;
        transcript.write_felt(&sum[2])?;
        transcript.write_felt(&sum[3])?;
        Ok(())
    }

    fn read_sum(&self, claim: F, transcript: &mut impl TranscriptRead<F>) -> Result<Vec<F>, Error> {
        let mut sum = [F::ZERO; 4];
        sum[0] = transcript.read_felt()?;
        sum[2] = transcript.read_felt()?;
        sum[3] = transcript.read_felt()?;
        sum[1] = claim - sum[0].double() - sum[2] - sum[3];
        Ok(sum.to_vec())
    }
}

macro_rules! pf {
    ($n:tt) => {
        F::from($n)
    };
    ($n:tt / $d:tt) => {
        F::from($n) * F::from($d).invert().unwrap()
    };
    (-$n:tt / $d:tt) => {
        -F::from($n) * F::from($d).invert().unwrap()
    };
}

use pf;

#[cfg(test)]
mod test {
    use crate::{
        etc::{
            self,
            fft::{omegas, prove_fft_layer_wiring, verify_fft_layer_wiring},
        },
        poly::{eq_poly, evaluate},
        transcript::Keccak256Transcript,
        util::{
            izip,
            test::{rand_vec, seeded_std_rng},
            Itertools, PrimeField,
        },
    };
    use halo2_curves::bn256::Fr;

    #[test]
    fn fft_layer_wiring() {
        (1..16).for_each(run_fft_layer_wiring::<Fr>)
    }

    #[test]
    fn fft_layer_init() {
        (1..10).for_each(run_fft_layer_init::<Fr>)
    }

    fn run_fft_layer_wiring<F: PrimeField>(log2_m: usize) {
        let mut rng = seeded_std_rng();
        let r_g = rand_vec(log2_m, &mut rng);
        let values = etc::fft::fft_layer_init(&r_g);
        let r_x = rand_vec(log2_m, &mut rng);
        let output_r_x = evaluate::<F>(values.last().unwrap(), &r_x);

        let proof = {
            let mut transcript = Keccak256Transcript::default();
            prove_fft_layer_wiring(&r_g, &values, &r_x, output_r_x, &mut transcript).unwrap();
            transcript.into_proof()
        };

        let result = {
            let mut transcript = Keccak256Transcript::from_proof(&proof);
            verify_fft_layer_wiring(&r_g, &r_x, output_r_x, &mut transcript)
        };
        assert_eq!(result, Ok(()));
    }

    fn run_fft_layer_init<F: PrimeField>(log2_m: usize) {
        let mut rng = seeded_std_rng();
        let r_g = rand_vec(log2_m, &mut rng);
        let expected = {
            let eq_r_g = eq_poly(&r_g, F::ONE);
            let omegas = omegas(log2_m);
            let omegas = |i| {
                let i = if i == 0 { 1 << log2_m } else { i };
                omegas.iter().cycle().step_by(i)
            };
            (0..1 << log2_m)
                .map(|i| F::sum(izip!(&eq_r_g, omegas(i)).map(|(lhs, rhs)| *lhs * rhs)))
                .collect_vec()
        };
        assert_eq!(
            expected,
            etc::fft::fft_layer_init(&r_g).into_iter().last().unwrap()
        );
    }
}
