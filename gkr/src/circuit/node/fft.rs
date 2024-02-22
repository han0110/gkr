//! Implementation of section 3 of [LXZ21] (aka zkCNN).
//!
//! [LXZ21]: https://eprint.iacr.org/2021/673

use crate::{
    circuit::node::{CombinedEvalClaim, EvalClaim, Node},
    poly::{box_dense_poly, eq_eval, eq_poly, evaluate, repeated_dense_poly, BoxMultilinearPoly},
    sum_check::{
        err_unmatched_evaluation, generic::Generic, prove_sum_check, quadratic::Quadratic,
        verify_sum_check, SumCheckFunction, SumCheckPoly,
    },
    transcript::{Transcript, TranscriptRead, TranscriptWrite},
    util::{
        arithmetic::{
            inner_product, powers, radix2_fft, squares, ExtensionField, Field, PrimeField,
        },
        chain, chain_par,
        collection::Hadamard,
        expression::Expression,
        izip, izip_par, Itertools,
    },
    Error,
};
use rayon::prelude::*;
use std::marker::PhantomData;

#[derive(Clone, Debug)]
pub struct FftNode<F, E> {
    log2_size: usize,
    omega: F,
    omegas: Vec<F>,
    n_inv: Option<F>,
    _marker: PhantomData<E>,
}

impl<F: PrimeField, E: ExtensionField<F>> Node<F, E> for FftNode<F, E> {
    fn is_input(&self) -> bool {
        false
    }

    fn log2_input_size(&self) -> usize {
        self.log2_size
    }

    fn log2_output_size(&self) -> usize {
        self.log2_size
    }

    fn evaluate(
        &self,
        inputs: Vec<&BoxMultilinearPoly<F, E>>,
    ) -> BoxMultilinearPoly<'static, F, E> {
        assert_eq!(inputs.len(), 1, "Unimplemented");

        let input = inputs[0];
        assert_eq!(input.len(), self.input_size());

        let mut value = input
            .as_dense()
            .map(|input| input.to_vec())
            .unwrap_or_else(|| (0..input.len()).into_par_iter().map(|b| input[b]).collect());
        radix2_fft(&mut value, self.omega);
        if let Some(n_inv) = self.n_inv {
            value.par_iter_mut().for_each(|value| *value *= n_inv);
        }
        box_dense_poly(value)
    }

    fn prove_claim_reduction<'a>(
        &self,
        claim: CombinedEvalClaim<E>,
        inputs: Vec<&BoxMultilinearPoly<F, E>>,
        transcript: &mut dyn TranscriptWrite<F, E>,
    ) -> Result<Vec<Vec<EvalClaim<E>>>, Error> {
        assert_eq!(inputs.len(), 1, "Unimplemented");

        let (w_interms, ws) = izip!(&claim.points, &claim.alphas)
            .map(|(point, alpha)| {
                let mut ws = self.wiring(point, *alpha);
                let w = ws.pop().unwrap();
                (ws, w)
            })
            .unzip::<_, _, Vec<_>, Vec<_>>();

        let (r_x, input_r_x, w_r_xs) = {
            let g = Quadratic::new(self.log2_size, vec![(None, 0, 1)]);
            let w = box_dense_poly(ws.par_iter().cloned().hada_sum());
            let polys = [SumCheckPoly::Base(inputs[0]), SumCheckPoly::Extension(&w)];
            let (_, r_x, evals) = prove_sum_check(&g, claim.value, polys, transcript)?;
            let w_r_xs = ws.iter().map(|w| evaluate(w, &r_x)).collect_vec();
            transcript.write_felt_ext(&evals[0])?;
            transcript.write_felt_exts(&w_r_xs)?;
            (r_x, evals[0], w_r_xs)
        };

        self.prove_wiring_eval(&claim, &w_interms, &r_x, &w_r_xs, transcript)?;

        Ok(vec![vec![EvalClaim::new(r_x, input_r_x)]])
    }

    fn verify_claim_reduction(
        &self,
        claim: CombinedEvalClaim<E>,
        transcript: &mut dyn TranscriptRead<F, E>,
    ) -> Result<Vec<Vec<EvalClaim<E>>>, Error> {
        let (r_x, input_r_x, w_r_xs) = {
            let g = Quadratic::new(self.log2_size, vec![(None, 0, 1)]);
            let (sub_claim, r_x) = verify_sum_check(&g, claim.value, transcript)?;
            let input_r_x = transcript.read_felt_ext()?;
            let w_r_xs = transcript.read_felt_exts(claim.points.len())?;
            if sub_claim != self.final_eval(&claim, input_r_x, &w_r_xs) {
                return Err(err_unmatched_evaluation());
            }

            (r_x, input_r_x, w_r_xs)
        };

        self.verify_wiring_eval(&claim, &r_x, &w_r_xs, transcript)?;

        Ok(vec![vec![EvalClaim::new(r_x, input_r_x)]])
    }
}

impl<F: PrimeField, E: ExtensionField<F>> FftNode<F, E> {
    pub fn forward(log2_size: usize) -> Self {
        let omega = root_of_unity(log2_size);
        let omegas = Vec::from_iter(powers(omega).take(1 << log2_size));
        Self {
            log2_size,
            omega,
            omegas,
            n_inv: None,
            _marker: PhantomData,
        }
    }

    pub fn inverse(log2_size: usize) -> Self {
        let omega = root_of_unity_inv(log2_size);
        let omegas = Vec::from_iter(powers(omega).take(1 << log2_size));
        Self {
            log2_size,
            omega,
            omegas,
            n_inv: Some(powers(F::TWO_INV).nth(log2_size).unwrap()),
            _marker: PhantomData,
        }
    }

    fn wiring(&self, r_g: &[E], alpha: E) -> Vec<Vec<E>> {
        let init = self.n_inv.map(|n_inv| alpha * n_inv).unwrap_or(alpha);
        let mut bufs = Vec::with_capacity(r_g.len());
        bufs.push(vec![init]);
        for (idx, r_g_i) in izip!(0..r_g.len(), r_g.iter()).rev() {
            let one_minus_r_g_i = E::ONE - r_g_i;
            let last = bufs.last().unwrap();
            let buf = izip_par!(chain_par![last, last], self.omegas(idx))
                .map(|(last, omega)| (one_minus_r_g_i + *r_g_i * omega) * last)
                .collect();
            bufs.push(buf);
        }
        bufs
    }

    fn final_eval(&self, claim: &CombinedEvalClaim<E>, input_r_x: E, w_r_xs: &[E]) -> E {
        input_r_x * inner_product::<E, E>(&claim.alphas, w_r_xs)
    }

    fn omegas(&self, log2_step: usize) -> impl IndexedParallelIterator<Item = &F> {
        self.omegas.par_iter().step_by(1 << log2_step)
    }

    fn prove_wiring_eval(
        &self,
        claim: &CombinedEvalClaim<E>,
        w_interms: &[Vec<Vec<E>>],
        r_x: &[E],
        w_r_xs: &[E],
        transcript: &mut dyn TranscriptWrite<F, E>,
    ) -> Result<(), Error> {
        let r_gs = &claim.points;

        let mut claims = w_r_xs.to_vec();
        let mut r_x = r_x.to_vec();
        for layer in 0..self.log2_size {
            let (claim, g) = self.wiring_sum_check_predicate(r_gs, &claims, layer, transcript);
            let polys = {
                let eq_r_x = box_dense_poly(eq_poly(&r_x, E::ONE));
                let omegas = box_dense_poly(self.omegas(layer).copied().collect::<Vec<_>>());
                let w_interms = w_interms
                    .iter()
                    .map(|w_interms| w_interms.iter().nth_back(layer).unwrap())
                    .map(|w| repeated_dense_poly(w, 1));
                chain![
                    [SumCheckPoly::Extension(eq_r_x), SumCheckPoly::Base(omegas)],
                    w_interms.map(SumCheckPoly::Extension)
                ]
                .collect_vec()
            };
            let (_, r_x_prime, evals) = prove_sum_check(&g, claim, polys, transcript)?;
            let w_interm_r_x_primes = evals[2..].to_vec();
            if layer == self.log2_size - 1 {
                break;
            }
            transcript.write_felt_exts(&w_interm_r_x_primes)?;
            claims = w_interm_r_x_primes;
            r_x = r_x_prime[..r_x_prime.len() - 1].to_vec();
        }

        Ok(())
    }

    fn verify_wiring_eval(
        &self,
        claim: &CombinedEvalClaim<E>,
        r_x: &[E],
        w_r_xs: &[E],
        transcript: &mut dyn TranscriptRead<F, E>,
    ) -> Result<(), Error> {
        let r_gs = &claim.points;
        let alphas = &claim.alphas;

        let mut claims = w_r_xs.to_vec();
        let mut r_x = r_x.to_vec();
        for layer in 0..self.log2_size {
            let (claim, g) = self.wiring_sum_check_predicate(r_gs, &claims, layer, transcript);
            let (sub_claim, r_x_prime) = verify_sum_check(&g, claim, transcript)?;
            let w_interm_r_x_primes = if layer == self.log2_size - 1 {
                self.wiring_gkr_initial_evals(alphas)
            } else {
                transcript.read_felt_exts(r_gs.len())?
            };
            if sub_claim
                != self.wiring_sum_check_final_eval(&g, &r_x, &r_x_prime, &w_interm_r_x_primes)
            {
                return Err(err_unmatched_evaluation());
            }
            claims = w_interm_r_x_primes;
            r_x = r_x_prime[..r_x_prime.len() - 1].to_vec();
        }

        Ok(())
    }

    fn wiring_sum_check_predicate(
        &self,
        r_gs: &[Vec<E>],
        claims: &[E],
        layer: usize,
        transcript: &mut (impl Transcript<F, E> + ?Sized),
    ) -> (E, Generic<F, E>) {
        let beta = if claims.len() == 1 {
            E::ONE
        } else {
            transcript.squeeze_challenge()
        };
        let claim = inner_product::<E, E>(powers(beta).take(claims.len()), claims);

        let expression = {
            let one = &Expression::constant(E::ONE);
            let r_g_is = &r_gs
                .iter()
                .map(|r_g| r_g[layer])
                .map(Expression::constant)
                .collect_vec();
            let eq_r_x = &Expression::poly(0);
            let omegas = &Expression::poly(1);
            let w_interms = &(2..).take(r_g_is.len()).map(Expression::poly).collect_vec();
            let expands = izip!(r_g_is, w_interms)
                .map(|(r_g_i, w_interm)| w_interm * (one - r_g_i + r_g_i * omegas));
            eq_r_x * Expression::distribute_powers(expands, beta)
        };

        (claim, Generic::new(r_gs[0].len() - layer, &expression))
    }

    fn wiring_sum_check_final_eval(
        &self,
        g: &Generic<F, E>,
        r_x: &[E],
        r_x_prime: &[E],
        w_interm_r_x_primes: &[E],
    ) -> E {
        let eq_eval = eq_eval([r_x, r_x_prime]);
        let omega = squares(self.omega).nth(self.log2_size - r_x.len()).unwrap();
        let omega_eval = omega_eval(omega, r_x_prime);
        let evals =
            chain![[eq_eval, omega_eval], w_interm_r_x_primes.iter().copied()].collect_vec();
        g.evaluate(&evals)
    }

    fn wiring_gkr_initial_evals(&self, alphas: &[E]) -> Vec<E> {
        self.n_inv
            .map(|n_inv| alphas.iter().map(|alpha| *alpha * n_inv).collect_vec())
            .unwrap_or_else(|| alphas.to_vec())
    }
}

fn root_of_unity<F: PrimeField>(k: usize) -> F {
    assert!(k <= F::S as usize);
    squares(F::ROOT_OF_UNITY).nth(F::S as usize - k).unwrap()
}

fn root_of_unity_inv<F: PrimeField>(k: usize) -> F {
    assert!(k <= F::S as usize);
    squares(F::ROOT_OF_UNITY_INV)
        .nth(F::S as usize - k)
        .unwrap()
}

fn omega_eval<F: Field, E: ExtensionField<F>>(omega: F, r_x_prime: &[E]) -> E {
    izip!(r_x_prime, squares(omega))
        .map(|(r_x_prime_i, omega)| E::ONE - r_x_prime_i + *r_x_prime_i * omega)
        .product()
}

#[cfg(test)]
mod test {
    use crate::{
        circuit::{
            node::{
                fft::{root_of_unity, FftNode},
                input::InputNode,
                NodeExt,
            },
            test::{run_circuit, TestData},
            Circuit,
        },
        poly::box_dense_poly,
        util::{
            arithmetic::{radix2_fft, ExtensionField, PrimeField},
            dev::rand_vec,
            RngCore,
        },
    };
    use goldilocks::{Goldilocks, GoldilocksExt2};

    #[test]
    fn fft_and_then_ifft() {
        run_circuit::<Goldilocks, GoldilocksExt2>(fft_and_then_ifft_circuit);
    }

    pub fn fft_and_then_ifft_circuit<F: PrimeField, E: ExtensionField<F>>(
        log2_input_size: usize,
        rng: &mut impl RngCore,
    ) -> TestData<F, E> {
        let nodes = vec![
            InputNode::new(log2_input_size, 1).boxed(),
            FftNode::forward(log2_input_size).boxed(),
            FftNode::inverse(log2_input_size).boxed(),
        ];
        let circuit = Circuit::linear(nodes);

        let input = rand_vec(1 << log2_input_size, rng);
        let input_prime = {
            let omega = root_of_unity(log2_input_size);
            let mut buf = input.clone();
            radix2_fft(&mut buf, omega);
            buf
        };
        let values = [input.clone(), input_prime, input.clone()]
            .into_iter()
            .map(box_dense_poly)
            .collect();

        (circuit, vec![box_dense_poly(input)], Some(values))
    }
}
