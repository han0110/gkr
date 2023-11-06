use crate::{
    circuit::Circuit,
    poly::{evaluate, MultilinearPoly},
    sum_check::{err_unmatched_evaluation, prove_sum_check, verify_sum_check, Quadratic},
    transcript::{TranscriptRead, TranscriptWrite},
    util::{inner_product, izip, Field, Itertools},
};
use std::{array, borrow::Cow, io};

pub mod circuit;
pub mod poly;
pub mod sum_check;
pub mod transcript;
pub mod util;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    InvalidSumCheck(String),
    Transcript(io::ErrorKind, String),
}

pub fn prove_gkr<F: Field>(
    circuit: &Circuit<F>,
    values: &[Vec<F>],
    r_g: &[F],
    output_r_g: F,
    transcript: &mut impl TranscriptWrite<F>,
) -> Result<[(Vec<F>, F); 2], Error> {
    assert!(!circuit.layers().is_empty());
    assert_eq!(values.len(), circuit.layers().len() + 1);

    let (output, inputs) = values.split_last().unwrap();
    let inputs = inputs.iter().map(MultilinearPoly::new).collect_vec();

    if cfg!(feature = "sanity-check") {
        assert_eq!(evaluate(output, r_g), output_r_g);
    }

    let mut r_gs = [r_g.to_vec(), vec![F::ZERO; r_g.len()]];
    let mut output_evals = [output_r_g, Default::default()];

    for (layer, input) in izip!(circuit.layers(), inputs).rev() {
        let alphas = if output_evals[1].is_zero_vartime() {
            [F::ONE, F::ZERO]
        } else {
            array::from_fn(|_| transcript.squeeze_challenge())
        };
        let eq_r_g_prime = layer.eq_r_g_prime(&r_gs, &alphas);
        let (claim, r_x_0, input_r_x_0) = {
            let claim = inner_product(&alphas, &output_evals);
            let [h_0, h_1] = layer.phase_1_polys(&input, &eq_r_g_prime);
            let polys = [Cow::Owned(h_0), Cow::Owned(h_1), Cow::Borrowed(&input)];
            let (claim, r_x_0, evals) = prove_sum_check::<_, Quadratic>(claim, polys, transcript)?;
            transcript.write_felt(&evals[2])?;
            (claim, r_x_0, evals[2])
        };
        let (r_x_1, input_r_x_1) = {
            let [h_0, h_1] = layer.phase_2_polys(&eq_r_g_prime, &r_x_0, &input_r_x_0);
            let polys = [h_0, h_1, input].map(Cow::Owned);
            let (_, r_x_1, evals) = prove_sum_check::<_, Quadratic>(claim, polys, transcript)?;
            transcript.write_felt(&evals[2])?;
            (r_x_1, evals[2])
        };
        r_gs = [r_x_0, r_x_1];
        output_evals = [input_r_x_0, input_r_x_1];
    }

    Ok(izip!(r_gs, output_evals).collect_vec().try_into().unwrap())
}

pub fn verify_gkr<F: Field>(
    circuit: &Circuit<F>,
    r_g: &[F],
    output_r_g: F,
    transcript: &mut impl TranscriptRead<F>,
) -> Result<[(Vec<F>, F); 2], Error> {
    assert!(!circuit.layers().is_empty());

    let mut r_gs = [r_g.to_vec(), vec![F::ZERO; r_g.len()]];
    let mut output_evals = [output_r_g, Default::default()];

    for layer in circuit.layers().iter().rev() {
        let alphas = if output_evals[1].is_zero_vartime() {
            [F::ONE, F::ZERO]
        } else {
            array::from_fn(|_| transcript.squeeze_challenge())
        };
        let (claim, r_x_0, input_r_x_0) = {
            let claim = inner_product(&alphas, &output_evals);
            let (claim, r_x_0) =
                verify_sum_check::<_, Quadratic>(claim, layer.log2_input_len(), transcript)?;
            (claim, r_x_0, transcript.read_felt()?)
        };
        let (claim, r_x_1, input_r_x_1) = {
            let (claim, r_x_1) =
                verify_sum_check::<_, Quadratic>(claim, layer.log2_input_len(), transcript)?;
            (claim, r_x_1, transcript.read_felt()?)
        };
        let r_xs = [r_x_0, r_x_1];
        let input_evals = [input_r_x_0, input_r_x_1];
        if claim != layer.evaluate(&r_gs, &alphas, &r_xs, &input_evals) {
            return Err(err_unmatched_evaluation());
        }
        r_gs = r_xs;
        output_evals = input_evals;
    }

    Ok(izip!(r_gs, output_evals).collect_vec().try_into().unwrap())
}

#[cfg(test)]
mod test {
    use crate::{
        circuit::{
            test::{grand_product_circuit, grand_sum_circuit, rand_circuit},
            Circuit,
        },
        poly::evaluate,
        prove_gkr,
        transcript::Keccak256Transcript,
        util::{
            chain,
            test::{rand_vec, seeded_std_rng},
            Itertools, PrimeField, RngCore,
        },
        verify_gkr,
    };
    use halo2_curves::bn256::Fr;

    #[test]
    fn grand_product() {
        let mut rng = seeded_std_rng();
        for log2_input_len in 1..16 {
            let (circuit, input, _) = grand_product_circuit(log2_input_len, &mut rng);
            run_gkr::<Fr>(&circuit, &input, &mut rng);
        }
    }

    #[test]
    fn grand_sum() {
        let mut rng = seeded_std_rng();
        for log2_input_len in 1..16 {
            let (circuit, input, _) = grand_sum_circuit(log2_input_len, &mut rng);
            run_gkr::<Fr>(&circuit, &input, &mut rng);
        }
    }

    #[test]
    fn rand() {
        let mut rng = seeded_std_rng();
        for _ in 1..16 {
            let (circuit, input, _) = rand_circuit(&mut rng);
            run_gkr::<Fr>(&circuit, &input, &mut rng);
        }
    }

    fn run_gkr<F: PrimeField>(circuit: &Circuit<F>, input: &[F], mut rng: impl RngCore) {
        let (values, r_g, output_r_g) = {
            let outputs = circuit.outputs(input);
            let values = chain![[input.to_vec()], outputs].collect_vec();
            let r_g = rand_vec(circuit.log2_output_len(), &mut rng);
            let output_r_g = evaluate(values.last().unwrap(), &r_g);
            (values, r_g, output_r_g)
        };

        let proof = {
            let mut transcript = Keccak256Transcript::default();
            prove_gkr(circuit, &values, &r_g, output_r_g, &mut transcript).unwrap();
            transcript.into_proof()
        };

        let queries = {
            let mut transcript = Keccak256Transcript::from_proof(&proof);
            verify_gkr(circuit, &r_g, output_r_g, &mut transcript).unwrap()
        };

        chain![queries].for_each(|(r, eval)| assert_eq!(evaluate(input, &r), eval));
    }
}
