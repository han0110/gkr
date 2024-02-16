use crate::{
    circuit::{node::EvalClaim, Circuit},
    poly::{BoxMultilinearPoly, MultilinearPoly},
    prove_gkr,
    transcript::StdRngTranscript,
    util::{
        arithmetic::{ExtensionField, PrimeField},
        dev::rand_vec,
        izip_eq, Itertools, RngCore,
    },
    verify_gkr,
};

pub fn run_gkr<F: PrimeField, E: ExtensionField<F>>(
    circuit: &Circuit<F, E>,
    inputs: &[BoxMultilinearPoly<F, E>],
    rng: impl RngCore,
) {
    let values = circuit.evaluate(inputs.iter().map(MultilinearPoly::clone_box).collect());
    run_gkr_with_values(circuit, &values, rng);
}

pub fn run_gkr_with_values<F: PrimeField, E: ExtensionField<F>>(
    circuit: &Circuit<F, E>,
    values: &[BoxMultilinearPoly<F, E>],
    mut rng: impl RngCore,
) {
    let output_claims = circuit
        .outputs()
        .map(|idx| {
            let point = rand_vec(circuit.nodes()[idx].log2_output_size(), &mut rng);
            let value = values[idx].evaluate(&point);
            EvalClaim::new(point, value)
        })
        .collect_vec();

    let proof = {
        let mut transcript = StdRngTranscript::default();
        prove_gkr(circuit, values, &output_claims, &mut transcript).unwrap();
        transcript.into_proof()
    };

    let input_claims = {
        let mut transcript = StdRngTranscript::from_proof(&proof);
        verify_gkr(circuit, &output_claims, &mut transcript).unwrap()
    };

    izip_eq!(circuit.inputs(), input_claims).for_each(|(input, claims)| {
        claims
            .iter()
            .for_each(|claim| assert_eq!(values[input].evaluate(claim.point()), claim.value()))
    });
}
