use crate::{
    circuit::{node::EvalClaim, Circuit},
    poly::evaluate,
    prove_gkr,
    transcript::StdRngTranscript,
    util::{arithmetic::PrimeField, izip_eq, test::rand_vec, Itertools, RngCore},
    verify_gkr,
};

pub fn run_gkr<F: PrimeField>(circuit: &Circuit<F>, inputs: Vec<Vec<F>>, mut rng: impl RngCore) {
    let (values, output_claims) = {
        let values = circuit.evaluate(inputs.clone());
        let output_claims = circuit
            .outputs()
            .map(|idx| {
                let point = rand_vec(circuit.nodes()[idx].log2_output_size(), &mut rng);
                let value = evaluate(&values[idx], &point);
                EvalClaim::new(point, value)
            })
            .collect_vec();
        (values, output_claims)
    };

    let proof = {
        let mut transcript = StdRngTranscript::default();
        prove_gkr(circuit, values, output_claims.clone(), &mut transcript).unwrap();
        transcript.into_proof()
    };

    let input_claims = {
        let mut transcript = StdRngTranscript::from_proof(&proof);
        verify_gkr(circuit, output_claims, &mut transcript).unwrap()
    };

    izip_eq!(&inputs, input_claims).for_each(|(input, claims)| {
        claims
            .iter()
            .for_each(|claim| assert_eq!(evaluate(input, claim.point()), claim.value()))
    });
}
