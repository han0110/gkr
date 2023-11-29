use crate::{
    circuit::{
        node::{CombinedEvalClaim, EvalClaim},
        Circuit,
    },
    poly::evaluate,
    transcript::{Transcript, TranscriptRead, TranscriptWrite},
    util::{izip_eq, Field, Itertools},
};
use std::{io, mem::take};

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
    values: Vec<Vec<F>>,
    output_claims: Vec<EvalClaim<F>>,
    transcript: &mut impl TranscriptWrite<F>,
) -> Result<Vec<Vec<EvalClaim<F>>>, Error> {
    circuit
        .topo_iter()
        .for_each(|(idx, node)| assert_eq!(values[idx].len(), node.output_size()));

    if cfg!(feature = "sanity-check") {
        izip_eq!(circuit.adj_mat().outputs(), &output_claims).for_each(|(idx, claim)| {
            assert_eq!(evaluate(&values[idx], claim.point()), claim.value())
        });
    }

    let mut claims = vec![Vec::new(); circuit.nodes().len()];
    izip_eq!(circuit.adj_mat().outputs(), output_claims)
        .for_each(|(idx, claim)| claims[idx] = vec![claim]);

    for (idx, node) in circuit.topo_iter().rev() {
        if node.is_input() {
            continue;
        }

        let claim = combined_claim(take(&mut claims[idx]), transcript);
        let inputs = Vec::from_iter(circuit.adj_mat().predec(idx).map(|idx| &values[idx]));
        let sub_claims = node.prove_claim_reduction(claim, inputs, transcript)?;

        izip_eq!(circuit.adj_mat().predec(idx), sub_claims)
            .for_each(|(idx, sub_claims)| claims[idx].extend(sub_claims));
    }

    let input_claims = Vec::from_iter(circuit.adj_mat().inputs().map(|idx| take(&mut claims[idx])));

    assert!(!claims.iter().any(|claims| !claims.is_empty()));

    Ok(input_claims)
}

pub fn verify_gkr<F: Field>(
    circuit: &Circuit<F>,
    output_claims: Vec<EvalClaim<F>>,
    transcript: &mut impl TranscriptRead<F>,
) -> Result<Vec<Vec<EvalClaim<F>>>, Error> {
    let mut claims = vec![Vec::new(); circuit.nodes().len()];
    izip_eq!(circuit.adj_mat().outputs(), output_claims)
        .for_each(|(idx, output_claims)| claims[idx] = vec![output_claims]);

    for (idx, node) in circuit.topo_iter().rev() {
        if node.is_input() {
            continue;
        }

        let claim = combined_claim(take(&mut claims[idx]), transcript);
        let sub_claims = node.verify_claim_reduction(claim, transcript)?;

        izip_eq!(circuit.adj_mat().predec(idx), sub_claims)
            .for_each(|(idx, sub_claims)| claims[idx].extend(sub_claims));
    }

    let input_claims = Vec::from_iter(circuit.adj_mat().inputs().map(|idx| take(&mut claims[idx])));

    assert!(!claims.iter().any(|claims| !claims.is_empty()));

    Ok(input_claims)
}

fn combined_claim<F: Field>(
    claims: Vec<EvalClaim<F>>,
    transcript: &mut impl Transcript<F>,
) -> CombinedEvalClaim<F> {
    let alphas = if claims.len() == 1 {
        vec![F::ONE]
    } else {
        transcript.squeeze_challenges(claims.len())
    };
    CombinedEvalClaim::new(claims, alphas)
}

#[cfg(test)]
mod test {
    use crate::{
        circuit::{node::EvalClaim, Circuit},
        poly::evaluate,
        prove_gkr,
        transcript::Keccak256Transcript,
        util::{izip_eq, test::rand_vec, Itertools, PrimeField, RngCore},
        verify_gkr,
    };

    pub fn run_gkr<F: PrimeField>(
        circuit: &Circuit<F>,
        inputs: Vec<Vec<F>>,
        mut rng: impl RngCore,
    ) {
        let (values, output_claims) = {
            let values = circuit.evaluate(inputs.clone());
            let output_claims = circuit
                .adj_mat()
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
            let mut transcript = Keccak256Transcript::default();
            prove_gkr(circuit, values, output_claims.clone(), &mut transcript).unwrap();
            transcript.into_proof()
        };

        let input_claims = {
            let mut transcript = Keccak256Transcript::from_proof(&proof);
            verify_gkr(circuit, output_claims, &mut transcript).unwrap()
        };

        izip_eq!(&inputs, input_claims).for_each(|(input, claims)| {
            claims
                .iter()
                .for_each(|claim| assert_eq!(evaluate(input, claim.point()), claim.value()))
        });
    }
}
