use crate::{
    circuit::{
        node::{CombinedEvalClaim, EvalClaim},
        Circuit,
    },
    poly::evaluate,
    transcript::{Transcript, TranscriptRead, TranscriptWrite},
    util::{arithmetic::Field, Itertools},
};
use std::{io, mem::take};

pub mod circuit;
pub mod poly;
pub mod sum_check;
pub mod transcript;
pub mod util;

#[cfg(any(test, feature = "dev"))]
pub mod dev;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    InvalidSumCheck(String),
    Transcript(io::ErrorKind, String),
}

pub fn prove_gkr<F: Field>(
    circuit: &Circuit<F>,
    values: &[Vec<F>],
    output_claims: &[EvalClaim<F>],
    transcript: &mut impl TranscriptWrite<F>,
) -> Result<Vec<Vec<EvalClaim<F>>>, Error> {
    circuit
        .topo_iter()
        .for_each(|(idx, node)| assert_eq!(values[idx].len(), node.output_size()));

    if cfg!(feature = "sanity-check") {
        izip_eq!(circuit.outputs(), output_claims).for_each(|(idx, claim)| {
            assert_eq!(evaluate(&values[idx], claim.point()), claim.value())
        });
    }

    let mut claims = vec![Vec::new(); circuit.nodes().len()];
    izip_eq!(circuit.outputs(), output_claims)
        .for_each(|(idx, claim)| claims[idx] = vec![claim.clone()]);

    for (idx, node) in circuit.topo_iter().rev() {
        if node.is_input() {
            continue;
        }

        let claim = combined_claim(take(&mut claims[idx]), transcript);
        let inputs = Vec::from_iter(circuit.predec(idx).map(|idx| &values[idx]));
        let sub_claims = node.prove_claim_reduction(claim, inputs, transcript)?;

        izip_eq!(circuit.predec(idx), sub_claims)
            .for_each(|(idx, sub_claims)| claims[idx].extend(sub_claims));
    }

    let input_claims = Vec::from_iter(circuit.inputs().map(|idx| take(&mut claims[idx])));

    assert!(!claims.iter().any(|claims| !claims.is_empty()));

    Ok(input_claims)
}

pub fn verify_gkr<F: Field>(
    circuit: &Circuit<F>,
    output_claims: &[EvalClaim<F>],
    transcript: &mut impl TranscriptRead<F>,
) -> Result<Vec<Vec<EvalClaim<F>>>, Error> {
    let mut claims = vec![Vec::new(); circuit.nodes().len()];
    izip_eq!(circuit.outputs(), output_claims)
        .for_each(|(idx, claim)| claims[idx] = vec![claim.clone()]);

    for (idx, node) in circuit.topo_iter().rev() {
        if node.is_input() {
            continue;
        }

        let claim = combined_claim(take(&mut claims[idx]), transcript);
        let sub_claims = node.verify_claim_reduction(claim, transcript)?;

        izip_eq!(circuit.predec(idx), sub_claims)
            .for_each(|(idx, sub_claims)| claims[idx].extend(sub_claims));
    }

    let input_claims = Vec::from_iter(circuit.inputs().map(|idx| take(&mut claims[idx])));

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
