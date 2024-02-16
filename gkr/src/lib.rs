use crate::{
    circuit::{
        node::{CombinedEvalClaim, EvalClaim},
        Circuit,
    },
    poly::{BoxMultilinearPoly, MultilinearPoly},
    transcript::{Transcript, TranscriptRead, TranscriptWrite},
    util::{
        arithmetic::{ExtensionField, Field},
        Itertools,
    },
};
use std::{io, mem::take};

pub mod circuit;
pub mod poly;
pub mod sum_check;
pub mod transcript;
pub mod util;

pub use ff_ext;

#[cfg(any(test, feature = "dev"))]
pub mod dev;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    InvalidSumCheck(String),
    Transcript(io::ErrorKind, String),
}

pub fn prove_gkr<F: Field, E: ExtensionField<F>>(
    circuit: &Circuit<F, E>,
    values: &[BoxMultilinearPoly<F, E>],
    output_claims: &[EvalClaim<E>],
    transcript: &mut impl TranscriptWrite<F, E>,
) -> Result<Vec<Vec<EvalClaim<E>>>, Error> {
    circuit
        .topo_iter()
        .for_each(|(idx, node)| assert_eq!(values[idx].len(), node.output_size()));

    if cfg!(feature = "sanity-check") {
        izip_eq!(circuit.outputs(), output_claims).for_each(|(idx, claim)| {
            assert_eq!(values[idx].evaluate(claim.point()), claim.value())
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
        let inputs = circuit.predec(idx).map(|idx| &values[idx]).collect();
        let sub_claims = node.prove_claim_reduction(claim, inputs, transcript)?;

        izip_eq!(circuit.predec(idx), sub_claims)
            .for_each(|(idx, sub_claims)| claims[idx].extend(sub_claims));
    }

    let input_claims = Vec::from_iter(circuit.inputs().map(|idx| take(&mut claims[idx])));

    assert!(!claims.iter().any(|claims| !claims.is_empty()));

    Ok(input_claims)
}

pub fn verify_gkr<F: Field, E: ExtensionField<F>>(
    circuit: &Circuit<F, E>,
    output_claims: &[EvalClaim<E>],
    transcript: &mut impl TranscriptRead<F, E>,
) -> Result<Vec<Vec<EvalClaim<E>>>, Error> {
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

fn combined_claim<F: Field, E: ExtensionField<F>>(
    claims: Vec<EvalClaim<E>>,
    transcript: &mut impl Transcript<F, E>,
) -> CombinedEvalClaim<E> {
    let alphas = if claims.len() == 1 {
        vec![E::ONE]
    } else {
        transcript.squeeze_challenges(claims.len())
    };
    CombinedEvalClaim::new(claims, alphas)
}
