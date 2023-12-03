use crate::{
    circuit::node::{CombinedEvalClaim, EvalClaim, Node},
    transcript::{TranscriptRead, TranscriptWrite},
    Error,
};

#[derive(Clone, Debug)]
pub struct InputNode {
    log2_size: usize,
}

impl InputNode {
    pub fn new(log2_size: usize) -> Self {
        Self { log2_size }
    }
}

impl<F> Node<F> for InputNode {
    fn is_input(&self) -> bool {
        true
    }

    fn log2_input_size(&self) -> usize {
        self.log2_size
    }

    fn log2_output_size(&self) -> usize {
        self.log2_size
    }

    fn evaluate(&self, _: Vec<&Vec<F>>) -> Vec<F> {
        unreachable!()
    }

    fn prove_claim_reduction(
        &self,
        _: CombinedEvalClaim<F>,
        _: Vec<&Vec<F>>,
        _: &mut dyn TranscriptWrite<F>,
    ) -> Result<Vec<Vec<EvalClaim<F>>>, Error> {
        unreachable!()
    }

    fn verify_claim_reduction(
        &self,
        _: CombinedEvalClaim<F>,
        _: &mut dyn TranscriptRead<F>,
    ) -> Result<Vec<Vec<EvalClaim<F>>>, Error> {
        unreachable!()
    }
}
