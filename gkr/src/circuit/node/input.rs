use crate::{
    circuit::node::{CombinedEvalClaim, EvalClaim, Node},
    poly::{BoxMultilinearPoly, DynMultilinearPoly},
    transcript::{TranscriptRead, TranscriptWrite},
    Error,
};

#[derive(Clone, Debug)]
pub struct InputNode {
    log2_size: usize,
    log2_reps: usize,
}

impl InputNode {
    pub fn new(log2_size: usize, num_reps: usize) -> Self {
        assert!(num_reps != 0);

        Self {
            log2_size,
            log2_reps: num_reps.next_power_of_two().ilog2() as usize,
        }
    }
}

impl<F> Node<F> for InputNode {
    fn is_input(&self) -> bool {
        true
    }

    fn log2_input_size(&self) -> usize {
        self.log2_size + self.log2_reps
    }

    fn log2_output_size(&self) -> usize {
        self.log2_size + self.log2_reps
    }

    fn evaluate(&self, _: Vec<&DynMultilinearPoly<F>>) -> BoxMultilinearPoly<'static, F> {
        unreachable!()
    }

    fn prove_claim_reduction(
        &self,
        _: CombinedEvalClaim<F>,
        _: Vec<&DynMultilinearPoly<F>>,
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
