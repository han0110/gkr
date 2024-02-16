use crate::{
    circuit::node::{CombinedEvalClaim, EvalClaim, Node},
    poly::BoxMultilinearPoly,
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

impl<F, E> Node<F, E> for InputNode {
    fn is_input(&self) -> bool {
        true
    }

    fn log2_input_size(&self) -> usize {
        self.log2_size + self.log2_reps
    }

    fn log2_output_size(&self) -> usize {
        self.log2_size + self.log2_reps
    }

    fn evaluate(&self, _: Vec<&BoxMultilinearPoly<F, E>>) -> BoxMultilinearPoly<'static, F, E> {
        unreachable!()
    }

    fn prove_claim_reduction(
        &self,
        _: CombinedEvalClaim<E>,
        _: Vec<&BoxMultilinearPoly<F, E>>,
        _: &mut dyn TranscriptWrite<F, E>,
    ) -> Result<Vec<Vec<EvalClaim<E>>>, Error> {
        unreachable!()
    }

    fn verify_claim_reduction(
        &self,
        _: CombinedEvalClaim<E>,
        _: &mut dyn TranscriptRead<F, E>,
    ) -> Result<Vec<Vec<EvalClaim<E>>>, Error> {
        unreachable!()
    }
}
