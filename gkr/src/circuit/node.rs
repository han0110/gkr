use crate::{
    poly::BoxMultilinearPoly,
    transcript::{TranscriptRead, TranscriptWrite},
    util::arithmetic::{inner_product, Field},
    Error,
};
use std::fmt::Debug;

mod fft;
mod input;
mod log_up;
mod vanilla;

pub use fft::FftNode;
pub use input::InputNode;
pub use log_up::LogUpNode;
pub use vanilla::{VanillaGate, VanillaNode};

#[auto_impl::auto_impl(&, Box)]
pub trait Node<F, E>: Debug {
    fn input_size(&self) -> usize {
        1 << self.log2_input_size()
    }

    fn output_size(&self) -> usize {
        1 << self.log2_output_size()
    }

    fn is_input(&self) -> bool;

    fn log2_input_size(&self) -> usize;

    fn log2_output_size(&self) -> usize;

    fn evaluate(&self, inputs: Vec<&BoxMultilinearPoly<F, E>>)
        -> BoxMultilinearPoly<'static, F, E>;

    fn prove_claim_reduction(
        &self,
        claim: CombinedEvalClaim<E>,
        inputs: Vec<&BoxMultilinearPoly<F, E>>,
        transcript: &mut dyn TranscriptWrite<F, E>,
    ) -> Result<Vec<Vec<EvalClaim<E>>>, Error>;

    fn verify_claim_reduction(
        &self,
        claim: CombinedEvalClaim<E>,
        transcript: &mut dyn TranscriptRead<F, E>,
    ) -> Result<Vec<Vec<EvalClaim<E>>>, Error>;
}

pub trait NodeExt<F, E>: Node<F, E> {
    fn boxed<'a>(self) -> Box<dyn Node<F, E> + 'a>
    where
        Self: 'a + Sized,
    {
        Box::new(self)
    }
}

impl<F, E, N: Node<F, E>> NodeExt<F, E> for N {
    fn boxed<'a>(self) -> Box<dyn Node<F, E> + 'a>
    where
        Self: 'a + Sized,
    {
        Box::new(self)
    }
}

#[derive(Clone, Debug, Default)]
pub struct EvalClaim<F> {
    point: Vec<F>,
    value: F,
}

impl<F> EvalClaim<F> {
    pub fn new(point: Vec<F>, value: F) -> Self {
        Self { point, value }
    }

    pub fn point(&self) -> &[F] {
        &self.point
    }

    pub fn value(&self) -> F
    where
        F: Copy,
    {
        self.value
    }
}

#[derive(Clone, Debug)]
pub struct CombinedEvalClaim<F> {
    points: Vec<Vec<F>>,
    alphas: Vec<F>,
    value: F,
}

impl<F: Field> CombinedEvalClaim<F> {
    pub fn new(claims: Vec<EvalClaim<F>>, alphas: Vec<F>) -> Self {
        assert!(!claims.is_empty());
        assert_eq!(claims.len(), alphas.len());
        let value = inner_product(&alphas, claims.iter().map(EvalClaim::value));
        let points = claims.into_iter().map(|claim| claim.point).collect();
        Self {
            points,
            alphas,
            value,
        }
    }
}
