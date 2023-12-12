use crate::{
    poly::{
        evaluate, merge, BoxMultilinearPoly, BoxMultilinearPolyOwned, MultilinearPoly,
        MultilinearPolyOwned,
    },
    util::arithmetic::Field,
};
use std::{fmt::Debug, marker::PhantomData, ops::Index};

#[derive(Clone, Debug)]
pub struct DenseMultilinearPoly<F, S: AsRef<[F]>> {
    evals: S,
    num_vars: usize,
    _marker: PhantomData<F>,
}

impl<F, S: AsRef<[F]>> DenseMultilinearPoly<F, S> {
    pub fn new(evals: S) -> Self {
        let num_vars = evals.as_ref().len().ilog2() as usize;
        assert_eq!(evals.as_ref().len(), 1 << num_vars);

        Self {
            evals,
            num_vars,
            _marker: PhantomData,
        }
    }
}

impl<F: Field> DenseMultilinearPoly<F, Vec<F>> {
    pub(crate) fn box_owned<'a>(self) -> BoxMultilinearPolyOwned<'a, F> {
        Box::new(self)
    }
}

impl<F, S: AsRef<[F]>> Index<usize> for DenseMultilinearPoly<F, S> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.evals.as_ref()[index]
    }
}

impl<F: Field, S: Clone + Debug + AsRef<[F]> + Send + Sync> MultilinearPoly<F>
    for DenseMultilinearPoly<F, S>
{
    fn clone_box(&self) -> BoxMultilinearPoly<F> {
        self.clone().boxed()
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn fix_var(&self, x_i: &F) -> BoxMultilinearPolyOwned<'static, F> {
        let evals = merge(self.evals.as_ref(), x_i);
        DenseMultilinearPoly::new(evals).box_owned()
    }

    fn evaluate(&self, x: &[F]) -> F {
        evaluate(self.evals.as_ref(), x)
    }

    fn as_dense(&self) -> Option<&[F]> {
        Some(self.evals.as_ref())
    }
}

impl<F: Field> MultilinearPolyOwned<F> for DenseMultilinearPoly<F, Vec<F>> {
    fn fix_var_in_place(&mut self, x_i: &F) {
        self.num_vars -= 1;
        self.evals = merge(&self.evals, x_i);
    }
}

pub fn box_dense_poly<'a, F, S>(evals: S) -> BoxMultilinearPoly<'a, F>
where
    F: Field,
    S: 'a + Clone + Debug + AsRef<[F]> + Send + Sync,
{
    DenseMultilinearPoly::new(evals).boxed()
}

pub fn repeated_dense_poly<'a, F, S>(evals: S, log2_reps: usize) -> BoxMultilinearPoly<'a, F>
where
    F: Field,
    S: 'a + Clone + Debug + AsRef<[F]> + Send + Sync,
{
    DenseMultilinearPoly::new(evals).repeated(log2_reps).boxed()
}
