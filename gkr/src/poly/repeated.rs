use crate::{
    poly::{
        BoxMultilinearPoly, BoxMultilinearPolyOwned, DenseMultilinearPoly, MultilinearPoly,
        MultilinearPolyOwned,
    },
    util::arithmetic::Field,
};
use std::{fmt::Debug, marker::PhantomData, ops::Index};

#[derive(Clone, Debug)]
pub struct RepeatedMultilinearPoly<F, T> {
    inner: T,
    log2_reps: usize,
    _marker: PhantomData<F>,
}

impl<F, T: MultilinearPoly<F>> RepeatedMultilinearPoly<F, T> {
    pub fn new(inner: T, log2_reps: usize) -> Self {
        Self {
            inner,
            log2_reps,
            _marker: PhantomData,
        }
    }
}

impl<F, T: MultilinearPoly<F>> RepeatedMultilinearPoly<F, T> {
    pub(crate) fn box_owned<'a>(self) -> BoxMultilinearPolyOwned<'a, F>
    where
        F: 'a,
        T: 'a,
        Self: MultilinearPolyOwned<F>,
    {
        Box::new(self)
    }
}

impl<F: Field, T: MultilinearPoly<F>> Index<usize> for RepeatedMultilinearPoly<F, T> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index < self.len());

        &self.inner[index % self.inner.len()]
    }
}

impl<F: Field, T: MultilinearPoly<F>> MultilinearPoly<F> for RepeatedMultilinearPoly<F, T> {
    fn clone_box(&self) -> BoxMultilinearPoly<F> {
        RepeatedMultilinearPoly::new(self.inner.clone_box(), self.log2_reps).boxed()
    }

    fn num_vars(&self) -> usize {
        self.inner.num_vars() + self.log2_reps
    }

    fn fix_var(&self, x_i: &F) -> BoxMultilinearPolyOwned<'static, F> {
        if self.inner.num_vars() > 0 {
            RepeatedMultilinearPoly::new(self.inner.fix_var(x_i), self.log2_reps).box_owned()
        } else {
            assert!(self.log2_reps > 0);

            let inner = DenseMultilinearPoly::new(vec![self.inner[0]]).box_owned();
            RepeatedMultilinearPoly::new(inner, self.log2_reps - 1).box_owned()
        }
    }

    fn evaluate(&self, x: &[F]) -> F {
        self.inner.evaluate(&x[..self.inner.num_vars()])
    }

    fn as_dense(&self) -> Option<&[F]> {
        None
    }
}

impl<F: Field> MultilinearPolyOwned<F>
    for RepeatedMultilinearPoly<F, BoxMultilinearPolyOwned<'static, F>>
{
    fn fix_var_in_place(&mut self, x_i: &F) {
        if self.inner.num_vars() > 0 {
            self.inner.fix_var_in_place(x_i);
        } else {
            assert!(self.log2_reps > 0);

            self.log2_reps -= 1;
        }
    }
}
