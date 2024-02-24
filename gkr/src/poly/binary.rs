use crate::{
    poly::{box_owned_dense_poly, BoxMultilinearPoly, BoxMultilinearPolyOwned, MultilinearPoly},
    util::arithmetic::{ExtensionField, Field},
};
use rayon::prelude::*;
use std::{fmt::Debug, ops::Index};

#[derive(Clone, Debug)]
pub struct BinaryMultilinearPoly<F> {
    evals: Vec<u64>,
    num_vars: usize,
    zero: F,
    one: F,
}

impl<F> BinaryMultilinearPoly<F> {
    pub const LOG2_BITS: usize = u64::BITS.ilog2() as usize;

    const MASK: usize = (1 << Self::LOG2_BITS) - 1;
}

impl<F: Field> BinaryMultilinearPoly<F> {
    pub fn new(evals: Vec<u64>, num_vars: usize) -> Self {
        if num_vars > Self::LOG2_BITS {
            assert_eq!(evals.len(), (1 << (num_vars - Self::LOG2_BITS)))
        } else {
            assert_eq!(evals.len(), 1);
            assert_eq!(evals[0], evals[0] & ((1 << (1 << num_vars)) - 1));
        }

        Self {
            num_vars,
            evals,
            zero: F::ZERO,
            one: F::ONE,
        }
    }

    #[inline(always)]
    fn bits<const N: usize>(&self, b: usize) -> usize {
        (self.evals[b >> Self::LOG2_BITS] as usize >> (b & Self::MASK)) & ((1 << N) - 1)
    }
}

impl<F: Field> Index<usize> for BinaryMultilinearPoly<F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index < 1 << self.num_vars);

        if (self.evals[index >> Self::LOG2_BITS] >> (index & Self::MASK)) & 1 == 1 {
            &self.one
        } else {
            &self.zero
        }
    }
}

impl<F: Field, E: ExtensionField<F>> MultilinearPoly<F, E> for BinaryMultilinearPoly<F> {
    fn clone_box(&self) -> BoxMultilinearPoly<F, E> {
        Box::new(self.clone())
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn fix_var(&self, x_i: &E) -> BoxMultilinearPolyOwned<'static, E> {
        let table = [E::ZERO, E::ONE - x_i, *x_i, E::ONE];
        let evals = (0..1 << self.num_vars)
            .into_par_iter()
            .step_by(2)
            .with_min_len(32)
            .map(|b| table[self.bits::<2>(b)])
            .collect();
        box_owned_dense_poly(evals)
    }

    fn fix_var_last(&self, x_i: &E) -> BoxMultilinearPolyOwned<'static, E> {
        let table = [E::ZERO, E::ONE - x_i, *x_i, E::ONE];
        let mid = 1 << (self.num_vars - 1);
        let evals = (0..mid)
            .into_par_iter()
            .with_min_len(32)
            .map(|b| table[self.bits::<1>(b) | self.bits::<1>(mid + b) << 1])
            .collect();
        box_owned_dense_poly(evals)
    }

    fn evaluate(&self, x: &[E]) -> E {
        assert_eq!(x.len(), self.num_vars);

        self.fix_var(&x[0]).evaluate(&x[1..])
    }
}
