use crate::{
    poly::{BoxMultilinearPoly, BoxMultilinearPolyOwned, DenseMultilinearPoly, MultilinearPoly},
    util::arithmetic::Field,
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

impl<F: Field> MultilinearPoly<F> for BinaryMultilinearPoly<F> {
    fn clone_box(&self) -> BoxMultilinearPoly<F> {
        Box::new(self.clone())
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn fix_var(&self, x_i: &F) -> BoxMultilinearPolyOwned<'static, F> {
        let table = [F::ZERO, F::ONE - x_i, *x_i, F::ONE];
        let evals = (0..1 << self.num_vars)
            .into_par_iter()
            .step_by(2)
            .with_min_len(32)
            .map(|b| table[(self.evals[b >> Self::LOG2_BITS] as usize >> (b & Self::MASK)) & 0b11])
            .collect();
        DenseMultilinearPoly::new(evals).box_owned()
    }

    fn evaluate(&self, x: &[F]) -> F {
        assert_eq!(x.len(), self.num_vars);

        self.fix_var(&x[0]).evaluate(&x[1..])
    }

    fn as_dense(&self) -> Option<&[F]> {
        None
    }
}
