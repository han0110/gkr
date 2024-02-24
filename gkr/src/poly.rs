#![allow(clippy::len_without_is_empty)]

use crate::{
    izip_par,
    util::arithmetic::{ExtensionField, Field},
};
use rayon::prelude::*;
use std::{fmt::Debug, ops::Index};

mod binary;
mod dense;
mod eq;
mod repeated;

pub use binary::BinaryMultilinearPoly;
pub use dense::{box_dense_poly, box_owned_dense_poly, repeated_dense_poly, DenseMultilinearPoly};
pub use eq::{eq_eval, eq_expand, eq_poly, PartialEqPoly};
pub use repeated::RepeatedMultilinearPoly;

pub type DynMultilinearPoly<'a, F, E = F> = dyn MultilinearPoly<F, E> + 'a;

pub type BoxMultilinearPoly<'a, F, E = F> = Box<DynMultilinearPoly<'a, F, E>>;

pub type DynMultilinearPolyOwned<'a, F> = dyn MultilinearPolyOwned<F> + 'a;

pub type BoxMultilinearPolyOwned<'a, F> = Box<DynMultilinearPolyOwned<'a, F>>;

#[auto_impl::auto_impl(&, Box)]
pub trait MultilinearPoly<F, E = F>: Debug + Send + Sync + Index<usize, Output = F> {
    fn num_vars(&self) -> usize;

    fn len(&self) -> usize {
        1 << self.num_vars()
    }

    fn fix_var(&self, x_i: &E) -> BoxMultilinearPolyOwned<'static, E>;

    fn fix_vars(&self, x: &[E]) -> BoxMultilinearPolyOwned<'static, E> {
        assert!(x.len() <= self.num_vars());
        let (x_first, x) = x.split_first().unwrap();
        let mut poly = self.fix_var(x_first);
        x.iter().for_each(|x_i| poly.fix_var_in_place(x_i));
        poly
    }

    fn fix_var_last(&self, x_i: &E) -> BoxMultilinearPolyOwned<'static, E>;

    fn fix_vars_last(&self, x: &[E]) -> BoxMultilinearPolyOwned<'static, E> {
        assert!(x.len() <= self.num_vars());
        let (x_last, x) = x.split_last().unwrap();
        let mut poly = self.fix_var_last(x_last);
        x.iter()
            .rev()
            .for_each(|x_i| poly.fix_var_last_in_place(x_i));
        poly
    }

    fn evaluate(&self, x: &[E]) -> E;

    fn as_dense(&self) -> Option<&[F]> {
        None
    }

    fn to_dense(&self) -> Vec<F>
    where
        F: Send + Sync + Copy,
    {
        self.as_dense()
            .map(|dense| dense.par_iter().copied().collect())
            .unwrap_or_else(|| {
                (0..1 << self.num_vars())
                    .into_par_iter()
                    .map(|b| self[b])
                    .collect()
            })
    }

    fn clone_box(&self) -> BoxMultilinearPoly<F, E>;
}

pub trait MultilinearPolyOwned<F>: MultilinearPoly<F> {
    fn fix_var_in_place(&mut self, x_i: &F);

    fn fix_var_last_in_place(&mut self, x_i: &F);
}

pub trait MultilinearPolyExt<F, E = F>: MultilinearPoly<F, E> {
    fn boxed<'a>(self) -> BoxMultilinearPoly<'a, F, E>
    where
        Self: 'a + Sized,
    {
        Box::new(self)
    }

    fn repeated<'a>(self, log2_reps: usize) -> RepeatedMultilinearPoly<Self, F, E>
    where
        Self: 'a + Sized,
    {
        RepeatedMultilinearPoly::new(self, log2_reps)
    }
}

impl<F, E, P: MultilinearPoly<F, E>> MultilinearPolyExt<F, E> for P {}

pub fn evaluate<F: Field, E: ExtensionField<F>>(evals: &[F], x: &[E]) -> E {
    assert_eq!(evals.len(), 1 << x.len());

    x.split_first()
        .map(|(x_0, x)| {
            let init = merge(evals, x_0);
            x.iter().fold(init, |evals, x_i| merge(&evals, x_i))[0]
        })
        .unwrap_or_else(|| E::from(evals[0]))
}

pub fn merge<F: Field, E: ExtensionField<F>>(evals: &[F], x_i: &E) -> Vec<E> {
    let merge = |evals: &[_]| *x_i * (evals[1] - evals[0]) + evals[0];
    evals.par_chunks(2).with_min_len(64).map(merge).collect()
}

pub fn merge_last<F: Field, E: ExtensionField<F>>(evals: &[F], x_i: &E) -> Vec<E> {
    let merge = |(lo, hi): (&_, &_)| *x_i * (*hi - lo) + lo;
    let (lo, hi) = evals.split_at(evals.len() >> 1);
    izip_par!(lo, hi).with_min_len(64).map(merge).collect()
}

pub fn merge_last_in_place<F: Field>(evals: &mut Vec<F>, x_i: &F) {
    let merge = |(lo, hi): (&mut _, &mut _)| *lo += *x_i * (*hi - lo as &_);
    let mid = evals.len() >> 1;
    let (lo, hi) = evals.split_at_mut(mid);
    izip_par!(lo, hi).with_min_len(64).for_each(merge);
    evals.truncate(mid);
}

macro_rules! impl_index {
    (<$($generics:tt),*>, $type:ty) => {
        impl<$($generics),*> Index<usize> for $type {
            type Output = F;

            fn index(&self, index: usize) -> &Self::Output {
                &(**self)[index]
            }
        }
    };
}

impl_index!(<'a, F, E>, &DynMultilinearPoly<'a, F, E>);
impl_index!(<'a, F, E>, BoxMultilinearPoly<'a, F, E>);
impl_index!(<'a, F, E>, &BoxMultilinearPoly<'a, F, E>);
impl_index!(<'a, F>, BoxMultilinearPolyOwned<'a, F>);
impl_index!(<'a, F>, &BoxMultilinearPolyOwned<'a, F>);
