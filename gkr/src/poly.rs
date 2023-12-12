use crate::util::arithmetic::Field;
use rayon::prelude::*;
use std::{borrow::Cow, fmt::Debug, ops::Index};

mod binary;
mod dense;
mod eq;
mod repeated;

pub use binary::BinaryMultilinearPoly;
pub use dense::{box_dense_poly, repeated_dense_poly, DenseMultilinearPoly};
pub use eq::{eq_eval, eq_expand, eq_poly, PartialEqPoly};
pub use repeated::RepeatedMultilinearPoly;

pub type DynMultilinearPoly<'a, F> = dyn MultilinearPoly<F> + 'a;

pub type BoxMultilinearPoly<'a, F> = Box<DynMultilinearPoly<'a, F>>;

pub type DynMultilinearPolyOwned<'a, F> = dyn MultilinearPolyOwned<F> + 'a;

pub type BoxMultilinearPolyOwned<'a, F> = Box<DynMultilinearPolyOwned<'a, F>>;

#[allow(clippy::len_without_is_empty)]
pub trait MultilinearPoly<F>: Debug + Send + Sync + Index<usize, Output = F> {
    fn num_vars(&self) -> usize;

    fn len(&self) -> usize {
        1 << self.num_vars()
    }

    fn fix_var(&self, x_i: &F) -> BoxMultilinearPolyOwned<'static, F>;

    fn evaluate(&self, x: &[F]) -> F;

    fn as_dense(&self) -> Option<&[F]>;

    fn clone_box(&self) -> BoxMultilinearPoly<F>;

    fn boxed<'a>(self) -> BoxMultilinearPoly<'a, F>
    where
        Self: 'a + Sized,
    {
        Box::new(self)
    }

    fn repeated<'a>(self, log2_reps: usize) -> RepeatedMultilinearPoly<F, Self>
    where
        F: Field,
        Self: 'a + Sized,
    {
        RepeatedMultilinearPoly::new(self, log2_reps)
    }
}

pub trait MultilinearPolyOwned<F>: MultilinearPoly<F> {
    fn fix_var_in_place(&mut self, x_i: &F);
}

pub fn evaluate<F: Field>(evals: &[F], x: &[F]) -> F {
    assert_eq!(evals.len(), 1 << x.len());

    x.iter()
        .fold(Cow::Borrowed(evals), |evals, x_i| merge(&evals, x_i).into())[0]
}

fn merge<F: Field>(evals: &[F], x_i: &F) -> Vec<F> {
    let merge = |evals: &[_]| (evals[1] - evals[0]) * x_i + evals[0];
    evals.par_chunks(2).with_min_len(64).map(merge).collect()
}

macro_rules! forward_impl {
    (<$($generics:tt),*>, $type:ty $(, { $($custom:tt)* })?) => {
        impl<'a, F> Index<usize> for $type {
            type Output = F;

            fn index(&self, index: usize) -> &Self::Output {
                &(**self)[index]
            }
        }

        impl<$($generics),*> MultilinearPoly<F> for $type {
            fn clone_box(&self) -> BoxMultilinearPoly<F> {
                (**self).clone_box()
            }

            fn num_vars(&self) -> usize {
                (**self).num_vars()
            }

            fn len(&self) -> usize {
                (**self).len()
            }

            fn fix_var(&self, x_i: &F) -> BoxMultilinearPolyOwned<'static, F> {
                (**self).fix_var(x_i)
            }

            fn evaluate(&self, x: &[F]) -> F {
                (**self).evaluate(x)
            }

            fn as_dense(&self) -> Option<&[F]> {
                (**self).as_dense()
            }

            $($($custom)*)?
        }
    };
}

forward_impl!(<'a, F>, &DynMultilinearPoly<'a, F>);
forward_impl!(<'a, F>, BoxMultilinearPoly<'a, F>, {
    fn boxed<'b>(self) -> BoxMultilinearPoly<'b, F> where Self: 'b { self }
});
forward_impl!(<'a, F>, BoxMultilinearPolyOwned<'a, F>);

#[cfg(test)]
pub(crate) mod test {
    use crate::{
        poly::MultilinearPoly,
        util::{arithmetic::Field, izip_eq, Itertools},
    };

    pub(crate) fn assert_polys_eq<'a, F: Field>(
        lhs: impl IntoIterator<Item = &'a (impl MultilinearPoly<F> + 'a)>,
        rhs: impl IntoIterator<Item = &'a (impl MultilinearPoly<F> + 'a)>,
    ) {
        izip_eq!(lhs, rhs).for_each(|(lhs, rhs)| {
            assert_eq!(lhs.num_vars(), rhs.num_vars());
            (0..lhs.len()).for_each(|b| assert_eq!(lhs[b], rhs[b]));
        });
    }
}
