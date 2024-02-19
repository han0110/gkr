use crate::util::arithmetic::{ExtensionField, Field};
use rayon::prelude::*;
use std::{fmt::Debug, ops::Index};

mod binary;
mod dense;
mod eq;
mod repeated;

pub use binary::BinaryMultilinearPoly;
pub use dense::{box_dense_poly, repeated_dense_poly, DenseMultilinearPoly};
pub use eq::{eq_eval, eq_expand, eq_poly, PartialEqPoly};
pub use repeated::RepeatedMultilinearPoly;

pub type DynMultilinearPoly<'a, F, E = F> = dyn MultilinearPoly<F, E> + 'a;

pub type BoxMultilinearPoly<'a, F, E = F> = Box<DynMultilinearPoly<'a, F, E>>;

pub type DynMultilinearPolyOwned<'a, F> = dyn MultilinearPolyOwned<F> + 'a;

pub type BoxMultilinearPolyOwned<'a, F> = Box<DynMultilinearPolyOwned<'a, F>>;

#[allow(clippy::len_without_is_empty)]
pub trait MultilinearPoly<F, E = F>: Debug + Send + Sync + Index<usize, Output = F> {
    fn num_vars(&self) -> usize;

    fn len(&self) -> usize {
        1 << self.num_vars()
    }

    fn fix_var(&self, x_i: &E) -> BoxMultilinearPolyOwned<'static, E>;

    fn evaluate(&self, x: &[E]) -> E;

    fn as_dense(&self) -> Option<&[F]>;

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

pub trait MultilinearPolyOwned<F>: MultilinearPoly<F> {
    fn fix_var_in_place(&mut self, x_i: &F);
}

pub fn evaluate<F: Field, E: ExtensionField<F>>(evals: &[F], x: &[E]) -> E {
    assert_eq!(evals.len(), 1 << x.len());

    x.split_first()
        .map(|(x_0, x)| {
            let init = merge(evals, x_0);
            x.iter().fold(init, |evals, x_i| merge(&evals, x_i))[0]
        })
        .unwrap_or_else(|| E::from_base(evals[0]))
}

pub fn merge<F: Field, E: ExtensionField<F>>(evals: &[F], x_i: &E) -> Vec<E> {
    let merge = |evals: &[_]| *x_i * (evals[1] - evals[0]) + evals[0];
    evals.par_chunks(2).with_min_len(64).map(merge).collect()
}

macro_rules! forward_impl {
    (<$($generics:tt),*>, $ext:tt, $type:ty $(, { $($custom:tt)* })?) => {
        impl<$($generics),*> Index<usize> for $type {
            type Output = F;

            fn index(&self, index: usize) -> &Self::Output {
                &(**self)[index]
            }
        }

        impl<$($generics),*> MultilinearPoly<F, $ext> for $type {
            fn clone_box(&self) -> BoxMultilinearPoly<F, $ext> {
                (**self).clone_box()
            }

            fn num_vars(&self) -> usize {
                (**self).num_vars()
            }

            fn len(&self) -> usize {
                (**self).len()
            }

            fn fix_var(&self, x_i: &$ext) -> BoxMultilinearPolyOwned<'static, $ext> {
                (**self).fix_var(x_i)
            }

            fn evaluate(&self, x: &[$ext]) -> $ext {
                (**self).evaluate(x)
            }

            fn as_dense(&self) -> Option<&[F]> {
                (**self).as_dense()
            }

            fn to_dense(&self) -> Vec<F>
            where
                F: Send + Sync + Copy,
            {
                (**self).to_dense()
            }

            $($($custom)*)?
        }
    };
}

forward_impl!(<'a, F, E>, E, &DynMultilinearPoly<'a, F, E>);
forward_impl!(<'a, F, E>, E, BoxMultilinearPoly<'a, F, E>, {
    fn boxed<'b>(self) -> BoxMultilinearPoly<'b, F, E> where Self: 'b { self }
});
forward_impl!(<'a, F, E>, E, &BoxMultilinearPoly<'a, F, E>);
forward_impl!(<'a, F>, F, BoxMultilinearPolyOwned<'a, F>);
forward_impl!(<'a, F>, F, &BoxMultilinearPolyOwned<'a, F>);

#[cfg(test)]
pub(crate) mod test {
    use crate::{
        izip_par,
        poly::MultilinearPoly,
        util::{
            arithmetic::{ExtensionField, Field},
            Itertools,
        },
    };
    use rayon::prelude::*;

    pub(crate) fn assert_polys_eq<F: Field, E: ExtensionField<F>>(
        lhs: impl IntoIterator<Item = impl MultilinearPoly<F, E>>,
        rhs: impl IntoIterator<Item = impl MultilinearPoly<F, E>>,
    ) {
        let lhs = lhs.into_iter().collect_vec();
        let rhs = rhs.into_iter().collect_vec();
        assert_eq!(lhs.len(), rhs.len());
        izip_par!(lhs, rhs).for_each(|(lhs, rhs)| assert_eq!(lhs.to_dense(), rhs.to_dense()));
    }
}
