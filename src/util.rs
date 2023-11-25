use rayon::prelude::*;
use std::{
    borrow::Borrow,
    borrow::Cow,
    iter::Sum,
    ops::{Add, AddAssign, Deref, DerefMut},
};

pub use halo2_curves::ff::{Field, PrimeField};
pub use itertools::{chain, izip, Itertools};
pub use rand_core::RngCore;

pub fn div_ceil(dividend: usize, divisor: usize) -> usize {
    (dividend + divisor - 1) / divisor
}

pub fn horner<F: Field>(vs: &[F], x: &F) -> F {
    vs.iter().rev().fold(F::ZERO, |acc, v| acc * x + v)
}

pub fn inner_product<F: Field>(
    lhs: impl IntoIterator<Item = impl Borrow<F>>,
    rhs: impl IntoIterator<Item = impl Borrow<F>>,
) -> F {
    F::sum(izip_eq!(lhs, rhs).map(|(lhs, rhs)| *lhs.borrow() * rhs.borrow()))
}

pub fn hadamard_add<F: Field>(lhs: Cow<[F]>, rhs: &[F]) -> Vec<F> {
    let mut lhs = lhs.into_owned();
    izip_par!(&mut lhs, rhs).for_each(|(lhs, rhs)| *lhs += rhs);
    lhs
}

#[derive(Clone, Copy, Debug)]
pub struct AdditiveArray<F, const N: usize>(pub [F; N]);

impl<F, const N: usize> Deref for AdditiveArray<F, N> {
    type Target = [F; N];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F, const N: usize> DerefMut for AdditiveArray<F, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<F: Copy + Default, const N: usize> Default for AdditiveArray<F, N> {
    fn default() -> Self {
        Self([F::default(); N])
    }
}

impl<F: Field, const N: usize> AddAssign for AdditiveArray<F, N> {
    fn add_assign(&mut self, rhs: Self) {
        izip!(&mut self.0, &rhs.0).for_each(|(acc, item)| *acc += item);
    }
}

impl<F: Field, const N: usize> Add for AdditiveArray<F, N> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<F: Field, const N: usize> Sum for AdditiveArray<F, N> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, item| acc + item).unwrap_or_default()
    }
}

#[derive(Clone, Debug)]
pub struct AdditiveVec<F>(pub Vec<F>);

impl<F> Deref for AdditiveVec<F> {
    type Target = Vec<F>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F> DerefMut for AdditiveVec<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<F: Clone + Default> AdditiveVec<F> {
    pub fn new(len: usize) -> Self {
        Self(vec![F::default(); len])
    }
}

impl<F: Field> AddAssign for AdditiveVec<F> {
    fn add_assign(&mut self, rhs: Self) {
        izip_eq!(&mut self.0, &rhs.0).for_each(|(acc, item)| *acc += item);
    }
}

impl<F: Field> Add for AdditiveVec<F> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

macro_rules! chain_par {
    () => {
        rayon::iter::empty()
    };
    ($first:expr $(, $rest:expr)* $(,)?) => {
        {
            let iter = rayon::iter::IntoParallelIterator::into_par_iter($first);
            $(
                let iter = rayon::iter::ParallelIterator::chain(
                    iter,
                    rayon::iter::IntoParallelIterator::into_par_iter($rest),
                );
            )*
            iter
        }
    };
}

macro_rules! izip_par {
    (@closure $p:pat => $tup:expr) => {
        |$p| $tup
    };
    (@closure $p:pat => ($($tup:tt)*) , $_iter:expr $(, $tail:expr)*) => {
        $crate::util::izip_par!(@closure ($p, b) => ($($tup)*, b) $(, $tail)*)
    };
    ($first:expr $(,)*) => {
        rayon::iter::IntoParallelIterator::into_par_iter($first)
    };
    ($first:expr, $second:expr $(,)*) => {
        $crate::util::izip_par!($first).zip($second)
    };
    ($first:expr $(, $rest:expr)* $(,)*) => {
        $crate::util::izip_par!($first)
            $(.zip($rest))*
            .map($crate::util::izip_par!(@closure a => (a) $(, $rest)*))
    };
}

macro_rules! izip_eq {
    (@closure $p:pat => $tup:expr) => {
        |$p| $tup
    };
    (@closure $p:pat => ($($tup:tt)*) , $_iter:expr $(, $tail:expr)*) => {
        $crate::util::izip_eq!(@closure ($p, b) => ($($tup)*, b) $(, $tail)*)
    };
    ($first:expr $(,)*) => {
        itertools::__std_iter::IntoIterator::into_iter($first)
    };
    ($first:expr, $second:expr $(,)*) => {
        $crate::util::izip_eq!($first).zip_eq($second)
    };
    ($first:expr $(, $rest:expr)* $(,)*) => {
        $crate::util::izip_eq!($first)
            $(.zip_eq($rest))*
            .map($crate::util::izip_eq!(@closure a => (a) $(, $rest)*))
    };
}

pub(crate) use {chain_par, izip_eq, izip_par};

#[cfg(test)]
pub mod test {
    use crate::util::Field;
    use rand::{
        distributions::uniform::SampleRange,
        rngs::{OsRng, StdRng},
        CryptoRng, Rng, RngCore, SeedableRng,
    };
    use std::{array, iter};

    pub fn std_rng() -> impl RngCore + CryptoRng {
        StdRng::from_seed(Default::default())
    }

    pub fn seeded_std_rng() -> impl RngCore + CryptoRng {
        StdRng::seed_from_u64(OsRng.next_u64())
    }

    pub fn rand_range(range: impl SampleRange<usize>, mut rng: impl RngCore) -> usize {
        rng.gen_range(range)
    }

    pub fn rand_bool(mut rng: impl RngCore) -> bool {
        rng.gen_bool(0.5)
    }

    pub fn rand_array<F: Field, const N: usize>(mut rng: impl RngCore) -> [F; N] {
        array::from_fn(|_| F::random(&mut rng))
    }

    pub fn rand_vec<F: Field>(n: usize, mut rng: impl RngCore) -> Vec<F> {
        iter::repeat_with(|| F::random(&mut rng)).take(n).collect()
    }
}
