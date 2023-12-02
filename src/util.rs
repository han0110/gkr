pub use itertools::{chain, izip, Itertools};
pub use rand_core::RngCore;

pub mod arithmetic;
pub mod collection;

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
    use crate::util::{arithmetic::Field, Itertools};
    use rand::{
        distributions::uniform::SampleRange,
        rngs::{OsRng, StdRng},
        CryptoRng, Rng, RngCore, SeedableRng,
    };
    use std::{array, hash::Hash, iter};

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

    pub fn rand_unique<T, R>(n: usize, f: impl Fn(&mut R) -> T, mut rng: R) -> Vec<T>
    where
        T: Clone + Eq + Hash,
        R: RngCore,
    {
        iter::repeat_with(|| f(&mut rng)).unique().take(n).collect()
    }
}
