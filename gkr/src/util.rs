pub use itertools::{chain, izip, Itertools};
pub use rand_chacha::ChaCha12Rng as StdRng;
pub use rand_core::{CryptoRng, RngCore, SeedableRng};

pub mod arithmetic;
pub mod collection;

#[cfg(any(test, feature = "dev"))]
pub mod dev;

#[macro_export]
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

#[macro_export]
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

#[macro_export]
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

pub use {chain_par, izip_eq, izip_par};
