use crate::util::{izip_eq, Itertools};
use std::{
    borrow::Borrow,
    iter::{self},
};

pub use halo2_curves::{
    ff::{Field, PrimeField},
    fft::best_fft as fft,
};

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

pub fn powers<F: Field>(base: F) -> impl Iterator<Item = F> {
    iter::successors(Some(F::ONE), move |power| Some(base * power))
}

pub fn squares<F: Field>(base: F) -> impl Iterator<Item = F> {
    iter::successors(Some(base), move |square| Some(square.square()))
}
