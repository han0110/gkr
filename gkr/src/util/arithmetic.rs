use crate::util::{izip, izip_eq, Itertools};
use std::{
    borrow::Borrow,
    iter::{self},
    mem,
};

pub use halo2_curves::{
    ff::{BatchInvert, Field, PrimeField},
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

pub fn bool_to_felt<F: Field>(bit: bool) -> F {
    if bit {
        F::ONE
    } else {
        F::ZERO
    }
}

pub fn try_felt_to_bool<F: Field>(felt: F) -> Option<bool> {
    if felt == F::ONE {
        Some(true)
    } else if felt == F::ZERO {
        Some(false)
    } else {
        None
    }
}

pub fn vander_mat_inv<F: Field>(points: Vec<F>) -> Vec<Vec<F>> {
    let poly_from_roots = |roots: &[F], scalar: F| {
        let mut poly = vec![F::ZERO; roots.len() + 1];
        *poly.last_mut().unwrap() = scalar;
        izip!(2.., roots).for_each(|(len, root)| {
            let mut buf = scalar;
            (0..poly.len() - 1).rev().take(len).for_each(|idx| {
                buf = poly[idx] - buf * root;
                mem::swap(&mut buf, &mut poly[idx])
            })
        });
        poly
    };

    let mut mat = vec![vec![F::ZERO; points.len()]; points.len()];
    izip!(0.., &points).for_each(|(j, point_j)| {
        let point_is = izip!(0.., &points)
            .filter(|(i, _)| *i != j)
            .map(|(_, point_i)| *point_i)
            .collect_vec();
        let scalar = F::product(point_is.iter().map(|point_i| *point_j - point_i))
            .invert()
            .unwrap();
        izip!(&mut mat, poly_from_roots(&point_is, scalar)).for_each(|(row, coeff)| row[j] = coeff)
    });
    mat
}
