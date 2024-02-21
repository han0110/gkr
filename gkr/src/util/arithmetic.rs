use crate::util::{izip, izip_eq, Itertools};
use rayon::{current_num_threads, prelude::*};
use std::{borrow::Borrow, iter, mem};

pub use ff_ext::{
    ff::{BatchInvert, Field, PrimeField},
    ExtensionField,
};

pub trait ParallelBatchInvert<F: Field> {
    fn par_batch_invert(&mut self);
}

impl<F: Field> ParallelBatchInvert<F> for [F] {
    fn par_batch_invert(&mut self) {
        let chunk_size = div_ceil(self.as_parallel_slice_mut().len(), current_num_threads());
        self.par_chunks_mut(chunk_size).for_each(|chunk| {
            chunk.batch_invert();
        });
    }
}

pub fn div_ceil(dividend: usize, divisor: usize) -> usize {
    (dividend + divisor - 1) / divisor
}

pub fn horner<F: Field, E: ExtensionField<F>>(vs: &[F], x: &E) -> E {
    vs.iter().rev().fold(E::ZERO, |acc, v| acc * x + v)
}

pub fn inner_product<F: Field, E: ExtensionField<F>>(
    lhs: impl IntoIterator<Item = impl Borrow<F>>,
    rhs: impl IntoIterator<Item = impl Borrow<E>>,
) -> E {
    E::sum(izip_eq!(lhs, rhs).map(|(lhs, rhs)| *rhs.borrow() * lhs.borrow()))
}

pub fn steps<F: Field>(start: F) -> impl Iterator<Item = F> {
    iter::successors(Some(start), move |acc| Some(F::ONE + acc))
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

pub fn radix2_fft<F: Field>(buf: &mut [F], omega: F) {
    let n = buf.len();
    let log2_n = n.ilog2() as usize;
    assert!(n.is_power_of_two());

    for k in 0..n {
        let rk = bitreverse(k, log2_n);
        if k < rk {
            buf.swap(rk, k);
        }
    }

    let twiddles = powers(omega).take(n / 2).collect_vec();

    recursive_butterfly_arithmetic(buf, 1, &twiddles);

    fn bitreverse(mut k: usize, log2_n: usize) -> usize {
        let mut rk = 0;
        for _ in 0..log2_n {
            rk = (rk << 1) | (k & 1);
            k >>= 1;
        }
        rk
    }

    fn recursive_butterfly_arithmetic<F: Field>(buf: &mut [F], step: usize, twiddles: &[F]) {
        let mid = buf.len() / 2;
        let (a, b) = buf.split_at_mut(mid);
        if mid == 1 {
            let t = b[0];
            b[0] = a[0];
            a[0] += &t;
            b[0] -= &t;
        } else {
            rayon::join(
                || recursive_butterfly_arithmetic(a, step * 2, twiddles),
                || recursive_butterfly_arithmetic(b, step * 2, twiddles),
            );

            let (a_0, a) = a.split_first_mut().unwrap();
            let (b_0, b) = b.split_first_mut().unwrap();
            let t = *b_0;
            *b_0 = *a_0;
            *a_0 += &t;
            *b_0 -= &t;

            izip!(a, b, twiddles.iter().step_by(step).skip(1)).for_each(|(a, b, twiddle)| {
                let t = *b * twiddle;
                *b = *a;
                *a += &t;
                *b -= &t;
            });
        }
    }
}
