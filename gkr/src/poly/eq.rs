use crate::util::{
    arithmetic::{div_ceil, Field},
    chain, izip_eq, Itertools,
};
use rayon::prelude::*;
use std::{borrow::Cow, fmt::Debug, ops::Deref};

#[derive(Clone, Debug)]
pub struct PartialEqPoly<F> {
    r_hi: Vec<F>,
    eq_r_lo: Vec<F>,
}

impl<F> Deref for PartialEqPoly<F> {
    type Target = Vec<F>;

    fn deref(&self) -> &Self::Target {
        &self.eq_r_lo
    }
}

impl<F: Field> PartialEqPoly<F> {
    pub fn new(r: &[F], mid: usize, scalar: F) -> Self {
        let (r_lo, r_hi) = r.split_at(mid);
        PartialEqPoly {
            r_hi: r_hi.to_vec(),
            eq_r_lo: eq_expand(&[scalar], r_lo),
        }
    }

    pub fn r_hi(&self) -> &[F] {
        &self.r_hi
    }

    pub fn expand(&self) -> Vec<F> {
        eq_expand(&self.eq_r_lo, &self.r_hi)
    }
}

pub fn eq_poly<F: Field>(y: &[F], scalar: F) -> Vec<F> {
    eq_expand(&[scalar], y)
}

pub fn eq_expand<F: Field>(poly: &[F], y: &[F]) -> Vec<F> {
    assert!(poly.len().is_power_of_two());

    let poly_num_vars = poly.len().ilog2() as usize;
    let num_vars = poly_num_vars + y.len();
    let lo_num_vars = div_ceil(num_vars, 2).max(poly_num_vars);

    let (lo, hi) = if poly_num_vars >= lo_num_vars {
        let hi = eq_expand_serial(&[F::ONE], y);
        (Cow::Borrowed(poly), hi)
    } else {
        let (lo, hi) = y.split_at(lo_num_vars - poly_num_vars);
        rayon::join(
            || eq_expand_serial(poly, lo).into(),
            || eq_expand_serial(&[F::ONE], hi),
        )
    };

    let lo_mask = (1 << lo_num_vars) - 1;
    (0..1 << num_vars)
        .into_par_iter()
        .map(|b| lo[b & lo_mask] * hi[b >> lo_num_vars])
        .collect()
}

fn eq_expand_serial<F: Field>(poly: &[F], y: &[F]) -> Vec<F> {
    y.iter()
        .fold(Cow::Borrowed(poly), |poly, y_i| {
            let one_minus_y_i = F::ONE - y_i;
            chain![
                poly.iter().map(|eval| *eval * one_minus_y_i),
                poly.iter().map(|eval| *eval * y_i),
            ]
            .collect::<Vec<_>>()
            .into()
        })
        .into()
}

pub fn eq_eval<'a, F: Field>(rs: impl IntoIterator<Item = &'a [F]>) -> F {
    let rs = rs.into_iter().collect_vec();
    match rs.len() {
        2 => F::product(izip_eq!(rs[0], rs[1]).map(|(&x, &y)| (x * y).double() + F::ONE - x - y)),
        3 => F::product(
            izip_eq!(rs[0], rs[1], rs[2])
                .map(|(&x, &y, &z)| F::ONE + x * (y + z - F::ONE) + y * (z - F::ONE) - z),
        ),
        _ => unimplemented!(),
    }
}
