use crate::{
    chain_par,
    util::{arithmetic::Field, izip_eq, Itertools},
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

    y.iter()
        .fold(Cow::Borrowed(poly), |poly, y_i| {
            let one_minus_y_i = F::ONE - y_i;
            chain_par![
                poly.par_iter().map(|eval| *eval * one_minus_y_i),
                poly.par_iter().map(|eval| *eval * y_i),
            ]
            .with_min_len(64)
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
