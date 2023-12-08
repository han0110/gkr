use crate::{
    chain_par,
    util::{arithmetic::Field, izip_eq, izip_par, Itertools},
};
use rayon::prelude::*;
use std::ops::Deref;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultilinearPoly<F> {
    num_vars: usize,
    evals: Vec<F>,
}

impl<F> MultilinearPoly<F> {
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }
}

impl<F: Clone> MultilinearPoly<F> {
    pub fn new(evals: Vec<F>) -> Self {
        let num_vars = if evals.is_empty() {
            0
        } else {
            let num_vars = evals.len().ilog2() as usize;
            assert_eq!(evals.len(), 1 << num_vars);
            num_vars
        };

        Self { evals, num_vars }
    }
}

impl<F: Field> MultilinearPoly<F> {
    pub fn fix_var(&mut self, x_i: &F) {
        self.num_vars -= 1;
        self.evals = izip_par!(self.evals.par_chunks(2))
            .with_min_len(64)
            .map(|eval| (eval[1] - eval[0]) * x_i + eval[0])
            .collect();
    }
}

impl<F> From<MultilinearPoly<F>> for Vec<F> {
    fn from(poly: MultilinearPoly<F>) -> Self {
        poly.evals
    }
}

impl<F> Deref for MultilinearPoly<F> {
    type Target = Vec<F>;

    fn deref(&self) -> &Self::Target {
        &self.evals
    }
}

pub fn evaluate<F: Field>(evals: &[F], x: &[F]) -> F {
    assert_eq!(evals.len(), 1 << x.len());

    if x.is_empty() {
        return evals[0];
    }

    let x_last = x.last().unwrap();
    let (lo, hi) = evals.split_at(evals.len() >> 1);
    let mut buf = lo.to_vec();
    izip_par!(&mut buf, hi).for_each(|(lo, hi)| *lo += (*hi - lo as &_) * x_last);
    x.iter().enumerate().rev().skip(1).for_each(|(idx, x_i)| {
        let (lo, hi) = buf.split_at_mut(1 << idx);
        izip_par!(lo, &hi[..1 << idx]).for_each(|(lo, hi)| *lo += (*hi - lo as &_) * x_i)
    });
    buf[0]
}

#[derive(Debug)]
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

pub fn eq_poly<F: Field>(y: &[F], scalar: F) -> MultilinearPoly<F> {
    MultilinearPoly::new(eq_expand(&[scalar], y))
}

pub fn eq_expand<F: Field>(poly: &[F], y: &[F]) -> Vec<F> {
    assert!(poly.len().is_power_of_two());

    y.iter().fold(poly.to_vec(), |poly, y_i| {
        let one_minus_y_i = F::ONE - y_i;
        chain_par![
            poly.par_iter().map(|eval| *eval * one_minus_y_i),
            poly.par_iter().map(|eval| *eval * y_i),
        ]
        .with_min_len(64)
        .collect()
    })
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
