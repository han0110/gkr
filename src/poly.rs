use crate::util::{izip, izip_eq, izip_par, Field, Itertools};
use rayon::prelude::*;
use std::{borrow::Cow, mem, ops::Deref};

#[derive(Clone, Debug)]
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
    pub fn new(evals: Cow<[F]>) -> Self {
        let evals = evals.into_owned();
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
        let mut buf = vec![F::ZERO; self.evals.len() >> 1];
        merge_into(&mut buf, &self.evals, x_i);
        self.num_vars -= 1;
        self.evals = buf;
    }
}

impl<F> Deref for MultilinearPoly<F> {
    type Target = Vec<F>;

    fn deref(&self) -> &Self::Target {
        &self.evals
    }
}

pub fn eq_poly<F: Field>(y: &[F], scalar: F) -> Vec<F> {
    eq_expand(&[scalar], y)
}

pub fn eq_expand<F: Field>(poly: &[F], y: &[F]) -> Vec<F> {
    assert!(poly.len().is_power_of_two());

    let mut buf = vec![F::ZERO; poly.len() << y.len()];
    buf[..poly.len()].copy_from_slice(poly);
    for (idx, y_i) in izip!(poly.len().ilog2().., y) {
        let (lo, hi) = buf[..2 << idx].split_at_mut(1 << idx);
        izip_par!(hi as &mut [_], lo as &[_]).for_each(|(hi, lo)| *hi = *lo * y_i);
        izip_par!(lo as &mut [_], hi as &[_]).for_each(|(lo, hi)| *lo -= hi);
    }
    buf
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

pub fn evaluate<F: Field>(evals: &[F], x: &[F]) -> F {
    assert_eq!(1 << x.len(), evals.len());

    let evals = &mut Cow::Borrowed(evals);
    let buf = &mut Vec::with_capacity(evals.len() >> 1);
    x.iter().for_each(|x_i| merge_in_place(evals, buf, x_i));
    evals[0]
}

fn merge_in_place<F: Field>(evals: &mut Cow<[F]>, buf: &mut Vec<F>, x_i: &F) {
    merge_into(buf, evals, x_i);
    if let Cow::Owned(_) = evals {
        mem::swap(evals.to_mut(), buf);
    } else {
        *evals = mem::replace(buf, Vec::with_capacity(buf.len() >> 1)).into();
    }
}

fn merge_into<F: Field>(target: &mut Vec<F>, evals: &[F], x_i: &F) {
    assert!(target.capacity() >= evals.len() >> 1);
    target.resize(evals.len() >> 1, F::ZERO);

    izip_par!(target, izip_par!(&evals[0..], &evals[1..]).step_by(2))
        .for_each(|(target, (eval_0, eval_1))| *target = (*eval_1 - eval_0) * x_i + eval_0);
}
