use crate::util::{
    div_ceil, izip, izip_eq, num_threads, parallelize, parallelize_iter, Field, Itertools,
};
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
    if scalar.is_zero_vartime() {
        return vec![F::ZERO; 1 << y.len()];
    }

    let expand = |next: &mut [F], buf: &[F], y_i: &F| {
        izip!(next.chunks_mut(2), buf).for_each(|(next, buf)| {
            next[1] = *buf * y_i;
            next[0] = *buf - next[1];
        })
    };

    let mut buf = vec![scalar];
    for y_i in y.iter().rev() {
        let mut next = vec![F::ZERO; 2 * buf.len()];
        let chunk_size = div_ceil(buf.len(), num_threads());
        parallelize_iter(
            izip!(next.chunks_mut(chunk_size << 1), buf.chunks(chunk_size)),
            |(next, buf)| expand(next, buf, y_i),
        );
        buf = next;
    }
    buf
}

pub fn eq_eval<F: Field, const N: usize>(rs: [&[F]; N]) -> F {
    match N {
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

    parallelize(target, |(target, start)| {
        let x_i = *x_i;
        izip!(target, evals[start << 1..].iter().tuples()).for_each(|(target, (eval_0, eval_1))| {
            *target = (*eval_1 - eval_0) * x_i + eval_0;
        })
    });
}
