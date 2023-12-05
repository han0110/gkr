use crate::util::{arithmetic::Field, izip, izip_eq, izip_par, Itertools};
use rayon::prelude::*;
use std::{
    array,
    iter::Sum,
    ops::{Add, AddAssign, Deref, DerefMut},
};

pub trait Hadamard<T>: Iterator {
    fn hada_sum(self) -> Vec<T>;
}

impl<F, V, I> Hadamard<F> for I
where
    F: Field,
    V: Into<Vec<F>> + AsRef<[F]>,
    I: Iterator<Item = V>,
{
    fn hada_sum(mut self) -> Vec<F> {
        let init = self.next().unwrap();
        self.fold(init.into(), |mut acc, item| {
            izip_par!(&mut acc, item.as_ref()).for_each(|(acc, item)| *acc += item);
            acc
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AdditiveArray<F, const N: usize>(pub [F; N]);

impl<F, const N: usize> Deref for AdditiveArray<F, N> {
    type Target = [F; N];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F, const N: usize> DerefMut for AdditiveArray<F, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<F: Default, const N: usize> Default for AdditiveArray<F, N> {
    fn default() -> Self {
        Self(array::from_fn(|_| F::default()))
    }
}

impl<F: AddAssign, const N: usize> AddAssign for AdditiveArray<F, N> {
    fn add_assign(&mut self, rhs: Self) {
        izip!(&mut self.0, rhs.0).for_each(|(acc, item)| *acc += item);
    }
}

impl<F: AddAssign, const N: usize> Add for AdditiveArray<F, N> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}

impl<F: AddAssign + Default, const N: usize> Sum for AdditiveArray<F, N> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, item| acc + item).unwrap_or_default()
    }
}

#[derive(Clone, Debug)]
pub struct AdditiveVec<F>(pub Vec<F>);

impl<F> Deref for AdditiveVec<F> {
    type Target = Vec<F>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F> DerefMut for AdditiveVec<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<F: Clone + Default> AdditiveVec<F> {
    pub fn new(len: usize) -> Self {
        Self(vec![F::default(); len])
    }
}

impl<F: AddAssign> AddAssign for AdditiveVec<F> {
    fn add_assign(&mut self, rhs: Self) {
        izip_eq!(&mut self.0, rhs.0).for_each(|(acc, item)| *acc += item);
    }
}

impl<F: AddAssign> Add for AdditiveVec<F> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self += rhs;
        self
    }
}
