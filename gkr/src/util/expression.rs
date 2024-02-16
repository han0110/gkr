use crate::util::{
    arithmetic::{powers, ExtensionField, Field},
    izip, Itertools,
};
use std::{
    borrow::Borrow,
    convert::identity,
    fmt::Debug,
    iter::{Product, Sum},
    ops::{Add, Deref, Mul, Neg, Sub},
};

#[derive(Clone, Debug)]
pub enum Expression<F, K> {
    Constant(F),
    Data(K),
    Neg(Box<Self>),
    Sum(Box<Self>, Box<Self>),
    Product(Box<Self>, Box<Self>),
}

impl<F, K> Expression<F, K> {
    pub fn constant(constant: F) -> Self {
        Self::Constant(constant)
    }

    pub fn distribute_powers(exprs: impl IntoIterator<Item = impl Borrow<Self>>, base: F) -> Self
    where
        F: Field,
        K: Clone,
    {
        izip!(exprs, powers(base))
            .map(|(expr, scalar)| expr.borrow() * Self::constant(scalar))
            .sum()
    }
}

impl<F: Clone, K: Clone> Expression<F, K> {
    pub fn evaluate<T>(
        &self,
        constant: &impl Fn(F) -> T,
        data: &impl Fn(K) -> T,
        neg: &impl Fn(T) -> T,
        sum: &impl Fn(T, T) -> T,
        product: &impl Fn(T, T) -> T,
    ) -> T {
        let evaluate = |expr: &Self| expr.evaluate(constant, data, neg, sum, product);
        match self {
            Expression::Constant(value) => constant(value.clone()),
            Expression::Data(key) => data(key.clone()),
            Expression::Neg(value) => neg(evaluate(value)),
            Expression::Sum(lhs, rhs) => sum(evaluate(lhs), evaluate(rhs)),
            Expression::Product(lhs, rhs) => product(evaluate(lhs), evaluate(rhs)),
        }
    }

    pub fn degree(&self) -> usize {
        self.evaluate(
            &|_| 0,
            &|_| 1,
            &|deg| deg,
            &|lhs, rhs| lhs.max(rhs),
            &|lhs, rhs| lhs + rhs,
        )
    }
}

impl<F> Expression<F, usize> {
    pub fn poly(poly: usize) -> Self {
        Self::Data(poly)
    }
}

macro_rules! impl_expression_ops {
    ($trait:ident, $op:ident, $variant:ident, $rhs_transformer:expr) => {
        impl<F: Clone, K: Clone> $trait<Expression<F, K>> for Expression<F, K> {
            type Output = Self;
            fn $op(self, rhs: Expression<F, K>) -> Self::Output {
                Expression::$variant((self).into(), $rhs_transformer(rhs).into())
            }
        }

        impl<F: Clone, K: Clone> $trait<Expression<F, K>> for &Expression<F, K> {
            type Output = Expression<F, K>;
            fn $op(self, rhs: Expression<F, K>) -> Self::Output {
                Expression::$variant((self.clone()).into(), $rhs_transformer(rhs).into())
            }
        }

        impl<F: Clone, K: Clone> $trait<&Expression<F, K>> for Expression<F, K> {
            type Output = Self;
            fn $op(self, rhs: &Expression<F, K>) -> Self::Output {
                Expression::$variant((self).into(), $rhs_transformer(rhs.clone()).into())
            }
        }

        impl<F: Clone, K: Clone> $trait<&Expression<F, K>> for &Expression<F, K> {
            type Output = Expression<F, K>;
            fn $op(self, rhs: &Expression<F, K>) -> Self::Output {
                Expression::$variant((self.clone()).into(), $rhs_transformer(rhs.clone()).into())
            }
        }
    };
}

impl_expression_ops!(Mul, mul, Product, identity);
impl_expression_ops!(Add, add, Sum, identity);
impl_expression_ops!(Sub, sub, Sum, Neg::neg);

impl<F: Clone, K: Clone> Neg for Expression<F, K> {
    type Output = Expression<F, K>;
    fn neg(self) -> Self::Output {
        Expression::Neg(Box::new(self))
    }
}

impl<F: Clone, K: Clone> Neg for &Expression<F, K> {
    type Output = Expression<F, K>;
    fn neg(self) -> Self::Output {
        -self.clone()
    }
}

impl<'a, F: Field, K: Clone + 'a> Sum<&'a Expression<F, K>> for Expression<F, K> {
    fn sum<I: Iterator<Item = &'a Expression<F, K>>>(iter: I) -> Self {
        iter.cloned().sum()
    }
}

impl<F: Field, K: Clone> Sum for Expression<F, K> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, item| acc + item)
            .unwrap_or_else(|| Expression::constant(F::ZERO))
    }
}

impl<'a, F: Field, K: Clone + 'a> Product<&'a Expression<F, K>> for Expression<F, K> {
    fn product<I: Iterator<Item = &'a Expression<F, K>>>(iter: I) -> Self {
        iter.cloned().product()
    }
}

impl<F: Field, K: Clone> Product for Expression<F, K> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, item| acc * item)
            .unwrap_or_else(|| Expression::constant(F::ONE))
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ExpressionRegistry<F, K> {
    offsets: Offsets,
    constants: Vec<F>,
    datas: Vec<K>,
    calcs: Vec<Calculation<usize>>,
    raw_calcs: Vec<Calculation<ValueSource>>,
}

impl<F: Field, K: Clone + Default + Eq> ExpressionRegistry<F, K> {
    pub(crate) fn new(expression: &Expression<F, K>) -> Self {
        let mut registry = Self {
            constants: vec![F::ZERO, F::ONE, F::ONE.double()],
            ..Default::default()
        };
        registry.register(expression);
        registry.offsets = Offsets::new(registry.constants.len(), registry.datas.len());
        registry.calcs = registry
            .raw_calcs
            .iter()
            .map(|calc| calc.indexed(&registry.offsets))
            .collect_vec();
        registry
    }

    pub(crate) fn offsets(&self) -> &Offsets {
        &self.offsets
    }

    pub(crate) fn datas(&self) -> &[K] {
        &self.datas
    }

    pub(crate) fn calcs(&self) -> &[Calculation<usize>] {
        &self.calcs
    }

    pub(crate) fn buffer<E: ExtensionField<F>>(&self) -> Vec<E> {
        let mut buf = vec![E::ZERO; self.offsets.calcs() + self.calcs.len()];
        izip!(&mut buf, &self.constants).for_each(|(buf, constant)| *buf = E::from_base(*constant));
        buf
    }

    fn register(&mut self, expression: &Expression<F, K>) {
        self.register_expression(expression);
    }

    fn register_value<T: Eq + Clone>(
        &mut self,
        field: impl FnOnce(&mut Self) -> &mut Vec<T>,
        item: &T,
    ) -> usize {
        let field = field(self);
        if let Some(idx) = field.iter().position(|lhs| lhs == item) {
            idx
        } else {
            let idx = field.len();
            field.push(item.clone());
            idx
        }
    }

    fn register_constant(&mut self, constant: impl Borrow<F>) -> ValueSource {
        ValueSource::Constant(self.register_value(|ev| &mut ev.constants, constant.borrow()))
    }

    fn register_polynomial(&mut self, data: impl Borrow<K>) -> ValueSource {
        ValueSource::Data(self.register_value(|ev| &mut ev.datas, data.borrow()))
    }

    fn register_calculation(&mut self, calc: Calculation<ValueSource>) -> ValueSource {
        ValueSource::Calculation(self.register_value(|ev| &mut ev.raw_calcs, &calc))
    }

    fn register_expression(&mut self, expr: &Expression<F, K>) -> ValueSource {
        match expr {
            Expression::Constant(constant) => self.register_constant(constant),
            Expression::Data(query) => self.register_polynomial(query),
            Expression::Neg(value) => {
                if let Expression::Constant(constant) = **value {
                    self.register_constant(-constant)
                } else {
                    let value = self.register_expression(value);
                    if let ValueSource::Constant(idx) = value {
                        self.register_constant(-self.constants[idx])
                    } else {
                        self.register_calculation(Calculation::Neg(value))
                    }
                }
            }
            Expression::Sum(lhs, rhs) => match (lhs.deref(), rhs.deref()) {
                (minuend, Expression::Neg(subtrahend)) | (Expression::Neg(subtrahend), minuend) => {
                    let minuend = self.register_expression(minuend);
                    let subtrahend = self.register_expression(subtrahend);
                    match (minuend, subtrahend) {
                        (ValueSource::Constant(minuend), ValueSource::Constant(subtrahend)) => self
                            .register_constant(
                                self.constants[minuend] - self.constants[subtrahend],
                            ),
                        (ValueSource::Constant(0), _) => {
                            self.register_calculation(Calculation::Neg(subtrahend))
                        }
                        (_, ValueSource::Constant(0)) => minuend,
                        _ => self.register_calculation(Calculation::Sub(minuend, subtrahend)),
                    }
                }
                _ => {
                    let lhs = self.register_expression(lhs);
                    let rhs = self.register_expression(rhs);
                    match (lhs, rhs) {
                        (ValueSource::Constant(lhs), ValueSource::Constant(rhs)) => {
                            self.register_constant(self.constants[lhs] + self.constants[rhs])
                        }
                        (ValueSource::Constant(0), other) | (other, ValueSource::Constant(0)) => {
                            other
                        }
                        _ => {
                            if lhs <= rhs {
                                self.register_calculation(Calculation::Add(lhs, rhs))
                            } else {
                                self.register_calculation(Calculation::Add(rhs, lhs))
                            }
                        }
                    }
                }
            },
            Expression::Product(lhs, rhs) => {
                let lhs = self.register_expression(lhs);
                let rhs = self.register_expression(rhs);
                match (lhs, rhs) {
                    (ValueSource::Constant(0), _) | (_, ValueSource::Constant(0)) => {
                        ValueSource::Constant(0)
                    }
                    (ValueSource::Constant(1), other) | (other, ValueSource::Constant(1)) => other,
                    (ValueSource::Constant(2), other) | (other, ValueSource::Constant(2)) => {
                        self.register_calculation(Calculation::Add(other, other))
                    }
                    (lhs, rhs) => {
                        if lhs <= rhs {
                            self.register_calculation(Calculation::Mul(lhs, rhs))
                        } else {
                            self.register_calculation(Calculation::Mul(rhs, lhs))
                        }
                    }
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Offsets(usize, usize);

impl Offsets {
    fn new(num_constants: usize, num_polys: usize) -> Self {
        let mut offset = Self::default();
        offset.0 = num_constants;
        offset.1 = offset.0 + num_polys;
        offset
    }

    pub(crate) fn datas(&self) -> usize {
        self.0
    }

    pub(crate) fn calcs(&self) -> usize {
        self.1
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Calculation<T> {
    Neg(T),
    Add(T, T),
    Sub(T, T),
    Mul(T, T),
}

impl Calculation<ValueSource> {
    fn indexed(&self, offsets: &Offsets) -> Calculation<usize> {
        use Calculation::*;
        match self {
            Neg(value) => Neg(value.indexed(offsets)),
            Add(lhs, rhs) => Add(lhs.indexed(offsets), rhs.indexed(offsets)),
            Sub(lhs, rhs) => Sub(lhs.indexed(offsets), rhs.indexed(offsets)),
            Mul(lhs, rhs) => Mul(lhs.indexed(offsets), rhs.indexed(offsets)),
        }
    }
}

impl Calculation<usize> {
    pub fn calculate<F: Field>(&self, buf: &mut [F], idx: usize) {
        use Calculation::*;
        buf[idx] = match self {
            Neg(idx) => -buf[*idx],
            Add(lhs, rhs) => buf[*lhs] + buf[*rhs],
            Sub(lhs, rhs) => buf[*lhs] - buf[*rhs],
            Mul(lhs, rhs) => buf[*lhs] * buf[*rhs],
        };
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum ValueSource {
    Constant(usize),
    Data(usize),
    Calculation(usize),
}

impl ValueSource {
    fn indexed(&self, offsets: &Offsets) -> usize {
        use ValueSource::*;
        match self {
            Constant(idx) => *idx,
            Data(idx) => offsets.datas() + idx,
            Calculation(idx) => offsets.calcs() + idx,
        }
    }
}
