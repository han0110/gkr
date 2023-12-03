use crate::{
    poly::MultilinearPoly,
    sum_check::SumCheckFunction,
    transcript::{TranscriptRead, TranscriptWrite},
    util::{
        arithmetic::{inner_product, powers, Field, PrimeField},
        chain,
        collection::AdditiveVec,
        izip, Itertools,
    },
    Error,
};
use rayon::prelude::*;
use std::{
    borrow::Borrow,
    fmt::Debug,
    iter::{Product, Sum},
    mem,
    ops::{Add, Deref, Mul, Neg, Sub},
};

#[derive(Clone, Debug)]
pub struct Generic<F: Field> {
    expression: Expression<F>,
    registry: ExpressionRegistry<F>,
    degree: usize,
    vander_mat_inv: Vec<Vec<F>>,
}

impl<F: Field> SumCheckFunction<F> for Generic<F> {
    fn degree(&self) -> usize {
        self.degree
    }

    fn compute_sum(&self, claim: F, polys: &[MultilinearPoly<F>]) -> Vec<F> {
        let registry = &self.registry;
        assert_eq!(polys.len(), registry.polys.len());

        if cfg!(feature = "sanity-check") {
            let evaluate = |b| {
                self.expression.evaluate(
                    &|constant| constant,
                    &|poly| polys[poly][b],
                    &|value| -value,
                    &|lhs, rhs| lhs + rhs,
                    &|lhs, rhs| lhs * rhs,
                )
            };
            assert_eq!(
                (0..polys[0].len()).into_par_iter().map(evaluate).sum::<F>(),
                claim,
            );
        }

        let buf = {
            let mut eval_buf = vec![F::ZERO; registry.offsets.calcs() + registry.calcs.len()];
            eval_buf[..registry.constants.len()].clone_from_slice(&registry.constants);
            let step_buf = vec![F::ZERO; registry.polys.len()];
            (eval_buf, step_buf)
        };

        let AdditiveVec(evals) = (0..polys[0].len() >> 1)
            .into_par_iter()
            .fold_with(
                (buf, AdditiveVec::new(self.degree)),
                |((mut eval_buf, mut step_buf), mut evals), b| {
                    izip!(
                        &mut eval_buf[registry.offsets.polys()..],
                        &mut step_buf,
                        &registry.polys,
                    )
                    .for_each(|(eval, step, poly)| {
                        *eval = polys[*poly][(b << 1) + 1];
                        *step = *eval - polys[*poly][b << 1];
                    });
                    izip!(registry.offsets.calcs().., &registry.indexed_calcs)
                        .for_each(|(idx, calc)| calc.calculate(&mut eval_buf, idx));
                    evals[0] += eval_buf.last().unwrap();

                    evals[1..].iter_mut().for_each(|eval| {
                        izip!(&mut eval_buf[registry.offsets.polys()..], &step_buf)
                            .for_each(|(eval, step)| *eval += step as &_);
                        izip!(registry.offsets.calcs().., &registry.indexed_calcs)
                            .for_each(|(idx, calc)| calc.calculate(&mut eval_buf, idx));
                        *eval += eval_buf.last().unwrap();
                    });

                    ((eval_buf, step_buf), evals)
                },
            )
            .map(|(_, evals)| evals)
            .reduce_with(|acc, item| acc + item)
            .unwrap();

        let evals = chain![[claim - evals[0]], evals].collect_vec();
        self.vander_mat_inv
            .iter()
            .map(|row| inner_product(row, &evals))
            .collect()
    }

    fn write_sum(
        &self,
        sum: &[F],
        transcript: &mut (impl TranscriptWrite<F> + ?Sized),
    ) -> Result<(), Error> {
        transcript.write_felt(&sum[0])?;
        transcript.write_felts(&sum[2..])?;
        Ok(())
    }

    fn read_sum(
        &self,
        claim: F,
        transcript: &mut (impl TranscriptRead<F> + ?Sized),
    ) -> Result<Vec<F>, Error> {
        let mut sum = vec![F::ZERO; self.degree + 1];
        sum[0] = transcript.read_felt()?;
        sum[2..].copy_from_slice(&transcript.read_felts(self.degree - 1)?);
        sum[1] = claim - sum[0].double() - sum[2..].iter().sum::<F>();
        Ok(sum)
    }
}

impl<F: PrimeField> Generic<F> {
    pub fn new(expression: &Expression<F>) -> Self {
        let registry = ExpressionRegistry::new(expression);
        let degree = expression.degree();
        assert!(degree >= 2);
        let vander_mat_inv = vander_mat_inv((0..).map(F::from).take(degree + 1).collect());
        Self {
            expression: expression.clone(),
            registry,
            degree,
            vander_mat_inv,
        }
    }

    pub fn expression(&self) -> &Expression<F> {
        &self.expression
    }
}

#[derive(Clone, Debug)]
pub enum Expression<F> {
    Constant(F),
    Polynomial(usize),
    Neg(Box<Expression<F>>),
    Sum(Box<Expression<F>>, Box<Expression<F>>),
    Product(Box<Expression<F>>, Box<Expression<F>>),
}

impl<F> Expression<F> {
    pub fn constant(constant: F) -> Self {
        Self::Constant(constant)
    }

    pub fn poly(poly: usize) -> Self {
        Self::Polynomial(poly)
    }

    pub fn distribute_powers(
        exprs: impl IntoIterator<Item = impl Borrow<Expression<F>>>,
        base: F,
    ) -> Self
    where
        F: Field,
    {
        izip!(exprs, powers(base))
            .map(|(expr, scalar)| expr.borrow() * Self::constant(scalar))
            .sum()
    }
}

impl<F: Field> Expression<F> {
    pub fn evaluate<T: Clone>(
        &self,
        constant: &impl Fn(F) -> T,
        poly: &impl Fn(usize) -> T,
        neg: &impl Fn(T) -> T,
        sum: &impl Fn(T, T) -> T,
        product: &impl Fn(T, T) -> T,
    ) -> T {
        let evaluate = |expr: &Expression<F>| expr.evaluate(constant, poly, neg, sum, product);
        match self {
            Expression::Constant(scalar) => constant(*scalar),
            Expression::Polynomial(idx) => poly(*idx),
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

macro_rules! impl_expression_ops {
    ($trait:ident, $op:ident, $variant:ident, $rhs:ty, $rhs_expr:expr) => {
        impl<F: Clone> $trait<$rhs> for Expression<F> {
            type Output = Expression<F>;
            fn $op(self, rhs: $rhs) -> Self::Output {
                Expression::$variant((self).into(), $rhs_expr(rhs).into())
            }
        }
        impl<F: Clone> $trait<$rhs> for &Expression<F> {
            type Output = Expression<F>;
            fn $op(self, rhs: $rhs) -> Self::Output {
                Expression::$variant((self.clone()).into(), $rhs_expr(rhs).into())
            }
        }
        impl<F: Clone> $trait<&$rhs> for Expression<F> {
            type Output = Expression<F>;
            fn $op(self, rhs: &$rhs) -> Self::Output {
                Expression::$variant((self).into(), $rhs_expr(rhs.clone()).into())
            }
        }
        impl<F: Clone> $trait<&$rhs> for &Expression<F> {
            type Output = Expression<F>;
            fn $op(self, rhs: &$rhs) -> Self::Output {
                Expression::$variant((self.clone()).into(), $rhs_expr(rhs.clone()).into())
            }
        }
    };
}

impl_expression_ops!(Mul, mul, Product, Expression<F>, std::convert::identity);
impl_expression_ops!(Add, add, Sum, Expression<F>, std::convert::identity);
impl_expression_ops!(Sub, sub, Sum, Expression<F>, Neg::neg);

impl<F: Clone> Neg for Expression<F> {
    type Output = Expression<F>;
    fn neg(self) -> Self::Output {
        -&self
    }
}

impl<F: Clone> Neg for &Expression<F> {
    type Output = Expression<F>;
    fn neg(self) -> Self::Output {
        Expression::Neg(Box::new(self.clone()))
    }
}

impl<'a, F: Field> Sum<&'a Expression<F>> for Expression<F> {
    fn sum<I: Iterator<Item = &'a Expression<F>>>(iter: I) -> Self {
        iter.cloned().sum()
    }
}

impl<F: Field> Sum for Expression<F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, item| acc + item)
            .unwrap_or_else(|| Expression::constant(F::ZERO))
    }
}

impl<'a, F: Field> Product<&'a Expression<F>> for Expression<F> {
    fn product<I: Iterator<Item = &'a Expression<F>>>(iter: I) -> Self {
        iter.cloned().product()
    }
}

impl<F: Field> Product for Expression<F> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, item| acc * item)
            .unwrap_or_else(|| Expression::constant(F::ONE))
    }
}

#[derive(Clone, Debug, Default)]
struct ExpressionRegistry<F: Field> {
    offsets: Offsets,
    constants: Vec<F>,
    polys: Vec<usize>,
    calcs: Vec<Calculation<ValueSource>>,
    indexed_calcs: Vec<Calculation<usize>>,
}

impl<F: Field> ExpressionRegistry<F> {
    pub fn new(expression: &Expression<F>) -> Self {
        let mut registry = Self {
            constants: vec![F::ZERO, F::ONE, F::ONE.double()],
            ..Default::default()
        };
        registry.register(expression);
        registry.offsets = Offsets::new(registry.constants.len(), registry.polys.len());
        registry.indexed_calcs = registry
            .calcs
            .iter()
            .map(|calc| calc.indexed(&registry.offsets))
            .collect_vec();
        registry
    }

    fn register(&mut self, expression: &Expression<F>) {
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

    fn register_polynomial(&mut self, poly: impl Borrow<usize>) -> ValueSource {
        ValueSource::Polynomial(self.register_value(|ev| &mut ev.polys, poly.borrow()))
    }

    fn register_calculation(&mut self, calc: Calculation<ValueSource>) -> ValueSource {
        ValueSource::Calculation(self.register_value(|ev| &mut ev.calcs, &calc))
    }

    fn register_expression(&mut self, expr: &Expression<F>) -> ValueSource {
        match expr {
            Expression::Constant(constant) => self.register_constant(constant),
            Expression::Polynomial(query) => self.register_polynomial(query),
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
struct Offsets(usize, usize);

impl Offsets {
    fn new(num_constants: usize, num_polys: usize) -> Self {
        let mut offset = Self::default();
        offset.0 = num_constants;
        offset.1 = offset.0 + num_polys;
        offset
    }

    fn polys(&self) -> usize {
        self.0
    }

    fn calcs(&self) -> usize {
        self.1
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum ValueSource {
    Constant(usize),
    Polynomial(usize),
    Calculation(usize),
}

impl ValueSource {
    fn indexed(&self, offsets: &Offsets) -> usize {
        use ValueSource::*;
        match self {
            Constant(idx) => *idx,
            Polynomial(idx) => offsets.polys() + idx,
            Calculation(idx) => offsets.calcs() + idx,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Calculation<T> {
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
    fn calculate<F: Field>(&self, buf: &mut [F], idx: usize) {
        use Calculation::*;
        buf[idx] = match self {
            Neg(idx) => -buf[*idx],
            Add(lhs, rhs) => buf[*lhs] + buf[*rhs],
            Sub(lhs, rhs) => buf[*lhs] - buf[*rhs],
            Mul(lhs, rhs) => buf[*lhs] * buf[*rhs],
        };
    }
}

fn vander_mat_inv<F: Field>(points: Vec<F>) -> Vec<Vec<F>> {
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
