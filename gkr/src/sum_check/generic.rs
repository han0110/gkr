use crate::{
    poly::MultilinearPoly,
    sum_check::{SumCheckFunction, SumCheckPoly},
    transcript::{TranscriptRead, TranscriptWrite},
    util::{
        arithmetic::{inner_product, steps, vander_mat_inv, ExtensionField, Field},
        chain,
        collection::AdditiveVec,
        expression::{Expression, ExpressionRegistry},
        izip, Itertools,
    },
    Error,
};
use rayon::prelude::*;
use std::fmt::Debug;

#[derive(Clone, Debug)]
pub struct Generic<F: Field, E: ExtensionField<F>> {
    num_vars: usize,
    expression: Expression<E, usize>,
    registry: ExpressionRegistry<E, usize>,
    degree: usize,
    vander_mat_inv: Vec<Vec<F>>,
}

impl<F: Field, E: ExtensionField<F>> SumCheckFunction<F, E> for Generic<F, E> {
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn degree(&self) -> usize {
        self.degree
    }

    fn compute_sum(
        &self,
        _: usize,
        claim: E,
        polys: &[SumCheckPoly<F, E, impl MultilinearPoly<F, E>, impl MultilinearPoly<E, E>>],
    ) -> Vec<E> {
        let registry = &self.registry;
        assert_eq!(polys.len(), registry.datas().len());

        if cfg!(feature = "sanity-check") {
            let evaluate = |b| {
                self.expression.evaluate(
                    &|constant| constant,
                    &|poly| match &polys[poly] {
                        SumCheckPoly::Base(poly) => E::from(poly[b]),
                        SumCheckPoly::Extension(poly) => poly[b],
                        _ => unreachable!(),
                    },
                    &|value| -value,
                    &|lhs, rhs| lhs + rhs,
                    &|lhs, rhs| lhs * rhs,
                )
            };
            assert_eq!(
                (0..polys[0].len()).into_par_iter().map(evaluate).sum::<E>(),
                claim,
            );
        }

        let buf = (registry.buffer(), vec![E::ZERO; registry.datas().len()]);
        let AdditiveVec(evals) = (0..polys[0].len() >> 1)
            .into_par_iter()
            .fold_with(
                (buf, AdditiveVec::new(self.degree)),
                |((mut eval_buf, mut step_buf), mut evals), b| {
                    izip!(
                        &mut eval_buf[registry.offsets().datas()..],
                        &mut step_buf,
                        registry.datas(),
                    )
                    .for_each(|(eval, step, poly)| match &polys[*poly] {
                        SumCheckPoly::Base(poly) => {
                            *eval = E::from(poly[(b << 1) + 1]);
                            *step = E::from(poly[(b << 1) + 1] - poly[b << 1]);
                        }
                        SumCheckPoly::Extension(poly) => {
                            *eval = poly[(b << 1) + 1];
                            *step = poly[(b << 1) + 1] - poly[b << 1];
                        }
                        _ => unreachable!(),
                    });
                    izip!(registry.offsets().calcs().., registry.calcs())
                        .for_each(|(idx, calc)| calc.calculate(&mut eval_buf, idx));
                    evals[0] += eval_buf.last().unwrap();

                    evals[1..].iter_mut().for_each(|eval| {
                        izip!(&mut eval_buf[registry.offsets().datas()..], &step_buf)
                            .for_each(|(eval, step)| *eval += step as &_);
                        izip!(registry.offsets().calcs().., registry.calcs())
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
        _: usize,
        sum: &[E],
        transcript: &mut (impl TranscriptWrite<F, E> + ?Sized),
    ) -> Result<(), Error> {
        transcript.write_felt_ext(&sum[0])?;
        transcript.write_felt_exts(&sum[2..])?;
        Ok(())
    }

    fn read_sum(
        &self,
        _: usize,
        claim: E,
        transcript: &mut (impl TranscriptRead<F, E> + ?Sized),
    ) -> Result<Vec<E>, Error> {
        let mut sum = vec![E::ZERO; self.degree + 1];
        sum[0] = transcript.read_felt_ext()?;
        sum[2..].copy_from_slice(&transcript.read_felt_exts(self.degree - 1)?);
        sum[1] = claim - sum[0].double() - sum[2..].iter().sum::<E>();
        Ok(sum)
    }
}

impl<F: Field, E: ExtensionField<F>> Generic<F, E> {
    pub fn new(num_vars: usize, expression: &Expression<E, usize>) -> Self {
        let registry = ExpressionRegistry::new(expression);
        let degree = expression.degree();
        assert!(degree >= 2);
        let vander_mat_inv = vander_mat_inv(steps(F::ZERO).take(degree + 1).collect());
        Self {
            num_vars,
            expression: expression.clone(),
            registry,
            degree,
            vander_mat_inv,
        }
    }

    pub fn expression(&self) -> &Expression<E, usize> {
        &self.expression
    }
}
