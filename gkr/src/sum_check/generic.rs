use crate::{
    poly::MultilinearPoly,
    sum_check::SumCheckFunction,
    transcript::{TranscriptRead, TranscriptWrite},
    util::{
        arithmetic::{inner_product, vander_mat_inv, Field, PrimeField},
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
pub struct Generic<F: Field> {
    num_vars: usize,
    expression: Expression<F, usize>,
    registry: ExpressionRegistry<F, usize>,
    degree: usize,
    vander_mat_inv: Vec<Vec<F>>,
}

impl<F: Field> SumCheckFunction<F> for Generic<F> {
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn degree(&self) -> usize {
        self.degree
    }

    fn compute_sum(
        &self,
        _: usize,
        claim: F,
        polys: &[&(impl MultilinearPoly<F> + ?Sized)],
    ) -> Vec<F> {
        let registry = &self.registry;
        assert_eq!(polys.len(), registry.datas().len());

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

        let buf = (registry.buffer(), vec![F::ZERO; registry.datas().len()]);
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
                    .for_each(|(eval, step, poly)| {
                        *eval = polys[*poly][(b << 1) + 1];
                        *step = *eval - polys[*poly][b << 1];
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
        sum: &[F],
        transcript: &mut (impl TranscriptWrite<F> + ?Sized),
    ) -> Result<(), Error> {
        transcript.write_felt(&sum[0])?;
        transcript.write_felts(&sum[2..])?;
        Ok(())
    }

    fn read_sum(
        &self,
        _: usize,
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
    pub fn new(num_vars: usize, expression: &Expression<F, usize>) -> Self {
        let registry = ExpressionRegistry::new(expression);
        let degree = expression.degree();
        assert!(degree >= 2);
        let vander_mat_inv = vander_mat_inv((0..).map(F::from).take(degree + 1).collect());
        Self {
            num_vars,
            expression: expression.clone(),
            registry,
            degree,
            vander_mat_inv,
        }
    }

    pub fn expression(&self) -> &Expression<F, usize> {
        &self.expression
    }
}
