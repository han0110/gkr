use crate::{
    poly::MultilinearPoly,
    sum_check::{eq_f::EqF, SumCheckFunction, SumCheckPoly},
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
    eq: Option<EqF<E>>,
}

impl<F: Field, E: ExtensionField<F>> SumCheckFunction<F, E> for Generic<F, E> {
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn degree(&self) -> usize {
        self.degree
    }

    fn evaluate(&self, evals: &[E]) -> E {
        self.expression.evaluate_felt(&|poly| evals[poly])
    }

    #[cfg(any(test, feature = "sanity-check"))]
    fn compute_sum(
        &self,
        round: usize,
        polys: &[SumCheckPoly<F, E, impl MultilinearPoly<F, E>, impl MultilinearPoly<E, E>>],
    ) -> E {
        use crate::op_sum_check_poly;

        let evaluate = |b| {
            self.expression.evaluate_felt(&|poly| {
                let poly = &polys[poly];
                op_sum_check_poly!(|poly| poly[b], |out| E::from(out))
            })
        };
        if let Some(eq) = self.eq() {
            let r_i = eq.r_i(round);
            let subset_i = eq.subset_i(round);
            (0..polys[0].len())
                .into_par_iter()
                .map(|b| {
                    let eval = evaluate(b);
                    let eq_eval = subset_i[b >> 1] * if b & 1 == 0 { E::ONE - r_i } else { r_i };
                    eq_eval * eval
                })
                .sum()
        } else {
            (0..polys[0].len()).into_par_iter().map(evaluate).sum()
        }
    }

    fn compute_round_poly(
        &self,
        round: usize,
        claim: E,
        polys: &[SumCheckPoly<F, E, impl MultilinearPoly<F, E>, impl MultilinearPoly<E, E>>],
    ) -> Vec<E> {
        let registry = &self.registry;
        assert_eq!(polys.len(), registry.datas().len());

        #[cfg(feature = "sanity-check")]
        assert_eq!(self.compute_sum(round, polys), claim);

        let buf = (registry.buffer(), vec![E::ZERO; registry.datas().len()]);
        let AdditiveVec(evals) = (0..polys[0].len() >> 1)
            .into_par_iter()
            .fold_with(
                (buf, AdditiveVec::new(self.degree)),
                |((mut eval_buf, mut step_buf), mut evals), b| {
                    let eq_eval = self.eq().map(|eq| eq.subset_i(round)[b]);
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
                    if let Some(eq_eval) = eq_eval {
                        evals[0] += eq_eval * eval_buf.last().unwrap()
                    } else {
                        evals[0] += eval_buf.last().unwrap()
                    };

                    evals[1..].iter_mut().for_each(|eval| {
                        izip!(&mut eval_buf[registry.offsets().datas()..], &step_buf)
                            .for_each(|(eval, step)| *eval += step as &_);
                        izip!(registry.offsets().calcs().., registry.calcs())
                            .for_each(|(idx, calc)| calc.calculate(&mut eval_buf, idx));
                        if let Some(eq_eval) = eq_eval {
                            *eval += eq_eval * eval_buf.last().unwrap()
                        } else {
                            *eval += eval_buf.last().unwrap()
                        };
                    });

                    ((eval_buf, step_buf), evals)
                },
            )
            .map(|(_, evals)| evals)
            .reduce_with(|acc, item| acc + item)
            .unwrap();

        let eval_0 = if let Some(eq) = self.eq() {
            eq.eval_0(round, claim, evals[0])
        } else {
            claim - evals[0]
        };
        let evals = chain![[eval_0], evals].collect_vec();
        self.vander_mat_inv
            .iter()
            .map(|row| inner_product(row, &evals))
            .collect()
    }

    fn write_round_poly(
        &self,
        _: usize,
        sum: &[E],
        transcript: &mut (impl TranscriptWrite<F, E> + ?Sized),
    ) -> Result<(), Error> {
        transcript.write_felt_ext(&sum[0])?;
        transcript.write_felt_exts(&sum[2..])?;
        Ok(())
    }

    fn read_round_poly(
        &self,
        round: usize,
        claim: E,
        transcript: &mut (impl TranscriptRead<F, E> + ?Sized),
    ) -> Result<Vec<E>, Error> {
        let mut sum = vec![E::ZERO; self.degree + 1];
        sum[0] = transcript.read_felt_ext()?;
        sum[2..].copy_from_slice(&transcript.read_felt_exts(self.degree - 1)?);
        let eval_1 = if let Some(eq) = self.eq() {
            eq.eval_1(round, claim, sum[0])
        } else {
            claim - sum[0]
        };
        sum[1] = eval_1 - sum[0] - sum[2..].iter().sum::<E>();
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
            eq: None,
        }
    }

    pub fn mul_by_eq(mut self, r_eq: &[E], is_proving: bool) -> Self {
        self.eq = Some(EqF::new(r_eq, is_proving));
        self
    }

    pub fn expression(&self) -> &Expression<E, usize> {
        &self.expression
    }

    pub fn eq(&self) -> Option<&EqF<E>> {
        self.eq.as_ref()
    }
}
