use criterion::{
    criterion_group, criterion_main, measurement::Measurement, BenchmarkGroup, BenchmarkId,
    Criterion,
};
use gkr::{
    poly::{box_dense_poly, eq_poly, BoxMultilinearPoly, MultilinearPoly},
    sum_check::{eq_f::EqF, prove_sum_check, quadratic::Quadratic, SumCheckFunction, SumCheckPoly},
    transcript::StdRngTranscript,
    util::{
        arithmetic::{ExtensionField, Field, PrimeField},
        chain,
        dev::{rand_vec, seeded_std_rng},
    },
};
use goldilocks::{Goldilocks, GoldilocksExt2};

fn run_sum_check<F: PrimeField, E: ExtensionField<F>, G: SumCheckFunction<F, E>>(
    group: &mut BenchmarkGroup<impl Measurement>,
    name: &str,
    setup: &impl Fn(&[E]) -> (G, Option<BoxMultilinearPoly<'static, E>>),
) {
    for num_vars in 16..20 {
        group.bench_with_input(BenchmarkId::new(name, num_vars), &num_vars, |b, _| {
            let mut rng = seeded_std_rng();
            let f = box_dense_poly(rand_vec(1 << num_vars, &mut rng));
            let r = rand_vec(num_vars, &mut rng);
            let claim = f.evaluate(&r);
            b.iter(|| {
                let (g, eq_poly) = setup(&r);
                let polys = chain![
                    [SumCheckPoly::Base(&f)],
                    eq_poly.as_ref().map(SumCheckPoly::Extension)
                ];
                let mut transcript = StdRngTranscript::default();
                prove_sum_check(&g, claim, polys, &mut transcript).unwrap();
            });
        });
    }
}

fn bench_sum_check(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum_check_eq_f");

    run_sum_check::<Goldilocks, GoldilocksExt2, _>(&mut group, "quadratic", &|r| {
        let g = Quadratic::new(r.len(), vec![(None, 0, 1)]);
        let eq_poly = box_dense_poly(eq_poly(r, GoldilocksExt2::ONE));
        (g, Some(eq_poly))
    });
    run_sum_check::<Goldilocks, GoldilocksExt2, _>(&mut group, "eq_f", &|r| {
        let g = EqF::new(r, true);
        (g, None)
    });
}

criterion_group!(benches, bench_sum_check);
criterion_main!(benches);
