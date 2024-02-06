use criterion::{
    criterion_group, criterion_main, measurement::Measurement, BenchmarkGroup, BenchmarkId,
    Criterion,
};
use gkr::{
    poly::{eq_poly, DenseMultilinearPoly, MultilinearPoly},
    sum_check::{eq_f::EqF, prove_sum_check, quadratic::Quadratic, SumCheckFunction},
    transcript::StdRngTranscript,
    util::{
        arithmetic::{Field, PrimeField},
        chain,
        dev::{rand_vec, seeded_std_rng},
    },
};
use halo2_curves::bn256::Fr;

fn run_sum_check<F: PrimeField, G: SumCheckFunction<F>>(
    group: &mut BenchmarkGroup<impl Measurement>,
    name: &str,
    setup: &impl Fn(&[F]) -> (G, Option<DenseMultilinearPoly<F, Vec<F>>>),
) {
    for num_vars in 16..20 {
        group.bench_with_input(BenchmarkId::new(name, num_vars), &num_vars, |b, _| {
            let mut rng = seeded_std_rng();
            let f = DenseMultilinearPoly::new(rand_vec(1 << num_vars, &mut rng));
            let r = rand_vec(num_vars, &mut rng);
            let claim = f.evaluate(&r);
            b.iter(|| {
                let (g, eq_poly) = setup(&r);
                let mut transcript = StdRngTranscript::<Vec<_>>::default();
                prove_sum_check(&g, claim, chain![[&f], &eq_poly], &mut transcript).unwrap();
            });
        });
    }
}

fn bench_sum_check(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum_check_eq_f");

    run_sum_check::<Fr, _>(&mut group, "quadratic", &|r| {
        let g = Quadratic::new(r.len());
        let eq_poly = DenseMultilinearPoly::new(eq_poly(r, Fr::ONE));
        (g, Some(eq_poly))
    });
    run_sum_check::<Fr, _>(&mut group, "eq_f", &|r| {
        let g = EqF::new(r, true);
        (g, None)
    });
}

criterion_group!(benches, bench_sum_check);
criterion_main!(benches);
