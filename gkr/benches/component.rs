use criterion::{
    black_box, criterion_group, criterion_main, measurement::Measurement, BenchmarkGroup,
    BenchmarkId, Criterion,
};
use gkr::{
    poly::{eq_poly, DenseMultilinearPoly, MultilinearPoly},
    sum_check::{quadratic::Quadratic, SumCheckFunction, SumCheckPoly},
    util::{
        arithmetic::Field,
        dev::{field_name, rand_vec, seeded_std_rng},
        RngCore,
    },
};
use goldilocks::GoldilocksExt2;
use halo2_curves::bn256;
use std::{array, ops::Range};

const RANGE: Range<usize> = 16..27;

fn bench_eq_poly(c: &mut Criterion) {
    let mut group = c.benchmark_group("eq_poly");

    let rng = seeded_std_rng();
    run::<GoldilocksExt2>(&mut group, rng.clone());
    run::<bn256::Fr>(&mut group, rng.clone());

    fn run<F: Field>(group: &mut BenchmarkGroup<impl Measurement>, mut rng: impl RngCore) {
        let r = std::iter::repeat_with(|| F::random(&mut rng))
            .take(RANGE.end)
            .collect::<Vec<_>>();

        for num_vars in RANGE {
            let r = &r[..num_vars];
            let id = BenchmarkId::new(field_name::<F>(), r.len());
            group.bench_with_input(id, &r, |b, r| {
                b.iter(|| eq_poly(black_box(r), black_box(F::ONE)))
            });
        }
    }
}

fn bench_fix_var(c: &mut Criterion) {
    let mut group = c.benchmark_group("fix_var");

    let rng = seeded_std_rng();
    run::<GoldilocksExt2>(&mut group, rng.clone());
    run::<bn256::Fr>(&mut group, rng.clone());

    fn run<F: Field>(group: &mut BenchmarkGroup<impl Measurement>, mut rng: impl RngCore) {
        let f = rand_vec::<F>(1 << RANGE.end, &mut rng);
        let r = F::random(&mut rng);

        for num_vars in RANGE {
            let f = DenseMultilinearPoly::new(&f[..1 << num_vars]);
            let id = BenchmarkId::new(field_name::<F>(), num_vars);
            group.bench_with_input(id, &r, |b, r| {
                b.iter(|| black_box(&f).fix_var(black_box(r)))
            });
        }
    }
}

fn bench_compute_sum_qudratic(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_sum_qudratic");

    let rng = seeded_std_rng();
    run::<GoldilocksExt2>(&mut group, rng.clone());
    run::<bn256::Fr>(&mut group, rng.clone());

    fn run<F: Field>(group: &mut BenchmarkGroup<impl Measurement>, mut rng: impl RngCore) {
        let [f, h] = array::from_fn(|_| rand_vec::<F>(1 << RANGE.end, &mut rng));

        for num_vars in RANGE {
            let [f, h] = [&f, &h].map(|poly| DenseMultilinearPoly::new(&poly[..1 << num_vars]));
            let polys = SumCheckPoly::bases([f, h]);
            let g = Quadratic::new(num_vars, vec![(None, 0, 1)]);
            let id = BenchmarkId::new(field_name::<F>(), num_vars);
            group.bench_function(id, |b| {
                b.iter(|| black_box(&g).compute_round_poly(0, F::ZERO, black_box(&polys)))
            });
        }
    }
}

criterion_group!(
    benches,
    bench_eq_poly,
    bench_fix_var,
    bench_compute_sum_qudratic
);
criterion_main!(benches);
