use criterion::{
    criterion_group, criterion_main, measurement::Measurement, BenchmarkGroup, BenchmarkId,
    Criterion,
};
use gkr::{
    circuit::node::EvalClaim,
    poly::MultilinearPoly,
    prove_gkr,
    transcript::StdRngTranscript,
    util::{
        arithmetic::PrimeField,
        dev::{rand_bytes, rand_vec, seeded_std_rng},
    },
};
use gkr_gadget::hash::keccak::{dev::keccak_circuit, Keccak};
use halo2_curves::bn256;

fn run_keccak256<F: PrimeField>(field_name: &str, group: &mut BenchmarkGroup<impl Measurement>) {
    let setup = |num_reps: usize| {
        let mut rng = seeded_std_rng();
        let keccak = Keccak::new(256, num_reps);
        let input = rand_bytes(num_reps * keccak.rate() - 1, &mut rng);
        let (circuit, values) = keccak_circuit(keccak, &input);
        let output_claims = {
            let output = values.last().unwrap();
            let point = rand_vec(output.len().ilog2() as usize, &mut rng);
            let value = output.evaluate(&point);
            vec![EvalClaim::new(point, value)]
        };
        (circuit, values, output_claims)
    };

    for num_reps in (5..10).map(|log2| 1 << log2) {
        let id = BenchmarkId::new(field_name, num_reps);
        let (circuit, values, output_claims) = setup(num_reps);
        group.bench_with_input(id, &num_reps, |b, _| {
            b.iter(|| {
                let mut transcript = StdRngTranscript::<Vec<_>>::default();
                prove_gkr::<F>(&circuit, &values, &output_claims, &mut transcript).unwrap();
            });
        });
    }
}

fn bench_keccak256(c: &mut Criterion) {
    let mut group = c.benchmark_group("keccak256");
    group.sample_size(10);

    run_keccak256::<bn256::Fr>("bn254", &mut group);
    run_keccak256::<goldilocks::GoldilocksExt2>("goldilocks_qe", &mut group);
}

criterion_group!(bench, bench_keccak256);
criterion_main!(bench);
