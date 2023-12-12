use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use gkr::{
    circuit::{node::EvalClaim, Circuit},
    poly::{BoxMultilinearPoly, MultilinearPoly},
    prove_gkr,
    transcript::StdRngTranscript,
    util::{
        arithmetic::PrimeField,
        dev::{rand_bytes, rand_vec, seeded_std_rng},
    },
};
use gkr_gadget::hash::keccak::{dev::keccak_circuit, Keccak};
use halo2_curves::bn256::Fr;

fn run_gkr<F: PrimeField>(
    circuit: &Circuit<F>,
    values: &[BoxMultilinearPoly<F>],
    output_claims: &[EvalClaim<F>],
) {
    let mut transcript = StdRngTranscript::<Vec<_>>::default();
    prove_gkr(circuit, values, output_claims, &mut transcript).unwrap();
}

fn keccak256(c: &mut Criterion) {
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

    let mut group = c.benchmark_group("keccak256");
    group.sample_size(10);
    for num_reps in (5..10).map(|log2| 1 << log2) {
        let id = BenchmarkId::from_parameter(num_reps);
        let (circuit, values, output_claims) = setup(num_reps);
        group.bench_with_input(id, &num_reps, |b, _| {
            b.iter(|| run_gkr::<Fr>(&circuit, &values, &output_claims));
        });
    }
}

criterion_group!(bench, keccak256);
criterion_main!(bench);
