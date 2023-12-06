use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use gkr::{
    circuit::{node::EvalClaim, Circuit},
    poly::evaluate,
    prove_gkr,
    transcript::StdRngTranscript,
    util::{
        arithmetic::PrimeField,
        test::{rand_bytes, rand_vec, seeded_std_rng},
    },
};
use gkr_gadget::hash::keccak::dev::{keccak_circuit, keccak_circuit_inputs};
use halo2_curves::bn256::Fr;
use pprof::criterion::{Output, PProfProfiler};

fn run_gkr<F: PrimeField>(circuit: &Circuit<F>, values: &[Vec<F>], output_claims: &[EvalClaim<F>]) {
    let mut transcript = StdRngTranscript::<Vec<_>>::default();
    prove_gkr(circuit, values, output_claims, &mut transcript).unwrap();
}

fn keccak256(c: &mut Criterion) {
    let setup = |num_reps: usize| {
        let mut rng = seeded_std_rng();
        let circuit = keccak_circuit(256, num_reps);
        let values = {
            let input = rand_bytes(num_reps * 136 - 1, &mut rng);
            circuit.evaluate(keccak_circuit_inputs(256, num_reps, &input))
        };
        let output_claims = {
            let output = values.last().unwrap();
            let point = rand_vec(output.len().ilog2() as usize, &mut rng);
            let value = evaluate(output, &point);
            vec![EvalClaim::new(point, value)]
        };
        (circuit, values, output_claims)
    };

    let mut group = c.benchmark_group("keccak256");
    group.sample_size(10);
    for num_reps in (5..6).map(|log2| 1 << log2) {
        let id = BenchmarkId::from_parameter(num_reps);
        let (circuit, values, output_claims) = setup(num_reps);
        group.bench_with_input(id, &num_reps, |b, _| {
            b.iter(|| run_gkr::<Fr>(&circuit, &values, &output_claims));
        });
    }
}

criterion_group! {
    name = bench;
    config = Criterion::default().with_profiler(PProfProfiler::new(10, Output::Flamegraph(None)));
    targets = keccak256
}
criterion_main!(bench);
