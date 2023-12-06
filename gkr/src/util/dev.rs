use halo2_curves::ff::Field;
use itertools::Itertools;
use rand::{
    distributions::uniform::SampleRange,
    rngs::{OsRng, StdRng},
    CryptoRng, Rng, RngCore, SeedableRng,
};
use std::{array, hash::Hash, iter};

pub fn std_rng() -> impl RngCore + CryptoRng {
    StdRng::from_seed(Default::default())
}

pub fn seeded_std_rng() -> impl RngCore + CryptoRng {
    StdRng::seed_from_u64(OsRng.next_u64())
}

pub fn rand_range(range: impl SampleRange<usize>, mut rng: impl RngCore) -> usize {
    rng.gen_range(range)
}

pub fn rand_bool(mut rng: impl RngCore) -> bool {
    rng.gen_bool(0.5)
}

pub fn rand_bytes(n: usize, mut rng: impl RngCore) -> Vec<u8> {
    iter::repeat_with(|| rng.next_u64().to_le_bytes())
        .flatten()
        .take(n)
        .collect()
}

pub fn rand_array<F: Field, const N: usize>(mut rng: impl RngCore) -> [F; N] {
    array::from_fn(|_| F::random(&mut rng))
}

pub fn rand_vec<F: Field>(n: usize, mut rng: impl RngCore) -> Vec<F> {
    iter::repeat_with(|| F::random(&mut rng)).take(n).collect()
}

pub fn rand_unique<T, R>(n: usize, f: impl Fn(&mut R) -> T, mut rng: R) -> Vec<T>
where
    T: Clone + Eq + Hash,
    R: RngCore,
{
    iter::repeat_with(|| f(&mut rng)).unique().take(n).collect()
}
