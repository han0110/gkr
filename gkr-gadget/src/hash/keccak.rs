use gkr::{
    circuit::{
        connect,
        node::{InputNode, LogUpNode, Node, VanillaGate, VanillaNode},
        Circuit, NodeId,
    },
    util::{
        arithmetic::{powers, ExtensionField, Field},
        chain, izip, Itertools,
    },
};
use std::iter;

#[derive(Clone, Copy, Debug)]
pub struct Keccak {
    num_bits: usize,
    rate: usize,
    perm: KeccakPerm,
}

impl Keccak {
    pub const STATE_SIZE: usize = KeccakPerm::STATE_SIZE;

    pub const LOG2_STATE_SIZE: usize = KeccakPerm::LOG2_STATE_SIZE;

    pub const fn v256(num_reps: usize) -> Self {
        Self::new(256, num_reps)
    }

    pub const fn new(num_bits: usize, num_reps: usize) -> Self {
        Self {
            num_bits,
            rate: 200 - num_bits / 4,
            perm: KeccakPerm::new(num_reps),
        }
    }

    pub const fn num_bits(&self) -> usize {
        self.num_bits
    }

    pub const fn rate(&self) -> usize {
        self.rate
    }

    pub const fn num_reps(&self) -> usize {
        self.perm.num_reps()
    }

    pub const fn log2_reps(&self) -> usize {
        self.perm.log2_reps()
    }

    pub const fn log2_size(&self) -> usize {
        self.perm.log2_size()
    }

    pub fn alloc_state<F: Field, E: ExtensionField<F>>(
        &self,
        circuit: &mut Circuit<F, E>,
    ) -> NodeId {
        circuit.insert(InputNode::new(Self::LOG2_STATE_SIZE, self.perm.num_reps))
    }
    pub fn alloc_input<F: Field, E: ExtensionField<F>>(
        &self,
        circuit: &mut Circuit<F, E>,
    ) -> NodeId {
        circuit.insert(InputNode::new(Self::LOG2_STATE_SIZE, self.perm.num_reps))
    }

    pub fn configure<F: Field, E: ExtensionField<F>>(
        &self,
        circuit: &mut Circuit<F, E>,
        state: NodeId,
        input: NodeId,
    ) -> NodeId {
        assert_eq!(circuit.node(state).log2_input_size(), self.log2_size());
        assert_eq!(circuit.node(input).log2_input_size(), self.log2_size());

        let state_prime = {
            let gates = chain![
                (0..self.rate / 8)
                    .flat_map(|idx| xor_gates(WordIdx::new(idx), WordIdx::new(idx).input(1))),
                (self.rate / 8..25).flat_map(|idx| relay_gates(WordIdx::new(idx)))
            ]
            .collect();
            VanillaNode::new(2, Self::LOG2_STATE_SIZE, gates, self.perm.num_reps)
        };

        let state_prime = circuit.insert(state_prime);

        connect!(circuit { state_prime <- state, input });

        self.perm.configure(circuit, state_prime)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct KeccakPerm {
    num_reps: usize,
    log2_reps: usize,
}

impl KeccakPerm {
    pub const STATE_SIZE: usize = 25 * 64;

    pub const LOG2_STATE_SIZE: usize = Self::STATE_SIZE.next_power_of_two().ilog2() as usize;

    #[rustfmt::skip]
    const RC: [u64; 24] = [
        0x0000000000000001, 0x0000000000008082, 0x800000000000808a, 0x8000000080008000,
        0x000000000000808b, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
        0x000000000000008a, 0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
        0x000000008000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003,
        0x8000000000008002, 0x8000000000000080, 0x000000000000800a, 0x800000008000000a,
        0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
    ];

    pub const fn new(num_reps: usize) -> Self {
        Self {
            num_reps,
            log2_reps: num_reps.next_power_of_two().ilog2() as usize,
        }
    }

    pub const fn num_reps(&self) -> usize {
        self.num_reps
    }

    pub const fn log2_reps(&self) -> usize {
        self.log2_reps
    }

    pub const fn log2_size(&self) -> usize {
        Self::LOG2_STATE_SIZE + self.log2_reps
    }

    pub fn configure<F: Field, E: ExtensionField<F>>(
        &self,
        circuit: &mut Circuit<F, E>,
        state: NodeId,
    ) -> NodeId {
        assert_eq!(circuit.node(state).log2_input_size(), self.log2_size());

        let mut theta_lookup_io = vec![];
        let state = Self::RC.into_iter().fold(state, |state, rc| {
            let state_prime = circuit.insert(InputNode::new(Self::LOG2_STATE_SIZE, self.num_reps));
            theta_lookup_io.push((state, state_prime));
            self.configure_rho_pi_chi_iota(circuit, state_prime, rc)
        });
        self.configure_theta(circuit, theta_lookup_io);
        state
    }

    fn configure_rho_pi_chi_iota<F: Field, E: ExtensionField<F>>(
        &self,
        circuit: &mut Circuit<F, E>,
        state: NodeId,
        rc: u64,
    ) -> NodeId {
        #[rustfmt::skip]
        let words = [
            (0, 0), (6, 44), (12, 43), (18, 21), (24, 14),
            (3, 28), (9, 20), (10, 3), (16, 45), (22, 61),
            (1, 1), (7, 6), (13, 25), (19, 8), (20, 18),
            (4, 27), (5, 36), (11, 10), (17, 15), (23, 56),
            (2, 62), (8, 55), (14, 39), (15, 41), (21, 2),
        ]
        .map(|(idx, rotate_left)| WordIdx::new(idx).rotate_left(rotate_left));
        let n_0 = {
            #[rustfmt::skip]
            let gates = [
                (1, 2), (2, 3), (3, 4), (4, 0), (0, 1),
                (6, 7), (7, 8), (8, 9), (9, 5), (5, 6),
                (11, 12), (12, 13), (13, 14), (14, 10), (10, 11),
                (16, 17), (17, 18), (18, 19), (19, 15), (15, 16),
                (21, 22), (22, 23), (23, 24), (24, 20), (20, 21),
            ]
            .into_iter()
            .flat_map(|(lhs, rhs)| not_lhs_and_rhs_gates(words[lhs], words[rhs]))
            .collect();
            VanillaNode::new(1, Self::LOG2_STATE_SIZE, gates, self.num_reps)
        };
        let n_1 = {
            let gates = chain![
                xor_with_rc_gates(words[0], WordIdx::new(0).input(1), rc),
                (1..25).flat_map(|idx| xor_gates(words[idx], WordIdx::new(idx).input(1)))
            ]
            .collect();
            VanillaNode::new(2, Self::LOG2_STATE_SIZE, gates, self.num_reps)
        };

        let [n_0, n_1] = [n_0, n_1].map(|node| circuit.insert(node));

        connect!(circuit {
            n_0 <- state;
            n_1 <- state, n_0;
        });

        n_1
    }

    fn configure_theta<F: Field, E: ExtensionField<F>>(
        &self,
        circuit: &mut Circuit<F, E>,
        theta_lookup_io: Vec<(NodeId, NodeId)>,
    ) {
        let f = {
            let gates = [
                ([0, 4, 9, 14, 19, 24], [1, 6, 11, 16, 21]),
                ([1, 0, 5, 10, 15, 20], [2, 7, 12, 17, 22]),
                ([2, 1, 6, 11, 16, 21], [3, 8, 13, 18, 23]),
                ([3, 2, 7, 12, 17, 22], [4, 9, 14, 19, 24]),
                ([4, 3, 8, 13, 18, 23], [0, 5, 10, 15, 20]),
                ([5, 4, 9, 14, 19, 24], [1, 6, 11, 16, 21]),
                ([6, 0, 5, 10, 15, 20], [2, 7, 12, 17, 22]),
                ([7, 1, 6, 11, 16, 21], [3, 8, 13, 18, 23]),
                ([8, 2, 7, 12, 17, 22], [4, 9, 14, 19, 24]),
                ([9, 3, 8, 13, 18, 23], [0, 5, 10, 15, 20]),
                ([10, 4, 9, 14, 19, 24], [1, 6, 11, 16, 21]),
                ([11, 0, 5, 10, 15, 20], [2, 7, 12, 17, 22]),
                ([12, 1, 6, 11, 16, 21], [3, 8, 13, 18, 23]),
                ([13, 2, 7, 12, 17, 22], [4, 9, 14, 19, 24]),
                ([14, 3, 8, 13, 18, 23], [0, 5, 10, 15, 20]),
                ([15, 4, 9, 14, 19, 24], [1, 6, 11, 16, 21]),
                ([16, 0, 5, 10, 15, 20], [2, 7, 12, 17, 22]),
                ([17, 1, 6, 11, 16, 21], [3, 8, 13, 18, 23]),
                ([18, 2, 7, 12, 17, 22], [4, 9, 14, 19, 24]),
                ([19, 3, 8, 13, 18, 23], [0, 5, 10, 15, 20]),
                ([20, 4, 9, 14, 19, 24], [1, 6, 11, 16, 21]),
                ([21, 0, 5, 10, 15, 20], [2, 7, 12, 17, 22]),
                ([22, 1, 6, 11, 16, 21], [3, 8, 13, 18, 23]),
                ([23, 2, 7, 12, 17, 22], [4, 9, 14, 19, 24]),
                ([24, 3, 8, 13, 18, 23], [0, 5, 10, 15, 20]),
            ]
            .into_iter()
            .enumerate()
            .flat_map(|(idx, (inputs, rotated_inputs))| {
                let mut inputs = inputs.map(|idx| WordIdx::new(idx).w_iter());
                let mut rotated_inputs =
                    rotated_inputs.map(|idx| WordIdx::new(idx).rotate_left(1).w_iter());
                let mut out = WordIdx::new(idx).input(1).w_iter();
                iter::repeat_with(move || {
                    let d_1 = izip!(
                        powers(F::ONE.double()),
                        chain![&mut inputs, &mut rotated_inputs, [&mut out]]
                    )
                    .map(|(scalar, wire)| (Some(scalar), wire.next().unwrap()))
                    .collect_vec();
                    VanillaGate::new(None, d_1, vec![])
                })
                .take(64)
            })
            .collect();
            VanillaNode::new(2, Self::LOG2_STATE_SIZE, gates, self.num_reps)
        };
        let fs = iter::repeat_with(|| circuit.insert(f.clone()))
            .take(theta_lookup_io.len())
            .collect_vec();
        for ((state, state_prime), f) in izip!(theta_lookup_io, fs.iter().copied()) {
            connect!(circuit { f <- state, state_prime });
        }

        let m = circuit.insert(InputNode::new(12, 1));
        let t = circuit.insert(InputNode::new(12, 1));
        let logup = circuit.insert(LogUpNode::new(12, self.log2_size(), fs.len()));
        for input in chain![[m, t], fs] {
            connect!(circuit { logup <- input });
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct WordIdx {
    idx: usize,
    rotate_left: usize,
    input: usize,
}

impl WordIdx {
    fn new(idx: usize) -> Self {
        Self {
            idx,
            ..Default::default()
        }
    }

    fn rotate_left(mut self, rotate_left: usize) -> Self {
        self.rotate_left = rotate_left;
        self
    }

    fn input(mut self, input: usize) -> Self {
        self.input = input;
        self
    }

    fn w_iter(self) -> impl Iterator<Item = (usize, usize)> {
        let bs = (self.idx * 64..(self.idx + 1) * 64)
            .cycle()
            .skip(64 - self.rotate_left);
        izip!(iter::repeat(self.input), bs).take(64)
    }
}

fn relay_gates<F: Field>(idx: WordIdx) -> impl Iterator<Item = VanillaGate<F>> {
    izip!(idx.w_iter()).map(move |w| VanillaGate::relay(w))
}

fn xor_gates<F: Field>(lhs: WordIdx, rhs: WordIdx) -> impl Iterator<Item = VanillaGate<F>> {
    izip!(lhs.w_iter(), rhs.w_iter()).map(move |(w_0, w_1)| VanillaGate::xor(w_0, w_1))
}

fn xor_with_rc_gates<F: Field>(
    lhs: WordIdx,
    rhs: WordIdx,
    rc: u64,
) -> impl Iterator<Item = VanillaGate<F>> {
    let bits = (0..64).map(move |idx| (rc >> idx) & 1 == 1);
    izip!(lhs.w_iter(), rhs.w_iter(), bits).map(move |(w_0, w_1, bit)| {
        if bit {
            VanillaGate::xnor(w_0, w_1)
        } else {
            VanillaGate::xor(w_0, w_1)
        }
    })
}

fn not_lhs_and_rhs_gates<F: Field>(
    lhs: WordIdx,
    rhs: WordIdx,
) -> impl Iterator<Item = VanillaGate<F>> {
    izip!(lhs.w_iter(), rhs.w_iter()).map(move |(w_0, w_1)| {
        VanillaGate::new(None, vec![(None, w_1)], vec![(Some(-F::ONE), w_0, w_1)])
    })
}

#[cfg(any(test, feature = "dev"))]
pub mod dev {
    use crate::hash::keccak::{Keccak, KeccakPerm};
    use gkr::{
        circuit::Circuit,
        poly::{box_dense_poly, BinaryMultilinearPoly, BoxMultilinearPoly, MultilinearPolyExt},
        util::{
            arithmetic::{ExtensionField, Field},
            chain, chain_par, izip, Itertools,
        },
    };
    use rayon::prelude::*;
    use std::{
        array::from_fn,
        iter, mem,
        sync::atomic::{AtomicU64, Ordering::Relaxed},
    };

    pub fn keccak_circuit<F: Field + From<u64>, E: ExtensionField<F>>(
        keccak: Keccak,
        input: &[u8],
    ) -> (Circuit<F, E>, Vec<BoxMultilinearPoly<'static, F, E>>) {
        let circuit = {
            let mut circuit = Circuit::default();
            let state = keccak.alloc_state(&mut circuit);
            let input = keccak.alloc_input(&mut circuit);
            keccak.configure(&mut circuit, state, input);
            circuit
        };
        let values = keccak_circuit_values(keccak, input);
        (circuit, values)
    }

    fn keccak_circuit_values<F: Field + From<u64>, E: ExtensionField<F>>(
        keccak: Keccak,
        input: &[u8],
    ) -> Vec<BoxMultilinearPoly<'static, F, E>> {
        let log2_size = keccak.log2_size();
        let rate = keccak.rate();

        let inputs = chain![
            chain![input.chunks(rate), [[].as_slice()]]
                .take(input.len() / rate + 1)
                .map(|chunk| {
                    let mut chunk = chunk.to_vec();
                    if chunk.len() != rate {
                        let offset = chunk.len();
                        chunk.resize(rate, 0);
                        chunk[offset] ^= 0x01;
                        chunk[rate - 1] ^= 0x80;
                    }
                    chain![
                        chunk.chunks_exact(8).map(u64_from_le_bytes),
                        iter::repeat(0)
                    ]
                    .take(25)
                    .collect_vec()
                    .try_into()
                    .unwrap()
                }),
            iter::repeat([0; 25])
        ]
        .take(1 << keccak.log2_reps())
        .collect::<Vec<_>>();

        let states = inputs
            .iter()
            .scan([0; 25], |state, input| {
                let mut next_state = from_fn(|idx| state[idx] ^ input.get(idx).unwrap_or(&0));
                tiny_keccak::keccakf(&mut next_state);
                mem::replace(state, next_state).into()
            })
            .collect_vec();

        let state_primes = izip!(&states, &inputs)
            .map(|(state, input)| from_fn(|idx| state[idx] ^ input.get(idx).unwrap_or(&0)))
            .collect_vec();

        let m = [(); 1 << 12].map(|_| AtomicU64::new(0));
        m[0].store(24 << log2_size, Relaxed);
        let (interms, fs) = state_primes
            .par_iter()
            .map(|state| {
                let (interms, fs) = izip!(KeccakPerm::RC)
                    .scan(*state, |state, rc| {
                        let (state_prime, f) = theta(state, &m);
                        let (n_0, n_1) = rho_pi_chi_iota(&state_prime, rc);
                        *state = n_1;
                        Some(([state_prime, n_0, n_1], f))
                    })
                    .unzip::<_, _, Vec<_>, Vec<_>>();
                (interms.into_iter().flatten().collect_vec(), fs)
            })
            .unzip::<_, _, Vec<_>, Vec<_>>();

        let bins = chain_par![
            [states, inputs, state_primes].map(|values| {
                values
                    .into_iter()
                    .flat_map(|value| chain![value, [0; 7]])
                    .collect()
            }),
            (0..72).into_par_iter().map(|idx| {
                interms
                    .iter()
                    .flat_map(|interms| chain![interms[idx], [0; 7]])
                    .collect::<Vec<_>>()
            })
        ]
        .map(|value| BinaryMultilinearPoly::new(value, log2_size).boxed());
        let fs = (0..24)
            .into_par_iter()
            .map(|idx| {
                fs.iter()
                    .flat_map(|fs| {
                        chain![fs[idx].iter().copied().map(F::from), iter::repeat(F::ZERO)]
                            .take(1 << Keccak::LOG2_STATE_SIZE)
                    })
                    .collect_vec()
            })
            .map(box_dense_poly);
        let m = m
            .into_par_iter()
            .map(|count| F::from(count.into_inner()))
            .collect();
        let t = (0..1u64 << 12)
            .into_par_iter()
            .map(|idx| (((idx & ((1 << 11) - 1)).count_ones() & 1) << 11) as u64 + idx)
            .map(F::from)
            .collect::<Vec<_>>();
        chain_par![bins, fs, [m, t, vec![F::ZERO]].map(box_dense_poly)].collect()
    }

    fn u64_from_le_bytes(bytes: &[u8]) -> u64 {
        let mut word = [0; 8];
        word.copy_from_slice(bytes);
        u64::from_le_bytes(word)
    }

    fn theta(state: &[u64; 25], m: &[AtomicU64; 1 << 12]) -> ([u64; 25], [u64; 1600]) {
        let mut num_nz = 0;
        let mut f = [0; 1600];
        let state_prime = [
            ([0, 4, 9, 14, 19, 24], [1, 6, 11, 16, 21]),
            ([1, 0, 5, 10, 15, 20], [2, 7, 12, 17, 22]),
            ([2, 1, 6, 11, 16, 21], [3, 8, 13, 18, 23]),
            ([3, 2, 7, 12, 17, 22], [4, 9, 14, 19, 24]),
            ([4, 3, 8, 13, 18, 23], [0, 5, 10, 15, 20]),
            ([5, 4, 9, 14, 19, 24], [1, 6, 11, 16, 21]),
            ([6, 0, 5, 10, 15, 20], [2, 7, 12, 17, 22]),
            ([7, 1, 6, 11, 16, 21], [3, 8, 13, 18, 23]),
            ([8, 2, 7, 12, 17, 22], [4, 9, 14, 19, 24]),
            ([9, 3, 8, 13, 18, 23], [0, 5, 10, 15, 20]),
            ([10, 4, 9, 14, 19, 24], [1, 6, 11, 16, 21]),
            ([11, 0, 5, 10, 15, 20], [2, 7, 12, 17, 22]),
            ([12, 1, 6, 11, 16, 21], [3, 8, 13, 18, 23]),
            ([13, 2, 7, 12, 17, 22], [4, 9, 14, 19, 24]),
            ([14, 3, 8, 13, 18, 23], [0, 5, 10, 15, 20]),
            ([15, 4, 9, 14, 19, 24], [1, 6, 11, 16, 21]),
            ([16, 0, 5, 10, 15, 20], [2, 7, 12, 17, 22]),
            ([17, 1, 6, 11, 16, 21], [3, 8, 13, 18, 23]),
            ([18, 2, 7, 12, 17, 22], [4, 9, 14, 19, 24]),
            ([19, 3, 8, 13, 18, 23], [0, 5, 10, 15, 20]),
            ([20, 4, 9, 14, 19, 24], [1, 6, 11, 16, 21]),
            ([21, 0, 5, 10, 15, 20], [2, 7, 12, 17, 22]),
            ([22, 1, 6, 11, 16, 21], [3, 8, 13, 18, 23]),
            ([23, 2, 7, 12, 17, 22], [4, 9, 14, 19, 24]),
            ([24, 3, 8, 13, 18, 23], [0, 5, 10, 15, 20]),
        ]
        .into_iter()
        .enumerate()
        .map(|(word_idx, (inputs, rotated_inputs))| {
            let inputs = inputs.map(|idx| state[idx]);
            let rotated_inputs = rotated_inputs.map(|idx| state[idx].rotate_left(1));
            let out = {
                let mut out = 0;
                inputs.map(|input| out ^= input);
                rotated_inputs.map(|input| out ^= input);
                out
            };
            for bit_idx in 0..64 {
                let value = chain![&inputs, &rotated_inputs, [&out]]
                    .rev()
                    .map(|word| (word >> bit_idx) & 1)
                    .reduce(|acc, bit| (acc << 1) + bit)
                    .unwrap();
                f[word_idx * 64 + bit_idx] = value;
                let idx = value & ((1 << 11) - 1);
                if idx != 0 {
                    num_nz += 1;
                    m[idx as usize].fetch_add(1, Relaxed);
                }
            }
            out
        })
        .collect_vec();
        m[0].fetch_sub(num_nz, Relaxed);
        (state_prime.try_into().unwrap(), f)
    }

    fn rho_pi_chi_iota(state: &[u64; 25], rc: u64) -> ([u64; 25], [u64; 25]) {
        #[rustfmt::skip]
        let state = [
            (0, 0), (6, 44), (12, 43), (18, 21), (24, 14),
            (3, 28), (9, 20), (10, 3), (16, 45), (22, 61),
            (1, 1), (7, 6), (13, 25), (19, 8), (20, 18),
            (4, 27), (5, 36), (11, 10), (17, 15), (23, 56),
            (2, 62), (8, 55), (14, 39), (15, 41), (21, 2),
        ]
        .map(|(idx, rotate_left)| state[idx].rotate_left(rotate_left));

        #[rustfmt::skip]
        let n_0 = [
            (1, 2), (2, 3), (3, 4), (4, 0), (0, 1),
            (6, 7), (7, 8), (8, 9), (9, 5), (5, 6),
            (11, 12), (12, 13), (13, 14), (14, 10), (10, 11),
            (16, 17), (17, 18), (18, 19), (19, 15), (15, 16),
            (21, 22), (22, 23), (23, 24), (24, 20), (20, 21),
        ]
        .map(|(lhs, rhs)| ((!state[lhs]) & (state[rhs])));

        let n_1 = from_fn(|idx| state[idx] ^ n_0[idx] ^ (if idx == 0 { rc } else { 0 }));

        (n_0, n_1)
    }
}

#[cfg(test)]
pub mod test {
    use crate::hash::keccak::{dev::keccak_circuit, Keccak};
    use gkr::{
        dev::run_gkr_with_values,
        poly::MultilinearPoly,
        util::{
            arithmetic::{try_felt_to_bool, ExtensionField, Field, PrimeField},
            dev::{rand_bytes, rand_range, seeded_std_rng},
            RngCore,
        },
    };
    use goldilocks::{Goldilocks, GoldilocksExt2};

    #[test]
    fn keccak256() {
        let mut rng = seeded_std_rng();
        for num_reps in (0..3).map(|log2| 1 << log2) {
            run_keccak::<Goldilocks, GoldilocksExt2>(256, num_reps, &mut rng);
        }
    }

    fn run_keccak<F: PrimeField, E: ExtensionField<F>>(
        num_bits: usize,
        num_reps: usize,
        mut rng: impl RngCore,
    ) {
        let keccak = Keccak::new(num_bits, num_reps);
        let input = rand_bytes(rand_range(0..num_reps * keccak.rate(), &mut rng), &mut rng);

        let (circuit, values) = keccak_circuit::<F, E>(keccak, &input);
        run_gkr_with_values(&circuit, &values, &mut rng);

        let offset = (input.len() / keccak.rate()) << Keccak::LOG2_STATE_SIZE;
        let output = extract_bytes(num_bits, offset, values.into_iter().nth(74).unwrap());
        assert_eq!(output, native_keccak(num_bits, &input));
    }

    fn native_keccak(num_bits: usize, input: &[u8]) -> Vec<u8> {
        use tiny_keccak::{Hasher, Keccak};
        let mut keccak = match num_bits {
            256 => Keccak::v256(),
            _ => unreachable!(),
        };
        let mut output = vec![0; 32];
        keccak.update(input);
        keccak.finalize(&mut output);
        output
    }

    fn extract_bytes<F: Field, E: ExtensionField<F>>(
        num_bits: usize,
        offset: usize,
        output: impl MultilinearPoly<F, E>,
    ) -> Vec<u8> {
        Vec::from_iter((offset..).take(num_bits).step_by(8).map(|offset| {
            (offset..offset + 8)
                .rev()
                .map(|b| try_felt_to_bool(output[b]).unwrap())
                .fold(0, |acc, bit| (acc << 1) + bit as u8)
        }))
    }
}
