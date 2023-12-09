use gkr::{
    circuit::{
        connect,
        node::{InputNode, Node, VanillaGate, VanillaNode},
        Circuit, NodeId,
    },
    util::{arithmetic::Field, chain, izip},
};
use std::iter;

pub struct Keccak {
    rate: usize,
    perm: KeccakPerm,
}

impl Keccak {
    pub const STATE_SIZE: usize = KeccakPerm::STATE_SIZE;

    pub const LOG2_STATE_SIZE: usize = KeccakPerm::LOG2_STATE_SIZE;

    pub const fn v256(num_reps: usize) -> Self {
        Self::new(256, num_reps)
    }

    const fn new(num_bits: usize, num_reps: usize) -> Self {
        Self {
            rate: Self::rate(num_bits),
            perm: KeccakPerm::new(num_reps),
        }
    }

    const fn rate(num_bits: usize) -> usize {
        200 - num_bits / 4
    }

    pub const fn log2_size(&self) -> usize {
        self.perm.log2_size()
    }

    pub fn alloc_state<F: Field>(&self, circuit: &mut Circuit<F>) -> NodeId {
        circuit.insert(InputNode::new(Self::LOG2_STATE_SIZE, self.perm.num_reps))
    }
    pub fn alloc_input<F: Field>(&self, circuit: &mut Circuit<F>) -> NodeId {
        circuit.insert(InputNode::new(Self::LOG2_STATE_SIZE, self.perm.num_reps))
    }

    pub fn configure<F: Field>(
        &self,
        circuit: &mut Circuit<F>,
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

    pub const fn log2_size(&self) -> usize {
        Self::LOG2_STATE_SIZE + self.log2_reps
    }

    pub fn configure<F: Field>(&self, circuit: &mut Circuit<F>, state: NodeId) -> NodeId {
        assert_eq!(circuit.node(state).log2_input_size(), self.log2_size());

        Self::RC.into_iter().fold(state, |state, rc| {
            let state = self.configure_theta(circuit, state);
            self.configure_rho_pi_chi_iota(circuit, state, rc)
        })
    }

    fn configure_theta<F: Field>(&self, circuit: &mut Circuit<F>, state: NodeId) -> NodeId {
        let n_1 = {
            let gates = [
                (0, 5),
                (10, 15),
                (1, 6),
                (11, 16),
                (2, 7),
                (12, 17),
                (3, 8),
                (13, 18),
                (4, 9),
                (14, 19),
            ]
            .into_iter()
            .flat_map(|(lhs, rhs)| xor_gates(WordIdx::new(lhs), WordIdx::new(rhs)))
            .collect();
            VanillaNode::new(1, Self::LOG2_STATE_SIZE, gates, self.num_reps)
        };
        let n_2 = {
            let gates = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
                .into_iter()
                .flat_map(|(lhs, rhs)| xor_gates(WordIdx::new(lhs), WordIdx::new(rhs)))
                .collect();
            VanillaNode::new(1, n_1.log2_sub_output_size(), gates, self.num_reps)
        };
        let n_3 = {
            let gates = [(20, 22), (21, 23), (22, 24), (23, 20), (24, 21)]
                .into_iter()
                .flat_map(|(lhs, rhs)| {
                    xor_gates(WordIdx::new(lhs), WordIdx::new(rhs).rotate_left(1))
                })
                .collect();
            VanillaNode::new(1, Self::LOG2_STATE_SIZE, gates, self.num_reps)
        };
        let n_4 = {
            let gates = [(0, 2), (1, 3), (2, 4), (3, 0), (4, 1)]
                .into_iter()
                .flat_map(|(lhs, rhs)| {
                    xor_gates(WordIdx::new(lhs), WordIdx::new(rhs).rotate_left(1))
                })
                .collect();
            VanillaNode::new(1, n_2.log2_sub_output_size(), gates, self.num_reps)
        };
        let n_5 = {
            let gates = chain![
                [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
                    .into_iter()
                    .flat_map(|(lhs, rhs)| xor_gates(
                        WordIdx::new(lhs),
                        WordIdx::new(rhs).input(1)
                    )),
                iter::repeat_with(VanillaGate::default)
            ]
            .take(1 << Self::LOG2_STATE_SIZE)
            .collect();
            VanillaNode::new(2, n_3.log2_sub_output_size(), gates, self.num_reps)
        };
        let n_6 = {
            let gates = izip!(0..25, iter::repeat([4, 0, 1, 2, 3]).flatten())
                .flat_map(|(lhs, rhs)| xor_gates(WordIdx::new(lhs), WordIdx::new(rhs).input(1)))
                .collect();
            VanillaNode::new(2, Self::LOG2_STATE_SIZE, gates, self.num_reps)
        };

        let [n_1, n_2, n_3, n_4, n_5, n_6] =
            [n_1, n_2, n_3, n_4, n_5, n_6].map(|node| circuit.insert(node.into_boxed()));

        connect!(circuit {
            n_1 <- state;
            n_2 <- n_1;
            n_3 <- state;
            n_4 <- n_2;
            n_5 <- n_3, n_4;
            n_6 <- state, n_5;
        });

        n_6
    }

    fn configure_rho_pi_chi_iota<F: Field>(
        &self,
        circuit: &mut Circuit<F>,
        state: NodeId,
        rc: u64,
    ) -> NodeId {
        let words = [
            (0, 0),
            (6, 44),
            (12, 43),
            (18, 21),
            (24, 14),
            (3, 28),
            (9, 20),
            (10, 3),
            (16, 45),
            (22, 61),
            (1, 1),
            (7, 6),
            (13, 25),
            (19, 8),
            (20, 18),
            (4, 27),
            (5, 36),
            (11, 10),
            (17, 15),
            (23, 56),
            (2, 62),
            (8, 55),
            (14, 39),
            (15, 41),
            (21, 2),
        ]
        .map(|(idx, rotate_left)| WordIdx::new(idx).rotate_left(rotate_left));
        let n_7 = {
            let gates = [
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 0),
                (0, 1),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 5),
                (5, 6),
                (11, 12),
                (12, 13),
                (13, 14),
                (14, 10),
                (10, 11),
                (16, 17),
                (17, 18),
                (18, 19),
                (19, 15),
                (15, 16),
                (21, 22),
                (22, 23),
                (23, 24),
                (24, 20),
                (20, 21),
            ]
            .into_iter()
            .flat_map(|(lhs, rhs)| not_lhs_and_rhs_gates(words[lhs], words[rhs]))
            .collect();
            VanillaNode::new(1, Self::LOG2_STATE_SIZE, gates, self.num_reps)
        };
        let n_8 = {
            let gates = chain![
                xor_with_rc_gates(words[0], WordIdx::new(0).input(1), rc),
                (1..25).flat_map(|idx| xor_gates(words[idx], WordIdx::new(idx).input(1)))
            ]
            .collect();
            VanillaNode::new(2, Self::LOG2_STATE_SIZE, gates, self.num_reps)
        };

        let [n_7, n_8] = [n_7, n_8].map(|node| circuit.insert(node));

        connect!(circuit {
            n_7 <- state;
            n_8 <- state, n_7;
        });

        n_8
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

#[cfg(test)]
pub mod test {
    use crate::hash::keccak::{
        dev::{keccak_circuit, keccak_circuit_inputs},
        Keccak,
    };
    use gkr::{
        dev::run_gkr,
        util::{
            arithmetic::{try_felt_to_bool, Field, PrimeField},
            dev::{rand_bytes, rand_range, seeded_std_rng},
            RngCore,
        },
    };
    use halo2_curves::bn256::Fr;

    #[test]
    fn keccak256() {
        let mut rng = seeded_std_rng();
        for num_reps in (0..4).map(|log2| 1 << log2) {
            run_keccak::<Fr>(256, num_reps, &mut rng);
        }
    }

    fn run_keccak<F: PrimeField>(num_bits: usize, num_reps: usize, mut rng: impl RngCore) {
        let rate = Keccak::rate(num_bits);
        let input = rand_bytes(rand_range(0..num_reps * rate, &mut rng), &mut rng);

        let circuit = keccak_circuit::<F>(num_bits, num_reps);
        let inputs = keccak_circuit_inputs(num_bits, num_reps, &input);
        run_gkr(&circuit, &inputs, &mut rng);

        let offset = (input.len() / rate) << Keccak::LOG2_STATE_SIZE;
        let output = felts_to_bytes(&circuit.evaluate(inputs).pop().unwrap()[offset..][..256]);
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

    fn felts_to_bytes<F: Field>(output: &[F]) -> Vec<u8> {
        Vec::from_iter(output.chunks_exact(8).map(|word| {
            word.iter()
                .rev()
                .copied()
                .map(try_felt_to_bool)
                .fold(0, |acc, bit| (acc << 1) + bit.unwrap() as u8)
        }))
    }
}

#[cfg(any(test, feature = "dev"))]
pub mod dev {
    use crate::hash::keccak::Keccak;
    use gkr::{
        circuit::Circuit,
        util::{
            arithmetic::{bool_to_felt, Field},
            chain, izip, Itertools,
        },
    };
    use std::iter;

    pub fn keccak_circuit<F: Field>(num_bits: usize, num_reps: usize) -> Circuit<F> {
        let mut circuit = Circuit::default();
        let keccak = Keccak::new(num_bits, num_reps);
        let state = keccak.alloc_state(&mut circuit);
        let input = keccak.alloc_input(&mut circuit);
        keccak.configure(&mut circuit, state, input);
        circuit
    }

    pub fn keccak_circuit_inputs<F: Field>(
        num_bits: usize,
        num_reps: usize,
        input: &[u8],
    ) -> Vec<Vec<F>> {
        let rate = Keccak::rate(num_bits);
        let inputs = chain![input.chunks(rate), [[].as_slice()]]
            .take(input.len() / rate + 1)
            .map(|chunk| {
                let mut chunk = chunk.to_vec();
                if chunk.len() != rate {
                    let offset = chunk.len();
                    chunk.resize(rate, 0);
                    chunk[offset] ^= 0x01;
                    chunk[rate - 1] ^= 0x80;
                }
                chunk.chunks_exact(8).map(u64_from_le_bytes).collect_vec()
            })
            .collect_vec();

        let states =
            inputs[..inputs.len() - 1]
                .iter()
                .fold(vec![vec![0; 25]], |mut states, chunk| {
                    let mut state: [_; 25] = states.last().unwrap().clone().try_into().unwrap();
                    izip!(&mut state, chunk).for_each(|(lhs, rhs)| *lhs ^= rhs);
                    tiny_keccak::keccakf(&mut state);
                    states.push(state.to_vec());
                    states
                });

        let len = num_reps.next_power_of_two() << Keccak::LOG2_STATE_SIZE;
        [states, inputs]
            .into_iter()
            .map(|values| {
                let padded_values = values.into_iter().flat_map(|value| {
                    chain![words_to_felts(value), iter::repeat(F::ZERO)]
                        .take(1 << Keccak::LOG2_STATE_SIZE)
                });
                Vec::from_iter(chain![padded_values, iter::repeat(F::ZERO)].take(len))
            })
            .collect()
    }

    fn u64_from_le_bytes(bytes: &[u8]) -> u64 {
        let mut word = [0; 8];
        word.copy_from_slice(bytes);
        u64::from_le_bytes(word)
    }

    fn words_to_felts<F: Field>(words: Vec<u64>) -> impl Iterator<Item = F> {
        words
            .into_iter()
            .flat_map(|word| (0..u64::BITS).map(move |idx| (word >> idx) & 1 == 1))
            .map(bool_to_felt)
    }
}
