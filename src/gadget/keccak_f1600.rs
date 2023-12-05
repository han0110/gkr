use crate::{
    circuit::{
        node::{Node, VanillaGate, VanillaNode},
        Circuit, NodeId,
    },
    util::{arithmetic::Field, chain, izip},
};
use std::iter;

pub const STATE_SIZE: usize = 1600;
pub const LOG2_STATE_SIZE: usize = STATE_SIZE.next_power_of_two().ilog2() as usize;

#[rustfmt::skip]
const RC: [u64; 24] = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a, 0x8000000080008000,
    0x000000000000808b, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008a, 0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800a, 0x800000008000000a,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
];

pub fn configure_keccak_f1600<F: Field>(
    circuit: &mut Circuit<F>,
    state: NodeId,
    num_reps: usize,
) -> NodeId {
    RC.into_iter().fold(state, |state, rc| {
        let state = configure_theta(circuit, state, num_reps);
        configure_rho_pi_chi_iota(circuit, state, rc, num_reps)
    })
}

fn configure_theta<F: Field>(circuit: &mut Circuit<F>, state: NodeId, num_reps: usize) -> NodeId {
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
        VanillaNode::new(1, LOG2_STATE_SIZE, gates, num_reps)
    };
    let n_2 = {
        let gates = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
            .into_iter()
            .flat_map(|(lhs, rhs)| xor_gates(WordIdx::new(lhs), WordIdx::new(rhs)))
            .collect();
        VanillaNode::new(1, n_1.log2_sub_output_size(), gates, num_reps)
    };
    let n_3_0 = {
        let gates = [(0, 2), (1, 3), (2, 4), (3, 0), (4, 1)]
            .into_iter()
            .flat_map(|(lhs, rhs)| xor_gates(WordIdx::new(lhs), WordIdx::new(rhs).rotate_left(1)))
            .collect();
        VanillaNode::new(1, n_2.log2_sub_output_size(), gates, num_reps)
    };
    let n_3_1 = {
        let gates = [(20, 22), (21, 23), (22, 24), (23, 20), (24, 21)]
            .into_iter()
            .flat_map(|(lhs, rhs)| xor_gates(WordIdx::new(lhs), WordIdx::new(rhs).rotate_left(1)))
            .collect();
        VanillaNode::new(1, LOG2_STATE_SIZE, gates, num_reps)
    };
    let n_4 = {
        let gates = chain![
            [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
                .into_iter()
                .flat_map(|(lhs, rhs)| xor_gates(WordIdx::new(lhs), WordIdx::new(rhs).input(1))),
            iter::repeat_with(VanillaGate::default)
        ]
        .take(1 << LOG2_STATE_SIZE)
        .collect();
        VanillaNode::new(2, n_3_0.log2_sub_output_size(), gates, num_reps)
    };
    let n_5 = {
        let gates = izip!(0..25, iter::repeat([4, 0, 1, 2, 3]).flatten())
            .flat_map(|(lhs, rhs)| xor_gates(WordIdx::new(lhs), WordIdx::new(rhs).input(1)))
            .collect();
        VanillaNode::new(2, LOG2_STATE_SIZE, gates, num_reps)
    };

    let [n_1, n_2, n_3_0, n_3_1, n_4, n_5] =
        [n_1, n_2, n_3_0, n_3_1, n_4, n_5].map(|node| circuit.insert(node.into_boxed()));

    circuit.link(state, n_1);
    circuit.link(n_1, n_2);
    circuit.link(n_2, n_3_0);
    circuit.link(state, n_3_1);
    circuit.link(n_3_0, n_4);
    circuit.link(n_3_1, n_4);
    circuit.link(state, n_5);
    circuit.link(n_4, n_5);

    n_5
}

fn configure_rho_pi_chi_iota<F: Field>(
    circuit: &mut Circuit<F>,
    state: NodeId,
    rc: u64,
    num_reps: usize,
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
    let n_6 = {
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
        VanillaNode::new(1, LOG2_STATE_SIZE, gates, num_reps)
    };
    let n_7 = {
        let gates = chain![
            xor_with_rc_gates(words[0], WordIdx::new(0).input(1), rc),
            (1..25).flat_map(|idx| xor_gates(words[idx], WordIdx::new(idx).input(1)))
        ]
        .collect();
        VanillaNode::new(2, LOG2_STATE_SIZE, gates, num_reps)
    };

    let [n_6, n_7] = [n_6, n_7].map(|node| circuit.insert(node.into_boxed()));

    circuit.link(state, n_6);
    circuit.link(state, n_7);
    circuit.link(n_6, n_7);

    n_7
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

    fn b_iter(self) -> impl Iterator<Item = usize> {
        (self.idx * 64..(self.idx + 1) * 64)
            .cycle()
            .skip(self.rotate_left)
            .take(64)
    }
}

fn xor_gates<F: Field>(lhs: WordIdx, rhs: WordIdx) -> impl Iterator<Item = VanillaGate<F>> {
    izip!(lhs.b_iter(), rhs.b_iter())
        .map(move |(b_0, b_1)| VanillaGate::xor((lhs.input, b_0), (rhs.input, b_1)))
}

fn xor_with_rc_gates<F: Field>(
    lhs: WordIdx,
    rhs: WordIdx,
    rc: u64,
) -> impl Iterator<Item = VanillaGate<F>> {
    let bits = (0..64).rev().map(move |idx| (rc >> idx) & 1 == 1);
    izip!(lhs.b_iter(), rhs.b_iter(), bits).map(move |(b_0, b_1, bit)| {
        let gate = if bit {
            VanillaGate::xnor
        } else {
            VanillaGate::xor
        };
        gate((lhs.input, b_0), (rhs.input, b_1))
    })
}

fn not_lhs_and_rhs_gates<F: Field>(
    lhs: WordIdx,
    rhs: WordIdx,
) -> impl Iterator<Item = VanillaGate<F>> {
    izip!(lhs.b_iter(), rhs.b_iter()).map(move |(b_0, b_1)| {
        VanillaGate::new(
            None,
            vec![(None, (rhs.input, b_1))],
            vec![(Some(-F::ONE), (lhs.input, b_0), (rhs.input, b_1))],
        )
    })
}

#[cfg(test)]
mod test {
    use crate::{
        circuit::{node::InputNode, Circuit},
        gadget::keccak_f1600::{configure_keccak_f1600, LOG2_STATE_SIZE},
        test::run_gkr,
        util::{
            arithmetic::{bool_to_felt, try_felt_to_bool, Field},
            chain,
            test::seeded_std_rng,
            // RngCore,
        },
    };
    use halo2_curves::bn256::Fr;
    use itertools::Itertools;
    use std::iter;

    #[test]
    fn keccak_f1600() {
        #[rustfmt::skip]
        const STATE: [u64; 25] = [
            0x00, 0x01, 0x02, 0x03, 0x04,
            0x05, 0x06, 0x07, 0x08, 0x09,
            0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
            0x0f, 0x10, 0x11, 0x12, 0x13,
            0x14, 0x15, 0x16, 0x17, 0x18,
        ];
        #[rustfmt::skip]
        const STATE_PRIME: [u64; 25] = [
            0x8374b05252ed8115, 0x1df7a676b6569400, 0xf765194b8a51797d, 0x20477b43d1760545, 0xd15f8ba4f3f6606a,
            0xa1d7144f7c8dd493, 0x30d193965138fd3f, 0x487e9472951be3be, 0x0cf3a858cbda7a5a, 0x2fe54e389bb17f88,
            0x0b7338de0d9f268f, 0x55efdff58b256d7f, 0xc8353e94eb2c3e6a, 0x2e2af6948c901f11, 0xe873de0cca309da6,
            0xf7afc26c944d31e2, 0xa0f5ea808cc415d7, 0x53f531437e3ed8cf, 0x777f1f3b43a4d221, 0xfd0ca63cb499e985,
            0xd4c055c0c5d12330, 0xa72fe58aa6e0a7df, 0x421af5937c9948a3, 0x5e16103071340888, 0xd153f43a297e4a33,
        ];
        let mut rng = seeded_std_rng();
        for log2_reps in 0..5 {
            let num_reps = 1 << log2_reps;
            let circuit = keccak_f1600_circuit::<Fr>(num_reps);
            let input = input(STATE, num_reps);
            let output = circuit.evaluate(vec![input.clone()]).pop().unwrap();
            assert_eq!(felts_to_words(&output)[..25], STATE_PRIME);
            run_gkr(&circuit, vec![input], &mut rng)
        }
    }

    fn keccak_f1600_circuit<F: Field>(num_reps: usize) -> Circuit<F> {
        let mut circuit = Circuit::default();
        let state = circuit.insert(InputNode::new(LOG2_STATE_SIZE, num_reps));
        configure_keccak_f1600(&mut circuit, state, num_reps);
        circuit
    }

    // fn rand_state(mut rng: impl RngCore) -> [u64; 25] {
    //     from_fn(|_| rng.next_u64())
    // }

    fn word_to_felts<F: Field>(word: u64) -> impl Iterator<Item = F> {
        (0..u64::BITS)
            .rev()
            .map(move |idx| (word >> idx) & 1 == 1)
            .map(bool_to_felt)
    }

    fn felts_to_words<F: Field>(output: &[F]) -> Vec<u64> {
        Vec::from_iter(output.chunks(64).map(|word| {
            word.iter()
                .copied()
                .map(try_felt_to_bool)
                .fold(0, |acc, bit| (acc << 1) + bit.unwrap() as u64)
        }))
    }

    fn input<F: Field>(state: [u64; 25], num_reps: usize) -> Vec<F> {
        let state_bits = state.into_iter().flat_map(word_to_felts).collect_vec();
        iter::repeat_with(|| {
            chain![state_bits.clone(), iter::repeat(F::ZERO)].take(1 << LOG2_STATE_SIZE)
        })
        .take(num_reps.next_power_of_two())
        .flatten()
        .collect()
    }
}
