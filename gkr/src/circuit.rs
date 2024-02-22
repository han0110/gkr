use crate::{
    circuit::{
        dag::DirectedAcyclicGraph,
        node::{Node, NodeExt},
    },
    poly::BoxMultilinearPoly,
    util::{
        arithmetic::{ExtensionField, Field},
        izip_eq, Itertools,
    },
};
use std::{iter, ops::Deref};

mod dag;
pub mod node;

pub use dag::NodeId;

#[derive(Debug, Default)]
pub struct Circuit<F, E> {
    dag: DirectedAcyclicGraph<Box<dyn Node<F, E>>>,
    topo: Vec<usize>,
}

impl<F: Field, E: ExtensionField<F>> Circuit<F, E> {
    pub fn linear(nodes: Vec<Box<dyn Node<F, E>>>) -> Self {
        let dag = DirectedAcyclicGraph::linear(nodes);
        let topo = dag.topo();
        Self { dag, topo }
    }

    pub fn insert(&mut self, node: impl Node<F, E> + 'static) -> NodeId {
        self.dag.insert(node.boxed())
    }

    pub fn connect(&mut self, from: NodeId, to: NodeId) {
        assert!(self.nodes()[from.0].log2_output_size() <= self.nodes()[to.0].log2_input_size());
        assert!(!self.nodes()[to.0].is_input());
        self.dag.connect(from, to);
        self.topo = self.dag.topo();
    }

    pub fn evaluate<'a>(
        &self,
        inputs: Vec<BoxMultilinearPoly<'a, F, E>>,
    ) -> Vec<BoxMultilinearPoly<'a, F, E>> {
        let mut values = Vec::from_iter(iter::repeat_with(|| None).take(self.nodes().len()));

        izip_eq!(self.inputs(), inputs).for_each(|(idx, input)| values[idx] = input.into());
        self.topo_iter()
            .filter(|(_, node)| !node.is_input())
            .for_each(|(idx, node)| {
                let inputs = self.predec(idx).map(|i| values[i].as_ref().unwrap());
                values[idx] = node.evaluate(inputs.collect()).into()
            });

        values.into_iter().map(Option::unwrap).collect()
    }

    pub(crate) fn topo_iter(&self) -> impl DoubleEndedIterator<Item = (usize, &dyn Node<F, E>)> {
        self.topo.iter().map(|idx| (*idx, &*self.nodes()[*idx]))
    }
}

impl<F, E> Deref for Circuit<F, E> {
    type Target = DirectedAcyclicGraph<Box<dyn Node<F, E>>>;

    fn deref(&self) -> &Self::Target {
        &self.dag
    }
}

#[macro_export]
macro_rules! connect {
    ($circuit:ident { $($to:ident <- $($from:ident),*);*$(;)? }) => {
        $($($circuit.connect($from, $to);)*)*
    };
    ($circuit:ident { $($($from:ident),* -> $to:ident);*$(;)? }) => {
        $($($circuit.connect($from, $to);)*)*
    };
}

pub use connect;

#[cfg(test)]
pub(super) mod test {
    use crate::{
        circuit::Circuit,
        dev::run_gkr_with_values,
        poly::BoxMultilinearPoly,
        util::{
            arithmetic::{ExtensionField, PrimeField},
            dev::{assert_polys_eq, seeded_std_rng},
        },
    };
    use rand::rngs::StdRng;

    pub(super) type TestData<F, E> = (
        Circuit<F, E>,
        Vec<BoxMultilinearPoly<'static, F, E>>,
        Option<Vec<BoxMultilinearPoly<'static, F, E>>>,
    );

    pub(super) fn run_circuit<F: PrimeField, E: ExtensionField<F>>(
        f: impl Fn(usize, &mut StdRng) -> TestData<F, E>,
    ) {
        let mut rng = seeded_std_rng();
        for num_vars in 1..10 {
            let (circuit, inputs, expected_values) = f(num_vars, &mut rng);
            let values = circuit.evaluate(inputs);
            if let Some(expected_values) = expected_values {
                assert_polys_eq(&values, &expected_values);
            }
            run_gkr_with_values(&circuit, &values, &mut rng);
        }
    }
}
