use crate::{
    circuit::{dag::DirectedAcyclicGraph, node::Node},
    poly::BoxMultilinearPoly,
    util::{arithmetic::Field, izip_eq, Itertools},
};
use std::{iter, ops::Deref};

mod dag;
pub mod node;

pub use dag::NodeId;

#[derive(Debug, Default)]
pub struct Circuit<F> {
    dag: DirectedAcyclicGraph<Box<dyn Node<F>>>,
    topo: Vec<usize>,
}

impl<F: Field> Circuit<F> {
    pub fn linear(nodes: Vec<Box<dyn Node<F>>>) -> Self {
        let dag = DirectedAcyclicGraph::linear(nodes);
        let topo = dag.topo();
        Self { dag, topo }
    }

    pub fn insert(&mut self, node: impl Node<F> + 'static) -> NodeId {
        self.dag.insert(node.boxed())
    }

    pub fn connect(&mut self, from: NodeId, to: NodeId) {
        assert_eq!(
            self.nodes()[from.0].log2_output_size(),
            self.nodes()[to.0].log2_input_size()
        );
        assert!(!self.nodes()[to.0].is_input());
        self.dag.connect(from, to);
        self.topo = self.dag.topo();
    }

    pub fn evaluate<'a>(
        &self,
        inputs: Vec<BoxMultilinearPoly<'a, F>>,
    ) -> Vec<BoxMultilinearPoly<'a, F>> {
        let mut values = Vec::from_iter(iter::repeat_with(|| None).take(self.nodes().len()));

        izip_eq!(self.inputs(), inputs).for_each(|(idx, input)| values[idx] = input.into());
        self.topo_iter()
            .filter(|(_, node)| !node.is_input())
            .for_each(|(idx, node)| {
                let inputs = self.predec(idx).map(|i| values[i].as_deref().unwrap());
                values[idx] = node.evaluate(inputs.collect()).into()
            });

        values.into_iter().map(Option::unwrap).collect()
    }

    pub(crate) fn topo_iter(&self) -> impl DoubleEndedIterator<Item = (usize, &dyn Node<F>)> {
        self.topo.iter().map(|idx| (*idx, &*self.nodes()[*idx]))
    }
}

impl<F> Deref for Circuit<F> {
    type Target = DirectedAcyclicGraph<Box<dyn Node<F>>>;

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
