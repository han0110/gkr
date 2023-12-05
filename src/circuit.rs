use crate::{
    circuit::{dag::DirectedAcyclicGraph, node::Node},
    util::{arithmetic::Field, izip_eq, Itertools},
};
use std::ops::Deref;

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
        let node = node.into_boxed();
        self.dag.insert(node)
    }

    pub fn link(&mut self, from: NodeId, to: NodeId) {
        assert_eq!(
            self.nodes()[from.0].log2_output_size(),
            self.nodes()[to.0].log2_input_size()
        );
        assert!(!self.nodes()[to.0].is_input());
        self.dag.link(from, to);
        self.topo = self.dag.topo();
    }

    pub fn evaluate(&self, inputs: Vec<Vec<F>>) -> Vec<Vec<F>> {
        let mut values = vec![Vec::new(); self.nodes().len()];

        izip_eq!(self.dag.inputs(), inputs).for_each(|(idx, input)| values[idx] = input);
        self.topo_iter()
            .filter(|(_, node)| !node.is_input())
            .for_each(|(idx, node)| {
                values[idx] = node.evaluate(self.dag.predec(idx).map(|idx| &values[idx]).collect())
            });

        values
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
