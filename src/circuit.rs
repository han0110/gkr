use crate::{
    circuit::{dag::AdjacencyMatrix, node::Node},
    util::Field,
};

mod dag;
pub mod node;

pub use dag::DirectedAcyclicGraph;

#[derive(Debug)]
pub struct Circuit<F> {
    dag: DirectedAcyclicGraph<Box<dyn Node<F>>>,
    topo: Vec<usize>,
}

impl<F: Field> Circuit<F> {
    pub fn new(dag: DirectedAcyclicGraph<Box<dyn Node<F>>>) -> Self {
        let topo = dag.topo();
        topo.iter()
            .for_each(|idx| assert!(!dag.nodes()[*idx].is_input() || dag.succ(*idx).count() > 0));
        Self { dag, topo }
    }

    pub fn nodes(&self) -> &[Box<dyn Node<F>>] {
        self.dag.nodes()
    }

    pub fn adj_mat(&self) -> &AdjacencyMatrix {
        self.dag.adj_mat()
    }

    pub fn topo_iter(&self) -> impl DoubleEndedIterator<Item = (usize, &dyn Node<F>)> {
        self.topo.iter().map(|idx| (*idx, &*self.nodes()[*idx]))
    }

    pub fn evaluate(&self, inputs: Vec<Vec<F>>) -> Vec<Vec<F>> {
        let mut inputs = inputs.into_iter();
        let values = self.topo_iter().fold(
            vec![Vec::new(); self.nodes().len()],
            |mut values, (idx, node)| {
                values[idx] = if node.is_input() {
                    inputs.next().unwrap()
                } else {
                    node.evaluate(self.dag.predec(idx).map(|idx| &values[idx]).collect())
                };
                values
            },
        );

        assert!(inputs.next().is_none());

        values
    }
}
