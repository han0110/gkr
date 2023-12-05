use crate::util::{chain, Itertools};

#[derive(Clone, Debug)]
pub struct DirectedAcyclicGraph<T> {
    nodes: Vec<T>,
    edges: Vec<Vec<(usize, Direction)>>,
}

impl<T> DirectedAcyclicGraph<T> {
    pub(super) fn linear(nodes: Vec<T>) -> Self {
        let edges = (0..nodes.len())
            .map(|idx| {
                chain![
                    (idx != 0).then_some((idx.saturating_sub(1), Direction::In)),
                    (idx != nodes.len() - 1).then_some((idx + 1, Direction::Out)),
                ]
                .collect()
            })
            .collect();
        Self { nodes, edges }
    }

    pub(crate) fn nodes(&self) -> &[T] {
        &self.nodes
    }

    pub(crate) fn inputs(&self) -> impl Iterator<Item = usize> + '_ {
        self.indegs().positions(|deg| deg == 0)
    }

    pub(crate) fn outputs(&self) -> impl Iterator<Item = usize> + '_ {
        self.outdegs().positions(|deg| deg == 0)
    }

    pub(crate) fn predec(&self, idx: usize) -> impl Iterator<Item = usize> + '_ {
        self.adjs(idx, Direction::In)
    }

    pub(crate) fn succ(&self, idx: usize) -> impl Iterator<Item = usize> + '_ {
        self.adjs(idx, Direction::Out)
    }

    pub(super) fn indegs(&self) -> impl Iterator<Item = usize> + '_ {
        self.degs(Direction::In)
    }

    pub(super) fn outdegs(&self) -> impl Iterator<Item = usize> + '_ {
        self.degs(Direction::Out)
    }

    pub(super) fn topo(&self) -> Vec<usize> {
        let mut topo = Vec::with_capacity(self.nodes.len());
        let mut indegs = self.indegs().collect_vec();
        let mut queue = self.inputs().collect_vec();

        while let Some(idx) = queue.pop() {
            topo.push(idx);
            self.succ(idx).for_each(|idx| {
                indegs[idx] -= 1;
                if indegs[idx] == 0 {
                    queue.push(idx);
                }
            });
        }

        assert_eq!(topo.len(), self.nodes.len());

        topo
    }

    fn adjs(&self, idx: usize, direction: Direction) -> impl Iterator<Item = usize> + '_ {
        self.edges[idx]
            .iter()
            .filter_map(move |edge| (edge.1 == direction).then_some(edge.0))
    }

    fn degs(&self, direction: Direction) -> impl Iterator<Item = usize> + '_ {
        self.edges
            .iter()
            .map(move |edges| edges.iter().filter(|edge| edge.1 == direction).count())
    }

    pub(super) fn insert(&mut self, node: T) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(node);
        self.edges.resize_with(self.nodes.len(), Vec::new);
        NodeId(id)
    }

    pub(super) fn link(&mut self, from: NodeId, to: NodeId) {
        let NodeId(from) = from;
        let NodeId(to) = to;
        assert_ne!(from, to);

        assert!(!self.edges[from].iter().any(|(idx, _)| *idx == to));
        assert!(!self.edges[to].iter().any(|(idx, _)| *idx == from));

        self.edges[from].push((to, Direction::Out));
        self.edges[to].push((from, Direction::In));
    }
}

impl<T> Default for DirectedAcyclicGraph<T> {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct NodeId(pub(super) usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Direction {
    In,
    Out,
}
