use crate::util::Itertools;
use std::ops::Deref;

#[derive(Clone, Debug)]
pub struct DirectedAcyclicGraph<T> {
    nodes: Vec<T>,
    adj_mat: AdjacencyMatrix,
}

impl<T> DirectedAcyclicGraph<T> {
    pub fn new(nodes: Vec<T>, adj_mat: AdjacencyMatrix) -> Self {
        assert_eq!(nodes.len(), adj_mat.0.len());
        assert!(adj_mat.topo().is_some());

        Self { nodes, adj_mat }
    }

    pub fn linear(nodes: Vec<T>) -> Self {
        let mut adj_mat = AdjacencyMatrix::new(nodes.len());
        (0..nodes.len()).for_each(|i| {
            if i != 0 {
                adj_mat.0[i][i - 1] = Some(Direction::In)
            }
            if i != nodes.len() - 1 {
                adj_mat.0[i][i + 1] = Some(Direction::Out)
            }
        });
        Self::new(nodes, adj_mat)
    }

    pub fn insert(&mut self, node: T) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(node);
        self.adj_mat.resize(self.nodes.len());
        NodeId(id)
    }

    pub fn link(&mut self, NodeId(from): NodeId, NodeId(to): NodeId) {
        let adj_mat = &mut self.adj_mat.0;
        assert!(matches!(adj_mat[from][to], None | Some(Direction::Out)));
        assert!(matches!(adj_mat[to][from], None | Some(Direction::In)));

        adj_mat[from][to] = Some(Direction::Out);
        adj_mat[to][from] = Some(Direction::In);

        assert!(self.adj_mat.topo().is_some());
    }

    pub fn nodes(&self) -> &[T] {
        &self.nodes
    }

    pub fn adj_mat(&self) -> &AdjacencyMatrix {
        &self.adj_mat
    }

    pub fn topo(&self) -> Vec<usize> {
        self.adj_mat.topo().unwrap()
    }
}

impl<T> Default for DirectedAcyclicGraph<T> {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            adj_mat: AdjacencyMatrix::default(),
        }
    }
}

impl<T> Deref for DirectedAcyclicGraph<T> {
    type Target = AdjacencyMatrix;

    fn deref(&self) -> &Self::Target {
        &self.adj_mat
    }
}

#[derive(Clone, Copy, Debug)]
pub struct NodeId(usize);

#[derive(Clone, Debug, Default)]
pub struct AdjacencyMatrix(Vec<Vec<Edge>>);

impl AdjacencyMatrix {
    pub fn new(len: usize) -> Self {
        Self(vec![vec![None; len]; len])
    }

    pub fn topo(&self) -> Option<Vec<usize>> {
        let mut topo = Vec::with_capacity(self.0.len());
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

        (topo.len() == self.0.len()).then_some(topo)
    }

    pub fn predec(&self, idx: usize) -> impl Iterator<Item = usize> + '_ {
        self.adjs(idx, Direction::In)
    }

    pub fn succ(&self, idx: usize) -> impl Iterator<Item = usize> + '_ {
        self.adjs(idx, Direction::Out)
    }

    pub fn indegs(&self) -> impl Iterator<Item = usize> + '_ {
        self.degs(Direction::In)
    }

    pub fn outdegs(&self) -> impl Iterator<Item = usize> + '_ {
        self.degs(Direction::Out)
    }

    pub fn inputs(&self) -> impl Iterator<Item = usize> + '_ {
        self.indegs().positions(|deg| deg == 0)
    }

    pub fn outputs(&self) -> impl Iterator<Item = usize> + '_ {
        self.outdegs().positions(|deg| deg == 0)
    }

    pub fn resize(&mut self, size: usize) {
        self.0.iter_mut().for_each(|e| e.resize(size, None));
        self.0.resize(size, vec![None; size]);
    }

    fn adjs(&self, idx: usize, direction: Direction) -> impl Iterator<Item = usize> + '_ {
        self.0[idx].iter().positions(move |e| *e == Some(direction))
    }

    fn degs(&self, direction: Direction) -> impl Iterator<Item = usize> + '_ {
        self.0
            .iter()
            .map(move |edges| edges.iter().filter(move |e| **e == Some(direction)).count())
    }
}

pub type Edge = Option<Direction>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    In,
    Out,
}
