use crate::{
    util::{arithmetic::PrimeField, RngCore, SeedableRng, StdRng},
    Error,
};
use std::{
    fmt::Debug,
    io::{self},
    iter,
};

pub trait Transcript<F>: Debug {
    fn common_felt(&mut self, felt: &F);

    fn common_felts(&mut self, felts: &[F]) {
        felts.iter().for_each(|felt| self.common_felt(felt));
    }

    fn squeeze_challenge(&mut self) -> F;

    fn squeeze_challenges(&mut self, n: usize) -> Vec<F> {
        iter::repeat_with(|| self.squeeze_challenge())
            .take(n)
            .collect()
    }
}

pub trait TranscriptWrite<F>: Transcript<F> {
    fn write_felt(&mut self, felt: &F) -> Result<(), Error>;

    fn write_felts(&mut self, felts: &[F]) -> Result<(), Error> {
        felts.iter().try_for_each(|felt| self.write_felt(felt))
    }
}

pub trait TranscriptRead<F>: Transcript<F> {
    fn read_felt(&mut self) -> Result<F, Error>;

    fn read_felts(&mut self, n: usize) -> Result<Vec<F>, Error> {
        iter::repeat_with(|| self.read_felt()).take(n).collect()
    }
}

pub type StdRngTranscript<S> = RngTranscript<S, StdRng>;

#[derive(Debug)]
pub struct RngTranscript<S, P> {
    stream: S,
    rng: P,
}

impl<P> RngTranscript<Vec<u8>, P> {
    pub fn into_proof(self) -> Vec<u8> {
        self.stream
    }
}

impl<'a, P> RngTranscript<&'a [u8], P>
where
    Self: Default,
{
    pub fn from_proof(proof: &'a [u8]) -> Self {
        Self {
            stream: proof,
            ..Default::default()
        }
    }
}

impl<S: Default> Default for RngTranscript<S, StdRng> {
    fn default() -> Self {
        Self {
            stream: S::default(),
            rng: StdRng::seed_from_u64(0),
        }
    }
}

impl<F: PrimeField, S: Debug, P: Debug + RngCore> Transcript<F> for RngTranscript<S, P> {
    fn squeeze_challenge(&mut self) -> F {
        F::random(&mut self.rng)
    }

    fn common_felt(&mut self, _: &F) {}
}

impl<F: PrimeField, R: Debug + io::Read, P: Debug + RngCore> TranscriptRead<F>
    for RngTranscript<R, P>
{
    fn read_felt(&mut self) -> Result<F, Error> {
        let mut repr = <F as PrimeField>::Repr::default();
        self.stream
            .read_exact(repr.as_mut())
            .map_err(|err| Error::Transcript(err.kind(), err.to_string()))?;
        repr.as_mut().reverse();
        let felt = F::from_repr_vartime(repr).ok_or_else(err_invalid_felt)?;
        Ok(felt)
    }
}

impl<F: PrimeField, W: Debug + io::Write, P: Debug + RngCore> TranscriptWrite<F>
    for RngTranscript<W, P>
{
    fn write_felt(&mut self, felt: &F) -> Result<(), Error> {
        let mut repr = felt.to_repr();
        repr.as_mut().reverse();
        self.stream
            .write_all(repr.as_ref())
            .map_err(|err| Error::Transcript(err.kind(), err.to_string()))
    }
}

fn err_invalid_felt() -> Error {
    Error::Transcript(
        io::ErrorKind::Other,
        "Invalid field element read from stream".to_string(),
    )
}
