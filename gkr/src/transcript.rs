use crate::{
    util::{
        arithmetic::{ExtensionField, PrimeField},
        RngCore, SeedableRng, StdRng,
    },
    Error, Itertools,
};
use std::{
    fmt::Debug,
    io::{self},
    iter,
};

pub trait Transcript<F, E>: Debug {
    fn common_felt(&mut self, felt: &F);

    fn common_felts(&mut self, felts: &[F]) {
        felts.iter().for_each(|felt| self.common_felt(felt));
    }

    fn squeeze_challenge(&mut self) -> E;

    fn squeeze_challenges(&mut self, n: usize) -> Vec<E> {
        iter::repeat_with(|| self.squeeze_challenge())
            .take(n)
            .collect()
    }
}

pub trait TranscriptWrite<F, E>: Transcript<F, E> {
    fn write_felt(&mut self, felt: &F) -> Result<(), Error>;

    fn write_felt_ext(&mut self, felt: &E) -> Result<(), Error>;

    fn write_felts(&mut self, felts: &[F]) -> Result<(), Error> {
        felts.iter().try_for_each(|felt| self.write_felt(felt))
    }

    fn write_felt_exts(&mut self, felts: &[E]) -> Result<(), Error> {
        felts.iter().try_for_each(|felt| self.write_felt_ext(felt))
    }
}

pub trait TranscriptRead<F, E>: Transcript<F, E> {
    fn read_felt(&mut self) -> Result<F, Error>;

    fn read_felt_ext(&mut self) -> Result<E, Error>;

    fn read_felts(&mut self, n: usize) -> Result<Vec<F>, Error> {
        iter::repeat_with(|| self.read_felt()).take(n).collect()
    }

    fn read_felt_exts(&mut self, n: usize) -> Result<Vec<E>, Error> {
        iter::repeat_with(|| self.read_felt_ext()).take(n).collect()
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

impl<'a> RngTranscript<&'a [u8], StdRng> {
    pub fn from_proof(proof: &'a [u8]) -> Self {
        Self::new(proof)
    }
}

impl<S> RngTranscript<S, StdRng> {
    pub fn new(stream: S) -> Self {
        Self {
            stream,
            rng: StdRng::seed_from_u64(0),
        }
    }
}

impl Default for RngTranscript<Vec<u8>, StdRng> {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

impl<F: PrimeField, E: ExtensionField<F>, S: Debug, P: Debug + RngCore> Transcript<F, E>
    for RngTranscript<S, P>
{
    fn squeeze_challenge(&mut self) -> E {
        let bases = iter::repeat_with(|| F::random(&mut self.rng))
            .take(E::DEGREE)
            .collect_vec();
        E::from_bases(&bases)
    }

    fn common_felt(&mut self, _: &F) {}
}

impl<F: PrimeField, E: ExtensionField<F>, R: Debug + io::Read, P: Debug + RngCore>
    TranscriptRead<F, E> for RngTranscript<R, P>
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

    fn read_felt_ext(&mut self) -> Result<E, Error> {
        let bases = iter::repeat_with(|| TranscriptRead::<F, E>::read_felt(self))
            .take(E::DEGREE)
            .try_collect::<_, Vec<_>, _>()?;
        Ok(E::from_bases(&bases))
    }
}

impl<F: PrimeField, E: ExtensionField<F>, W: Debug + io::Write, P: Debug + RngCore>
    TranscriptWrite<F, E> for RngTranscript<W, P>
{
    fn write_felt(&mut self, felt: &F) -> Result<(), Error> {
        let mut repr = felt.to_repr();
        repr.as_mut().reverse();
        self.stream
            .write_all(repr.as_ref())
            .map_err(|err| Error::Transcript(err.kind(), err.to_string()))
    }

    fn write_felt_ext(&mut self, felt: &E) -> Result<(), Error> {
        felt.as_bases()
            .iter()
            .try_for_each(|base| TranscriptWrite::<F, E>::write_felt(self, base))
    }
}

fn err_invalid_felt() -> Error {
    Error::Transcript(
        io::ErrorKind::Other,
        "Invalid field element read from stream".to_string(),
    )
}
