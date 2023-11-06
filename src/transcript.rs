use crate::{util::PrimeField, Error};
use num_bigint::BigUint;
use sha3::{
    digest::{Digest, FixedOutputReset},
    Keccak256,
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

pub type Keccak256Transcript<S> = FiatShamirTranscript<Keccak256, S>;

#[derive(Debug, Default)]
pub struct FiatShamirTranscript<H, S> {
    state: H,
    stream: S,
}

impl<H> FiatShamirTranscript<H, Vec<u8>> {
    pub fn into_proof(self) -> Vec<u8> {
        self.stream
    }
}

impl<'a, H: Default> FiatShamirTranscript<H, &'a [u8]> {
    pub fn from_proof(proof: &'a [u8]) -> Self {
        Self {
            stream: proof,
            ..Default::default()
        }
    }
}

impl<H: Debug + Digest + FixedOutputReset, F: PrimeField, S: Debug> Transcript<F>
    for FiatShamirTranscript<H, S>
{
    fn squeeze_challenge(&mut self) -> F {
        let hash = self.state.finalize_fixed_reset();
        Digest::update(&mut self.state, &hash);
        felt_from_le_bytes(&(BigUint::from_bytes_be(&hash) % modulus::<F>()).to_bytes_le())
    }

    fn common_felt(&mut self, felt: &F) {
        Digest::update(&mut self.state, felt.to_repr());
    }
}

impl<H: Debug + Digest + FixedOutputReset, F: PrimeField, R: Debug + io::Read> TranscriptRead<F>
    for FiatShamirTranscript<H, R>
{
    fn read_felt(&mut self) -> Result<F, Error> {
        let mut repr = <F as PrimeField>::Repr::default();
        self.stream
            .read_exact(repr.as_mut())
            .map_err(|err| Error::Transcript(err.kind(), err.to_string()))?;
        repr.as_mut().reverse();
        let felt = F::from_repr_vartime(repr).ok_or_else(err_invalid_felt)?;
        self.common_felt(&felt);
        Ok(felt)
    }
}

impl<H: Debug + Digest + FixedOutputReset, F: PrimeField, W: Debug + io::Write> TranscriptWrite<F>
    for FiatShamirTranscript<H, W>
{
    fn write_felt(&mut self, felt: &F) -> Result<(), Error> {
        self.common_felt(felt);
        let mut repr = felt.to_repr();
        repr.as_mut().reverse();
        self.stream
            .write_all(repr.as_ref())
            .map_err(|err| Error::Transcript(err.kind(), err.to_string()))
    }
}

fn modulus<F: PrimeField>() -> BigUint {
    BigUint::from_bytes_le((-F::ONE).to_repr().as_ref()) + 1u64
}

fn felt_from_le_bytes<F: PrimeField>(le_bytes: &[u8]) -> F {
    let mut repr = F::Repr::default();
    assert!(le_bytes.len() <= repr.as_ref().len());
    repr.as_mut()[..le_bytes.len()].copy_from_slice(le_bytes);
    F::from_repr(repr).unwrap()
}

fn err_invalid_felt() -> Error {
    Error::Transcript(
        io::ErrorKind::Other,
        "Invalid field element read from stream".to_string(),
    )
}
