use ff::Field;
use std::{
    ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
    slice,
};

pub use ff;

pub trait ExtensionField<F>:
    Field
    + Add<F, Output = Self>
    + Sub<F, Output = Self>
    + Mul<F, Output = Self>
    + for<'a> Add<&'a F, Output = Self>
    + for<'a> Sub<&'a F, Output = Self>
    + for<'a> Mul<&'a F, Output = Self>
    + AddAssign<F>
    + SubAssign<F>
    + MulAssign<F>
    + for<'a> AddAssign<&'a F>
    + for<'a> SubAssign<&'a F>
    + for<'a> MulAssign<&'a F>
{
    const DEGREE: usize;

    fn from_base(base: F) -> Self;

    fn from_bases(bases: &[F]) -> Self;

    fn as_bases(&self) -> &[F];
}

impl<F: Field> ExtensionField<F> for F {
    const DEGREE: usize = 1;

    fn from_base(base: F) -> Self {
        base
    }

    fn from_bases(bases: &[F]) -> Self {
        debug_assert_eq!(bases.len(), 1);
        bases[0]
    }

    fn as_bases(&self) -> &[F] {
        slice::from_ref(self)
    }
}

mod impl_goldilocks {
    use crate::{ff::Field, ExtensionField};
    use goldilocks::{Goldilocks, GoldilocksExt2};

    impl ExtensionField<Goldilocks> for GoldilocksExt2 {
        const DEGREE: usize = 2;

        fn from_base(base: Goldilocks) -> Self {
            Self([base, Goldilocks::ZERO])
        }

        fn from_bases(bases: &[Goldilocks]) -> Self {
            debug_assert_eq!(bases.len(), 2);
            Self([bases[0], bases[1]])
        }

        fn as_bases(&self) -> &[Goldilocks] {
            self.0.as_slice()
        }
    }
}
