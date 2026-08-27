// Copyright (c) IBM
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

//! Working precision for this crate's arbitrary-precision arithmetic is **explicit, not
//! ambient**: there is no global or thread-local precision anywhere. Every value that needs
//! one either carries a [`Prec`] field (the per-synthesis structs in `region.rs`,
//! `gridsynth.rs`, `protocol::*`, and the result types in `accuracy.rs`) or is computed via a
//! method on [`Prec`] (here and in `math.rs`; the ring accessors in `ring::*` still take one as
//! a parameter, since precision is a property of a ring element's float *projection*, not of
//! the exact ring element itself).
//!
//! Two things make this affordable instead of an infestation of parameters:
//!
//! - [`Prec::ib`] is the *only* load-bearing coercion. `FBig::from(IBig)` is precision-0,
//!   which `dashu_float` defines as **unlimited** -- useful for exact integers, but "can lead
//!   to very huge significands" the moment it enters arithmetic with a bounded value. Every
//!   `IBig` seed into this crate's float arithmetic goes through `Prec::ib` for exactly this
//!   reason.
//! - [`Prec::fb`] (re-pinning an existing `FBig`) is needed far less than it looks: dashu
//!   propagates precision through `+ - * /` as `Context::max(lhs, rhs)`, so once a value is
//!   pinned it stays pinned through arithmetic with other values at the same precision.
//!   Re-pinning only matters at a genuine precision *boundary* -- a fresh `FBig::try_from(f64)`
//!   literal, or a value that might have crossed threads/calls at a different precision.
//!
//! `GridSynthConfig::prec` is the source of truth for one synthesis; `gridsynth_gates`/
//! `gridsynth_unitary` build every per-synthesis struct from it once, and everything
//! downstream reads `self.prec` rather than reaching for anything ambient.

use dashu_float::{round::mode::HalfEven, Context, FBig};
use dashu_int::IBig;

/// Working precision, in bits, for one synthesis. Threaded explicitly -- there is no ambient
/// or global precision anywhere in this crate. `Copy` so it can be passed around as freely as
/// a `usize`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Prec(pub usize);

impl Prec {
    /// `IBig` -> `FBig` at this precision. Load-bearing, not defensive: `FBig::from(IBig)` is
    /// precision-0 (unlimited), which `dashu_float` warns can produce huge significands, so
    /// this is what bounds the significand of every integer seed entering float arithmetic.
    pub fn ib(self, x: IBig) -> FBig<HalfEven> {
        FBig::from(x).with_precision(self.0).value()
    }

    /// Re-pin an existing `FBig` to this precision. Dashu propagates precision as
    /// `max(lhs, rhs)` through arithmetic, so this is a no-op on a value already at `self` --
    /// use it at genuine precision boundaries (a fresh `f64` literal, a value that may have
    /// crossed a call at a different precision), not defensively after every operation.
    pub fn fb(self, x: FBig<HalfEven>) -> FBig<HalfEven> {
        x.with_precision(self.0).value()
    }

    /// A `dashu_float::Context` at this precision, for APIs (`sqrt`, `ln`, ...) that take one
    /// directly rather than going through an existing `FBig`'s precision.
    pub fn ctx(self) -> Context<HalfEven> {
        Context::new(self.0)
    }

    pub fn bits(self) -> usize {
        self.0
    }

    /// `pi` at this precision, via `dashu_float`'s Chudnovsky-algorithm implementation.
    /// Recomputed per call -- there is no cache, since a cache keyed by an explicit parameter
    /// buys nothing a caller can't do itself (memoize on whatever per-synthesis struct is
    /// calling this repeatedly, if it matters).
    pub fn pi(self) -> FBig<HalfEven> {
        FBig::pi(self.0)
    }

    pub fn tau(self) -> FBig<HalfEven> {
        2 * self.pi()
    }

    pub fn cos(self, x: &FBig<HalfEven>) -> FBig<HalfEven> {
        self.ctx()
            .cos(x.repr(), None)
            .expect("cos of a finite FBig cannot fail")
            .value()
    }

    pub fn sin(self, x: &FBig<HalfEven>) -> FBig<HalfEven> {
        self.ctx()
            .sin(x.repr(), None)
            .expect("sin of a finite FBig cannot fail")
            .value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dashu_float::round::mode::HalfEven;
    use dashu_float::FBig;
    use dashu_int::ops::Abs;
    use rand::Rng;
    use std::f64::consts::PI as PI_F64;

    const PREC: Prec = Prec(1000);

    fn to_fbig(x: f64) -> FBig<HalfEven> {
        FBig::<HalfEven>::try_from(x)
            .unwrap()
            .with_precision(PREC.bits())
            .value()
    }

    fn approx_eq(a: &FBig<HalfEven>, b: &FBig<HalfEven>, tol_bits: usize) -> bool {
        let diff = (a - b).abs();
        let tol = PREC.ib(IBig::ONE) / PREC.fb(FBig::from(1u64 << tol_bits));
        diff <= tol
    }

    #[test]
    fn test_sin_random() {
        // This test asserts against a fixed bit-tolerance, `PREC`, that has nothing to do with
        // whatever precision any other concurrently-running test happens to be using -- each
        // test builds its own `Prec` value, so there is no shared state to race on.
        //
        // The tolerance itself must also stay safely below the *reference*'s own precision:
        // `expected` is `x_f64.sin()`, an f64 (~53 bits of mantissa) computed by the
        // platform's libm, which is typically only guaranteed correctly-rounded to within
        // ~1 ULP, not exactly-rounded -- so comparing against it at 50 bits (only ~3 bits of
        // margin below f64's own precision) occasionally fails from ordinary libm rounding
        // noise over enough random trials, unrelated to `PREC`. 40 bits leaves a comfortable
        // margin while still being a far stricter tolerance (~1 part in 10^12) than any
        // practical use of `Prec::sin` needs.
        let mut rng = rand::rng();
        for _ in 0..100 {
            let x_f64 = rng.random_range(-10.0 * PI_F64..=10.0 * PI_F64);
            let x = to_fbig(x_f64);
            let expected = to_fbig(x_f64.sin());
            let result = PREC.sin(&x);
            assert!(
                approx_eq(&result, &expected, 40),
                "sin({}) = {}, expected {}, diff = {}",
                x_f64,
                result,
                expected,
                (&result - &expected).abs()
            );
        }
    }

    #[test]
    fn test_cos_random() {
        // See test_sin_random's comment: same tolerance-margin reasoning applies here.
        let mut rng = rand::rng();
        for _ in 0..100 {
            let x_f64 = rng.random_range(-10.0 * PI_F64..=10.0 * PI_F64);
            let x = to_fbig(x_f64);
            let expected = to_fbig(x_f64.cos());
            let result = PREC.cos(&x);
            assert!(
                approx_eq(&result, &expected, 40),
                "cos({}) = {}, expected {}, diff = {}",
                x_f64,
                result,
                expected,
                (&result - &expected).abs()
            );
        }
    }
}
