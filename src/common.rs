// Copyright (c) IBM
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

//! Working precision for this crate's arbitrary-precision arithmetic is **explicit, not
//! ambient**: there is no global or thread-local precision anywhere. Every value that needs
//! one either carries a [`Prec`] field (the per-synthesis structs in `region.rs`,
//! `gridsynth.rs`, `protocol::*`, and the result types in `accuracy.rs`) or is computed via a
//! method on [`Prec`] that seeds a *fresh* value at that precision (here and in `math.rs`:
//! [`Prec::pi`], `Prec::sqrt2`, `Prec::floorsqrt`, ...; the ring accessors in `ring::*` still
//! take a `Prec` parameter for the same reason, since precision is a property of a ring
//! element's float *projection*, not of the exact ring element itself).
//!
//! For an operation on an *existing* value (`sin`, `cos`, `sqrt`, `ln`, ...), call the
//! inherent method on the `FBig` directly (`x.sin()`, not a `Prec` wrapper) -- every `FBig`
//! carries its own precision in its attached `Context`, and dashu propagates precision through
//! `+ - * /` as `Context::max(lhs, rhs)`, so a value already built via `Prec::ib`/`Prec::fb`
//! (or arithmetic on such values) already carries the right context by construction.
//!
//! [`Prec::ib`] is the *only* load-bearing coercion. `FBig::from(IBig)` is precision-0, which
//! `dashu_float` defines as **unlimited** -- useful for exact integers, but "can lead to very
//! huge significands" the moment it enters arithmetic with a bounded value. Every `IBig` seed
//! into this crate's float arithmetic goes through `Prec::ib` for exactly this reason.
//! [`Prec::fb`] (re-pinning an existing `FBig`) is needed far less than it looks -- only at a
//! genuine precision *boundary*, e.g. a fresh `FBig::try_from(f64)` literal, or a value that
//! may have crossed a call at a different precision.
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
}
