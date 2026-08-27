// Copyright (c) 2024-2025 Shun Yamamoto and Nobuyuki Yoshioka, and IBM
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

use crate::common::Prec;
use dashu_float::round::mode::HalfEven;
use dashu_float::FBig;
use dashu_int::IBig;
use nalgebra::{Matrix2, Vector2};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

use std::fmt::{Debug, Display, Formatter};

#[derive(Debug, Clone, PartialEq)]
pub struct Interval {
    pub l: FBig<HalfEven>,
    pub r: FBig<HalfEven>,
    pub prec: Prec,
}

impl Interval {
    pub fn new(l: FBig<HalfEven>, r: FBig<HalfEven>, prec: Prec) -> Self {
        Self { l, r, prec }
    }

    pub fn width(&self) -> FBig<HalfEven> {
        &self.r - &self.l
    }

    pub fn fatten(&self, eps: &FBig<HalfEven>) -> Self {
        Self {
            l: &self.l - eps,
            r: &self.r + eps,
            prec: self.prec,
        }
    }

    pub fn within(&self, x: &FBig<HalfEven>) -> bool {
        (self.l <= *x) && (*x <= self.r)
    }

    pub fn scale(&self, factor: &FBig<HalfEven>) -> Self {
        let zero = self.prec.ib(IBig::ZERO);
        if *factor >= zero {
            Self {
                l: &self.l * factor,
                r: &self.r * factor,
                prec: self.prec,
            }
        } else {
            Self {
                l: &self.r * factor,
                r: &self.l * factor,
                prec: self.prec,
            }
        }
    }

    pub fn sub_ref(&self, x: &FBig<HalfEven>) -> Self {
        Interval {
            l: &self.l - x,
            r: &self.r - x,
            prec: self.prec,
        }
    }
}

impl fmt::Display for Interval {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{}, {}]", self.l, self.r)
    }
}

// Interval + Interval
impl Add for Interval {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            l: &self.l + &rhs.l,
            r: &self.r + &rhs.r,
            prec: self.prec,
        }
    }
}

// Interval + f64
impl Add<f64> for Interval {
    type Output = Self;
    fn add(self, rhs: f64) -> Self {
        let rhs_float = self.prec.fb(FBig::<HalfEven>::try_from(rhs).unwrap());
        Self {
            l: &self.l + &rhs_float,
            r: &self.r + &rhs_float,
            prec: self.prec,
        }
    }
}

// Interval + FBig
impl Add<FBig<HalfEven>> for Interval {
    type Output = Self;
    fn add(self, rhs: FBig<HalfEven>) -> Self {
        Self {
            l: &self.l + &rhs,
            r: &self.r + &rhs,
            prec: self.prec,
        }
    }
}

// f64 + Interval
impl Add<Interval> for f64 {
    type Output = Interval;
    fn add(self, rhs: Interval) -> Interval {
        rhs + self
    }
}

// FBig + Interval
impl Add<Interval> for FBig<HalfEven> {
    type Output = Interval;
    fn add(self, rhs: Interval) -> Interval {
        rhs + self
    }
}

// Interval - Interval（= Interval + (-Interval)）
impl Sub for Interval {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}

// f64 - Interval
impl Sub<Interval> for f64 {
    type Output = Interval;
    fn sub(self, rhs: Interval) -> Interval {
        -rhs + self
    }
}

// FBig - Interval
impl Sub<Interval> for FBig<HalfEven> {
    type Output = Interval;
    fn sub(self, rhs: Interval) -> Interval {
        -rhs + self
    }
}

// Interval - f64
impl Sub<f64> for Interval {
    type Output = Interval;
    fn sub(self, rhs: f64) -> Interval {
        let rhs_float = self.prec.fb(FBig::<HalfEven>::try_from(rhs).unwrap());
        let prec = self.prec;
        self + (-Interval::new(rhs_float.clone(), rhs_float, prec))
    }
}

// Interval - IBig
impl Sub<IBig> for Interval {
    type Output = Interval;
    fn sub(self, rhs: IBig) -> Interval {
        let prec = self.prec;
        self + (-Interval::new(prec.ib(rhs.clone()), prec.ib(rhs), prec))
    }
}

// Interval - FBig
impl Sub<FBig<HalfEven>> for Interval {
    type Output = Interval;
    fn sub(self, rhs: FBig<HalfEven>) -> Interval {
        let prec = self.prec;
        self + (-Interval::new(rhs.clone(), rhs, prec))
    }
}

// -Interval
impl Neg for Interval {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            l: -self.r,
            r: -self.l,
            prec: self.prec,
        }
    }
}

// Interval * f64
impl Mul<f64> for Interval {
    type Output = Interval;
    fn mul(self, rhs: f64) -> Interval {
        let rhs_fbig = self.prec.fb(FBig::<HalfEven>::try_from(rhs).unwrap());
        if rhs >= 0.0 {
            Interval {
                l: self.l * rhs_fbig.clone(),
                r: self.r * rhs_fbig,
                prec: self.prec,
            }
        } else {
            Interval {
                l: self.r * rhs_fbig.clone(),
                r: self.l * rhs_fbig,
                prec: self.prec,
            }
        }
    }
}

// Interval * FBig
impl Mul<FBig<HalfEven>> for Interval {
    type Output = Interval;
    fn mul(self, rhs: FBig<HalfEven>) -> Interval {
        if rhs >= self.prec.ib(IBig::ZERO) {
            Interval {
                l: self.l * rhs.clone(),
                r: self.r * rhs,
                prec: self.prec,
            }
        } else {
            Interval {
                l: self.r * rhs.clone(),
                r: self.l * rhs,
                prec: self.prec,
            }
        }
    }
}

// f64 * Interval
impl Mul<Interval> for f64 {
    type Output = Interval;
    fn mul(self, rhs: Interval) -> Interval {
        rhs * self
    }
}

// FBig * Interval
impl Mul<Interval> for FBig<HalfEven> {
    type Output = Interval;
    fn mul(self, rhs: Interval) -> Interval {
        rhs * self
    }
}

// Interval / FBig
impl Div<FBig<HalfEven>> for Interval {
    type Output = Interval;
    fn div(self, rhs: FBig<HalfEven>) -> Interval {
        if rhs > self.prec.ib(IBig::ZERO) {
            Interval {
                l: self.l / rhs.clone(),
                r: self.r / rhs,
                prec: self.prec,
            }
        } else {
            Interval {
                l: self.r / rhs.clone(),
                r: self.l / rhs,
                prec: self.prec,
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Rectangle {
    pub x: Interval,
    pub y: Interval,
}

impl Rectangle {
    pub fn new(
        x_l: FBig<HalfEven>,
        x_r: FBig<HalfEven>,
        y_l: FBig<HalfEven>,
        y_r: FBig<HalfEven>,
        prec: Prec,
    ) -> Self {
        Self {
            x: Interval::new(x_l, x_r, prec),
            y: Interval::new(y_l, y_r, prec),
        }
    }

    pub fn area(&self) -> FBig<HalfEven> {
        self.x.width() * self.y.width()
    }

    fn scale(&self, factor: FBig<HalfEven>) -> Self {
        if factor >= self.x.prec.ib(IBig::ZERO) {
            Self {
                x: self.x.scale(&factor),
                y: self.y.scale(&factor),
            }
        } else {
            Self {
                x: self.y.scale(&factor),
                y: self.x.scale(&factor),
            }
        }
    }
}

impl fmt::Display for Rectangle {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}×{}", self.x, self.y)
    }
}

// Rectangle * FBig
impl Mul<FBig<HalfEven>> for Rectangle {
    type Output = Self;

    fn mul(self, rhs: FBig<HalfEven>) -> Self::Output {
        self.scale(rhs)
    }
}

// FBig * Rectangle
impl Mul<Rectangle> for FBig<HalfEven> {
    type Output = Rectangle;

    fn mul(self, rhs: Rectangle) -> Rectangle {
        rhs.scale(self)
    }
}

#[derive(Debug, Clone)]
pub struct Ellipse {
    pub d: Matrix2<FBig<HalfEven>>,
    pub p: Vector2<FBig<HalfEven>>,
    pub prec: Prec,
}

impl Ellipse {
    pub fn new(d: Matrix2<FBig<HalfEven>>, p: Vector2<FBig<HalfEven>>, prec: Prec) -> Self {
        Self { d, p, prec }
    }

    pub fn from(
        d00: FBig<HalfEven>,
        d01: FBig<HalfEven>,
        d10: FBig<HalfEven>,
        d11: FBig<HalfEven>,
        px: FBig<HalfEven>,
        py: FBig<HalfEven>,
        prec: Prec,
    ) -> Self {
        Self {
            d: Matrix2::new(d00, d01, d10, d11),
            p: Vector2::new(px, py),
            prec,
        }
    }

    pub fn px(&self) -> &FBig<HalfEven> {
        &self.p[0]
    }

    pub fn py(&self) -> &FBig<HalfEven> {
        &self.p[1]
    }

    pub fn a(&self) -> &FBig<HalfEven> {
        &self.d[(0, 0)]
    }

    pub fn b(&self) -> &FBig<HalfEven> {
        &self.d[(0, 1)]
    }

    pub fn d(&self) -> &FBig<HalfEven> {
        &self.d[(1, 1)]
    }

    pub fn skew(&self) -> FBig<HalfEven> {
        self.b().powi(IBig::from(2))
    }

    pub fn bias(&self) -> FBig<HalfEven> {
        self.d() / self.a()
    }
    pub fn inside(&self, v: &Vector2<FBig<HalfEven>>) -> bool {
        let x = &v[0] - &self.p[0];
        let y = &v[1] - &self.p[1];
        let a = self.a();
        let b = self.b();
        let d = self.d();
        let x_sq = &x * &x;
        let term1 = a * &x_sq;

        let x_y = &x * &y;
        let two_b = 2 * b;
        let term2 = &two_b * &x_y;

        let y_sq = &y * &y;
        let term3 = d * &y_sq;

        let sum12 = &term1 + &term2;
        let value = &sum12 + &term3;
        value <= (FBig::<HalfEven>::from(1))
    }

    pub fn bbox(&self) -> Rectangle {
        let sqrt_det = self.sqrt_det();
        let w = self.d().sqrt() / &sqrt_det;
        let h = self.a().sqrt() / &sqrt_det;
        let px_minus_w = self.px() - &w;
        let px_plus_w = self.px() + &w;
        let py_minus_h = self.py() - &h;
        let py_plus_h = self.py() + &h;
        Rectangle {
            x: Interval::new(px_minus_w, px_plus_w, self.prec),
            y: Interval::new(py_minus_h, py_plus_h, self.prec),
        }
    }

    pub fn sqrt_det(&self) -> FBig<HalfEven> {
        let det = self.d() * self.a() - self.b().powi(IBig::from(2));
        det.sqrt()
    }

    pub fn area(&self) -> FBig<HalfEven> {
        self.prec.pi() / self.sqrt_det()
    }

    pub fn normalize(&self) -> Self {
        let factor = self.sqrt_det();
        let factor_sqrt = factor.sqrt();
        Ellipse::new(
            self.d.clone() / factor,
            self.p.clone() * factor_sqrt,
            self.prec,
        )
    }
}

// Ellipse * FBig
impl Mul<FBig<HalfEven>> for Ellipse {
    type Output = Ellipse;
    fn mul(self, rhs: FBig<HalfEven>) -> Ellipse {
        let inv_rhs: FBig<HalfEven> = 1 / &rhs;
        let inv_rhs_sq = inv_rhs.powi(IBig::from(2));
        Ellipse::new(self.d * inv_rhs_sq, self.p * rhs, self.prec)
    }
}

// FBig * Ellipse
impl Mul<Ellipse> for FBig<HalfEven> {
    type Output = Ellipse;
    fn mul(self, rhs: Ellipse) -> Ellipse {
        rhs * self
    }
}

// Ellipse / FBig
impl Div<FBig<HalfEven>> for Ellipse {
    type Output = Ellipse;
    fn div(self, rhs: FBig<HalfEven>) -> Ellipse {
        let rhs_sq = rhs.clone().powi(IBig::from(2));
        Ellipse::new(self.d * rhs_sq, self.p / rhs, self.prec)
    }
}

impl Display for Ellipse {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Ellipse {{\n  d: [[{}, {}],\n      [{}, {}]],\n  p: ({}, {})\n}}",
            self.d[(0, 0)].to_decimal().value(),
            self.d[(0, 1)].to_decimal().value(),
            self.d[(1, 0)].to_decimal().value(),
            self.d[(1, 1)].to_decimal().value(),
            self.p[0].to_decimal().value(),
            self.p[1].to_decimal().value(),
        )
    }
}
