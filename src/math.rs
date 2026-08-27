// Copyright (c) 2024-2025 Shun Yamamoto and Nobuyuki Yoshioka, and IBM
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

use crate::common::Prec;
use dashu_float::{round::mode::HalfEven, FBig};
use dashu_int::IBig;
use std::cmp::Ordering;

impl Prec {
    pub fn sqrt2(self) -> FBig<HalfEven> {
        self.ib(IBig::from(2)).sqrt()
    }

    pub fn sign(self, x: FBig<HalfEven>) -> i8 {
        match x.partial_cmp(&self.ib(IBig::ZERO)) {
            Some(Ordering::Greater) => 1,
            Some(Ordering::Less) => -1,
            _ => 0,
        }
    }

    pub fn floorsqrt(self, x: FBig<HalfEven>) -> IBig {
        assert!(x >= self.ib(IBig::ZERO), "Negative input to floorsqrt");
        self.binary_search_sqrt(x)
    }

    fn binary_search_sqrt(self, x: FBig<HalfEven>) -> IBig {
        let mut ok = IBig::ZERO;
        let mut ng: IBig =
            <FBig<HalfEven> as TryInto<IBig>>::try_into(x.ceil()).unwrap() + IBig::from(1);
        while &ng - &ok > IBig::ONE {
            let mid = &ok + (&ng - &ok) / IBig::from(2);
            let mid_sqr = self.ib(&mid * &mid);
            if mid_sqr <= x {
                ok = mid;
            } else {
                ng = mid;
            }
        }
        ok
    }

    pub fn pow_sqrt2(self, k: i64) -> FBig<HalfEven> {
        let k_div_2 = k >> 1;
        let k_mod_2 = k & 1;
        let base =
            self.ib(IBig::ONE) << k_div_2.try_into().expect("Shift amount must fit in usize");
        if k_mod_2 != 0 {
            base * self.sqrt2()
        } else {
            base
        }
    }

    pub fn floorlog(self, x: FBig<HalfEven>, y: FBig<HalfEven>) -> (IBig, FBig<HalfEven>) {
        assert!(x > self.ib(IBig::ZERO), "math domain error");
        let m = self.compute_precision_requirement(&x, &y);
        let pow_y = compute_powers(&y, m);
        self.compute_logarithm(x, y, pow_y)
    }

    fn compute_precision_requirement(self, x: &FBig<HalfEven>, y: &FBig<HalfEven>) -> usize {
        let mut tmp = y.clone();
        let mut m = 0;
        while x >= &tmp || x * &tmp < self.ib(IBig::ONE) {
            tmp = &tmp * &tmp;
            m += 1;
        }
        m
    }

    fn compute_logarithm(
        self,
        x: FBig<HalfEven>,
        y: FBig<HalfEven>,
        pow_y: Vec<FBig<HalfEven>>,
    ) -> (IBig, FBig<HalfEven>) {
        let (mut n, mut r) = if x >= self.ib(IBig::ONE) {
            (IBig::ZERO, x)
        } else {
            (IBig::NEG_ONE, x * pow_y.iter().fold(y, |acc, p| acc * p))
        };

        for p in pow_y {
            n <<= 1;
            if r > p {
                r /= p;
                n += 1;
            }
        }
        (n, r)
    }
}

pub fn ntz(n: &IBig) -> i64 {
    match n.trailing_zeros() {
        Some(k) => k as i64,
        None => 0,
    }
}

pub fn rounddiv(x: IBig, y: &IBig) -> IBig {
    let adjustment = y / 2;
    if (x >= IBig::ZERO && y > &IBig::ZERO) || (x <= IBig::ZERO && y < &IBig::ZERO) {
        (x + adjustment) / y
    } else {
        (x - adjustment) / y
    }
}

fn compute_powers(y: &FBig<HalfEven>, m: usize) -> Vec<FBig<HalfEven>> {
    let mut pow_y = Vec::with_capacity(m);
    pow_y.push(y.clone());
    for i in 1..m {
        let last = &pow_y[i - 1];
        pow_y.push(last * last);
    }
    pow_y.reverse();
    pow_y
}

/// Solves the quadratic equation ax^2 + bx + c = 0 for real roots.
///
/// # Arguments
///
/// - `a`: Coefficient of x^2, assumed to be non-zero for a valid quadratic equation.
/// - `b`: Coefficient of x.
/// - `c`: Constant term.
///
/// # Returns
///
/// An `Option` containing a tuple of two roots if they exist; otherwise, `None` if the roots are not real (i.e., the discriminant is negative).
pub fn solve_quadratic(
    prec: Prec,
    a: &FBig<HalfEven>,
    b: &FBig<HalfEven>,
    c: &FBig<HalfEven>,
) -> Option<(FBig<HalfEven>, FBig<HalfEven>)> {
    let zero = prec.ib(IBig::ZERO);
    let two = prec.ib(IBig::from(2));
    let four = prec.ib(IBig::from(4));

    // Handle degenerate and easy cases
    if a == &zero {
        // Linear: b x + c = 0
        if b == &zero {
            return None;
        }
        let x = -c.clone() / b.clone();
        return Some((x.clone(), x));
    }
    if c == &zero {
        // Roots are 0 and -b/a; avoids 0/0 in the stable branch
        return Some((zero.clone(), -b.clone() / a.clone()));
    }

    // Discriminant
    let disc = b.clone() * b.clone() - four.clone() * a.clone() * c.clone();
    if disc < zero {
        return None;
    }
    let sqrt_d = disc.sqrt();

    // Stable computation:
    // s = -b - sign(b) * sqrt(d)
    // x1 = s / (2a), x2 = (2c) / s
    let s = if b >= &zero {
        -b.clone() - sqrt_d.clone()
    } else {
        -b.clone() + sqrt_d.clone()
    };

    // s == 0 can only happen when c == 0 (handled above), so safe here
    let x1 = s.clone() / (two.clone() * a.clone());
    let x2 = (two * c.clone()) / s;
    if b >= &zero {
        Some((x1, x2))
    } else {
        Some((x2, x1))
    }
}
