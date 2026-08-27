// Copyright (c) 2024-2025 Shun Yamamoto and Nobuyuki Yoshioka, and IBM
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

use dashu_float::round::mode::HalfEven;
use dashu_float::FBig;
use dashu_int::IBig;

use crate::common::Prec;
use crate::grid_op::GridOp;
use crate::odgp::{
    first_solve_scaled_odgp, solve_scaled_odgp, solve_scaled_odgp_with_parity_k_ne_0,
};
use crate::region::{Ellipse, Interval, Rectangle};
use crate::ring::{DOmega, DRootTwo};

/// See Remark 5.4, page 5, Ross and Selinger arXiv:1403.2975v3
pub trait Region {
    /// An ellipse bounding the region A
    fn ellipse(&self) -> Ellipse;

    /// Returns `true` if `u` is inside the region A
    fn inside(&self, u: &DOmega) -> bool;

    /// Intersection of the line with the region A
    /// Given L(t) = u + tv, return endpoints of the interval {t | L(t) ∈ A}
    fn intersect(&self, u: &DOmega, v: &DOmega) -> Option<(FBig<HalfEven>, FBig<HalfEven>)>;
}

pub fn solve_tdgp<'a>(
    set_a: &'a impl Region,
    set_b: &'a impl Region,
    op_g: &'a GridOp,
    bbox_a: &'a Rectangle,
    bbox_b: &'a Rectangle,
    k: i64,
    _verbose: bool,
) -> Option<impl Iterator<Item = DOmega> + 'a> {
    let prec = bbox_a.x.prec;
    let alpha0 = first_solve_scaled_odgp(&bbox_a.x, &bbox_b.x, k + 1)?;
    let _k_ibig = IBig::from(k);
    let dx = DRootTwo::power_of_inv_sqrt2(k);
    let op_g_inv_result = op_g.inv();

    let op_g_inv = op_g_inv_result.unwrap();
    let zero_droottwo = DRootTwo::from_int(IBig::ZERO);
    let v = op_g_inv * DOmega::from_droottwo_vector(&dx, &zero_droottwo, k);
    let v_conj_sq2 = v.conj_sq2().clone();

    let bbox_a_new = bbox_a
        .y
        .fatten(&(bbox_a.y.width() / prec.ib(IBig::from(10000))));
    let bbox_b_new = bbox_b
        .y
        .fatten(&(bbox_b.y.width() / prec.ib(IBig::from(10000))));
    let sol_y = solve_scaled_odgp(&bbox_a_new, &bbox_b_new, k + 1);

    let sol_sufficient = sol_y.flat_map(move |y| {
        newproc(
            prec,
            y,
            set_a,
            set_b,
            op_g,
            alpha0.clone(),
            v_conj_sq2.clone(),
            k,
        )
        .into_iter()
        .flatten()
    });

    let solutions = sol_sufficient
        .map(|z| op_g.inv().unwrap() * z)
        .filter(|z| set_a.inside(z) && set_b.inside(z.conj_sq2()));
    Some(solutions)
}

// `prec` becoming explicit (no ambient/global precision anywhere in this crate) pushed this
// already-large private helper from 7 to 8 parameters.
#[allow(clippy::too_many_arguments)]
fn newproc<'a>(
    prec: Prec,
    beta: DRootTwo,
    set_a: &'a impl Region,
    set_b: &'a impl Region,
    op_g: &'a GridOp,
    alpha0: DRootTwo,
    //    alpha0: &'a DRootTwo,
    v_conj_sq2: DOmega,
    //    v_conj_sq2: &'a DOmega,
    k: i64,
) -> Option<impl Iterator<Item = DOmega> + 'a> {
    let alpha0 = alpha0.clone();
    let droot_zero = DRootTwo::from_int(IBig::ZERO);
    let dx = DRootTwo::power_of_inv_sqrt2(k);
    let z0 = op_g.inv().unwrap() * DOmega::from_droottwo_vector(&alpha0, &beta, k + 1);
    let v = op_g.inv().unwrap() * DOmega::from_droottwo_vector(&dx, &droot_zero, k);

    let t_a = set_a.intersect(&z0, &v);
    let t_b = set_b.intersect(z0.conj_sq2(), &v_conj_sq2);
    if t_a.is_none() || t_b.is_none() {
        return None;
    }
    let (t_a, t_b) = (t_a.unwrap(), t_b.unwrap());

    let parity = (&beta - &alpha0).mul_by_sqrt2_power_renewing_denomexp(k);
    let (mut int_a, mut int_b) = (
        Interval::new(t_a.0, t_a.1, prec),
        Interval::new(t_b.0, t_b.1, prec),
    );
    let dt_a = get_dt_x(prec, k, &int_b);
    let dt_b = get_dt_x(prec, k, &int_a);
    int_a = int_a.fatten(&dt_a);
    int_b = int_b.fatten(&dt_b);

    let sol_t = solve_scaled_odgp_with_parity_k_ne_0(int_a, int_b, 1, &parity);
    let sol_x = sol_t.map(move |alpha| alpha * dx.clone() + alpha0.clone());
    let sol_xx = sol_x.map(move |alpha| DOmega::from_droottwo_vector(&alpha, &beta, k));
    Some(sol_xx)
}

fn get_dt_x(prec: Prec, k: i64, int_y: &Interval) -> FBig<HalfEven> {
    let ten = prec.ib(IBig::from(10));
    let shift_k = IBig::from(1) << (k as usize);
    let width_product = shift_k * int_y.width();
    let max_val = {
        if ten > width_product {
            &ten
        } else {
            &width_product
        }
    };
    prec.fb(&ten / max_val)
}
