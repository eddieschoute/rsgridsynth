use crate::common::Prec;
use crate::diophantine::Caches;
use crate::gate::GateSeq;
use dashu_float::round::mode::HalfEven;
use dashu_float::FBig;
use dashu_int::{IBig, UBig};
use rand::{rngs::StdRng, SeedableRng};
use std::str::FromStr;

#[derive(Debug)]
pub struct DiophantineData {
    pub diophantine_timeout: u128,
    pub factoring_timeout: u128,
    pub rng: StdRng,
    /// Memo caches for the number-theoretic search, owned by this config so concurrent
    /// syntheses never share them. See [`Caches`].
    pub caches: Caches,
}

#[derive(Debug)]
pub struct GridSynthConfig {
    pub theta: FBig<HalfEven>,
    pub epsilon: FBig<HalfEven>,
    /// The working precision this config was built for. `gridsynth_gates`/`gridsynth_unitary`
    /// build every per-synthesis value from this -- there is no ambient or global precision
    /// anywhere in this crate, so a config built on one thread and run on another still
    /// computes at the intended precision.
    pub prec: Prec,
    pub verbose: bool,
    pub measure_time: bool,
    pub diophantine_data: DiophantineData,
    pub up_to_phase: bool,
}

/// The result of running the gridsynth algorithm. Accuracy against a target angle is not
/// computed here -- call [`crate::accuracy::AchievedDiamondError::achieved_diamond_error`] on
/// demand instead.
pub struct GridSynthResult {
    /// The synthesized gate sequence. An empty sequence represents the identity.
    pub gates: GateSeq,

    /// The global phase factor.
    pub global_phase: bool,

    /// The working precision this result was synthesized at, carried so
    /// `AchievedDiamondError::achieved_diamond_error` can measure at the right precision
    /// regardless of which thread calls it.
    pub prec: Prec,
}

pub fn parse_decimal_with_exponent(input: &str) -> Option<(IBig, IBig)> {
    let input = input.trim();
    let (sign, body) = if let Some(s) = input.strip_prefix('-') {
        (-1, s)
    } else if let Some(s) = input.strip_prefix('+') {
        (1, s)
    } else {
        (1, input)
    };

    let (base_str, exp_str) = match body.split_once(['e', 'E']) {
        Some((b, e)) => (b, e),
        None => (body, "0"),
    };

    let mut parts = base_str.split('.');
    let int_part = match parts.next() {
        Some(part) => part,
        _ => return None,
    };
    let frac_part: &str = parts.next().unwrap_or_default();
    if parts.next().is_some() {
        return None;
    }
    let digits = format!("{}{}", int_part, frac_part);
    let decimal_digits = frac_part.len() as i32;

    let exponent: i32 = exp_str.parse().ok()?;
    let scale = exponent - decimal_digits;

    let mut numerator = IBig::from_str(&digits).ok()? * sign;
    let mut denominator = IBig::from(1);

    match scale.cmp(&0) {
        std::cmp::Ordering::Greater => {
            numerator *= IBig::from(10u8).pow(scale as usize);
        }
        std::cmp::Ordering::Less => {
            denominator = IBig::from(10u8).pow((-scale) as usize);
        }
        std::cmp::Ordering::Equal => {}
    }

    Some((numerator, denominator))
}

/// Lower bound on working precision. Precision that is too low can cause hard failures
/// (stack overflow, SIGABRT), so the epsilon-derived estimate below is clamped up to this.
/// The target accuracy is deliberately not tied to the working precision beyond this floor.
pub const MIN_PREC_BITS: usize = 16;

/// The working precision (in bits) this crate uses to hit a target accuracy
/// `epsilon = epsilon_num / epsilon_den`. The magic factor 12 safely overapproximates the
/// bits needed per decimal digit. Shared by [`config_from_theta_epsilon`] and the CLI
/// (`main.rs`) so the two formulas cannot drift apart.
///
/// Note: this underflows (`usize` subtraction) for `epsilon >= 1`, a pre-existing limitation
/// carried over unchanged from before this helper was extracted.
pub fn prec_bits_for_epsilon(epsilon_num: &IBig, epsilon_den: &IBig) -> usize {
    let calculated =
        12 * (epsilon_den.ilog(&UBig::from(10u8)) - epsilon_num.ilog(&UBig::from(10u8)));
    calculated.max(MIN_PREC_BITS)
}

/// Creates the default config to easily call the code from other rust packages.
/// `seed` is used to set single RNG that is used through the call to `gridsynth`.
pub fn config_from_theta_epsilon(
    theta: f64,
    epsilon: f64,
    seed: u64,
    verbose: bool,
    up_to_phase: bool,
) -> GridSynthConfig {
    let (theta_num, theta_den) = parse_decimal_with_exponent(&theta.to_string()).unwrap();

    // `theta` is built at a fixed precision independent of the epsilon-derived working
    // precision computed below -- reproducing this crate's historical precision lifecycle
    // (theta used to be built right after a `reset_prec_bits()` to the crate's default,
    // before `set_prec_bits` installed the epsilon-derived value). This is a known
    // inconsistency, preserved here rather than fixed, so this refactor does not also change
    // numeric output; see the module docs on working precision.
    const THETA_PREC: Prec = Prec(1000);
    let theta = THETA_PREC.ib(theta_num) / THETA_PREC.ib(theta_den);
    let (epsilon_num, epsilon_den) = parse_decimal_with_exponent(&epsilon.to_string()).unwrap();
    let prec = Prec(prec_bits_for_epsilon(&epsilon_num, &epsilon_den));
    let epsilon = prec.ib(epsilon_num) / prec.ib(epsilon_den);
    let diophantine_timeout = 200u128;
    let factoring_timeout = 50u128;
    let time = false;

    let rng: StdRng = SeedableRng::seed_from_u64(seed);
    let diophantine_data = DiophantineData {
        diophantine_timeout,
        factoring_timeout,
        rng,
        caches: Caches::default(),
    };

    GridSynthConfig {
        theta,
        epsilon,
        prec,
        verbose,
        measure_time: time,
        diophantine_data,
        up_to_phase,
    }
}
