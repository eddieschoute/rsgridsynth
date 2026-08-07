"""Ad-hoc differential check of rsgridsynth's Stage 1-3 Rust implementations against
real Qualtran output, for the same (theta, epsilon, q) inputs. Exploratory verification
tool, not a committed CI fixture generator (see ../generate_qualtran_fixtures for that).

Usage: uv run tools/differential_check/check.py
"""

import math

import qualtran.rotation_synthesis as rs

PI = math.pi
Q = 1 - 2**-7  # matches Rust's exact_q(7)

THETAS = [0.1, 3 * PI / 32, 5 * PI / 32, 7 * PI / 32, PI / 3, PI / 6]
EPSILONS = [1e-4, 1e-6, 1e-8]
BUDGET = {1e-4: (40, 90), 1e-6: (50, 120), 1e-8: (60, 150)}

print("protocol,theta,epsilon,t_count,diamond_distance,q_or_p")
for eps in EPSILONS:
    dps, max_n = BUDGET[eps]
    config = rs.with_dps(dps)
    for theta in THETAS:
        try:
            ch = rs.mixed_diagonal_protocol(theta, eps, max_n, config)
            print(
                f"mixed_diagonal,{theta},{eps},{float(ch.expected_num_ts(config))},"
                f"{float(ch.diamond_norm_distance_to_rz(theta, config))},{float(ch.probability)}"
            )
        except Exception as e:
            print(f"mixed_diagonal,{theta},{eps},ERROR,{type(e).__name__}:{e},")
        try:
            ch = rs.fallback_protocol(theta, eps, Q, max_n, config)
            q = float(ch.success_probability(config))
            print(
                f"fallback,{theta},{eps},{float(ch.expected_num_ts(config))},"
                f"{float(ch.diamond_norm_distance_to_rz(theta, config))},{q}"
            )
        except Exception as e:
            print(f"fallback,{theta},{eps},ERROR,{type(e).__name__}:{e},")
        try:
            ch = rs.mixed_fallback_protocol(theta, eps, Q, max_n, config)
            print(
                f"mixed_fallback,{theta},{eps},{float(ch.expected_num_ts(config))},"
                f"{float(ch.diamond_norm_distance_to_rz(theta, config))},{float(ch.probability)}"
            )
        except Exception as e:
            print(f"mixed_fallback,{theta},{eps},ERROR,{type(e).__name__}:{e},")
