"""General integration rule for performance detail and course specialization.

This does not contain race-specific horse values. It defines how the two
independent signals coexist without increasing total evidence weight.
"""
from __future__ import annotations


def split_performance_course_weight(
    performance_score: float,
    course_score: float,
    *,
    original_weight: float = 0.12,
    course_share: float = 0.40,
    course_reliability: float = 1.0,
) -> float:
    """Split an existing performance-detail budget instead of adding weight.

    ``course_share`` remains the fixed maximum allocation (default 40%).  The
    actual allocation is attenuated by pre-race evidence reliability, so sparse
    or absent course history cannot displace generic performance at full force.
    No target-race result is used in this attenuation.
    """
    if not 0.0 <= performance_score <= 1.0 or not 0.0 <= course_score <= 1.0:
        raise ValueError("scores must be in [0, 1]")
    if not 0.0 <= course_share <= 1.0:
        raise ValueError("course_share must be in [0, 1]")
    if not 0.0 <= course_reliability <= 1.0:
        raise ValueError("course_reliability must be in [0, 1]")

    effective_course_share = course_share * course_reliability
    return original_weight * (
        (1.0 - effective_course_share) * performance_score
        + effective_course_share * course_score
    )
