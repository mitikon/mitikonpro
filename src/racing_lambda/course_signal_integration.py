"""General integration rule for performance detail and course specialization.

This does not contain Ibis-specific horse values. It defines how the two
independent signals coexist without increasing total evidence weight.
"""
from __future__ import annotations


def split_performance_course_weight(
    performance_score: float,
    course_score: float,
    *,
    original_weight: float = 0.12,
    course_share: float = 0.40,
) -> float:
    """Split an existing performance-detail budget instead of adding weight.

    Default keeps 60% of the 0.12 budget on generic clock/closing performance
    and allocates 40% to independent target-course specialization. This avoids
    simply inflating the model because a new feature was discovered.
    """
    if not 0.0 <= performance_score <= 1.0 or not 0.0 <= course_score <= 1.0:
        raise ValueError("scores must be in [0, 1]")
    if not 0.0 <= course_share <= 1.0:
        raise ValueError("course_share must be in [0, 1]")
    return original_weight * ((1.0 - course_share) * performance_score + course_share * course_score)
