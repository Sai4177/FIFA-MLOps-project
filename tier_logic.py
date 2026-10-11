"""Rule-based FIFA tier logic.

Used (1) as the label definition for the ML problem and (2) as the small
pure-Python program for CI Practical-1 (unit tests + GitHub Actions).
"""

ELITE_MAX_RANK = 20
STRONG_MAX_RANK = 60
MID_MAX_RANK = 120


def rank_to_tier(rank):
    """Map a FIFA world rank (1 = best) to a tier name."""
    if isinstance(rank, bool) or not isinstance(rank, (int, float)):
        raise TypeError("rank must be a number")
    if rank < 1:
        raise ValueError("rank must be >= 1")
    if rank <= ELITE_MAX_RANK:
        return "Elite"
    if rank <= STRONG_MAX_RANK:
        return "Strong"
    if rank <= MID_MAX_RANK:
        return "Mid"
    return "Developing"


if __name__ == "__main__":
    print("Rank 5   ->", rank_to_tier(5))
    print("Rank 150 ->", rank_to_tier(150))
