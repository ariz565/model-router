"""synthetic/mathstats.py — the pure-stdlib primitives the Gaussian copula
engine is built on.

Tested independently of anything about tables or columns: if `normal_cdf` /
`inverse_normal_cdf` don't round-trip and `cholesky_decomposition` doesn't
reconstruct its input, nothing built on top of them can be trusted, so those
properties are checked directly against known reference values rather than
only indirectly through the engine.
"""

from __future__ import annotations

import math

import pytest

from modelrouter.synthetic.mathstats import (
    NotPositiveDefiniteError,
    cholesky_decomposition,
    inverse_normal_cdf,
    normal_cdf,
)


def test_normal_cdf_of_zero_is_one_half():
    assert normal_cdf(0.0) == pytest.approx(0.5)


def test_normal_cdf_matches_known_reference_quantiles():
    # 1.959963985... is the standard normal 97.5th percentile.
    assert normal_cdf(1.959963985) == pytest.approx(0.975, abs=1e-6)
    assert normal_cdf(-1.959963985) == pytest.approx(0.025, abs=1e-6)


def test_inverse_normal_cdf_of_one_half_is_zero():
    assert inverse_normal_cdf(0.5) == pytest.approx(0.0, abs=1e-12)


def test_inverse_normal_cdf_matches_known_reference_quantiles():
    assert inverse_normal_cdf(0.975) == pytest.approx(1.959963985, abs=1e-8)
    assert inverse_normal_cdf(0.025) == pytest.approx(-1.959963985, abs=1e-8)
    # A far-tail probability, where Acklam's approximation is least accurate
    # before the Halley refinement step.
    assert inverse_normal_cdf(1e-6) == pytest.approx(-4.7534243, abs=1e-6)


def test_inverse_normal_cdf_rejects_boundary_and_out_of_range_probabilities():
    """Clamping at the boundary would return ±inf silently, which would poison
    every downstream correlation computation with a value that looks numeric
    but isn't."""
    for bad in (0.0, 1.0, -0.1, 1.1):
        with pytest.raises(ValueError):
            inverse_normal_cdf(bad)


def test_normal_cdf_and_inverse_round_trip_to_machine_precision():
    """The Halley refinement step in `inverse_normal_cdf` exists specifically to
    push accuracy past Acklam's ~1e-9 approximation error; this asserts it
    actually does."""
    for i in range(1, 1000):
        p = i / 1000
        assert normal_cdf(inverse_normal_cdf(p)) == pytest.approx(p, abs=1e-9)


def test_cholesky_reconstructs_a_positive_definite_matrix_exactly():
    matrix = [[1.0, 0.5], [0.5, 1.0]]
    lower = cholesky_decomposition(matrix)
    n = len(matrix)
    reconstructed = [
        [sum(lower[i][k] * lower[j][k] for k in range(n)) for j in range(n)]
        for i in range(n)
    ]
    for i in range(n):
        for j in range(n):
            assert reconstructed[i][j] == pytest.approx(matrix[i][j], abs=1e-12)


def test_cholesky_is_lower_triangular():
    matrix = [[1.0, 0.3, 0.2], [0.3, 1.0, 0.4], [0.2, 0.4, 1.0]]
    lower = cholesky_decomposition(matrix)
    for i in range(len(lower)):
        for j in range(i + 1, len(lower)):
            assert lower[i][j] == 0.0


def test_cholesky_fixes_a_mildly_non_positive_definite_matrix_via_jitter():
    """A realistic near-PD correlation matrix (estimation noise, not a logical
    contradiction) should be fixed by a small ridge nudge, and the fix should
    barely perturb the original values."""
    matrix = [[1.0, 0.6, 0.59], [0.6, 1.0, 0.6], [0.59, 0.6, 1.0]]
    lower = cholesky_decomposition(matrix)
    n = len(matrix)
    reconstructed = [
        [sum(lower[i][k] * lower[j][k] for k in range(n)) for j in range(n)]
        for i in range(n)
    ]
    for i in range(n):
        for j in range(n):
            assert reconstructed[i][j] == pytest.approx(matrix[i][j], abs=1e-6)


def test_cholesky_raises_rather_than_silently_flattening_an_inconsistent_matrix():
    """corr(A,B)=0.9, corr(B,C)=0.9, corr(A,C)=-0.9 cannot exist simultaneously
    for any real variables. Fixing this would require jitter large enough to
    overwrite the correlations themselves -- worse than failing loudly, which
    is why the jitter ceiling is capped well below that point."""
    contradictory = [[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]]
    with pytest.raises(NotPositiveDefiniteError):
        cholesky_decomposition(contradictory)


def test_cholesky_rejects_a_non_square_input():
    with pytest.raises(ValueError):
        cholesky_decomposition([[1.0, 0.5], [0.5, 1.0], [0.1, 0.1]])


def test_cholesky_handles_the_identity_matrix():
    lower = cholesky_decomposition([[1.0, 0.0], [0.0, 1.0]])
    assert lower == [[1.0, 0.0], [0.0, 1.0]]
