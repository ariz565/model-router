"""Pure-stdlib statistical primitives for the Gaussian copula engine —
`normal_cdf`, `inverse_normal_cdf`, and `cholesky_decomposition`.

**Why these three, and why they're enough.** A Gaussian copula generates
CORRELATED values by (1) drawing independent standard-normal noise, (2)
correlating it by multiplying with the Cholesky factor of the target
correlation matrix, then (3) mapping each correlated normal back through its
own marginal distribution. That pipeline needs exactly these three primitives
and nothing else — no scipy, no numpy. `engines/copula.py` is where they get
assembled into a generator; this module owns only the math, so it can be
tested (and trusted) independently of anything about tables or columns.

**Why pure stdlib rather than a dependency.** Same reasoning as
`profiling.py`'s "no numpy, no pandas": this platform's default path has to
run inside a customer's environment with nothing installed beyond Python
itself, and `StatisticalEngine` already proves the zero-dependency default is
viable. A copula engine that pulls in scipy just for `erf`, `ppf`, and `cholesky`
would make "no real data leaves and no exotic dependency tree" a claim that's
only true for the OTHER engine.
"""

from __future__ import annotations

import math

__all__ = ["normal_cdf", "inverse_normal_cdf", "cholesky_decomposition", "NotPositiveDefiniteError"]


def normal_cdf(x: float) -> float:
    """Φ(x) — the standard normal CDF, via the identity Φ(x) = ½(1 + erf(x/√2)).

    `math.erf` is stdlib since Python 3.2, which is what makes this a one-line
    function instead of a numerical-integration routine."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# Acklam's rational approximation to Φ⁻¹ — the standard reference algorithm for
# a probit function with no special-function library available. Coefficients
# are the published constants; see Peter Acklam's "An algorithm for computing
# the inverse normal cumulative distribution function" (2003).
_ACKLAM_A = (
    -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
    1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00,
)
_ACKLAM_B = (
    -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
    6.680131188771972e+01, -1.328068155288572e+01,
)
_ACKLAM_C = (
    -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
    -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00,
)
_ACKLAM_D = (
    7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
    3.754408661907416e+00,
)
_ACKLAM_LOW = 0.02425


def inverse_normal_cdf(p: float) -> float:
    """Φ⁻¹(p) — the probit function, accurate to ~1.15e-9 from Acklam's
    approximation alone, then sharpened to full double precision with a single
    Halley refinement step (using `normal_cdf`, which IS exact to machine
    precision via `erf`). One correction step is standard practice for this
    algorithm and costs one `erf` call, which is negligible next to sampling an
    entire dataset.

    Raises rather than clamping at the boundaries: `p <= 0` or `p >= 1` means
    the caller passed a probability that cannot correspond to a finite normal
    quantile, and returning ±inf silently would poison every downstream
    correlation computation with a value that looks numeric but isn't."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in the open interval (0, 1), got {p}")

    high = 1.0 - _ACKLAM_LOW
    if p < _ACKLAM_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((_ACKLAM_C[0]*q+_ACKLAM_C[1])*q+_ACKLAM_C[2])*q+_ACKLAM_C[3])*q+_ACKLAM_C[4])*q+_ACKLAM_C[5]) / \
            ((((_ACKLAM_D[0]*q+_ACKLAM_D[1])*q+_ACKLAM_D[2])*q+_ACKLAM_D[3])*q+1.0)
    elif p <= high:
        q = p - 0.5
        r = q * q
        x = (((((_ACKLAM_A[0]*r+_ACKLAM_A[1])*r+_ACKLAM_A[2])*r+_ACKLAM_A[3])*r+_ACKLAM_A[4])*r+_ACKLAM_A[5])*q / \
            (((((_ACKLAM_B[0]*r+_ACKLAM_B[1])*r+_ACKLAM_B[2])*r+_ACKLAM_B[3])*r+_ACKLAM_B[4])*r+1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((_ACKLAM_C[0]*q+_ACKLAM_C[1])*q+_ACKLAM_C[2])*q+_ACKLAM_C[3])*q+_ACKLAM_C[4])*q+_ACKLAM_C[5]) / \
             ((((_ACKLAM_D[0]*q+_ACKLAM_D[1])*q+_ACKLAM_D[2])*q+_ACKLAM_D[3])*q+1.0)

    # One step of Halley's method on f(x) = Φ(x) - p, using φ(x) as f'(x) and
    # f''(x)/f'(x) = -x for the standard normal density. This is the standard
    # refinement Acklam's own writeup recommends for full machine precision.
    error = normal_cdf(x) - p
    if error != 0.0:
        density = math.exp(-x * x / 2.0) / math.sqrt(2.0 * math.pi)
        x -= error / (density * (1.0 + x * error / (2.0 * density * density)))
    return x


class NotPositiveDefiniteError(ValueError):
    """The matrix is not positive-definite even after ridge regularization."""


def cholesky_decomposition(
    matrix: list[list[float]], *, max_attempts: int = 8, initial_jitter: float = 1e-10,
) -> list[list[float]]:
    """Lower-triangular `L` such that `L @ L.T == matrix`, with an automatic
    ridge-regularization fallback for a matrix that is not quite positive-
    definite.

    **Why the fallback is necessary, not defensive-programming theater.** The
    input here is an EMPIRICAL correlation matrix built from real column pairs
    (see `engines/copula.py`). Empirical correlation matrices over more than a
    couple of columns are routinely not exactly positive-definite — rounding,
    the sparse-correlation threshold in `profiling.py` (pairs below it are
    recorded as exactly 0.0 rather than their tiny true value), or plain
    estimation noise can all produce a matrix with a slightly negative
    eigenvalue. Refusing to decompose such a matrix would make the copula
    engine unusable on exactly the realistic inputs it exists to handle.

    The fix is the standard one: add a small multiple of the identity (`jitter`
    on the diagonal) and retry, growing the jitter geometrically until the
    decomposition succeeds or `max_attempts` is exhausted. This nudges the
    matrix towards positive-definiteness by the smallest amount that works,
    rather than picking one large fixed jitter that would flatten genuine
    correlations on every input, well-conditioned or not.

    The defaults cap the largest jitter tried at ~1e-3 — small next to a
    correlation matrix's unit diagonal. That ceiling is deliberate: a matrix
    that still isn't positive-definite after a 1e-3 nudge isn't suffering from
    estimation noise, it is internally inconsistent (e.g. `corr(A,B)=0.9`,
    `corr(B,C)=0.9`, `corr(A,C)=-0.9` cannot exist simultaneously for any real
    variables). Raising past that ceiling would mean growing jitter into
    values comparable to the diagonal itself, which "fixes" the matrix by
    silently overwriting the very correlations this engine exists to
    preserve — worse than failing loudly."""
    if not matrix or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    jitter = 0.0
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return _cholesky_once(matrix, jitter)
        except NotPositiveDefiniteError as error:
            last_error = error
            jitter = initial_jitter if jitter == 0.0 else jitter * 10.0
    raise NotPositiveDefiniteError(
        f"matrix is not positive-definite after {max_attempts} ridge-regularization "
        f"attempts (final jitter={jitter:.2e})"
    ) from last_error


def _cholesky_once(matrix: list[list[float]], jitter: float) -> list[list[float]]:
    n = len(matrix)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            partial = sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                diagonal = matrix[i][i] + jitter - partial
                if diagonal <= 0.0:
                    raise NotPositiveDefiniteError(
                        f"non-positive pivot at ({i}, {i}) with jitter={jitter:.2e}"
                    )
                lower[i][j] = math.sqrt(diagonal)
            else:
                lower[i][j] = (matrix[i][j] - partial) / lower[j][j]
    return lower
