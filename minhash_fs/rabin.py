"""
Rabin fingerprinting [32] over GF(2).

A fingerprint function is defined by a random irreducible polynomial p(x) of
degree k over GF(2). The input byte string is read as a polynomial over GF(2)
and the fingerprint is the remainder of its division by p(x).

Polynomials are represented as Python integers, bit i standing for the
coefficient of x^i, so that addition is XOR and multiplication is carry-less.

The collision probability for two strings of shingle length l is bounded by
    P_collision <= max(|s1|, |s2|) / (2^(l-1))                      (Equation 8)
which is why a large degree (61 by default, as in the experiments) is used.
"""

from __future__ import annotations

import numpy as np

__all__ = ["is_irreducible", "random_irreducible_polynomial", "RabinFingerprint"]


def _poly_degree(p: int) -> int:
    return p.bit_length() - 1


def _poly_mulmod(a: int, b: int, mod: int) -> int:
    """Carry-less multiplication of a and b modulo the polynomial ``mod``."""
    deg_mod = _poly_degree(mod)
    result = 0
    while b:
        if b & 1:
            result ^= a
        b >>= 1
        a <<= 1
        if _poly_degree(a) >= deg_mod:
            a ^= mod << (_poly_degree(a) - deg_mod)
    return result


def _poly_powmod(base: int, exponent: int, mod: int) -> int:
    result = 1
    while exponent:
        if exponent & 1:
            result = _poly_mulmod(result, base, mod)
        base = _poly_mulmod(base, base, mod)
        exponent >>= 1
    return result


def _poly_gcd(a: int, b: int) -> int:
    while b:
        a, b = b, _poly_mod(a, b)
    return a


def _poly_mod(a: int, b: int) -> int:
    deg_b = _poly_degree(b)
    while a and _poly_degree(a) >= deg_b:
        a ^= b << (_poly_degree(a) - deg_b)
    return a


def is_irreducible(p: int) -> bool:
    """Rabin's irreducibility test for a polynomial over GF(2)."""
    n = _poly_degree(p)
    if n < 1 or not (p & 1):          # a reducible polynomial has the factor x
        return False
    # x^(2^n) = x  (mod p)
    x_pow = _poly_powmod(2, 1 << n, p)
    if x_pow != 2:
        return False
    # gcd(x^(2^(n/q)) - x, p) = 1 for every prime divisor q of n
    for q in _prime_divisors(n):
        h = _poly_powmod(2, 1 << (n // q), p) ^ 2
        if h and _poly_degree(_poly_gcd(p, h)) != 0:
            return False
    return True


def _prime_divisors(n: int):
    divisors, d = set(), 2
    while d * d <= n:
        while n % d == 0:
            divisors.add(d)
            n //= d
        d += 1
    if n > 1:
        divisors.add(n)
    return divisors


def random_irreducible_polynomial(degree: int, rng: np.random.Generator) -> int:
    """Draw a uniformly random irreducible polynomial of the requested degree."""
    if degree < 2:
        raise ValueError("degree must be >= 2")
    high = 1 << degree
    while True:
        candidate = high | int(rng.integers(0, high)) | 1   # monic, non-zero constant
        if is_irreducible(candidate):
            return candidate


class RabinFingerprint:
    """A single Rabin fingerprint function.

    Parameters
    ----------
    polynomial : int
        Irreducible polynomial over GF(2) defining the function.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> f = RabinFingerprint(random_irreducible_polynomial(61, rng))
    >>> f(b"hello") == f(b"hello")
    True
    """

    __slots__ = ("polynomial", "degree")

    def __init__(self, polynomial: int):
        self.polynomial = int(polynomial)
        self.degree = _poly_degree(self.polynomial)

    def __call__(self, data: bytes) -> int:
        """Fingerprint of a byte string: the remainder of data / p(x)."""
        acc = 0
        deg = self.degree
        poly = self.polynomial
        for byte in data:
            acc = (acc << 8) ^ byte
            while acc.bit_length() - 1 >= deg:
                acc ^= poly << (acc.bit_length() - 1 - deg)
        return acc

    def __repr__(self) -> str:          # pragma: no cover - cosmetic only
        return f"RabinFingerprint(degree={self.degree}, poly=0x{self.polynomial:x})"


def make_fingerprint_family(n_functions: int, degree: int = 61,
                            random_state: int | None = 42) -> list[RabinFingerprint]:
    """Draw ``n_functions`` independent Rabin fingerprint functions.

    The experiments in the paper use ``n_functions=400``; the same family (i.e.
    the same ``random_state``) is reused across all datasets so that the sketch
    generation is identical everywhere.
    """
    rng = np.random.default_rng(random_state)
    polynomials, seen = [], set()
    while len(polynomials) < n_functions:
        p = random_irreducible_polynomial(degree, rng)
        if p not in seen:
            seen.add(p)
            polynomials.append(p)
    return [RabinFingerprint(p) for p in polynomials]
