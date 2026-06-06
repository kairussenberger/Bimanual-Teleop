"""One-Euro filter tests (spec invariant #3: all tracking input is filtered).

Pins the two properties that make One-Euro the right choice for teleop:
  - steady signal -> heavy smoothing (jitter killed, no DC offset),
  - fast signal   -> light smoothing (lag killed; higher beta = less lag).

    uv run pytest tests/test_filters.py -q
"""
from __future__ import annotations

import numpy as np

from bimanual_teleop.filters import OneEuroFilter, OneEuroVecFilter


HZ = 100.0
DT = 1.0 / HZ


def test_first_sample_passthrough():
    f = OneEuroFilter()
    assert f({"x": 3.0}, 0.0) == {"x": 3.0}
    fv = OneEuroVecFilter()
    assert np.allclose(fv([1.0, 2.0, 3.0], 0.0), [1.0, 2.0, 3.0])


def test_constant_input_no_dc_offset():
    """A constant signal must pass through exactly — a filter that biased steady
    state would make the arm droop away from where the wrist is held."""
    f = OneEuroFilter(mincutoff=1.0, beta=0.5)
    out = None
    for i in range(200):
        out = f({"x": 5.0}, i * DT)
    assert abs(out["x"] - 5.0) < 1e-9


def test_step_response_no_overshoot_and_converges():
    """After a sudden jump the output moves monotonically toward the new value and
    never overshoots it (One-Euro is a convex low-pass)."""
    f = OneEuroFilter(mincutoff=1.0, beta=0.0)
    for i in range(50):
        f({"x": 0.0}, i * DT)
    prev = 0.0
    for i in range(50, 400):
        y = f({"x": 1.0}, i * DT)["x"]
        assert 0.0 <= y <= 1.0          # bounded by [old, new] -> no overshoot
        assert y >= prev - 1e-12        # monotonic toward target
        prev = y
    assert prev > 0.99                  # eventually converges


def test_higher_beta_reduces_lag_on_fast_motion():
    """The defining One-Euro property: on a fast ramp, a larger beta raises the
    cutoff with speed, so the output lags the true signal LESS."""
    slow = OneEuroFilter(mincutoff=1.0, beta=0.0)
    fast = OneEuroFilter(mincutoff=1.0, beta=2.0)
    vel = 5.0                            # units/s — a brisk wrist move
    lag_slow = lag_fast = 0.0
    for i in range(300):
        t = i * DT
        x = vel * t
        ys = slow({"x": x}, t)["x"]
        yf = fast({"x": x}, t)["x"]
        lag_slow, lag_fast = abs(x - ys), abs(x - yf)
    assert lag_fast < lag_slow          # beta cuts lag


def test_steady_jitter_is_attenuated():
    """Zero-mean jitter around a steady value is reduced (the whole point)."""
    rng = np.random.default_rng(0)
    f = OneEuroFilter(mincutoff=1.0, beta=0.0)
    noise = rng.normal(0.0, 1.0, size=500)
    outs = []
    for i, n in enumerate(noise):
        outs.append(f({"x": 10.0 + n}, i * DT)["x"])
    outs = np.array(outs[100:])          # drop warm-up
    assert np.std(outs) < 0.5 * np.std(noise[100:])


def test_vector_matches_dict_channelwise():
    """OneEuroVecFilter must equal OneEuroFilter run per-channel (same math)."""
    rng = np.random.default_rng(1)
    fd = OneEuroFilter(mincutoff=1.5, beta=0.7)
    fv = OneEuroVecFilter(mincutoff=1.5, beta=0.7)
    for i in range(200):
        t = i * DT
        x = rng.normal(size=3) + np.array([i * 0.01, -i * 0.005, 2.0])
        d = fd({"a": x[0], "b": x[1], "c": x[2]}, t)
        v = fv(x, t)
        assert np.allclose([d["a"], d["b"], d["c"]], v, atol=1e-12)


def test_duplicate_timestamp_does_not_blow_up():
    """A repeated/zero-dt timestamp (stale frame) must not divide-by-zero."""
    f = OneEuroFilter()
    f({"x": 0.0}, 1.0)
    y = f({"x": 1.0}, 1.0)              # same t -> dt clamped, finite output
    assert np.isfinite(y["x"])
    fv = OneEuroVecFilter()
    fv([0.0], 2.0)
    assert np.isfinite(fv([1.0], 2.0)).all()
