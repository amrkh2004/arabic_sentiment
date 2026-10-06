"""
Unit tests for Monitoring and Statistical Drift Detection.
"""

import numpy as np

from scripts.monitor_drift import calculate_psi


def test_psi_identical_distributions():
    """
    Verifies that Population Stability Index (PSI) is zero for identical distributions.
    """
    p = np.array([0.5, 0.3, 0.2])
    psi = calculate_psi(p, p)
    assert np.isclose(psi, 0.0, atol=1e-5)


def test_psi_shifted_distributions():
    """
    Verifies that PSI is significantly positive when class distributions diverge.
    """
    baseline = np.array([0.7, 0.2, 0.1])
    shifted = np.array([0.1, 0.2, 0.7])
    psi = calculate_psi(baseline, shifted)
    assert psi > 0.25  # High drift threshold
