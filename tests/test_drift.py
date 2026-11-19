"""Unit tests for the PSI drift detector."""

from __future__ import annotations

from app.drift import DriftMonitor


def test_drift_low_on_in_distribution_and_high_on_ood() -> None:
    monitor = DriftMonitor(reference_path=None, window_size=50, threshold=0.20)

    # In-distribution-like lengths & high confidences spanning the reference bins
    in_lengths = [10, 30, 30, 50, 50, 80, 80, 120, 180]
    in_confs = [0.60, 0.72, 0.85, 0.92, 0.97, 0.98, 0.99, 0.96, 0.97]
    for _ in range(5):
        monitor.record(in_lengths, in_confs)
    st_normal = monitor.status()
    assert st_normal["drift_detected"] is False

    # Shift to out-of-distribution: very long texts and low confidences
    for _ in range(50):
        monitor.record([450, 480], [0.52, 0.54])
    st_drift = monitor.status()
    assert st_drift["max_psi"] > 0.20
    assert st_drift["drift_detected"] is True
