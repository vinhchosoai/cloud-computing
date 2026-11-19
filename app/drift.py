"""Sliding-window Population Stability Index (PSI) drift detector."""

from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path
from threading import Lock

import numpy as np


def calculate_psi(expected_probs: list[float], actual_values: list[float], edges: list[float]) -> float:
    if len(actual_values) < 10:
        return 0.0
    counts, _ = np.histogram(actual_values, bins=edges)
    total = counts.sum()
    if total == 0:
        return 0.0
    actual_probs = (counts + 1e-6) / (total + 1e-6 * len(counts))
    psi = 0.0
    for exp_p, act_p in zip(expected_probs, actual_probs):
        e = max(float(exp_p), 1e-6)
        a = max(float(act_p), 1e-6)
        psi += (a - e) * math.log(a / e)
    return max(0.0, float(psi))


class DriftMonitor:
    def __init__(self, reference_path: Path | None, window_size: int = 200, threshold: float = 0.20):
        self.window_size = window_size
        self.threshold = threshold
        self._lengths: deque[float] = deque(maxlen=window_size)
        self._confidences: deque[float] = deque(maxlen=window_size)
        self._lock = Lock()

        if reference_path and reference_path.exists():
            ref = json.loads(reference_path.read_text(encoding="utf-8"))
            self.length_edges = ref["text_length_edges"]
            self.length_probs = ref["text_length_probs"]
            self.conf_edges = ref["confidence_edges"]
            self.conf_probs = ref["confidence_probs"]
        else:
            # Sensible fallback reference distributions if file is absent (e.g. unit tests)
            self.length_edges = [0, 20, 40, 60, 100, 150, 250, 513]
            self.length_probs = [0.10, 0.22, 0.24, 0.24, 0.14, 0.05, 0.01]
            self.conf_edges = [0.5, 0.65, 0.8, 0.9, 0.95, 1.01]
            self.conf_probs = [0.05, 0.07, 0.10, 0.13, 0.65]

    def record(self, lengths: list[int], confidences: list[float]) -> tuple[float, float, float]:
        with self._lock:
            for length in lengths:
                self._lengths.append(float(length))
            for conf in confidences:
                self._confidences.append(float(conf))
            return self._compute_locked()

    def status(self) -> dict[str, float | int | bool]:
        with self._lock:
            length_psi, conf_psi, max_psi = self._compute_locked()
            return {
                "window_size": self.window_size,
                "current_samples": len(self._lengths),
                "length_psi": round(length_psi, 4),
                "confidence_psi": round(conf_psi, 4),
                "max_psi": round(max_psi, 4),
                "threshold": self.threshold,
                "drift_detected": max_psi > self.threshold,
            }

    def _compute_locked(self) -> tuple[float, float, float]:
        length_psi = calculate_psi(self.length_probs, list(self._lengths), self.length_edges)
        conf_psi = calculate_psi(self.conf_probs, list(self._confidences), self.conf_edges)
        return length_psi, conf_psi, max(length_psi, conf_psi)

