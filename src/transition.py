"""Standalone transition detection module for (position, value) curves.

Implements multiple detection methods for identifying transition points in
signal curves (e.g., probe confidence over token positions). Each method
segments a trace into three phases: reasoning, transition, and post-hoc.

This module has NO dependencies on project-specific code. It only uses
numpy, scipy, and ruptures, making it reusable for any (position, value)
curve analysis.

Detection methods:
    - derivative: Savitzky-Golay smoothed first derivative, onset detection
    - pelt: PELT change-point detection (ruptures library)
    - binseg: Binary segmentation change-point detection (ruptures library)
    - sigmoid: Logistic sigmoid curve fitting (scipy)

Usage:
    >>> import numpy as np
    >>> from src.transition import detect_transitions
    >>> x = np.arange(200)
    >>> y = 1.0 / (1.0 + np.exp(-0.1 * (x - 100)))
    >>> results = detect_transitions(x, y)
    >>> results['sigmoid']['classification']
    'clear'
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import ruptures as rpt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration values
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = {
    "derivative": {
        "window_length": 21,
        "polyorder": 3,
        "slope_threshold": 0.01,
    },
    "pelt": {
        "model": "l2",
        "penalty": 10,
        "min_size": 5,
    },
    "binseg": {
        "model": "l2",
        "n_bkps": 2,
        "min_size": 5,
    },
    "sigmoid": {
        "r2_thresholds": [0.75, 0.80, 0.85, 0.90, 0.95],
        "maxfev": 5000,
    },
    "classification": {
        "no_signal_max": 0.70,
        "clear_plateau_min": 0.90,
        "plateau_tail_n": 10,
    },
}


def _get_config(config: dict | None, method: str) -> dict:
    """Merge user config with defaults for a specific method."""
    defaults = _DEFAULT_CONFIG.get(method, {}).copy()
    if config is not None and method in config:
        defaults.update(config[method])
    return defaults


def _get_classification_config(config: dict | None) -> dict:
    """Get classification thresholds from config or defaults."""
    defaults = _DEFAULT_CONFIG["classification"].copy()
    if config is not None and "classification" in config:
        defaults.update(config["classification"])
    return defaults


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def _classify_trace(
    values: np.ndarray,
    transition_found: bool,
    plateau_found: bool,
    cls_config: dict,
) -> str:
    """Classify a trace into one of three categories.

    Args:
        values: Raw signal values.
        transition_found: Whether a transition point was detected.
        plateau_found: Whether a plateau region was detected.
        cls_config: Classification thresholds dict with keys:
            no_signal_max, clear_plateau_min, plateau_tail_n.

    Returns:
        One of: "no-signal", "clear", "noisy-signal".
    """
    no_signal_max = cls_config["no_signal_max"]

    if np.max(values) < no_signal_max:
        return "no-signal"
    if transition_found and plateau_found:
        return "clear"
    return "noisy-signal"


def _detect_plateau(
    values: np.ndarray,
    cls_config: dict,
) -> bool:
    """Check if the tail of the signal forms a plateau above threshold.

    Args:
        values: Raw signal values.
        cls_config: Classification config with clear_plateau_min and
            plateau_tail_n keys.

    Returns:
        True if the mean of the last N values exceeds the plateau threshold.
    """
    tail_n = cls_config["plateau_tail_n"]
    min_val = cls_config["clear_plateau_min"]
    if len(values) < tail_n:
        tail_n = len(values)
    tail = values[-tail_n:]
    return float(np.mean(tail)) >= min_val


# ---------------------------------------------------------------------------
# Segment builder
# ---------------------------------------------------------------------------

def _build_segments(
    positions: np.ndarray,
    transition_idx: int | None,
    plateau_idx: int | None,
) -> list[tuple[Any, Any, str]]:
    """Build three-phase segment list from detected boundaries.

    Args:
        positions: Array of position values (not indices).
        transition_idx: Index into positions array for transition onset,
            or None if no transition detected.
        plateau_idx: Index into positions array for plateau onset,
            or None if no plateau detected.

    Returns:
        List of (start_position, end_position, label) tuples where label
        is one of: "reasoning", "transition", "post-hoc".
    """
    start = positions[0]
    end = positions[-1]

    if transition_idx is None:
        return [(start, end, "reasoning")]

    transition_pos = positions[transition_idx]

    if plateau_idx is None:
        return [
            (start, transition_pos, "reasoning"),
            (transition_pos, end, "transition"),
        ]

    plateau_pos = positions[plateau_idx]
    return [
        (start, transition_pos, "reasoning"),
        (transition_pos, plateau_pos, "transition"),
        (plateau_pos, end, "post-hoc"),
    ]


# ---------------------------------------------------------------------------
# Detection methods
# ---------------------------------------------------------------------------

def _derivative_method(
    positions: np.ndarray,
    values: np.ndarray,
    config: dict | None,
) -> dict:
    """Derivative-based transition detection using Savitzky-Golay filter.

    Computes a smoothed first derivative of the signal to find the first
    point where the slope exceeds a threshold. Smoothing is applied ONLY
    to the derivative computation; plateau detection uses raw values.

    Args:
        positions: Array of position values.
        values: Array of raw signal values.
        config: Optional config dict. Relevant keys under "derivative":
            window_length (int), polyorder (int), slope_threshold (float).

    Returns:
        Dict with keys: transition_point, plateau_onset, segments,
        classification, method_specific.
    """
    cfg = _get_config(config, "derivative")
    cls_cfg = _get_classification_config(config)

    window_length = cfg["window_length"]
    polyorder = cfg["polyorder"]
    slope_threshold = cfg["slope_threshold"]

    n = len(values)

    # Adjust window_length for short signals: must be odd and <= n, minimum 5
    if window_length > n:
        window_length = n if n % 2 == 1 else n - 1
    if window_length < 5:
        window_length = 5
    if window_length > n:
        # Signal is too short even for minimum window
        return {
            "transition_point": None,
            "plateau_onset": None,
            "segments": _build_segments(positions, None, None),
            "classification": "noisy-signal" if np.max(values) >= cls_cfg["no_signal_max"] else "no-signal",
            "method_specific": {
                "derivative": np.zeros_like(values),
                "note": "signal too short for derivative computation",
            },
        }

    # Ensure polyorder < window_length
    if polyorder >= window_length:
        polyorder = window_length - 1

    # Compute smoothed first derivative
    # delta = mean spacing between positions for proper scaling
    dx = np.mean(np.diff(positions.astype(float)))
    if dx == 0:
        dx = 1.0

    derivative = savgol_filter(values, window_length=window_length, polyorder=polyorder, deriv=1, delta=dx)

    # Find first point where slope exceeds threshold
    transition_idx = None
    for i in range(len(derivative)):
        if derivative[i] >= slope_threshold:
            transition_idx = i
            break

    # Plateau detection on RAW values
    plateau_found = _detect_plateau(values, cls_cfg)

    # Find plateau onset index: last point before tail where values first
    # stay above plateau threshold continuously to the end
    plateau_idx = None
    if plateau_found:
        tail_n = cls_cfg["plateau_tail_n"]
        if tail_n > n:
            tail_n = n
        plateau_min = cls_cfg["clear_plateau_min"]
        # Walk backwards from end to find where values first drop below threshold
        for i in range(n - 1, -1, -1):
            if values[i] < plateau_min:
                plateau_idx = i + 1 if i + 1 < n else i
                break
        if plateau_idx is None:
            # All values above threshold -- plateau is entire signal
            plateau_idx = 0

    classification = _classify_trace(values, transition_idx is not None, plateau_found, cls_cfg)

    return {
        "transition_point": int(positions[transition_idx]) if transition_idx is not None else None,
        "plateau_onset": int(positions[plateau_idx]) if plateau_idx is not None else None,
        "segments": _build_segments(positions, transition_idx, plateau_idx),
        "classification": classification,
        "method_specific": {
            "derivative": derivative,
            "slope_threshold": slope_threshold,
            "window_length": window_length,
            "polyorder": polyorder,
        },
    }


def _pelt_method(
    positions: np.ndarray,
    values: np.ndarray,
    config: dict | None,
) -> dict:
    """PELT change-point detection on raw signal values.

    Uses the ruptures library's Pelt algorithm with L2 cost model.
    Operates on raw values with no smoothing.

    Args:
        positions: Array of position values.
        values: Array of raw signal values.
        config: Optional config dict. Relevant keys under "pelt":
            model (str), penalty (float), min_size (int).

    Returns:
        Dict with keys: transition_point, plateau_onset, segments,
        classification, method_specific.
    """
    cfg = _get_config(config, "pelt")
    cls_cfg = _get_classification_config(config)

    model = cfg["model"]
    penalty = cfg.get("penalty", 10)
    min_size = cfg["min_size"]

    # ruptures expects 2D signal
    signal = values.reshape(-1, 1)

    try:
        algo = rpt.Pelt(model=model, min_size=min_size).fit(signal)
        change_points_raw = algo.predict(pen=penalty)
    except Exception as exc:
        logger.warning("PELT failed (n=%d, pen=%s): %s", len(values), penalty, exc)
        classification = "no-signal" if np.max(values) < cls_cfg["no_signal_max"] else "noisy-signal"
        return {
            "transition_point": None,
            "plateau_onset": None,
            "segments": _build_segments(positions, None, None),
            "classification": classification,
            "method_specific": {
                "change_points_indices": [],
                "change_points_positions": [],
                "penalty": penalty,
                "model": model,
                "n_change_points": 0,
                "error": str(exc),
            },
        }

    # ruptures always appends len(signal) as last element; filter it out
    change_points = [cp for cp in change_points_raw if cp < len(signal)]

    # First change point is candidate transition
    transition_idx = change_points[0] if change_points else None

    # For plateau: use last change point if multiple exist
    plateau_idx = change_points[-1] if len(change_points) > 1 else None

    # Plateau detection on RAW values
    plateau_found = _detect_plateau(values, cls_cfg)

    # If plateau detected but no plateau_idx from change points, estimate
    if plateau_found and plateau_idx is None and transition_idx is not None:
        plateau_min = cls_cfg["clear_plateau_min"]
        for i in range(len(values) - 1, -1, -1):
            if values[i] < plateau_min:
                plateau_idx = i + 1 if i + 1 < len(values) else i
                break

    classification = _classify_trace(
        values,
        transition_found=transition_idx is not None,
        plateau_found=plateau_found,
        cls_config=cls_cfg,
    )

    return {
        "transition_point": int(positions[transition_idx]) if transition_idx is not None else None,
        "plateau_onset": int(positions[plateau_idx]) if plateau_idx is not None else None,
        "segments": _build_segments(positions, transition_idx, plateau_idx),
        "classification": classification,
        "method_specific": {
            "change_points_indices": change_points,
            "change_points_positions": [int(positions[cp]) for cp in change_points] if change_points else [],
            "penalty": penalty,
            "model": model,
            "n_change_points": len(change_points),
        },
    }


def _binseg_method(
    positions: np.ndarray,
    values: np.ndarray,
    config: dict | None,
) -> dict:
    """Binary segmentation change-point detection on raw signal values.

    Uses the ruptures library's Binseg algorithm with L2 cost model.
    Operates on raw values with no smoothing.

    Args:
        positions: Array of position values.
        values: Array of raw signal values.
        config: Optional config dict. Relevant keys under "binseg":
            model (str), n_bkps (int), min_size (int).

    Returns:
        Dict with keys: transition_point, plateau_onset, segments,
        classification, method_specific.
    """
    cfg = _get_config(config, "binseg")
    cls_cfg = _get_classification_config(config)

    model = cfg["model"]
    n_bkps = cfg["n_bkps"]
    min_size = cfg["min_size"]

    # ruptures expects 2D signal
    signal = values.reshape(-1, 1)

    try:
        algo = rpt.Binseg(model=model, min_size=min_size).fit(signal)
        change_points_raw = algo.predict(n_bkps=n_bkps)
    except Exception as exc:
        # Signal too short for requested breakpoints or other ruptures error
        logger.warning("BinSeg failed (n=%d, n_bkps=%d): %s", len(values), n_bkps, exc)
        classification = "no-signal" if np.max(values) < cls_cfg["no_signal_max"] else "noisy-signal"
        return {
            "transition_point": None,
            "plateau_onset": None,
            "segments": _build_segments(positions, None, None),
            "classification": classification,
            "method_specific": {
                "change_points_indices": [],
                "change_points_positions": [],
                "n_bkps_requested": n_bkps,
                "n_change_points": 0,
                "model": model,
                "error": str(exc),
            },
        }

    # Filter out the terminal element (len(signal))
    change_points = [cp for cp in change_points_raw if cp < len(signal)]

    # First change point is candidate transition
    transition_idx = change_points[0] if change_points else None

    # Last change point (if distinct from first) is plateau onset candidate
    plateau_idx = change_points[-1] if len(change_points) > 1 else None

    # Plateau detection on RAW values
    plateau_found = _detect_plateau(values, cls_cfg)

    # If plateau detected but no plateau_idx from change points, estimate
    if plateau_found and plateau_idx is None and transition_idx is not None:
        plateau_min = cls_cfg["clear_plateau_min"]
        for i in range(len(values) - 1, -1, -1):
            if values[i] < plateau_min:
                plateau_idx = i + 1 if i + 1 < len(values) else i
                break

    classification = _classify_trace(
        values,
        transition_found=transition_idx is not None,
        plateau_found=plateau_found,
        cls_config=cls_cfg,
    )

    return {
        "transition_point": int(positions[transition_idx]) if transition_idx is not None else None,
        "plateau_onset": int(positions[plateau_idx]) if plateau_idx is not None else None,
        "segments": _build_segments(positions, transition_idx, plateau_idx),
        "classification": classification,
        "method_specific": {
            "change_points_indices": change_points,
            "change_points_positions": [int(positions[cp]) for cp in change_points] if change_points else [],
            "n_bkps_requested": n_bkps,
            "n_change_points": len(change_points),
            "model": model,
        },
    }


def _sigmoid(x: np.ndarray, L: float, k: float, x0: float, b: float) -> np.ndarray:
    """Logistic sigmoid function: L / (1 + exp(-k * (x - x0))) + b."""
    return L / (1.0 + np.exp(-k * (x - x0))) + b


def _sigmoid_method(
    positions: np.ndarray,
    values: np.ndarray,
    config: dict | None,
) -> dict:
    """Sigmoid curve fitting for transition detection.

    Fits a logistic sigmoid L/(1 + exp(-k*(x - x0))) + b to the signal.
    The midpoint x0 is the transition point and k controls sharpness.

    Args:
        positions: Array of position values.
        values: Array of raw signal values.
        config: Optional config dict. Relevant keys under "sigmoid":
            r2_thresholds (list[float]), maxfev (int).

    Returns:
        Dict with keys: transition_point, plateau_onset, segments,
        classification, method_specific.
    """
    cfg = _get_config(config, "sigmoid")
    cls_cfg = _get_classification_config(config)

    r2_thresholds = cfg["r2_thresholds"]
    maxfev = cfg["maxfev"]

    x = positions.astype(float)
    y = values.astype(float)

    # Initial guesses
    L_init = float(np.max(y) - np.min(y))
    k_init = 1.0 / (float(np.std(x)) + 1e-8)
    x0_init = float(np.median(x))
    b_init = float(np.min(y))

    p0 = [L_init, k_init, x0_init, b_init]

    # Bounds
    bounds_lower = [0.0, 0.0, float(x.min()), -0.1]
    bounds_upper = [1.5, np.inf, float(x.max()), 0.5]

    try:
        popt, _ = curve_fit(
            _sigmoid,
            x, y,
            p0=p0,
            bounds=(bounds_lower, bounds_upper),
            maxfev=maxfev,
        )

        L_fit, k_fit, x0_fit, b_fit = popt

        # Compute R-squared
        y_pred = _sigmoid(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-12))

        # Report results for each threshold
        threshold_results = {}
        for thresh in r2_thresholds:
            threshold_results[thresh] = {
                "passes": bool(r2 >= thresh),
                "r2": float(r2),
            }

        # Primary threshold for classification: use 0.85 (middle of range)
        # or the median threshold if custom thresholds provided
        primary_threshold = 0.85
        if r2_thresholds:
            sorted_thresholds = sorted(r2_thresholds)
            primary_threshold = sorted_thresholds[len(sorted_thresholds) // 2]

        sigmoid_fits = bool(r2 >= primary_threshold)

        # Transition point is x0 (midpoint of sigmoid)
        # Find the index closest to x0 for segment building
        transition_idx = int(np.argmin(np.abs(positions.astype(float) - x0_fit)))

        # Plateau detection on RAW values
        plateau_found = _detect_plateau(values, cls_cfg)

        # Estimate plateau onset: where sigmoid reaches 90% of its max
        plateau_idx = None
        if plateau_found:
            plateau_level = b_fit + 0.9 * L_fit
            above_plateau = np.where(y_pred >= plateau_level)[0]
            if len(above_plateau) > 0:
                plateau_idx = int(above_plateau[0])
            else:
                # Fallback: walk backwards on raw values
                plateau_min = cls_cfg["clear_plateau_min"]
                for i in range(len(values) - 1, -1, -1):
                    if values[i] < plateau_min:
                        plateau_idx = i + 1 if i + 1 < len(values) else i
                        break

        classification = _classify_trace(
            values,
            transition_found=sigmoid_fits,
            plateau_found=plateau_found,
            cls_config=cls_cfg,
        )

        return {
            "transition_point": int(positions[transition_idx]) if sigmoid_fits else None,
            "plateau_onset": int(positions[plateau_idx]) if plateau_idx is not None else None,
            "segments": _build_segments(
                positions,
                transition_idx if sigmoid_fits else None,
                plateau_idx,
            ),
            "classification": classification,
            "method_specific": {
                "L": float(L_fit),
                "k": float(k_fit),
                "x0": float(x0_fit),
                "b": float(b_fit),
                "r2": float(r2),
                "primary_threshold": primary_threshold,
                "threshold_results": threshold_results,
                "fitted_values": y_pred,
                "sharpness": float(k_fit),
            },
        }

    except (RuntimeError, ValueError, TypeError) as exc:
        logger.warning("Sigmoid fit failed: %s", exc)

        classification = "no-signal" if np.max(values) < cls_cfg["no_signal_max"] else "noisy-signal"

        return {
            "transition_point": None,
            "plateau_onset": None,
            "segments": _build_segments(positions, None, None),
            "classification": classification,
            "method_specific": {
                "error": str(exc),
                "r2": None,
                "threshold_results": {},
                "sharpness": None,
            },
        }


# ---------------------------------------------------------------------------
# Method dispatch
# ---------------------------------------------------------------------------

_METHOD_DISPATCH = {
    "derivative": _derivative_method,
    "pelt": _pelt_method,
    "binseg": _binseg_method,
    "sigmoid": _sigmoid_method,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_transitions(
    positions: np.ndarray,
    values: np.ndarray,
    methods: list[str] | None = None,
    config: dict | None = None,
) -> dict:
    """Run all specified detection methods on a (positions, values) curve.

    This is the primary entry point for transition detection. It runs each
    requested method independently and returns structured results for each.

    Args:
        positions: Array of token positions (integers or floats). Must be
            the same length as values.
        values: Array of signal values (e.g., probe confidence in [0, 1]).
            Must be the same length as positions.
        methods: List of method names to run. Defaults to all four methods:
            ["derivative", "pelt", "binseg", "sigmoid"].
        config: Optional dict of method-specific parameters. Keys should be
            method names mapping to parameter dicts. If None, uses defaults.
            Example: {"derivative": {"slope_threshold": 0.02},
                      "classification": {"no_signal_max": 0.65}}

    Returns:
        Dict keyed by method name, each containing:
            - transition_point: int or None (position value, not index)
            - plateau_onset: int or None (position value)
            - segments: list of (start, end, label) tuples
            - classification: "clear" | "no-signal" | "noisy-signal"
            - method_specific: dict of method-specific metadata

    Raises:
        ValueError: If positions and values have different lengths, or if
            an unknown method name is requested.
    """
    if methods is None:
        methods = ["derivative", "pelt", "binseg", "sigmoid"]

    positions = np.asarray(positions)
    values = np.asarray(values, dtype=float)

    if len(positions) != len(values):
        raise ValueError(
            f"positions and values must have the same length, "
            f"got {len(positions)} and {len(values)}"
        )

    if len(positions) == 0:
        raise ValueError("positions and values must be non-empty")

    unknown = [m for m in methods if m not in _METHOD_DISPATCH]
    if unknown:
        raise ValueError(f"Unknown detection methods: {unknown}")

    results = {}
    for method_name in methods:
        method_fn = _METHOD_DISPATCH[method_name]
        results[method_name] = method_fn(positions, values, config)

    return results


def sweep_pelt_penalties(
    positions: np.ndarray,
    values: np.ndarray,
    penalties: list[float] | None = None,
    model: str = "l2",
    min_size: int = 5,
) -> dict[float, dict]:
    """Run PELT with multiple penalty values and return results for each.

    Convenience function that calls detect_transitions with the PELT method
    for each penalty value in the sweep list.

    Args:
        positions: Array of position values.
        values: Array of signal values.
        penalties: List of penalty values to sweep. Defaults to
            [1, 5, 10, 20, 50, 100].
        model: Cost model for PELT. Default "l2".
        min_size: Minimum segment size. Default 5.

    Returns:
        Dict keyed by penalty value, each containing the PELT method result
        dict (transition_point, classification, etc.).
    """
    if penalties is None:
        penalties = [1, 5, 10, 20, 50, 100]

    positions = np.asarray(positions)
    values = np.asarray(values, dtype=float)

    results = {}
    for pen in penalties:
        config = {
            "pelt": {
                "model": model,
                "penalty": pen,
                "min_size": min_size,
            },
        }
        full_result = detect_transitions(
            positions, values, methods=["pelt"], config=config
        )
        results[pen] = full_result["pelt"]

    return results
