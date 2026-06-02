"""Drop-jump event detection for OpenCap DJ trials.

Borrows the signal-processing recipe from Couro's
`app/services/ecs/track_mode/utils/gait_analyzer.py`
(gaussian_filter1d + savgol_filter + scipy.signal.find_peaks on the ankle
vertical trajectory), but specialised for a single-impact DJ rather than
a cyclic gait pattern.

DJ structure on the ankle-y signal (image coordinates, y-down):
    standing on box        → low y (high position)
    dropping               → y increasing
    initial contact (IC)   → first major peak in y (foot on ground)
    loading                → ankle dorsiflexes, knee/hip flex
    push-off (TO)          → upward velocity, y starts decreasing
    flight                 → y decreasing fast
    second contact         → next peak (sometimes — depends on capture window)

We anchor ankle-ROM measurement to a window starting at IC and lasting
through the loading-to-pushoff phase (~400ms), so the regression sees
biomech-relevant motion instead of whatever happened across the full clip.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


@dataclass(frozen=True)
class DJEvents:
    """Detected DJ events with frame indices (or None if not detected)."""
    initial_contact: int | None        # frame of first ground contact
    pushoff: int | None                # frame of takeoff (vy turns negative)
    loading_window: tuple[int, int] | None  # (start, end) for ankle ROM


def _smooth(arr: np.ndarray) -> np.ndarray:
    """Same two-stage smoothing Couro uses (gaussian then savgol)."""
    a = gaussian_filter1d(arr, sigma=1.0)
    if len(a) >= 7:
        try:
            a = savgol_filter(a, window_length=7, polyorder=2, mode="interp")
        except Exception:
            pass
    return a


def detect_dj_events_multi_signal(
    signals: list[np.ndarray],
    fps: float,
    *,
    min_loading_ms: float = 100.0,
    max_loading_ms: float = 600.0,
) -> DJEvents:
    """Try multiple signals in order; return the first successful detection.

    Use case: ankle_y is the primary impact signal, but on dead-on cameras
    (Cam2 front-center) ankle keypoint quality drops and we fall back to
    hip-center vertical or any reliable proxy.
    """
    for sig in signals:
        ev = detect_dj_events(sig, fps,
                              min_loading_ms=min_loading_ms,
                              max_loading_ms=max_loading_ms)
        if ev.initial_contact is not None and ev.loading_window is not None:
            return ev
    return DJEvents(None, None, None)


def detect_dj_events(
    ankle_y: np.ndarray,
    fps: float,
    *,
    min_loading_ms: float = 100.0,
    max_loading_ms: float = 600.0,
) -> DJEvents:
    """Detect initial contact + pushoff from ankle vertical trajectory.

    Args:
        ankle_y: (T,) per-frame ankle y-pixel (image coords, y-down).
                 Either right ankle or left — usually doesn't matter for DJ
                 since both contact within ~50ms.
        fps:     video frame rate
    Returns:
        DJEvents with detected frames, or None for any event not found.
    """
    if len(ankle_y) < 10 or not np.any(np.isfinite(ankle_y)):
        return DJEvents(None, None, None)

    finite = np.isfinite(ankle_y)
    if not finite.all():
        # Linearly fill gaps so smoothing doesn't crater
        ankle_y = ankle_y.copy()
        idx = np.arange(len(ankle_y))
        ankle_y[~finite] = np.interp(idx[~finite], idx[finite], ankle_y[finite])

    ay_s = _smooth(ankle_y)
    vy_s = _smooth(np.gradient(ay_s) * fps)  # px/sec

    # Initial contact = first prominent peak in y (foot at lowest image position).
    traj_range = float(np.ptp(ay_s))
    if traj_range <= 0:
        return DJEvents(None, None, None)
    prominence = max(traj_range * 0.10, 3.0)  # 10% of range or 3 px floor
    peaks, _ = signal.find_peaks(
        ay_s,
        prominence=prominence,
        distance=int(0.10 * fps),  # ≥100ms separation
    )
    if peaks.size == 0:
        return DJEvents(None, None, None)
    ic = int(peaks[0])

    # Pushoff = first frame after IC where vy turns clearly negative (foot rising).
    # Search within [IC + min_loading, IC + max_loading].
    search_start = ic + int(min_loading_ms / 1000.0 * fps)
    search_end = min(len(vy_s) - 1, ic + int(max_loading_ms / 1000.0 * fps))
    pushoff: int | None = None
    if search_end > search_start:
        # Smooth threshold: vy < -0.20 * |vy peak| during the search window
        vy_window = vy_s[search_start:search_end]
        if vy_window.size > 0:
            threshold = -0.20 * np.nanmax(np.abs(vy_s))
            crossings = np.where(vy_window < threshold)[0]
            if crossings.size > 0:
                pushoff = int(search_start + crossings[0])

    loading: tuple[int, int] | None = None
    if pushoff is not None:
        loading = (ic, pushoff)
    elif search_end > search_start:
        # Fallback: use full max_loading_ms window if pushoff not found
        loading = (ic, search_end)

    return DJEvents(initial_contact=ic, pushoff=pushoff, loading_window=loading)


def event_anchored_for_metric(
    series: np.ndarray, events: DJEvents,
) -> dict[str, float]:
    """Per-metric event-anchored features from one angle time-series.

    Returns NaN-filled keys when the loading window can't be determined.
    """
    nan_out = {
        "at_contact": float("nan"),
        "rom_loading": float("nan"),
        "min_loading": float("nan"),
        "max_loading": float("nan"),
        "end_loading": float("nan"),
        "loading_duration_frames": float("nan"),
    }
    if events.loading_window is None:
        return nan_out
    a, b = events.loading_window
    a = max(0, int(a)); b = min(len(series), int(b))
    if b - a < 2:
        return nan_out
    window = series[a:b]
    if not np.any(np.isfinite(window)):
        return nan_out
    return {
        "at_contact": float(series[a]) if np.isfinite(series[a]) else float("nan"),
        "rom_loading": float(np.nanmax(window) - np.nanmin(window)),
        "min_loading": float(np.nanmin(window)),
        "max_loading": float(np.nanmax(window)),
        "end_loading": float(series[b - 1]) if np.isfinite(series[b - 1]) else float("nan"),
        "loading_duration_frames": float(b - a),
    }


def event_anchored_features(
    ankle_angle: np.ndarray,
    knee_angle: np.ndarray,
    hip_angle: np.ndarray,
    events: DJEvents,
) -> dict[str, float]:
    """Pull biomech-relevant features from a known loading window.

    All angle arrays are (T,) in degrees, aligned to the same frame index.
    """
    if events.loading_window is None:
        return {
            "ankle_rom_loading": float("nan"),
            "ankle_at_contact": float("nan"),
            "ankle_min_loading": float("nan"),
            "knee_at_contact": float("nan"),
            "knee_rom_loading": float("nan"),
            "hip_at_contact": float("nan"),
            "loading_duration_frames": float("nan"),
        }

    a, b = events.loading_window
    a = max(0, int(a))
    b = min(len(ankle_angle), int(b))
    if b - a < 2:
        return {
            "ankle_rom_loading": float("nan"),
            "ankle_at_contact": float("nan"),
            "ankle_min_loading": float("nan"),
            "knee_at_contact": float("nan"),
            "knee_rom_loading": float("nan"),
            "hip_at_contact": float("nan"),
            "loading_duration_frames": float("nan"),
        }

    aw = ankle_angle[a:b]
    kw = knee_angle[a:b]
    hw = hip_angle[a:b]

    def _nan_range(x):
        if not np.any(np.isfinite(x)):
            return float("nan")
        return float(np.nanmax(x) - np.nanmin(x))

    return {
        "ankle_rom_loading": _nan_range(aw),
        "ankle_at_contact": float(ankle_angle[a]) if np.isfinite(ankle_angle[a]) else float("nan"),
        "ankle_min_loading": float(np.nanmin(aw)) if np.any(np.isfinite(aw)) else float("nan"),
        "knee_at_contact": float(knee_angle[a]) if np.isfinite(knee_angle[a]) else float("nan"),
        "knee_rom_loading": _nan_range(kw),
        "hip_at_contact": float(hip_angle[a]) if np.isfinite(hip_angle[a]) else float("nan"),
        "loading_duration_frames": float(b - a),
    }
