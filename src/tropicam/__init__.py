"""TropiCam -- a deterministic, CPU-only, training-free event-vision engine.

The time surface is built by ``T <- max(T, t)``, which is tropical (max-plus)
addition. TropiCam takes that literally and uses max-plus as the engine's
native arithmetic rather than layering ordinary linear algebra on top.

See ``docs/tropicam-master-doc.md`` for the full project document; ROADMAP.md
tracks what is actually built.
"""

from .events import EVENT_DTYPE, ON, OFF, make_events, event_rate, duration_us
from .time_surface import TimeSurface, scatter_max, MAX_UNAMBIGUOUS_AGE_US
from .metrics import (
    angular_error,
    endpoint_error,
    normal_flow_error,
    regime_agreement,
    check_certificate,
    contrast_rate_floor,
)
from .harness import Record, ResultLog, Sweep, rng_for
from .synthetic import (
    Scene,
    translating_bar,
    moving_dot,
    translating_corner,
    inject_background_noise,
    inject_threshold_mismatch,
)

__version__ = "0.0.1"

__all__ = [
    "EVENT_DTYPE", "ON", "OFF", "make_events", "event_rate", "duration_us",
    "TimeSurface", "scatter_max", "MAX_UNAMBIGUOUS_AGE_US",
    "Scene", "translating_bar", "moving_dot", "translating_corner",
    "inject_background_noise", "inject_threshold_mismatch",
    "angular_error", "endpoint_error", "normal_flow_error",
    "regime_agreement", "check_certificate", "contrast_rate_floor",
    "Record", "ResultLog", "Sweep", "rng_for",
]
