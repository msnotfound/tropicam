//! TropiCam core: deterministic, training-free event-vision primitives.
//!
//! No GPU, no neural network, no allocation on the ingest path. The time
//! surface is built by `T <- max(T, t)`, which is tropical (max-plus)
//! addition; this crate takes that literally and keeps the semiring's
//! arithmetic as the engine's native arithmetic.
//!
//! The Python package under `src/tropicam/` is the reference implementation:
//! it is the executable specification this crate is differentially tested
//! against (`tests/test_rust_parity.py`). Where the two disagree, one of them
//! is a bug -- that is the point of keeping both.

#![deny(unsafe_op_in_unsafe_fn)]

pub mod events;
pub mod time_surface;

pub use events::{assert_event_layout, events_from_bytes, Event, EVENT_SIZE};
pub use time_surface::{IngestStats, TimeSurface, MAX_UNAMBIGUOUS_AGE_US};
