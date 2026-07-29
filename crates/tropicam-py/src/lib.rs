//! Python bindings for the TropiCam core.
//!
//! Deliberately thin: no vision logic lives here, only marshalling. Events
//! cross the boundary **zero-copy** -- the Python side hands over a `uint8`
//! view of its packed structured array and Rust reinterprets it in place.
//! Nothing is parsed, converted, or reallocated per event.
//!
//! Built with plain `cargo` (see `scripts/build_rust.sh`); maturin is not
//! required, keeping the project's zero-install-friction property intact.

use pyo3::exceptions::{PyBufferError, PyIndexError, PyValueError};
use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use tropicam_core as core_lib;

/// Per-pixel most-recent-event timestamps, backed by the Rust core.
#[pyclass(name = "TimeSurface", module = "tropicam_rs")]
struct PyTimeSurface {
    inner: core_lib::TimeSurface,
}

#[pymethods]
impl PyTimeSurface {
    #[new]
    #[pyo3(signature = (height, width, n_polarities = 2))]
    fn new(height: usize, width: usize, n_polarities: usize) -> PyResult<Self> {
        if height == 0 || width == 0 || n_polarities == 0 {
            return Err(PyValueError::new_err(
                "height, width and n_polarities must be positive",
            ));
        }
        Ok(Self {
            inner: core_lib::TimeSurface::new(height, width, n_polarities),
        })
    }

    /// Consume a batch of events from a packed `EVENT_DTYPE` buffer.
    ///
    /// Expects the raw bytes of the structured array (pass
    /// ``events.view(np.uint8)``). The 9-byte packed layout is contract-
    /// checked on both sides; a mismatch raises rather than silently
    /// misreading every field.
    fn update_bytes(&mut self, obj: &Bound<'_, PyAny>) -> PyResult<(u64, u64, u64)> {
        // Buffer protocol rather than `&[u8]`: the latter accepts only
        // `bytes`, which would force a copy of every batch. This takes a
        // NumPy uint8 view directly.
        let buf = PyBuffer::<u8>::get(obj)?;
        if !buf.is_c_contiguous() {
            return Err(PyBufferError::new_err(
                "event buffer must be C-contiguous; pass events.view(np.uint8)",
            ));
        }
        let len = buf.item_count();
        if len % core_lib::EVENT_SIZE != 0 {
            return Err(PyBufferError::new_err(format!(
                "buffer of {} bytes is not a whole number of {}-byte events",
                len,
                core_lib::EVENT_SIZE
            )));
        }
        // SAFETY: the buffer is C-contiguous with `len` u8 items, and the GIL
        // is held for the duration of this call, so the exporter cannot
        // resize or free it. We only read.
        let bytes = unsafe { core::slice::from_raw_parts(buf.buf_ptr() as *const u8, len) };
        let events = core_lib::events_from_bytes(bytes)
            .ok_or_else(|| PyBufferError::new_err("malformed event buffer"))?;
        let s = self.inner.extend(events);
        Ok((s.consumed, s.updated, s.out_of_bounds))
    }

    /// Raw timestamps as bytes, shape (planes, height, width) uint32.
    fn timestamps_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, bytemuck_cast(self.inner.timestamps()))
    }

    /// Polarity-agnostic surface (max across planes) as uint32 bytes.
    fn merged_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, bytemuck_cast(&self.inner.merged()))
    }

    /// Ages via wrapping unsigned subtraction, as uint32 bytes.
    fn age_bytes<'py>(&self, py: Python<'py>, now: u32) -> Bound<'py, PyBytes> {
        PyBytes::new(py, bytemuck_cast(&self.inner.age(now)))
    }

    fn plane_bytes<'py>(&self, py: Python<'py>, p: usize) -> PyResult<Bound<'py, PyBytes>> {
        if p >= self.inner.planes() {
            return Err(PyIndexError::new_err("polarity plane out of range"));
        }
        Ok(PyBytes::new(py, bytemuck_cast(self.inner.plane(p))))
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    #[getter]
    fn now(&self) -> u32 {
        self.inner.now()
    }
    #[getter]
    fn width(&self) -> usize {
        self.inner.width()
    }
    #[getter]
    fn height(&self) -> usize {
        self.inner.height()
    }
    #[getter]
    fn n_polarities(&self) -> usize {
        self.inner.planes()
    }
    #[getter]
    fn nbytes(&self) -> usize {
        self.inner.nbytes()
    }

    fn __repr__(&self) -> String {
        format!(
            "TimeSurface({}x{}, {} planes, {:.1} MB, now={}us) [rust]",
            self.inner.height(),
            self.inner.width(),
            self.inner.planes(),
            self.inner.nbytes() as f64 / 1e6,
            self.inner.now()
        )
    }
}

/// `&[u32] -> &[u8]` without pulling in the bytemuck dependency.
fn bytemuck_cast(v: &[u32]) -> &[u8] {
    // SAFETY: u32 has no padding and no invalid bit patterns, so any u32
    // slice is a valid byte slice of 4x the length. Read-only, same lifetime.
    unsafe { core::slice::from_raw_parts(v.as_ptr() as *const u8, core::mem::size_of_val(v)) }
}

/// Byte size of one event; the Python side asserts its dtype matches.
#[pyfunction]
fn event_size() -> usize {
    core_lib::EVENT_SIZE
}

#[pymodule]
fn tropicam_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    core_lib::assert_event_layout();
    m.add_class::<PyTimeSurface>()?;
    m.add_function(wrap_pyfunction!(event_size, m)?)?;
    m.add("__doc__", "TropiCam native core (Rust).")?;
    Ok(())
}
