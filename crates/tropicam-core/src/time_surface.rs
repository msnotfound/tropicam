//! `T(x, y) <- max(T(x, y), t)` -- tropical addition, one event at a time.
//!
//! This is where the native port actually earns its keep, and the reason is
//! worth stating precisely because it is a project claim, not a micro-
//! optimisation.
//!
//! In NumPy, a correct scatter-max over a batch needs either `np.maximum.at`
//! (unbuffered, slow) or a lexsort-and-reduce -- **O(n log n)** in the batch
//! size, because vectorised fancy-index assignment resolves duplicate pixels
//! by last write rather than by max.
//!
//! In Rust the same operation is a load, a compare, and a conditional store.
//! No sort, no allocation, no batching required:
//!
//! ```text
//! let slot = &mut self.t[idx];
//! if ev.t > *slot { *slot = ev.t; }
//! ```
//!
//! That is **strictly constant work per event**, independent of batch size,
//! sensor resolution, and stream history -- which is the O(1)-per-event bound
//! the project scope commits to. The NumPy path cannot make that claim at
//! all; it was always a prototype standing in for this.

/// Timestamps wrap at 2^32 us; ages above this alias (~71.6 minutes).
pub const MAX_UNAMBIGUOUS_AGE_US: u32 = u32::MAX;

/// Per-pixel most-recent-event timestamps, one plane per polarity.
///
/// Pre-allocated, flat, contiguous, row-major. Never reallocated after
/// construction, so ingest performs no allocation whatsoever.
pub struct TimeSurface {
    t: Vec<u32>,
    width: usize,
    height: usize,
    planes: usize,
    plane_stride: usize,
    now: u32,
}

/// Outcome of ingesting a batch: enough to verify the work bound in tests.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IngestStats {
    /// Events consumed.
    pub consumed: u64,
    /// Events that actually advanced a pixel's timestamp.
    pub updated: u64,
    /// Events dropped for being out of bounds.
    pub out_of_bounds: u64,
}

impl TimeSurface {
    pub fn new(height: usize, width: usize, planes: usize) -> Self {
        assert!(height > 0 && width > 0 && planes > 0);
        let plane_stride = height * width;
        Self {
            t: vec![0u32; plane_stride * planes],
            width,
            height,
            planes,
            plane_stride,
            now: 0,
        }
    }

    #[inline(always)]
    fn index(&self, x: u16, y: u16, p: u8) -> Option<usize> {
        let (x, y, p) = (x as usize, y as usize, p as usize);
        if x >= self.width || y >= self.height || p >= self.planes {
            return None;
        }
        Some(p * self.plane_stride + y * self.width + x)
    }

    /// Consume one event. Constant work; no allocation, no sort, no branchy
    /// bookkeeping. Order of arrival within a batch does not matter.
    #[inline(always)]
    pub fn push(&mut self, ev: crate::Event) -> bool {
        let (x, y, t, p) = (ev.x(), ev.y(), ev.t(), ev.p());
        match self.index(x, y, p) {
            None => false,
            Some(idx) => {
                // SAFETY: index() bounds-checked against the allocation.
                let slot = unsafe { self.t.get_unchecked_mut(idx) };
                if t > *slot {
                    *slot = t;
                }
                if t > self.now {
                    self.now = t;
                }
                true
            }
        }
    }

    /// Consume a batch. Exactly `push` in a loop -- kept separate only so the
    /// FFI boundary is crossed once per batch instead of once per event.
    pub fn extend(&mut self, events: &[crate::Event]) -> IngestStats {
        let mut stats = IngestStats {
            consumed: events.len() as u64,
            ..Default::default()
        };
        for &ev in events {
            let t = ev.t();
            match self.index(ev.x(), ev.y(), ev.p()) {
                None => stats.out_of_bounds += 1,
                Some(idx) => {
                    let slot = unsafe { self.t.get_unchecked_mut(idx) };
                    if t > *slot {
                        *slot = t;
                        stats.updated += 1;
                    }
                    if t > self.now {
                        self.now = t;
                    }
                }
            }
        }
        stats
    }

    pub fn reset(&mut self) {
        self.t.fill(0);
        self.now = 0;
    }

    // -- read -----------------------------------------------------------

    #[inline]
    pub fn timestamps(&self) -> &[u32] {
        &self.t
    }

    pub fn plane(&self, p: usize) -> &[u32] {
        let start = p * self.plane_stride;
        &self.t[start..start + self.plane_stride]
    }

    /// Polarity-agnostic surface: the max across planes.
    ///
    /// Tropical addition, which is the only merge respecting the semiring.
    /// Averaging the planes would be the clock-time mistake in miniature.
    pub fn merged(&self) -> Vec<u32> {
        let mut out = self.plane(0).to_vec();
        for p in 1..self.planes {
            for (o, &v) in out.iter_mut().zip(self.plane(p)) {
                if v > *o {
                    *o = v;
                }
            }
        }
        out
    }

    /// Age per pixel via wrapping unsigned subtraction -- no rolling epoch,
    /// hence no periodic O(N) stop-the-world sweep in the p99 plot.
    pub fn age(&self, now: u32) -> Vec<u32> {
        self.t.iter().map(|&v| now.wrapping_sub(v)).collect()
    }

    #[inline]
    pub fn now(&self) -> u32 {
        self.now
    }
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }
    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }
    #[inline]
    pub fn planes(&self) -> usize {
        self.planes
    }
    #[inline]
    pub fn nbytes(&self) -> usize {
        self.t.len() * core::mem::size_of::<u32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Event;

    #[test]
    fn max_not_last_write_wins() {
        // The footgun, in Rust: oldest timestamp arrives last and must lose.
        let mut ts = TimeSurface::new(8, 8, 2);
        ts.extend(&[
            Event::new(3, 4, 500, 1),
            Event::new(3, 4, 900, 1),
            Event::new(3, 4, 100, 1),
        ]);
        assert_eq!(ts.plane(1)[4 * 8 + 3], 900);
    }

    #[test]
    fn order_independent() {
        let a = {
            let mut s = TimeSurface::new(16, 16, 2);
            s.extend(&[
                Event::new(1, 1, 10, 1),
                Event::new(1, 1, 90, 1),
                Event::new(2, 2, 50, 0),
            ]);
            s.timestamps().to_vec()
        };
        let b = {
            let mut s = TimeSurface::new(16, 16, 2);
            s.extend(&[
                Event::new(2, 2, 50, 0),
                Event::new(1, 1, 90, 1),
                Event::new(1, 1, 10, 1),
            ]);
            s.timestamps().to_vec()
        };
        assert_eq!(a, b);
    }

    #[test]
    fn wrapping_age_across_the_boundary() {
        let mut ts = TimeSurface::new(4, 4, 2);
        ts.extend(&[Event::new(0, 0, u32::MAX - 99, 1)]);
        // now has wrapped past zero; true age is 350us.
        let age = ts.age(250);
        assert_eq!(age[ts.plane_stride], 350);
    }

    #[test]
    fn polarity_planes_independent_and_merge_is_max() {
        let mut ts = TimeSurface::new(8, 8, 2);
        ts.extend(&[Event::new(2, 3, 400, 1), Event::new(2, 3, 800, 0)]);
        let i = 3 * 8 + 2;
        assert_eq!(ts.plane(1)[i], 400);
        assert_eq!(ts.plane(0)[i], 800);
        assert_eq!(ts.merged()[i], 800);
    }

    #[test]
    fn out_of_bounds_dropped_not_wrapped() {
        let mut ts = TimeSurface::new(8, 8, 2);
        let stats = ts.extend(&[Event::new(99, 0, 5, 1), Event::new(0, 99, 5, 1)]);
        assert_eq!(stats.out_of_bounds, 2);
        assert_eq!(ts.timestamps().iter().copied().max(), Some(0));
    }

    #[test]
    fn footprint_matches_budget() {
        // Doc section 7.3: ~7.4 MB at 1280x720, two 32-bit polarity planes.
        let ts = TimeSurface::new(720, 1280, 2);
        assert!(ts.nbytes() > 7_000_000 && ts.nbytes() < 7_500_000);
    }
}
