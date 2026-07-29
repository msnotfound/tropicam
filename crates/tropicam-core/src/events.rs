//! Event representation, ABI-compatible with the Python `EVENT_DTYPE`.

/// One event: pixel coordinates, timestamp (microseconds), polarity.
///
/// `#[repr(C, packed)]` is deliberate and load-bearing. NumPy's default
/// structured dtype is packed: 9 bytes, fields at offsets 0/2/4/8. A plain
/// `repr(C)` would align to 12 bytes and silently misread every event that
/// crosses the FFI boundary. `assert_event_layout` pins this down, and the
/// Python side asserts the mirror image, so a layout drift fails loudly on
/// both sides rather than producing plausible garbage.
///
/// Reads are by value; taking a reference into a packed struct is UB, so the
/// accessors below copy. At 9 bytes that is free.
#[repr(C, packed)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Event {
    pub x: u16,
    pub y: u16,
    pub t: u32,
    pub p: u8,
}

impl Event {
    #[inline(always)]
    pub fn new(x: u16, y: u16, t: u32, p: u8) -> Self {
        Self { x, y, t, p }
    }

    #[inline(always)]
    pub fn x(self) -> u16 {
        self.x
    }
    #[inline(always)]
    pub fn y(self) -> u16 {
        self.y
    }
    #[inline(always)]
    pub fn t(self) -> u32 {
        self.t
    }
    #[inline(always)]
    pub fn p(self) -> u8 {
        self.p
    }
}

/// Byte size of one event as seen by both Rust and NumPy.
pub const EVENT_SIZE: usize = 9;

/// Panics unless the struct layout matches NumPy's packed dtype exactly.
pub fn assert_event_layout() {
    assert_eq!(
        core::mem::size_of::<Event>(),
        EVENT_SIZE,
        "Event must be 9 bytes to match NumPy's packed EVENT_DTYPE"
    );
}

/// Reinterpret a raw byte buffer as events.
///
/// # Safety contract (checked, not assumed)
/// Returns `None` unless the buffer length is an exact multiple of
/// `EVENT_SIZE`. `Event` is packed, so there is no alignment requirement --
/// which is precisely why the packed layout is worth the ergonomic cost.
pub fn events_from_bytes(bytes: &[u8]) -> Option<&[Event]> {
    if bytes.len() % EVENT_SIZE != 0 {
        return None;
    }
    // SAFETY: Event is repr(C, packed) with no padding, no invalid bit
    // patterns (all fields are plain integers), and alignment 1. The length
    // check above guarantees the slice covers whole events.
    Some(unsafe {
        core::slice::from_raw_parts(bytes.as_ptr() as *const Event, bytes.len() / EVENT_SIZE)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_matches_numpy() {
        assert_event_layout();
        assert_eq!(core::mem::align_of::<Event>(), 1);
    }

    #[test]
    fn roundtrip_bytes() {
        let evs = [Event::new(1, 2, 300, 1), Event::new(4, 5, 600, 0)];
        let bytes = unsafe {
            core::slice::from_raw_parts(evs.as_ptr() as *const u8, evs.len() * EVENT_SIZE)
        };
        let back = events_from_bytes(bytes).unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!(back[1].t(), 600);
        assert_eq!(back[0].x(), 1);
    }

    #[test]
    fn rejects_ragged_buffer() {
        assert!(events_from_bytes(&[0u8; 10]).is_none());
    }
}
