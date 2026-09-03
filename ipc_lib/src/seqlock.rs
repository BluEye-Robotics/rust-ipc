//! The tyndall seqlock, byte for byte.
//!
//! [`SeqLock<T>`] is the shared-memory record `tyndall/ipc/seq_lock.h`
//! defines — one writer, any number of readers, a reader retries while a
//! write is in progress — and the two must agree on every byte of it: the
//! record lives in `/dev/shm` and is written and read by C++ and Rust
//! processes alike. The layout tracked here is tyndall's after its PR #18
//! (commit `c7fadae`, "Fix seq_lock cacheline padding and CACHELINE_BYTES on
//! aarch64", issue #16):
//!
//! ```text
//! offset 0                   seq    unsigned, 4 bytes
//! offset alignof(size_t)     size   size_t
//! offset CACHELINE_BYTES     entry  STORAGE, alignas(CACHELINE_BYTES)
//! sizeof = CACHELINE_BYTES + sizeof(STORAGE), rounded up to CACHELINE_BYTES
//! ```
//!
//! `entry` starts exactly one cacheline in on every ABI, whatever the storage
//! type's own alignment, so the sequence counter and the payload never share
//! a cacheline. [`CACHELINE_BYTES`] is 32 on armv7 (i.MX, Cortex-A9) and 64
//! on aarch64 (Jetson Orin, Cortex-A78AE) and x86_64, as in
//! `tyndall/ipc/smp.h`.
//!
//! Before that tyndall change — and before this crate's 0.3.0 — the header
//! was padded on the assumption that `seq` and `size` pack back to back, and
//! aarch64 used a 32-byte cacheline, so on 64-bit targets `entry` landed at
//! offset 36/40 (aarch64) or 68/72 (x86_64). A 0.2.x writer and a
//! post-#18 C++ reader disagree on every 64-bit target, and vice versa; the
//! two sides have to move together, which is why this is a breaking version.
//!
//! The layout is pinned at compile time: the `const` assertions below (the
//! Rust counterpart of tyndall's `tests/ipc/seq_lock_layout.cpp`) fail
//! `cargo check` for a target whose ABI does not produce it, and the
//! alignment guard on the storage type (tyndall's `static_assert`) fails the
//! build of any program that opens a record over an over-aligned type.

use std::cell::UnsafeCell;
use std::mem::{align_of, offset_of, size_of};
use std::sync::atomic::{AtomicU32, Ordering};

use log::debug;

/// The cacheline size the record layout is built on (`CACHELINE_BYTES` in
/// `tyndall/ipc/smp.h`).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub const CACHELINE_BYTES: usize = 64;

/// armv7 Cortex-A9 (i.MX): 32-byte cache lines.
#[cfg(target_arch = "arm")]
pub const CACHELINE_BYTES: usize = 32;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "arm")))]
compile_error!("ipc_lib: tyndall defines no CACHELINE_BYTES for this target architecture");

/// tyndall's `alignas(CACHELINE_BYTES) STORAGE entry`: the wrapper that
/// forces `entry` onto its own cacheline. Rust's `align` attribute takes a
/// literal, hence one definition per cacheline size; the constant above is
/// the single source the assertions below hold both to.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[repr(C, align(64))]
struct CachelineAligned<T>(T);

#[cfg(target_arch = "arm")]
#[repr(C, align(32))]
struct CachelineAligned<T>(T);

/// `tyndall::seq_lock<STORAGE>` (see the module doc for the layout).
#[repr(C)]
pub struct SeqLock<T: Copy + Default> {
    seq: AtomicU32,
    /// `sizeof(STORAGE)`, stored by every write as tyndall's writer does
    /// (its `ipc_read` tool reads it to know how much to print); no reader
    /// depends on it.
    size: UnsafeCell<usize>,
    entry: CachelineAligned<UnsafeCell<T>>,
}

#[derive(Default)]
pub struct SeqLockState {
    prev_seq: u32,
    has_read_once: bool,
}

// The layout, pinned at compile time for every target this builds for — the
// Rust counterpart of tyndall's tests/ipc/seq_lock_layout.cpp. Storage types
// spanning the alignment classes: 1-byte, 4-byte, and 8-byte (the Vector3
// case that first exposed tyndall issue #16).
const _: () = {
    assert!(align_of::<CachelineAligned<u8>>() == CACHELINE_BYTES);
    assert!(offset_of!(SeqLock<u8>, entry) == CACHELINE_BYTES);
    assert!(offset_of!(SeqLock<f32>, entry) == CACHELINE_BYTES);
    assert!(offset_of!(SeqLock<[f64; 3]>, entry) == CACHELINE_BYTES);
    assert!(align_of::<SeqLock<u8>>() == CACHELINE_BYTES);
    assert!(align_of::<SeqLock<[f64; 3]>>() == CACHELINE_BYTES);
    // sizeof: one cacheline of header, then the entry, rounded up.
    assert!(size_of::<SeqLock<u8>>() == 2 * CACHELINE_BYTES);
    assert!(size_of::<SeqLock<[f64; 3]>>() == 2 * CACHELINE_BYTES);
    // The header itself is as C lays it out: `size` after `seq`, at its own
    // alignment (4 bytes of padding on 64-bit targets, none on armv7).
    assert!(offset_of!(SeqLock<u8>, seq) == 0);
    assert!(offset_of!(SeqLock<u8>, size) == align_of::<usize>());
};

impl<T: Copy + Default> SeqLock<T> {
    /// tyndall's `static_assert(alignof(STORAGE) <= CACHELINE_BYTES)`: a
    /// storage type aligned beyond the cacheline would push `entry` past
    /// offset `CACHELINE_BYTES` and off the C++ layout. An associated const
    /// of a generic type is evaluated when a use of it is compiled, so it is
    /// referenced from every path that touches a record — [`Self::segment_size`]
    /// (opening one), `write` and `read` — and fails the build of the
    /// offending program (not `cargo check`, which does not instantiate
    /// generics).
    const STORAGE_FITS_CACHELINE: () = assert!(
        align_of::<T>() <= CACHELINE_BYTES,
        "STORAGE alignment exceeds the cacheline; entry cannot be cacheline-isolated"
    );

    /// The size of the shared-memory segment holding one record:
    /// `sizeof(seq_lock<STORAGE>)`. The one way to size a segment, so that
    /// merely opening a record over an over-aligned storage type trips the
    /// guard, as instantiating tyndall's `shmem_buf` does its `static_assert`.
    pub const fn segment_size() -> usize {
        let () = Self::STORAGE_FITS_CACHELINE;
        size_of::<Self>()
    }

    pub fn write(&self, value: T) {
        let () = Self::STORAGE_FITS_CACHELINE;
        self.seq.fetch_add(1, Ordering::Release);
        unsafe {
            *self.size.get() = size_of::<T>();
            *self.entry.0.get() = value;
        }
        self.seq.fetch_add(1, Ordering::Release);
    }

    pub fn read(&self, state: &mut SeqLockState, always_update_entry: bool) -> Result<T, i32> {
        let () = Self::STORAGE_FITS_CACHELINE;
        let mut entry = T::default();
        loop {
            let seq1: u32 = self.seq.load(Ordering::Acquire);
            if seq1 & 1 != 0 {
                debug!("SeqLock is being written to, retrying...");
                std::hint::spin_loop();
                continue;
            }

            if seq1 != state.prev_seq || always_update_entry {
                entry = unsafe { *self.entry.0.get() };
            }

            let seq2 = self.seq.load(Ordering::Acquire);
            if seq1 != seq2 {
                debug!("SeqLock read failed, seq mismatch: {seq1} != {seq2}");
                std::hint::spin_loop();
                continue;
            }

            if seq1 != state.prev_seq {
                state.prev_seq = seq1;
                state.has_read_once = true;
                return Ok(entry);
            } else if seq1 == 0 && !state.has_read_once {
                return Err(libc::ENOMSG);
            } else {
                return Err(libc::EAGAIN);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The storage types of tyndall's seq_lock_layout.cpp.
    #[derive(Clone, Copy, Default, Debug, PartialEq)]
    #[repr(C)]
    struct Vec3 {
        x: f64,
        y: f64,
        z: f64,
    }

    #[derive(Clone, Copy, Default, Debug, PartialEq)]
    #[repr(C)]
    struct Mixed {
        a: i64,
        b: f64,
        c: u8,
        d: u64,
    }

    /// A serialised-message record (libblunux's 512-byte buffer).
    #[derive(Clone, Copy)]
    #[repr(C)]
    struct Bytes512([u8; 512]);

    impl Default for Bytes512 {
        fn default() -> Self {
            Self([0; 512])
        }
    }

    /// `sizeof(seq_lock<STORAGE>)` in C++: the cacheline header plus the
    /// storage, rounded up to the cacheline (the struct's alignment).
    fn cxx_sizeof<T>() -> usize {
        (CACHELINE_BYTES + size_of::<T>()).div_ceil(CACHELINE_BYTES) * CACHELINE_BYTES
    }

    fn check_layout<T: Copy + Default>(name: &str) {
        let offset = offset_of!(SeqLock<T>, entry);
        println!(
            "SeqLock<{name}>: offset_of(entry)={offset} CACHELINE_BYTES={CACHELINE_BYTES} size={}",
            size_of::<SeqLock<T>>()
        );
        assert_eq!(
            offset, CACHELINE_BYTES,
            "{name}: entry must start one cacheline in"
        );
        assert_eq!(
            offset % CACHELINE_BYTES,
            0,
            "{name}: entry must be cacheline-aligned"
        );
        assert_eq!(
            align_of::<SeqLock<T>>(),
            CACHELINE_BYTES,
            "{name}: struct alignment"
        );
        assert_eq!(size_of::<SeqLock<T>>(), cxx_sizeof::<T>(), "{name}: sizeof");
    }

    /// tyndall's tests/ipc/seq_lock_layout.cpp, over the same storage types.
    #[test]
    fn entry_starts_exactly_one_cacheline_in() {
        check_layout::<u8>("char");
        check_layout::<f32>("float");
        check_layout::<Vec3>("vec3");
        check_layout::<Mixed>("mixed");
        check_layout::<Bytes512>("serialized record");
    }

    /// The constant follows tyndall/ipc/smp.h after its PR #18.
    #[test]
    fn cacheline_bytes_follows_tyndall_smp_h() {
        let expected = match std::env::consts::ARCH {
            "x86_64" | "aarch64" => 64,
            "arm" => 32,
            other => panic!("no expectation for {other}"),
        };
        assert_eq!(CACHELINE_BYTES, expected);
    }

    /// A record on the stack, zero-initialised as a fresh shared-memory
    /// segment is: the write/read contract over the new layout.
    #[test]
    fn write_then_read_round_trips_through_the_aligned_entry() {
        // SAFETY: every field is valid when zeroed (an atomic, a usize, and a
        // storage type whose zero pattern is a valid value in these tests).
        let lock: SeqLock<Vec3> = unsafe { std::mem::zeroed() };
        let mut state = SeqLockState::default();
        assert_eq!(lock.read(&mut state, true), Err(libc::ENOMSG));
        let value = Vec3 {
            x: 1.5,
            y: -2.5,
            z: 3.25,
        };
        lock.write(value);
        assert_eq!(lock.read(&mut state, true), Ok(value));
        assert_eq!(lock.read(&mut state, true), Err(libc::EAGAIN));
        // The writer stores sizeof(STORAGE), as tyndall's does.
        assert_eq!(unsafe { *lock.size.get() }, size_of::<Vec3>());
        assert_eq!(SeqLock::<Vec3>::segment_size(), size_of::<SeqLock<Vec3>>());
        // The entry sits where a C++ reader looks for it.
        let base = &lock as *const SeqLock<Vec3> as usize;
        let entry = lock.entry.0.get() as usize;
        assert_eq!(entry - base, CACHELINE_BYTES);
    }
}
