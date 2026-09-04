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
//!
//! # Memory ordering
//!
//! The barriers match tyndall's, which are the Linux seqlock's. Four of them,
//! and each one closes a specific hole on a weakly ordered machine (armv7 and
//! aarch64 — x86's store order makes two of them free):
//!
//! ```text
//! writer                          tyndall                 here
//!   seq -> odd                     WRITE_ONCE              fetch_add(AcqRel)
//!   ---- store-store ----          smp_wmb()               the AcqRel above
//!   size, entry                    plain stores            plain stores
//!   ---- store-store ----          smp_wmb()               the Release below
//!   seq -> even                    WRITE_ONCE              fetch_add(Release)
//!
//! reader
//!   load seq1 (spin while odd)     READ_ONCE               load(Acquire)
//!   ---- load-load ----            smp_rmb()               the Acquire above
//!   read entry                     plain load              plain load
//!   ---- load-load ----            smp_rmb()               fence(Acquire)
//!   load seq2, retry if != seq1    READ_ONCE               load(Acquire)
//! ```
//!
//! The two that are easy to get wrong are the *first* writer barrier and the
//! *second* reader barrier, because neither is implied by the release/acquire
//! pair that carries the payload:
//!
//! * a plain `Release` on the odd increment orders everything *before* it,
//!   which is the previous write, not the payload stores that follow. Without
//!   `AcqRel` those stores may become visible first, and a reader that samples
//!   `seq` either side of them sees the same even number over a half-written
//!   entry and accepts it;
//! * an `Acquire` on the second `seq` load orders what comes *after* it, not
//!   the entry read before it. Without [`fence`] the entry read may drift past
//!   the validation and pick up bytes from the next write.
//!
//! Both are silent when they go wrong: a torn record, not a crash. Nothing
//! here is UB — the storage is `Copy` plain data — so the failure is wrong
//! numbers reaching a consumer.

use std::cell::UnsafeCell;
use std::mem::{align_of, offset_of, size_of};
use std::sync::atomic::{fence, AtomicU32, Ordering};

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
        // `AcqRel`, not `Release`: the acquire half is tyndall's `smp_wmb()`
        // between the odd sequence number and the payload (see the module
        // doc). A plain `Release` would let the stores below become visible
        // first, and a reader sampling `seq` either side of them would accept
        // a half-written entry.
        self.seq.fetch_add(1, Ordering::AcqRel);
        unsafe {
            *self.size.get() = size_of::<T>();
            *self.entry.0.get() = value;
        }
        // `Release` is tyndall's second `smp_wmb()`: the payload stores above
        // are visible before the even sequence number that publishes them.
        self.seq.fetch_add(1, Ordering::Release);
    }

    pub fn read(&self, state: &mut SeqLockState, always_update_entry: bool) -> Result<T, i32> {
        let () = Self::STORAGE_FITS_CACHELINE;
        let mut entry = T::default();
        loop {
            // `Acquire` is tyndall's first `smp_rmb()`: the entry read below
            // cannot be hoisted above this sample of the sequence number.
            let seq1: u32 = self.seq.load(Ordering::Acquire);
            if seq1 & 1 != 0 {
                debug!("SeqLock is being written to, retrying...");
                std::hint::spin_loop();
                continue;
            }

            if seq1 != state.prev_seq || always_update_entry {
                entry = unsafe { *self.entry.0.get() };
            }

            // tyndall's second `smp_rmb()`: the entry read above must not
            // drift below this load, or the validation would be checking a
            // sequence number the bytes did not come from. The `Acquire` on
            // the load itself orders what follows it, not what precedes it.
            fence(Ordering::Acquire);
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
    /// A payload wide enough to tear: every word carries the same counter, so
    /// any mix of two writes is visible as a word that disagrees with the
    /// first one.
    #[derive(Clone, Copy, Default, Debug)]
    #[repr(C)]
    struct Wide([u64; 32]);

    impl Wide {
        fn of(counter: u64) -> Self {
            Self([counter; 32])
        }

        /// The counter, if every word agrees.
        fn counter(&self) -> Option<u64> {
            let first = self.0[0];
            self.0.iter().all(|&word| word == first).then_some(first)
        }
    }

    /// A record shared between the threads of the test, the way [`crate::IPC`]
    /// shares one between processes.
    ///
    /// `SeqLock` is deliberately neither `Send` nor `Sync` on its own — it is
    /// a raw shared-memory record, and it is `IPC` that owns the mapping and
    /// asserts the contract (one writer, any number of readers). The test
    /// makes the same assertion for the same reason.
    struct Shared(SeqLock<Wide>);

    // SAFETY: exactly the contract `IPC` documents. One writer thread, three
    // reader threads, and the seqlock's own atomics are the synchronisation.
    unsafe impl Send for Shared {}
    unsafe impl Sync for Shared {}

    /// A reader never observes a mix of two writes.
    ///
    /// This is a stress test, not a proof, and it is worth being precise about
    /// what it does and does not establish. A missing barrier is a reordering
    /// the hardware is *allowed* to perform, not one it must: the test can
    /// only fail when a machine actually takes the liberty. Measured, with the
    /// writer's `AcqRel` reverted to `Release`, it still passes on an Apple
    /// M-series host, whose cores rarely reorder stores in practice, and it
    /// would never fail on x86, whose store order hides the writer's half of
    /// the bug outright.
    ///
    /// So the evidence that the barriers are right is the code generation, not
    /// this test: on aarch64 the writer's first increment must emit `ldaddal`
    /// (acquire-release) rather than `ldaddl` (release only), and the reader's
    /// [`fence`] must emit `dmb ishld`, which is exactly tyndall's `smp_rmb()`.
    ///
    /// The test earns its two seconds by catching the gross regressions that
    /// no amount of reasoning protects against — losing the retry loop,
    /// publishing before the payload is stored, a future rewrite of `read`
    /// that drops the second sequence check — and it may yet catch a real
    /// reordering on the i.MX or a Jetson runner.
    #[test]
    fn a_reader_never_sees_a_half_written_entry() {
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;

        // SAFETY: as in the round-trip test, every field is valid zeroed.
        let lock: Arc<Shared> = Arc::new(Shared(unsafe { std::mem::zeroed() }));
        let stop = Arc::new(AtomicBool::new(false));

        let writer = {
            let lock = Arc::clone(&lock);
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                let mut counter: u64 = 1;
                while !stop.load(Ordering::Relaxed) {
                    lock.0.write(Wide::of(counter));
                    counter = counter.wrapping_add(1);
                }
                counter
            })
        };

        let readers: Vec<_> = (0..3)
            .map(|_| {
                let lock = Arc::clone(&lock);
                let stop = Arc::clone(&stop);
                std::thread::spawn(move || {
                    let mut state = SeqLockState::default();
                    let mut reads = 0u64;
                    while !stop.load(Ordering::Relaxed) {
                        if let Ok(entry) = lock.0.read(&mut state, true) {
                            assert!(
                                entry.counter().is_some(),
                                "torn entry: words disagree, first four are {:?}",
                                &entry.0[..4]
                            );
                            reads += 1;
                        }
                    }
                    reads
                })
            })
            .collect();

        std::thread::sleep(std::time::Duration::from_secs(2));
        stop.store(true, Ordering::Relaxed);

        let writes = writer.join().expect("writer panicked");
        let reads: u64 = readers
            .into_iter()
            .map(|reader| reader.join().expect("reader saw a torn entry"))
            .sum();
        println!("{writes} writes, {reads} consistent reads across 3 readers");
        // A run that raced nothing proves nothing; fail loudly rather than
        // pass on an empty test.
        assert!(writes > 1000, "the writer barely ran ({writes} writes)");
        assert!(reads > 1000, "the readers barely ran ({reads} reads)");
    }

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
