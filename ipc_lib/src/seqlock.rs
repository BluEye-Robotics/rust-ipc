use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU32, Ordering};
use log::{debug};

#[cfg(target_arch = "x86_64")]
const CACHELINE_BYTES: usize = 64;

#[cfg(target_arch = "aarch64")]
const CACHELINE_BYTES: usize = 32;

#[cfg(target_arch = "arm")]
const CACHELINE_BYTES: usize = 32;

#[cfg(target_arch = "x86_64")]
#[repr(C, align(64))]
pub struct SeqLock<T: Copy + Default> {
    seq: AtomicU32,
    size: usize,
    _padding:
        [u8; CACHELINE_BYTES - std::mem::size_of::<AtomicU32>() - std::mem::size_of::<usize>()],
    entry: UnsafeCell<T>,
}

#[cfg(not(target_arch = "x86_64"))]
#[repr(C, align(32))]
pub struct SeqLock<T: Copy + Default> {
    seq: AtomicU32,
    size: usize,
    _padding:
        [u8; CACHELINE_BYTES - std::mem::size_of::<AtomicU32>() - std::mem::size_of::<usize>()],
    entry: UnsafeCell<T>,
}
pub struct SeqLockState {
    prev_seq: u32,
    has_read_once: bool,
}

impl<T: Copy + Default> SeqLock<T> {
    pub fn write(&self, value: T) {
        self.seq.fetch_add(1, Ordering::Release);
        unsafe {
            *self.entry.get() = value;
        }
        self.seq.fetch_add(1, Ordering::Release);
    }

    pub fn read(&self, state: &mut SeqLockState, always_update_entry: bool) -> Result<T, i32> {
        let mut entry = T::default();
        loop {
            let seq1: u32 = self.seq.load(Ordering::Acquire);
            if seq1 & 1 != 0 {
                debug!("SeqLock is being written to, retrying...");
                std::hint::spin_loop();
                continue;
            }

            if seq1 != state.prev_seq || always_update_entry {
                entry = unsafe { *self.entry.get() };
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

impl Default for SeqLockState {
    fn default() -> Self {
        SeqLockState {
            prev_seq: 0,
            has_read_once: false,
        }
    }
}
