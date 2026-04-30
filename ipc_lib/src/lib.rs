// ipc_lib/src/lib.rs

use libc::{
    c_void, close, ftruncate, mmap, munmap, off_t, shm_open, MAP_FAILED, MAP_SHARED, O_CREAT,
    O_RDWR, PROT_READ, PROT_WRITE,
};
use std::ffi::CString;
use std::mem::size_of;
use std::ptr::null_mut;
mod seqlock;
use seqlock::{SeqLock, SeqLockState};

/// Cross-platform errno getter
fn last_errno() -> i32 {
    #[cfg(target_os = "linux")]
    unsafe {
        *libc::__errno_location()
    }

    #[cfg(target_os = "macos")]
    unsafe {
        *libc::__error()
    }
}

/// Status codes for IPC read operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IPCStatus {
    /// New data was successfully read.
    Success,
    /// No new data since last read (sequence unchanged).
    EAgain,
    /// Topic has never been written to.
    ENomsg,
}

/// The shared memory prefix used by tyndall IPC.
///
/// This is "ipc" + sha1("ipc") to reduce collision risk in /dev/shm.
const IPC_SHMEM_PREFIX: &str = "ipc1ef42bc4e0bbfeb0ac34bc3642732768cf6f77b7";

/// FNV-1a 32-bit hash, matching tyndall's `hash_fnv1a_32`.
fn fnv1a_32(data: &[u8]) -> u32 {
    let mut hash: u32 = 2166136261;
    for &byte in data {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(16777619);
    }
    hash
}

/// Convert a topic name to the tyndall shared memory name.
///
/// Algorithm (matching `id_rtid_prepare` in tyndall/ipc/id.h):
/// 1. Strip leading slashes
/// 2. Compute FNV-1a hash of the stripped id (with internal slashes intact)
/// 3. Construct: `IPC_SHMEM_PREFIX + "_" + id + "_" + hash`
/// 4. Replace all remaining slashes with underscores
fn topic_to_shm_name(topic: &str) -> String {
    let id = topic.trim_start_matches('/');
    let hash = fnv1a_32(id.as_bytes());
    let name = format!("{}_{}_{}",IPC_SHMEM_PREFIX, id, hash);
    name.replace('/', "_")
}

/// Simple RAII wrapper for shared memory mapping of type T.
///
/// Provides lock-free single-writer / multiple-reader IPC via a SeqLock
/// stored in POSIX shared memory. Compatible with tyndall's C++ IPC.
pub struct IPC<T: Copy + Default> {
    ptr: *mut SeqLock<T>,
    state: SeqLockState,
    size: usize,
}

// SAFETY: The shared memory region pointed to by `ptr` is valid for the lifetime
// of this struct (unmapped on Drop). The SeqLock uses proper atomic operations
// (AtomicU32 with Acquire/Release ordering) for synchronization, making it safe
// to access from any thread.
unsafe impl<T: Copy + Default> Send for IPC<T> {}

// SAFETY: The SeqLock provides synchronization via atomics. Write is safe for a
// single writer (API contract), and reads are always safe (spin on odd seq).
unsafe impl<T: Copy + Default> Sync for IPC<T> {}

impl<T: Copy + Default> IPC<T> {
    /// Open or create a tyndall IPC topic.
    ///
    /// The topic name follows tyndall conventions (e.g. `/my/topic`).
    /// Leading slashes are stripped and the name is hashed to produce
    /// a unique shared memory path in `/dev/shm/`.
    pub fn new(topic: &str) -> Result<Self, String> {
        let shm_name = topic_to_shm_name(topic);
        log::info!("Opening IPC topic '{}' -> shm '/{}'", topic, shm_name);

        let cname =
            CString::new(format!("/{}", shm_name)).map_err(|_| "Invalid shm name".to_string())?;

        let mut create = false;
        let mut fd = unsafe { shm_open(cname.as_ptr(), O_RDWR, 0o666) };
        if fd < 0 {
            fd = unsafe { shm_open(cname.as_ptr(), O_CREAT | O_RDWR, 0o666) };
            create = true;
        }

        if fd < 0 {
            return Err(format!("shm_open failed: errno {}", last_errno()));
        }

        let size = size_of::<SeqLock<T>>() as off_t;

        if create {
            let ret = unsafe { ftruncate(fd, size) };
            if ret < 0 {
                unsafe {
                    libc::shm_unlink(cname.as_ptr());
                    close(fd);
                }
                return Err(format!("ftruncate failed: errno {}", last_errno()));
            }
        }

        let ptr = unsafe {
            mmap(
                null_mut(),
                size as usize,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                fd,
                0,
            )
        };

        unsafe { close(fd) };

        if ptr == MAP_FAILED {
            return Err(format!("mmap failed: errno {}", last_errno()));
        }

        Ok(Self {
            ptr: ptr as *mut SeqLock<T>,
            state: SeqLockState::default(),
            size: size as usize,
        })
    }

    /// Read the latest value from the IPC topic.
    ///
    /// Returns `IPCStatus::Success` if new data was read into `obj`,
    /// `IPCStatus::EAgain` if no new data is available, or
    /// `IPCStatus::ENomsg` if the topic has never been written to.
    pub fn get(&mut self, obj: &mut T) -> IPCStatus {
        unsafe {
            match (*self.ptr).read(&mut self.state, true) {
                Ok(val) => {
                    *obj = val;
                    IPCStatus::Success
                }
                Err(libc::EAGAIN) => IPCStatus::EAgain,
                Err(libc::ENOMSG) => IPCStatus::ENomsg,
                Err(_) => IPCStatus::ENomsg,
            }
        }
    }

    /// Write a value to the IPC topic.
    ///
    /// This atomically publishes the value so that all readers will see it.
    /// Only one writer should exist per topic.
    pub fn set(&self, value: T) {
        unsafe {
            (*self.ptr).write(value);
        }
    }
}

impl<T: Copy + Default> Drop for IPC<T> {
    fn drop(&mut self) {
        unsafe {
            munmap(self.ptr as *mut c_void, self.size);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fnv1a_32() {
        // Verify against known FNV-1a values
        assert_eq!(fnv1a_32(b""), 2166136261);
        assert_eq!(fnv1a_32(b"a"), 0xe40c292c);
    }

    #[test]
    fn test_topic_to_shm_name() {
        // Leading slashes stripped, internal slashes become underscores, hash appended
        let name = topic_to_shm_name("/my/topic");
        assert!(name.starts_with(IPC_SHMEM_PREFIX));
        assert!(name.contains("my_topic_")); // slashes replaced
        assert!(!name.contains('/')); // no slashes in final name

        // Same topic with or without leading slash gives same name
        assert_eq!(topic_to_shm_name("/my/topic"), topic_to_shm_name("my/topic"));
        assert_eq!(topic_to_shm_name("//my/topic"), topic_to_shm_name("my/topic"));

        // Different topics with same underscore pattern get different hashes
        assert_ne!(topic_to_shm_name("my/topic"), topic_to_shm_name("my_topic"));
    }
}
