// ipc_lib/src/lib.rs

use libc::{
    c_void, close, ftruncate, mmap, munmap, off_t, shm_open, MAP_FAILED, MAP_SHARED, O_CREAT,
    O_RDWR, PROT_READ, PROT_WRITE,
};
use std::ffi::CString;
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
const IPC_SHMEM_PREFIX: &str = "tyn";

/// Convert a topic name to the tyndall shared memory name, as
/// `id_rtid_prepare` in tyndall/ipc/id.h does: the prefix and `_`, then the
/// topic without its leading slashes, with each remaining `/` replaced by `%`.
///
/// `/controller/control_force` becomes `tyn_controller%control_force`. A topic
/// can't contain `%` (see [`IPC::new`]), so different topics get different
/// names.
fn topic_to_shm_name(topic: &str) -> String {
    let id = topic.trim_start_matches('/');
    format!("{IPC_SHMEM_PREFIX}_{id}").replace('/', "%")
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
    /// Leading slashes are stripped and the remaining slashes become `%`
    /// to produce a unique shared memory path in `/dev/shm/`. A topic
    /// containing `%` is refused.
    pub fn new(topic: &str) -> Result<Self, String> {
        if topic.contains('%') {
            return Err(format!(
                "Invalid topic '{topic}': '%' stands for '/' in shm names"
            ));
        }
        let shm_name = topic_to_shm_name(topic);
        log::info!("Opening IPC topic '{topic}' -> shm '/{shm_name}'");

        let cname =
            CString::new(format!("/{shm_name}")).map_err(|_| "Invalid shm name".to_string())?;

        let mut create = false;
        let mut fd = unsafe { shm_open(cname.as_ptr(), O_RDWR, 0o666) };
        if fd < 0 {
            fd = unsafe { shm_open(cname.as_ptr(), O_CREAT | O_RDWR, 0o666) };
            create = true;
        }

        if fd < 0 {
            return Err(format!("shm_open failed: errno {}", last_errno()));
        }

        // Through `segment_size`, not `size_of`: that is where the storage
        // type's alignment guard is evaluated (see seqlock.rs).
        let size = SeqLock::<T>::segment_size() as off_t;

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

/// Async polling API (requires the `tokio` feature).
#[cfg(feature = "tokio")]
impl<T: Copy + Default + Send + 'static> IPC<T> {
    /// Create a tokio channel that receives new values as they appear on the IPC topic.
    ///
    /// Spawns an internal polling loop that checks for new data at the given `poll_interval`.
    /// Returns a `tokio::sync::mpsc::Receiver<T>` that yields values whenever
    /// `IPCStatus::Success` is returned.
    ///
    /// The polling stops when the receiver is dropped.
    pub fn into_stream(
        mut self,
        poll_interval: std::time::Duration,
        buffer: usize,
    ) -> tokio::sync::mpsc::Receiver<T> {
        let (tx, rx) = tokio::sync::mpsc::channel(buffer);

        tokio::spawn(async move {
            let mut obj = T::default();
            loop {
                match self.get(&mut obj) {
                    IPCStatus::Success => {
                        if tx.send(obj).await.is_err() {
                            break; // receiver dropped
                        }
                    }
                    _ => {}
                }
                tokio::time::sleep(poll_interval).await;
            }
        });

        rx
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
    fn test_topic_to_shm_name() {
        // The names tyndall builds (tests/ipc/ipc.cpp pins the same ones)
        assert_eq!(
            topic_to_shm_name("/controller/control_force"),
            "tyn_controller%control_force"
        );
        assert_eq!(topic_to_shm_name("fpscalc/main"), "tyn_fpscalc%main");

        // Same topic with or without leading slashes gives the same name
        assert_eq!(
            topic_to_shm_name("//my/topic"),
            topic_to_shm_name("my/topic")
        );

        // '%' for '/' keeps topics apart that '_' would merge
        assert_ne!(topic_to_shm_name("my/topic"), topic_to_shm_name("my_topic"));
    }

    #[test]
    fn test_topic_with_percent_is_refused() {
        assert!(IPC::<f64>::new("/my%topic").is_err());
    }
}
