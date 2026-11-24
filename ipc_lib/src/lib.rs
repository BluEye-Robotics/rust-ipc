// ipc_lib/src/lib.rs

use libc::{
    c_int, c_void, close, fstat, ftruncate, mmap, munmap, off_t, shm_open, MAP_FAILED, MAP_SHARED,
    O_CREAT, O_EXCL, O_RDWR, PROT_READ, PROT_WRITE,
};
use std::ffi::CString;
use std::ptr::null_mut;
mod seqlock;
use seqlock::{SeqLock, SeqLockState, CACHELINE_BYTES};

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

/// The name passed to `shm_open`: `/` and [`topic_to_shm_name`], shortened on
/// macOS as tyndall's `fix_shm_name()` (tyndall/ipc/shmem.h) does.
fn shm_open_name(topic: &str) -> Vec<u8> {
    let name = format!("/{}", topic_to_shm_name(topic)).into_bytes();
    #[cfg(target_os = "macos")]
    let name = shorten_for_macos(name);
    name
}

/// macOS allows at most 31 characters in a shm name (`PSHMNAMLEN`), and
/// `shm_open` fails with `ENAMETOOLONG` beyond.
#[cfg(any(target_os = "macos", test))]
const MACOS_SHM_NAME_MAX: usize = 31;

/// Shorten a shm name that is too long for macOS exactly as tyndall does: keep
/// its first 22 bytes and append `_` and the FNV-1a hash of the whole name as
/// 8 hex digits, so that names with the same start stay apart.
#[cfg(any(target_os = "macos", test))]
fn shorten_for_macos(name: Vec<u8>) -> Vec<u8> {
    if name.len() <= MACOS_SHM_NAME_MAX {
        return name;
    }
    let hash = format!("_{:08x}", fnv1a_32(&name));
    let mut short = name[..MACOS_SHM_NAME_MAX - hash.len()].to_vec();
    short.extend_from_slice(hash.as_bytes());
    short
}

/// FNV-1a 32-bit hash, matching tyndall's `hash_fnv1a_32`.
#[cfg(any(target_os = "macos", test))]
fn fnv1a_32(data: &[u8]) -> u32 {
    let mut hash: u32 = 2166136261;
    for &byte in data {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(16777619);
    }
    hash
}

/// tyndall's list of the shm names created on macOS, where `ipc_cleanup()`
/// has no `/dev/shm` to list (tyndall/ipc/shmem_mac_ipc_names.h).
#[cfg(target_os = "macos")]
const IPC_NAMES_FILE: &str = "/tmp/tyndall_ipc_names";

/// Add a created shm name to [`IPC_NAMES_FILE`] the way tyndall's
/// `append_ipc_name()` does: under `flock`, one name per line, sorted, each
/// once. tyndall's `ipc_cleanup()` then unlinks Rust's segments too.
#[cfg(target_os = "macos")]
fn record_created_name(name: &[u8]) -> std::io::Result<()> {
    use std::collections::BTreeSet;
    use std::fs::OpenOptions;
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::os::unix::fs::OpenOptionsExt;
    use std::os::unix::io::AsRawFd;

    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o666)
        .open(IPC_NAMES_FILE)?;
    // Released when the file closes.
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    let mut contents = Vec::new();
    file.read_to_end(&mut contents)?;
    let mut names: BTreeSet<Vec<u8>> = contents
        .split(|&b| b == b'\n')
        .filter(|line| !line.is_empty())
        .map(<[u8]>::to_vec)
        .collect();
    if names.insert(name.to_vec()) {
        let mut out = Vec::new();
        for n in &names {
            out.extend_from_slice(n);
            out.push(b'\n');
        }
        file.set_len(0)?;
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&out)?;
    }
    Ok(())
}

/// Room for what tyndall's C++ keeps behind the record: the type hash and the
/// debug format, which tyndall's `static_assert` keeps below one cacheline.
const TYNDALL_TAILER_ROOM: usize = CACHELINE_BYTES;

/// The size of the segment behind `fd`. macOS reports it rounded up to a page.
fn segment_len(fd: c_int) -> Result<off_t, String> {
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    if unsafe { fstat(fd, &mut st) } != 0 {
        return Err(format!("fstat failed: errno {}", last_errno()));
    }
    Ok(st.st_size)
}

/// Make sure the segment behind `fd` holds the `needed` bytes this process
/// maps. `observed` is its size as this process last saw it. Returns whether
/// this call sized the segment.
///
/// Every process that opens a segment does this, not only the one that made
/// it, as tyndall's C++ does on Linux. Otherwise a process that opens the
/// segment between its creator's `shm_open` and `ftruncate`, or after a
/// creator died in between, maps a segment of size 0: `mmap` fails on macOS,
/// and the first access raises SIGBUS on Linux.
///
/// The segment grows to `target`, which leaves room for tyndall's tailer, so
/// sizing it late never cuts off a tailer that a C++ process has already
/// written behind the record. macOS sizes a segment only once: when another
/// process got there first, `ftruncate` fails, and the size it set decides.
fn ensure_sized(fd: c_int, observed: off_t, needed: off_t, target: off_t) -> Result<bool, String> {
    if observed >= needed {
        return Ok(false);
    }
    if unsafe { ftruncate(fd, target) } == 0 {
        return Ok(true);
    }
    let errno = last_errno();
    let now = segment_len(fd)?;
    if now >= needed {
        Ok(false)
    } else {
        Err(format!(
            "ftruncate failed: errno {errno}; the segment has {now} of {needed} bytes"
        ))
    }
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
    /// containing `%` is refused. On macOS a name over 31 characters is
    /// shortened as tyndall shortens it, and a segment this call creates is
    /// recorded for tyndall's `ipc_cleanup()`.
    pub fn new(topic: &str) -> Result<Self, String> {
        if topic.contains('%') {
            return Err(format!(
                "Invalid topic '{topic}': '%' stands for '/' in shm names"
            ));
        }
        let cname =
            CString::new(shm_open_name(topic)).map_err(|_| "Invalid shm name".to_string())?;
        log::info!(
            "Opening IPC topic '{topic}' -> shm '{}'",
            cname.to_string_lossy()
        );

        // O_EXCL, so that `create` is only true for the process that made the
        // segment. Only that one may unlink it again on failure.
        let mut create = true;
        let mut fd = unsafe { shm_open(cname.as_ptr(), O_CREAT | O_EXCL | O_RDWR, 0o666) };
        if fd < 0 && last_errno() == libc::EEXIST {
            create = false;
            fd = unsafe { shm_open(cname.as_ptr(), O_RDWR, 0o666) };
        }

        if fd < 0 {
            return Err(format!("shm_open failed: errno {}", last_errno()));
        }

        // Through `segment_size`, not `size_of`: that is where the storage
        // type's alignment guard is evaluated (see seqlock.rs).
        let size = SeqLock::<T>::segment_size();

        let sized = segment_len(fd).and_then(|observed| {
            ensure_sized(
                fd,
                observed,
                size as off_t,
                (size + TYNDALL_TAILER_ROOM) as off_t,
            )
        });
        let sized = match sized {
            Ok(sized) => sized,
            Err(e) => {
                unsafe {
                    if create {
                        libc::shm_unlink(cname.as_ptr());
                    }
                    close(fd);
                }
                return Err(e);
            }
        };

        // A segment whose creator died before sizing it is recorded by the
        // process that sized it.
        #[cfg(target_os = "macos")]
        if create || sized {
            if let Err(e) = record_created_name(cname.as_bytes()) {
                log::warn!(
                    "Could not record '{}' for cleanup: {e}",
                    cname.to_string_lossy()
                );
            }
        }
        #[cfg(not(target_os = "macos"))]
        let _ = sized;

        let ptr = unsafe { mmap(null_mut(), size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0) };

        unsafe { close(fd) };

        if ptr == MAP_FAILED {
            return Err(format!("mmap failed: errno {}", last_errno()));
        }

        Ok(Self {
            ptr: ptr as *mut SeqLock<T>,
            state: SeqLockState::default(),
            size,
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

    #[test]
    fn test_fnv1a_32() {
        assert_eq!(fnv1a_32(b""), 2166136261);
        assert_eq!(fnv1a_32(b"a"), 0xe40c292c);
    }

    #[test]
    fn test_macos_names_are_shortened_as_tyndall_does() {
        // Expected values printed by tyndall's fix_shm_name() on macOS.
        let short = |name: &str| String::from_utf8(shorten_for_macos(name.into())).unwrap();
        assert_eq!(
            short("/tyn_controller%control_force"),
            "/tyn_controller%control_force"
        );
        assert_eq!(
            short("/tyn_guestport%dvl%position_local"),
            "/tyn_guestport%dvl%pos_5dd4d2c6"
        );
        assert_eq!(
            short("/tyn_camera_control%set_camera_parameters"),
            "/tyn_camera_control%se_496b6005"
        );
        assert_eq!(
            short("/tyn_test%a_topic_longer_than_macos_allows%first"),
            "/tyn_test%a_topic_long_15b42a6c"
        );
        assert_eq!(
            short("/tyn_test%a_topic_longer_than_macos_allows%second"),
            "/tyn_test%a_topic_long_5995b29a"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_macos_records_created_names_for_tyndall_cleanup() {
        let topic = format!(
            "/rust_ipc_test/a_topic_longer_than_macos_allows_{}",
            std::process::id()
        );
        let name = shm_open_name(&topic);
        assert_eq!(name.len(), MACOS_SHM_NAME_MAX);
        let ipc = IPC::<u32>::new(&topic).expect("a long topic opens on macOS");
        ipc.set(7);
        let listed = std::fs::read(IPC_NAMES_FILE).unwrap();
        assert!(listed
            .split(|&b| b == b'\n')
            .any(|line| line == name.as_slice()));
        drop(ipc);
        let cname = CString::new(name.clone()).unwrap();
        assert_eq!(unsafe { libc::shm_unlink(cname.as_ptr()) }, 0);
        forget_recorded_name(&name);
    }

    /// Take a test's line out of tyndall's name list again, so that tyndall's
    /// cleanup doesn't try it.
    #[cfg(target_os = "macos")]
    fn forget_recorded_name(name: &[u8]) {
        let fd = std::fs::File::open(IPC_NAMES_FILE).unwrap();
        use std::os::unix::io::AsRawFd;
        assert_eq!(unsafe { libc::flock(fd.as_raw_fd(), libc::LOCK_EX) }, 0);
        let rest: Vec<u8> = std::fs::read(IPC_NAMES_FILE)
            .unwrap()
            .split(|&b| b == b'\n')
            .filter(|line| !line.is_empty() && *line != name)
            .flat_map(|line| line.iter().copied().chain([b'\n']))
            .collect();
        std::fs::write(IPC_NAMES_FILE, rest).unwrap();
    }

    /// A topic of its own for each test, so that tests running at the same
    /// time, or a run that left a segment behind, don't share one.
    fn test_topic(test: &str) -> String {
        format!("/rust_ipc_test/{test}_{}", std::process::id())
    }

    /// Create the segment of `topic` as another process would and leave it
    /// unsized. Returns its `shm_open` name and descriptor.
    fn create_unsized(topic: &str) -> (Vec<u8>, c_int) {
        let name = shm_open_name(topic);
        let cname = CString::new(name.clone()).unwrap();
        let fd = unsafe { shm_open(cname.as_ptr(), O_CREAT | O_EXCL | O_RDWR, 0o600) };
        assert!(fd >= 0, "shm_open failed: errno {}", last_errno());
        assert_eq!(segment_len(fd).unwrap(), 0);
        (name, fd)
    }

    fn remove_segment(name: &[u8], fd: c_int) {
        let cname = CString::new(name.to_vec()).unwrap();
        unsafe {
            close(fd);
            libc::shm_unlink(cname.as_ptr());
        }
        #[cfg(target_os = "macos")]
        forget_recorded_name(name);
    }

    #[test]
    fn test_opens_a_segment_nobody_has_sized() {
        // As if another process had created the segment and not sized it yet,
        // or had died before sizing it. Before every process sized it, the
        // write below raised SIGBUS on Linux, and macOS refused the mapping.
        let topic = test_topic("unsized");
        let (name, fd) = create_unsized(&topic);

        let writer = IPC::<[u64; 4]>::new(&topic).expect("an unsized segment opens");
        let needed = SeqLock::<[u64; 4]>::segment_size() as off_t;
        assert!(segment_len(fd).unwrap() >= needed);
        writer.set([1, 2, 3, 4]);

        let mut reader = IPC::<[u64; 4]>::new(&topic).unwrap();
        let mut value = [0; 4];
        assert_eq!(reader.get(&mut value), IPCStatus::Success);
        assert_eq!(value, [1, 2, 3, 4]);

        drop((writer, reader));
        remove_segment(&name, fd);
    }

    #[test]
    fn test_late_sizing_keeps_a_tyndall_tailer() {
        // This process created the segment and found it unsized. Before it
        // sizes it, a C++ process sizes it to its own size and writes its
        // tailer behind the record. Sizing to just the record would cut the
        // tailer off on Linux.
        let needed = SeqLock::<[u64; 4]>::segment_size();
        let cpp_len = needed + TYNDALL_TAILER_ROOM - 1; // the largest tailer tyndall allows
        let (name, fd) = create_unsized(&test_topic("late_sizing"));
        assert_eq!(unsafe { ftruncate(fd, cpp_len as off_t) }, 0);
        let map = unsafe {
            mmap(
                null_mut(),
                cpp_len,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                fd,
                0,
            )
        };
        assert_ne!(map, MAP_FAILED);
        let tailer = unsafe {
            std::slice::from_raw_parts_mut((map as *mut u8).add(needed), cpp_len - needed)
        };
        tailer.fill(0xa5);

        let target = (needed + TYNDALL_TAILER_ROOM) as off_t;
        let sized = ensure_sized(fd, 0, needed as off_t, target);
        assert!(sized.is_ok(), "{sized:?}");
        assert!(segment_len(fd).unwrap() >= cpp_len as off_t);
        assert!(tailer.iter().all(|&b| b == 0xa5), "the tailer was cut off");

        unsafe { munmap(map, cpp_len) };
        remove_segment(&name, fd);
    }
}
