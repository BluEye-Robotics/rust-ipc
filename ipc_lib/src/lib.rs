// ipc_lib/src/lib.rs

use libc::{
    c_void, close, ftruncate, mmap, munmap, off_t, shm_open, shm_unlink, MAP_FAILED, MAP_SHARED,
    O_CREAT, O_RDWR, PROT_READ, PROT_WRITE,
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

// Create enum with SUCCESS, EAGAIN, ENOMSG
// Error codes for IPC operations
#[derive(Debug)]
pub enum IPCStatus {
    Success,
    EAgain,
    ENomsg,
}

/// Simple RAII wrapper for shared memory mapping of type T
pub struct IPC<T: Copy + Default> {
    ptr: *mut SeqLock<T>,
    state: SeqLockState,
    size: usize,
    name: CString,
}

impl<T: Copy + Default> IPC<T> {
    pub fn new(name: &str) -> Result<Self, String> {
        println!("Open shm topic {}", name);
        let name = format!("/tyn_{}", name.trim_start_matches('/').replace('/', "%"));

        let cname = CString::new(name).map_err(|_| "Invalid shm name")?;
        let mut create = false;
        let mut fd = unsafe { shm_open(cname.as_ptr(), O_RDWR, 0o666) };
        // Append PREFIX to the name to avoid conflicts
        if fd < 0 {
            fd = unsafe { shm_open(cname.as_ptr(), O_CREAT | O_RDWR, 0o666) };
            create = true;
        }

        if fd < 0 {
            return Err(format!("shm_open failed: errno {}", last_errno()));
        }

        let size = size_of::<T>() as off_t;

        if create {
            let ret = unsafe { ftruncate(fd, size) };
            if ret < 0 {
                unsafe { shm_unlink(cname.as_ptr()) };
                unsafe { close(fd) };
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
            name: cname,
        })
    }

    pub fn get(&mut self, obj: &mut T) -> IPCStatus {
        unsafe {
            match (*self.ptr).read(&mut self.state, true) {
                Ok(val) => {
                    *obj = val;
                    IPCStatus::Success
                }
                Err(libc::EAGAIN) => IPCStatus::EAgain,
                Err(libc::ENOMSG) => IPCStatus::ENomsg,
                Err(_) => IPCStatus::ENomsg, // fallback for other errors
            }
        }
    }

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
