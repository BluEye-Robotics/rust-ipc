// ipc_lib/src/lib.rs

use libc::{
    c_void, close, ftruncate, mmap, munmap, off_t, shm_open, shm_unlink, MAP_FAILED,
    MAP_SHARED, O_CREAT, O_RDWR, PROT_READ, PROT_WRITE,
};
use std::ffi::CString;
use std::mem::size_of;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicBool, Ordering};

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

/// Simple RAII wrapper for shared memory mapping of type T
pub struct IPC<T> {
    ptr: *mut T,
    size: usize,
    name: CString,
    created: AtomicBool,
}

impl<T> IPC<T> {
    /// Create or open shared memory region named `name`
    /// If `create` is true, attempts to create and truncate
    pub fn new(name: &str) -> Result<Self, String> {
        let cname = CString::new(name).map_err(|_| "Invalid shm name")?;
        let mut create = false;
        let mut fd = unsafe { shm_open(cname.as_ptr(), O_RDWR, 0o666) };

        if fd < 0{
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
            ptr: ptr as *mut T,
            size: size as usize,
            name: cname,
            created: AtomicBool::new(create),
        })
    }

    /// Get mutable reference to shared struct
    pub fn get(&self) -> &mut T {
        unsafe { &mut *self.ptr }
    }
}


impl<T> Drop for IPC<T> {
    fn drop(&mut self) {
        unsafe {
            munmap(self.ptr as *mut c_void, self.size);
        }
    }
}
 