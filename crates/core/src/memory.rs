use crate::error::{Result, invalid};

/// The memory a new allocation can use, or `None` when it cannot be read.
///
/// Linux: `MemAvailable` from `/proc/meminfo`, lowered to the headroom of the tightest cgroup v2
/// `memory.max` above this process and of its own data and address-space limits (`ulimit -d`,
/// `ulimit -v`). Windows: the smaller of the available physical memory and the remaining commit
/// charge (`GlobalMemoryStatusEx`). macOS: the installed physical memory (`hw.memsize`), since
/// macOS reports no single figure for reclaimable memory. `None` on other platforms.
pub fn available_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    let bytes = linux_available();
    #[cfg(windows)]
    let bytes = windows_available();
    #[cfg(target_os = "macos")]
    let bytes = macos_available();
    #[cfg(not(any(target_os = "linux", windows, target_os = "macos")))]
    let bytes = None;
    bytes
}

#[cfg(target_os = "linux")]
fn linux_available() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = text.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    let system = kb.checked_mul(1024)?;
    Some(
        [cgroup_headroom(), rlimit_headroom()]
            .into_iter()
            .flatten()
            .fold(system, u64::min),
    )
}

#[cfg(windows)]
fn windows_available() -> Option<u64> {
    #[repr(C)]
    #[derive(Default)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
    }
    let mut status = MemoryStatusEx {
        length: std::mem::size_of::<MemoryStatusEx>() as u32,
        ..Default::default()
    };
    // SAFETY: `status` is a correctly sized MEMORYSTATUSEX with `dwLength` set, as the call requires.
    let ok = unsafe { GlobalMemoryStatusEx(&mut status) } != 0;
    (ok && status.avail_phys > 0).then(|| status.avail_phys.min(status.avail_page_file))
}

#[cfg(target_os = "macos")]
fn macos_available() -> Option<u64> {
    use std::ffi::{c_char, c_int, c_void};
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }
    let mut bytes: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    // SAFETY: `hw.memsize` is a 64-bit integer and `len` states the size of the buffer it fills.
    let rc = unsafe {
        sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&mut bytes as *mut u64).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (rc == 0 && len == std::mem::size_of::<u64>() && bytes > 0).then_some(bytes)
}

#[cfg(target_os = "linux")]
fn rlimit_headroom() -> Option<u64> {
    let limits = std::fs::read_to_string("/proc/self/limits").ok()?;
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let soft = |name: &str| -> Option<u64> {
        let rest = limits.lines().find_map(|l| l.strip_prefix(name))?;
        rest.split_whitespace().next()?.parse().ok()
    };
    let used = |key: &str| -> u64 {
        status
            .lines()
            .find_map(|l| l.strip_prefix(key))
            .and_then(|r| r.split_whitespace().next()?.parse::<u64>().ok())
            .map_or(0, |kb| kb.saturating_mul(1024))
    };
    [
        soft("Max data size").map(|l| l.saturating_sub(used("VmData:"))),
        soft("Max address space").map(|l| l.saturating_sub(used("VmSize:"))),
    ]
    .into_iter()
    .flatten()
    .min()
}

#[cfg(target_os = "linux")]
fn cgroup_headroom() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let path = text.lines().find_map(|l| l.strip_prefix("0::"))?;
    let root = std::path::Path::new("/sys/fs/cgroup");
    let mut dir = root.join(path.trim_start_matches('/'));
    let read = |d: &std::path::Path, f: &str| -> Option<u64> {
        std::fs::read_to_string(d.join(f)).ok()?.trim().parse().ok()
    };
    let mut least: Option<u64> = None;
    while dir.starts_with(root) {
        if let (Some(max), Some(current)) = (read(&dir, "memory.max"), read(&dir, "memory.current"))
        {
            let free = max.saturating_sub(current);
            least = Some(least.map_or(free, |l| l.min(free)));
        }
        if !dir.pop() {
            break;
        }
    }
    least
}

/// Raises `InvalidInput` when `needed` exceeds `memory`, or the available memory when `memory` is None.
pub fn check(needed: u64, memory: Option<u64>, what: &str) -> Result<()> {
    match memory.or_else(available_bytes) {
        Some(budget) if needed > budget => invalid(format!(
            "{what} needs {needed} bytes but the memory budget is {budget} bytes; pass memory= or use a smaller grid"
        )),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_from_argument_or_system() {
        assert!(check(10, Some(10), "grid").is_ok());
        let e = check(11, Some(10), "grid").unwrap_err().to_string();
        assert!(
            e.contains("grid needs 11 bytes") && e.contains("budget is 10 bytes"),
            "{e}"
        );
        if cfg!(any(target_os = "linux", windows, target_os = "macos")) {
            let available = available_bytes().unwrap();
            assert!(available > 0);
            assert!(check(u64::MAX, None, "grid").is_err());
            assert!(check(1, None, "grid").is_ok());
        }
        #[cfg(target_os = "linux")]
        {
            let available = available_bytes().unwrap();
            // Other test threads allocate and free between the two reads.
            let slack = 256 << 20;
            if let Some(cgroup) = cgroup_headroom() {
                assert!(available <= cgroup + slack, "{available} > {cgroup}");
            }
            if let Some(rlimit) = rlimit_headroom() {
                assert!(available <= rlimit + slack, "{available} > {rlimit}");
            }
        }
    }
}
