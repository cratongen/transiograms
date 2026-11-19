use crate::error::{Result, invalid};

/// `MemAvailable` from `/proc/meminfo` on Linux, lowered to the headroom of the tightest cgroup v2
/// `memory.max` above this process and of its own data and address-space limits (`ulimit -d`,
/// `ulimit -v`); `None` elsewhere or on any failure.
pub fn available_bytes() -> Option<u64> {
    if !cfg!(target_os = "linux") {
        return None;
    }
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
        if cfg!(target_os = "linux") {
            let available = available_bytes().unwrap();
            assert!(available > 0);
            assert!(check(u64::MAX, None, "grid").is_err());
            assert!(check(1, None, "grid").is_ok());
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
