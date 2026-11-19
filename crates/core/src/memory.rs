use crate::error::{Result, invalid};

/// `MemAvailable` from `/proc/meminfo` on Linux, lowered to the headroom of the tightest cgroup v2
/// `memory.max` above this process; `None` elsewhere or on any failure.
pub fn available_bytes() -> Option<u64> {
    if !cfg!(target_os = "linux") {
        return None;
    }
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = text.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    let system = kb.checked_mul(1024)?;
    Some(cgroup_headroom().map_or(system, |c| c.min(system)))
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
            if let Some(cgroup) = cgroup_headroom() {
                assert!(available <= cgroup);
            }
        }
    }
}
