use crate::error::{Result, invalid};

/// `MemAvailable` from `/proc/meminfo` on Linux; `None` elsewhere or on any failure.
pub fn available_bytes() -> Option<u64> {
    if !cfg!(target_os = "linux") {
        return None;
    }
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = text.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    kb.checked_mul(1024)
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
        }
    }
}
