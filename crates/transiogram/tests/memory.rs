use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use varvite_core::{Error, Grid, RateMatrix, SequenceSet};
use varvite_transiogram::{PairFilter, PairSpec, TransitionModel, downhole, experimental, on_grid};

/// Tracks the live heap and its peak, and refuses any allocation that would take the live heap
/// past 1.5 GB, so an absurd request aborts this test binary instead of exhausting the machine.
struct Capped;

const CAP: usize = 1536 << 20;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static SERIAL: Mutex<()> = Mutex::new(());

fn take(n: usize) -> bool {
    let now = LIVE.fetch_add(n, Ordering::SeqCst) + n;
    if now > CAP {
        LIVE.fetch_sub(n, Ordering::SeqCst);
        return false;
    }
    PEAK.fetch_max(now, Ordering::SeqCst);
    true
}

unsafe impl GlobalAlloc for Capped {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if !take(l.size()) {
            return std::ptr::null_mut();
        }
        let p = unsafe { System.alloc(l) };
        if p.is_null() {
            LIVE.fetch_sub(l.size(), Ordering::SeqCst);
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        if !take(l.size()) {
            return std::ptr::null_mut();
        }
        let p = unsafe { System.alloc_zeroed(l) };
        if p.is_null() {
            LIVE.fetch_sub(l.size(), Ordering::SeqCst);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) };
        LIVE.fetch_sub(l.size(), Ordering::SeqCst);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        if n > l.size() && !take(n - l.size()) {
            return std::ptr::null_mut();
        }
        let q = unsafe { System.realloc(p, l, n) };
        if q.is_null() {
            if n > l.size() {
                LIVE.fetch_sub(n - l.size(), Ordering::SeqCst);
            }
        } else if n < l.size() {
            LIVE.fetch_sub(l.size() - n, Ordering::SeqCst);
        }
        q
    }
}

#[global_allocator]
static GLOBAL: Capped = Capped;

fn spec(n_lags: usize) -> PairSpec {
    PairSpec {
        direction: None,
        lag: 1.0,
        n_lags,
        lag_tolerance: 0.5,
        angle_tolerance: 22.5,
        bandwidth: f64::INFINITY,
        vertical_bandwidth: f64::INFINITY,
        directed: true,
        pairs: PairFilter::All,
    }
}

fn rejected<T: std::fmt::Debug>(r: varvite_core::Result<T>, what: &str) {
    match r {
        Err(Error::InvalidInput(m)) => assert!(m.contains(what), "{m}"),
        other => panic!("expected InvalidInput naming {what}, got {other:?}"),
    }
}

#[test]
fn point_estimator_rejects_an_absurd_lag_count_before_allocating() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let xyz = [[0.0; 3], [1.0, 0.0, 0.0]];
    let r = experimental(&xyz, &[0, 1], None, 2, &spec(1 << 36));
    rejected(r, "experimental transiogram");
}

#[test]
fn point_estimator_rejects_an_absurd_category_count_before_allocating() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let xyz = [[0.0; 3], [1.0, 0.0, 0.0]];
    let r = experimental(&xyz, &[0, 1], None, 1 << 20, &spec(1));
    rejected(r, "experimental transiogram");
}

#[test]
fn grid_estimator_rejects_an_absurd_lag_count_before_allocating() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let grid = Grid::new([0.0; 3], [1.0; 3], [2, 1, 1]).unwrap();
    let r = on_grid(&[0, 1], &grid, None, [1, 0, 0], 1 << 36, 2);
    rejected(r, "grid transiogram");
}

#[test]
fn downhole_estimator_rejects_an_absurd_category_count_before_allocating() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let k = 1 << 20;
    let seqs = SequenceSet::from_regular(&[vec![0, 1]], k).unwrap();
    let r = downhole(&seqs, varvite_core::Direction::Up, &[1.0], 0.0);
    rejected(r, "downhole transiogram");
}

#[test]
fn point_estimator_peak_does_not_grow_with_the_number_of_blocks() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let n = 64 * 1024;
    let xyz: Vec<[f64; 3]> = (0..n).map(|i| [1000.0 * i as f64, 0.0, 0.0]).collect();
    let codes: Vec<u32> = (0..n).map(|i| (i % 3) as u32).collect();
    let n_lags = 2000;
    let base = LIVE.load(Ordering::SeqCst);
    PEAK.store(base, Ordering::SeqCst);
    let e = experimental(&xyz, &codes, None, 3, &spec(n_lags)).unwrap();
    let used = PEAK.load(Ordering::SeqCst) - base;
    assert_eq!(e.counts[999].sum(), 2.0 * (n - 1) as f64);
    assert_eq!(e.counts[1999].sum(), 2.0 * (n - 2) as f64);
    let per_set = n_lags * (size_of::<varvite_core::Mat>() + 9 * 8 + 8);
    let blocks = n / 1024;
    let threads = rayon::current_num_threads();
    assert!(
        used < (threads + 8) * per_set + n * 200,
        "{used} bytes for {blocks} blocks of {per_set} bytes"
    );
}

fn two_category(lengths: [[f64; 2]; 3]) -> TransitionModel {
    let p = [0.5, 0.5];
    let rates: Vec<RateMatrix> = lengths
        .iter()
        .map(|l| RateMatrix::max_entropy(&p, l).unwrap())
        .collect();
    TransitionModel::new(p.to_vec(), rates.try_into().unwrap(), [0.0; 3], 1).unwrap()
}

/// AGENTS.md: raise InvalidInput before allocating when a request exceeds the budget. Every
/// budget the table accepts must hold its real peak; tight budgets are refused.
#[test]
fn table_stays_within_the_memory_budget_it_accepted() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let m = two_category([[1e3; 2], [1e3; 2], [1e3; 2]]);
    let grid = Grid::new([0.0; 3], [1.0; 3], [24, 24, 24]).unwrap();
    let offsets = 47u64 * 47 * 47 - 1;
    let mut accepted = 0;
    for per_offset in [8u64, 64, 128, 192, 256, 512] {
        let budget = offsets * per_offset + 1024;
        let base = LIVE.load(Ordering::SeqCst);
        PEAK.store(base, Ordering::SeqCst);
        match m.table(&grid, 0.01, Some(budget)) {
            Ok(t) => {
                let used = (PEAK.load(Ordering::SeqCst) - base) as u64;
                assert_eq!(t.offsets.len() as u64, offsets);
                assert!(
                    used <= budget,
                    "table accepted a budget of {budget} bytes and allocated {used} bytes"
                );
                accepted += 1;
            }
            Err(e) => {
                assert!(per_offset < 512, "refused {budget} bytes: {e}");
                assert!(e.to_string().contains("transition table"), "{e}");
            }
        }
    }
    assert!(accepted > 0);
}
