use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use varvite_core::*;
use varvite_tpsim::*;
use varvite_transiogram::TransitionModel;

struct Counting;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            let now = CURRENT.fetch_add(layout.size(), Ordering::SeqCst) + layout.size();
            PEAK.fetch_max(now, Ordering::SeqCst);
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        CURRENT.fetch_sub(layout.size(), Ordering::SeqCst);
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn peak_during<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let base = CURRENT.load(Ordering::SeqCst);
    PEAK.store(base, Ordering::SeqCst);
    let out = f();
    (out, PEAK.load(Ordering::SeqCst) - base)
}

fn model() -> TransitionModel {
    TransitionModel::from_geology(
        &[0.5, 0.3, 0.2],
        [&[4.0, 3.0, 2.0], &[4.0, 3.0, 2.0], &[2.0, 1.0, 0.8]],
        [None, None, None],
        0,
        [0.0; 3],
    )
    .unwrap()
}

/// Smallest budget the call accepts, by bisection: `fails(b)` must be monotone in `b`.
fn smallest_accepted(fails: impl Fn(u64) -> bool) -> u64 {
    let (mut lo, mut hi) = (0u64, 1u64 << 40);
    assert!(fails(lo) && !fails(hi));
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if fails(mid) { lo = mid } else { hi = mid }
    }
    hi
}

/// With `memory = Some(B)` and B the smallest budget the library accepts, the bytes the call holds
/// at its peak must stay within B.
#[test]
fn predict_and_simulate_stay_within_the_budget_they_accept() {
    let model = model();
    let grid = Grid::new([0.0; 3], [1.0, 1.0, 0.5], [30, 30, 10]).unwrap();
    let hard = Snapped {
        cells: vec![0, 4500],
        codes: vec![0, 1],
        conflicts: 0,
        outside: 0,
    };
    let base = Options {
        n_realizations: 2,
        max_data: 4,
        closeness_limit: 0.3,
        quench: None,
        ..Options::default()
    };
    simulate(&model, &grid, None, &hard, &base).unwrap();
    let mut over = vec![];
    for quench in [None, Some(QuenchOptions::default())] {
        let with = |memory| Options {
            memory: Some(memory),
            quench: quench.clone(),
            ..base.clone()
        };
        let budget = smallest_accepted(|b| simulate(&model, &grid, None, &hard, &with(b)).is_err());
        let (r, peak) = peak_during(|| simulate(&model, &grid, None, &hard, &with(budget)));
        eprintln!(
            "simulate, quench {}: budget {budget} bytes, peak {peak} bytes",
            quench.is_some()
        );
        assert!(r.is_ok(), "{r:?}");
        if peak as u64 > budget {
            over.push(format!(
                "simulate (quench {}): peak {peak} > budget {budget}",
                quench.is_some()
            ));
        }
    }
    let with = |memory| Options {
        memory: Some(memory),
        ..base.clone()
    };
    let budget = smallest_accepted(|b| predict(&model, &grid, None, &hard, &with(b)).is_err());
    let (r, peak) = peak_during(|| predict(&model, &grid, None, &hard, &with(budget)));
    eprintln!("predict: budget {budget} bytes, peak {peak} bytes");
    assert!(r.is_ok(), "{r:?}");
    if peak as u64 > budget {
        over.push(format!("predict: peak {peak} > budget {budget}"));
    }
    assert!(over.is_empty(), "{over:?}");
}
