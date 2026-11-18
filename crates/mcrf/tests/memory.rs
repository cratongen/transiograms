use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Mutex;
use std::sync::atomic::{AtomicI64, Ordering};

use varvite_core::{Grid, Mat, RateMatrix, Snapped};
use varvite_mcrf::{Neighbourhood, Options, predict, simulate};
use varvite_transiogram::TransitionModel;

struct Counting;

static CURRENT: AtomicI64 = AtomicI64::new(0);
static PEAK: AtomicI64 = AtomicI64::new(0);
static SERIAL: Mutex<()> = Mutex::new(());

thread_local! {
    static TRACKED: Cell<bool> = const { Cell::new(false) };
}

/// Counts only the thread inside `peak_of`, so the test harness's own threads add no noise.
fn tracked() -> bool {
    TRACKED.try_with(Cell::get).unwrap_or(false)
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() && tracked() {
            let size = layout.size() as i64;
            let now = CURRENT.fetch_add(size, Ordering::SeqCst) + size;
            PEAK.fetch_max(now, Ordering::SeqCst);
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        if tracked() {
            CURRENT.fetch_sub(layout.size() as i64, Ordering::SeqCst);
        }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

const P: [f64; 3] = [0.5, 0.3, 0.2];

fn model() -> TransitionModel {
    let z = RateMatrix::new(Mat::from_row_slice(
        3,
        3,
        &[-0.25, 0.15, 0.10, 0.35, -0.5, 0.15, 0.1, 0.375, -0.475],
    ))
    .unwrap();
    let lateral = RateMatrix::max_entropy(&P, &[3.0, 2.0, 1.5]).unwrap();
    TransitionModel::new(P.to_vec(), [lateral.clone(), lateral, z], [0.0; 3], 2).unwrap()
}

fn empty() -> Snapped {
    Snapped {
        cells: vec![],
        codes: vec![],
        conflicts: 0,
        outside: 0,
    }
}

/// Peak bytes allocated by `f` above what was live when it started, on one rayon thread.
fn peak_of<T>(f: impl FnOnce() -> T + Send) -> (T, usize)
where
    T: Send,
{
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    pool.install(|| {
        CURRENT.store(0, Ordering::SeqCst);
        PEAK.store(0, Ordering::SeqCst);
        TRACKED.with(|t| t.set(true));
        let out = f();
        TRACKED.with(|t| t.set(false));
        (out, PEAK.load(Ordering::SeqCst) as usize)
    })
}

#[test]
fn simulate_with_huge_realization_count_returns_an_error_instead_of_panicking() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let grid = Grid::new([0.0; 3], [1.0; 3], [4, 1, 1]).unwrap();
    let opts = Options {
        n_realizations: 1usize << 62,
        closeness_limit: 0.3,
        ..Options::default()
    };
    let r = std::panic::catch_unwind(|| simulate(&model(), &grid, None, &empty(), &opts));
    match r {
        Ok(Err(_)) => {}
        Ok(Ok(real)) => panic!(
            "accepted n = 2^62 and returned {} codes for n * n_cells = 2^64 cells",
            real.codes.len()
        ),
        Err(_) => panic!("simulate panicked on n_realizations = 2^62 instead of returning Err"),
    }
}

/// Smallest budget `TransitionModel::table` accepts, by bisection.
fn smallest_table_budget(model: &TransitionModel, grid: &Grid, limit: f64) -> u64 {
    let (mut lo, mut hi) = (0u64, 1u64 << 32);
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if model.table(grid, limit, Some(mid)).is_ok() {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi
}

#[test]
fn simulate_never_allocates_more_than_the_budget_it_accepted() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let grid = Grid::new([0.0; 3], [1.0; 3], [60, 60, 60]).unwrap();
    let model = model();
    let table = model.table(&grid, 0.3, None).unwrap();
    let budget = (grid.n_cells() as u64 * 4 + table.offsets.len() as u64 * (9 + 4) * 8)
        .max(smallest_table_budget(&model, &grid, 0.3));
    let opts = Options {
        n_realizations: 1,
        closeness_limit: 0.3,
        memory: Some(budget),
        ..Options::default()
    };
    let (result, peak) = peak_of(|| simulate(&model, &grid, None, &empty(), &opts));
    assert!(
        result.is_ok(),
        "the budget the library itself states was refused: {:?}",
        result.err()
    );
    assert!(
        peak as u64 <= budget,
        "simulate accepted a budget of {budget} bytes but allocated {peak} bytes at peak"
    );
}

#[test]
fn predict_never_allocates_more_than_the_budget_it_accepted() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let grid = Grid::new([0.0; 3], [1.0; 3], [60, 60, 60]).unwrap();
    let model = model();
    let n = grid.n_cells() as u64;
    let table = model.table(&grid, 0.3, None).unwrap();
    let table_bytes = table.offsets.len() as u64 * (9 + 4) * 8;
    let budget = n * (3 * 8 + 8 + 4) + table_bytes;
    let opts = Options {
        closeness_limit: 0.3,
        memory: Some(budget),
        ..Options::default()
    };
    let (result, peak) = peak_of(|| predict(&model, &grid, None, &empty(), &opts));
    assert!(result.is_ok(), "{:?}", result.err());
    assert!(
        peak as u64 <= budget,
        "predict accepted a budget of {budget} bytes but allocated {peak} bytes at peak"
    );
}

fn scattered(n_cells: usize, every: usize) -> Snapped {
    let cells: Vec<usize> = (0..n_cells).step_by(every).collect();
    let codes = cells.iter().map(|&c| (c % 3) as u32).collect();
    Snapped {
        cells,
        codes,
        conflicts: 0,
        outside: 0,
    }
}

#[test]
fn simulate_peak_stays_within_its_stated_budget_with_hard_data_and_mask() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let grid = Grid::new([0.0; 3], [1.0; 3], [60, 60, 60]).unwrap();
    let n_cells = grid.n_cells();
    let model = model();
    let mask: Vec<bool> = (0..n_cells).map(|c| c % 7 != 3).collect();
    let mut hard = scattered(n_cells, 97);
    let keep: Vec<usize> = (0..hard.cells.len())
        .filter(|&i| mask[hard.cells[i]])
        .collect();
    hard.cells = keep.iter().map(|&i| hard.cells[i]).collect();
    hard.codes = keep.iter().map(|&i| hard.codes[i]).collect();
    for neighbourhood in [Neighbourhood::Sectors, Neighbourhood::Nearest(6)] {
        let n = 3u64;
        let m = model.table(&grid, 0.3, None).unwrap().offsets.len() as u64;
        let found = match neighbourhood {
            Neighbourhood::Sectors => 8,
            Neighbourhood::Nearest(q) => q as u64,
        }
        .min(m);
        let budget = n * n_cells as u64 * 4 + n * 8 + m * (24 + 9 * 8 + 1) + n * (found * 8 + 24);
        let opts = Options {
            neighbourhood,
            n_realizations: n as usize,
            closeness_limit: 0.3,
            memory: Some(budget),
            ..Options::default()
        };
        let (result, peak) = peak_of(|| simulate(&model, &grid, Some(&mask), &hard, &opts));
        assert!(result.is_ok(), "{neighbourhood:?}: {:?}", result.err());
        assert!(
            peak as u64 <= budget,
            "{neighbourhood:?}: budget {budget} bytes, peak {peak} bytes"
        );
    }
}

#[test]
fn predict_peak_stays_within_its_stated_budget_with_hard_data() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let grid = Grid::new([0.0; 3], [1.0; 3], [60, 60, 60]).unwrap();
    let n = grid.n_cells() as u64;
    let model = model();
    let hard = scattered(grid.n_cells(), 53);
    let budget = n * (3 * 8 + 8 + 4);
    let opts = Options {
        neighbourhood: Neighbourhood::Nearest(6),
        closeness_limit: 0.3,
        memory: Some(budget),
        ..Options::default()
    };
    let (result, peak) = peak_of(|| predict(&model, &grid, None, &hard, &opts));
    assert!(result.is_ok(), "{:?}", result.err());
    assert!(
        peak as u64 <= budget,
        "budget {budget} bytes, peak {peak} bytes"
    );
}

#[test]
fn simulate_and_predict_reject_budgets_below_their_output() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let grid = Grid::new([0.0; 3], [1.0; 3], [60, 60, 60]).unwrap();
    let n = grid.n_cells() as u64;
    let opts = |memory: u64, n_realizations: usize| Options {
        n_realizations,
        closeness_limit: 0.3,
        memory: Some(memory),
        ..Options::default()
    };
    let (r, peak) = peak_of(|| simulate(&model(), &grid, None, &empty(), &opts(2 * n * 4 + 15, 2)));
    let e = r.unwrap_err();
    assert!(
        e.to_string()
            .contains(&format!("needs {} bytes", 2 * n * 4 + 16)),
        "{e}"
    );
    assert!(peak < 4096, "allocated {peak} bytes before refusing");
    let (r, peak) = peak_of(|| predict(&model(), &grid, None, &empty(), &opts(n * 36 - 1, 1)));
    let e = r.unwrap_err();
    assert!(
        e.to_string().contains(&format!("needs {} bytes", n * 36)),
        "{e}"
    );
    assert!(peak < 4096, "allocated {peak} bytes before refusing");
}
