//! The measured gap between T-PROGS ensembles and their model, quoted in
//! docs/guide/tp-simulation.md. A change that moves a number out of its band must update that
//! page.

use transiograms_core::*;
use transiograms_tpsim::{Options, QuenchOptions, quench_lags, simulate};
use transiograms_transiogram::{FitQuality, TransitionModel, fit_quality};

fn m5_model() -> TransitionModel {
    let pi = Mat::from_row_slice(
        3,
        3,
        &[
            0.0,
            0.6,
            0.4,
            0.7,
            0.0,
            0.3,
            0.2105263158,
            0.7894736842,
            0.0,
        ],
    );
    TransitionModel::from_geology(
        &[0.5, 0.3, 0.2],
        [&[12.0, 8.0, 6.0], &[12.0, 8.0, 6.0], &[2.0, 1.0, 0.8]],
        [None, None, Some(&pi)],
        2,
        [0.0; 3],
    )
    .unwrap()
}

const AXES: [[i64; 3]; 3] = [[1, 0, 0], [0, 0, 1], [0, 0, -1]];

fn report(quench: Option<QuenchOptions>) -> (TransitionModel, Grid, FitQuality) {
    let model = m5_model();
    let grid = Grid::new([0.0; 3], [3.0, 3.0, 0.5], [48, 48, 32]).unwrap();
    let empty = grid.snap(&[], &[], None).unwrap();
    let opts = Options {
        n_realizations: 10,
        seed: 1,
        quench,
        ..Options::default()
    };
    let sims = simulate(&model, &grid, None, &empty, &opts).unwrap();
    let q = fit_quality(&model, &grid, None, &sims, &AXES, 6).unwrap();
    (model, grid, q)
}

/// Method limit: sequential indicator simulation from cokriging leaves minority categories too
/// continuous at mid lags, and the quench fits only the lags it targets (closeness above
/// `limit`, here the adjacent cells). Measured at seed 1 on the 48 x 48 x 32 M5 grid,
/// 10 realizations, default quench: proportion error 0.0001, transiogram error 0.026
/// (15 standard errors), largest at lags 2 to 4, while lag 1 is within 0.001.
#[test]
fn default_quench_gap_is_the_documented_limit() {
    let (model, grid, q) = report(Some(QuenchOptions::default()));
    let (p, e, z) = (
        q.proportion_error(),
        q.transiogram_error(),
        q.max_standard_errors(),
    );
    assert!(p < 0.001, "proportion error {p:.5}");
    assert!((0.02..0.032).contains(&e), "transiogram error {e:.4}");
    assert!(z > 10.0, "{z:.1} standard errors");
    let targeted = quench_lags(&model, &grid, QuenchOptions::default().limit);
    for (d, (m, t)) in q.offsets.iter().zip(q.mean.iter().zip(&q.model)) {
        if targeted.contains(d) {
            let gap = (m - t).abs().max();
            assert!(gap < 0.002, "quenched offset {d:?}: {gap:.4}");
        }
    }
}

/// Without the quench: proportion error 0.018, transiogram error 0.075 at the same seed.
#[test]
fn unquenched_gap_is_the_documented_limit() {
    let (_, _, q) = report(None);
    let (p, e) = (q.proportion_error(), q.transiogram_error());
    assert!((0.012..0.024).contains(&p), "proportion error {p:.4}");
    assert!((0.06..0.09).contains(&e), "transiogram error {e:.4}");
}
