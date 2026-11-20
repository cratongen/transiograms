//! The measured gap between MCRF ensembles and their model, quoted in docs/guide/mcrf-mcp.md.
//! A change that moves a number out of its band must update that page.

use transiograms_core::*;
use transiograms_mcrf::{Neighbourhood, Options, simulate};
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

fn report(count: [usize; 3], n: usize, nb: Neighbourhood, dirs: &[[i64; 3]]) -> FitQuality {
    let model = m5_model();
    let grid = Grid::new([0.0; 3], [3.0, 3.0, 0.5], count).unwrap();
    let empty = grid.snap(&[], &[], None).unwrap();
    let opts = Options {
        n_realizations: n,
        seed: 1,
        neighbourhood: nb,
        ..Options::default()
    };
    let sims = simulate(&model, &grid, None, &empty, &opts).unwrap();
    fit_quality(&model, &grid, None, &sims, dirs, 6).unwrap()
}

const AXES: [[i64; 3]; 3] = [[1, 0, 0], [0, 0, 1], [0, 0, -1]];

/// Along a single column the nearest cell above and below screen every other cell of a Markov
/// chain, so Eq. 14 over the two vertical sectors is the exact Markov bridge. The simulation
/// must then reproduce the model within sampling error; this rules out indexing, search or
/// path bugs as the cause of the 3D gap below.
#[test]
fn one_column_sectors_simulation_reproduces_the_markov_chain() {
    let q = report([1, 1, 20_000], 20, Neighbourhood::Sectors, &AXES[1..]);
    let (z, e, p) = (
        q.max_standard_errors(),
        q.transiogram_error(),
        q.proportion_error(),
    );
    assert!(
        z < 4.0 && e < 0.01 && p < 0.005,
        "z {z:.2} e {e:.4} p {p:.4}"
    );
}

/// Method limit: with more than one non-collinear neighbour, conditional independence counts
/// shared information twice, so unconditional 3D realizations come out too continuous and
/// drift from the target proportions. Measured at seed 1 on the 48 x 48 x 32 M5 grid,
/// 10 realizations: proportion error 0.038, transiogram error 0.190 (31 standard errors).
#[test]
fn unconditional_sectors_gap_is_the_documented_limit() {
    let q = report([48, 48, 32], 10, Neighbourhood::Sectors, &AXES);
    let (p, e, z) = (
        q.proportion_error(),
        q.transiogram_error(),
        q.max_standard_errors(),
    );
    assert!((0.03..0.05).contains(&p), "proportion error {p:.4}");
    assert!((0.17..0.21).contains(&e), "transiogram error {e:.4}");
    assert!(z > 20.0, "{z:.1} standard errors");
    let mean = q.mean_proportions();
    assert!(
        mean[0] < 0.47,
        "majority category under-represented: {mean:?}"
    );
    let up = q.offsets.iter().position(|&d| d == [0, 0, 1]).unwrap();
    let t11 = q.mean[up][(1, 1)];
    assert!(t11 > q.model[up][(1, 1)] + 0.1, "vertical t_11(1) {t11:.3}");
}

/// The gap depends on the neighbourhood: Nearest(4) 0.088 at the same seed and grid.
#[test]
fn unconditional_nearest_four_gap_is_the_documented_limit() {
    let e = report([48, 48, 32], 10, Neighbourhood::Nearest(4), &AXES).transiogram_error();
    assert!((0.07..0.11).contains(&e), "transiogram error {e:.4}");
}
