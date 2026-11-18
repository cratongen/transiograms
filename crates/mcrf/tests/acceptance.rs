//! M6: MCP rows are distributions, hard cells one-hot, and an MCRF ensemble reproduces the
//! proportions and the hard data on the tpsim acceptance model.

use varvite_core::rng::tag;
use varvite_core::synthetic::markov_codes;
use varvite_core::{Code, Grid, Mat, Snapped, Stream, TransitionMatrix};
use varvite_mcrf::{Options, predict, simulate};
use varvite_transiogram::TransitionModel;

const P: [f64; 3] = [0.5, 0.3, 0.2];
const SIZE: [f64; 3] = [4.0, 4.0, 1.0];
const COUNT: [usize; 3] = [96, 96, 32];

fn model() -> TransitionModel {
    let golden_embedded = Mat::from_row_slice(3, 3, &[0.0, 0.6, 0.4, 0.7, 0.0, 0.3, 0.2, 0.8, 0.0]);
    let lateral = [12.0, 8.0, 6.0];
    TransitionModel::from_geology(
        &P,
        [&lateral, &lateral, &[2.0, 1.0, 0.8]],
        [None, None, Some(&golden_embedded)],
        2,
        [0.0; 3],
    )
    .unwrap()
}

/// 16 vertical holes on a regular 4 x 4 pattern, each a stationary draw of the model's z chain.
fn holes(model: &TransitionModel, grid: &Grid) -> Snapped {
    let up = TransitionMatrix::new(model.transition([0.0, 0.0, grid.size[2]])).unwrap();
    let (sx, sy) = (grid.count[0] / 4, grid.count[1] / 4);
    let mut coords = Vec::new();
    let mut codes: Vec<Code> = Vec::new();
    for h in 0..16 {
        let (i, j) = (sx / 2 + sx * (h / 4), sy / 2 + sy * (h % 4));
        let mut stream = Stream::new(0, &[tag::SAMPLE, h as u64]);
        let column = markov_codes(&up, None, grid.count[2], &mut stream).unwrap();
        for (z, &c) in column.iter().enumerate() {
            coords.push(grid.centroid(grid.index([i, j, z])));
            codes.push(c);
        }
    }
    grid.snap(&coords, &codes, None).unwrap()
}

/// tpsim.md section 5: `Var(p_hat_k) ~ |D|^-2 sum_x sum_y p_k (t_kk(y - x) - p_k)`.
fn proportion_sd(model: &TransitionModel, grid: &Grid, limit: f64) -> Vec<f64> {
    let table = model.table(grid, limit, None).unwrap();
    let n = grid.n_cells() as f64;
    (0..3)
        .map(|k| {
            let mut sum = n * P[k] * (1.0 - P[k]);
            for (d, t) in table.offsets.iter().zip(&table.transition) {
                let pairs: f64 = (0..3)
                    .map(|v| (grid.count[v] as i64 - d[v].abs()) as f64)
                    .product();
                sum += pairs * P[k] * (t[(k, k)] - P[k]);
            }
            (sum / (n * n)).sqrt()
        })
        .collect()
}

#[test]
fn m6_acceptance() {
    let grid = Grid::new([0.0; 3], SIZE, COUNT).unwrap();
    let model = model();
    let opts = Options {
        n_realizations: 20,
        seed: 0,
        ..Options::default()
    };
    for (k, sd) in proportion_sd(&model, &grid, opts.closeness_limit)
        .iter()
        .enumerate()
    {
        assert!(3.0 * sd < 0.02, "category {k}: 3 sigma = {}", 3.0 * sd);
    }

    let hard = holes(&model, &grid);
    assert_eq!(hard.cells.len(), 16 * COUNT[2]);
    let pred = predict(&model, &grid, None, &hard, &opts).unwrap();
    for row in pred.probabilities.chunks(3) {
        assert!(row.iter().all(|&p| p >= 0.0));
        assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
    for (&cell, &code) in hard.cells.iter().zip(&hard.codes) {
        let row = &pred.probabilities[cell * 3..cell * 3 + 3];
        for k in 0..3 {
            assert_eq!(row[k], (k as Code == code) as u8 as f64);
        }
    }

    let real = simulate(&model, &grid, None, &hard, &opts).unwrap();
    let n_cells = grid.n_cells();
    let mut counts = [0f64; 3];
    for r in 0..real.n {
        let codes = &real.codes[r * n_cells..(r + 1) * n_cells];
        for (&cell, &code) in hard.cells.iter().zip(&hard.codes) {
            assert_eq!(codes[cell], code, "realization {r} cell {cell}");
        }
        for &c in codes {
            counts[c as usize] += 1.0;
        }
    }
    let total: f64 = counts.iter().sum();
    for k in 0..3 {
        let prop = counts[k] / total;
        assert!(
            (prop - P[k]).abs() < 0.02,
            "category {k}: {prop} vs {}",
            P[k]
        );
    }
}
