use rayon::prelude::*;
use varvite_core::{Code, Grid, MISSING, Mat, Prediction, Result, Snapped, memory};
use varvite_transiogram::TransitionModel;

use crate::Options;
use crate::conditional::resolve;
use crate::search::Search;

const BLOCK: usize = 4096;

pub fn predict(
    model: &TransitionModel,
    grid: &Grid,
    mask: Option<&[bool]>,
    hard: &Snapped,
    opts: &Options,
) -> Result<Prediction> {
    let informed = crate::hard_codes(model, grid, mask, hard)?;
    let k = model.k();
    let n = grid.n_cells();
    memory::check(
        (n as u64) * (k as u64 * 8 + 8 + 4),
        opts.memory,
        "MCP prediction",
    )?;
    let table = model.table(grid, opts.closeness_limit, opts.memory)?;
    let search = Search::new(&table, grid, opts.neighbourhood);
    let p = &model.proportions;
    let mut probabilities = vec![f64::NAN; n * k];
    probabilities
        .par_chunks_mut(BLOCK * k)
        .enumerate()
        .for_each(|(b, block)| {
            let mut found: Vec<(usize, Code)> = Vec::new();
            let mut terms: Vec<(&Mat, Code)> = Vec::new();
            for (i, row) in block.chunks_mut(k).enumerate() {
                let cell = b * BLOCK + i;
                if mask.is_some_and(|m| !m[cell]) {
                    continue;
                }
                let code = informed[cell];
                if code != MISSING {
                    row.fill(0.0);
                    row[code as usize] = 1.0;
                    continue;
                }
                search.find(grid.ijk(cell), &informed, &mut found);
                terms.clear();
                terms.extend(found.iter().map(|&(t, c)| (&table.transition[t], c)));
                resolve(p, &terms, row);
            }
        });
    Ok(Prediction::from_probabilities(k, probabilities, 0.0))
}
