use std::mem::size_of;

use rayon::prelude::*;
use transiograms_core::{Code, Grid, MISSING, Mat, Prediction, Result, Snapped, invalid, memory};
use transiograms_transiogram::TransitionModel;

use crate::conditional::resolve;
use crate::search::Search;
use crate::{Options, product, sum};

const BLOCK: usize = 4096;
const WHAT: &str = "MCP prediction";

pub fn predict(
    model: &TransitionModel,
    grid: &Grid,
    mask: Option<&[bool]>,
    hard: &Snapped,
    opts: &Options,
) -> Result<Prediction> {
    crate::validate(model, grid, mask, hard)?;
    let k = model.k();
    let n = grid.n_cells();
    let (n64, k64) = (n as u64, k as u64);
    let probabilities_bytes = product(&[n64, k64, 8], WHAT)?;
    let output = sum(&[probabilities_bytes, product(&[n64, 8 + 4], WHAT)?], WHAT)?;
    memory::check(output, opts.memory, WHAT)?;

    let table = model.table(grid, opts.closeness_limit, opts.memory)?;
    let m = table.offsets.len();
    if m >= u32::MAX as usize {
        return invalid(format!(
            "{m} table offsets exceed the {} supported",
            u32::MAX
        ));
    }
    let search = Search::new(&table.offsets, grid, opts.neighbourhood);
    let per_offset =
        (size_of::<[i64; 3]>() + size_of::<f64>() + size_of::<Mat>() + 1) as u64 + k64 * k64 * 8;
    let blocks = n.div_ceil(BLOCK) as u64;
    let computing = sum(
        &[
            product(&[n64, size_of::<Code>() as u64], WHAT)?,
            probabilities_bytes,
            product(&[m as u64, per_offset], WHAT)?,
            product(
                &[
                    blocks,
                    search.capacity() as u64,
                    size_of::<(u32, Code)>() as u64,
                ],
                WHAT,
            )?,
        ],
        WHAT,
    )?;
    memory::check(output.max(computing), opts.memory, WHAT)?;

    let mut informed = vec![MISSING; n];
    crate::place_hard(hard, &mut informed);
    let p = &model.proportions;
    let transition = |t: u32| table.transition[t as usize].as_slice();
    let mut probabilities = vec![f64::NAN; n * k];
    probabilities
        .par_chunks_mut(BLOCK * k)
        .enumerate()
        .for_each(|(b, block)| {
            let mut found = Vec::with_capacity(search.capacity());
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
                resolve(p, &found, transition, row);
            }
        });
    drop(search);
    drop(table);
    drop(informed);
    Ok(Prediction::from_probabilities(k, probabilities, 0.0))
}
