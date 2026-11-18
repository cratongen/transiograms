use std::mem::size_of;

use rayon::prelude::*;
use varvite_core::rng::{splitmix, tag};
use varvite_core::{
    Code, Grid, MISSING, Mat, Realizations, Result, Snapped, Stream, invalid, memory,
};
use varvite_transiogram::{Table, TransitionModel};

use crate::conditional::resolve;
use crate::search::Search;
use crate::{Neighbourhood, Options, product, sum};

const WHAT: &str = "MCRF simulation";
const ROUNDS: usize = 6;

/// Seeded pseudo-random permutation of `0..4^half_bits` (a balanced Feistel network), so the
/// random path costs no memory per cell.
pub(crate) struct Path {
    half_bits: u32,
    keys: [u64; ROUNDS],
}

impl Path {
    pub(crate) fn new(n_cells: usize, stream: &mut Stream) -> Self {
        let mut half_bits = 1;
        while (1u128 << (2 * half_bits)) < n_cells as u128 {
            half_bits += 1;
        }
        Self {
            half_bits,
            keys: std::array::from_fn(|_| stream.next_u64()),
        }
    }

    pub(crate) fn len(&self) -> u64 {
        1 << (2 * self.half_bits)
    }

    pub(crate) fn at(&self, i: u64) -> u64 {
        let mask = (1u64 << self.half_bits) - 1;
        let (mut l, mut r) = (i >> self.half_bits, i & mask);
        for key in self.keys {
            (l, r) = (r, l ^ (splitmix(r ^ key) & mask));
        }
        (l << self.half_bits) | r
    }
}

pub fn simulate(
    model: &TransitionModel,
    grid: &Grid,
    mask: Option<&[bool]>,
    hard: &Snapped,
    opts: &Options,
) -> Result<Realizations> {
    crate::validate(model, grid, mask, hard)?;
    let k = model.k();
    let n_cells = grid.n_cells();
    let n = opts.n_realizations;
    let (n64, k64) = (n as u64, k as u64);
    let codes_bytes = product(&[n64, n_cells as u64, size_of::<Code>() as u64], WHAT)?;
    let fixed = sum(&[codes_bytes, product(&[n64, 8], WHAT)?], WHAT)?;
    memory::check(fixed, opts.memory, WHAT)?;

    let table = model.table(grid, opts.closeness_limit, opts.memory)?;
    let m = table.offsets.len();
    if m >= u32::MAX as usize {
        return invalid(format!(
            "{m} table offsets exceed the {} supported",
            u32::MAX
        ));
    }
    let (m64, kk) = (m as u64, k64 * k64 * 8);
    let offset_bytes = size_of::<[i64; 3]>() as u64;
    let converting = product(
        &[m64, offset_bytes + size_of::<Mat>() as u64 + 2 * kk],
        WHAT,
    )?;
    let sectors = if grid.is_2d() { 4 } else { 8 };
    let found = match opts.neighbourhood {
        Neighbourhood::Sectors => sectors,
        Neighbourhood::Nearest(q) => q,
    }
    .min(m) as u64;
    let scratch = found * size_of::<(u32, Code)>() as u64 + k64 * 8;
    let running = sum(
        &[
            fixed,
            product(&[m64, offset_bytes + kk + 1], WHAT)?,
            product(&[n64, scratch], WHAT)?,
        ],
        WHAT,
    )?;
    memory::check(running.max(converting), opts.memory, WHAT)?;

    let Table {
        offsets,
        closeness,
        transition,
    } = table;
    drop(closeness);
    let mut flat = Vec::with_capacity(m * k * k);
    for t in &transition {
        flat.extend_from_slice(t.as_slice());
    }
    drop(transition);
    let transition = |t: u32| &flat[t as usize * k * k..(t as usize + 1) * k * k];
    let search = Search::new(&offsets, grid, opts.neighbourhood);
    let p = &model.proportions;

    let mut out = vec![MISSING; n * n_cells];
    out.par_chunks_mut(n_cells.max(1))
        .enumerate()
        .for_each(|(r, codes)| {
            let r = r as u64;
            crate::place_hard(hard, codes);
            let path = Path::new(n_cells, &mut Stream::new(opts.seed, &[r, tag::PATH]));
            let mut draw = Stream::new(opts.seed, &[r, tag::DRAW]);
            let mut found = Vec::with_capacity(search.capacity());
            let mut probabilities = vec![0.0; k];
            for i in 0..path.len() {
                let cell = path.at(i) as usize;
                if cell >= n_cells || codes[cell] != MISSING || mask.is_some_and(|m| !m[cell]) {
                    continue;
                }
                search.find(grid.ijk(cell), codes, &mut found);
                resolve(p, &found, transition, &mut probabilities);
                codes[cell] = draw.categorical(&probabilities) as Code;
            }
        });

    Ok(Realizations {
        n,
        n_cells,
        k,
        codes: out,
        corrected: vec![0.0; n],
        objective: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_is_a_bijection_by_brute_force() {
        for n_cells in [1usize, 2, 3, 4, 5, 15, 16, 17, 63, 64, 65, 1000] {
            for seed in 0..5 {
                let path = Path::new(n_cells, &mut Stream::new(seed, &[tag::PATH]));
                assert!(path.len() >= n_cells as u64 && path.len() < 4 * n_cells.max(2) as u64);
                let mut seen = vec![false; path.len() as usize];
                for i in 0..path.len() {
                    let c = path.at(i) as usize;
                    assert!(!seen[c], "n_cells {n_cells} seed {seed}: {c} visited twice");
                    seen[c] = true;
                }
            }
        }
    }

    #[test]
    fn first_cell_of_the_path_is_uniform() {
        let n_cells = 5usize;
        let draws = 50_000u64;
        let mut counts = [0f64; 5];
        for seed in 0..draws {
            let path = Path::new(n_cells, &mut Stream::new(seed, &[0, tag::PATH]));
            let first = (0..path.len())
                .map(|i| path.at(i) as usize)
                .find(|&c| c < n_cells)
                .unwrap();
            counts[first] += 1.0;
        }
        let expected = draws as f64 / n_cells as f64;
        let chi2: f64 = counts
            .iter()
            .map(|c| (c - expected).powi(2) / expected)
            .sum();
        assert!(
            chi2 < 18.47,
            "chi-square {chi2} with 4 dof, counts {counts:?}"
        );
    }
}
