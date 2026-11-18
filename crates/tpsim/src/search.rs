//! Closeness-ranked neighbour search over the offset table.

use std::borrow::Cow;
use std::collections::HashMap;

use varvite_core::{Code, Grid, MISSING, Mat};
use varvite_transiogram::{Table, TransitionModel};

/// The offset table with an index for pair lookups; lags outside it fall back to the model.
pub(crate) struct Neighbourhood<'a> {
    pub model: &'a TransitionModel,
    pub grid: &'a Grid,
    pub table: &'a Table,
    index: HashMap<[i64; 3], usize>,
}

impl<'a> Neighbourhood<'a> {
    pub fn new(model: &'a TransitionModel, grid: &'a Grid, table: &'a Table) -> Self {
        let index = table
            .offsets
            .iter()
            .enumerate()
            .map(|(i, &d)| (d, i))
            .collect();
        Self {
            model,
            grid,
            table,
            index,
        }
    }

    /// `T(d * size)`.
    pub fn transition(&self, d: [i64; 3]) -> Cow<'_, Mat> {
        if d == [0; 3] {
            return Cow::Owned(Mat::identity(self.model.k(), self.model.k()));
        }
        match self.index.get(&d) {
            Some(&i) => Cow::Borrowed(&self.table.transition[i]),
            None => Cow::Owned(self.model.transition(self.grid.lag(d))),
        }
    }

    /// The first `max_data` informed cells in table order, at most `max_per_octant` per octant
    /// when it is positive; `out` receives `(offset from cell, code)`.
    pub fn search(
        &self,
        cell: usize,
        informed: &[Code],
        max_data: usize,
        max_per_octant: usize,
        out: &mut Vec<([i64; 3], Code)>,
    ) {
        out.clear();
        if max_data == 0 {
            return;
        }
        let mut per_octant = [0usize; 8];
        for &d in &self.table.offsets {
            let Some(other) = self.grid.offset(cell, d) else {
                continue;
            };
            let code = informed[other];
            if code == MISSING {
                continue;
            }
            if max_per_octant > 0 {
                let o = octant(d);
                if per_octant[o] >= max_per_octant {
                    continue;
                }
                per_octant[o] += 1;
            }
            out.push((d, code));
            if out.len() == max_data {
                break;
            }
        }
    }
}

fn octant(d: [i64; 3]) -> usize {
    (0..3).fold(0, |o, a| o | (usize::from(d[a] < 0) << a))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn octants_follow_sign_patterns() {
        assert_eq!(octant([0, 0, 0]), 0);
        assert_eq!(octant([1, 2, 3]), 0);
        assert_eq!(octant([-1, 0, 0]), 1);
        assert_eq!(octant([0, -1, 0]), 2);
        assert_eq!(octant([-1, -1, -1]), 7);
    }
}
