//! Sector (quadrant or octant) and nearest-neighbour search over the offset table.

use varvite_core::{Code, Grid, MISSING};
use varvite_transiogram::Table;

use crate::Neighbourhood;

pub(crate) struct Search<'a> {
    table: &'a Table,
    count: [usize; 3],
    neighbourhood: Neighbourhood,
    sector: Vec<u8>,
    reachable: usize,
}

impl<'a> Search<'a> {
    pub(crate) fn new(table: &'a Table, grid: &Grid, neighbourhood: Neighbourhood) -> Self {
        let dims = if grid.is_2d() { 2 } else { 3 };
        let sector: Vec<u8> = table
            .offsets
            .iter()
            .map(|d| (0..dims).map(|a| ((d[a] >= 0) as u8) << a).sum())
            .collect();
        let mut seen = [false; 8];
        for &s in &sector {
            seen[s as usize] = true;
        }
        Self {
            table,
            count: grid.count,
            neighbourhood,
            sector,
            reachable: seen.iter().filter(|&&s| s).count(),
        }
    }

    /// Informed neighbours of `ijk` as `(table index, code)` in table order (closeness descending).
    pub(crate) fn find(&self, ijk: [usize; 3], informed: &[Code], out: &mut Vec<(usize, Code)>) {
        out.clear();
        let wanted = match self.neighbourhood {
            Neighbourhood::Sectors => self.reachable,
            Neighbourhood::Nearest(n) => n,
        };
        if wanted == 0 {
            return;
        }
        let mut filled = [false; 8];
        let [nx, ny, nz] = self.count.map(|c| c as i64);
        for (t, d) in self.table.offsets.iter().enumerate() {
            let (i, j, k) = (
                ijk[0] as i64 + d[0],
                ijk[1] as i64 + d[1],
                ijk[2] as i64 + d[2],
            );
            if i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz {
                continue;
            }
            let code = informed[(i + nx * (j + ny * k)) as usize];
            if code == MISSING {
                continue;
            }
            if let Neighbourhood::Sectors = self.neighbourhood {
                let s = self.sector[t] as usize;
                if filled[s] {
                    continue;
                }
                filled[s] = true;
            }
            out.push((t, code));
            if out.len() == wanted {
                return;
            }
        }
    }
}
