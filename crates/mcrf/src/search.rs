//! Sector (quadrant or octant) and nearest-neighbour search over the offset table.

use transiograms_core::{Code, Grid, MISSING};

use crate::Neighbourhood;

pub(crate) struct Search<'a> {
    offsets: &'a [[i64; 3]],
    count: [usize; 3],
    neighbourhood: Neighbourhood,
    sector: Vec<u8>,
    reachable: usize,
}

impl<'a> Search<'a> {
    /// `offsets` must be in table order and hold fewer than `u32::MAX` entries.
    pub(crate) fn new(offsets: &'a [[i64; 3]], grid: &Grid, neighbourhood: Neighbourhood) -> Self {
        let dims = if grid.is_2d() { 2 } else { 3 };
        let mut sector = Vec::with_capacity(offsets.len());
        sector.extend(
            offsets
                .iter()
                .map(|d| (0..dims).map(|a| ((d[a] >= 0) as u8) << a).sum::<u8>()),
        );
        let mut seen = [false; 8];
        for &s in &sector {
            seen[s as usize] = true;
        }
        Self {
            offsets,
            count: grid.count,
            neighbourhood,
            sector,
            reachable: seen.iter().filter(|&&s| s).count(),
        }
    }

    /// Upper bound on the neighbours `find` returns, so a buffer of this capacity never grows.
    pub(crate) fn capacity(&self) -> usize {
        self.wanted().min(self.offsets.len())
    }

    fn wanted(&self) -> usize {
        match self.neighbourhood {
            Neighbourhood::Sectors => self.reachable,
            Neighbourhood::Nearest(n) => n,
        }
    }

    /// Informed neighbours of `ijk` as `(table index, code)` in table order (closeness descending).
    pub(crate) fn find(&self, ijk: [usize; 3], informed: &[Code], out: &mut Vec<(u32, Code)>) {
        out.clear();
        let wanted = self.wanted();
        if wanted == 0 {
            return;
        }
        let mut filled = [false; 8];
        let [nx, ny, nz] = self.count.map(|c| c as i64);
        for (t, d) in self.offsets.iter().enumerate() {
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
            out.push((t as u32, code));
            if out.len() == wanted {
                return;
            }
        }
    }
}
