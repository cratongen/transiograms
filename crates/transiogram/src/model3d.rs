use serde::{Deserialize, Serialize};
use varvite_core::{Error, Mat, RateMatrix, Result, invalid};

use crate::model1d::check_proportions;
use varvite_core::numerics::expm_rate;

const AXES: [&str; 3] = ["x'", "y'", "z'"];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransitionModel {
    pub proportions: Vec<f64>,
    /// Principal x', y', z' (z' up), each the +direction generator.
    pub rates: [RateMatrix; 3],
    /// Azimuth, dip, rake in degrees.
    pub rotation: [f64; 3],
    pub background: usize,
}

fn comb(a: f64, u: [f64; 3], b: f64, v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| a * u[i] + b * v[i])
}

fn dot(u: [f64; 3], v: [f64; 3]) -> f64 {
    u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
}

fn norm(h: [f64; 3]) -> f64 {
    dot(h, h).sqrt()
}

fn on_axis(v: usize, e: Error) -> Error {
    match e {
        Error::InvalidInput(m) => Error::InvalidInput(format!("axis {}: {m}", AXES[v])),
        Error::Convergence(m) => Error::Convergence(format!("axis {}: {m}", AXES[v])),
    }
}

impl TransitionModel {
    /// Each rate matrix valid with `p^T R = 0` within `1e-10 max|r|`.
    pub fn new(
        proportions: Vec<f64>,
        rates: [RateMatrix; 3],
        rotation: [f64; 3],
        background: usize,
    ) -> Result<Self> {
        let k = proportions.len();
        check_proportions(&proportions, k)?;
        if background >= k {
            return invalid(format!("background {background} is not a category of {k}"));
        }
        if let Some(a) = rotation.iter().position(|a| !a.is_finite()) {
            return invalid(format!(
                "rotation angle {a} = {} is not finite",
                rotation[a]
            ));
        }
        for (v, r) in rates.iter().enumerate() {
            if r.k() != k {
                return invalid(format!(
                    "axis {} rates are {}x{}, proportions have {k} categories",
                    AXES[v],
                    r.k(),
                    r.k()
                ));
            }
            RateMatrix::new(r.matrix().clone()).map_err(|e| on_axis(v, e))?;
            let m = r.matrix();
            let scale = m.iter().fold(0.0, |a: f64, x| a.max(x.abs()));
            for j in 0..k {
                let s: f64 = (0..k).map(|i| proportions[i] * m[(i, j)]).sum();
                if s.abs() > 1e-10 * scale {
                    return invalid(format!(
                        "axis {} rates miss p^T R = 0 by {s} in column {j}",
                        AXES[v]
                    ));
                }
            }
        }
        Ok(Self {
            proportions,
            rates,
            rotation,
            background,
        })
    }

    /// Per axis: embedded given -> background fill; None -> maximum entropy (symmetric).
    pub fn from_geology(
        proportions: &[f64],
        mean_lengths: [&[f64]; 3],
        embedded: [Option<&Mat>; 3],
        background: usize,
        rotation: [f64; 3],
    ) -> Result<Self> {
        let k = proportions.len();
        check_proportions(proportions, k)?;
        if background >= k {
            return invalid(format!("background {background} is not a category of {k}"));
        }
        let mut rates = Vec::with_capacity(3);
        for v in 0..3 {
            let l = mean_lengths[v];
            if l.len() != k {
                return invalid(format!(
                    "axis {}: give {k} mean lengths, not {}",
                    AXES[v],
                    l.len()
                ));
            }
            let r = match embedded[v] {
                None => RateMatrix::max_entropy(proportions, l),
                Some(e) => {
                    if e.nrows() != k || e.ncols() != k {
                        return invalid(format!(
                            "axis {}: embedded matrix is {}x{}, not {k}x{k}",
                            AXES[v],
                            e.nrows(),
                            e.ncols()
                        ));
                    }
                    let q = background;
                    for i in (0..k).filter(|&i| i != q) {
                        if !(l[i] > 0.0 && l[i].is_finite()) {
                            return invalid(format!(
                                "axis {}: mean length {i} = {} must be positive and finite",
                                AXES[v], l[i]
                            ));
                        }
                        for j in (0..k).filter(|&j| j != q && j != i) {
                            if !(0.0..=1.0).contains(&e[(i, j)]) {
                                return invalid(format!(
                                    "axis {}: embedded entry ({i}, {j}) = {} is outside [0, 1]",
                                    AXES[v],
                                    e[(i, j)]
                                ));
                            }
                        }
                    }
                    let partial = Mat::from_fn(k, k, |i, j| {
                        if i == q || j == q {
                            0.0
                        } else if i == j {
                            -1.0 / l[i]
                        } else {
                            e[(i, j)] / l[i]
                        }
                    });
                    RateMatrix::from_partial(&partial, proportions, q)
                }
            };
            rates.push(r.map_err(|e| on_axis(v, e))?);
        }
        let rates: [RateMatrix; 3] = rates.try_into().expect("three axes");
        Self::new(proportions.to_vec(), rates, rotation, background)
    }

    pub fn k(&self) -> usize {
        self.proportions.len()
    }

    fn axes(&self) -> [[f64; 3]; 3] {
        let [a, d, r] = self.rotation.map(f64::to_radians);
        let x1 = [a.cos(), -a.sin(), 0.0];
        let y1 = [a.sin(), a.cos(), 0.0];
        let z1 = [0.0, 0.0, 1.0];
        let y2 = comb(d.cos(), y1, -d.sin(), z1);
        let z2 = comb(d.sin(), y1, d.cos(), z1);
        let x3 = comb(r.cos(), x1, -r.sin(), z2);
        let z3 = comb(r.sin(), x1, r.cos(), z2);
        [x3, y2, z3]
    }

    pub fn to_principal(&self, h: [f64; 3]) -> [f64; 3] {
        self.axes().map(|axis| dot(h, axis))
    }

    /// `h != 0`.
    pub fn rates_along(&self, h: [f64; 3]) -> Result<RateMatrix> {
        let length = norm(h);
        if !(length > 0.0 && length.is_finite()) {
            return invalid(format!(
                "rates_along needs a non-zero finite lag, got {h:?}"
            ));
        }
        let k = self.k();
        if k == 1 {
            return Ok(self.rates[0].clone());
        }
        let hp = self.to_principal(h);
        let axis = |v: usize| {
            if hp[v] >= 0.0 {
                self.rates[v].clone()
            } else {
                self.rates[v].reversed(&self.proportions)
            }
        };
        let active: Vec<usize> = (0..3).filter(|&v| hp[v] != 0.0).collect();
        match active.len() {
            0 => return invalid(format!("lag {h:?} has no principal component")),
            1 => return Ok(axis(active[0])),
            _ => {}
        }
        let c = hp.map(|x| x.abs() / length);
        let mats: Vec<(f64, Mat)> = active
            .iter()
            .map(|&v| (c[v], axis(v).matrix().clone()))
            .collect();
        let q = self.background;
        let partial = Mat::from_fn(k, k, |i, j| {
            if i == q || j == q {
                return 0.0;
            }
            let s = mats
                .iter()
                .map(|(c, m)| (c * m[(i, j)]).powi(2))
                .sum::<f64>()
                .sqrt();
            if i == j { -s } else { s }
        });
        RateMatrix::from_partial(&partial, &self.proportions, q)
            .map_err(|e| Error::InvalidInput(format!("interpolated rates along {h:?}: {e}")))
    }

    /// `T(0) = I`. Panics when the interpolated rates along `h` are invalid (see `rates_along`).
    pub fn transition(&self, h: [f64; 3]) -> Mat {
        let k = self.k();
        if h == [0.0; 3] {
            return Mat::identity(k, k);
        }
        let r = self
            .rates_along(h)
            .unwrap_or_else(|e| panic!("transition({h:?}): {e}"));
        expm_rate(r.matrix(), norm(h))
    }

    pub(crate) fn try_closeness(&self, h: [f64; 3]) -> Result<f64> {
        let k = self.k();
        if k == 1 || h == [0.0; 3] {
            return Ok(1.0);
        }
        let trace = self.rates_along(h)?.matrix().trace();
        Ok((norm(h) * trace / (k - 1) as f64).exp())
    }

    /// `det T(h)^(1/(k-1))`; 1 when k = 1.
    pub fn closeness(&self, h: [f64; 3]) -> f64 {
        self.try_closeness(h)
            .unwrap_or_else(|e| panic!("closeness({h:?}): {e}"))
    }

    /// NaN for every category at `h = 0`, where no direction is defined.
    pub fn mean_lengths(&self, h: [f64; 3]) -> Vec<f64> {
        if h == [0.0; 3] {
            return vec![f64::NAN; self.k()];
        }
        self.rates_along(h)
            .unwrap_or_else(|e| panic!("mean_lengths({h:?}): {e}"))
            .mean_lengths()
    }
}
