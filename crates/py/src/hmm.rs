//! Emission distributions, `HiddenMarkovModel`, and the sequence-model plumbing shared with the
//! HSMM binding (input parsing, clamps, outputs in input row order).

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use varvite_core::rng::tag;
use varvite_core::{Direction, Error, Mat, Result as VResult, Stream};
use varvite_hmm::{
    Categorical, Clamp, CovarianceType, Decoder, Emission, Family, FitOptions, Gaussian,
    GaussianMixture, Hmm, Implementation, Ordering, Params, Poisson, Priors, Sequence, n_observed,
};

use crate::args::{RowMap, column, direction, floats, mat, mat2, matrix, sequences, to_rust, vec1};
use crate::core::{json, unjson};
use crate::{convergence, invalid};

// ---------------------------------------------------------------- distributions

pub(crate) fn covariance_type(s: &str) -> PyResult<CovarianceType> {
    match s {
        "full" => Ok(CovarianceType::Full),
        "diag" => Ok(CovarianceType::Diag),
        "spherical" => Ok(CovarianceType::Spherical),
        _ => Err(invalid(format!(
            "covariance_type must be \"full\", \"diag\" or \"spherical\", not {s:?}"
        ))),
    }
}

fn covariance_name(c: CovarianceType) -> &'static str {
    match c {
        CovarianceType::Full => "full",
        CovarianceType::Diag => "diag",
        CovarianceType::Spherical => "spherical",
    }
}

/// `d x d` covariance from a scalar, a `(d,)` diagonal or a `(d, d)` matrix.
fn covariance(obj: &Bound<PyAny>, d: usize, ct: CovarianceType) -> PyResult<Mat> {
    if let Ok(v) = obj.extract::<f64>() {
        return Ok(Mat::identity(d, d) * v);
    }
    let ndim: usize = obj
        .py()
        .import("numpy")?
        .call_method1("ndim", (obj,))?
        .extract()?;
    let m = match ndim {
        1 => Mat::from_diagonal(&varvite_core::nalgebra::DVector::from_vec(floats(
            obj,
            "covariance",
        )?)),
        _ => mat(obj, "covariance")?,
    };
    if m.shape() != (d, d) {
        return Err(invalid(format!(
            "covariance must be a scalar, ({d},) or ({d}, {d}), not {:?}",
            m.shape()
        )));
    }
    if ct == CovarianceType::Spherical && ndim == 2 {
        return Err(invalid("a spherical covariance is a scalar"));
    }
    Ok(m)
}

fn gaussian(
    mean: Vec<f64>,
    cov: &Bound<PyAny>,
    ct: CovarianceType,
    frozen: bool,
) -> PyResult<Gaussian> {
    let d = mean.len();
    Ok(Gaussian {
        covariance: covariance(cov, d, ct)?,
        mean,
        covariance_type: ct,
        frozen,
    })
}

fn covariance_py<'py>(py: Python<'py>, g: &Gaussian) -> Bound<'py, PyAny> {
    let c = &g.covariance;
    match g.covariance_type {
        CovarianceType::Full => mat2(py, c).into_any(),
        CovarianceType::Diag => vec1(py, c.diagonal().iter().copied().collect()).into_any(),
        CovarianceType::Spherical => pyo3::types::PyFloat::new(py, c[(0, 0)]).into_any(),
    }
}

fn checked(e: Emission) -> PyResult<Emission> {
    e.validate().map_err(to_rust)?;
    Ok(e)
}

/// Observations of one distribution: `(n, d)` rows; categorical codes with -1 as NaN.
fn observations(x: &Bound<PyAny>, e: &Emission) -> PyResult<(usize, Vec<f64>)> {
    let d = e.dim();
    let (n, dd, mut v) = matrix(x, "x")?;
    let (n, dd) = if dd == 1 && d > 1 && n == d {
        (1, d)
    } else {
        (n, dd)
    };
    if dd != d {
        return Err(invalid(format!("x must have {d} channels, found {dd}")));
    }
    if matches!(e, Emission::Categorical(_)) {
        for x in &mut v {
            if *x == -1.0 {
                *x = f64::NAN;
            }
        }
    }
    Ok((n, v))
}

fn log_density<'py>(
    py: Python<'py>,
    e: &Emission,
    x: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (n, v) = observations(x, e)?;
    let d = e.dim();
    Ok(vec1(
        py,
        (0..n)
            .map(|i| e.log_density(&v[i * d..(i + 1) * d]))
            .collect(),
    ))
}

fn sample<'py>(py: Python<'py>, e: &Emission, n: usize, seed: u64) -> PyResult<Bound<'py, PyAny>> {
    let d = e.dim();
    let categorical = matches!(e, Emission::Categorical(_));
    let need = (n as u64).checked_mul(d as u64 * 8 * (1 + categorical as u64));
    varvite_core::memory::check(need.unwrap_or(u64::MAX), None, "samples").map_err(to_rust)?;
    let mut s = Stream::new(seed, &[tag::SAMPLE, 0]);
    let mut v: Vec<f64> = Vec::with_capacity(n * d);
    for _ in 0..n {
        v.extend(e.sample(&mut s));
    }
    if categorical {
        return Ok(PyArray1::from_vec(py, v.iter().map(|&x| x as i64).collect()).into_any());
    }
    Ok(PyArray1::from_vec(py, v).reshape([n, d])?.into_any())
}

/// Categorical emission over symbols ``0..m-1``.
///
/// Parameters
/// ----------
/// probabilities : array_like
///     Probability of each symbol, summing to 1.
/// pseudocount : float
///     Added to every symbol count in the M-step.
/// frozen : bool
///     Keep the parameters fixed while fitting.
#[pyclass(module = "varvite", name = "Categorical", frozen, eq, from_py_object)]
#[derive(Clone, PartialEq)]
pub struct PyCategorical(pub Categorical);

#[pymethods]
impl PyCategorical {
    #[new]
    #[pyo3(signature = (probabilities, *, pseudocount=0.0, frozen=false))]
    fn new(probabilities: &Bound<PyAny>, pseudocount: f64, frozen: bool) -> PyResult<Self> {
        let c = Categorical {
            probabilities: floats(probabilities, "probabilities")?,
            pseudocount,
            frozen,
        };
        checked(Emission::Categorical(c.clone()))?;
        Ok(Self(c))
    }
    #[getter]
    fn probabilities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec1(py, self.0.probabilities.clone())
    }
    #[getter]
    fn pseudocount(&self) -> f64 {
        self.0.pseudocount
    }
    #[getter]
    fn frozen(&self) -> bool {
        self.0.frozen
    }
    /// Log probability of each code in ``x`` (-1 or NaN for missing gives 0).
    fn log_density<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        log_density(py, &Emission::Categorical(self.0.clone()), x)
    }
    /// ``n`` seeded draws, int64 ``(n,)``.
    #[pyo3(signature = (n, *, seed=0))]
    fn sample<'py>(&self, py: Python<'py>, n: usize, seed: u64) -> PyResult<Bound<'py, PyAny>> {
        sample(py, &Emission::Categorical(self.0.clone()), n, seed)
    }
    fn to_json(&self) -> PyResult<String> {
        json(&self.0)
    }
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        let c: Categorical = unjson(text, "Categorical")?;
        checked(Emission::Categorical(c.clone()))?;
        Ok(Self(c))
    }
    fn __repr__(&self) -> String {
        format!("Categorical(probabilities={:?})", self.0.probabilities)
    }
}

/// Gaussian emission.
///
/// Parameters
/// ----------
/// mean : array_like
///     ``(d,)`` mean.
/// covariance : array_like or float
///     ``(d, d)`` for "full", ``(d,)`` for "diag", a scalar for "spherical".
/// covariance_type : {"full", "diag", "spherical"}
/// frozen : bool
///     Keep the parameters fixed while fitting.
#[pyclass(module = "varvite", name = "Gaussian", frozen, eq, from_py_object)]
#[derive(Clone, PartialEq)]
pub struct PyGaussian(pub Gaussian);

#[pymethods]
impl PyGaussian {
    #[new]
    #[pyo3(signature = (mean, covariance, *, covariance_type="full", frozen=false))]
    fn new(
        mean: &Bound<PyAny>,
        covariance: &Bound<PyAny>,
        covariance_type: &str,
        frozen: bool,
    ) -> PyResult<Self> {
        let g = gaussian(
            floats(mean, "mean")?,
            covariance,
            self::covariance_type(covariance_type)?,
            frozen,
        )?;
        checked(Emission::Gaussian(g.clone()))?;
        Ok(Self(g))
    }
    #[getter]
    fn mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec1(py, self.0.mean.clone())
    }
    /// As constructed: ``(d, d)``, ``(d,)`` or a float.
    #[getter]
    fn covariance<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        covariance_py(py, &self.0)
    }
    #[getter]
    fn covariance_type(&self) -> &'static str {
        covariance_name(self.0.covariance_type)
    }
    #[getter]
    fn frozen(&self) -> bool {
        self.0.frozen
    }
    /// Log density of each row of ``x`` (``(n, d)``, or ``(n,)`` when ``d = 1``); NaN channels
    /// are marginalised.
    fn log_density<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        log_density(py, &Emission::Gaussian(self.0.clone()), x)
    }
    /// ``n`` seeded draws, ``(n, d)``.
    #[pyo3(signature = (n, *, seed=0))]
    fn sample<'py>(&self, py: Python<'py>, n: usize, seed: u64) -> PyResult<Bound<'py, PyAny>> {
        sample(py, &Emission::Gaussian(self.0.clone()), n, seed)
    }
    fn to_json(&self) -> PyResult<String> {
        json(&self.0)
    }
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        let g: Gaussian = unjson(text, "Gaussian")?;
        checked(Emission::Gaussian(g.clone()))?;
        Ok(Self(g))
    }
    fn __repr__(&self) -> String {
        format!(
            "Gaussian(mean={:?}, covariance_type={:?})",
            self.0.mean,
            self.covariance_type()
        )
    }
}

/// Poisson emission with independent channels.
///
/// Parameters
/// ----------
/// rate : array_like
///     ``(d,)`` rates.
/// frozen : bool
#[pyclass(module = "varvite", name = "Poisson", frozen, eq, from_py_object)]
#[derive(Clone, PartialEq)]
pub struct PyPoisson(pub Poisson);

#[pymethods]
impl PyPoisson {
    #[new]
    #[pyo3(signature = (rate, *, frozen=false))]
    fn new(rate: &Bound<PyAny>, frozen: bool) -> PyResult<Self> {
        let (rate, _) = crate::args::scalar_or_floats(rate, "rate")?;
        let p = Poisson { rate, frozen };
        checked(Emission::Poisson(p.clone()))?;
        Ok(Self(p))
    }
    #[getter]
    fn rate<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec1(py, self.0.rate.clone())
    }
    #[getter]
    fn frozen(&self) -> bool {
        self.0.frozen
    }
    /// Log probability of each row of counts in ``x``.
    fn log_density<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        log_density(py, &Emission::Poisson(self.0.clone()), x)
    }
    /// ``n`` seeded draws, ``(n, d)``.
    #[pyo3(signature = (n, *, seed=0))]
    fn sample<'py>(&self, py: Python<'py>, n: usize, seed: u64) -> PyResult<Bound<'py, PyAny>> {
        sample(py, &Emission::Poisson(self.0.clone()), n, seed)
    }
    fn to_json(&self) -> PyResult<String> {
        json(&self.0)
    }
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        let p: Poisson = unjson(text, "Poisson")?;
        checked(Emission::Poisson(p.clone()))?;
        Ok(Self(p))
    }
    fn __repr__(&self) -> String {
        format!("Poisson(rate={:?})", self.0.rate)
    }
}

/// Mixture of Gaussians sharing one covariance type.
///
/// Parameters
/// ----------
/// weights : array_like
///     ``(m,)`` component weights summing to 1.
/// means : array_like
///     ``(m, d)`` component means.
/// covariances : array_like
///     ``(m, d, d)`` for "full", ``(m, d)`` for "diag", ``(m,)`` for "spherical".
/// covariance_type : {"full", "diag", "spherical"}
/// frozen : bool
#[pyclass(
    module = "varvite",
    name = "GaussianMixture",
    frozen,
    eq,
    from_py_object
)]
#[derive(Clone, PartialEq)]
pub struct PyGaussianMixture(pub GaussianMixture);

#[pymethods]
impl PyGaussianMixture {
    #[new]
    #[pyo3(signature = (weights, means, covariances, *, covariance_type="full", frozen=false))]
    fn new(
        weights: &Bound<PyAny>,
        means: &Bound<PyAny>,
        covariances: &Bound<PyAny>,
        covariance_type: &str,
        frozen: bool,
    ) -> PyResult<Self> {
        let ct = self::covariance_type(covariance_type)?;
        let weights = floats(weights, "weights")?;
        let (m, d, mu) = matrix(means, "means")?;
        let np = means.py().import("numpy")?;
        let covs = np.call_method1("asarray", (covariances, "float64"))?;
        let n_cov: usize = covs.len()?;
        if n_cov != m {
            return Err(invalid(format!(
                "give {m} covariances, one per mean, not {n_cov}"
            )));
        }
        let components = (0..m)
            .map(|i| {
                gaussian(
                    mu[i * d..(i + 1) * d].to_vec(),
                    &covs.get_item(i)?,
                    ct,
                    false,
                )
            })
            .collect::<PyResult<Vec<_>>>()?;
        let g = GaussianMixture {
            weights,
            components,
            frozen,
        };
        checked(Emission::Mixture(g.clone()))?;
        Ok(Self(g))
    }
    #[getter]
    fn weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec1(py, self.0.weights.clone())
    }
    /// ``(m, d)``.
    #[getter]
    fn means<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let c = &self.0.components;
        let d = c.first().map_or(0, |g| g.mean.len());
        mat2(py, &Mat::from_fn(c.len(), d, |i, j| c[i].mean[j]))
    }
    /// One covariance per component, in the constructed form.
    #[getter]
    fn covariances<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let items: Vec<Bound<PyAny>> = self
            .0
            .components
            .iter()
            .map(|g| covariance_py(py, g))
            .collect();
        py.import("numpy")?.call_method1("array", (items,))
    }
    #[getter]
    fn covariance_type(&self) -> &'static str {
        covariance_name(
            self.0
                .components
                .first()
                .map_or(CovarianceType::Full, |g| g.covariance_type),
        )
    }
    #[getter]
    fn frozen(&self) -> bool {
        self.0.frozen
    }
    /// Log mixture density of each row of ``x``.
    ///
    /// Parameters
    /// ----------
    /// x : array_like
    ///     ``(n, d)`` observations, or ``(n,)`` when ``d = 1``. NaN channels are marginalised
    ///     in every component.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     ``(n,)`` values of ``log sum_k w_k N(x; mu_k, Sigma_k)``.
    fn log_density<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        log_density(py, &Emission::Mixture(self.0.clone()), x)
    }
    /// ``n`` seeded draws, ``(n, d)``.
    #[pyo3(signature = (n, *, seed=0))]
    fn sample<'py>(&self, py: Python<'py>, n: usize, seed: u64) -> PyResult<Bound<'py, PyAny>> {
        sample(py, &Emission::Mixture(self.0.clone()), n, seed)
    }
    fn to_json(&self) -> PyResult<String> {
        json(&self.0)
    }
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        let g: GaussianMixture = unjson(text, "GaussianMixture")?;
        checked(Emission::Mixture(g.clone()))?;
        Ok(Self(g))
    }
    fn __repr__(&self) -> String {
        format!("GaussianMixture(n_components={})", self.0.components.len())
    }
}

pub(crate) fn emission_of(obj: &Bound<PyAny>) -> PyResult<Emission> {
    if let Ok(c) = obj.cast::<PyCategorical>() {
        return Ok(Emission::Categorical(c.get().0.clone()));
    }
    if let Ok(g) = obj.cast::<PyGaussian>() {
        return Ok(Emission::Gaussian(g.get().0.clone()));
    }
    if let Ok(p) = obj.cast::<PyPoisson>() {
        return Ok(Emission::Poisson(p.get().0.clone()));
    }
    if let Ok(m) = obj.cast::<PyGaussianMixture>() {
        return Ok(Emission::Mixture(m.get().0.clone()));
    }
    Err(invalid(
        "distributions must be Categorical, Gaussian, Poisson or GaussianMixture objects",
    ))
}

pub(crate) fn emission_py(py: Python, e: &Emission) -> PyResult<Py<PyAny>> {
    Ok(match e {
        Emission::Categorical(c) => Py::new(py, PyCategorical(c.clone()))?.into_any(),
        Emission::Gaussian(g) => Py::new(py, PyGaussian(g.clone()))?.into_any(),
        Emission::Poisson(p) => Py::new(py, PyPoisson(p.clone()))?.into_any(),
        Emission::Mixture(m) => Py::new(py, PyGaussianMixture(m.clone()))?.into_any(),
    })
}

/// `n_states` and the given emissions from `distributions` (an int or a sequence of objects).
pub(crate) fn distributions(obj: &Bound<PyAny>) -> PyResult<(usize, Option<Vec<Emission>>)> {
    if let Ok(k) = obj.extract::<usize>() {
        if k == 0 {
            return Err(invalid("give at least one state"));
        }
        return Ok((k, None));
    }
    let items: Vec<Bound<PyAny>> = obj
        .try_iter()
        .map_err(|_| invalid("distributions must be an int or a sequence of distributions"))?
        .collect::<PyResult<_>>()?;
    if items.is_empty() {
        return Err(invalid("give at least one distribution"));
    }
    let e = items
        .iter()
        .map(emission_of)
        .collect::<PyResult<Vec<_>>>()?;
    Ok((e.len(), Some(e)))
}

// ---------------------------------------------------------------- shared sequence plumbing

/// Model-independent settings shared by the HMM and HSMM classes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct Common {
    pub n_states: usize,
    pub emission: String,
    pub covariance_type: CovarianceType,
    pub n_symbols: Option<usize>,
    pub n_mix: usize,
    pub max_iter: usize,
    pub tol: f64,
    pub n_init: usize,
    pub seed: u64,
    pub params: String,
    pub init_params: Option<String>,
    pub start_prior: f64,
    pub transition_prior: f64,
    pub covariance_prior: f64,
    pub pseudocount: f64,
    pub direction: Direction,
    pub ordering: String,
}

impl Common {
    pub fn categorical(&self) -> bool {
        self.emission == "categorical"
    }

    pub fn priors(&self) -> Priors {
        Priors {
            start: self.start_prior,
            transition: self.transition_prior,
            covariance: self.covariance_prior,
        }
    }

    pub fn ordering(&self) -> PyResult<Ordering> {
        match self.ordering.as_str() {
            "none" => Ok(Ordering::None),
            "mean" => Ok(Ordering::Mean),
            "occupancy" => Ok(Ordering::Occupancy),
            o => Err(invalid(format!(
                "ordering must be \"none\", \"mean\" or \"occupancy\", not {o:?}"
            ))),
        }
    }

    /// `init_params`, or every letter of `all` minus those of the parts given explicitly.
    pub fn init_letters(&self, all: &str, given: &[(char, bool)]) -> String {
        match &self.init_params {
            Some(p) => p.clone(),
            None => all
                .chars()
                .filter(|c| !given.iter().any(|(g, yes)| g == c && *yes))
                .collect(),
        }
    }

    pub fn family(&self, given: Option<&[Emission]>, data: &[Sequence]) -> PyResult<Family> {
        if let Some(e) = given.and_then(|e| e.first()) {
            return Ok(match e {
                Emission::Categorical(c) => Family::Categorical {
                    n_symbols: c.probabilities.len(),
                },
                Emission::Gaussian(g) => Family::Gaussian {
                    covariance_type: g.covariance_type,
                },
                Emission::Poisson(_) => Family::Poisson,
                Emission::Mixture(m) => Family::Mixture {
                    n_mix: m.components.len(),
                    covariance_type: m.components[0].covariance_type,
                },
            });
        }
        let ct = self.covariance_type;
        match self.emission.as_str() {
            "gaussian" => Ok(Family::Gaussian {
                covariance_type: ct,
            }),
            "poisson" => Ok(Family::Poisson),
            "mixture" => Ok(Family::Mixture {
                n_mix: self.n_mix,
                covariance_type: ct,
            }),
            _ => {
                let seen = data
                    .iter()
                    .flat_map(|s| &s.x)
                    .filter(|x| !x.is_nan())
                    .fold(0.0f64, |m, &x| m.max(x + 1.0)) as usize;
                Ok(Family::Categorical {
                    n_symbols: self.n_symbols.unwrap_or(seen.max(1)),
                })
            }
        }
    }

    pub fn fit_options(&self, init_letters: &str) -> PyResult<FitOptions> {
        Ok(FitOptions {
            max_iter: self.max_iter,
            tol: self.tol,
            n_init: self.n_init,
            seed: self.seed,
            params: Params::parse(&self.params).map_err(to_rust)?,
            init_params: Params::parse(init_letters).map_err(to_rust)?,
            priors: self.priors(),
            implementation: Implementation::Log,
            ordering: self.ordering()?,
        })
    }
}

pub(crate) fn emission_name(s: &str) -> PyResult<String> {
    match s {
        "gaussian" | "categorical" | "poisson" | "mixture" => Ok(s.to_string()),
        _ => Err(invalid(format!(
            "emission must be \"gaussian\", \"categorical\", \"poisson\" or \"mixture\", not {s:?}"
        ))),
    }
}

/// Sequences, their row map and the clamps from `labels` and `allowed`.
pub(crate) struct Input {
    pub seqs: Vec<Sequence>,
    pub map: RowMap,
    pub clamps: Option<Vec<Clamp>>,
}

impl Input {
    pub fn clamp(&self, i: usize) -> Option<&Clamp> {
        self.clamps.as_ref().map(|c| &c[i])
    }
}

/// Per-hole arrays (when `sequences` was a list) are concatenated to match row numbers.
fn per_row<'py>(obj: &Bound<'py, PyAny>, listed: bool) -> PyResult<Bound<'py, PyAny>> {
    if listed && (obj.cast::<PyList>().is_ok() || obj.cast::<PyTuple>().is_ok()) {
        return obj
            .py()
            .import("numpy")?
            .call_method1("concatenate", (obj,));
    }
    Ok(obj.clone())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn input<'py>(
    sequences_: &Bound<'py, PyAny>,
    holes: Option<&Bound<'py, PyAny>>,
    labels: Option<&Bound<'py, PyAny>>,
    allowed: Option<&Bound<'py, PyAny>>,
    data: Option<&Bound<'py, PyAny>>,
    direction: Direction,
    d_expected: Option<usize>,
    k: usize,
    categorical: bool,
) -> PyResult<Input> {
    let (mut seqs, map) = sequences(sequences_, holes, data, direction, d_expected)?;
    if categorical {
        for s in &mut seqs {
            for x in &mut s.x {
                if *x == -1.0 {
                    *x = f64::NAN;
                }
            }
        }
    }
    let n = map.n_rows;
    let listed = map.list.is_some();
    let labels = labels
        .filter(|o| !o.is_none())
        .map(|o| -> PyResult<Vec<f64>> {
            floats(&per_row(&column(data, o, "labels")?, listed)?, "labels")
        })
        .transpose()?;
    if let Some(l) = &labels {
        if l.len() != n {
            return Err(invalid(format!(
                "labels has {} values for {n} rows",
                l.len()
            )));
        }
        if let Some(r) = l.iter().position(|&v| {
            !(v.is_nan() || v == -1.0 || (v >= 0.0 && v.fract() == 0.0 && v < k as f64))
        }) {
            return Err(invalid(format!(
                "labels must be states in 0..{k} or -1, found {} at row {r}",
                l[r]
            )));
        }
    }
    let allowed = allowed
        .filter(|o| !o.is_none())
        .map(|o| -> PyResult<Vec<bool>> {
            let a = per_row(&column(data, o, "allowed")?, listed)?;
            let a = o
                .py()
                .import("numpy")?
                .call_method1("asarray", (a, "bool"))?;
            let shape: Vec<usize> = a.getattr("shape")?.extract()?;
            if shape != [n, k] {
                return Err(invalid(format!(
                    "allowed must have shape ({n}, {k}), not {shape:?}"
                )));
            }
            a.call_method1("reshape", (-1i64,))?
                .call_method0("tolist")?
                .extract()
        })
        .transpose()?;
    let clamps = (labels.is_some() || allowed.is_some()).then(|| {
        map.rows
            .iter()
            .map(|rows| Clamp {
                k,
                allowed: rows
                    .iter()
                    .flat_map(|&r| {
                        let lab = labels.as_ref().map(|l| l[r]).filter(|v| *v >= 0.0);
                        let allowed = &allowed;
                        (0..k).map(move |j| {
                            lab.is_none_or(|v| v as usize == j)
                                && allowed.as_ref().is_none_or(|a| a[r * k + j])
                        })
                    })
                    .collect(),
            })
            .collect()
    });
    Ok(Input { seqs, map, clamps })
}

/// Per-step values (direction order, one Vec per sequence) back to input rows: one array, or a
/// list of arrays when the input was a list.
pub(crate) fn rows_out<'py, T: numpy::Element + Copy>(
    py: Python<'py>,
    map: &RowMap,
    per_seq: &[Vec<T>],
    width: usize,
    fill: T,
) -> PyResult<Bound<'py, PyAny>> {
    let mut out = vec![fill; map.n_rows * width];
    for (rows, values) in map.rows.iter().zip(per_seq) {
        for (t, &r) in rows.iter().enumerate() {
            out[r * width..(r + 1) * width].copy_from_slice(&values[t * width..(t + 1) * width]);
        }
    }
    let shape = |n: usize| if width == 1 { vec![n] } else { vec![n, width] };
    match &map.list {
        None => Ok(PyArray1::from_vec(py, out)
            .reshape(shape(map.n_rows))?
            .into_any()),
        Some(lengths) => {
            let mut items = Vec::with_capacity(lengths.len());
            let mut at = 0;
            for &len in lengths {
                let piece = out[at * width..(at + len) * width].to_vec();
                items.push(
                    PyArray1::from_vec(py, piece)
                        .reshape(shape(len))?
                        .into_any(),
                );
                at += len;
            }
            Ok(PyList::new(py, items)?.into_any())
        }
    }
}

/// Keeps the error class and names the sequence instead of the engine's index 0.
pub(crate) fn at(i: usize, e: Error) -> PyErr {
    let name = |m: String| {
        if m.contains("sequence 0") {
            m.replacen("sequence 0", &format!("sequence {i}"), 1)
        } else {
            format!("sequence {i}: {m}")
        }
    };
    match e {
        Error::InvalidInput(m) => invalid(name(m)),
        Error::Convergence(m) => convergence(name(m)),
    }
}

/// What the HMM and HSMM classes share once a model exists.
pub(crate) trait Engine: Sync {
    fn k(&self) -> usize;
    fn emissions(&self) -> &[Emission];
    fn log_likelihood(&self, seq: &Sequence) -> VResult<f64>;
    /// `(gamma, entropy, log_likelihood)`.
    fn posterior(&self, seq: &Sequence, clamp: Option<&Clamp>) -> VResult<(Mat, f64, f64)>;
    fn viterbi(&self, seq: &Sequence, clamp: Option<&Clamp>) -> VResult<(Vec<u32>, f64)>;
    fn decode(&self, seq: &Sequence, clamp: Option<&Clamp>, algorithm: &str) -> VResult<Vec<u32>>;
    fn sample_posterior(
        &self,
        seq: &Sequence,
        clamp: Option<&Clamp>,
        n: usize,
        seed: u64,
        index: u64,
    ) -> VResult<Vec<Vec<u32>>>;
    fn sample(&self, lengths: &[usize], seed: u64) -> Vec<(Vec<u32>, Sequence)>;
    fn n_params(&self, params: Params) -> usize;
}

pub(crate) fn each<T: Send>(
    inp: &Input,
    f: impl Fn(usize, &Sequence, Option<&Clamp>) -> VResult<T> + Sync,
) -> PyResult<Vec<T>> {
    let results: Vec<VResult<T>> = (0..inp.seqs.len())
        .into_par_iter()
        .map(|i| f(i, &inp.seqs[i], inp.clamp(i)))
        .collect();
    results
        .into_iter()
        .enumerate()
        .map(|(i, r)| r.map_err(|e| at(i, e)))
        .collect()
}

pub(crate) fn score_of(py: Python, e: &impl Engine, inp: &Input) -> PyResult<f64> {
    let lls = py.detach(|| each(inp, |_, s, _| e.log_likelihood(s)))?;
    Ok(lls.iter().sum())
}

pub(crate) fn criteria(
    py: Python,
    e: &impl Engine,
    inp: &Input,
    params: &str,
) -> PyResult<(f64, usize, usize, f64)> {
    let params = Params::parse(params).map_err(to_rust)?;
    let res = py.detach(|| each(inp, |_, s, _| e.posterior(s, None)))?;
    let ll = res.iter().map(|r| r.2).sum();
    let entropy = res.iter().map(|r| r.1).sum();
    Ok((ll, e.n_params(params), n_observed(&inp.seqs), entropy))
}

pub(crate) fn predict_of<'py>(
    py: Python<'py>,
    e: &impl Engine,
    inp: &Input,
    algorithm: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let paths = py.detach(|| each(inp, |_, s, c| e.decode(s, c, algorithm)))?;
    let paths: Vec<Vec<i64>> = paths
        .into_iter()
        .map(|p| p.into_iter().map(|s| s as i64).collect())
        .collect();
    rows_out(py, &inp.map, &paths, 1, -1i64)
}

pub(crate) fn proba_of<'py>(
    py: Python<'py>,
    e: &impl Engine,
    inp: &Input,
) -> PyResult<Bound<'py, PyAny>> {
    let k = e.k();
    let g = py.detach(|| each(inp, |_, s, c| e.posterior(s, c)))?;
    let flat: Vec<Vec<f64>> = g
        .iter()
        .map(|(g, _, _)| {
            (0..g.nrows())
                .flat_map(|t| (0..k).map(move |j| g[(t, j)]))
                .collect()
        })
        .collect();
    rows_out(py, &inp.map, &flat, k, f64::NAN)
}

pub(crate) fn viterbi_of<'py>(
    py: Python<'py>,
    e: &impl Engine,
    inp: &Input,
) -> PyResult<Bound<'py, PyTuple>> {
    let res = py.detach(|| each(inp, |_, s, c| e.viterbi(s, c)))?;
    let paths: Vec<Vec<i64>> = res
        .iter()
        .map(|(p, _)| p.iter().map(|&s| s as i64).collect())
        .collect();
    let lp: Vec<f64> = res.iter().map(|r| r.1).collect();
    PyTuple::new(
        py,
        [
            rows_out(py, &inp.map, &paths, 1, -1i64)?,
            vec1(py, lp).into_any(),
        ],
    )
}

pub(crate) fn sample_posterior_of<'py>(
    py: Python<'py>,
    e: &impl Engine,
    inp: &Input,
    n: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyAny>> {
    let rows = inp.map.n_rows;
    let need = (n as u64).checked_mul(rows as u64 * 8);
    varvite_core::memory::check(need.unwrap_or(u64::MAX), None, "posterior samples")
        .map_err(to_rust)?;
    let draws = py.detach(|| each(inp, |i, s, c| e.sample_posterior(s, c, n, seed, i as u64)))?;
    let mut out = vec![-1i64; n * rows];
    for (map_rows, paths) in inp.map.rows.iter().zip(&draws) {
        for (d, path) in paths.iter().enumerate() {
            for (t, &r) in map_rows.iter().enumerate() {
                out[d * rows + r] = path[t] as i64;
            }
        }
    }
    Ok(PyArray1::from_vec(py, out).reshape([n, rows])?.into_any())
}

pub(crate) fn sample_of<'py>(
    py: Python<'py>,
    e: &impl Engine,
    lengths: &Bound<'py, PyAny>,
    seed: u64,
    direction: Direction,
) -> PyResult<Bound<'py, PyTuple>> {
    let lengths: Vec<usize> = match lengths.extract::<usize>() {
        Ok(n) => vec![n],
        Err(_) => lengths
            .extract()
            .map_err(|_| invalid("lengths must be an int or a sequence of ints"))?,
    };
    let d = e.emissions()[0].dim();
    let total: u64 = lengths.iter().map(|&l| l as u64).sum();
    varvite_core::memory::check(total.saturating_mul(8 * (d as u64 + 1)), None, "samples")
        .map_err(to_rust)?;
    let out = py.detach(|| e.sample(&lengths, seed));
    let categorical = matches!(e.emissions()[0], Emission::Categorical(_));
    let (mut states, mut obs) = (Vec::new(), Vec::new());
    for (mut s, mut x) in out {
        if direction == Direction::Up {
            s.reverse();
            x.x = x.x.chunks(d.max(1)).rev().flatten().copied().collect();
        }
        states.push(
            PyArray1::from_vec(py, s.into_iter().map(|v| v as i64).collect::<Vec<i64>>())
                .into_any(),
        );
        obs.push(if categorical {
            PyArray1::from_vec(py, x.x.iter().map(|&v| v as i64).collect::<Vec<i64>>()).into_any()
        } else {
            PyArray1::from_vec(py, x.x).reshape([x.t, d])?.into_any()
        });
    }
    PyTuple::new(
        py,
        [
            PyList::new(py, states)?.into_any(),
            PyList::new(py, obs)?.into_any(),
        ],
    )
}

pub(crate) fn distributions_py<'py>(
    py: Python<'py>,
    e: &[Emission],
) -> PyResult<Bound<'py, PyList>> {
    PyList::new(
        py,
        e.iter()
            .map(|e| emission_py(py, e))
            .collect::<PyResult<Vec<_>>>()?,
    )
}

pub(crate) fn set_pseudocount(emissions: &mut [Emission], pseudocount: f64) {
    for e in emissions {
        if let Emission::Categorical(c) = e {
            c.pseudocount = pseudocount;
        }
    }
}

// ---------------------------------------------------------------- HiddenMarkovModel

struct Engined<'a>(&'a Hmm);

impl Engine for Engined<'_> {
    fn k(&self) -> usize {
        self.0.k()
    }
    fn emissions(&self) -> &[Emission] {
        &self.0.emissions
    }
    fn log_likelihood(&self, seq: &Sequence) -> VResult<f64> {
        self.0.log_likelihood(seq, None, Implementation::Log)
    }
    fn posterior(&self, seq: &Sequence, clamp: Option<&Clamp>) -> VResult<(Mat, f64, f64)> {
        let p = self.0.posterior(seq, clamp, Implementation::Log)?;
        Ok((p.gamma, p.entropy, p.log_likelihood))
    }
    fn viterbi(&self, seq: &Sequence, clamp: Option<&Clamp>) -> VResult<(Vec<u32>, f64)> {
        self.0.viterbi(seq, clamp)
    }
    fn decode(&self, seq: &Sequence, clamp: Option<&Clamp>, algorithm: &str) -> VResult<Vec<u32>> {
        let d = match algorithm {
            "map" => Decoder::Map,
            "posterior_viterbi" => Decoder::PosteriorViterbi,
            _ => Decoder::Viterbi,
        };
        self.0.decode(seq, clamp, d)
    }
    fn sample_posterior(
        &self,
        seq: &Sequence,
        clamp: Option<&Clamp>,
        n: usize,
        seed: u64,
        index: u64,
    ) -> VResult<Vec<Vec<u32>>> {
        self.0.sample_posterior(seq, clamp, n, seed, index)
    }
    fn sample(&self, lengths: &[usize], seed: u64) -> Vec<(Vec<u32>, Sequence)> {
        self.0.sample(lengths, seed)
    }
    fn n_params(&self, params: Params) -> usize {
        self.0.n_params(params)
    }
}

/// Hidden Markov model of a geological state sequence observed through logs or assays.
///
/// Parameters
/// ----------
/// distributions : int or sequence of distributions
///     The number of states, or one emission object per state (``Categorical``, ``Gaussian``,
///     ``Poisson`` or ``GaussianMixture``).
/// start, transition : array_like, optional
///     Start probabilities ``(K,)`` and transition matrix ``(K, K)``.
/// emission : {"gaussian", "categorical", "poisson", "mixture"}
///     Family fitted when ``distributions`` is an int.
/// covariance_type : {"full", "diag", "spherical"}
/// n_symbols : int, optional
///     Categorical symbols; the largest observed code plus 1 by default.
/// n_mix : int
///     Components per state of a mixture.
/// max_iter, tol, n_init, seed
///     Baum-Welch iterations, gain per observed step that stops it, seeded restarts.
/// params, init_params : str
///     Letters of the parts fitted (s start, t transition, e emissions) and initialised; by
///     default ``init_params`` is "ste" minus the parts given.
/// start_prior, transition_prior, covariance_prior : float
///     Dirichlet concentrations (>= 1) and the covariance shrinkage of the MAP M-step.
/// pseudocount : float
///     Categorical emissions: added to symbol counts.
/// direction : {"up", "down"}
///     "up" runs each hole from its deepest row to its shallowest.
/// implementation : {"log", "scaled"}
///     Forward-backward arithmetic used while fitting.
/// ordering : {"none", "mean", "occupancy"}
///     Relabel the fitted states by first-channel mean or by stationary occupancy.
#[pyclass(module = "varvite", name = "HiddenMarkovModel", eq, from_py_object)]
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct PyHiddenMarkovModel {
    common: Common,
    implementation: String,
    start: Option<Vec<f64>>,
    #[serde(with = "varvite_core::serde_mat::option")]
    transition: Option<Mat>,
    emissions: Option<Vec<Emission>>,
    model: Option<Hmm>,
    history: Vec<f64>,
    converged: bool,
    n_iter: usize,
}

impl PyHiddenMarkovModel {
    pub(crate) fn common(&self) -> &Common {
        &self.common
    }

    pub fn fitted(&self) -> PyResult<&Hmm> {
        self.model.as_ref().ok_or_else(|| {
            invalid("the model is not fitted; call fit or give start, transition and distributions")
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn input<'py>(
        &self,
        sequences: &Bound<'py, PyAny>,
        holes: Option<&Bound<'py, PyAny>>,
        labels: Option<&Bound<'py, PyAny>>,
        allowed: Option<&Bound<'py, PyAny>>,
        data: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Input> {
        let d = self
            .model
            .as_ref()
            .map(|m| &m.emissions)
            .or(self.emissions.as_ref())
            .map(|e| e[0].dim());
        input(
            sequences,
            holes,
            labels,
            allowed,
            data,
            self.common.direction,
            d,
            self.common.n_states,
            self.common.categorical(),
        )
    }
}

#[pymethods]
impl PyHiddenMarkovModel {
    #[new]
    #[pyo3(signature = (distributions, *, start=None, transition=None, emission="gaussian", covariance_type="diag", n_symbols=None, n_mix=1, max_iter=100, tol=1e-6, n_init=1, seed=0, params="ste", init_params=None, start_prior=1.0, transition_prior=1.0, covariance_prior=1e-6, pseudocount=0.0, direction="up", implementation="log", ordering="none"))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        distributions: &Bound<PyAny>,
        start: Option<&Bound<PyAny>>,
        transition: Option<&Bound<PyAny>>,
        emission: &str,
        covariance_type: &str,
        n_symbols: Option<usize>,
        n_mix: usize,
        max_iter: usize,
        tol: f64,
        n_init: usize,
        seed: u64,
        params: &str,
        init_params: Option<String>,
        start_prior: f64,
        transition_prior: f64,
        covariance_prior: f64,
        pseudocount: f64,
        direction: &str,
        implementation: &str,
        ordering: &str,
    ) -> PyResult<Self> {
        let (k, emissions) = self::distributions(distributions)?;
        if !matches!(implementation, "log" | "scaled") {
            return Err(invalid(format!(
                "implementation must be \"log\" or \"scaled\", not {implementation:?}"
            )));
        }
        let emission = match &emissions {
            Some(e) => match e[0] {
                Emission::Categorical(_) => "categorical".to_string(),
                Emission::Gaussian(_) => "gaussian".to_string(),
                Emission::Poisson(_) => "poisson".to_string(),
                Emission::Mixture(_) => "mixture".to_string(),
            },
            None => emission_name(emission)?,
        };
        let common = Common {
            n_states: k,
            emission,
            covariance_type: self::covariance_type(covariance_type)?,
            n_symbols,
            n_mix,
            max_iter,
            tol,
            n_init,
            seed,
            params: params.to_string(),
            init_params,
            start_prior,
            transition_prior,
            covariance_prior,
            pseudocount,
            direction: self::direction(direction)?,
            ordering: ordering.to_string(),
        };
        common.ordering()?;
        Params::parse(params).map_err(to_rust)?;
        if let Some(p) = &common.init_params {
            Params::parse(p).map_err(to_rust)?;
        }
        let start = start
            .filter(|o| !o.is_none())
            .map(|o| floats(o, "start"))
            .transpose()?;
        let transition = transition
            .filter(|o| !o.is_none())
            .map(|o| mat(o, "transition"))
            .transpose()?;
        let model = match (&start, &transition, &emissions) {
            (Some(s), Some(t), Some(e)) => {
                Some(Hmm::new(s.clone(), t.clone(), e.clone()).map_err(to_rust)?)
            }
            _ => None,
        };
        Ok(Self {
            common,
            implementation: implementation.to_string(),
            start,
            transition,
            emissions,
            model,
            history: vec![],
            converged: false,
            n_iter: 0,
        })
    }

    /// Baum-Welch (EM) fit over all holes, with seeded restarts.
    ///
    /// Parameters
    /// ----------
    /// sequences : array_like, list of array_like or str
    ///     One array per hole (rows top-down, ``(t,)`` or ``(t, d)``), or one array with
    ///     ``holes``.
    /// holes : array_like or str, optional
    ///     Per-row hole ids when ``sequences`` is one array.
    /// labels : array_like or str, optional
    ///     Per-row known state, -1 where unknown (logged core intervals).
    /// allowed : array_like, optional
    ///     ``(rows, K)`` booleans: the states each row may take.
    /// data : mapping, optional
    ///     Strings above name its columns.
    ///
    /// Returns
    /// -------
    /// HiddenMarkovModel
    ///     This model, fitted.
    #[pyo3(signature = (sequences, *, holes=None, labels=None, allowed=None, data=None))]
    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        sequences: &Bound<'py, PyAny>,
        holes: Option<&Bound<'py, PyAny>>,
        labels: Option<&Bound<'py, PyAny>>,
        allowed: Option<&Bound<'py, PyAny>>,
        data: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let d = slf.emissions.as_ref().map(|e| e[0].dim());
        let c = slf.common.clone();
        let inp = input(
            sequences,
            holes,
            labels,
            allowed,
            data,
            c.direction,
            d,
            c.n_states,
            c.categorical(),
        )?;
        let family = c.family(slf.emissions.as_deref(), &inp.seqs)?;
        let letters = c.init_letters(
            "ste",
            &[
                ('s', slf.start.is_some()),
                ('t', slf.transition.is_some()),
                ('e', slf.emissions.is_some()),
            ],
        );
        let mut opts = c.fit_options(&letters)?;
        if slf.implementation == "scaled" {
            opts.implementation = Implementation::Scaled;
        }
        let (start, transition, emissions) = (
            slf.start.clone(),
            slf.transition.clone(),
            slf.emissions.clone(),
        );
        let (model, report) = py
            .detach(|| -> VResult<_> {
                let clamps = inp.clamps.as_deref();
                let mut stream = Stream::new(c.seed, &[tag::INIT, 0]);
                let mut m = Hmm::initialize(
                    c.n_states,
                    &family,
                    &inp.seqs,
                    clamps,
                    &opts.priors,
                    &mut stream,
                )?;
                match emissions {
                    Some(e) => m.emissions = e,
                    None => set_pseudocount(&mut m.emissions, c.pseudocount),
                }
                if let Some(s) = start {
                    m.start = s;
                }
                if let Some(t) = transition {
                    m.transition = t;
                }
                let mut m = Hmm::new(m.start, m.transition, m.emissions)?;
                let report = m.fit(&family, &inp.seqs, clamps, &opts)?;
                Ok((m, report))
            })
            .map_err(to_rust)?;
        slf.history = report.history;
        slf.converged = report.converged;
        slf.n_iter = report.n_iter;
        slf.model = Some(model);
        Ok(slf)
    }

    /// Total log-likelihood over holes (unclamped).
    #[pyo3(signature = (sequences, *, holes=None, data=None))]
    fn score(
        &self,
        py: Python,
        sequences: &Bound<PyAny>,
        holes: Option<&Bound<PyAny>>,
        data: Option<&Bound<PyAny>>,
    ) -> PyResult<f64> {
        let inp = self.input(sequences, holes, None, None, data)?;
        score_of(py, &Engined(self.fitted()?), &inp)
    }

    /// Log-likelihood of each hole (unclamped).
    #[pyo3(signature = (sequences, *, holes=None, data=None))]
    fn score_samples<'py>(
        &self,
        py: Python<'py>,
        sequences: &Bound<'py, PyAny>,
        holes: Option<&Bound<'py, PyAny>>,
        data: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let inp = self.input(sequences, holes, None, None, data)?;
        let e = Engined(self.fitted()?);
        let lls = py.detach(|| each(&inp, |_, s, _| e.log_likelihood(s)))?;
        Ok(vec1(py, lls))
    }

    /// Most likely states per row.
    ///
    /// Parameters
    /// ----------
    /// algorithm : {"viterbi", "map", "posterior_viterbi"}
    ///     Best joint path, per-row posterior argmax, or the best path through rows with
    ///     non-zero posterior.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray or list of numpy.ndarray
    ///     int64 states in input row order.
    #[pyo3(signature = (sequences, *, holes=None, labels=None, allowed=None, data=None, algorithm="viterbi"))]
    #[allow(clippy::too_many_arguments)]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        sequences: &Bound<'py, PyAny>,
        holes: Option<&Bound<'py, PyAny>>,
        labels: Option<&Bound<'py, PyAny>>,
        allowed: Option<&Bound<'py, PyAny>>,
        data: Option<&Bound<'py, PyAny>>,
        algorithm: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !matches!(algorithm, "viterbi" | "map" | "posterior_viterbi") {
            return Err(invalid(format!(
                "algorithm must be \"viterbi\", \"map\" or \"posterior_viterbi\", not {algorithm:?}"
            )));
        }
        let inp = self.input(sequences, holes, labels, allowed, data)?;
        predict_of(py, &Engined(self.fitted()?), &inp, algorithm)
    }

    /// Posterior state probabilities per row, ``(rows, K)``.
    #[pyo3(signature = (sequences, *, holes=None, labels=None, allowed=None, data=None))]
    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        sequences: &Bound<'py, PyAny>,
        holes: Option<&Bound<'py, PyAny>>,
        labels: Option<&Bound<'py, PyAny>>,
        allowed: Option<&Bound<'py, PyAny>>,
        data: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inp = self.input(sequences, holes, labels, allowed, data)?;
        proba_of(py, &Engined(self.fitted()?), &inp)
    }

    /// Viterbi path per row and its log-probability per hole.
    #[pyo3(signature = (sequences, *, holes=None, labels=None, allowed=None, data=None))]
    fn viterbi<'py>(
        &self,
        py: Python<'py>,
        sequences: &Bound<'py, PyAny>,
        holes: Option<&Bound<'py, PyAny>>,
        labels: Option<&Bound<'py, PyAny>>,
        allowed: Option<&Bound<'py, PyAny>>,
        data: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let inp = self.input(sequences, holes, labels, allowed, data)?;
        viterbi_of(py, &Engined(self.fitted()?), &inp)
    }

    /// Seeded state paths and observations, one per length, rows top-down.
    ///
    /// Returns
    /// -------
    /// tuple of (list of numpy.ndarray, list of numpy.ndarray)
    ///     States (int64) and observations (``(t, d)``; int64 ``(t,)`` codes when categorical).
    #[pyo3(signature = (lengths, *, seed=0))]
    fn sample<'py>(
        &self,
        py: Python<'py>,
        lengths: &Bound<'py, PyAny>,
        seed: u64,
    ) -> PyResult<Bound<'py, PyTuple>> {
        sample_of(
            py,
            &Engined(self.fitted()?),
            lengths,
            seed,
            self.common.direction,
        )
    }

    /// ``n`` seeded state paths drawn from the posterior, int64 ``(n, rows)``.
    #[pyo3(signature = (sequences, *, n=100, seed=0, holes=None, labels=None, allowed=None, data=None))]
    #[allow(clippy::too_many_arguments)]
    fn sample_posterior<'py>(
        &self,
        py: Python<'py>,
        sequences: &Bound<'py, PyAny>,
        n: usize,
        seed: u64,
        holes: Option<&Bound<'py, PyAny>>,
        labels: Option<&Bound<'py, PyAny>>,
        allowed: Option<&Bound<'py, PyAny>>,
        data: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inp = self.input(sequences, holes, labels, allowed, data)?;
        sample_posterior_of(py, &Engined(self.fitted()?), &inp, n, seed)
    }

    /// Akaike information criterion (lower is better).
    #[pyo3(signature = (sequences, *, holes=None, data=None))]
    fn aic(
        &self,
        py: Python,
        sequences: &Bound<PyAny>,
        holes: Option<&Bound<PyAny>>,
        data: Option<&Bound<PyAny>>,
    ) -> PyResult<f64> {
        let inp = self.input(sequences, holes, None, None, data)?;
        let (ll, p, _, _) = criteria(py, &Engined(self.fitted()?), &inp, &self.common.params)?;
        Ok(varvite_hsmm::aic(ll, p))
    }

    /// Bayesian information criterion (lower is better). Depth samples are autocorrelated, so
    /// BIC tends to favour too many states on dense logs.
    #[pyo3(signature = (sequences, *, holes=None, data=None))]
    fn bic(
        &self,
        py: Python,
        sequences: &Bound<PyAny>,
        holes: Option<&Bound<PyAny>>,
        data: Option<&Bound<PyAny>>,
    ) -> PyResult<f64> {
        let inp = self.input(sequences, holes, None, None, data)?;
        let (ll, p, n, _) = criteria(py, &Engined(self.fitted()?), &inp, &self.common.params)?;
        Ok(varvite_hsmm::bic(ll, p, n))
    }

    /// Integrated completed likelihood: BIC plus twice the path entropy (lower is better).
    #[pyo3(signature = (sequences, *, holes=None, data=None))]
    fn icl(
        &self,
        py: Python,
        sequences: &Bound<PyAny>,
        holes: Option<&Bound<PyAny>>,
        data: Option<&Bound<PyAny>>,
    ) -> PyResult<f64> {
        let inp = self.input(sequences, holes, None, None, data)?;
        let (ll, p, n, h) = criteria(py, &Engined(self.fitted()?), &inp, &self.common.params)?;
        Ok(varvite_hsmm::icl(ll, p, n, h))
    }

    #[getter]
    fn n_states(&self) -> usize {
        self.common.n_states
    }
    #[getter]
    fn direction(&self) -> &'static str {
        match self.common.direction {
            Direction::Up => "up",
            Direction::Down => "down",
        }
    }
    #[getter]
    fn start<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(vec1(py, self.fitted()?.start.clone()))
    }
    #[getter]
    fn transition<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(mat2(py, &self.fitted()?.transition))
    }
    #[getter]
    fn distributions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        distributions_py(py, &self.fitted()?.emissions)
    }
    /// Objective (log-likelihood plus log prior) after each EM iteration.
    #[getter]
    fn history<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec1(py, self.history.clone())
    }
    #[getter]
    fn converged(&self) -> bool {
        self.converged
    }
    #[getter]
    fn n_iter(&self) -> usize {
        self.n_iter
    }

    fn to_json(&self) -> PyResult<String> {
        json(self)
    }
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        let m: Self = unjson(text, "HiddenMarkovModel")?;
        if let Some(h) = &m.model {
            Hmm::new(h.start.clone(), h.transition.clone(), h.emissions.clone())
                .map_err(to_rust)?;
        }
        Ok(m)
    }
    fn __repr__(&self) -> String {
        format!(
            "HiddenMarkovModel(n_states={}, emission={:?}, fitted={})",
            self.common.n_states,
            self.common.emission,
            self.model.is_some()
        )
    }
}

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyCategorical>()?;
    m.add_class::<PyGaussian>()?;
    m.add_class::<PyPoisson>()?;
    m.add_class::<PyGaussianMixture>()?;
    m.add_class::<PyHiddenMarkovModel>()?;
    Ok(())
}
