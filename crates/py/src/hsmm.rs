//! Duration laws, `HiddenSemiMarkovModel` and `select_states`.

use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use serde::{Deserialize, Serialize};
use varvite_core::rng::tag;
use varvite_core::{Mat, Result as VResult, Stream};
use varvite_hmm::{Clamp, Emission, Params, Sequence};
use varvite_hsmm::{
    Duration, DurationDist, DurationFamily, Hsmm, ModelKind, SelectionSpec, select_states as select,
};

use crate::args::{direction, floats, mat, mat2, to_rust, vec1};
use crate::core::{json, unjson};
use crate::hmm::{
    Common, Engine, Input, PyHiddenMarkovModel, covariance_type, criteria, distributions,
    distributions_py, each, emission_name, input, predict_of, proba_of, sample_of,
    sample_posterior_of, score_of, set_pseudocount, viterbi_of,
};
use crate::invalid;

fn law_checked(d: DurationDist) -> PyResult<DurationDist> {
    d.law.validate(d.law_support()).map_err(to_rust)?;
    if !(d.pseudocount >= 0.0 && d.pseudocount.is_finite()) {
        return Err(invalid("pseudocount must be finite and non-negative"));
    }
    Ok(d)
}

trait Support {
    fn law_support(&self) -> usize;
}

impl Support for DurationDist {
    /// A support on which every valid law has mass: the empirical pmf length, else 1.
    fn law_support(&self) -> usize {
        match &self.law {
            Duration::Empirical { pmf } => pmf.len().max(1),
            _ => 1,
        }
    }
}

fn pmf_of<'py>(
    py: Python<'py>,
    d: &DurationDist,
    max_duration: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if max_duration == 0 {
        return Err(invalid("max_duration must be at least 1"));
    }
    varvite_core::memory::check(
        (max_duration as u64).saturating_mul(24),
        None,
        "duration pmf",
    )
    .map_err(to_rust)?;
    d.law.validate(max_duration).map_err(to_rust)?;
    Ok(vec1(
        py,
        d.law
            .log_pmf(max_duration)
            .into_iter()
            .map(f64::exp)
            .collect(),
    ))
}

fn mean_of(d: &DurationDist, max_duration: usize) -> PyResult<f64> {
    if max_duration == 0 {
        return Err(invalid("max_duration must be at least 1"));
    }
    varvite_core::memory::check(
        (max_duration as u64).saturating_mul(24),
        None,
        "duration mean",
    )
    .map_err(to_rust)?;
    d.law.validate(max_duration).map_err(to_rust)?;
    Ok(d.law.mean(max_duration))
}

macro_rules! duration_class {
    ($ty:ident, $name:literal, { $($own:tt)* }) => {
        #[pymethods]
        impl $ty {
            $($own)*

            /// Probabilities of durations ``1..=max_duration``, renormalised on that support.
            fn pmf<'py>(&self, py: Python<'py>, max_duration: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
                pmf_of(py, &self.0, max_duration)
            }
            /// Mean duration of the law truncated to ``1..=max_duration``.
            fn mean(&self, max_duration: usize) -> PyResult<f64> {
                mean_of(&self.0, max_duration)
            }
            #[getter]
            fn frozen(&self) -> bool {
                self.0.frozen
            }
            fn to_json(&self) -> PyResult<String> {
                json(&self.0)
            }
            #[staticmethod]
            fn from_json(text: &str) -> PyResult<Self> {
                let d: DurationDist = unjson(text, $name)?;
                let d = law_checked(d)?;
                match (&d.law, $name) {
                    (Duration::Geometric { .. }, "Geometric")
                    | (Duration::Poisson { .. }, "ShiftedPoisson")
                    | (Duration::NegativeBinomial { .. }, "NegativeBinomial")
                    | (Duration::Empirical { .. }, "EmpiricalDuration") => Ok(Self(d)),
                    _ => Err(invalid(format!("not {} JSON", $name))),
                }
            }
            fn __repr__(&self) -> String {
                let args = match &self.0.law {
                    Duration::Geometric { p } => format!("p={p}"),
                    Duration::Poisson { rate } => format!("rate={rate}"),
                    Duration::NegativeBinomial { r, p } => format!("r={r}, p={p}"),
                    Duration::Empirical { pmf } => format!("pmf of length {}", pmf.len()),
                };
                format!("{}({args})", $name)
            }
        }
    };
}

/// Geometric duration ``p (1 - p)^(u - 1)`` on ``u = 1, 2, ...``: the duration an HMM implies.
///
/// Parameters
/// ----------
/// p : float
///     ``0 < p <= 1``.
/// frozen : bool
///     Keep the law fixed while fitting.
#[pyclass(module = "varvite", name = "Geometric", frozen, eq, from_py_object)]
#[derive(Clone, PartialEq)]
pub struct PyGeometric(DurationDist);

duration_class!(PyGeometric, "Geometric", {
    #[new]
    #[pyo3(signature = (p, *, frozen=false))]
    fn new(p: f64, frozen: bool) -> PyResult<Self> {
        law_checked(DurationDist {
            law: Duration::Geometric { p },
            frozen,
            pseudocount: 0.0,
        })
        .map(Self)
    }
    #[getter]
    fn p(&self) -> f64 {
        match self.0.law {
            Duration::Geometric { p } => p,
            _ => f64::NAN,
        }
    }
});

/// Shifted Poisson duration ``e^-rate rate^(u-1) / (u-1)!`` on ``u = 1, 2, ...``.
///
/// Parameters
/// ----------
/// rate : float
///     ``rate >= 0``; the mean is ``1 + rate``.
/// frozen : bool
#[pyclass(
    module = "varvite",
    name = "ShiftedPoisson",
    frozen,
    eq,
    from_py_object
)]
#[derive(Clone, PartialEq)]
pub struct PyShiftedPoisson(DurationDist);

duration_class!(PyShiftedPoisson, "ShiftedPoisson", {
    #[new]
    #[pyo3(signature = (rate, *, frozen=false))]
    fn new(rate: f64, frozen: bool) -> PyResult<Self> {
        law_checked(DurationDist {
            law: Duration::Poisson { rate },
            frozen,
            pseudocount: 0.0,
        })
        .map(Self)
    }
    #[getter]
    fn rate(&self) -> f64 {
        match self.0.law {
            Duration::Poisson { rate } => rate,
            _ => f64::NAN,
        }
    }
});

/// Shifted negative binomial duration ``C(u + r - 2, u - 1) p^r (1 - p)^(u - 1)``.
///
/// Parameters
/// ----------
/// r : float
///     ``r > 0``.
/// p : float
///     ``0 < p <= 1``.
/// frozen : bool
#[pyclass(
    module = "varvite",
    name = "NegativeBinomial",
    frozen,
    eq,
    from_py_object
)]
#[derive(Clone, PartialEq)]
pub struct PyNegativeBinomial(DurationDist);

duration_class!(PyNegativeBinomial, "NegativeBinomial", {
    #[new]
    #[pyo3(signature = (r, p, *, frozen=false))]
    fn new(r: f64, p: f64, frozen: bool) -> PyResult<Self> {
        law_checked(DurationDist {
            law: Duration::NegativeBinomial { r, p },
            frozen,
            pseudocount: 0.0,
        })
        .map(Self)
    }
    #[getter]
    fn r(&self) -> f64 {
        match self.0.law {
            Duration::NegativeBinomial { r, .. } => r,
            _ => f64::NAN,
        }
    }
    #[getter]
    fn p(&self) -> f64 {
        match self.0.law {
            Duration::NegativeBinomial { p, .. } => p,
            _ => f64::NAN,
        }
    }
});

/// Empirical duration: ``pmf[u - 1]`` on ``u = 1..len(pmf)``.
///
/// Parameters
/// ----------
/// pmf : array_like
///     Non-negative weights, renormalised.
/// pseudocount : float
///     Added to every duration count in the M-step.
/// frozen : bool
#[pyclass(
    module = "varvite",
    name = "EmpiricalDuration",
    frozen,
    eq,
    from_py_object
)]
#[derive(Clone, PartialEq)]
pub struct PyEmpiricalDuration(DurationDist);

duration_class!(PyEmpiricalDuration, "EmpiricalDuration", {
    #[new]
    #[pyo3(signature = (pmf, *, pseudocount=0.0, frozen=false))]
    fn new(pmf: &Bound<PyAny>, pseudocount: f64, frozen: bool) -> PyResult<Self> {
        law_checked(DurationDist {
            law: Duration::Empirical {
                pmf: floats(pmf, "pmf")?,
            },
            frozen,
            pseudocount,
        })
        .map(Self)
    }
    #[getter]
    fn pseudocount(&self) -> f64 {
        self.0.pseudocount
    }
});

fn duration_of(obj: &Bound<PyAny>) -> PyResult<DurationDist> {
    if let Ok(d) = obj.cast::<PyGeometric>() {
        return Ok(d.get().0.clone());
    }
    if let Ok(d) = obj.cast::<PyShiftedPoisson>() {
        return Ok(d.get().0.clone());
    }
    if let Ok(d) = obj.cast::<PyNegativeBinomial>() {
        return Ok(d.get().0.clone());
    }
    if let Ok(d) = obj.cast::<PyEmpiricalDuration>() {
        return Ok(d.get().0.clone());
    }
    Err(invalid(
        "durations must be Geometric, ShiftedPoisson, NegativeBinomial or EmpiricalDuration objects",
    ))
}

fn duration_py(py: Python, d: &DurationDist) -> PyResult<Py<PyAny>> {
    Ok(match d.law {
        Duration::Geometric { .. } => Py::new(py, PyGeometric(d.clone()))?.into_any(),
        Duration::Poisson { .. } => Py::new(py, PyShiftedPoisson(d.clone()))?.into_any(),
        Duration::NegativeBinomial { .. } => Py::new(py, PyNegativeBinomial(d.clone()))?.into_any(),
        Duration::Empirical { .. } => Py::new(py, PyEmpiricalDuration(d.clone()))?.into_any(),
    })
}

fn duration_family(s: &str) -> PyResult<DurationFamily> {
    match s {
        "geometric" => Ok(DurationFamily::Geometric),
        "poisson" => Ok(DurationFamily::Poisson),
        "negative_binomial" => Ok(DurationFamily::NegativeBinomial),
        "empirical" => Ok(DurationFamily::Empirical),
        _ => Err(invalid(format!(
            "duration must be \"geometric\", \"poisson\", \"negative_binomial\" or \"empirical\", not {s:?}"
        ))),
    }
}

fn family_of_law(d: &Duration) -> DurationFamily {
    match d {
        Duration::Geometric { .. } => DurationFamily::Geometric,
        Duration::Poisson { .. } => DurationFamily::Poisson,
        Duration::NegativeBinomial { .. } => DurationFamily::NegativeBinomial,
        Duration::Empirical { .. } => DurationFamily::Empirical,
    }
}

/// `max_duration` per state: an int for all, a sequence, or None for the longest sequence.
fn max_durations(
    obj: Option<&Bound<PyAny>>,
    k: usize,
    data: Option<&[Sequence]>,
) -> PyResult<Option<Vec<usize>>> {
    let Some(obj) = obj.filter(|o| !o.is_none()) else {
        return Ok(data.map(|d| vec![d.iter().map(|s| s.t).max().unwrap_or(1).max(1); k]));
    };
    let m: Vec<usize> = match obj.extract::<usize>() {
        Ok(m) => vec![m; k],
        Err(_) => obj
            .extract()
            .map_err(|_| invalid("max_duration must be an int, a sequence of ints or None"))?,
    };
    if m.len() != k {
        return Err(invalid(format!(
            "give {k} max_duration values, not {}",
            m.len()
        )));
    }
    Ok(Some(m))
}

struct Engined<'a>(&'a Hsmm);

impl Engine for Engined<'_> {
    fn k(&self) -> usize {
        self.0.start.len()
    }
    fn emissions(&self) -> &[Emission] {
        &self.0.emissions
    }
    fn log_likelihood(&self, seq: &Sequence) -> VResult<f64> {
        self.0.log_likelihood(seq, None)
    }
    fn posterior(&self, seq: &Sequence, clamp: Option<&Clamp>) -> VResult<(Mat, f64, f64)> {
        let p = self.0.posterior(seq, clamp)?;
        Ok((p.gamma, p.entropy, p.log_likelihood))
    }
    fn viterbi(&self, seq: &Sequence, clamp: Option<&Clamp>) -> VResult<(Vec<u32>, f64)> {
        self.0.viterbi(seq, clamp)
    }
    fn decode(&self, seq: &Sequence, clamp: Option<&Clamp>, algorithm: &str) -> VResult<Vec<u32>> {
        if algorithm == "viterbi" {
            return Ok(self.0.viterbi(seq, clamp)?.0);
        }
        let g = self.0.posterior(seq, clamp)?.gamma;
        Ok((0..g.nrows())
            .map(|t| {
                (0..g.ncols()).fold(0, |b, j| if g[(t, j)] > g[(t, b)] { j } else { b }) as u32
            })
            .collect())
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

/// Hidden semi-Markov model: an HMM whose states last an explicit number of steps (bed
/// thickness), with right-censored first and last segments.
///
/// Parameters
/// ----------
/// distributions : int or sequence of distributions
///     The number of states, or one emission object per state.
/// durations : sequence of duration laws, optional
///     One ``Geometric``, ``ShiftedPoisson``, ``NegativeBinomial`` or ``EmpiricalDuration`` per
///     state.
/// start, transition : array_like, optional
///     Start probabilities and a transition matrix with a zero diagonal.
/// duration : {"geometric", "poisson", "negative_binomial", "empirical"}
///     Duration family fitted when ``durations`` is not given.
/// max_duration : int, sequence of int or None
///     Longest duration per state; None means the longest sequence (no truncation).
/// params, init_params : str
///     Letters of the parts fitted and initialised: s start, t transition, e emissions,
///     d durations.
/// emission, covariance_type, n_symbols, n_mix, max_iter, tol, n_init, seed, start_prior,
/// transition_prior, covariance_prior, pseudocount, direction, ordering
///     As in ``HiddenMarkovModel``.
#[pyclass(module = "varvite", name = "HiddenSemiMarkovModel", eq, from_py_object)]
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct PyHiddenSemiMarkovModel {
    common: Common,
    duration: String,
    max_duration: Option<Vec<usize>>,
    start: Option<Vec<f64>>,
    #[serde(with = "varvite_core::serde_mat::option")]
    transition: Option<Mat>,
    emissions: Option<Vec<Emission>>,
    durations: Option<Vec<DurationDist>>,
    model: Option<Hsmm>,
    history: Vec<f64>,
    converged: bool,
    n_iter: usize,
}

impl PyHiddenSemiMarkovModel {
    fn fitted(&self) -> PyResult<&Hsmm> {
        self.model.as_ref().ok_or_else(|| {
            invalid("the model is not fitted; call fit or give start, transition, distributions, durations and max_duration")
        })
    }

    fn family(&self) -> PyResult<DurationFamily> {
        match self.durations.as_ref().and_then(|d| d.first()) {
            Some(d) => Ok(family_of_law(&d.law)),
            None => duration_family(&self.duration),
        }
    }

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
impl PyHiddenSemiMarkovModel {
    #[new]
    #[pyo3(signature = (distributions, durations=None, *, start=None, transition=None, emission="gaussian", duration="geometric", covariance_type="diag", n_symbols=None, n_mix=1, max_duration=None, max_iter=100, tol=1e-6, n_init=1, seed=0, params="sted", init_params=None, start_prior=1.0, transition_prior=1.0, covariance_prior=1e-6, pseudocount=0.0, direction="up", ordering="none"))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        distributions: &Bound<PyAny>,
        durations: Option<&Bound<PyAny>>,
        start: Option<&Bound<PyAny>>,
        transition: Option<&Bound<PyAny>>,
        emission: &str,
        duration: &str,
        covariance_type: &str,
        n_symbols: Option<usize>,
        n_mix: usize,
        max_duration: Option<&Bound<PyAny>>,
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
        ordering: &str,
    ) -> PyResult<Self> {
        let (k, emissions) = self::distributions(distributions)?;
        duration_family(duration)?;
        let emission = match &emissions {
            Some(e) => match e[0] {
                Emission::Categorical(_) => "categorical".to_string(),
                Emission::Gaussian(_) => "gaussian".to_string(),
                Emission::Poisson(_) => "poisson".to_string(),
                Emission::Mixture(_) => "mixture".to_string(),
            },
            None => emission_name(emission)?,
        };
        let durations = durations
            .filter(|o| !o.is_none())
            .map(|o| -> PyResult<Vec<DurationDist>> {
                let items: Vec<Bound<PyAny>> = o.try_iter()?.collect::<PyResult<_>>()?;
                items.iter().map(duration_of).collect()
            })
            .transpose()?;
        if let Some(d) = &durations
            && d.len() != k
        {
            return Err(invalid(format!(
                "give {k} durations, one per state, not {}",
                d.len()
            )));
        }
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
        let max_duration = max_durations(max_duration, k, None)?;
        let start = start
            .filter(|o| !o.is_none())
            .map(|o| floats(o, "start"))
            .transpose()?;
        let transition = transition
            .filter(|o| !o.is_none())
            .map(|o| mat(o, "transition"))
            .transpose()?;
        let model = match (&start, &transition, &emissions, &durations, &max_duration) {
            (Some(s), Some(t), Some(e), Some(d), Some(m)) => Some(
                Hsmm::new(s.clone(), t.clone(), d.clone(), e.clone(), m.clone())
                    .map_err(to_rust)?,
            ),
            _ => None,
        };
        Ok(Self {
            common,
            duration: duration.to_string(),
            max_duration,
            start,
            transition,
            emissions,
            durations,
            model,
            history: vec![],
            converged: false,
            n_iter: 0,
        })
    }

    /// The HSMM equivalent to ``hmm``: geometric durations with ``p = 1 - a_jj`` and the
    /// transition matrix renormalised off the diagonal.
    #[staticmethod]
    #[pyo3(signature = (hmm, *, max_duration))]
    fn from_hmm(hmm: &PyHiddenMarkovModel, max_duration: usize) -> PyResult<Self> {
        let h = hmm.fitted()?;
        let model = Hsmm::from_hmm(h, max_duration).map_err(to_rust)?;
        let k = h.k();
        let common = hmm.common().clone();
        Ok(Self {
            common: Common {
                n_states: k,
                params: "sted".to_string(),
                ..common
            },
            duration: "geometric".to_string(),
            max_duration: Some(model.max_duration.clone()),
            start: Some(model.start.clone()),
            transition: Some(model.transition.clone()),
            emissions: Some(model.emissions.clone()),
            durations: Some(model.durations.clone()),
            model: Some(model),
            history: vec![],
            converged: false,
            n_iter: 0,
        })
    }

    /// EM fit over all holes, with seeded restarts. Arguments as in
    /// ``HiddenMarkovModel.fit``.
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
        let dfam = slf.family()?;
        let max_duration = match &slf.max_duration {
            Some(m) => m.clone(),
            None => max_durations(None, c.n_states, Some(&inp.seqs))?.expect("data given"),
        };
        let letters = c.init_letters(
            "sted",
            &[
                ('s', slf.start.is_some()),
                ('t', slf.transition.is_some()),
                ('e', slf.emissions.is_some()),
                ('d', slf.durations.is_some()),
            ],
        );
        let opts = c.fit_options(&letters)?;
        let (start, transition, emissions, durations) = (
            slf.start.clone(),
            slf.transition.clone(),
            slf.emissions.clone(),
            slf.durations.clone(),
        );
        let (model, report) = py
            .detach(|| -> VResult<_> {
                let clamps = inp.clamps.as_deref();
                let mut stream = Stream::new(c.seed, &[tag::INIT, 0]);
                let mut m = Hsmm::initialize(
                    c.n_states,
                    &family,
                    dfam,
                    &inp.seqs,
                    clamps,
                    &opts.priors,
                    &max_duration,
                    &mut stream,
                )?;
                match emissions {
                    Some(e) => m.emissions = e,
                    None => set_pseudocount(&mut m.emissions, c.pseudocount),
                }
                match durations {
                    Some(d) => m.durations = d,
                    None => {
                        for d in &mut m.durations {
                            d.pseudocount = c.pseudocount;
                        }
                    }
                }
                if let Some(s) = start {
                    m.start = s;
                }
                if let Some(t) = transition {
                    m.transition = t;
                }
                let mut m = Hsmm::new(
                    m.start,
                    m.transition,
                    m.durations,
                    m.emissions,
                    m.max_duration,
                )?;
                let report = m.fit(&family, dfam, &inp.seqs, clamps, &opts)?;
                Ok((m, report))
            })
            .map_err(to_rust)?;
        slf.history = report.history;
        slf.converged = report.converged;
        slf.n_iter = report.n_iter;
        slf.model = Some(model);
        Ok(slf)
    }

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

    /// Most likely states per row: ``algorithm`` "viterbi" (best segmentation) or "map"
    /// (per-row posterior argmax).
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
        if !matches!(algorithm, "viterbi" | "map") {
            return Err(invalid(format!(
                "algorithm must be \"viterbi\" or \"map\", not {algorithm:?}"
            )));
        }
        let inp = self.input(sequences, holes, labels, allowed, data)?;
        predict_of(py, &Engined(self.fitted()?), &inp, algorithm)
    }

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
    #[getter]
    fn durations<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let d = &self.fitted()?.durations;
        PyList::new(
            py,
            d.iter()
                .map(|d| duration_py(py, d))
                .collect::<PyResult<Vec<_>>>()?,
        )
    }
    #[getter]
    fn max_duration<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        Ok(PyArray1::from_vec(
            py,
            self.fitted()?
                .max_duration
                .iter()
                .map(|&m| m as i64)
                .collect(),
        ))
    }
    /// Mass of each untruncated duration law above its ``max_duration``.
    #[getter]
    fn truncated_mass<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(vec1(py, self.fitted()?.truncated_mass()))
    }
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
        let m: Self = unjson(text, "HiddenSemiMarkovModel")?;
        if let Some(h) = &m.model {
            Hsmm::new(
                h.start.clone(),
                h.transition.clone(),
                h.durations.clone(),
                h.emissions.clone(),
                h.max_duration.clone(),
            )
            .map_err(to_rust)?;
        }
        Ok(m)
    }
    fn __repr__(&self) -> String {
        format!(
            "HiddenSemiMarkovModel(n_states={}, emission={:?}, fitted={})",
            self.common.n_states,
            self.common.emission,
            self.model.is_some()
        )
    }
}

/// Fits models with each number of states and tabulates information criteria and
/// cross-validated log-likelihood.
///
/// Depth samples are autocorrelated, so BIC tends to favour too many states on dense logs; the
/// cross-validated score is the honest check.
///
/// Parameters
/// ----------
/// sequences, holes, data
///     As in ``HiddenMarkovModel.fit``.
/// labels, allowed
///     Clamps; only with a single entry in ``states``.
/// states : sequence of int
/// model : {"hmm", "hsmm"}
/// emission, covariance_type, n_symbols, n_mix, duration, max_duration
///     Model family, as in the classes.
/// cv_folds : int, optional
///     None leaves one hole out, 0 skips cross-validation, k > 0 uses k seeded folds.
/// n_init, seed, max_iter, tol, direction
///     Fit settings.
///
/// Returns
/// -------
/// dict
///     ``states``, ``log_likelihood``, ``n_params``, ``aic``, ``bic``, ``icl``,
///     ``cv_log_likelihood`` and ``cv_per_step``.
#[pyfunction]
#[pyo3(signature = (sequences, *, holes=None, data=None, labels=None, allowed=None, states=None, model="hmm", emission="gaussian", covariance_type="diag", n_symbols=None, n_mix=1, duration="geometric", max_duration=None, cv_folds=None, n_init=5, seed=0, max_iter=100, tol=1e-6, direction="up"))]
#[allow(clippy::too_many_arguments)]
fn select_states<'py>(
    py: Python<'py>,
    sequences: &Bound<'py, PyAny>,
    holes: Option<&Bound<'py, PyAny>>,
    data: Option<&Bound<'py, PyAny>>,
    labels: Option<&Bound<'py, PyAny>>,
    allowed: Option<&Bound<'py, PyAny>>,
    states: Option<Vec<usize>>,
    model: &str,
    emission: &str,
    covariance_type: &str,
    n_symbols: Option<usize>,
    n_mix: usize,
    duration: &str,
    max_duration: Option<usize>,
    cv_folds: Option<usize>,
    n_init: usize,
    seed: u64,
    max_iter: usize,
    tol: f64,
    direction: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let states = states.unwrap_or_else(|| (1..7).collect());
    let clamped = labels.is_some_and(|o| !o.is_none()) || allowed.is_some_and(|o| !o.is_none());
    if clamped && states.len() != 1 {
        return Err(invalid(
            "labels and allowed fix the number of states; give a single entry in states",
        ));
    }
    let k = states.first().copied().unwrap_or(1);
    let common = Common {
        n_states: k,
        emission: emission_name(emission)?,
        covariance_type: self::covariance_type(covariance_type)?,
        n_symbols,
        n_mix,
        max_iter,
        tol,
        n_init,
        seed,
        params: if model == "hsmm" { "sted" } else { "ste" }.to_string(),
        init_params: None,
        start_prior: 1.0,
        transition_prior: 1.0,
        covariance_prior: 1e-6,
        pseudocount: 0.0,
        direction: self::direction(direction)?,
        ordering: "none".to_string(),
    };
    let inp = input(
        sequences,
        holes,
        labels,
        allowed,
        data,
        common.direction,
        None,
        k,
        common.categorical(),
    )?;
    let kind = match model {
        "hmm" => ModelKind::Hmm,
        "hsmm" => ModelKind::Hsmm {
            duration: duration_family(duration)?,
            max_duration,
        },
        _ => {
            return Err(invalid(format!(
                "model must be \"hmm\" or \"hsmm\", not {model:?}"
            )));
        }
    };
    let spec = SelectionSpec {
        states,
        family: common.family(None, &inp.seqs)?,
        kind,
        fit: common.fit_options(&common.params)?,
        cv_folds,
    };
    let t = py
        .detach(|| select(&inp.seqs, inp.clamps.as_deref(), &spec))
        .map_err(to_rust)?;
    let ints = |v: &[usize]| PyArray1::from_vec(py, v.iter().map(|&x| x as i64).collect());
    let d = PyDict::new(py);
    d.set_item("states", ints(&t.states))?;
    d.set_item("log_likelihood", vec1(py, t.log_likelihood))?;
    d.set_item("n_params", ints(&t.n_params))?;
    d.set_item("aic", vec1(py, t.aic))?;
    d.set_item("bic", vec1(py, t.bic))?;
    d.set_item("icl", vec1(py, t.icl))?;
    d.set_item("cv_log_likelihood", vec1(py, t.cv_log_likelihood))?;
    d.set_item("cv_per_step", vec1(py, t.cv_per_step))?;
    Ok(d)
}

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyGeometric>()?;
    m.add_class::<PyShiftedPoisson>()?;
    m.add_class::<PyNegativeBinomial>()?;
    m.add_class::<PyEmpiricalDuration>()?;
    m.add_class::<PyHiddenSemiMarkovModel>()?;
    m.add_function(wrap_pyfunction!(select_states, m)?)?;
    Ok(())
}
