//! `Categories` and `Grid`.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};
use transiograms_core::{Categories, Grid, MISSING};

use crate::args::{codes, coords, to_rust};
use crate::invalid;

pub(crate) fn json<T: serde::Serialize>(v: &T) -> PyResult<String> {
    serde_json::to_string(v).map_err(invalid)
}

pub(crate) fn unjson<T: serde::de::DeserializeOwned>(text: &str, what: &str) -> PyResult<T> {
    serde_json::from_str(text).map_err(|e| invalid(format!("not {what} JSON: {e}")))
}

/// Equality through JSON, so NaN entries (written as null) compare equal.
pub(crate) fn same_json<T: serde::Serialize>(a: &T, b: &T) -> PyResult<bool> {
    Ok(json(a)? == json(b)?)
}

/// A label as text: None and NaN are missing; integral floats lose their decimal point.
fn label(obj: &Bound<PyAny>) -> PyResult<Option<String>> {
    if obj.is_none() {
        return Ok(None);
    }
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(Some(s.to_str()?.to_string()));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(Some(i.to_string()));
    }
    match obj.extract::<f64>() {
        Ok(f) if f.is_nan() => Ok(None),
        Ok(f) if f.fract() == 0.0 && f.abs() < 9e15 => Ok(Some((f as i64).to_string())),
        Ok(f) => Ok(Some(f.to_string())),
        Err(_) => Ok(Some(obj.str()?.to_str()?.to_string())),
    }
}

fn labels(obj: &Bound<PyAny>) -> PyResult<Vec<Option<String>>> {
    let list = obj
        .py()
        .import("numpy")?
        .call_method1("asarray", (obj, "object"))?
        .call_method1("reshape", (-1i64,))?
        .call_method0("tolist")?;
    list.cast::<PyList>()?.iter().map(|o| label(&o)).collect()
}

/// Named categories: a code is the index of its name.
///
/// Parameters
/// ----------
/// names : sequence of str
///     Category names in code order, unique and non-empty.
/// colors : sequence of str, optional
///     One matplotlib color per name.
///
/// Raises
/// ------
/// InvalidInput
///     When a name is empty or repeated, or ``colors`` has another length.
///
/// Examples
/// --------
/// >>> cats = tg.Categories(["sand", "silt", "clay"])
/// >>> len(cats), cats.names
/// (3, ['sand', 'silt', 'clay'])
/// >>> codes = cats.encode(["clay", None, "sand"])
/// >>> codes
/// array([ 2, -1,  0])
/// >>> cats.decode(codes)
/// array(['clay', None, 'sand'], dtype=object)
/// >>> tg.Categories.from_labels(["b", "a", "b"]).names
/// ['a', 'b']
/// >>> tg.Categories.from_json(cats.to_json()) == cats
/// True
#[pyclass(
    module = "transiograms",
    name = "Categories",
    frozen,
    eq,
    from_py_object
)]
#[derive(Clone, PartialEq)]
pub struct PyCategories(pub Categories);

#[pymethods]
impl PyCategories {
    #[new]
    #[pyo3(signature = (names, *, colors=None))]
    fn new(names: &Bound<PyAny>, colors: Option<Vec<String>>) -> PyResult<Self> {
        if colors.is_none() && crate::protocol::is_protocol(names)? {
            return crate::protocol::categories_from_protocol(names).map(Self);
        }
        let names: Vec<String> = names.extract()?;
        Categories::new(names, colors).map(Self).map_err(to_rust)
    }

    /// Categories from the sorted unique non-missing labels; numeric order when every label is an
    /// integer.
    ///
    /// Parameters
    /// ----------
    /// labels : array_like
    ///     Labels of any type; None and NaN are missing.
    ///
    /// Returns
    /// -------
    /// Categories
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When there is no non-missing label.
    #[staticmethod]
    pub fn from_labels(labels: &Bound<PyAny>) -> PyResult<Self> {
        let l = self::labels(labels)?;
        Categories::from_labels(l.iter().map(Option::as_deref))
            .map(Self)
            .map_err(to_rust)
    }

    /// Category names in code order.
    #[getter]
    fn names(&self) -> Vec<String> {
        self.0.names.clone()
    }

    /// One matplotlib color per category, or None.
    #[getter]
    fn colors(&self) -> Option<Vec<String>> {
        self.0.colors.clone()
    }

    /// Codes of ``labels``.
    ///
    /// Parameters
    /// ----------
    /// labels : array_like
    ///     Category names; None and NaN are missing.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     int64 codes, -1 for missing.
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When a label is not one of the names.
    fn encode<'py>(
        &self,
        py: Python<'py>,
        labels: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let l = self::labels(labels)?;
        let codes = self
            .0
            .encode(l.iter().map(Option::as_deref))
            .map_err(to_rust)?;
        Ok(PyArray1::from_vec(
            py,
            codes
                .into_iter()
                .map(|c| if c == MISSING { -1 } else { c as i64 })
                .collect(),
        ))
    }

    /// Names of ``codes``.
    ///
    /// Parameters
    /// ----------
    /// codes : array_like
    ///     Codes ``0..k-1``, -1 or NaN for missing.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Object array of names, None for missing.
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When a code is not ``-1`` or in ``0..k-1``.
    fn decode<'py>(
        &self,
        py: Python<'py>,
        codes: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let codes = self::codes(codes, "codes")?;
        transiograms_core::codes::check_codes(&codes, self.0.len()).map_err(to_rust)?;
        let names: Vec<Option<&str>> = codes.iter().map(|&c| self.0.name(c)).collect();
        py.import("numpy")?.call_method1("array", (names, "object"))
    }

    fn __len__(&self) -> usize {
        self.0.len()
    }

    /// JSON text of this object; ``Categories.from_json`` reads it back.
    ///
    /// Returns
    /// -------
    /// str
    fn to_json(&self) -> PyResult<String> {
        json(&self.0)
    }

    /// Categories from the text written by ``to_json``.
    ///
    /// Parameters
    /// ----------
    /// text : str
    ///
    /// Returns
    /// -------
    /// Categories
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When ``text`` is not Categories JSON or describes an invalid Categories.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        let c: Categories = unjson(text, "Categories")?;
        Categories::new(c.names, c.colors)
            .map(Self)
            .map_err(to_rust)
    }

    fn __repr__(&self) -> String {
        format!("Categories({:?})", self.0.names)
    }

    /// Protocol v1 dictionary of this category table (``kind="categories"``).
    ///
    /// Returns
    /// -------
    /// dict
    fn __minetypes__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = crate::protocol::category_entry_dict(py, &self.0.names, self.0.colors.as_deref())?;
        d.set_item("protocol", 1)?;
        d.set_item("kind", "categories")?;
        d.set_item("columns", PyDict::new(py))?;
        d.set_item("units", PyDict::new(py))?;
        d.set_item("categories", PyDict::new(py))?;
        d.set_item("crs", py.None())?;
        d.set_item("length_unit", py.None())?;
        Ok(d)
    }
}

/// Axis-aligned grid, cells x fastest, then y, then z.
///
/// Parameters
/// ----------
/// origin : sequence of 3 floats
///     Corner of the first cell (minimum x, y, z).
/// size : sequence of 3 floats
///     Cell size along x, y and z.
/// count : sequence of 3 ints
///     Number of cells along x, y and z; 1 along z for a 2D grid.
///
/// Raises
/// ------
/// InvalidInput
///     When a size is not positive and finite or a count is 0.
///
/// Examples
/// --------
/// >>> grid = tg.Grid((0.0, 0.0, 0.0), (10.0, 10.0, 1.0), (3, 2, 4))
/// >>> grid.n_cells, grid.shape
/// (24, (4, 2, 3))
/// >>> grid.coords[:2]
/// array([[ 5. ,  5. ,  0.5],
///        [15. ,  5. ,  0.5]])
/// >>> cells, codes = grid.snap([[4.0, 6.0, 0.2], [99.0, 0.0, 0.0]], [1, 2])
/// >>> cells, codes
/// (array([0], dtype=uint64), array([1]))
/// >>> tg.Grid.from_json(grid.to_json()) == grid
/// True
#[pyclass(module = "transiograms", name = "Grid", frozen, eq, from_py_object)]
#[derive(Clone, PartialEq)]
pub struct PyGrid(pub Grid);

#[pymethods]
impl PyGrid {
    #[new]
    #[pyo3(signature = (origin, size=None, count=None))]
    fn new(
        origin: &Bound<PyAny>,
        size: Option<[f64; 3]>,
        count: Option<[usize; 3]>,
    ) -> PyResult<Self> {
        if size.is_none() && count.is_none() && crate::protocol::is_protocol(origin)? {
            let (grid, mask, _, _, _, _) = crate::protocol::resolve_grid_target(origin)?;
            if mask.is_some() {
                return Err(invalid(
                    "masked block_model cannot be converted to a plain Grid",
                ));
            }
            return Ok(Self(grid));
        }
        let origin: [f64; 3] = origin.extract()?;
        let Some(size) = size else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Grid() missing required positional arguments: 'size' and 'count'",
            ));
        };
        let Some(count) = count else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Grid() missing required positional argument: 'count'",
            ));
        };
        Grid::new(origin, size, count).map(Self).map_err(to_rust)
    }

    /// Corner of the first cell ``(x, y, z)``.
    #[getter]
    fn origin(&self) -> (f64, f64, f64) {
        let o = self.0.origin;
        (o[0], o[1], o[2])
    }

    /// Cell size along x, y and z.
    #[getter]
    fn size(&self) -> (f64, f64, f64) {
        let s = self.0.size;
        (s[0], s[1], s[2])
    }

    /// Number of cells along x, y and z.
    #[getter]
    fn count(&self) -> (usize, usize, usize) {
        let c = self.0.count;
        (c[0], c[1], c[2])
    }

    /// Total number of cells.
    #[getter]
    fn n_cells(&self) -> usize {
        self.0.n_cells()
    }

    /// `(nz, ny, nx)`, so `values.reshape(grid.shape)` indexes `[k, j, i]`.
    #[getter]
    fn shape(&self) -> (usize, usize, usize) {
        let c = self.0.count;
        (c[2], c[1], c[0])
    }

    /// Cell centroids, shape ``(n_cells, 3)``, in cell order. Raises InvalidInput before
    /// allocating more than the available memory.
    #[getter]
    fn coords<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let n = self.0.n_cells();
        let need = (n as u64).saturating_mul(24);
        transiograms_core::memory::check(need, None, "Grid.coords").map_err(to_rust)?;
        let flat: Vec<f64> = (0..n).flat_map(|i| self.0.centroid(i)).collect();
        Ok(Array2::from_shape_vec((n, 3), flat)
            .unwrap()
            .into_pyarray(py))
    }

    /// Snap point data to cells.
    ///
    /// Points outside the grid or mask and missing codes are skipped; several points in a cell keep
    /// the one nearest to the centroid, ties to the first.
    ///
    /// Parameters
    /// ----------
    /// coords : array_like
    ///     ``(n, 3)`` point coordinates.
    /// codes : array_like
    ///     ``(n,)`` category codes, -1 or NaN for missing.
    /// mask : sequence of bool, optional
    ///     ``(n_cells,)`` active cells.
    ///
    /// Returns
    /// -------
    /// cells : numpy.ndarray
    ///     uint64 cell indices, ascending.
    /// codes : numpy.ndarray
    ///     int64 code of each snapped cell.
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When ``coords`` is not ``(n, 3)``, the lengths differ or ``mask`` has another length.
    #[pyo3(signature = (coords, codes=None, *, mask=None))]
    fn snap<'py>(
        &self,
        py: Python<'py>,
        coords: &Bound<'py, PyAny>,
        codes: Option<&Bound<'py, PyAny>>,
        mask: Option<Vec<bool>>,
    ) -> PyResult<(Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<i64>>)> {
        let (xyz, codes) = if crate::protocol::is_protocol(coords)? {
            let pts = crate::protocol::points_from_protocol(coords, codes, None, None, None)?;
            (pts.coords, pts.codes)
        } else {
            let Some(codes) = codes else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "snap() missing 1 required positional argument: 'codes'",
                ));
            };
            (
                self::coords(coords, "coords")?,
                self::codes(codes, "codes")?,
            )
        };
        let s = self
            .0
            .snap(&xyz, &codes, mask.as_deref())
            .map_err(to_rust)?;
        Ok((
            PyArray1::from_vec(py, s.cells.into_iter().map(|c| c as u64).collect()),
            PyArray1::from_vec(py, s.codes.into_iter().map(|c| c as i64).collect()),
        ))
    }

    /// JSON text of this object; ``Grid.from_json`` reads it back.
    ///
    /// Returns
    /// -------
    /// str
    fn to_json(&self) -> PyResult<String> {
        json(&self.0)
    }

    /// Grid from the text written by ``to_json``.
    ///
    /// Parameters
    /// ----------
    /// text : str
    ///
    /// Returns
    /// -------
    /// Grid
    ///
    /// Raises
    /// ------
    /// InvalidInput
    ///     When ``text`` is not Grid JSON or describes an invalid Grid.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        let g: Grid = unjson(text, "Grid")?;
        Grid::new(g.origin, g.size, g.count)
            .map(Self)
            .map_err(to_rust)
    }

    fn __repr__(&self) -> String {
        format!(
            "Grid(origin={:?}, size={:?}, count={:?})",
            self.0.origin, self.0.size, self.0.count
        )
    }

    /// Protocol v1 dictionary of this grid (``kind="grid"``).
    ///
    /// Returns
    /// -------
    /// dict
    fn __minetypes__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("protocol", 1)?;
        d.set_item("kind", "grid")?;
        d.set_item("origin", self.origin())?;
        d.set_item("size", self.size())?;
        d.set_item("count", self.count())?;
        d.set_item("rotation", (0.0, 0.0, 0.0))?;
        d.set_item("columns", PyDict::new(py))?;
        d.set_item("units", PyDict::new(py))?;
        d.set_item("categories", PyDict::new(py))?;
        d.set_item("crs", py.None())?;
        d.set_item("length_unit", py.None())?;
        Ok(d)
    }
}

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyCategories>()?;
    m.add_class::<PyGrid>()?;
    m.add_function(wrap_pyfunction!(_sequence_runs, m)?)?;
    m.add_function(wrap_pyfunction!(_sequences, m)?)?;
    Ok(())
}

/// Test hook for the shared sequence keywords: `(hole, rows, codes)` of every run.
#[pyfunction]
#[pyo3(signature = (codes, *, holes=None, from_=None, to=None, depth=None, step=None, gap=1e-6, categories=None, data=None, direction="up"))]
#[allow(clippy::too_many_arguments)]
fn _sequence_runs<'py>(
    codes: Bound<'py, PyAny>,
    holes: Option<Bound<'py, PyAny>>,
    from_: Option<Bound<'py, PyAny>>,
    to: Option<Bound<'py, PyAny>>,
    depth: Option<Bound<'py, PyAny>>,
    step: Option<f64>,
    gap: f64,
    categories: Option<Bound<'py, PyAny>>,
    data: Option<Bound<'py, PyAny>>,
    direction: &str,
) -> PyResult<Vec<(usize, Vec<usize>, Vec<u32>)>> {
    let dir = crate::args::direction(direction)?;
    let set = crate::args::sequence_set(crate::args::SequenceArgs {
        codes,
        holes,
        from_,
        to,
        depth,
        step,
        gap,
        categories,
        data,
    })?;
    Ok(set
        .runs(dir, gap)
        .into_iter()
        .map(|r| (r.hole, r.rows, r.codes))
        .collect())
}

/// Test hook for HMM sequence input: `(t, d, values, rows)` per sequence, and the row count.
#[pyfunction]
#[pyo3(signature = (sequences, *, holes=None, data=None, direction="up"))]
fn _sequences(
    sequences: &Bound<PyAny>,
    holes: Option<&Bound<PyAny>>,
    data: Option<&Bound<PyAny>>,
    direction: &str,
) -> PyResult<(Vec<(usize, usize, Vec<f64>, Vec<usize>)>, usize)> {
    let dir = crate::args::direction(direction)?;
    let (seqs, map) = crate::args::sequences(sequences, holes, data, dir, None)?;
    Ok((
        seqs.into_iter()
            .zip(map.rows)
            .map(|(s, r)| (s.t, s.d, s.x, r))
            .collect(),
        map.n_rows,
    ))
}
