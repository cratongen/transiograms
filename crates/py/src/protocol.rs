//! `__minetypes__` protocol v1 producer and consumer validation.

use std::collections::HashSet;

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyList, PyString, PyType};
use transiograms_core::{Categories, Code, Grid, MISSING};

use crate::args::{categories_k, codes, column, coords, to_rust};
use crate::core::{PyCategories, PyGrid};
use crate::invalid;

pub const ALL_KINDS: &[&str] = &[
    "table",
    "categories",
    "grid",
    "block_model",
    "point_set",
    "mesh",
    "polylines",
    "drillholes",
    "intervals",
    "composites",
    "seismic_volume",
    "prestack_volume",
];

pub const COMMON_KEYS: &[&str] = &[
    "protocol",
    "kind",
    "columns",
    "units",
    "categories",
    "crs",
    "length_unit",
];

/// Whether `obj` is a `__minetypes__` protocol object or a protocol dict.
pub fn is_protocol(obj: &Bound<PyAny>) -> PyResult<bool> {
    if obj.is_none() || obj.is_instance_of::<PyType>() {
        return Ok(false);
    }
    if obj.hasattr("__minetypes__")? {
        return Ok(true);
    }
    if let Ok(d) = obj.cast::<PyDict>() {
        if d.contains("protocol")? {
            return Ok(true);
        }
        if let Some(k) = d.get_item("kind")?
            && k.cast::<PyString>().is_ok()
        {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Calls `obj.__minetypes__()` when present, or casts `obj` to a `PyDict`.
pub fn protocol_dict<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    if !obj.is_instance_of::<PyType>() && obj.hasattr("__minetypes__")? {
        let res = obj.call_method0("__minetypes__")?;
        return res
            .cast_into::<PyDict>()
            .map_err(|_| invalid("__minetypes__() must return a dict"));
    }
    if let Ok(d) = obj.cast::<PyDict>() {
        return Ok(d.clone());
    }
    Err(invalid("expected a __minetypes__ protocol object or dict"))
}

/// Canonical CRS key of a declared CRS label (ECOSYSTEM.md "CRS and unit labels").
pub fn crs_key(label: &str) -> PyResult<String> {
    let t = label.trim();
    if t.is_empty() {
        return Err(invalid("crs must not be empty or whitespace"));
    }
    if let Some((auth, code)) = t.split_once(':') {
        let code = code.trim();
        if !auth.is_empty()
            && !code.is_empty()
            && auth.chars().all(|c| c.is_ascii_alphanumeric())
            && code.chars().all(|c| c.is_ascii_alphanumeric())
        {
            return Ok(format!("{}:{code}", auth.to_ascii_uppercase()));
        }
    }
    if let Some(epsg) = wkt_top_level_epsg(t) {
        return Ok(format!("EPSG:{epsg}"));
    }
    let mut out = String::new();
    let mut in_quotes = false;
    let mut in_ws = false;
    for ch in t.chars() {
        if ch == '"' {
            if in_ws {
                out.push(' ');
                in_ws = false;
            }
            in_quotes = !in_quotes;
            out.push(ch);
        } else if in_quotes {
            out.push(ch);
        } else if ch.is_whitespace() {
            in_ws = true;
        } else {
            if in_ws {
                out.push(' ');
                in_ws = false;
            }
            out.push(ch.to_ascii_uppercase());
        }
    }
    Ok(out)
}

fn wkt_top_level_epsg(t: &str) -> Option<String> {
    let open = t.find(['[', '('])?;
    let prefix = t[..open].trim();
    if prefix.is_empty() || !prefix.chars().all(|c| c.is_ascii_alphabetic() || c == '_') {
        return None;
    }
    if !t.ends_with([']', ')']) {
        return None;
    }
    let inner = t[open + 1..t.len() - 1].trim();
    let mut depth = 0i32;
    let mut in_quotes = false;
    let mut last_comma = None;
    for (idx, ch) in inner.char_indices() {
        if ch == '"' {
            in_quotes = !in_quotes;
        } else if !in_quotes {
            match ch {
                '[' | '(' => depth += 1,
                ']' | ')' => depth -= 1,
                ',' if depth == 0 => last_comma = Some(idx),
                _ => {}
            }
        }
    }
    if depth != 0 || in_quotes {
        return None;
    }
    let last_item = match last_comma {
        Some(c) => inner[c + 1..].trim(),
        None => inner,
    };
    let item_open = last_item.find(['[', '('])?;
    if !last_item.ends_with([']', ')']) {
        return None;
    }
    let kw = last_item[..item_open].trim();
    if !kw.eq_ignore_ascii_case("ID") && !kw.eq_ignore_ascii_case("AUTHORITY") {
        return None;
    }
    let args = last_item[item_open + 1..last_item.len() - 1].trim();
    let (auth, code) = args.split_once(',')?;
    let auth = auth.trim().trim_matches('"').trim();
    let code = code.trim().trim_matches('"').trim();
    if auth.eq_ignore_ascii_case("EPSG")
        && !code.is_empty()
        && code.chars().all(|c| c.is_ascii_alphanumeric())
    {
        Some(code.to_string())
    } else {
        None
    }
}

/// Combines two optional CRS labels, raising `InvalidInput` on mismatch.
pub fn combine_crs(a: Option<&str>, b: Option<&str>) -> PyResult<Option<String>> {
    match (a, b) {
        (None, None) => Ok(None),
        (Some(a), None) => {
            crs_key(a)?;
            Ok(Some(a.to_string()))
        }
        (None, Some(b)) => {
            crs_key(b)?;
            Ok(Some(b.to_string()))
        }
        (Some(a), Some(b)) => {
            let ka = crs_key(a)?;
            let kb = crs_key(b)?;
            if ka != kb {
                return Err(invalid(format!("crs mismatch: {a:?} vs {b:?}")));
            }
            Ok(Some(a.to_string()))
        }
    }
}

/// Combines two optional length_unit labels, raising `InvalidInput` on mismatch.
pub fn combine_length_unit(a: Option<&str>, b: Option<&str>) -> PyResult<Option<String>> {
    let check = |u: &str| -> PyResult<String> {
        let s = u.trim();
        if s.is_empty() {
            return Err(invalid("length_unit must not be empty or whitespace"));
        }
        Ok(s.to_string())
    };
    match (a, b) {
        (None, None) => Ok(None),
        (Some(a), None) => Ok(Some(check(a)?)),
        (None, Some(b)) => Ok(Some(check(b)?)),
        (Some(a), Some(b)) => {
            let sa = check(a)?;
            let sb = check(b)?;
            if sa != sb {
                return Err(invalid(format!("length_unit mismatch: {a:?} vs {b:?}")));
            }
            Ok(Some(sa))
        }
    }
}

/// Validates the 7 common keys and returns `(kind, crs, length_unit)`.
pub fn validate_common(
    d: &Bound<PyDict>,
    expected_kinds: &[&str],
) -> PyResult<(String, Option<String>, Option<String>)> {
    for &k in COMMON_KEYS {
        if !d.contains(k)? {
            return Err(invalid(format!("missing protocol key {k:?}")));
        }
    }
    let p = d.get_item("protocol")?.unwrap();
    if p.is_instance_of::<PyBool>() || p.extract::<i64>().ok() != Some(1) {
        return Err(invalid(format!("protocol must be 1, got {}", p.repr()?)));
    }
    let k_obj = d.get_item("kind")?.unwrap();
    let Ok(k_str) = k_obj.cast::<PyString>() else {
        return Err(invalid("kind must be a string"));
    };
    let kind = k_str.to_str()?.to_string();
    if !ALL_KINDS.contains(&kind.as_str()) {
        return Err(invalid(format!("unknown kind {kind:?}")));
    }
    if !expected_kinds.contains(&kind.as_str()) {
        return Err(invalid(format!(
            "kind {kind:?} is not accepted here; expected {}",
            expected_kinds.join(", ")
        )));
    }
    let crs_obj = d.get_item("crs")?.unwrap();
    let crs = if crs_obj.is_none() {
        None
    } else {
        let Ok(s) = crs_obj.cast::<PyString>() else {
            return Err(invalid("crs must be a string or None"));
        };
        let s = s.to_str()?;
        crs_key(s)?;
        Some(s.to_string())
    };
    let lu_obj = d.get_item("length_unit")?.unwrap();
    let length_unit = if lu_obj.is_none() {
        None
    } else {
        let Ok(s) = lu_obj.cast::<PyString>() else {
            return Err(invalid("length_unit must be a string or None"));
        };
        let s = s.to_str()?;
        if s.trim().is_empty() {
            return Err(invalid("length_unit must not be empty or whitespace"));
        }
        Some(s.trim().to_string())
    };
    for k in ["columns", "units", "categories"] {
        if d.get_item(k)?.unwrap().cast::<PyDict>().is_err() {
            return Err(invalid(format!("{k} must be a dict")));
        }
    }
    Ok((kind, crs, length_unit))
}

fn string_seq(obj: &Bound<PyAny>, key: &str) -> PyResult<Vec<String>> {
    if obj.cast::<PyString>().is_ok()
        || obj.cast::<PyBytes>().is_ok()
        || obj.cast::<PyDict>().is_ok()
    {
        return Err(invalid(format!("{key} must be a sequence of strings")));
    }
    let items: Vec<Bound<PyAny>> = obj
        .try_iter()
        .map_err(|_| invalid(format!("{key} must be a sequence of strings")))?
        .collect::<PyResult<_>>()?;
    let mut out = Vec::with_capacity(items.len());
    for (i, item) in items.into_iter().enumerate() {
        let Ok(s) = item.cast::<PyString>() else {
            return Err(invalid(format!("{key}[{i}] must be a string")));
        };
        let s = s.to_str()?.to_string();
        if s.is_empty() {
            return Err(invalid(format!("{key}[{i}] must not be empty")));
        }
        out.push(s);
    }
    Ok(out)
}

/// Validates a category entry (`names`, `colors`, `mapping`, `other`) and returns `Categories`.
pub fn validate_category_entry(d: &Bound<PyDict>, ctx: &str) -> PyResult<Categories> {
    for k in ["names", "colors", "mapping", "other"] {
        if !d.contains(k)? {
            return Err(invalid(format!("{ctx}: missing protocol key {k:?}")));
        }
    }
    let names = string_seq(&d.get_item("names")?.unwrap(), &format!("{ctx} \"names\""))?;
    if names.is_empty() {
        return Err(invalid(format!("{ctx} \"names\" must not be empty")));
    }
    let mut seen = HashSet::with_capacity(names.len());
    for n in &names {
        if !seen.insert(n.as_str()) {
            return Err(invalid(format!("{ctx} \"names\" contains duplicate {n:?}")));
        }
    }
    let colors_obj = d.get_item("colors")?.unwrap();
    let colors = if colors_obj.is_none() {
        None
    } else {
        let c = string_seq(&colors_obj, &format!("{ctx} \"colors\""))?;
        if c.len() != names.len() {
            return Err(invalid(format!(
                "{ctx} \"colors\" has {} entries for {} names",
                c.len(),
                names.len()
            )));
        }
        Some(c)
    };
    let mapping_obj = d.get_item("mapping")?.unwrap();
    let Ok(mapping) = mapping_obj.cast::<PyDict>() else {
        return Err(invalid(format!("{ctx} \"mapping\" must be a dict")));
    };
    for (mk, mv) in mapping.iter() {
        let (Ok(ks), Ok(vs)) = (mk.cast::<PyString>(), mv.cast::<PyString>()) else {
            return Err(invalid(format!(
                "{ctx} \"mapping\" keys and values must be strings"
            )));
        };
        let (k, v) = (ks.to_str()?, vs.to_str()?);
        if k.is_empty() || seen.contains(k) {
            return Err(invalid(format!(
                "{ctx} \"mapping\" key {k:?} must be non-empty and not in \"names\""
            )));
        }
        if !seen.contains(v) {
            return Err(invalid(format!(
                "{ctx} \"mapping\" value {v:?} is not in \"names\""
            )));
        }
    }
    let other_obj = d.get_item("other")?.unwrap();
    if !other_obj.is_none() {
        let Ok(os) = other_obj.cast::<PyString>() else {
            return Err(invalid(format!("{ctx} \"other\" must be a string or None")));
        };
        let o = os.to_str()?;
        if Some(o) != names.last().map(String::as_str) {
            return Err(invalid(format!(
                "{ctx} \"other\" {o:?} must equal the last name {:?}",
                names.last().unwrap()
            )));
        }
    }
    Categories::new(names, colors).map_err(to_rust)
}

/// Builds a category entry dict `{names, colors, mapping, other}`.
pub fn category_entry_dict<'py>(
    py: Python<'py>,
    names: &[String],
    colors: Option<&[String]>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("names", PyList::new(py, names)?)?;
    match colors {
        Some(c) => d.set_item("colors", PyList::new(py, c)?)?,
        None => d.set_item("colors", py.None())?,
    }
    d.set_item("mapping", PyDict::new(py))?;
    d.set_item("other", py.None())?;
    Ok(d)
}

/// Validates a `categories` protocol object or dict and returns `Categories`.
pub fn categories_from_protocol(obj: &Bound<PyAny>) -> PyResult<Categories> {
    let d = protocol_dict(obj)?;
    validate_common(&d, &["categories"])?;
    for k in ["columns", "units", "categories"] {
        let sub = d.get_item(k)?.unwrap().cast_into::<PyDict>()?;
        if !sub.is_empty() {
            return Err(invalid(format!("kind \"categories\" must have empty {k}")));
        }
    }
    validate_category_entry(&d, "categories")
}

/// Checks that `arr` is a C-contiguous, native-endian `numpy.ndarray`, returning `(shape, dtype_name, dtype_kind)`.
pub fn check_array(
    arr: &Bound<PyAny>,
    key: &str,
    expected_ndim: Option<usize>,
) -> PyResult<(Vec<usize>, String, String)> {
    let py = arr.py();
    let np = py.import("numpy")?;
    let ndarray_ty = np.getattr("ndarray")?;
    if !arr.is_instance(&ndarray_ty)? {
        return Err(invalid(format!("{key} must be a numpy.ndarray")));
    }
    let flags = arr.getattr("flags")?;
    let c_contig: bool = flags.getattr("c_contiguous")?.extract()?;
    if !c_contig {
        return Err(invalid(format!("{key} must be C-contiguous")));
    }
    let dtype = arr.getattr("dtype")?;
    let byteorder: String = dtype.getattr("byteorder")?.extract()?;
    if byteorder != "=" && byteorder != "|" {
        return Err(invalid(format!("{key} must be native-endian")));
    }
    let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    if let Some(ndim) = expected_ndim
        && shape.len() != ndim
    {
        return Err(invalid(format!(
            "{key} must be {ndim}-D, got shape {shape:?}"
        )));
    }
    let name: String = dtype.getattr("name")?.extract()?;
    let kind: String = dtype.getattr("kind")?.extract()?;
    Ok((shape, name, kind))
}

fn check_int32_codes(arr: &Bound<PyAny>, k: usize, key: &str) -> PyResult<Vec<i32>> {
    let a: PyReadonlyArray1<i32> = arr
        .extract()
        .map_err(|_| invalid(format!("{key} must be a 1-D int32 array")))?;
    let slice = a.as_slice()?;
    for (row, &v) in slice.iter().enumerate() {
        if v < -1 || v >= k as i32 {
            return Err(invalid(format!(
                "{key} code {v} at row {row} is outside -1..{}",
                k.saturating_sub(1)
            )));
        }
    }
    Ok(slice.to_vec())
}

fn check_coords_array(
    arr: &Bound<PyAny>,
    key: &str,
    expected_rows: Option<usize>,
) -> PyResult<Vec<[f64; 3]>> {
    let (shape, dtype_name, _) = check_array(arr, key, Some(2))?;
    if dtype_name != "float64" {
        return Err(invalid(format!(
            "{key} must have dtype float64, got {dtype_name}"
        )));
    }
    if shape[1] != 3 || expected_rows.is_some_and(|r| shape[0] != r) {
        return Err(invalid(format!(
            "{key} has shape {shape:?}, expected ({}, 3)",
            expected_rows.map_or_else(|| "N".to_string(), |r| r.to_string())
        )));
    }
    let a: PyReadonlyArray2<f64> = arr.extract()?;
    let slice = a.as_slice()?;
    if slice.iter().any(|v| !v.is_finite()) {
        return Err(invalid(format!("{key} must contain finite coordinates")));
    }
    Ok((0..shape[0])
        .map(|i| [slice[3 * i], slice[3 * i + 1], slice[3 * i + 2]])
        .collect())
}

fn validate_table_fields(
    d: &Bound<PyDict>,
    n_rows: usize,
    extra_cat_cols: &[(&str, Bound<PyAny>)],
) -> PyResult<()> {
    let py = d.py();
    let np = py.import("numpy")?;
    let columns = d
        .get_item("columns")?
        .ok_or_else(|| invalid("missing protocol key \"columns\""))?
        .cast_into::<PyDict>()
        .map_err(|_| invalid("columns must be a dict"))?;
    let units = d
        .get_item("units")?
        .ok_or_else(|| invalid("missing protocol key \"units\""))?
        .cast_into::<PyDict>()
        .map_err(|_| invalid("units must be a dict"))?;
    let categories = d
        .get_item("categories")?
        .ok_or_else(|| invalid("missing protocol key \"categories\""))?
        .cast_into::<PyDict>()
        .map_err(|_| invalid("categories must be a dict"))?;

    let mut cat_map = std::collections::HashMap::new();
    for (ck, cv) in categories.iter() {
        let Ok(cs) = ck.cast::<PyString>() else {
            return Err(invalid("categories key must be a string"));
        };
        let cname = cs.to_str()?.to_string();
        if !columns.contains(&ck)? && !extra_cat_cols.iter().any(|(n, _)| *n == cname) {
            return Err(invalid(format!(
                "categories key {cname:?} is not in columns"
            )));
        }
        let Ok(cdict) = cv.cast::<PyDict>() else {
            return Err(invalid(format!("categories[{cname:?}] must be a dict")));
        };
        let cat = validate_category_entry(cdict, &format!("categories[{cname:?}]"))?;
        cat_map.insert(cname, cat);
    }

    for (uk, uv) in units.iter() {
        let Ok(us) = uk.cast::<PyString>() else {
            return Err(invalid("units key must be a string"));
        };
        let uname = us.to_str()?;
        if !columns.contains(&uk)? {
            return Err(invalid(format!("units key {uname:?} is not in columns")));
        }
        let Ok(uval) = uv.cast::<PyString>() else {
            return Err(invalid(format!("units[{uname:?}] must be a string")));
        };
        if uval.to_str()?.trim().is_empty() {
            return Err(invalid(format!("units[{uname:?}] must not be empty")));
        }
    }

    for (col_k, col_v) in columns.iter() {
        let Ok(col_s) = col_k.cast::<PyString>() else {
            return Err(invalid("columns key must be a string"));
        };
        let col_name = col_s.to_str()?;
        if col_name.is_empty() {
            return Err(invalid("columns key must not be empty"));
        }
        let key_label = format!("columns[{col_name:?}]");
        let (shape, dtype_name, dtype_kind) = check_array(&col_v, &key_label, Some(1))?;
        if shape[0] != n_rows {
            return Err(invalid(format!(
                "{key_label} has length {}, expected {n_rows}",
                shape[0]
            )));
        }
        if let Some(cat) = cat_map.get(col_name) {
            if dtype_name != "int32" {
                return Err(invalid(format!(
                    "{key_label} is in categories and must have dtype int32, got {dtype_name}"
                )));
            }
            check_int32_codes(&col_v, cat.len(), &key_label)?;
        } else {
            let allowed = matches!(
                dtype_name.as_str(),
                "float64" | "float32" | "int64" | "int32" | "bool"
            ) || dtype_kind == "U"
                || dtype_kind == "T";
            if !allowed {
                return Err(invalid(format!(
                    "{key_label} has unsupported dtype {dtype_name}"
                )));
            }
            if matches!(dtype_name.as_str(), "float64" | "float32") {
                let has_inf: bool = np
                    .call_method1("isinf", (&col_v,))?
                    .call_method0("any")?
                    .extract()?;
                if has_inf {
                    return Err(invalid(format!("{key_label} must not contain +/- Inf")));
                }
            }
        }
    }

    for (ename, earr) in extra_cat_cols {
        let cat = cat_map.get(*ename).ok_or_else(|| {
            invalid(format!(
                "{ename} is set and must have an entry in categories"
            ))
        })?;
        check_int32_codes(earr, cat.len(), ename)?;
    }

    Ok(())
}

fn triple_f64(obj: &Bound<PyAny>, key: &str) -> PyResult<[f64; 3]> {
    if obj.cast::<PyString>().is_ok() || obj.cast::<PyBytes>().is_ok() {
        return Err(invalid(format!("{key} must be 3 finite floats")));
    }
    let items: Vec<Bound<PyAny>> = obj
        .try_iter()
        .map_err(|_| invalid(format!("{key} must be 3 finite floats")))?
        .collect::<PyResult<_>>()?;
    if items.len() != 3 {
        return Err(invalid(format!(
            "{key} must have length 3, got {}",
            items.len()
        )));
    }
    let mut out = [0.0; 3];
    for (i, item) in items.into_iter().enumerate() {
        if item.is_instance_of::<PyBool>() {
            return Err(invalid(format!("{key}[{i}] must be a float, not bool")));
        }
        let v: f64 = item
            .extract()
            .map_err(|_| invalid(format!("{key}[{i}] must be a float")))?;
        if !v.is_finite() {
            return Err(invalid(format!("{key}[{i}] must be finite, got {v}")));
        }
        out[i] = v;
    }
    Ok(out)
}

fn triple_usize(obj: &Bound<PyAny>, key: &str) -> PyResult<[usize; 3]> {
    if obj.cast::<PyString>().is_ok() || obj.cast::<PyBytes>().is_ok() {
        return Err(invalid(format!("{key} must be 3 positive integers")));
    }
    let items: Vec<Bound<PyAny>> = obj
        .try_iter()
        .map_err(|_| invalid(format!("{key} must be 3 positive integers")))?
        .collect::<PyResult<_>>()?;
    if items.len() != 3 {
        return Err(invalid(format!(
            "{key} must have length 3, got {}",
            items.len()
        )));
    }
    let mut out = [0usize; 3];
    for (i, item) in items.into_iter().enumerate() {
        if item.is_instance_of::<PyBool>() {
            return Err(invalid(format!("{key}[{i}] must be an integer, not bool")));
        }
        let v: i64 = item
            .extract()
            .map_err(|_| invalid(format!("{key}[{i}] must be an integer")))?;
        if v < 1 {
            return Err(invalid(format!("{key}[{i}] must be >= 1, got {v}")));
        }
        out[i] = v as usize;
    }
    if out
        .iter()
        .try_fold(1i64, |a, &c| a.checked_mul(c as i64))
        .is_none()
    {
        return Err(invalid(format!("{key} product exceeds 2^63 - 1")));
    }
    Ok(out)
}

fn validate_grid_geometry(d: &Bound<PyDict>, kind: &str) -> PyResult<(Grid, [f64; 3])> {
    if d.contains("coords")? {
        return Err(invalid(format!("{kind} must not carry \"coords\"")));
    }
    for k in ["origin", "size", "count", "rotation"] {
        if !d.contains(k)? {
            return Err(invalid(format!("missing protocol key {k:?}")));
        }
    }
    let origin = triple_f64(&d.get_item("origin")?.unwrap(), "origin")?;
    let size = triple_f64(&d.get_item("size")?.unwrap(), "size")?;
    for (i, &s) in size.iter().enumerate() {
        if s <= 0.0 {
            return Err(invalid(format!("size[{i}] must be > 0, got {s}")));
        }
    }
    let count = triple_usize(&d.get_item("count")?.unwrap(), "count")?;
    let rotation = triple_f64(&d.get_item("rotation")?.unwrap(), "rotation")?;
    let grid = Grid::new(origin, size, count).map_err(to_rust)?;
    Ok((grid, rotation))
}

/// Validates a `grid` or `block_model` dict and returns `(grid, rotation, layout, masked_index, n_rows)`.
pub fn validate_grid_or_block_model(
    d: &Bound<PyDict>,
    kind: &str,
) -> PyResult<(Grid, [f64; 3], String, Option<Vec<u64>>, usize)> {
    let (grid, rotation) = validate_grid_geometry(d, kind)?;
    if kind == "grid" {
        let n_rows = grid.n_cells();
        validate_table_fields(d, n_rows, &[])?;
        return Ok((grid, rotation, "regular".to_string(), None, n_rows));
    }
    for k in ["layout", "index", "parent", "offset", "scale", "subgrid"] {
        if !d.contains(k)? {
            return Err(invalid(format!("missing protocol key {k:?}")));
        }
    }
    let layout_obj = d.get_item("layout")?.unwrap();
    let Ok(layout_s) = layout_obj.cast::<PyString>() else {
        return Err(invalid("layout must be a string"));
    };
    let layout = layout_s.to_str()?.to_string();
    let n_parent = grid.n_cells() as u64;
    match layout.as_str() {
        "regular" => {
            for k in ["index", "parent", "offset", "scale", "subgrid"] {
                if !d.get_item(k)?.unwrap().is_none() {
                    return Err(invalid(format!(
                        "block_model with layout=\"regular\" must have {k}=None"
                    )));
                }
            }
            let n_rows = grid.n_cells();
            validate_table_fields(d, n_rows, &[])?;
            Ok((grid, rotation, layout, None, n_rows))
        }
        "masked" => {
            for k in ["parent", "offset", "scale", "subgrid"] {
                if !d.get_item(k)?.unwrap().is_none() {
                    return Err(invalid(format!(
                        "block_model with layout=\"masked\" must have {k}=None"
                    )));
                }
            }
            let idx_obj = d.get_item("index")?.unwrap();
            let (shape, dtype_name, _) = check_array(&idx_obj, "index", Some(1))?;
            if dtype_name != "uint64" {
                return Err(invalid(format!(
                    "index must have dtype uint64, got {dtype_name}"
                )));
            }
            let idx_arr: PyReadonlyArray1<u64> = idx_obj.extract()?;
            let idx = idx_arr.as_slice()?;
            for w in idx.windows(2) {
                if w[1] <= w[0] {
                    return Err(invalid("index must be strictly increasing"));
                }
            }
            if let Some(&last) = idx.last()
                && last >= n_parent
            {
                return Err(invalid(format!(
                    "index value {last} is out of bounds for {n_parent} parent cells"
                )));
            }
            let n_rows = shape[0];
            validate_table_fields(d, n_rows, &[])?;
            Ok((grid, rotation, layout, Some(idx.to_vec()), n_rows))
        }
        "subblocked" => {
            if !d.get_item("index")?.unwrap().is_none() {
                return Err(invalid(
                    "block_model with layout=\"subblocked\" must have index=None",
                ));
            }
            let parent_obj = d.get_item("parent")?.unwrap();
            let (p_shape, p_dtype, _) = check_array(&parent_obj, "parent", Some(1))?;
            if p_dtype != "uint64" {
                return Err(invalid(format!(
                    "parent must have dtype uint64, got {p_dtype}"
                )));
            }
            let p_arr: PyReadonlyArray1<u64> = parent_obj.extract()?;
            let parent = p_arr.as_slice()?;
            for w in parent.windows(2) {
                if w[1] < w[0] {
                    return Err(invalid("parent must be non-decreasing"));
                }
            }
            if let Some(&last) = parent.last()
                && last >= n_parent
            {
                return Err(invalid(format!(
                    "parent value {last} is out of bounds for {n_parent} parent cells"
                )));
            }
            let n_rows = p_shape[0];
            let off_obj = d.get_item("offset")?.unwrap();
            let (o_shape, o_dtype, _) = check_array(&off_obj, "offset", Some(2))?;
            if o_dtype != "float32" || o_shape != [n_rows, 3] {
                return Err(invalid(format!(
                    "offset must be float32 of shape ({n_rows}, 3), got {o_dtype} {o_shape:?}"
                )));
            }
            let sc_obj = d.get_item("scale")?.unwrap();
            let (s_shape, s_dtype, _) = check_array(&sc_obj, "scale", Some(2))?;
            if s_dtype != "float32" || s_shape != [n_rows, 3] {
                return Err(invalid(format!(
                    "scale must be float32 of shape ({n_rows}, 3), got {s_dtype} {s_shape:?}"
                )));
            }
            let off_arr: PyReadonlyArray2<f32> = off_obj.extract()?;
            let sc_arr: PyReadonlyArray2<f32> = sc_obj.extract()?;
            let off = off_arr.as_slice()?;
            let sc = sc_arr.as_slice()?;
            for i in 0..off.len() {
                let (c, s) = (off[i], sc[i]);
                if !(c.is_finite() && c > 0.0 && c < 1.0) {
                    return Err(invalid(format!(
                        "offset must be finite and in (0, 1), got {c}"
                    )));
                }
                if !(s.is_finite() && s > 0.0 && s <= 1.0) {
                    return Err(invalid(format!(
                        "scale must be finite and in (0, 1], got {s}"
                    )));
                }
                if c - 0.5 * s < -1e-6 || c + 0.5 * s > 1.0 + 1e-6 {
                    return Err(invalid(
                        "offset and scale extend outside the parent cell bounds",
                    ));
                }
            }
            let sg_obj = d.get_item("subgrid")?.unwrap();
            if !sg_obj.is_none() {
                let sg = triple_usize(&sg_obj, "subgrid")?;
                for a in 0..3 {
                    if sg[a] > u32::MAX as usize {
                        return Err(invalid(format!("subgrid[{a}] must be <= 2^32 - 1")));
                    }
                    let m = sg[a] as f64;
                    for r in 0..n_rows {
                        let (c, s) = (off[3 * r + a] as f64, sc[3 * r + a] as f64);
                        for edge in [(c - 0.5 * s) * m, (c + 0.5 * s) * m] {
                            let nearest = edge.round();
                            if (edge - nearest).abs() > 1e-5 || nearest < 0.0 || nearest > m {
                                return Err(invalid(
                                    "subgrid does not align with offset and scale",
                                ));
                            }
                        }
                    }
                }
            }
            validate_table_fields(d, n_rows, &[])?;
            Ok((grid, rotation, layout, None, n_rows))
        }
        _ => Err(invalid(format!(
            "layout must be \"regular\", \"masked\" or \"subblocked\", got {layout:?}"
        ))),
    }
}

/// Extracts `(Grid, mask, crs, length_unit)` from a `tg.Grid`, `grid`, or `block_model` object.
pub fn resolve_grid_target<'py>(
    obj: &Bound<'py, PyAny>,
) -> PyResult<(
    Grid,
    Option<Vec<bool>>,
    Option<String>,
    Option<String>,
    Option<Bound<'py, PyDict>>,
    Option<Vec<u64>>,
)> {
    if let Ok(g) = obj.cast::<PyGrid>() {
        return Ok((g.get().0.clone(), None, None, None, None, None));
    }
    if is_protocol(obj)? {
        let d = protocol_dict(obj)?;
        let (kind, crs, length_unit) = validate_common(&d, &["grid", "block_model"])?;
        let (grid, rotation, layout, index, _) = validate_grid_or_block_model(&d, &kind)?;
        if rotation.iter().any(|&r| r != 0.0) {
            return Err(invalid(format!(
                "rotation {rotation:?} is not (0, 0, 0); transiograms Grid is axis-aligned"
            )));
        }
        if layout == "subblocked" {
            return Err(invalid(
                "layout \"subblocked\" cannot be used as a regular simulation grid target",
            ));
        }
        let mask = index.as_ref().map(|idx| {
            let mut m = vec![false; grid.n_cells()];
            for &i in idx {
                m[i as usize] = true;
            }
            m
        });
        return Ok((grid, mask, crs, length_unit, Some(d), index));
    }
    Err(invalid(
        "grid must be a tg.Grid or a \"grid\" or \"block_model\" protocol object",
    ))
}

/// Combines an optional protocol mask with an explicit user `mask=` argument.
pub fn merge_masks(
    proto_mask: Option<Vec<bool>>,
    user_mask: Option<&Bound<PyAny>>,
    n_cells: usize,
) -> PyResult<Option<Vec<bool>>> {
    let user = crate::args::mask(user_mask, n_cells)?;
    Ok(match (proto_mask, user) {
        (None, None) => None,
        (Some(m), None) | (None, Some(m)) => Some(m),
        (Some(a), Some(b)) => Some(a.into_iter().zip(b).map(|(x, y)| x && y).collect()),
    })
}

/// Validates a `holes` sequence: `H >= 0` distinct, non-empty strings.
pub fn validate_holes(obj: &Bound<PyAny>, key: &str) -> PyResult<Vec<String>> {
    if obj.cast::<PyString>().is_ok()
        || obj.cast::<PyBytes>().is_ok()
        || obj.cast::<PyDict>().is_ok()
    {
        return Err(invalid(format!("{key} must be a sequence of hole ids")));
    }
    let items: Vec<Bound<PyAny>> = obj
        .try_iter()
        .map_err(|_| invalid(format!("{key} must be a sequence of hole ids")))?
        .collect::<PyResult<_>>()?;
    let mut holes = Vec::with_capacity(items.len());
    let mut seen = HashSet::with_capacity(items.len());
    for (i, item) in items.into_iter().enumerate() {
        let Ok(s) = item.cast::<PyString>() else {
            return Err(invalid(format!("{key}[{i}] must be a string")));
        };
        let s = s.to_str()?.to_string();
        if s.is_empty() {
            return Err(invalid(format!("{key}[{i}] must not be empty")));
        }
        if !seen.insert(s.clone()) {
            return Err(invalid(format!("{key} contains duplicate hole id {s:?}")));
        }
        holes.push(s);
    }
    Ok(holes)
}

fn validate_hole_from_to(
    d: &Bound<PyDict>,
    holes: &[String],
    require_sorted: bool,
) -> PyResult<(Vec<i32>, Vec<f64>, Vec<f64>)> {
    for k in ["hole_index", "from_depth", "to_depth"] {
        if !d.contains(k)? {
            return Err(invalid(format!("missing protocol key {k:?}")));
        }
    }
    let hi_obj = d.get_item("hole_index")?.unwrap();
    let (hi_shape, hi_dtype, _) = check_array(&hi_obj, "hole_index", Some(1))?;
    if hi_dtype != "int32" {
        return Err(invalid(format!(
            "hole_index must have dtype int32, got {hi_dtype}"
        )));
    }
    let n_rows = hi_shape[0];
    let hi_arr: PyReadonlyArray1<i32> = hi_obj.extract()?;
    let hi = hi_arr.as_slice()?.to_vec();
    let h_len = holes.len() as i32;
    for (r, &h) in hi.iter().enumerate() {
        if h < 0 || h >= h_len {
            return Err(invalid(format!(
                "hole_index[{r}] = {h} is out of range for {h_len} holes"
            )));
        }
    }
    let fd_obj = d.get_item("from_depth")?.unwrap();
    let (fd_shape, fd_dtype, _) = check_array(&fd_obj, "from_depth", Some(1))?;
    if fd_dtype != "float64" || fd_shape != [n_rows] {
        return Err(invalid(format!(
            "from_depth must be float64 of shape ({n_rows},), got {fd_dtype} {fd_shape:?}"
        )));
    }
    let td_obj = d.get_item("to_depth")?.unwrap();
    let (td_shape, td_dtype, _) = check_array(&td_obj, "to_depth", Some(1))?;
    if td_dtype != "float64" || td_shape != [n_rows] {
        return Err(invalid(format!(
            "to_depth must be float64 of shape ({n_rows},), got {td_dtype} {td_shape:?}"
        )));
    }
    let fd_arr: PyReadonlyArray1<f64> = fd_obj.extract()?;
    let td_arr: PyReadonlyArray1<f64> = td_obj.extract()?;
    let fd = fd_arr.as_slice()?.to_vec();
    let td = td_arr.as_slice()?.to_vec();
    for r in 0..n_rows {
        if !(fd[r].is_finite() && fd[r] >= 0.0) {
            return Err(invalid(format!(
                "from_depth[{r}] = {} must be finite and >= 0",
                fd[r]
            )));
        }
        if !(td[r].is_finite() && td[r] > fd[r]) {
            return Err(invalid(format!(
                "to_depth[{r}] = {} must be finite and > from_depth[{r}] = {}",
                td[r], fd[r]
            )));
        }
        if require_sorted && r > 0 {
            if hi[r] < hi[r - 1] {
                return Err(invalid("hole_index must be sorted non-decreasing"));
            }
            if hi[r] == hi[r - 1] && fd[r] < td[r - 1] - 1e-9 {
                return Err(invalid(format!(
                    "from_depth[{r}] = {} overlaps previous to_depth = {} in hole {:?}",
                    fd[r],
                    td[r - 1],
                    holes[hi[r] as usize]
                )));
            }
        }
    }
    Ok((hi, fd, td))
}

/// Validates an `intervals` protocol dict (top-level or nested inside `drillholes`).
pub fn validate_interval_dict(
    d: &Bound<PyDict>,
    parent_holes: Option<&[String]>,
) -> PyResult<(Vec<String>, Vec<i32>, Vec<f64>, Vec<f64>)> {
    let holes = match parent_holes {
        Some(h) => h.to_vec(),
        None => {
            if !d.contains("holes")? {
                return Err(invalid("missing protocol key \"holes\""));
            }
            validate_holes(&d.get_item("holes")?.unwrap(), "holes")?
        }
    };
    let (hi, fd, td) = validate_hole_from_to(d, &holes, true)?;
    validate_table_fields(d, hi.len(), &[])?;
    Ok((holes, hi, fd, td))
}

/// Validates a `composites` protocol dict.
pub fn validate_composites_dict(
    d: &Bound<PyDict>,
) -> PyResult<(Vec<String>, Vec<[f64; 3]>, Vec<i32>, Vec<f64>, Vec<f64>)> {
    for k in [
        "holes",
        "coords",
        "hole_index",
        "from_depth",
        "to_depth",
        "length",
        "domain",
    ] {
        if !d.contains(k)? {
            return Err(invalid(format!("missing protocol key {k:?}")));
        }
    }
    let holes = validate_holes(&d.get_item("holes")?.unwrap(), "holes")?;
    let (hi, fd, td) = validate_hole_from_to(d, &holes, true)?;
    let n_rows = hi.len();
    let xyz = check_coords_array(&d.get_item("coords")?.unwrap(), "coords", Some(n_rows))?;
    let len_obj = d.get_item("length")?.unwrap();
    let (l_shape, l_dtype, _) = check_array(&len_obj, "length", Some(1))?;
    if l_dtype != "float64" || l_shape != [n_rows] {
        return Err(invalid(format!(
            "length must be float64 of shape ({n_rows},), got {l_dtype} {l_shape:?}"
        )));
    }
    let len_arr: PyReadonlyArray1<f64> = len_obj.extract()?;
    let lengths = len_arr.as_slice()?;
    for r in 0..n_rows {
        let span = td[r] - fd[r];
        if !(lengths[r].is_finite() && lengths[r] > 0.0 && lengths[r] <= span + 1e-9) {
            return Err(invalid(format!(
                "length[{r}] = {} must lie in (0, to_depth - from_depth]",
                lengths[r]
            )));
        }
    }
    let dom_obj = d.get_item("domain")?.unwrap();
    let mut extra = Vec::new();
    if !dom_obj.is_none() {
        let (dom_shape, dom_dtype, _) = check_array(&dom_obj, "domain", Some(1))?;
        if dom_dtype != "int32" || dom_shape != [n_rows] {
            return Err(invalid(format!(
                "domain must be int32 of shape ({n_rows},), got {dom_dtype} {dom_shape:?}"
            )));
        }
        extra.push(("domain", dom_obj));
    }
    validate_table_fields(d, n_rows, &extra)?;

    // Check grade_length rule on float64 grade columns in composites.
    let columns = d.get_item("columns")?.unwrap().cast_into::<PyDict>()?;
    let categories = d.get_item("categories")?.unwrap().cast_into::<PyDict>()?;
    for (ck, cv) in columns.iter() {
        let cname = ck.cast::<PyString>()?.to_str()?;
        let (_, dt, _) = check_array(&cv, &format!("columns[{cname:?}]"), Some(1))?;
        if dt == "float64" && !cname.ends_with("_length") && !categories.contains(&ck)? {
            let gl_name = format!("{cname}_length");
            if !columns.contains(&gl_name)? {
                return Err(invalid(format!(
                    "composites grade column {cname:?} requires {gl_name:?} in columns"
                )));
            }
        }
        if let Some(base) = cname.strip_suffix("_length") {
            let Some(g_obj) = columns.get_item(base)? else {
                return Err(invalid(format!(
                    "columns[{cname:?}] has no matching grade column {base:?}"
                )));
            };
            let (_, gl_dtype, _) = check_array(&cv, &format!("columns[{cname:?}]"), Some(1))?;
            if gl_dtype != "float64" {
                return Err(invalid(format!(
                    "columns[{cname:?}] must have dtype float64, got {gl_dtype}"
                )));
            }
            let g_arr: PyReadonlyArray1<f64> = g_obj.extract()?;
            let gl_arr: PyReadonlyArray1<f64> = cv.extract()?;
            let (g_s, gl_s) = (g_arr.as_slice()?, gl_arr.as_slice()?);
            for r in 0..n_rows {
                if g_s[r].is_nan() {
                    if !(gl_s[r].is_nan() || gl_s[r] == 0.0) {
                        return Err(invalid(format!(
                            "columns[{cname:?}][{r}] must be NaN or 0 where {base:?} is NaN"
                        )));
                    }
                } else if !(gl_s[r] >= 0.0 && gl_s[r] <= lengths[r] + 1e-9) {
                    return Err(invalid(format!(
                        "columns[{cname:?}][{r}] = {} must lie in [0, length]",
                        gl_s[r]
                    )));
                }
            }
        }
    }
    Ok((holes, xyz, hi, fd, td))
}

/// Validates a `drillholes` protocol dict.
pub fn validate_drillholes_dict(d: &Bound<PyDict>) -> PyResult<Vec<String>> {
    for k in [
        "holes",
        "coords",
        "survey",
        "intervals",
        "tables",
        "path_coords",
        "path_depths",
        "path_offsets",
        "method",
    ] {
        if !d.contains(k)? {
            return Err(invalid(format!("missing protocol key {k:?}")));
        }
    }
    let holes = validate_holes(&d.get_item("holes")?.unwrap(), "holes")?;
    let h_count = holes.len();
    check_coords_array(&d.get_item("coords")?.unwrap(), "coords", Some(h_count))?;
    validate_table_fields(d, h_count, &[])?;

    let survey_obj = d.get_item("survey")?.unwrap();
    if !survey_obj.is_none() {
        let Ok(s_dict) = survey_obj.cast::<PyDict>() else {
            return Err(invalid("survey must be a dict or None"));
        };
        for k in ["hole_index", "depth", "azimuth", "dip"] {
            if !s_dict.contains(k)? {
                return Err(invalid(format!("survey: missing protocol key {k:?}")));
            }
        }
        let shi_obj = s_dict.get_item("hole_index")?.unwrap();
        let (s_shape, shi_dtype, _) = check_array(&shi_obj, "survey[\"hole_index\"]", Some(1))?;
        if shi_dtype != "int32" {
            return Err(invalid(format!(
                "survey[\"hole_index\"] must have dtype int32, got {shi_dtype}"
            )));
        }
        let s_len = s_shape[0];
        let shi_arr: PyReadonlyArray1<i32> = shi_obj.extract()?;
        let shi = shi_arr.as_slice()?;
        for (i, &h) in shi.iter().enumerate() {
            if h < 0 || h >= h_count as i32 {
                return Err(invalid(format!(
                    "survey[\"hole_index\"][{i}] = {h} is out of range for {h_count} holes"
                )));
            }
        }
        let check_s_f64 = |name: &str| -> PyResult<Vec<f64>> {
            let obj = s_dict.get_item(name)?.unwrap();
            let (sh, dt, _) = check_array(&obj, &format!("survey[{name:?}]"), Some(1))?;
            if dt != "float64" || sh != [s_len] {
                return Err(invalid(format!(
                    "survey[{name:?}] must be float64 of shape ({s_len},), got {dt} {sh:?}"
                )));
            }
            let a: PyReadonlyArray1<f64> = obj.extract()?;
            let s = a.as_slice()?;
            if s.iter().any(|v| !v.is_finite()) {
                return Err(invalid(format!("survey[{name:?}] must be finite")));
            }
            Ok(s.to_vec())
        };
        let depth = check_s_f64("depth")?;
        let azimuth = check_s_f64("azimuth")?;
        let dip = check_s_f64("dip")?;
        for i in 0..s_len {
            if depth[i] < 0.0 {
                return Err(invalid(format!(
                    "survey[\"depth\"][{i}] = {} must be >= 0",
                    depth[i]
                )));
            }
            if !(0.0..360.0).contains(&azimuth[i]) {
                return Err(invalid(format!(
                    "survey[\"azimuth\"][{i}] = {} must lie in [0, 360)",
                    azimuth[i]
                )));
            }
            if !(-90.0..=90.0).contains(&dip[i]) {
                return Err(invalid(format!(
                    "survey[\"dip\"][{i}] = {} must lie in [-90, 90]",
                    dip[i]
                )));
            }
            if i > 0 {
                if shi[i] < shi[i - 1] {
                    return Err(invalid(
                        "survey[\"hole_index\"] must be sorted non-decreasing",
                    ));
                }
                if shi[i] == shi[i - 1] && depth[i] <= depth[i - 1] {
                    return Err(invalid(
                        "survey[\"depth\"] must be strictly increasing within a hole",
                    ));
                }
            }
        }
    }

    let iv_obj = d.get_item("intervals")?.unwrap();
    if !iv_obj.is_none() {
        let Ok(iv_dict) = iv_obj.cast::<PyDict>() else {
            return Err(invalid("intervals must be a dict or None"));
        };
        validate_interval_dict(iv_dict, Some(&holes))?;
    }

    let tables_obj = d.get_item("tables")?.unwrap();
    let Ok(tables_dict) = tables_obj.cast::<PyDict>() else {
        return Err(invalid("tables must be a dict"));
    };
    for (tk, tv) in tables_dict.iter() {
        let Ok(ts) = tk.cast::<PyString>() else {
            return Err(invalid("tables key must be a string"));
        };
        if ts.to_str()?.is_empty() {
            return Err(invalid("tables key must not be empty"));
        }
        let Ok(t_dict) = tv.cast::<PyDict>() else {
            return Err(invalid("tables entry must be a dict"));
        };
        validate_interval_dict(t_dict, Some(&holes))?;
    }

    let po_obj = d.get_item("path_offsets")?.unwrap();
    let (po_shape, po_dtype, _) = check_array(&po_obj, "path_offsets", Some(1))?;
    if po_dtype != "uint32" || po_shape != [h_count + 1] {
        return Err(invalid(format!(
            "path_offsets must be uint32 of shape ({},), got {po_dtype} {po_shape:?}",
            h_count + 1
        )));
    }
    let po_arr: PyReadonlyArray1<u32> = po_obj.extract()?;
    let po = po_arr.as_slice()?;
    if po[0] != 0 {
        return Err(invalid("path_offsets[0] must be 0"));
    }
    for h in 0..h_count {
        if po[h + 1] < po[h] + 2 {
            return Err(invalid(format!(
                "path_offsets must give at least 2 vertices for hole {h}"
            )));
        }
    }
    let w_count = po[h_count] as usize;
    check_coords_array(
        &d.get_item("path_coords")?.unwrap(),
        "path_coords",
        Some(w_count),
    )?;
    let pd_obj = d.get_item("path_depths")?.unwrap();
    let (pd_shape, pd_dtype, _) = check_array(&pd_obj, "path_depths", Some(1))?;
    if pd_dtype != "float64" || pd_shape != [w_count] {
        return Err(invalid(format!(
            "path_depths must be float64 of shape ({w_count},), got {pd_dtype} {pd_shape:?}"
        )));
    }
    let pd_arr: PyReadonlyArray1<f64> = pd_obj.extract()?;
    let pd = pd_arr.as_slice()?;
    if pd.iter().any(|v| !v.is_finite()) {
        return Err(invalid("path_depths must be finite"));
    }
    for h in 0..h_count {
        let (start, end) = (po[h] as usize, po[h + 1] as usize);
        if pd[start] != 0.0 {
            return Err(invalid(format!(
                "path_depths must start at 0 at the collar of hole {h}, got {}",
                pd[start]
            )));
        }
        for i in start + 1..end {
            if pd[i] <= pd[i - 1] {
                return Err(invalid(format!(
                    "path_depths must be strictly increasing within hole {h}"
                )));
            }
        }
    }
    let m_obj = d.get_item("method")?.unwrap();
    let Ok(m_str) = m_obj.cast::<PyString>() else {
        return Err(invalid("method must be a string"));
    };
    let m = m_str.to_str()?;
    if !matches!(
        m,
        "minimum_curvature" | "tangential" | "balanced_tangential"
    ) {
        return Err(invalid(format!(
            "method must be \"minimum_curvature\", \"tangential\" or \"balanced_tangential\", got {m:?}"
        )));
    }
    Ok(holes)
}

/// Selects the interval table from a validated `drillholes` dict.
fn select_drillhole_interval_dict<'py>(
    d: &Bound<'py, PyDict>,
    col_name: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let iv_obj = d.get_item("intervals")?.unwrap();
    let tables = d.get_item("tables")?.unwrap().cast_into::<PyDict>()?;
    if let Some(name) = col_name {
        if !iv_obj.is_none() {
            let iv = iv_obj.cast::<PyDict>()?;
            let cols = iv.get_item("columns")?.unwrap().cast_into::<PyDict>()?;
            if cols.contains(name)? {
                return Ok(iv.clone());
            }
        }
        for (_, tv) in tables.iter() {
            let td = tv.cast_into::<PyDict>()?;
            let cols = td.get_item("columns")?.unwrap().cast_into::<PyDict>()?;
            if cols.contains(name)? {
                return Ok(td);
            }
        }
        if iv_obj.is_none() && tables.is_empty() {
            return Err(invalid("drillholes has no intervals or tables"));
        }
        return Err(invalid(format!(
            "codes names column {name:?}, which data does not have"
        )));
    }
    if !iv_obj.is_none() {
        let iv = iv_obj.cast::<PyDict>()?;
        let iv_cats = iv.get_item("categories")?.unwrap().cast_into::<PyDict>()?;
        if !iv_cats.is_empty() || tables.is_empty() {
            return Ok(iv.clone());
        }
        for (_, tv) in tables.iter() {
            let td = tv.cast_into::<PyDict>()?;
            let cats = td.get_item("categories")?.unwrap().cast_into::<PyDict>()?;
            if !cats.is_empty() {
                return Ok(td);
            }
        }
        return Ok(iv.clone());
    }
    if tables.len() == 1 {
        let (_, tv) = tables.iter().next().unwrap();
        return Ok(tv.cast_into::<PyDict>()?);
    }
    for (_, tv) in tables.iter() {
        let td = tv.cast_into::<PyDict>()?;
        let cats = td.get_item("categories")?.unwrap().cast_into::<PyDict>()?;
        if !cats.is_empty() {
            return Ok(td);
        }
    }
    Err(invalid("drillholes has no intervals"))
}

/// Picks a categorical or code column from an interval/composite/point_set/grid dict.
pub fn resolve_code_column<'py>(
    iv_dict: &Bound<'py, PyDict>,
    code_arg: Option<&Bound<'py, PyAny>>,
    user_categories: Option<&Bound<'py, PyAny>>,
) -> PyResult<(Vec<Code>, usize, Option<Categories>)> {
    let cols = iv_dict
        .get_item("columns")?
        .unwrap()
        .cast_into::<PyDict>()?;
    let cats = iv_dict
        .get_item("categories")?
        .unwrap()
        .cast_into::<PyDict>()?;
    let domain_opt = iv_dict.get_item("domain")?.filter(|o| !o.is_none());

    let (col_arr, chosen_name): (Bound<'py, PyAny>, Option<String>) = match code_arg {
        Some(arg) if !arg.is_none() => {
            if let Ok(s) = arg.cast::<PyString>() {
                let name = s.to_str()?;
                if let Some(arr) = cols.get_item(name)? {
                    (arr, Some(name.to_string()))
                } else if name == "domain"
                    && let Some(dom) = &domain_opt
                {
                    (dom.clone(), Some("domain".to_string()))
                } else {
                    return Err(invalid(format!(
                        "codes names column {name:?}, which data does not have"
                    )));
                }
            } else {
                (arg.clone(), None)
            }
        }
        _ => {
            if cats.len() == 1 {
                let (ck, _) = cats.iter().next().unwrap();
                let name = ck.cast::<PyString>()?.to_str()?.to_string();
                if let Some(arr) = cols.get_item(&name)? {
                    (arr, Some(name))
                } else if name == "domain"
                    && let Some(dom) = &domain_opt
                {
                    (dom.clone(), Some(name))
                } else {
                    return Err(invalid(format!("column {name:?} not found")));
                }
            } else if cols.len() == 1 {
                let (ck, cv) = cols.iter().next().unwrap();
                let name = ck.cast::<PyString>()?.to_str()?.to_string();
                (cv, Some(name))
            } else if let Some(dom) = &domain_opt {
                (dom.clone(), Some("domain".to_string()))
            } else {
                let mut int_cols = Vec::new();
                for (ck, cv) in cols.iter() {
                    let name = ck.cast::<PyString>()?.to_str()?.to_string();
                    let dt: String = cv.getattr("dtype")?.getattr("name")?.extract()?;
                    if dt == "int32" || dt == "int64" {
                        int_cols.push((name, cv));
                    }
                }
                if int_cols.len() == 1 {
                    let (name, cv) = int_cols.pop().unwrap();
                    (cv, Some(name))
                } else {
                    return Err(invalid(
                        "protocol object has multiple columns; pass the code column name",
                    ));
                }
            }
        }
    };

    let mut proto_cat = None;
    if let Some(name) = &chosen_name
        && let Some(c_entry) = cats.get_item(name)?
    {
        proto_cat = Some(validate_category_entry(
            c_entry.cast::<PyDict>()?,
            &format!("categories[{name:?}]"),
        )?);
    }

    // If the column in the protocol object is a unicode text column, encode it via user_categories or from_labels.
    let dtype_kind: Option<String> = col_arr
        .getattr("dtype")
        .ok()
        .and_then(|d| d.getattr("kind").ok())
        .and_then(|k| k.extract().ok());
    if matches!(dtype_kind.as_deref(), Some("U" | "T")) && chosen_name.is_some() {
        let cat = match user_categories.filter(|o| !o.is_none()) {
            Some(uc) => {
                if let Ok(c) = uc.cast::<PyCategories>() {
                    c.get().0.clone()
                } else if is_protocol(uc)? {
                    categories_from_protocol(uc)?
                } else {
                    PyCategories::from_labels(&col_arr)?.0
                }
            }
            None => PyCategories::from_labels(&col_arr)?.0,
        };
        let list: Vec<String> = col_arr.call_method0("tolist")?.extract()?;
        let encoded = cat
            .encode(list.iter().map(|s| Some(s.as_str())))
            .map_err(to_rust)?;
        let k = categories_k(user_categories, &encoded)?;
        return Ok((encoded, k, Some(cat)));
    }

    let c = codes(&col_arr, "codes")?;
    let k = if user_categories.is_some_and(|o| !o.is_none()) {
        categories_k(user_categories, &c)?
    } else if let Some(cat) = &proto_cat {
        let k = cat.len();
        transiograms_core::codes::check_codes(&c, k).map_err(to_rust)?;
        k
    } else {
        categories_k(None, &c)?
    };
    Ok((c, k, proto_cat))
}

pub struct ProtocolSequence<'py> {
    pub holes_per_row: Vec<String>,
    pub from: Vec<f64>,
    pub to: Vec<f64>,
    pub codes: Vec<Code>,
    pub k: usize,
    pub iv_columns: Bound<'py, PyDict>,
    pub collar_columns: Option<Bound<'py, PyDict>>,
    pub hole_index: Vec<i32>,
}

/// Extracts interval sequence fields from a `drillholes`, `intervals`, or `composites` protocol object.
pub fn sequence_from_protocol<'py>(
    proto_obj: &Bound<'py, PyAny>,
    code_arg: Option<&Bound<'py, PyAny>>,
    user_categories: Option<&Bound<'py, PyAny>>,
) -> PyResult<ProtocolSequence<'py>> {
    let d = protocol_dict(proto_obj)?;
    let (kind, _, _) = validate_common(&d, &["drillholes", "intervals", "composites"])?;
    let col_name = code_arg
        .and_then(|a| a.cast::<PyString>().ok())
        .and_then(|s| s.to_str().ok());
    let (holes, hi, from, to, iv_dict, collar_columns) = match kind.as_str() {
        "intervals" => {
            let (holes, hi, fd, td) = validate_interval_dict(&d, None)?;
            (holes, hi, fd, td, d.clone(), None)
        }
        "composites" => {
            let (holes, _, hi, fd, td) = validate_composites_dict(&d)?;
            (holes, hi, fd, td, d.clone(), None)
        }
        "drillholes" => {
            let holes = validate_drillholes_dict(&d)?;
            let iv_dict = select_drillhole_interval_dict(&d, col_name)?;
            let (_, hi, fd, td) = validate_interval_dict(&iv_dict, Some(&holes))?;
            let collar_cols = d.get_item("columns")?.unwrap().cast_into::<PyDict>()?;
            (holes, hi, fd, td, iv_dict, Some(collar_cols))
        }
        _ => unreachable!(),
    };
    let holes_per_row: Vec<String> = hi.iter().map(|&i| holes[i as usize].clone()).collect();
    let (codes, k, _) = resolve_code_column(&iv_dict, code_arg, user_categories)?;
    if codes.len() != holes_per_row.len() {
        return Err(invalid(format!(
            "codes has {} values for {} rows",
            codes.len(),
            holes_per_row.len()
        )));
    }
    let iv_columns = iv_dict
        .get_item("columns")?
        .unwrap()
        .cast_into::<PyDict>()?;
    Ok(ProtocolSequence {
        holes_per_row,
        from,
        to,
        codes,
        k,
        iv_columns,
        collar_columns,
        hole_index: hi,
    })
}

/// Extracts `(columns_dict, default_holes_per_row)` from a protocol object for HMM/HSMM inputs.
pub fn table_columns_from_protocol<'py>(
    proto_obj: &Bound<'py, PyAny>,
    seq_arg: Option<&Bound<'py, PyAny>>,
) -> PyResult<(Bound<'py, PyDict>, Option<Vec<String>>)> {
    let d = protocol_dict(proto_obj)?;
    let (kind, _, _) = validate_common(
        &d,
        &[
            "drillholes",
            "intervals",
            "composites",
            "point_set",
            "table",
        ],
    )?;
    let col_name = seq_arg
        .and_then(|a| a.cast::<PyString>().ok())
        .and_then(|s| s.to_str().ok());
    match kind.as_str() {
        "intervals" => {
            let (holes, hi, _, _) = validate_interval_dict(&d, None)?;
            let cols = d.get_item("columns")?.unwrap().cast_into::<PyDict>()?;
            let h = hi.iter().map(|&i| holes[i as usize].clone()).collect();
            Ok((cols, Some(h)))
        }
        "composites" => {
            let (holes, _, hi, _, _) = validate_composites_dict(&d)?;
            let cols = d.get_item("columns")?.unwrap().cast_into::<PyDict>()?;
            let h = hi.iter().map(|&i| holes[i as usize].clone()).collect();
            Ok((cols, Some(h)))
        }
        "drillholes" => {
            let holes = validate_drillholes_dict(&d)?;
            let iv_dict = select_drillhole_interval_dict(&d, col_name)?;
            let (_, hi, _, _) = validate_interval_dict(&iv_dict, Some(&holes))?;
            let cols = iv_dict
                .get_item("columns")?
                .unwrap()
                .cast_into::<PyDict>()?;
            let h = hi.iter().map(|&i| holes[i as usize].clone()).collect();
            Ok((cols, Some(h)))
        }
        "point_set" => {
            if !d.contains("coords")? {
                return Err(invalid("missing protocol key \"coords\""));
            }
            let xyz = check_coords_array(&d.get_item("coords")?.unwrap(), "coords", None)?;
            validate_table_fields(&d, xyz.len(), &[])?;
            let cols = d.get_item("columns")?.unwrap().cast_into::<PyDict>()?;
            Ok((cols, None))
        }
        "table" => {
            if !d.contains("rows")? {
                return Err(invalid("missing protocol key \"rows\""));
            }
            let r_obj = d.get_item("rows")?.unwrap();
            if r_obj.is_instance_of::<PyBool>() {
                return Err(invalid("rows must be a non-negative integer"));
            }
            let rows: isize = r_obj
                .extract()
                .map_err(|_| invalid("rows must be a non-negative integer"))?;
            if rows < 0 {
                return Err(invalid("rows must be >= 0"));
            }
            validate_table_fields(&d, rows as usize, &[])?;
            let cols = d.get_item("columns")?.unwrap().cast_into::<PyDict>()?;
            Ok((cols, None))
        }
        _ => unreachable!(),
    }
}

pub struct ProtocolPoints {
    pub coords: Vec<[f64; 3]>,
    pub codes: Vec<Code>,
    pub k: usize,
    pub categories: Option<Categories>,
    pub hole_indices: Option<Vec<u32>>,
    pub crs: Option<String>,
    pub length_unit: Option<String>,
}

/// Resolves point data from a `point_set` or `composites` protocol object (passed as `coords` or `data`).
pub fn points_from_protocol<'py>(
    coords_arg: &Bound<'py, PyAny>,
    codes_arg: Option<&Bound<'py, PyAny>>,
    holes_arg: Option<&Bound<'py, PyAny>>,
    user_categories: Option<&Bound<'py, PyAny>>,
    data_arg: Option<&Bound<'py, PyAny>>,
) -> PyResult<ProtocolPoints> {
    let (proto_obj, explicit_coords) = if is_protocol(coords_arg)? {
        if let Some(d_arg) = data_arg.filter(|o| !o.is_none())
            && is_protocol(d_arg)?
            && !coords_arg.is(d_arg)
        {
            let d1 = protocol_dict(coords_arg)?;
            let d2 = protocol_dict(d_arg)?;
            let (_, crs1, lu1) = validate_common(&d1, &["point_set", "composites"])?;
            let (_, crs2, lu2) = validate_common(&d2, &["point_set", "composites"])?;
            combine_crs(crs1.as_deref(), crs2.as_deref())?;
            combine_length_unit(lu1.as_deref(), lu2.as_deref())?;
        }
        (coords_arg, None)
    } else if let Some(d_arg) = data_arg.filter(|o| !o.is_none())
        && is_protocol(d_arg)?
    {
        let explicit = if coords_arg.is_none()
            || coords_arg
                .cast::<PyString>()
                .is_ok_and(|s| s.to_str().ok() == Some("coords"))
        {
            None
        } else {
            Some(coords_arg)
        };
        (d_arg, explicit)
    } else {
        return Err(invalid(
            "expected a point_set or composites protocol object",
        ));
    };

    let d = protocol_dict(proto_obj)?;
    let (kind, crs, length_unit) = validate_common(&d, &["point_set", "composites"])?;
    let (mut xyz, default_hole_idx) = match kind.as_str() {
        "point_set" => {
            if !d.contains("coords")? {
                return Err(invalid("missing protocol key \"coords\""));
            }
            let xyz = check_coords_array(&d.get_item("coords")?.unwrap(), "coords", None)?;
            validate_table_fields(&d, xyz.len(), &[])?;
            (xyz, None)
        }
        "composites" => {
            let (_, xyz, hi, _, _) = validate_composites_dict(&d)?;
            let h_u32: Vec<u32> = hi.into_iter().map(|i| i as u32).collect();
            (xyz, Some(h_u32))
        }
        _ => unreachable!(),
    };
    if let Some(exp) = explicit_coords {
        xyz = coords(exp, "coords")?;
    }
    let (codes, k, categories) = resolve_code_column(&d, codes_arg, user_categories)?;
    if xyz.len() != codes.len() {
        return Err(invalid(format!(
            "{} coordinates but {} codes",
            xyz.len(),
            codes.len()
        )));
    }
    let cols = d.get_item("columns")?.unwrap().cast_into::<PyDict>()?;
    let hole_indices = match holes_arg.filter(|o| !o.is_none()) {
        Some(h) => {
            let h_col = column(Some(cols.as_any()), h, "holes")?;
            let ids = crate::args::hole_ids(Some(&h_col), codes.len())?;
            Some(crate::args::hole_indices(&ids))
        }
        None => default_hole_idx,
    };
    Ok(ProtocolPoints {
        coords: xyz,
        codes,
        k,
        categories,
        hole_indices,
        crs,
        length_unit,
    })
}

/// Builds a read-only 1-D `int32` NumPy array from internal `Code` values (`MISSING -> -1`).
pub fn readonly_i32<'py>(py: Python<'py>, codes: &[Code]) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let arr = PyArray1::from_vec(
        py,
        codes
            .iter()
            .map(|&c| if c == MISSING { -1 } else { c as i32 })
            .collect(),
    );
    arr.as_any().getattr("flags")?.setattr("writeable", false)?;
    Ok(arr)
}
