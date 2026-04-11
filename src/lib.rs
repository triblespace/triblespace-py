use std::{
    borrow::Cow,
    collections::HashMap,
    hash::Hash,
    sync::{Arc, LazyLock},
};

use itertools::Itertools;
use parking_lot::{ArcMutexGuard, Mutex, RawMutex, RwLock};
use pyo3::{
    exceptions::{PyKeyError, PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyBytes, PyType},
};
use triblespace::core as ts_core;
use ts_core::metadata;
use ts_core::id::IdOwner;
use triblespace::prelude::*;
use ts_core::query::{
    constantconstraint::ConstantConstraint, Binding, Constraint, ContainsConstraint, Query,
    TriblePattern, Variable, VariableId,
    equalityconstraint::EqualityConstraint,
    unionconstraint::UnionConstraint,
    RegularPathConstraint, PathOp,
};
use ts_core::value::{schemas::UnknownValue, RawValue};

use hex::FromHex;

struct PyPtrIdentity<T>(pub Py<T>);

impl<T> PartialEq for PyPtrIdentity<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.as_ptr() == other.0.as_ptr()
    }
}

impl<T> Eq for PyPtrIdentity<T> {}

impl<T> Hash for PyPtrIdentity<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.as_ptr().hash(state);
    }
}

static TYPE_TO_ENTITY: LazyLock<Mutex<HashMap<PyPtrIdentity<PyType>, Id>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

static TO_VALUE_CONVERTERS: LazyLock<Mutex<HashMap<(Id, Id), Py<PyAny>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

static FROM_VALUE_CONVERTERS: LazyLock<Mutex<HashMap<(Id, Id), Py<PyAny>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

static TO_BLOB_CONVERTERS: LazyLock<Mutex<HashMap<(Id, Id), Py<PyAny>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

static FROM_BLOB_CONVERTERS: LazyLock<Mutex<HashMap<(Id, Id), Py<PyAny>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

#[pyfunction]
pub fn get_value_schema(context: &PyTribleSet, attr_id: &PyId) -> PyResult<PyId> {
    let data = context.0.lock();
    match find!((value_schema: Id),
    pattern!(&*data, [
        {&(attr_id.0) @
            metadata::value_schema: ?value_schema}
    ]))
    .exactly_one()
    {
        Ok((schema_id,)) => Ok(PyId(schema_id)),
        Err(results) => match results.count() {
            0 => Err(PyErr::new::<PyKeyError, _>(
                "attribute should be registered first",
            )),
            _ => Err(PyErr::new::<PyKeyError, _>(
                "multiple value schemas for attribute",
            )),
        },
    }
}

#[pyfunction]
pub fn get_blob_schema(context: &PyTribleSet, attr_id: &PyId) -> PyResult<Option<PyId>> {
    let data = context.0.lock();
    match find!((blob_schema: Id),
    pattern!(&*data, [
        {&(attr_id.0) @
            metadata::blob_schema: ?blob_schema}
    ]))
    .at_most_one()
    {
        Ok(None) => Ok(None),
        Ok(Some((schema_id,))) => Ok(Some(PyId(schema_id))),
        Err(results) => match results.count() {
            _ => Err(PyErr::new::<PyKeyError, _>(
                "multiple blob schemas for attribute",
            )),
        },
    }
}

/// Returns pairs of (attr_id, name_handle_hex) for all named attributes.
/// The name_handle is a 32-byte blob reference that can be resolved
/// via a blob store to get the actual string.
#[pyfunction]
pub fn get_label_name_handles(context: &PyTribleSet) -> PyResult<Vec<(PyId, String)>> {
    let data = context.0.lock();
    use ts_core::value::Value;
    use ts_core::value::schemas::hash::{Blake3, Handle};
    use ts_core::blob::schemas::longstring::LongString;
    let results: Vec<(PyId, String)> = find!(
        (attr_id: Id, name_handle: Value<Handle<Blake3, LongString>>),
        pattern!(&*data, [{
            ?attr_id @
            metadata::name: ?name_handle
        }])
    )
    .map(|(aid, nh)| (PyId(aid), hex::encode_upper(nh.raw)))
    .collect();
    Ok(results)
}

#[pyfunction]
pub fn register_type(type_id: &PyId, typ: Py<PyType>) {
    let mut type_to_entity = TYPE_TO_ENTITY.lock();
    type_to_entity.insert(PyPtrIdentity(typ), type_id.0);
}

#[pyfunction]
pub fn register_to_value_converter(
    value_schema_id: &PyId,
    typ: Py<PyType>,
    converter: Py<PyAny>,
) -> PyResult<()> {
    let type_id = {
        let type_to_entity = TYPE_TO_ENTITY.lock();
        let Some(entity) = type_to_entity.get(&PyPtrIdentity(typ)) else {
            return Err(PyErr::new::<PyKeyError, _>("no such type registered"));
        };
        entity.clone()
    };
    TO_VALUE_CONVERTERS
        .lock()
        .insert((value_schema_id.0, type_id), converter);
    Ok(())
}

#[pyfunction]
pub fn register_from_value_converter(
    value_schema_id: &PyId,
    typ: Py<PyType>,
    converter: Py<PyAny>,
) -> PyResult<()> {
    let type_id = {
        let type_to_entity = TYPE_TO_ENTITY.lock();
        let Some(entity) = type_to_entity.get(&PyPtrIdentity(typ)) else {
            return Err(PyErr::new::<PyKeyError, _>("no such type registered"));
        };
        entity.clone()
    };
    FROM_VALUE_CONVERTERS
        .lock()
        .insert((value_schema_id.0, type_id), converter);
    Ok(())
}

#[pyfunction]
pub fn register_to_blob_converter(
    blob_schema_id: &PyId,
    typ: Py<PyType>,
    converter: Py<PyAny>,
) -> PyResult<()> {
    let type_id = {
        let type_to_entity = TYPE_TO_ENTITY.lock();
        let Some(entity) = type_to_entity.get(&PyPtrIdentity(typ)) else {
            return Err(PyErr::new::<PyKeyError, _>("no such type registered"));
        };
        entity.clone()
    };
    TO_BLOB_CONVERTERS
        .lock()
        .insert((blob_schema_id.0, type_id), converter);
    Ok(())
}

#[pyfunction]
pub fn register_from_blob_converter(
    blob_schema_id: &PyId,
    typ: Py<PyType>,
    converter: Py<PyAny>,
) -> PyResult<()> {
    let type_id = {
        let type_to_entity = TYPE_TO_ENTITY.lock();
        let Some(entity) = type_to_entity.get(&PyPtrIdentity(typ)) else {
            return Err(PyErr::new::<PyKeyError, _>("no such type registered"));
        };
        entity.clone()
    };
    FROM_BLOB_CONVERTERS
        .lock()
        .insert((blob_schema_id.0, type_id), converter);
    Ok(())
}

#[pyfunction]
pub fn metadata_description() -> PyTribleSet {
    // TODO: use the Describe trait to build metadata description
    PyTribleSet(Mutex::new(TribleSet::new()))
}

#[derive(Debug, Copy, Clone)]
#[pyclass(frozen, name = "Id")]
pub struct PyId(Id);

#[pymethods]
impl PyId {
    #[new]
    fn new(bytes: &[u8]) -> Result<Self, PyErr> {
        let Ok(id) = bytes.try_into() else {
            return Err(PyValueError::new_err("ids should be 16 bytes"));
        };
        let Some(id) = Id::new(id) else {
            return Err(PyValueError::new_err(
                "id must be non-nil (contain non-zero bytes)",
            ));
        };
        Ok(PyId(id))
    }

    #[staticmethod]
    pub fn hex(hex: &str) -> Result<Self, PyErr> {
        let Ok(id) = <[u8; 16]>::from_hex(hex) else {
            return Err(PyValueError::new_err("failed to parse hex id"));
        };
        let Some(id) = Id::new(id) else {
            return Err(PyValueError::new_err(
                "id must be non-nil (contain non-zero bytes)",
            ));
        };
        Ok(PyId(id))
    }

    pub fn bytes(&self) -> Cow<[u8]> {
        Cow::Borrowed(self.0.as_ref())
    }

    pub fn to_hex(&self) -> String {
        hex::encode_upper(self.0)
    }

    fn __repr__(&self) -> String {
        format!("Id({})", &self.to_hex()[..8])
    }

    fn __hash__(&self) -> u64 {
        use std::hash::Hasher;
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.0.hash(&mut h);
        h.finish()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

#[pyclass(frozen, name = "IdOwner")]
pub struct PyIdOwner(Arc<Mutex<IdOwner>>);

#[pymethods]
impl PyIdOwner {
    #[new]
    fn new() -> Self {
        PyIdOwner(Arc::new(Mutex::new(IdOwner::new())))
    }

    pub fn rngid(&self) -> PyId {
        let owned_id = rngid();
        let id = self.0.lock_arc().insert(owned_id);
        PyId(id)
    }

    pub fn ufoid(&self) -> PyId {
        let owned_id = ufoid();
        let id = self.0.lock_arc().insert(owned_id);
        PyId(id)
    }

    pub fn fucid(&self) -> PyId {
        let owned_id = fucid();
        let id = self.0.lock_arc().insert(owned_id);
        PyId(id)
    }

    pub fn has(&self, v: PyRef<'_, PyVariable>) -> PyConstraint {
        PyConstraint {
            constraint: Arc::new(self.0.lock_arc().has(Variable::new(v.0.read().index))),
        }
    }

    pub fn owns(&self, id: PyRef<'_, PyId>) -> bool {
        self.0.lock_arc().owns(&id.0)
    }

    pub fn lock(&self) -> PyIdOwnerGuard {
        PyIdOwnerGuard(Mutex::new(Some(self.0.lock_arc())))
    }
}

#[pyclass(frozen, name = "IdOwnerGuard")]
pub struct PyIdOwnerGuard(Mutex<Option<ArcMutexGuard<RawMutex, IdOwner>>>);

#[pymethods]
impl PyIdOwnerGuard {
    pub fn rngid(&self) -> PyResult<PyId> {
        let owned_id = rngid();
        if let Some(guard) = &mut *self.0.lock() {
            let id = guard.insert(owned_id);
            Ok(PyId(id))
        } else {
            Err(PyErr::new::<PyRuntimeError, _>("guard has been released"))
        }
    }

    pub fn ufoid(&self) -> PyResult<PyId> {
        let owned_id = ufoid();
        if let Some(guard) = &mut *self.0.lock() {
            let id = guard.insert(owned_id);
            Ok(PyId(id))
        } else {
            Err(PyErr::new::<PyRuntimeError, _>("guard has been released"))
        }
    }

    pub fn fucid(&self) -> PyResult<PyId> {
        let owned_id = fucid();
        if let Some(guard) = &mut *self.0.lock() {
            let id = guard.insert(owned_id);
            Ok(PyId(id))
        } else {
            Err(PyErr::new::<PyRuntimeError, _>("guard has been released"))
        }
    }

    pub fn has(&self, v: PyRef<'_, PyVariable>) -> PyResult<PyConstraint> {
        if let Some(guard) = &mut *self.0.lock() {
            Ok(PyConstraint {
                constraint: Arc::new(guard.has(Variable::new(v.0.read().index))),
            })
        } else {
            Err(PyErr::new::<PyRuntimeError, _>("guard has been released"))
        }
    }

    pub fn owns(&self, id: PyRef<'_, PyId>) -> PyResult<bool> {
        if let Some(guard) = &mut *self.0.lock() {
            Ok(guard.owns(&id.0))
        } else {
            Err(PyErr::new::<PyRuntimeError, _>("guard has been released"))
        }
    }

    pub fn release(&self) {
        *self.0.lock() = None;
    }

    pub fn __enter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    pub fn __exit__(
        &self,
        _exc_type: PyObject,
        _exc_value: PyObject,
        _traceback: PyObject,
    ) -> bool {
        self.release();
        false
    }
}

#[pyclass(frozen, name = "Value")]
pub struct PyValue {
    value: RawValue,
    _value_schema: Id,
    _blob_schema: Option<Id>,
}

#[pymethods]
impl PyValue {
    /// Create a raw value from 32 bytes.
    #[new]
    fn new(bytes: &[u8]) -> PyResult<Self> {
        if bytes.len() != 32 {
            return Err(PyValueError::new_err("values must be 32 bytes"));
        }
        let mut raw = [0u8; 32];
        raw.copy_from_slice(bytes);
        // Use a dummy schema for raw values.
        let dummy_schema = Id::new([0xFF; 16]).unwrap();
        Ok(PyValue { value: raw, _value_schema: dummy_schema, _blob_schema: None })
    }

    #[pyo3(signature = (value, value_schema, blob_schema=None))]
    #[staticmethod]
    fn of(
        py: Python<'_>,
        value: Bound<'_, PyAny>,
        value_schema: PyRef<'_, PyId>,
        blob_schema: Option<PyRef<'_, PyId>>,
    ) -> PyResult<Self> {
        let value_schema = value_schema.0;
        let blob_schema = blob_schema.map(|s| s.0);
        let type_id = {
            let typ = value.get_type().unbind();
            let type_to_entity = TYPE_TO_ENTITY.lock();
            let Some(entity) = type_to_entity.get(&PyPtrIdentity(typ)) else {
                return Err(PyErr::new::<PyKeyError, _>("no such type registered"));
            };
            entity.clone()
        };
        let converters = TO_VALUE_CONVERTERS.lock();
        let Some(converter) = converters.get(&(value_schema, type_id)) else {
            return Err(PyErr::new::<PyKeyError, _>(format!(
                "no converter to schema from type`{value_schema:X}` and type `{type_id:X}`"
            )));
        };
        let bytes = converter.call(py, (value,), None)?;
        let bytes = bytes.downcast_bound::<PyBytes>(py)?;
        let value: RawValue = bytes.as_bytes().try_into()?;
        Ok(Self {
            value,
            _value_schema: value_schema,
            _blob_schema: blob_schema,
        })
    }

    fn to(&self, py: Python<'_>, typ: Py<PyType>) -> PyResult<Py<PyAny>> {
        let type_id = {
            let type_to_entity = TYPE_TO_ENTITY.lock();
            let Some(entity) = type_to_entity.get(&PyPtrIdentity(typ)) else {
                return Err(PyErr::new::<PyKeyError, _>("no such type registered"));
            };
            entity.clone()
        };
        let converters = FROM_VALUE_CONVERTERS.lock();
        let Some(converter) = converters.get(&(self._value_schema, type_id)) else {
            return Err(PyErr::new::<PyKeyError, _>(
                "no converter from schema to type",
            ));
        };
        let bytes = PyBytes::new(py, &self.value);
        converter.call(py, (bytes,), None)
    }

    pub fn value_schema(&self) -> PyId {
        PyId(self._value_schema)
    }

    pub fn blob_schema(&self) -> Option<PyId> {
        self._blob_schema.map(|s| PyId(s))
    }

    pub fn is_handle(&self) -> bool {
        self._blob_schema.is_some()
    }

    fn __hash__(&self) -> u64 {
        use std::hash::Hasher;
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.value.hash(&mut h);
        h.finish()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.value == other.value
    }

    /// Create a Value from a short string (max 31 bytes UTF-8).
    #[staticmethod]
    fn from_str(s: &str) -> PyResult<Self> {
        let val: Value<valueschemas::ShortString> = TryToValue::try_to_value(s)
            .map_err(|_| PyValueError::new_err("string too long for ShortString (max 31 bytes)"))?;
        // Use a well-known schema ID for ShortString.
        let schema_id = <valueschemas::ShortString as ConstId>::ID;
        Ok(PyValue {
            value: val.raw,
            _value_schema: schema_id,
            _blob_schema: None,
        })
    }

    /// Extract a short string from this Value.
    fn to_str(&self) -> PyResult<String> {
        let val = Value::<valueschemas::ShortString>::new(self.value);
        let s: String = TryFromValue::try_from_value(&val)
            .map_err(|_| PyValueError::new_err("not a valid ShortString"))?;
        Ok(s)
    }

    /// Create a Value from an Id (GenId schema — 16 zero bytes + 16-byte id).
    #[staticmethod]
    fn from_id(id: &PyId) -> PyResult<Self> {
        let mut raw = [0u8; 32];
        let id_bytes: [u8; 16] = id.0.into();
        raw[16..32].copy_from_slice(&id_bytes);
        let schema_id = <valueschemas::GenId as ConstId>::ID;
        Ok(PyValue {
            value: raw,
            _value_schema: schema_id,
            _blob_schema: None,
        })
    }

    /// Extract an Id from this Value (GenId schema).
    fn to_id(&self) -> PyResult<PyId> {
        let val = Value::<valueschemas::GenId>::new(self.value);
        let id: Id = TryFromValue::try_from_value(&val)
            .map_err(|_| PyValueError::new_err("not a valid GenId"))?;
        Ok(PyId(id))
    }

    /// Create a Value from an f64 (F64 schema — 8-byte LE in first 8 bytes).
    #[staticmethod]
    fn from_f64(v: f64) -> Self {
        let val: Value<valueschemas::F64> = ToValue::to_value(v);
        PyValue {
            value: val.raw,
            _value_schema: <valueschemas::F64 as ConstId>::ID,
            _blob_schema: None,
        }
    }

    /// Extract an f64 from this Value (F64 schema).
    fn to_f64(&self) -> PyResult<f64> {
        let val = Value::<valueschemas::F64>::new(self.value);
        let v: f64 = TryFromValue::try_from_value(&val)
            .map_err(|_| PyValueError::new_err("not a valid F64"))?;
        Ok(v)
    }

    /// Create a Value from a bool (Boolean schema).
    #[staticmethod]
    fn from_bool(v: bool) -> Self {
        let val: Value<valueschemas::Boolean> = ToValue::to_value(v);
        PyValue {
            value: val.raw,
            _value_schema: <valueschemas::Boolean as ConstId>::ID,
            _blob_schema: None,
        }
    }

    /// Extract a bool from this Value (Boolean schema).
    fn to_bool(&self) -> PyResult<bool> {
        let val = Value::<valueschemas::Boolean>::new(self.value);
        let v: bool = TryFromValue::try_from_value(&val)
            .map_err(|e| PyValueError::new_err(format!("not a valid Boolean: {e:?}")))?;
        Ok(v)
    }

    /// Create a Value from raw 32 bytes with an explicit schema.
    #[staticmethod]
    fn from_raw(bytes: &[u8], schema: &PySchema) -> PyResult<Self> {
        if bytes.len() != 32 {
            return Err(PyValueError::new_err("values must be 32 bytes"));
        }
        let mut raw = [0u8; 32];
        raw.copy_from_slice(bytes);
        Ok(PyValue { value: raw, _value_schema: schema.id, _blob_schema: None })
    }

    /// Get raw 32-byte value.
    pub fn raw_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.value)
    }

    fn __repr__(&self) -> String {
        let schema = self._value_schema;
        // Schema-aware repr
        if schema == <valueschemas::ShortString as ConstId>::ID {
            if let Ok(s) = self.to_str() {
                return format!("Value({s:?})");
            }
        }
        if schema == <valueschemas::GenId as ConstId>::ID {
            if let Ok(id) = self.to_id() {
                return format!("Value(Id({}))", &id.to_hex()[..8]);
            }
        }
        if schema == <valueschemas::F64 as ConstId>::ID {
            if let Ok(v) = self.to_f64() {
                return format!("Value({v})");
            }
        }
        if schema == <valueschemas::Boolean as ConstId>::ID {
            if let Ok(v) = self.to_bool() {
                return format!("Value({v})");
            }
        }
        // Fallback: try string, then id, then hex
        if let Ok(s) = self.to_str() {
            if !s.is_empty() {
                return format!("Value({s:?})");
            }
        }
        if self.value[..16] == [0u8; 16] && self.value[16..] != [0u8; 16] {
            if let Ok(id) = self.to_id() {
                return format!("Value(Id({}))", &id.to_hex()[..8]);
            }
        }
        format!("Value(0x{})", hex::encode(&self.value[..8]))
    }

    pub fn bytes(&self) -> Cow<[u8]> {
        (&self.value).into()
    }
}

#[pyclass(frozen, name = "TribleSet")]
pub struct PyTribleSet(Mutex<TribleSet>);

#[pymethods]
impl PyTribleSet {
    fn __repr__(&self) -> String {
        let len = self.0.lock().len();
        format!("TribleSet({len} tribles)")
    }

    #[new]
    pub fn new() -> Self {
        PyTribleSet(Mutex::new(TribleSet::new()))
    }

    #[staticmethod]
    pub fn empty() -> Self {
        PyTribleSet(Mutex::new(TribleSet::new()))
    }

    pub fn __add__(&self, other: &Self) -> Self {
        let mut result = self.0.lock().clone();
        result += other.0.lock().clone();
        PyTribleSet(Mutex::new(result))
    }

    pub fn __iadd__(&self, other: &Self) {
        let mut set = self.0.lock();
        *set += other.0.lock().clone();
    }

    pub fn __len__(&self) -> usize {
        return self.0.lock().eav.len() as usize;
    }

    pub fn fork(&self) -> Self {
        PyTribleSet(Mutex::new(self.0.lock().clone()))
    }

    pub fn add(&self, e: &PyId, a: &PyId, v: &PyValue) -> PyResult<()> {
        self.0.lock().insert(
            &(Trible::new(
                ExclusiveId::force_ref(&e.0),
                &a.0,
                Value::<UnknownValue>::as_transmute_raw(&v.value),
            )),
        );
        Ok(())
    }

    pub fn consume(&self, other: &Self) {
        let other_set: TribleSet = std::mem::replace(&mut other.0.lock(), TribleSet::new());
        let mut set = self.0.lock();
        *set += other_set;
    }

    /// Iterate all triples as (entity: Id, attribute: Id, value: Value) tuples.
    pub fn triples(&self) -> Vec<(PyId, PyId, PyValue)> {
        let set = self.0.lock();
        let mut result = Vec::new();
        for trible in set.eav.iter() {
            let mut e_raw = [0u8; 16];
            let mut a_raw = [0u8; 16];
            let mut v_raw = [0u8; 32];
            e_raw.copy_from_slice(&trible[..16]);
            a_raw.copy_from_slice(&trible[16..32]);
            v_raw.copy_from_slice(&trible[32..64]);
            let e = Id::new(e_raw).unwrap();
            let a = Id::new(a_raw).unwrap();
            // Use a dummy schema for iteration — caller interprets.
            let dummy = Id::new([0xFF; 16]).unwrap();
            result.push((
                PyId(e),
                PyId(a),
                PyValue { value: v_raw, _value_schema: dummy, _blob_schema: None },
            ));
        }
        result
    }

    pub fn pattern(
        &self,
        ev: PyRef<'_, PyVariable>,
        av: PyRef<'_, PyVariable>,
        vv: PyRef<'_, PyVariable>,
    ) -> PyConstraint {
        PyConstraint {
            constraint: Arc::new(self.0.lock().pattern(
                Variable::new(ev.0.read().index),
                Variable::new(av.0.read().index),
                Variable::<UnknownValue>::new(vv.0.read().index),
            )),
        }
    }

    /// Returns true if the set contains the given (e, a, v) trible.
    pub fn contains(&self, e: &PyId, a: &PyId, v: &PyValue) -> bool {
        let trible = Trible::new(
            ExclusiveId::force_ref(&e.0),
            &a.0,
            Value::<UnknownValue>::as_transmute_raw(&v.value),
        );
        self.0.lock().contains(&trible)
    }

    /// Returns true if the set is empty.
    pub fn is_empty(&self) -> bool {
        self.0.lock().is_empty()
    }

    /// Returns a new TribleSet containing only tribles present in both sets.
    #[pyo3(name = "intersect")]
    pub fn set_intersect(&self, other: &Self) -> Self {
        PyTribleSet(Mutex::new(self.0.lock().intersect(&other.0.lock())))
    }

    /// Returns a new TribleSet with tribles in self but not in other.
    pub fn difference(&self, other: &Self) -> Self {
        PyTribleSet(Mutex::new(self.0.lock().difference(&other.0.lock())))
    }

    /// Equality test — two sets are equal iff they contain the same tribles.
    pub fn __eq__(&self, other: &Self) -> bool {
        *self.0.lock() == *other.0.lock()
    }

    fn __bool__(&self) -> bool {
        !self.0.lock().is_empty()
    }

    /// Operator - for set difference.
    pub fn __sub__(&self, other: &Self) -> Self {
        self.difference(other)
    }

    /// Operator & for set intersection.
    pub fn __and__(&self, other: &Self) -> Self {
        self.set_intersect(other)
    }

    /// Fast fingerprint for equality/caching (not stable across processes).
    pub fn __hash__(&self) -> u64 {
        let fp = self.0.lock().fingerprint();
        // TribleSetFingerprint wraps Option<u128> — convert to u64 for Python hash
        use std::hash::Hasher;
        let mut h = std::collections::hash_map::DefaultHasher::new();
        fp.hash(&mut h);
        h.finish()
    }

    /// Constrain a value variable to a byte range [min, max] (VEA index).
    pub fn value_in_range(
        &self,
        variable: PyRef<'_, PyVariable>,
        min: &PyValue,
        max: &PyValue,
    ) -> PyConstraint {
        PyConstraint {
            constraint: Arc::new(self.0.lock().value_in_range(
                Variable::<UnknownValue>::new(variable.0.read().index),
                Value::<UnknownValue>::new(min.value),
                Value::<UnknownValue>::new(max.value),
            )),
        }
    }

    /// Constrain an entity variable to an ID range [min, max] (EAV index).
    pub fn entity_in_range(
        &self,
        variable: PyRef<'_, PyVariable>,
        min: &PyId,
        max: &PyId,
    ) -> PyConstraint {
        PyConstraint {
            constraint: Arc::new(self.0.lock().entity_in_range(
                Variable::<valueschemas::GenId>::new(variable.0.read().index),
                min.0,
                max.0,
            )),
        }
    }

    /// Constrain an attribute variable to an ID range [min, max] (AEV index).
    pub fn attribute_in_range(
        &self,
        variable: PyRef<'_, PyVariable>,
        min: &PyId,
        max: &PyId,
    ) -> PyConstraint {
        PyConstraint {
            constraint: Arc::new(self.0.lock().attribute_in_range(
                Variable::<valueschemas::GenId>::new(variable.0.read().index),
                min.0,
                max.0,
            )),
        }
    }
}

// ── Pile (persistent storage) ─────────────────────────────────────────

use ts_core::repo::pile::Pile;
use ts_core::repo::{
    BlobStore, BlobStoreGet, BlobStorePut, BranchStore, Repository, Workspace,
    CommitHandle, CommitSet,
    ancestors as repo_ancestors, parents as repo_parents,
    nth_ancestors as repo_nth_ancestors, symmetric_diff as repo_symmetric_diff,
    union as repo_selector_union, intersect as repo_selector_intersect,
    difference as repo_selector_difference,
};
use ts_core::value::schemas::hash::{Blake3, Handle};
use ts_core::blob::schemas::simplearchive::SimpleArchive;

#[pyclass(name = "Pile")]
pub struct PyPile {
    path: String,
    pile: Mutex<Option<Pile<Blake3>>>,
}

#[pymethods]
impl PyPile {
    /// Open a pile file. Creates it if it doesn't exist.
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let path_str = path.to_string();
        let path = std::path::Path::new(path);
        // Create file if it doesn't exist.
        if !path.exists() {
            std::fs::File::create(path)
                .map_err(|e| PyRuntimeError::new_err(format!("create: {e}")))?;
        }
        let pile = Pile::<Blake3>::open(path)
            .map_err(|e| PyRuntimeError::new_err(format!("open: {e:?}")))?;
        Ok(PyPile { path: path_str, pile: Mutex::new(Some(pile)) })
    }

    /// Close the pile, flushing pending writes.
    fn close(&self) -> PyResult<()> {
        let pile = self.pile.lock().take()
            .ok_or_else(|| PyRuntimeError::new_err("pile already closed"))?;
        pile.close().map_err(|e| PyRuntimeError::new_err(format!("close: {e:?}")))?;
        Ok(())
    }

    /// Checkout a branch by name, returning its TribleSet.
    fn checkout(&self, branch_name: &str) -> PyResult<PyTribleSet> {
        // Take the pile out temporarily — Repository needs ownership.
        let pile = self.pile.lock().take()
            .ok_or_else(|| PyRuntimeError::new_err("pile is closed"))?;
        let signing_key = ed25519_dalek::SigningKey::from_bytes(&[0u8; 32]);
        let mut repo = Repository::new(pile, signing_key, TribleSet::new())
            .map_err(|e| PyRuntimeError::new_err(format!("repo: {e:?}")))?;
        let bid = repo.ensure_branch(branch_name, None)
            .map_err(|e| PyRuntimeError::new_err(format!("branch: {e:?}")))?;
        let mut ws = repo.pull(bid)
            .map_err(|e| PyRuntimeError::new_err(format!("pull: {e:?}")))?;
        let head = ws.head().ok_or_else(|| PyRuntimeError::new_err("branch has no commits"))?;
        let co = ws.checkout(ts_core::repo::ancestors(head))
            .map_err(|e| PyRuntimeError::new_err(format!("checkout: {e:?}")))?;
        let result = co.into_facts();
        // Put the pile back.
        *self.pile.lock() = Some(repo.into_storage());
        Ok(PyTribleSet(Mutex::new(result)))
    }

    /// List all branch names.
    fn branches(&self) -> PyResult<Vec<String>> {
        let mut guard = self.pile.lock();
        let pile = guard.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("pile is closed"))?;

        let branch_ids: Vec<Id> = pile.branches()
            .map_err(|e| PyRuntimeError::new_err(format!("branches: {e:?}")))?
            .filter_map(|r| r.ok())
            .collect();

        let mut names = Vec::new();
        for bid in branch_ids {
            let Some(head) = pile.head(bid)
                .map_err(|e| PyRuntimeError::new_err(format!("head: {e:?}")))? else { continue };

            let reader = pile.reader()
                .map_err(|e| PyRuntimeError::new_err(format!("reader: {e:?}")))?;
            let Ok(meta) = reader.get::<TribleSet, SimpleArchive>(head) else { continue };

            use ts_core::blob::schemas::longstring::LongString;
            use ts_core::value::schemas::hash::Handle;
            let name_handle = find!(
                h: Value<Handle<Blake3, LongString>>,
                pattern!(&meta, [{ _?e @ ts_core::metadata::name: ?h }])
            ).next();
            let Some(nh) = name_handle else { continue };
            let Ok(name_view) = reader.get::<anybytes::View<str>, LongString>(nh) else { continue };
            names.push(name_view.as_ref().to_string());
        }
        Ok(names)
    }

    /// Commit a TribleSet to a branch. Creates the branch if it doesn't exist.
    fn commit(&self, branch_name: &str, data: &PyTribleSet) -> PyResult<()> {
        let pile = self.pile.lock().take()
            .ok_or_else(|| PyRuntimeError::new_err("pile is closed"))?;
        let signing_key = ed25519_dalek::SigningKey::from_bytes(&[0u8; 32]);
        let mut repo = Repository::new(pile, signing_key, TribleSet::new())
            .map_err(|e| PyRuntimeError::new_err(format!("repo: {e:?}")))?;
        let bid = repo.ensure_branch(branch_name, None)
            .map_err(|e| PyRuntimeError::new_err(format!("branch: {e:?}")))?;
        let mut ws = repo.pull(bid)
            .map_err(|e| PyRuntimeError::new_err(format!("pull: {e:?}")))?;

        // Create a commit with the data.
        let data_set = data.0.lock().clone();
        ws.commit(data_set, "python commit");
        repo.push(&mut ws)
            .map_err(|_| PyRuntimeError::new_err("push failed"))?;

        *self.pile.lock() = Some(repo.into_storage());
        Ok(())
    }

    /// Read a blob by its handle value. Returns bytes or None.
    fn get_blob(&self, py: Python<'_>, handle: &PyValue) -> PyResult<Option<Py<PyBytes>>> {
        let mut guard = self.pile.lock();
        let pile = guard.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("pile is closed"))?;
        let reader = pile.reader()
            .map_err(|e| PyRuntimeError::new_err(format!("reader: {e:?}")))?;
        use ts_core::blob::schemas::UnknownBlob;
        use ts_core::value::schemas::hash::Handle;
        let handle_val = Value::<Handle<Blake3, UnknownBlob>>::new(handle.value);
        match reader.get::<anybytes::Bytes, UnknownBlob>(handle_val) {
            Ok(bytes) => Ok(Some(PyBytes::new(py, bytes.as_ref()).into())),
            Err(_) => Ok(None),
        }
    }

    /// Store bytes as a blob, returning its blake3 handle.
    fn put_blob(&self, data: &[u8]) -> PyResult<PyValue> {
        let mut guard = self.pile.lock();
        let pile = guard.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("pile is closed"))?;
        use ts_core::blob::schemas::UnknownBlob;
        use ts_core::value::schemas::hash::Handle;
        let blob = ts_core::blob::Blob::<UnknownBlob>::new(data.to_vec().into());
        let handle: Value<Handle<Blake3, UnknownBlob>> = pile.put(blob)
            .map_err(|e| PyRuntimeError::new_err(format!("put: {e:?}")))?;
        Ok(PyValue {
            value: handle.raw,
            _value_schema: <Handle<Blake3, UnknownBlob> as ConstId>::ID,
            _blob_schema: Some(<UnknownBlob as ConstId>::ID),
        })
    }

    /// Store a string as a LongString blob, returning its handle.
    fn put_string(&self, s: &str) -> PyResult<PyValue> {
        let mut guard = self.pile.lock();
        let pile = guard.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("pile is closed"))?;
        use ts_core::blob::schemas::longstring::LongString;
        use ts_core::value::schemas::hash::Handle;
        let blob = ts_core::blob::Blob::<LongString>::new(s.to_string().into());
        let handle: Value<Handle<Blake3, LongString>> = pile.put(blob)
            .map_err(|e| PyRuntimeError::new_err(format!("put: {e:?}")))?;
        Ok(PyValue {
            value: handle.raw,
            _value_schema: <Handle<Blake3, LongString> as ConstId>::ID,
            _blob_schema: Some(<LongString as ConstId>::ID),
        })
    }

    /// Read a LongString blob by its handle, returning as string.
    fn get_string(&self, handle: &PyValue) -> PyResult<Option<String>> {
        let mut guard = self.pile.lock();
        let pile = guard.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("pile is closed"))?;
        let reader = pile.reader()
            .map_err(|e| PyRuntimeError::new_err(format!("reader: {e:?}")))?;
        use ts_core::blob::schemas::longstring::LongString;
        use ts_core::value::schemas::hash::Handle;
        let handle_val = Value::<Handle<Blake3, LongString>>::new(handle.value);
        match reader.get::<anybytes::View<str>, LongString>(handle_val) {
            Ok(view) => Ok(Some(view.as_ref().to_string())),
            Err(_) => Ok(None),
        }
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(&self, _exc_type: Option<&Bound<'_, PyAny>>, _exc_val: Option<&Bound<'_, PyAny>>, _exc_tb: Option<&Bound<'_, PyAny>>) -> PyResult<bool> {
        if let Some(pile) = self.pile.lock().take() {
            let _ = pile.close();
        }
        Ok(false)
    }

    fn __repr__(&self) -> String {
        if self.pile.lock().is_some() {
            format!("Pile({:?})", self.path)
        } else {
            format!("Pile({:?}, closed)", self.path)
        }
    }
}

// ── Repository / Workspace / CommitSet / Checkout ─────────────────────

type PileBlake3 = Pile<Blake3>;

/// A Repository wraps a Pile with branch/commit/workspace operations.
#[pyclass(name = "Repository")]
pub struct PyRepository {
    path: String,
    repo: Mutex<Option<Repository<PileBlake3>>>,
}

#[pymethods]
impl PyRepository {
    /// Open or create a Repository at the given pile path.
    ///
    /// `signing_key`: optional 32-byte ed25519 signing key. Defaults to a
    /// dummy zero key (commits will be unsigned).
    #[new]
    #[pyo3(signature = (path, signing_key=None))]
    fn new(path: &str, signing_key: Option<&[u8]>) -> PyResult<Self> {
        let path_str = path.to_string();
        let path_obj = std::path::Path::new(path);
        if !path_obj.exists() {
            std::fs::File::create(path_obj)
                .map_err(|e| PyRuntimeError::new_err(format!("create: {e}")))?;
        }
        let pile = Pile::<Blake3>::open(path_obj)
            .map_err(|e| PyRuntimeError::new_err(format!("open: {e:?}")))?;
        let key_bytes: [u8; 32] = match signing_key {
            Some(b) => b.try_into()
                .map_err(|_| PyValueError::new_err("signing key must be exactly 32 bytes"))?,
            None => [0u8; 32],
        };
        let key = ed25519_dalek::SigningKey::from_bytes(&key_bytes);
        let repo = Repository::new(pile, key, TribleSet::new())
            .map_err(|e| PyRuntimeError::new_err(format!("repository: {e:?}")))?;
        Ok(PyRepository { path: path_str, repo: Mutex::new(Some(repo)) })
    }

    /// Pull a workspace for the given branch (creates the branch if missing).
    fn pull(&self, branch_name: &str) -> PyResult<PyWorkspace> {
        let mut guard = self.repo.lock();
        let repo = guard.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("repository is closed"))?;
        let bid = repo.ensure_branch(branch_name, None)
            .map_err(|e| PyRuntimeError::new_err(format!("ensure_branch: {e:?}")))?;
        let ws = repo.pull(bid)
            .map_err(|e| PyRuntimeError::new_err(format!("pull: {e:?}")))?;
        Ok(PyWorkspace { ws: Mutex::new(Some(ws)) })
    }

    /// Pull a workspace by branch ID directly.
    fn pull_by_id(&self, branch_id: &PyId) -> PyResult<PyWorkspace> {
        let mut guard = self.repo.lock();
        let repo = guard.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("repository is closed"))?;
        let ws = repo.pull(branch_id.0)
            .map_err(|e| PyRuntimeError::new_err(format!("pull: {e:?}")))?;
        Ok(PyWorkspace { ws: Mutex::new(Some(ws)) })
    }

    /// Create a new named branch (errors if it already exists).
    /// `from_commit` optionally initializes the branch at a specific commit.
    /// Returns the new branch's id.
    #[pyo3(signature = (branch_name, from_commit=None))]
    fn create_branch(&self, branch_name: &str, from_commit: Option<&PyValue>) -> PyResult<PyId> {
        let mut guard = self.repo.lock();
        let repo = guard.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("repository is closed"))?;
        let commit = from_commit.map(|v| Value::<Handle<Blake3, SimpleArchive>>::new(v.value));
        let bid = repo.create_branch(branch_name, commit)
            .map_err(|e| PyRuntimeError::new_err(format!("create_branch: {e:?}")))?;
        Ok(PyId(*bid))
    }

    /// Push a workspace to the repo. Auto-retries on conflict by merging.
    fn push(&self, workspace: &PyWorkspace) -> PyResult<()> {
        let mut guard = self.repo.lock();
        let repo = guard.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("repository is closed"))?;
        let mut ws_guard = workspace.ws.lock();
        let ws = ws_guard.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("workspace is consumed"))?;
        repo.push(ws).map_err(|e| PyRuntimeError::new_err(format!("push: {e:?}")))?;
        Ok(())
    }

    /// Single-attempt CAS push.
    /// Returns None on success, or a conflict workspace if the branch advanced.
    /// The conflict workspace has the new branch state — merge your changes
    /// into it and try_push again.
    fn try_push(&self, workspace: &PyWorkspace) -> PyResult<Option<PyWorkspace>> {
        let mut guard = self.repo.lock();
        let repo = guard.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("repository is closed"))?;
        let mut ws_guard = workspace.ws.lock();
        let ws = ws_guard.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("workspace is consumed"))?;
        let conflict = repo.try_push(ws)
            .map_err(|e| PyRuntimeError::new_err(format!("try_push: {e:?}")))?;
        Ok(conflict.map(|cws| PyWorkspace { ws: Mutex::new(Some(cws)) }))
    }

    /// List all branch IDs in the repository.
    fn branch_ids(&self) -> PyResult<Vec<PyId>> {
        let mut guard = self.repo.lock();
        let repo = guard.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("repository is closed"))?;
        // Repository doesn't expose branches directly; use the storage.
        let storage = repo.storage_mut();
        let bids: Vec<Id> = storage.branches()
            .map_err(|e| PyRuntimeError::new_err(format!("branches: {e:?}")))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(bids.into_iter().map(PyId).collect())
    }

    /// List all branches as (name, id) tuples. Branches without names are skipped.
    fn branches(&self) -> PyResult<Vec<(String, PyId)>> {
        let mut guard = self.repo.lock();
        let repo = guard.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("repository is closed"))?;
        let storage = repo.storage_mut();
        let bids: Vec<Id> = storage.branches()
            .map_err(|e| PyRuntimeError::new_err(format!("branches: {e:?}")))?
            .filter_map(|r| r.ok())
            .collect();
        let mut result = Vec::new();
        for bid in bids {
            let Some(meta_handle) = storage.head(bid)
                .map_err(|e| PyRuntimeError::new_err(format!("head: {e:?}")))? else { continue };
            let reader = storage.reader()
                .map_err(|e| PyRuntimeError::new_err(format!("reader: {e:?}")))?;
            let Ok(meta) = reader.get::<TribleSet, SimpleArchive>(meta_handle) else { continue };
            use ts_core::blob::schemas::longstring::LongString;
            let name_handle = find!(
                h: Value<Handle<Blake3, LongString>>,
                pattern!(&meta, [{ _?e @ ts_core::metadata::name: ?h }])
            ).next();
            let Some(nh) = name_handle else { continue };
            let Ok(name_view) = reader.get::<anybytes::View<str>, LongString>(nh) else { continue };
            result.push((name_view.as_ref().to_string(), PyId(bid)));
        }
        Ok(result)
    }

    /// Get the current head commit handle of a branch by name. Returns None if the branch has no commits.
    fn branch_head(&self, branch_name: &str) -> PyResult<Option<PyValue>> {
        let mut guard = self.repo.lock();
        let repo = guard.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("repository is closed"))?;
        let bid = repo.ensure_branch(branch_name, None)
            .map_err(|e| PyRuntimeError::new_err(format!("ensure_branch: {e:?}")))?;
        let ws = repo.pull(bid)
            .map_err(|e| PyRuntimeError::new_err(format!("pull: {e:?}")))?;
        Ok(ws.head().map(|h| PyValue {
            value: h.raw,
            _value_schema: <Handle<Blake3, SimpleArchive> as ConstId>::ID,
            _blob_schema: Some(<SimpleArchive as ConstId>::ID),
        }))
    }

    fn close(&self) -> PyResult<()> {
        let repo = self.repo.lock().take()
            .ok_or_else(|| PyRuntimeError::new_err("already closed"))?;
        let pile = repo.into_storage();
        pile.close().map_err(|e| PyRuntimeError::new_err(format!("close: {e:?}")))?;
        Ok(())
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> { slf }

    fn __exit__(&self, _et: Option<&Bound<'_, PyAny>>, _ev: Option<&Bound<'_, PyAny>>, _tb: Option<&Bound<'_, PyAny>>) -> PyResult<bool> {
        if let Some(repo) = self.repo.lock().take() {
            let _ = repo.into_storage().close();
        }
        Ok(false)
    }

    fn __repr__(&self) -> String {
        if self.repo.lock().is_some() {
            format!("Repository({:?})", self.path)
        } else {
            format!("Repository({:?}, closed)", self.path)
        }
    }
}

/// A Workspace is a local mutable view of a branch with staged commits.
#[pyclass(name = "Workspace")]
pub struct PyWorkspace {
    ws: Mutex<Option<Workspace<PileBlake3>>>,
}

#[pymethods]
impl PyWorkspace {
    /// The current head commit handle (None if no commits).
    #[getter]
    fn head(&self) -> PyResult<Option<PyValue>> {
        let guard = self.ws.lock();
        let ws = guard.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("workspace is consumed"))?;
        Ok(ws.head().map(|h| PyValue {
            value: h.raw,
            _value_schema: <Handle<Blake3, SimpleArchive> as ConstId>::ID,
            _blob_schema: Some(<SimpleArchive as ConstId>::ID),
        }))
    }

    /// The branch ID this workspace is for.
    #[getter]
    fn branch_id(&self) -> PyResult<PyId> {
        let guard = self.ws.lock();
        let ws = guard.as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("workspace is consumed"))?;
        Ok(PyId(ws.branch_id()))
    }

    /// Stage a commit with the given facts and message.
    #[pyo3(signature = (facts, message=""))]
    fn commit(&self, facts: &PyTribleSet, message: &str) -> PyResult<()> {
        let mut guard = self.ws.lock();
        let ws = guard.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("workspace is consumed"))?;
        let data = facts.0.lock().clone();
        ws.commit(data, message);
        Ok(())
    }

    /// Merge another workspace into this one. Creates a merge commit with
    /// both heads as parents and copies all staged blobs.
    fn merge(&self, other: &PyWorkspace) -> PyResult<Option<PyValue>> {
        let mut self_guard = self.ws.lock();
        let mut other_guard = other.ws.lock();
        let ws = self_guard.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("workspace is consumed"))?;
        let other_ws = other_guard.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("other workspace is consumed"))?;
        let handle = ws.merge(other_ws)
            .map_err(|e| PyRuntimeError::new_err(format!("merge: {e:?}")))?;
        Ok(Some(PyValue {
            value: handle.raw,
            _value_schema: <Handle<Blake3, SimpleArchive> as ConstId>::ID,
            _blob_schema: Some(<SimpleArchive as ConstId>::ID),
        }))
    }

    /// Get all facts reachable from the workspace's head as a TribleSet.
    /// Convenience for `checkout(ancestors(head)).facts`.
    #[getter]
    fn facts(&self) -> PyResult<PyTribleSet> {
        let mut guard = self.ws.lock();
        let ws = guard.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("workspace is consumed"))?;
        let Some(head) = ws.head() else {
            return Ok(PyTribleSet(Mutex::new(TribleSet::new())));
        };
        let co = ws.checkout(repo_ancestors(head))
            .map_err(|e| PyRuntimeError::new_err(format!("checkout: {e:?}")))?;
        Ok(PyTribleSet(Mutex::new(co.into_facts())))
    }

    /// Checkout the facts at the given commits (handles, CommitSet, or list).
    fn checkout(&self, spec: &Bound<'_, PyAny>) -> PyResult<PyCheckout> {
        let mut guard = self.ws.lock();
        let ws = guard.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("workspace is consumed"))?;
        let commits = pyobj_to_commitset(spec)?;
        let co = ws.checkout(commits)
            .map_err(|e| PyRuntimeError::new_err(format!("checkout: {e:?}")))?;
        Ok(PyCheckout {
            facts: co.facts().clone(),
            commits: co.commits(),
        })
    }

    /// Compute all ancestors of the given commit (or workspace head).
    #[pyo3(signature = (commit=None))]
    fn ancestors(&self, commit: Option<&PyValue>) -> PyResult<PyCommitSet> {
        let mut guard = self.ws.lock();
        let ws = guard.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("workspace is consumed"))?;
        let start = match commit {
            Some(v) => Value::<Handle<Blake3, SimpleArchive>>::new(v.value),
            None => ws.head().ok_or_else(|| PyRuntimeError::new_err("no head"))?,
        };
        let co = ws.checkout(repo_ancestors(start))
            .map_err(|e| PyRuntimeError::new_err(format!("checkout: {e:?}")))?;
        Ok(PyCommitSet { set: co.commits() })
    }

    /// Compute the direct parents of the given commit (or workspace head).
    #[pyo3(signature = (commit=None))]
    fn parents(&self, commit: Option<&PyValue>) -> PyResult<PyCommitSet> {
        let mut guard = self.ws.lock();
        let ws = guard.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("workspace is consumed"))?;
        let start = match commit {
            Some(v) => Value::<Handle<Blake3, SimpleArchive>>::new(v.value),
            None => ws.head().ok_or_else(|| PyRuntimeError::new_err("no head"))?,
        };
        let co = ws.checkout(repo_parents(start))
            .map_err(|e| PyRuntimeError::new_err(format!("checkout: {e:?}")))?;
        Ok(PyCommitSet { set: co.commits() })
    }

    /// Walk back N parent steps from the given commit (or head).
    #[pyo3(signature = (n, commit=None))]
    fn nth_ancestors(&self, n: usize, commit: Option<&PyValue>) -> PyResult<PyCommitSet> {
        let mut guard = self.ws.lock();
        let ws = guard.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("workspace is consumed"))?;
        let start = match commit {
            Some(v) => Value::<Handle<Blake3, SimpleArchive>>::new(v.value),
            None => ws.head().ok_or_else(|| PyRuntimeError::new_err("no head"))?,
        };
        let co = ws.checkout(repo_nth_ancestors(start, n))
            .map_err(|e| PyRuntimeError::new_err(format!("checkout: {e:?}")))?;
        Ok(PyCommitSet { set: co.commits() })
    }

    /// Look up a blob by handle (UnknownBlob — raw bytes).
    fn get_blob(&self, py: Python<'_>, handle: &PyValue) -> PyResult<Option<Py<PyBytes>>> {
        let mut guard = self.ws.lock();
        let ws = guard.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("workspace is consumed"))?;
        use ts_core::blob::schemas::UnknownBlob;
        let handle_val = Value::<Handle<Blake3, UnknownBlob>>::new(handle.value);
        match ws.get::<anybytes::Bytes, UnknownBlob>(handle_val) {
            Ok(bytes) => Ok(Some(PyBytes::new(py, bytes.as_ref()).into())),
            Err(_) => Ok(None),
        }
    }

    /// Look up a LongString blob by handle.
    fn get_string(&self, handle: &PyValue) -> PyResult<Option<String>> {
        let mut guard = self.ws.lock();
        let ws = guard.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("workspace is consumed"))?;
        use ts_core::blob::schemas::longstring::LongString;
        let handle_val = Value::<Handle<Blake3, LongString>>::new(handle.value);
        match ws.get::<anybytes::View<str>, LongString>(handle_val) {
            Ok(view) => Ok(Some(view.as_ref().to_string())),
            Err(_) => Ok(None),
        }
    }

    /// Get the message string for a specific commit handle, or None if it has no message.
    fn commit_message(&self, commit: &PyValue) -> PyResult<Option<String>> {
        let mut guard = self.ws.lock();
        let ws = guard.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("workspace is consumed"))?;
        let handle = Value::<Handle<Blake3, SimpleArchive>>::new(commit.value);
        let meta: TribleSet = match ws.get::<TribleSet, SimpleArchive>(handle) {
            Ok(m) => m,
            Err(_) => return Ok(None),
        };
        use ts_core::blob::schemas::longstring::LongString;
        let msg_handle = find!(
            h: Value<Handle<Blake3, LongString>>,
            pattern!(&meta, [{ _?e @ ts_core::repo::message: ?h }])
        ).next();
        let Some(mh) = msg_handle else { return Ok(None) };
        match ws.get::<anybytes::View<str>, LongString>(mh) {
            Ok(view) => Ok(Some(view.as_ref().to_string())),
            Err(_) => Ok(None),
        }
    }

    /// Get the parent commit handles of a commit.
    fn commit_parents(&self, commit: &PyValue) -> PyResult<Vec<PyValue>> {
        let mut guard = self.ws.lock();
        let ws = guard.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("workspace is consumed"))?;
        let handle = Value::<Handle<Blake3, SimpleArchive>>::new(commit.value);
        let meta: TribleSet = match ws.get::<TribleSet, SimpleArchive>(handle) {
            Ok(m) => m,
            Err(_) => return Ok(vec![]),
        };
        let parents: Vec<_> = find!(
            p: Value<Handle<Blake3, SimpleArchive>>,
            pattern!(&meta, [{ _?e @ ts_core::repo::parent: ?p }])
        ).collect();
        Ok(parents.into_iter().map(|p| PyValue {
            value: p.raw,
            _value_schema: <Handle<Blake3, SimpleArchive> as ConstId>::ID,
            _blob_schema: Some(<SimpleArchive as ConstId>::ID),
        }).collect())
    }

    /// Get the raw commit metadata TribleSet — exposes all attributes
    /// (parents, message, content, signed_by, timestamp, etc.) for custom queries.
    fn commit_metadata(&self, commit: &PyValue) -> PyResult<PyTribleSet> {
        let mut guard = self.ws.lock();
        let ws = guard.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("workspace is consumed"))?;
        let handle = Value::<Handle<Blake3, SimpleArchive>>::new(commit.value);
        let meta: TribleSet = ws.get::<TribleSet, SimpleArchive>(handle)
            .map_err(|e| PyRuntimeError::new_err(format!("get commit: {e:?}")))?;
        Ok(PyTribleSet(Mutex::new(meta)))
    }

    /// Get the content facts of a single commit.
    fn commit_facts(&self, commit: &PyValue) -> PyResult<PyTribleSet> {
        let mut guard = self.ws.lock();
        let ws = guard.as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("workspace is consumed"))?;
        let handle = Value::<Handle<Blake3, SimpleArchive>>::new(commit.value);
        let meta: TribleSet = ws.get::<TribleSet, SimpleArchive>(handle)
            .map_err(|e| PyRuntimeError::new_err(format!("get commit: {e:?}")))?;
        let content_handle = find!(
            c: Value<Handle<Blake3, SimpleArchive>>,
            pattern!(&meta, [{ _?e @ ts_core::repo::content: ?c }])
        ).next();
        let Some(ch) = content_handle else {
            return Ok(PyTribleSet(Mutex::new(TribleSet::new())));
        };
        let content: TribleSet = ws.get::<TribleSet, SimpleArchive>(ch)
            .map_err(|e| PyRuntimeError::new_err(format!("get content: {e:?}")))?;
        Ok(PyTribleSet(Mutex::new(content)))
    }

    fn __repr__(&self) -> String {
        let guard = self.ws.lock();
        match guard.as_ref() {
            Some(ws) => {
                let head_str = ws.head().map(|h| {
                    format!("{}", &hex::encode(&h.raw[..4]))
                }).unwrap_or_else(|| "None".to_string());
                format!("Workspace(head={head_str})")
            }
            None => "Workspace(consumed)".to_string(),
        }
    }
}

/// A set of commit handles. Supports union, intersection, difference operators.
#[pyclass(name = "CommitSet")]
pub struct PyCommitSet {
    set: CommitSet,
}

#[pymethods]
impl PyCommitSet {
    fn __len__(&self) -> usize {
        self.set.len() as usize
    }

    fn __bool__(&self) -> bool {
        self.set.len() > 0
    }

    fn __iter__(slf: PyRef<'_, Self>, py: Python<'_>) -> Py<PyCommitSetIter> {
        let handles: Vec<PyValue> = slf.set.iter().map(|raw| PyValue {
            value: *raw,
            _value_schema: <Handle<Blake3, SimpleArchive> as ConstId>::ID,
            _blob_schema: Some(<SimpleArchive as ConstId>::ID),
        }).collect();
        Py::new(py, PyCommitSetIter {
            handles: Mutex::new(handles.into_iter().collect::<std::collections::VecDeque<_>>()),
        }).unwrap()
    }

    fn __or__(&self, other: &Self) -> Self {
        let mut s = self.set.clone();
        s.union(other.set.clone());
        PyCommitSet { set: s }
    }

    fn __and__(&self, other: &Self) -> Self {
        PyCommitSet { set: self.set.intersect(&other.set) }
    }

    fn __sub__(&self, other: &Self) -> Self {
        PyCommitSet { set: self.set.difference(&other.set) }
    }

    fn __contains__(&self, handle: &PyValue) -> bool {
        // CommitSet is a PATCH; check membership via has_prefix
        self.set.has_prefix(&handle.value)
    }

    fn __repr__(&self) -> String {
        format!("CommitSet({} commits)", self.set.len())
    }
}

#[pyclass(name = "CommitSetIter")]
pub struct PyCommitSetIter {
    handles: Mutex<std::collections::VecDeque<PyValue>>,
}

#[pymethods]
impl PyCommitSetIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> { slf }
    fn __next__(&self) -> Option<PyValue> {
        self.handles.lock().pop_front()
    }
}

/// The result of a checkout: facts (TribleSet) + commits (CommitSet).
#[pyclass(name = "Checkout")]
pub struct PyCheckout {
    facts: TribleSet,
    commits: CommitSet,
}

#[pymethods]
impl PyCheckout {
    #[getter]
    fn facts(&self) -> PyTribleSet {
        PyTribleSet(Mutex::new(self.facts.clone()))
    }

    #[getter]
    fn commits(&self) -> PyCommitSet {
        PyCommitSet { set: self.commits.clone() }
    }

    fn __repr__(&self) -> String {
        format!("Checkout({} tribles, {} commits)", self.facts.len(), self.commits.len())
    }
}

/// Convert a Python object to a CommitSet (accepts: PyValue/handle, PyCommitSet, list of either).
fn pyobj_to_commitset(obj: &Bound<'_, PyAny>) -> PyResult<CommitSet> {
    if let Ok(cs) = obj.extract::<PyRef<PyCommitSet>>() {
        return Ok(cs.set.clone());
    }
    if let Ok(v) = obj.extract::<PyRef<PyValue>>() {
        let mut set = CommitSet::new();
        set.insert(&ts_core::patch::Entry::new(&v.value));
        return Ok(set);
    }
    if let Ok(list) = obj.downcast::<pyo3::types::PyList>() {
        let mut set = CommitSet::new();
        for item in list.iter() {
            let v = item.extract::<PyRef<PyValue>>()?;
            set.insert(&ts_core::patch::Entry::new(&v.value));
        }
        return Ok(set);
    }
    Err(PyTypeError::new_err("expected Value, CommitSet, or list of Values"))
}

pub struct InnerVariable {
    index: usize,
    name: String,

    _value_schema: Option<Id>,
    _blob_schema: Option<Id>,
}

#[pyclass(frozen, name = "Variable")]
pub struct PyVariable(RwLock<InnerVariable>);

// class Variable:
//     def __init__(self, index, name=None):
//         self.index = index
//         self.name = name
//         self.value_schema = None
//         self.blob_schema = None

//     def annotate_schemas(self, value_schema, blob_schema = None):
//         if self.value_schema is None:
//             self.value_schema = value_schema
//             self.blob_schema = blob_schema
//         else:
//             if self.value_schema != value_schema:
//                 raise TypeError(
//                     "variable"
//                     + name
//                     + " annotated with conflicting value schemas"
//                     + str(self.schema)
//                     + " and "
//                     + str(schema)
//                 )
//             if self.blob_schema != blob_schema:
//                 raise TypeError(
//                     "variable"
//                     + name
//                     + " annotated with conflicting blob schemas"
//                     + str(self.schema)
//                     + " and "
//                     + str(schema)
//                 )

#[pymethods]
impl PyVariable {
    /// Get the variable's index (for use with `constant()`).
    pub fn _index(&self) -> usize {
        self.0.read().index
    }

    #[new]
    pub fn new(index: usize, name: String) -> Self {
        PyVariable(RwLock::new(InnerVariable {
            index,
            name,
            _value_schema: None,
            _blob_schema: None,
        }))
    }

    #[pyo3(signature = (value_schema, blob_schema=None))]
    pub fn annotate_schemas(&self, value_schema: PyId, blob_schema: Option<&PyId>) {
        let mut variable = self.0.write();
        variable._value_schema = Some(value_schema.0);
        variable._blob_schema = blob_schema.map(|id| id.0);
    }
}

pub struct InnerVariableContext {
    variables: Vec<Py<PyVariable>>,
}

#[pyclass(frozen, name = "VariableContext")]
pub struct PyVariableContext(RwLock<InnerVariableContext>);

#[pymethods]
impl PyVariableContext {
    #[new]
    pub fn new() -> Self {
        PyVariableContext(RwLock::new(InnerVariableContext {
            variables: Vec::new(),
        }))
    }

    #[pyo3(signature = (name))]
    pub fn fresh_variable(&self, py: Python<'_>, name: String) -> Py<PyVariable> {
        let mut variable_context = self.0.write();

        let next_index = variable_context.variables.len();

        let variable = Py::new(py, PyVariable::new(next_index, name)).unwrap();
        variable_context.variables.push(variable.clone_ref(py));

        return variable;
    }

    pub fn check_schemas(&self) -> PyResult<()> {
        let variable_context = self.0.read();
        for v in &variable_context.variables {
            let variable = v.get().0.read();

            if variable._value_schema.is_none() {
                let name = &variable.name;
                let msg = format!("missing value schema for variable {name}");
                return Err(PyTypeError::new_err(msg));
            }
        }
        Ok(())
    }
}

#[pyclass(frozen, name = "Query")]
pub struct PyQuery {
    query: Mutex<
        Query<
            Arc<dyn Constraint<'static> + Send + Sync>,
            Box<dyn Fn(&Binding) -> Option<Vec<PyValue>> + Send>,
            Vec<PyValue>,
        >,
    >,
}

#[pyclass(frozen)]
pub struct PyConstraint {
    constraint: Arc<dyn Constraint<'static> + Send + Sync>,
}

/// Build a constraint for the intersection of the provided constraints.
#[pyfunction]
pub fn constant(index: usize, constant: &Bound<'_, PyValue>) -> PyConstraint {
    let constraint = Arc::new(ConstantConstraint::new(
        Variable::<UnknownValue>::new(index),
        Value::<UnknownValue>::new(constant.get().value),
    ));

    PyConstraint { constraint }
}

/// Build a constraint for the intersection of the provided constraints.
#[pyfunction]
#[pyo3(name = "intersect")]
pub fn py_intersect(constraints: Vec<Py<PyConstraint>>) -> PyConstraint {
    let constraints = constraints
        .iter()
        .map(|py| py.get().constraint.clone())
        .collect();
    let constraint = Arc::new(IntersectionConstraint::new(constraints));

    PyConstraint { constraint }
}

/// Constrain two variables to have the same value.
#[pyfunction]
pub fn equality(a: &PyVariable, b: &PyVariable) -> PyConstraint {
    PyConstraint {
        constraint: Arc::new(EqualityConstraint::new(
            a.0.read().index,
            b.0.read().index,
        )),
    }
}

/// Build a constraint for the union (OR) of the provided constraints.
/// All constraints must bind the same variables.
#[pyfunction]
#[pyo3(name = "union")]
pub fn py_union(constraints: Vec<Py<PyConstraint>>) -> PyConstraint {
    let constraints: Vec<_> = constraints
        .iter()
        .map(|py| py.get().constraint.clone())
        .collect();
    let constraint = Arc::new(UnionConstraint::new(constraints));
    PyConstraint { constraint }
}

/// Build a path constraint for graph traversal.
///
/// ops is a list of path operations (postfix encoded):
///   ("attr", Id)    — single attribute hop
///   ("concat",)     — compose two preceding sub-expressions
///   ("union",)      — match either of two preceding sub-expressions
///   ("star",)       — zero or more repetitions
///   ("plus",)       — one or more repetitions
#[pyfunction]
#[pyo3(name = "path")]
pub fn py_path(
    kb: &PyTribleSet,
    start: &PyVariable,
    end: &PyVariable,
    ops: Vec<(String, Option<PyId>)>,
) -> PyResult<PyConstraint> {
    let ops: Vec<PathOp> = ops.iter().map(|(name, id)| {
        match name.as_str() {
            "attr" => {
                let id = id.as_ref().expect("attr op requires an Id").0;
                PathOp::Attr(id.raw())
            }
            "concat" => PathOp::Concat,
            "union" => PathOp::Union,
            "star" => PathOp::Star,
            "plus" => PathOp::Plus,
            other => panic!("unknown path op: {other}"),
        }
    }).collect();
    Ok(PyConstraint {
        constraint: Arc::new(RegularPathConstraint::new(
            kb.0.lock().clone(),
            Variable::<valueschemas::GenId>::new(start.0.read().index),
            Variable::<valueschemas::GenId>::new(end.0.read().index),
            &ops,
        )),
    })
}

/// Constrain a variable to values in the given set.
#[pyfunction]
pub fn value_set(variable: &PyVariable, values: Vec<Py<PyValue>>) -> PyConstraint {
    use std::collections::HashSet;
    let index = variable.0.read().index;
    let raw_values: HashSet<RawValue> = values.iter().map(|v| v.get().value).collect();
    PyConstraint {
        constraint: Arc::new(RawValueSetConstraint { index, values: raw_values }),
    }
}

/// Hide variables from the outer query — like Rust's ignore!() macro.
#[pyfunction]
#[pyo3(name = "ignore")]
pub fn py_ignore(variables: Vec<Py<PyVariable>>, constraint: &PyConstraint) -> PyConstraint {
    let mut ignored = ts_core::query::VariableSet::new_empty();
    for v in &variables {
        ignored.set(v.get().0.read().index);
    }
    let inner = constraint.constraint.clone();
    PyConstraint {
        constraint: Arc::new(PyIgnoreConstraint { ignored, inner }),
    }
}

struct PyIgnoreConstraint {
    ignored: ts_core::query::VariableSet,
    inner: Arc<dyn Constraint<'static> + Send + Sync>,
}

impl<'a> Constraint<'a> for PyIgnoreConstraint {
    fn variables(&self) -> ts_core::query::VariableSet {
        self.inner.variables().difference(self.ignored)
    }
    fn estimate(&self, variable: VariableId, binding: &Binding) -> Option<usize> {
        self.inner.estimate(variable, binding)
    }
    fn propose(&self, variable: VariableId, binding: &Binding, proposals: &mut Vec<RawValue>) {
        self.inner.propose(variable, binding, proposals)
    }
    fn confirm(&self, variable: VariableId, binding: &Binding, proposals: &mut Vec<RawValue>) {
        self.inner.confirm(variable, binding, proposals)
    }
}

struct RawValueSetConstraint {
    index: VariableId,
    values: std::collections::HashSet<RawValue>,
}

impl<'a> Constraint<'a> for RawValueSetConstraint {
    fn variables(&self) -> ts_core::query::VariableSet {
        ts_core::query::VariableSet::new_singleton(self.index)
    }
    fn estimate(&self, variable: VariableId, _binding: &Binding) -> Option<usize> {
        if self.index == variable { Some(self.values.len()) } else { None }
    }
    fn propose(&self, variable: VariableId, _binding: &Binding, proposals: &mut Vec<RawValue>) {
        if self.index == variable {
            proposals.extend(self.values.iter().copied());
        }
    }
    fn confirm(&self, variable: VariableId, _binding: &Binding, proposals: &mut Vec<RawValue>) {
        if self.index == variable {
            proposals.retain(|v| self.values.contains(v));
        }
    }
}

/// Find solutions for the provided constraint.
#[pyfunction]
pub fn solve(projected: Vec<Py<PyVariable>>, constraint: &PyConstraint) -> PyResult<PyQuery> {
    let constraint = constraint.constraint.clone();

    let postprocessing = Box::new(move |binding: &Binding| {
        let mut vec = vec![];
        for v in &projected {
            let v = v.get();
            let value = *binding
                .get(v.0.read().index)
                .expect("constraint should contain projected variables");

            vec.push(PyValue {
                value,
                _value_schema: v
                    .0
                    .read()
                    ._value_schema
                    .expect("variable with uninitialized value schema"),
                _blob_schema: v.0.read()._blob_schema,
            });
        }
        Some(vec)
    }) as Box<dyn Fn(&Binding) -> Option<Vec<PyValue>> + Send>;

    let query = ts_core::query::Query::new(constraint, postprocessing);

    Ok(PyQuery {
        query: Mutex::new(query),
    })
}

#[pymethods]
impl PyQuery {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(&self) -> Option<Vec<PyValue>> {
        self.query.lock().next()
    }
}

/// Blake3 hash of the given bytes, returned as 32 raw bytes.
#[pyfunction]
fn py_blake3(py: Python<'_>, data: &[u8]) -> Py<PyBytes> {
    let digest = blake3::hash(data);
    PyBytes::new(py, digest.as_bytes()).into()
}

/// A schema type — wraps the ConstId from Rust.
///
/// Leaf schemas (GenId, ShortString, etc.) are module-level singletons.
/// Compound schemas are built with Handle(hash_schema, blob_schema).
#[pyclass(name = "Schema", frozen)]
#[derive(Clone)]
pub struct PySchema {
    id: Id,
    label: String,
}

#[pymethods]
impl PySchema {
    /// The schema's ConstId.
    #[getter]
    fn id(&self) -> PyId {
        PyId(self.id)
    }

    /// Derive an attribute ID from a name string and this schema.
    ///
    /// Matches Rust's Attribute::from_name exactly.
    fn _attribute_id(&self, name: &str) -> PyId {
        let field_handle = blake3::hash(name.as_bytes());
        let mut hasher = blake3::Hasher::new();
        hasher.update(field_handle.as_bytes());
        hasher.update(&self.id.raw());
        let digest = hasher.finalize();
        let bytes = digest.as_bytes();
        let mut raw = [0u8; 16];
        raw.copy_from_slice(&bytes[16..32]);
        PyId(Id::new(raw).expect("attribute_from_name produced nil ID"))
    }

    fn __repr__(&self) -> String {
        format!("Schema({}, {})", self.label, PyId(self.id).__repr__())
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.id.raw().hash(&mut h);
        h.finish()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

fn derive_handle_id(hash_id: Id, blob_id: Id) -> Id {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&hash_id.raw());
    hasher.update(&blob_id.raw());
    let digest = hasher.finalize();
    let bytes = digest.as_bytes();
    let mut raw = [0u8; 16];
    raw.copy_from_slice(&bytes[16..32]);
    Id::new(raw).expect("derived handle schema id must be non-nil")
}

/// Build a Handle schema from a hash schema and a blob schema.
#[pyfunction]
#[pyo3(name = "Handle")]
fn py_handle(hash_schema: &PySchema, blob_schema: &PySchema) -> PySchema {
    PySchema {
        id: derive_handle_id(hash_schema.id, blob_schema.id),
        label: format!("Handle<{}, {}>", hash_schema.label, blob_schema.label),
    }
}

macro_rules! schema_const {
    ($id:expr, $label:expr) => {
        PySchema { id: ts_core::id_hex!($id), label: String::from($label) }
    };
}

static SCHEMAS: LazyLock<Vec<(&str, PySchema)>> = LazyLock::new(|| vec![
    // Value schemas
    ("GenId",            schema_const!("B08EE1D45EB081E8C47618178AFE0D81", "GenId")),
    ("ShortString",      schema_const!("2D848DB0AF112DB226A6BF1A3640D019", "ShortString")),
    ("F64",              schema_const!("C80A60F4A6F2FBA5A8DB2531A923EC70", "F64")),
    ("Boolean",          schema_const!("73B414A3E25B0C0F9E4D6B0694DC33C5", "Boolean")),
    ("NsTAIInterval",    schema_const!("2170014368272A2B1B18B86B1F1F1CB5", "NsTAIInterval")),
    ("U256LE",           schema_const!("49E70B4DBD84DC7A3E0BDDABEC8A8C6E", "U256LE")),
    ("U256BE",           schema_const!("DC3CFB719B05F019FB8101A6F471A982", "U256BE")),
    ("I256LE",           schema_const!("DB94325A37D96037CBFC6941A4C3B66D", "I256LE")),
    ("I256BE",           schema_const!("CE3A7839231F1EB390E9E8E13DAED782", "I256BE")),
    ("R256LE",           schema_const!("0A9B43C5C2ECD45B257CDEFC16544358", "R256LE")),
    ("R256BE",           schema_const!("CA5EAF567171772C1FFD776E9C7C02D1", "R256BE")),
    ("F256LE",           schema_const!("D9A419D3CAA0D8E05D8DAB950F5E80F2", "F256LE")),
    ("F256BE",           schema_const!("A629176D4656928D96B155038F9F2220", "F256BE")),
    ("LineLocation",     schema_const!("DFAED173A908498CB893A076EAD3E578", "LineLocation")),
    ("ED25519PublicKey", schema_const!("69A872254E01B4C1ED36E08E40445E93", "ED25519PublicKey")),
    ("ED25519RComponent",schema_const!("995A86FFC83DB95ECEAA17E226208897", "ED25519RComponent")),
    ("ED25519SComponent",schema_const!("10D35B0B628E9E409C549D8EC1FB3598", "ED25519SComponent")),
    ("UnknownValue",     schema_const!("4EC697E8599AC79D667C722E2C8BEBF4", "UnknownValue")),
    // Hash protocol schemas
    ("Blake3",           schema_const!("4160218D6C8F620652ECFBD7FDC7BDB3", "Blake3")),
    // Blob schemas
    ("LongString",       schema_const!("8B173C65B7DB601A11E8A190BD774A79", "LongString")),
    ("SimpleArchive",    schema_const!("8F4A27C8581DADCBA1ADA8BA228069B6", "SimpleArchive")),
    ("FileBytes",        schema_const!("5DE76157AE4FDEA830019916805E80A4", "FileBytes")),
    ("WasmCode",         schema_const!("DEE50FAD0CFFA4F8FD542DD18D9B7E52", "WasmCode")),
    ("UnknownBlob",      schema_const!("EAB14005141181B0C10C4B5DD7985F8D", "UnknownBlob")),
]);

/// The `tribles` python module.
#[pymodule]
#[pyo3(name = "triblespace")]
pub fn triblespace_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTribleSet>()?;
    m.add_class::<PyId>()?;
    m.add_class::<PyIdOwner>()?;
    m.add_class::<PyValue>()?;
    m.add_class::<PyVariable>()?;
    m.add_class::<PyVariableContext>()?;
    m.add_class::<PyConstraint>()?;
    m.add_class::<PyQuery>()?;
    m.add_class::<PyPile>()?;
    m.add_class::<PyRepository>()?;
    m.add_class::<PyWorkspace>()?;
    m.add_class::<PyCommitSet>()?;
    m.add_class::<PyCommitSetIter>()?;
    m.add_class::<PyCheckout>()?;
    m.add_class::<PySchema>()?;
    m.add_function(wrap_pyfunction!(py_blake3, m)?)?;
    m.add_function(wrap_pyfunction!(py_handle, m)?)?;
    for (name, schema) in SCHEMAS.iter() {
        m.add(name, schema.clone())?;
    }
    m.add_function(wrap_pyfunction!(register_type, m)?)?;
    m.add_function(wrap_pyfunction!(register_to_value_converter, m)?)?;
    m.add_function(wrap_pyfunction!(register_from_value_converter, m)?)?;
    m.add_function(wrap_pyfunction!(register_to_blob_converter, m)?)?;
    m.add_function(wrap_pyfunction!(register_from_blob_converter, m)?)?;
    m.add_function(wrap_pyfunction!(get_value_schema, m)?)?;
    m.add_function(wrap_pyfunction!(get_blob_schema, m)?)?;
    m.add_function(wrap_pyfunction!(get_label_name_handles, m)?)?;
    m.add_function(wrap_pyfunction!(metadata_description, m)?)?;
    m.add_function(wrap_pyfunction!(constant, m)?)?;
    m.add_function(wrap_pyfunction!(py_intersect, m)?)?;
    m.add_function(wrap_pyfunction!(equality, m)?)?;
    m.add_function(wrap_pyfunction!(py_union, m)?)?;
    m.add_function(wrap_pyfunction!(py_path, m)?)?;
    m.add_function(wrap_pyfunction!(value_set, m)?)?;
    m.add_function(wrap_pyfunction!(py_ignore, m)?)?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_submodule(m)?;
    Ok(())
}
