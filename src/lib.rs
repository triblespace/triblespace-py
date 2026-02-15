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
use triblespace::metadata::metadata;
use triblespace::{
    id::IdOwner,
    prelude::*,
    query::{
        constantconstraint::ConstantConstraint, Binding, Constraint, ContainsConstraint, Query,
        TriblePattern, Variable,
    },
    value::{schemas::UnknownValue, RawValue},
};

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
    match find!((value_schema: Id),
    metadata::pattern!(&context.0.lock(), [
        {(attr_id.0) @
            attr_value_schema: value_schema}
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
    match find!((blob_schema: Id),
    metadata::pattern!(&context.0.lock(), [
        {(attr_id.0) @
            attr_blob_schema: blob_schema}
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

#[pyfunction]
pub fn get_label_names(context: &PyTribleSet) -> PyResult<HashMap<String, PyId>> {
    find!((name: String, attr_id: Id),
    metadata::pattern!(&context.0.lock(), [
        {attr_id @
            attr_name: name
        }]))
    .into_group_map()
    .into_iter()
    .map(|(name, ids)| {
        if ids.len() > 1 {
            Err(PyErr::new::<PyKeyError, _>(
                "multiple attributes with the same name",
            ))
        } else {
            Ok((name, PyId(ids[0])))
        }
    })
    .collect()
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
    PyTribleSet(Mutex::new(triblespace::metadata::metadata::description()))
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

    pub fn bytes(&self) -> Cow<[u8]> {
        (&self.value).into()
    }
}

#[pyclass(frozen, name = "TribleSet")]
pub struct PyTribleSet(Mutex<TribleSet>);

#[pymethods]
impl PyTribleSet {
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
            Box<dyn Fn(&Binding) -> Vec<PyValue> + Send>,
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
pub fn intersect(constraints: Vec<Py<PyConstraint>>) -> PyConstraint {
    let constraints = constraints
        .iter()
        .map(|py| py.get().constraint.clone())
        .collect();
    let constraint = Arc::new(IntersectionConstraint::new(constraints));

    PyConstraint { constraint }
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
        vec
    }) as Box<dyn Fn(&Binding) -> Vec<PyValue> + Send>;

    let query = triblespace::query::Query::new(constraint, postprocessing);

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

/// The `tribles` python module.
#[pymodule]
#[pyo3(name = "tribles")]
pub fn tribles_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTribleSet>()?;
    m.add_class::<PyId>()?;
    m.add_class::<PyIdOwner>()?;
    m.add_class::<PyValue>()?;
    m.add_class::<PyVariable>()?;
    m.add_class::<PyVariableContext>()?;
    m.add_class::<PyConstraint>()?;
    m.add_class::<PyQuery>()?;
    m.add_function(wrap_pyfunction!(register_type, m)?)?;
    m.add_function(wrap_pyfunction!(register_to_value_converter, m)?)?;
    m.add_function(wrap_pyfunction!(register_from_value_converter, m)?)?;
    m.add_function(wrap_pyfunction!(register_to_blob_converter, m)?)?;
    m.add_function(wrap_pyfunction!(register_from_blob_converter, m)?)?;
    m.add_function(wrap_pyfunction!(get_value_schema, m)?)?;
    m.add_function(wrap_pyfunction!(get_blob_schema, m)?)?;
    m.add_function(wrap_pyfunction!(get_label_names, m)?)?;
    m.add_function(wrap_pyfunction!(metadata_description, m)?)?;
    m.add_function(wrap_pyfunction!(constant, m)?)?;
    m.add_function(wrap_pyfunction!(intersect, m)?)?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_submodule(m)?;
    Ok(())
}
