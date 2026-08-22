"""TribleSpace — a knowledge graph in Python.

Low-level bindings from the Rust core, plus Pythonic convenience wrappers.

Quick start:

    import triblespace as ts

    # Define attributes (derived from name + schema, matching Rust)
    name   = ts.ShortString.attribute('name')
    friend = ts.GenId.attribute('friend')

    # Build entities
    alice = ts.entity({name: "Alice"})
    bob   = ts.entity({name: "Bob", friend: alice.id})

    # Add to knowledge base
    kb = ts.TribleSet()
    kb += alice + bob

    # Query
    e, n = ts.vars('entity', 'name')
    for row in ts.query([n], kb.where(e, name, n)):
        print(row[0].to_str())
"""

# Re-export everything from the native module.
from .triblespace import *
from .triblespace import (
    TribleSet, Id, IdOwner, Value, Variable, VariableContext,
    PyConstraint as Constraint, Query, Schema, Pile,
    constant, intersect, solve,
    equality, union, value_set,
    get_value_schema, get_blob_schema, get_label_name_handles,
    metadata_description,
    register_type, register_to_value_converter, register_from_value_converter,
    register_to_blob_converter, register_from_blob_converter,
    py_blake3 as blake3,
    Handle,
    # Leaf encoding singletons.
    GenId, ShortString, F64, Boolean, NsTAIInterval,
    U256LE, U256BE, I256LE, I256BE, R256LE, R256BE, F256LE, F256BE,
    LineLocation, ED25519PublicKey, ED25519RComponent, ED25519SComponent,
    UnknownInline,
    Blake3 as Blake3Schema,
    UTF8String, SimpleArchive, RawBytes, WasmCode, UnknownBlob,
)

# ── Variable + Constraint combinators ─────────────────────────────────

# Shared variable context for the module.
_var_ctx = VariableContext()
_var_cache = {}


def var(name):
    """Get or create a named query variable.

    Variables are module-global singletons by name — var('x') always
    returns the same Variable.

        e = ts.var('entity')
        n = ts.var('name')
    """
    if name not in _var_cache:
        v = _var_ctx.fresh_variable(name)
        _var_cache[name] = v
    return _var_cache[name]


def vars(*names):
    """Create multiple variables at once.

        e, n, p = ts.vars('entity', 'name', 'phase')
    """
    return tuple(var(n) for n in names)


def reset_vars():
    """Reset the variable context. Call between independent queries."""
    global _var_ctx, _var_cache
    _var_ctx = VariableContext()
    _var_cache = {}


class ConstraintBuilder:
    """Wraps a PyConstraint with & (and) and | (or) operators.

        c = kb.where(e, name, n) & kb.where(e, phase, p)
    """
    __slots__ = ('_inner',)

    def __init__(self, inner):
        self._inner = inner

    def __and__(self, other):
        if isinstance(other, ConstraintBuilder):
            return ConstraintBuilder(intersect([self._inner, other._inner]))
        return NotImplemented

    def __or__(self, other):
        """Union (OR). Both branches must bind exactly the same variables.
        Use explicit variables (not string constants) in both branches."""
        if isinstance(other, ConstraintBuilder):
            try:
                return ConstraintBuilder(union([self._inner, other._inner]))
            except Exception as e:
                raise ValueError(
                    "OR requires both branches to bind the same variables. "
                    "Use explicit variables instead of constants in where()."
                ) from e
        return NotImplemented

    def __repr__(self):
        return f"Constraint(...)"


class Fragment:
    """A set of facts with exported entity Ids.

    Returned by entity(). Composable with + and +=.

        alice = ts.entity({name: "Alice"})
        bob = ts.entity({name: "Bob"})

        alice.id          # single entity's Id
        alice.ids         # list of all exported Ids

        combined = alice + bob
        combined.ids      # [alice.id, bob.id]

        kb = kb + combined  # unions all facts into kb
    """
    __slots__ = ('ids', 'facts')

    def __init__(self, ids, facts):
        self.ids = ids if isinstance(ids, list) else [ids]
        self.facts = facts

    @property
    def id(self):
        """The root entity Id (first exported Id)."""
        return self.ids[0] if self.ids else None

    def __add__(self, other):
        if isinstance(other, Fragment):
            return Fragment(self.ids + other.ids, self.facts + other.facts)
        return NotImplemented

    def __len__(self):
        return len(self.facts)

    def __repr__(self):
        n = len(self.ids)
        ids_str = self.ids[0].to_hex()[:8] if n == 1 else f"{n} entities"
        return f"Fragment({ids_str}..., {len(self.facts)} tribles)"


# ── TribleSet.where() for pattern building ────────────────────────────

_hidden_counter = [0]

def _ensure_var(spec):
    """Convert a spec to a Variable, creating hidden vars for constants."""
    if isinstance(spec, Variable):
        return spec, []
    # It's a constant — create a hidden variable + constant constraint.
    _hidden_counter[0] += 1
    hv = _var_ctx.fresh_variable(f"_h{_hidden_counter[0]}")
    if isinstance(spec, Attribute):
        hv.annotate_schemas(spec.schema.id if spec.schema else GenId.id, None)
        return hv, [constant(hv._index(), Value.from_id(spec.id))]
    elif isinstance(spec, Id):
        hv.annotate_schemas(GenId.id, None)
        return hv, [constant(hv._index(), Value.from_id(spec))]
    elif isinstance(spec, Value):
        return hv, [constant(hv._index(), spec)]
    elif isinstance(spec, str):
        # Treat strings starting with ? as variables, otherwise as string constants.
        if spec.startswith('?'):
            return var(spec[1:]), []
        hv.annotate_schemas(ShortString.id, None)
        return hv, [constant(hv._index(), Value.from_str(spec))]
    raise TypeError(f"unexpected pattern element: {type(spec).__name__}")


def _tribleset_where(self, entity, attribute, value):
    """Build a constraint for a triple pattern.

    Each position can be:
    - Variable (from ts.var()) — free variable
    - Attribute — fixed attribute
    - Id — fixed entity or value
    - Value — fixed value
    - str starting with "?" — variable shorthand
    - str — fixed ShortString value

    Returns a ConstraintBuilder that supports & for intersection.

        c = kb.where(e, name, n) & kb.where(e, phase, p)
    """
    ev, ec = _ensure_var(entity)
    av, ac = _ensure_var(attribute)
    vv, vc = _ensure_var(value)
    pat = self.pattern(ev, av, vv)
    all_constraints = [pat] + ec + ac + vc
    if len(all_constraints) == 1:
        return ConstraintBuilder(all_constraints[0])
    return ConstraintBuilder(intersect(all_constraints))

TribleSet.where = _tribleset_where
TribleSet.__iter__ = lambda self: iter(self.triples())



def exists(constraint):
    """Check if any solutions exist for the constraint. Short-circuits."""
    if isinstance(constraint, ConstraintBuilder):
        constraint = constraint._inner
    # Solve with empty projection — just check if there's at least one solution.
    q = solve([], constraint)
    result = next(q, None) is not None
    return result


def query(projected, constraint):
    """Run a query. Returns an iterator of result rows.

        for name, phase in ts.query([n, p], constraint):
            print(name.to_str())
    """
    if isinstance(constraint, ConstraintBuilder):
        constraint = constraint._inner
    # Auto-annotate projected variables that don't have schemas yet.
    proj_list = list(projected)
    for v in proj_list:
        if v.annotate_schemas is not None:
            try:
                v.annotate_schemas(GenId.id, None)
            except:
                pass
    return solve(proj_list, constraint)


# Patch __add__ so kb + fragment works (returns new TribleSet).
_original_add = TribleSet.__add__

def _patched_add(self, other):
    if isinstance(other, Fragment):
        return _original_add(self, other.facts)
    return _original_add(self, other)

TribleSet.__add__ = _patched_add

# Patch consume so kb.consume(fragment) works for in-place mutation.
_original_consume = TribleSet.consume

def _patched_consume(self, other):
    if isinstance(other, Fragment):
        _original_consume(self, other.facts)
    else:
        _original_consume(self, other)

TribleSet.consume = _patched_consume


_original_schema_attribute_id = Schema._attribute_id

def _schema_attribute(self, name):
    """Derive a named Attribute from this schema.

        name = ts.ShortString.attribute('name')
        friend = ts.GenId.attribute('friend')
    """
    attr = object.__new__(Attribute)
    attr.id = _original_schema_attribute_id(self, name)
    attr.label = name
    attr.schema = self
    return attr

Schema.attribute = _schema_attribute


class Attribute:
    """A named attribute with a known schema.

    Two ways to create:

        # Explicit hex ID (matches Rust attributes! macro):
        title = ts.Attribute('EE18CEC15C18438A2FAB670E2E46E00C', schema=ts.ShortString)

        # Derived from name via Schema.attribute():
        name = ts.ShortString.attribute('name')
        friend = ts.GenId.attribute('friend')
        desc = ts.Handle(ts.UTF8String).attribute('description')
    """
    __slots__ = ('id', 'label', 'schema')

    def __init__(self, hex_id, *, schema):
        self.id = Id.hex(hex_id)
        self.label = hex_id[:8] + '...'
        self.schema = schema

    def __hash__(self):
        return hash(self.id.to_hex())

    def __eq__(self, other):
        return isinstance(other, Attribute) and self.id.to_hex() == other.id.to_hex()

    def __repr__(self):
        return f"Attribute({self.label!r}, {self.id.to_hex()[:8]}...)"


def find(kb, *pattern_spec):
    """Pythonic query interface.

    Pattern spec alternates between:
    - String starting with "?" → variable (projected in results)
    - Attribute → fixed attribute in the pattern
    - Id → fixed entity/value
    - Value → fixed value

    Triples are specified as consecutive (entity, attribute, value) groups.

    Example:
        for entity, name in find(kb, "?e", name_attr, "?v"):
            print(name.to_str())
    """
    ctx = VariableContext()
    variables = {}
    projected = []
    constraints = []

    # Parse pattern spec into triples.
    items = list(pattern_spec)
    if len(items) % 3 != 0:
        raise ValueError("pattern must have 3 items per triple (entity, attr, value)")

    triple_count = len(items) // 3
    for i in range(triple_count):
        e_spec, a_spec, v_spec = items[i*3], items[i*3+1], items[i*3+2]

        parts = []
        for pos, spec in enumerate([e_spec, a_spec, v_spec]):
            if isinstance(spec, str) and spec.startswith("?"):
                vname = spec[1:]
                if vname not in variables:
                    v = ctx.fresh_variable(vname)
                    variables[vname] = v
                    projected.append((vname, v))
                parts.append(("var", variables[vname]))
            elif isinstance(spec, Attribute):
                parts.append(("const_id", spec.id, spec.schema.id if spec.schema else GenId.id))
            elif isinstance(spec, Id):
                parts.append(("const_id", spec, GenId.id))
            elif isinstance(spec, Value):
                parts.append(("const_val", spec))
            else:
                raise TypeError(f"unexpected pattern element: {type(spec)}")

        # We need all three positions as variables for kb.pattern().
        # For constants, create hidden variables and add constant constraints.
        triple_vars = []
        for pos, part in enumerate(parts):
            if part[0] == "var":
                triple_vars.append(part[1])
            elif part[0] == "const_id":
                hv = ctx.fresh_variable(f"_hidden_{i}_{pos}")
                hv.annotate_schemas(part[2], None)
                triple_vars.append(hv)
                constraints.append(constant(hv._index(), Value.from_id(part[1])))
            elif part[0] == "const_val":
                hv = ctx.fresh_variable(f"_hidden_{i}_{pos}")
                triple_vars.append(hv)
                constraints.append(constant(hv._index(), part[1]))

        # Add the pattern constraint.
        constraints.append(kb.pattern(triple_vars[0], triple_vars[1], triple_vars[2]))

    # Annotate projected variables with schemas from their attributes.
    for vname, v in projected:
        # Try to infer schema from the pattern context.
        # For now, just use GenId as default (values will be raw).
        if not hasattr(v, '_schema_set'):
            v.annotate_schemas(GenId.id, None)

    proj_vars = [v for _, v in projected]
    combined = intersect(constraints) if len(constraints) > 1 else constraints[0]
    q = solve(proj_vars, combined)
    return q


# Module-level owner for convenience. Users who need explicit ownership
# can still use IdOwner/guard directly.
_default_owner = IdOwner()


def mint_id():
    """Mint a new random Id."""
    guard = _default_owner.lock()
    return guard.rngid()



# ── Metadata attributes — matching Rust metadata module ──────────────

class _Metadata:
    """Canonical metadata attributes from triblespace-core."""
    def __init__(self):
        hbl = Handle(UTF8String)
        self.description = Attribute('AE94660A55D2EE3C428D2BB299E02EC3', schema=hbl)
        self.value_encoding = Attribute('213F89E3F49628A105B3830BD3A6612C', schema=GenId)
        self.blob_encoding = Attribute('43C134652906547383054B1E31E23DF4', schema=GenId)
        self.hash_schema = Attribute('51C08CFABB2C848CE0B4A799F0EFE5EA', schema=GenId)
        self.name = Attribute('7FB28C0B48E1924687857310EE230414', schema=hbl)
        self.attribute = Attribute('F10DE6D8E60E0E86013F1B867173A85C', schema=GenId)
        self.source = Attribute('A56350FD00EC220B4567FE15A5CD68B8', schema=hbl)
        self.source_module = Attribute('BCB94C7439215641A3E9760CE3F4F432', schema=hbl)
        self.json_kind = Attribute('A7AFC8C0FAD017CE7EC19587AF682CFF', schema=ShortString)
        self.tag = Attribute('91C50E9FBB1F73E892EBD5FFDE46C251', schema=GenId)
        self.created_at = Attribute('9B1E79DFD065F643954141593CD8B9E0', schema=NsTAIInterval)
        self.updated_at = Attribute('93B7372E3443063392CD801B03A8D390', schema=NsTAIInterval)
        self.started_at = Attribute('06973030ACA83A7B2B4FC8BEBB31F77A', schema=NsTAIInterval)
        self.finished_at = Attribute('9B06AA4060EF9928A923FC7E6A6B6438', schema=NsTAIInterval)
        self.expires_at = Attribute('89FEC3B560336BA88B10759DECD3155F', schema=NsTAIInterval)

metadata = _Metadata()



def _coerce_value(val, attr=None):
    """Auto-convert Python values to triblespace Values."""
    if isinstance(val, Value):
        return val
    if isinstance(val, str):
        return Value.from_str(val)
    if isinstance(val, Id):
        return Value.from_id(val)
    if isinstance(val, bool):
        return Value.from_bool(val)
    if isinstance(val, float):
        return Value.from_f64(val)
    if isinstance(val, int):
        # Ints that fit in f64 — good enough for now.
        return Value.from_f64(float(val))
    raise TypeError(f"can't coerce {type(val).__name__} to Value")


def entity(facts, *, id=None):
    """Build a Fragment for one entity. Auto-mints an Id if not provided.

    Returns a Fragment — union into your KB with +=.

    Values auto-converted: str→ShortString, Id→GenId, Value→as-is.

    Example:
        alice = ts.entity({name: "Alice", friend: bob.id})
        kb += alice
        print(alice.id)  # the entity's Id
    """
    eid = id if id is not None else mint_id()
    result = TribleSet()
    for attr, val in facts.items():
        if isinstance(attr, Attribute):
            attr_id = attr.id
        elif isinstance(attr, Id):
            attr_id = attr
        else:
            raise TypeError(f"attribute must be Attribute or Id, got {type(attr).__name__}")
        result.add(eid, attr_id, _coerce_value(val, attr))
    return Fragment(eid, result)
