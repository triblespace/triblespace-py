"""TribleSpace — a knowledge graph in Python.

Low-level bindings from the Rust core, plus Pythonic convenience wrappers.

Quick start:

    import triblespace as ts

    kb = ts.TribleSet()
    owner = ts.IdOwner()
    guard = owner.lock()

    # Define attributes
    name = ts.Attribute("name", guard)
    age = ts.Attribute("age", guard)
    friend = ts.Attribute("friend", guard, is_id=True)

    # Add facts
    alice = guard.rngid()
    bob = guard.rngid()
    kb.add(alice, name.id, ts.Value.from_str("Alice"))
    kb.add(bob, name.id, ts.Value.from_str("Bob"))
    kb.add(alice, friend.id, ts.Value.from_id(bob))

    # Query
    for entity, val in ts.find(kb, "?entity", name, "?value"):
        print(val.to_str())
"""

# Re-export everything from the native module.
from .triblespace import *
from .triblespace import (
    TribleSet, Id, IdOwner, Value, Variable, VariableContext,
    PyConstraint as Constraint, Query,
    constant, intersect, solve,
    get_value_schema, get_blob_schema, get_label_name_handles,
    metadata_description,
    register_type, register_to_value_converter, register_from_value_converter,
    register_to_blob_converter, register_from_blob_converter,
)

# Well-known schema IDs.
SCHEMA_GENID = Id.hex("3BE6EEE252CD6A886B0E4E33F3D5BF3F")
SCHEMA_SHORT_STRING = Id.hex("3E4174825DCB6A1C5D0B3F360753D648")

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
        hv.annotate_schemas(spec.schema, None)
        return hv, [constant(hv._index(), Value.from_id(spec.id))]
    elif isinstance(spec, Id):
        hv.annotate_schemas(SCHEMA_GENID, None)
        return hv, [constant(hv._index(), Value.from_id(spec))]
    elif isinstance(spec, Value):
        return hv, [constant(hv._index(), spec)]
    elif isinstance(spec, str):
        # Treat strings starting with ? as variables, otherwise as string constants.
        if spec.startswith('?'):
            return var(spec[1:]), []
        hv.annotate_schemas(SCHEMA_SHORT_STRING, None)
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
                v.annotate_schemas(SCHEMA_GENID, None)
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


class Attribute:
    """A named attribute with a known schema.

    Convenience wrapper that mints an Id and remembers the schema
    so queries can auto-annotate variables.

        name_attr = Attribute("name", guard)
        kb.add(entity, name_attr.id, Value.from_str("Alice"))
    """
    def __init__(self, label=None, *, id=None, is_id=False, schema=None):
        if id is not None:
            self.id = id
        elif label is not None:
            # Deterministic ID from attribute name.
            import hashlib
            h = hashlib.blake2b(label.encode(), digest_size=16).digest()
            self.id = Id(h)
        else:
            raise ValueError("provide label (for derived id) or id= (explicit)")
        self.label = label
        if schema is not None:
            self.schema = schema
        elif is_id:
            self.schema = SCHEMA_GENID
        else:
            self.schema = SCHEMA_SHORT_STRING

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
                parts.append(("const_id", spec.id, spec.schema))
            elif isinstance(spec, Id):
                parts.append(("const_id", spec, SCHEMA_GENID))
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
            v.annotate_schemas(SCHEMA_GENID, None)

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



def _coerce_value(val, attr=None):
    """Auto-convert Python values to triblespace Values."""
    if isinstance(val, Value):
        return val
    if isinstance(val, str):
        return Value.from_str(val)
    if isinstance(val, Id):
        return Value.from_id(val)
    if isinstance(val, (int, float)):
        return Value.from_str(str(val))
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
