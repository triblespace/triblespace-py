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

class Attribute:
    """A named attribute with a known schema.

    Convenience wrapper that mints an Id and remembers the schema
    so queries can auto-annotate variables.

        name_attr = Attribute("name", guard)
        kb.add(entity, name_attr.id, Value.from_str("Alice"))
    """
    def __init__(self, label, guard=None, *, id=None, is_id=False, schema=None):
        if id is not None:
            self.id = id
        elif guard is not None:
            self.id = guard.rngid()
        else:
            raise ValueError("provide either guard (to mint) or id (existing)")
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


def entity(kb, entity_id, facts):
    """Add multiple attribute-value pairs for an entity.

    facts: dict mapping Attribute (or Id) → value
    Values auto-converted: str→ShortString, Id→GenId, Value→as-is

    Example:
        ts.entity(kb, alice, {
            name: "Alice",
            age: "30",
            friend: bob,
        })
    """
    for attr, val in facts.items():
        if isinstance(attr, Attribute):
            attr_id = attr.id
        elif isinstance(attr, Id):
            attr_id = attr
        else:
            raise TypeError(f"attribute must be Attribute or Id, got {type(attr).__name__}")
        kb.add(entity_id, attr_id, _coerce_value(val, attr))


def add_entity(kb, guard, facts):
    """Create a new entity with the given facts. Returns the entity Id.

    Example:
        alice = ts.add_entity(kb, guard, {name: "Alice", friend: bob})
    """
    eid = guard.rngid()
    entity(kb, eid, facts)
    return eid
