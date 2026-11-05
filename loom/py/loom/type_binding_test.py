# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Complete type/attribute SSA ownership through shared graph translation."""

from loom.dialect.test import test_array_type, test_options_attr
from loom.ir import (
    F32,
    CanonicalAttrDict,
    DialectType,
    DynamicDim,
    DynamicEncoding,
    FunctionType,
    GroupType,
    ParameterizedAttrArray,
    PoolType,
    Predicate,
    PredicateArg,
    PredicateListAttr,
    RegisterType,
    ShapedType,
    StaticDim,
    TypeKind,
)
from loom.type_binding import (
    iter_value_binding_occurrences,
    iter_value_bindings,
    remap_value_bindings,
)


def test_nested_type_and_attribute_bindings_translate_together() -> None:
    vector = ShapedType(TypeKind.VECTOR, F32, (DynamicDim(1),))
    options = test_options_attr(mode="fast", element_type=vector)
    predicate = Predicate("eq", (PredicateArg("value", 2), PredicateArg("const", 1)))
    metadata = CanonicalAttrDict(
        (
            ("predicates", PredicateListAttr((predicate,))),
            ("shape", vector),
            ("options", ParameterizedAttrArray((options, options))),
        )
    )
    array = test_array_type(element_type=vector, metadata=metadata)
    view = ShapedType(
        TypeKind.VIEW, F32, (DynamicDim(1), StaticDim(4)), DynamicEncoding(3)
    )
    register = RegisterType(1, 2, 4, value_type=array)
    roots = [
        DialectType("test.ref", (register,)),
        view,
        PoolType(),
        GroupType((DynamicDim(1), StaticDim(8))),
    ]
    mapped, mapped_view, mapped_pool, mapped_group = remap_value_bindings(
        roots, {1: 11, 2: 12, 3: 13}
    )
    mapped_array = mapped.params[0].value_type
    mapped_vector = mapped_array.get("element_type")
    mapped_metadata = mapped_array.get("metadata")
    assert mapped_vector.dims == (DynamicDim(11),)
    assert mapped_metadata["shape"] is mapped_vector
    mapped_options = mapped_metadata["options"].values
    assert mapped_options[0] is mapped_options[1]
    assert mapped_options[0].get("element_type") is mapped_vector
    assert mapped_metadata["predicates"][0].args == (
        PredicateArg("value", 12),
        PredicateArg("const", 1),
    )
    assert not mapped_array.has("alignment")
    assert mapped_view.encoding == DynamicEncoding(13)
    assert mapped_pool is roots[2]
    assert mapped_group.dims == (DynamicDim(11), StaticDim(8))
    assert vector.dims == (DynamicDim(1),)
    assert (
        mapped.params[0].descriptor_set_stable_id == register.descriptor_set_stable_id
    )
    assert mapped.params[0].register_class_id == register.register_class_id
    assert mapped.params[0].unit_count == register.unit_count


def test_shared_graph_translation_is_iterative_and_preserves_sharing() -> None:
    leaf = ShapedType(TypeKind.VECTOR, F32, (DynamicDim(1),))
    root = leaf
    depth = 2048
    for _ in range(depth):
        root = FunctionType((root, root), (root,))
    mapped, shared, unchanged = remap_value_bindings([root, root, F32], {1: 2})
    assert mapped is shared
    assert unchanged is F32
    for _ in range(depth):
        assert mapped.arg_types[0] is mapped.arg_types[1]
        assert mapped.arg_types[0] is mapped.result_types[0]
        mapped = mapped.arg_types[0]
    assert mapped.dims == (DynamicDim(2),)
    assert list(iter_value_bindings(root)) == [DynamicDim(1)]
    assert remap_value_bindings([root], {7: 8})[0] is root


def test_unbound_and_external_identities_remain_distinct() -> None:
    unbound = DynamicDim()
    shape = ShapedType(TypeKind.VIEW, F32, (unbound, DynamicDim(4)), DynamicEncoding(5))
    [mapped] = remap_value_bindings([shape], {4: 9})
    assert mapped.dims == (unbound, DynamicDim(9))
    assert mapped.encoding is shape.encoding
    assert set(iter_value_bindings(mapped)) == {
        unbound,
        DynamicDim(9),
        DynamicEncoding(5),
    }


def test_empty_predicate_lists_preserve_their_attribute_kind() -> None:
    empty = PredicateListAttr()
    assert empty != ()
    assert empty != []
    assert hash(empty) == hash(PredicateListAttr())
    assert list(iter_value_bindings(empty)) == []
    assert remap_value_bindings([empty], {1: 2})[0] is empty
    metadata = CanonicalAttrDict((("predicates", empty), ("integers", ())))
    assert metadata["predicates"] is empty
    assert metadata["integers"] == ()


def test_occurrence_traversal_preserves_structure_and_repeated_references() -> None:
    repeated = ShapedType(TypeKind.VECTOR, F32, (DynamicDim(2), DynamicDim(2)))
    root = FunctionType(
        (repeated,),
        (
            ShapedType(
                TypeKind.VIEW,
                F32,
                (DynamicDim(3),),
                DynamicEncoding(4),
            ),
            DialectType(
                "test.predicated",
                (
                    Predicate(
                        "eq",
                        (PredicateArg("value", 5), PredicateArg("const", 0)),
                    ),
                ),
            ),
        ),
    )

    assert list(iter_value_binding_occurrences(root)) == [
        DynamicDim(2),
        DynamicDim(2),
        DynamicDim(3),
        DynamicEncoding(4),
        PredicateArg("value", 5),
    ]
