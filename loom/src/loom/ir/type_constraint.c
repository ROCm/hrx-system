// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/type_constraint.h"

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/ir/scalar_type.h"

const char* loom_type_constraint_name(loom_type_constraint_t constraint) {
  static const char* const names[] = {
      [LOOM_TYPE_CONSTRAINT_TILE] = "tile",
      [LOOM_TYPE_CONSTRAINT_TENSOR] = "tensor",
      [LOOM_TYPE_CONSTRAINT_INTEGER] = "integer",
      [LOOM_TYPE_CONSTRAINT_FLOAT] = "float",
      [LOOM_TYPE_CONSTRAINT_BITWISE_SCALAR] = "non-i1 bitwise scalar",
      [LOOM_TYPE_CONSTRAINT_SCALAR] = "scalar",
      [LOOM_TYPE_CONSTRAINT_INDEX] = "index",
      [LOOM_TYPE_CONSTRAINT_OFFSET] = "offset",
      [LOOM_TYPE_CONSTRAINT_ADDRESS] = "address",
      [LOOM_TYPE_CONSTRAINT_ANY] = "any",
      [LOOM_TYPE_CONSTRAINT_ANY_ENCODING] = "encoding",
      [LOOM_TYPE_CONSTRAINT_POOL] = "pool",
      [LOOM_TYPE_CONSTRAINT_REGISTER] = "register",
      [LOOM_TYPE_CONSTRAINT_I1] = "i1",
      [LOOM_TYPE_CONSTRAINT_I32] = "i32",
      [LOOM_TYPE_CONSTRAINT_VECTOR] = "vector",
      [LOOM_TYPE_CONSTRAINT_RANK_ONE_VECTOR] = "rank-1 vector",
      [LOOM_TYPE_CONSTRAINT_ALL_STATIC_VECTOR] = "all-static vector shape",
      [LOOM_TYPE_CONSTRAINT_ALL_STATIC_RANK_ONE_VECTOR] =
          "all-static rank-1 vector",
      [LOOM_TYPE_CONSTRAINT_VIEW] = "view",
      [LOOM_TYPE_CONSTRAINT_BUFFER] = "buffer",
      [LOOM_TYPE_CONSTRAINT_INDEX_OR_NON_I1_INTEGER_SCALAR] =
          "index or non-i1 integer scalar",
      [LOOM_TYPE_CONSTRAINT_INDEX_OR_NON_I1_INTEGER_ELEMENT] =
          "index or non-i1 integer element type",
      [LOOM_TYPE_CONSTRAINT_INTEGER_ELEMENT] = "integer_element",
      [LOOM_TYPE_CONSTRAINT_FLOAT_ELEMENT] = "float_element",
      [LOOM_TYPE_CONSTRAINT_BITWISE_ELEMENT] = "non-i1 bitwise element type",
      [LOOM_TYPE_CONSTRAINT_I1_ELEMENT] = "i1_element",
      [LOOM_TYPE_CONSTRAINT_I8_ELEMENT] = "i8 element type",
      [LOOM_TYPE_CONSTRAINT_I32_ELEMENT] = "i32 element type",
      [LOOM_TYPE_CONSTRAINT_F16_OR_BF16_ELEMENT] = "f16 or bf16 element type",
      [LOOM_TYPE_CONSTRAINT_F32_ELEMENT] = "f32 element type",
      [LOOM_TYPE_CONSTRAINT_ENCODING_LAYOUT] = "encoding<layout>",
      [LOOM_TYPE_CONSTRAINT_ENCODING_SCHEMA] = "encoding<schema>",
      [LOOM_TYPE_CONSTRAINT_ENCODING_STORAGE] = "encoding<storage>",
      [LOOM_TYPE_CONSTRAINT_ENCODING_TRANSFORM] = "encoding<transform>",
      [LOOM_TYPE_CONSTRAINT_STORAGE] = "storage",
      [LOOM_TYPE_CONSTRAINT_BYTE_PATTERN_SCALAR] =
          "8/16/32/64-bit scalar pattern",
  };
  static_assert(IREE_ARRAYSIZE(names) == LOOM_TYPE_CONSTRAINT_COUNT_,
                "constraint names out of sync with enum");
  if (constraint < LOOM_TYPE_CONSTRAINT_COUNT_) return names[constraint];
  return "unknown";
}

bool loom_type_satisfies_constraint(loom_type_t type,
                                    loom_type_constraint_t constraint) {
  switch (constraint) {
    case LOOM_TYPE_CONSTRAINT_ANY:
      return true;
    case LOOM_TYPE_CONSTRAINT_TILE:
      return loom_type_is_tile(type);
    case LOOM_TYPE_CONSTRAINT_TENSOR:
      return loom_type_is_tensor(type);
    case LOOM_TYPE_CONSTRAINT_VECTOR:
      return loom_type_is_vector(type);
    case LOOM_TYPE_CONSTRAINT_RANK_ONE_VECTOR:
      return loom_type_is_vector(type) && loom_type_rank(type) == 1;
    case LOOM_TYPE_CONSTRAINT_ALL_STATIC_VECTOR:
      if (!loom_type_is_vector(type)) return false;
      for (uint8_t i = 0; i < loom_type_rank(type); ++i) {
        if (loom_type_dim_is_dynamic_at(type, i)) return false;
      }
      return true;
    case LOOM_TYPE_CONSTRAINT_ALL_STATIC_RANK_ONE_VECTOR:
      if (!loom_type_is_vector(type) || loom_type_rank(type) != 1) return false;
      return !loom_type_dim_is_dynamic_at(type, 0);
    case LOOM_TYPE_CONSTRAINT_VIEW:
      return loom_type_is_view(type);
    case LOOM_TYPE_CONSTRAINT_BUFFER:
      return loom_type_is_buffer(type);
    case LOOM_TYPE_CONSTRAINT_SCALAR:
      return loom_type_is_scalar(type);
    case LOOM_TYPE_CONSTRAINT_INDEX:
      return loom_type_is_scalar(type) &&
             loom_type_element_type(type) == LOOM_SCALAR_TYPE_INDEX;
    case LOOM_TYPE_CONSTRAINT_OFFSET:
      return loom_type_is_scalar(type) &&
             loom_type_element_type(type) == LOOM_SCALAR_TYPE_OFFSET;
    case LOOM_TYPE_CONSTRAINT_ADDRESS:
      return loom_type_is_scalar(type) &&
             (loom_type_element_type(type) == LOOM_SCALAR_TYPE_INDEX ||
              loom_type_element_type(type) == LOOM_SCALAR_TYPE_OFFSET);
    case LOOM_TYPE_CONSTRAINT_INTEGER:
      return loom_type_is_scalar(type) &&
             loom_scalar_type_is_integer(loom_type_element_type(type));
    case LOOM_TYPE_CONSTRAINT_FLOAT:
      return loom_type_is_scalar(type) &&
             loom_scalar_type_is_float(loom_type_element_type(type));
    case LOOM_TYPE_CONSTRAINT_BITWISE_SCALAR: {
      if (!loom_type_is_scalar(type)) return false;
      const loom_scalar_type_t scalar_type = loom_type_element_type(type);
      return scalar_type == LOOM_SCALAR_TYPE_INDEX ||
             (scalar_type != LOOM_SCALAR_TYPE_I1 &&
              (loom_scalar_type_is_integer(scalar_type) ||
               loom_scalar_type_is_float(scalar_type)));
    }
    case LOOM_TYPE_CONSTRAINT_BYTE_PATTERN_SCALAR: {
      if (!loom_type_is_scalar(type)) return false;
      const loom_scalar_type_t scalar_type = loom_type_element_type(type);
      return loom_scalar_type_set_contains(
          LOOM_SCALAR_TYPE_SET_INTEGER_PAYLOAD | LOOM_SCALAR_TYPE_SET_FLOAT,
          scalar_type);
    }
    case LOOM_TYPE_CONSTRAINT_INDEX_OR_NON_I1_INTEGER_SCALAR: {
      if (!loom_type_is_scalar(type)) return false;
      const loom_scalar_type_t scalar_type = loom_type_element_type(type);
      if (scalar_type == LOOM_SCALAR_TYPE_INDEX) return true;
      return scalar_type != LOOM_SCALAR_TYPE_I1 &&
             loom_scalar_type_is_integer(scalar_type);
    }
    case LOOM_TYPE_CONSTRAINT_ANY_ENCODING:
      return loom_type_is_encoding(type);
    case LOOM_TYPE_CONSTRAINT_ENCODING_LAYOUT:
      return loom_type_is_encoding(type) &&
             loom_type_encoding_role(type) == LOOM_ENCODING_ROLE_ADDRESS_LAYOUT;
    case LOOM_TYPE_CONSTRAINT_ENCODING_SCHEMA:
      return loom_type_is_encoding(type) &&
             loom_type_encoding_role(type) == LOOM_ENCODING_ROLE_STORAGE_SCHEMA;
    case LOOM_TYPE_CONSTRAINT_ENCODING_STORAGE:
      return loom_type_is_encoding(type) &&
             loom_type_encoding_role(type) ==
                 LOOM_ENCODING_ROLE_PHYSICAL_STORAGE;
    case LOOM_TYPE_CONSTRAINT_ENCODING_TRANSFORM:
      return loom_type_is_encoding(type) &&
             loom_type_encoding_role(type) ==
                 LOOM_ENCODING_ROLE_NUMERIC_TRANSFORM;
    case LOOM_TYPE_CONSTRAINT_POOL:
      return loom_type_is_pool(type);
    case LOOM_TYPE_CONSTRAINT_REGISTER:
      return loom_type_is_register(type);
    case LOOM_TYPE_CONSTRAINT_STORAGE:
      return loom_type_is_storage(type);
    case LOOM_TYPE_CONSTRAINT_I1:
      return loom_type_is_scalar(type) &&
             loom_type_element_type(type) == LOOM_SCALAR_TYPE_I1;
    case LOOM_TYPE_CONSTRAINT_I32:
      return loom_type_is_scalar(type) &&
             loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32;
    case LOOM_TYPE_CONSTRAINT_INTEGER_ELEMENT:
      return loom_type_is_shaped(type) &&
             loom_scalar_type_is_integer(loom_type_element_type(type));
    case LOOM_TYPE_CONSTRAINT_FLOAT_ELEMENT:
      return loom_type_is_shaped(type) &&
             loom_scalar_type_is_float(loom_type_element_type(type));
    case LOOM_TYPE_CONSTRAINT_BITWISE_ELEMENT: {
      if (!loom_type_is_shaped(type)) return false;
      const loom_scalar_type_t element_type = loom_type_element_type(type);
      return element_type == LOOM_SCALAR_TYPE_INDEX ||
             (element_type != LOOM_SCALAR_TYPE_I1 &&
              (loom_scalar_type_is_integer(element_type) ||
               loom_scalar_type_is_float(element_type)));
    }
    case LOOM_TYPE_CONSTRAINT_INDEX_OR_NON_I1_INTEGER_ELEMENT: {
      if (!loom_type_is_shaped(type)) return false;
      const loom_scalar_type_t element_type = loom_type_element_type(type);
      if (element_type == LOOM_SCALAR_TYPE_INDEX) return true;
      return element_type != LOOM_SCALAR_TYPE_I1 &&
             loom_scalar_type_is_integer(element_type);
    }
    case LOOM_TYPE_CONSTRAINT_I1_ELEMENT:
      return loom_type_is_shaped(type) &&
             loom_type_element_type(type) == LOOM_SCALAR_TYPE_I1;
    case LOOM_TYPE_CONSTRAINT_I8_ELEMENT:
      return loom_type_is_shaped(type) &&
             loom_type_element_type(type) == LOOM_SCALAR_TYPE_I8;
    case LOOM_TYPE_CONSTRAINT_I32_ELEMENT:
      return loom_type_is_shaped(type) &&
             loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32;
    case LOOM_TYPE_CONSTRAINT_F16_OR_BF16_ELEMENT:
      return loom_type_is_shaped(type) &&
             (loom_type_element_type(type) == LOOM_SCALAR_TYPE_F16 ||
              loom_type_element_type(type) == LOOM_SCALAR_TYPE_BF16);
    case LOOM_TYPE_CONSTRAINT_F32_ELEMENT:
      return loom_type_is_shaped(type) &&
             loom_type_element_type(type) == LOOM_SCALAR_TYPE_F32;
    default:
      return false;
  }
}
