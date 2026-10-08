// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMCXX_ENCODING_H_
#define LOOMCXX_ENCODING_H_

#include <loomcxx/encoding_type.h>

namespace loom::encoding {

// Numeric formats for block scale operands. Enum names match the encoding
// dialect; their C++ numeric values are independent of compiler ordinals.
enum class numeric_format { e8m0, f16, bf16, f32 };

// Physical ordering of encoded payload elements.
enum class payload_packing {
  dense_lanes,
  little_endian_nibbles,
  big_endian_nibbles,
};

// Logical groups sharing an auxiliary scale.
enum class scale_topology { none, tensor_global, group_1d, block_1d };

// Affine interpretation applied after numeric decoding.
enum class affine_policy { none, scale_only };

// E2M1 schema parameters, defaulting to one MXFP4 group of 32 values. Packed
// word counts derive from the payload size unless overridden. Runtime scales
// belong to vector.decode's auxiliary record.
struct f4e2m1 {
  // Logical value count decoded from the physical payload.
  unsigned payload_elements = 32;
  // Number of packed 32-bit words, rounded up to cover the logical payload.
  unsigned payload_registers = (payload_elements + 7ull) / 8;
  // Nibble ordering within each payload word.
  encoding::payload_packing payload_packing =
      encoding::payload_packing::little_endian_nibbles;
  // Numeric format of the explicit scale values.
  numeric_format scale_format = numeric_format::e8m0;
  // Number of consecutive values sharing a scale.
  unsigned scale_group_elements = 32;
  // Number of named scale operands, independently of each operand's lanes.
  unsigned scale_operands = 1;
  // Logical organization of the scale groups.
  encoding::scale_topology scale_topology = encoding::scale_topology::block_1d;
  // Apply the block scale to decoded numeric values.
  affine_policy affine = affine_policy::scale_only;
  // Preserve the E8M0 minimum-scale interpretation when its code is zero.
  bool zero_scale_fallback = true;
};

// Finite E4M3 schema parameters, defaulting to one MXFP8 group of 32 values.
// Payload lanes have the ordinary float8_e4m3fn_t vector representation.
struct f8e4m3fn {
  // Logical value count, equal to the FP8 payload's lane count.
  unsigned payload_elements = 32;
  // Numeric format of the explicit scale values.
  numeric_format scale_format = numeric_format::e8m0;
  // Number of consecutive values sharing a scale.
  unsigned scale_group_elements = 32;
  // Number of named scale operands, independently of each operand's lanes.
  unsigned scale_operands = 1;
  // Logical organization of the scale groups.
  encoding::scale_topology scale_topology = encoding::scale_topology::block_1d;
  // Apply the block scale to decoded numeric values.
  affine_policy affine = affine_policy::scale_only;
};

// Defines a first-class schema from static parameters. The aggregate is a C++
// constant template argument; the returned value can pass through ordinary
// helpers and records. Custom families use the same loom::op binding with a
// parameter aggregate whose fields match the family's public schema.
template <f4e2m1 Parameters>
[[loom::op("encoding.define", "encoding.f4e2m1")]]
type::encoding<role::schema> define();

template <f8e4m3fn Parameters>
[[loom::op("encoding.define", "encoding.f8e4m3fn")]]
type::encoding<role::schema> define();

// Composes an address layout and storage schema for the same logical rank.
// Dynamic schema auxiliaries remain explicit operands of decode operations.
template <type::size_type Rank>
[[loom::op("encoding.define")]]
type::encoding<role::storage, Rank> define(
    type::encoding<role::layout, Rank> layout,
    type::encoding<role::schema> schema);

}  // namespace loom::encoding

#endif  // LOOMCXX_ENCODING_H_
