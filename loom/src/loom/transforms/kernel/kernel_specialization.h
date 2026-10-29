// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Mandatory compile-time arguments of independently compiled kernel products.

#ifndef LOOM_TRANSFORMS_KERNEL_KERNEL_SPECIALIZATION_H_
#define LOOM_TRANSFORMS_KERNEL_KERNEL_SPECIALIZATION_H_

#include "loom/decision/class_partition.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// One command-side launch fact environment.
typedef struct loom_kernel_class_site_t {
  // Existing O(1) fact table for the containing command function.
  const loom_value_fact_table_t* facts;

  // Complete source argument IDs in signature order, including staged values.
  const loom_value_id_t* argument_values;
} loom_kernel_class_site_t;

typedef enum loom_kernel_specialization_value_kind_e {
  // An exact scalar reconstructed from its facts.
  LOOM_KERNEL_SPECIALIZATION_VALUE_KIND_EXACT_SCALAR = 0,
  // A closed operand-free ConstantLike operation cloned into the product.
  LOOM_KERNEL_SPECIALIZATION_VALUE_KIND_CLOSED_LITERAL = 1,
  // A vector reconstructed by splatting one exact scalar element.
  LOOM_KERNEL_SPECIALIZATION_VALUE_KIND_UNIFORM_SPLAT = 2,
} loom_kernel_specialization_value_kind_t;

typedef union loom_kernel_specialization_value_payload_u {
  // Source operation for CLOSED_LITERAL values.
  const loom_op_t* literal;
  // Exact scalar or splat-element facts for the other value kinds.
  loom_value_facts_t scalar_facts;
} loom_kernel_specialization_value_payload_t;

typedef struct loom_kernel_specialization_value_t {
  // Fact-specialized source type, including concrete dependent dimensions.
  loom_type_t type;
  // Reconstruction mechanism selected at the command-side source boundary.
  loom_kernel_specialization_value_kind_t kind;
  // Payload selected by |kind|.
  loom_kernel_specialization_value_payload_t payload;
} loom_kernel_specialization_value_t;

// Interns mandatory argument tuples and seeds the initial partition. Unlike
// optional decisions, distinct tuples cannot be collapsed at the class limit.
// Returned per-site tuples and the partition are arena-owned; literal payloads
// borrow site_module until the independent product is materialized.
iree_status_t loom_kernel_specializations_collect(
    const loom_module_t* site_module, const loom_kernel_class_site_t* sites,
    uint16_t specialization_count, loom_decision_class_partition_t* partition,
    iree_arena_allocator_t* arena,
    const loom_kernel_specialization_value_t*** out_site_specializations);

// Binds a collected tuple into an independently owned function, specializing
// dependent types and the signature contract before removing the staged prefix.
iree_status_t loom_kernel_specializations_apply(
    const loom_module_t* site_module,
    const loom_kernel_specialization_value_t* specializations,
    uint16_t specialization_count, loom_func_like_t function,
    loom_rewriter_t* rewriter, iree_arena_allocator_t* arena);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_KERNEL_KERNEL_SPECIALIZATION_H_
