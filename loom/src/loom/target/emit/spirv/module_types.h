// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SPIR-V module type and constant emission.
//
// The type context owns uniquing for SPIR-V type declarations and emitter-owned
// integer constants within one module builder. It is deliberately independent
// of low-function traversal so packet emission and ABI materialization can
// share a compact numeric type cache without exposing module-emitter internals.

#ifndef LOOM_TARGET_EMIT_SPIRV_MODULE_TYPES_H_
#define LOOM_TARGET_EMIT_SPIRV_MODULE_TYPES_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/target/arch/spirv/scalar_types.h"
#include "loom/target/arch/spirv/value_types.h"
#include "loom/target/emit/spirv/module_builder.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_spirv_type_cache_entry_t loom_spirv_type_cache_entry_t;
typedef struct loom_spirv_integer_constant_cache_entry_t
    loom_spirv_integer_constant_cache_entry_t;

enum {
  // Fixed byte length of the raw-BDA PushConstant root header.
  LOOM_SPIRV_BDA_ROOT_BYTE_LENGTH = 32,
  // Byte offset of the optional inline-constant array in the raw-BDA root.
  LOOM_SPIRV_BDA_ROOT_CONSTANT_BYTE_OFFSET = 32,
  // Struct member index of the optional inline-constant array.
  LOOM_SPIRV_BDA_ROOT_CONSTANT_MEMBER_INDEX = 6,
};

typedef struct loom_spirv_type_context_t {
  // Scratch arena used for cache storage.
  iree_arena_allocator_t* scratch_arena;
  // Sectioned module builder receiving declarations and allocated IDs.
  loom_spirv_module_builder_t* builder;
  // Cached SPIR-V type declaration rows.
  loom_spirv_type_cache_entry_t* type_cache_entries;
  // Number of entries in |type_cache_entries|.
  iree_host_size_t type_cache_count;
  // Allocated capacity of |type_cache_entries|.
  iree_host_size_t type_cache_capacity;
  // Cached emitter-owned integer constants.
  loom_spirv_integer_constant_cache_entry_t* integer_constant_cache_entries;
  // Number of entries in |integer_constant_cache_entries|.
  iree_host_size_t integer_constant_cache_count;
  // Allocated capacity of |integer_constant_cache_entries|.
  iree_host_size_t integer_constant_cache_capacity;
} loom_spirv_type_context_t;

// Initializes |out_context| with cache storage allocated from |scratch_arena|.
void loom_spirv_type_context_initialize(loom_spirv_module_builder_t* builder,
                                        iree_arena_allocator_t* scratch_arena,
                                        loom_spirv_type_context_t* out_context);

iree_status_t loom_spirv_emit_type_void(loom_spirv_type_context_t* context,
                                        uint32_t* out_type_id);
iree_status_t loom_spirv_emit_type_bool(loom_spirv_type_context_t* context,
                                        uint32_t* out_type_id);
iree_status_t loom_spirv_emit_type_int(loom_spirv_type_context_t* context,
                                       uint32_t bit_width, uint32_t signedness,
                                       uint32_t* out_type_id);
iree_status_t loom_spirv_emit_type_float(loom_spirv_type_context_t* context,
                                         uint32_t bit_width,
                                         loom_spirv_fp_encoding_t fp_encoding,
                                         uint32_t* out_type_id);
iree_status_t loom_spirv_emit_type_scalar(loom_spirv_type_context_t* context,
                                          loom_spirv_scalar_type_t scalar_type,
                                          uint32_t* out_type_id);
iree_status_t loom_spirv_emit_type_vector(loom_spirv_type_context_t* context,
                                          uint32_t element_type_id,
                                          uint32_t component_count,
                                          uint32_t* out_type_id);
iree_status_t loom_spirv_emit_type_cooperative_matrix(
    loom_spirv_type_context_t* context, loom_spirv_value_type_t value_type,
    uint32_t* out_type_id);
iree_status_t loom_spirv_emit_type_i32(loom_spirv_type_context_t* context,
                                       uint32_t* out_type_id);
iree_status_t loom_spirv_emit_type_u32(loom_spirv_type_context_t* context,
                                       uint32_t* out_type_id);
iree_status_t loom_spirv_emit_type_u64(loom_spirv_type_context_t* context,
                                       uint32_t* out_type_id);
iree_status_t loom_spirv_emit_type_runtime_array(
    loom_spirv_type_context_t* context, uint32_t element_type_id,
    uint32_t array_stride, uint32_t* out_type_id);
iree_status_t loom_spirv_emit_type_array(loom_spirv_type_context_t* context,
                                         uint32_t element_type_id,
                                         uint32_t length_id,
                                         uint32_t array_stride,
                                         uint32_t* out_type_id);
iree_status_t loom_spirv_emit_type_struct(loom_spirv_type_context_t* context,
                                          const uint32_t* member_type_ids,
                                          uint8_t member_count,
                                          uint32_t* out_type_id,
                                          bool* out_emitted);
iree_status_t loom_spirv_emit_decorate_block(loom_spirv_type_context_t* context,
                                             uint32_t type_id);
iree_status_t loom_spirv_emit_decorate_member_offset(
    loom_spirv_type_context_t* context, uint32_t type_id,
    uint32_t member_ordinal, uint32_t byte_offset);
iree_status_t loom_spirv_emit_type_function(loom_spirv_type_context_t* context,
                                            uint32_t result_type_id,
                                            const uint32_t* parameter_type_ids,
                                            uint8_t parameter_count,
                                            uint32_t* out_type_id);
iree_status_t loom_spirv_emit_type_pointer(loom_spirv_type_context_t* context,
                                           uint32_t storage_class,
                                           uint32_t pointee_type_id,
                                           uint32_t pointer_array_stride,
                                           uint32_t* out_type_id);
iree_status_t loom_spirv_emit_type_ptr_storage_buffer_descriptor_struct(
    loom_spirv_type_context_t* context, uint32_t field_type_id,
    uint32_t* out_type_id);
iree_status_t loom_spirv_emit_type_ptr_storage_buffer_descriptor_field(
    loom_spirv_type_context_t* context, uint32_t field_type_id,
    uint32_t* out_type_id);
iree_status_t loom_spirv_emit_type_ptr_push_constant_bda_root(
    loom_spirv_type_context_t* context, uint16_t bda_constant_word_count,
    uint32_t* out_type_id);
iree_status_t loom_spirv_emit_type_ptr_push_constant_u32(
    loom_spirv_type_context_t* context, uint32_t* out_type_id);
iree_status_t loom_spirv_emit_type_ptr_push_constant_u64(
    loom_spirv_type_context_t* context, uint32_t* out_type_id);
iree_status_t
loom_spirv_emit_type_ptr_physical_storage_buffer_bda_address_table(
    loom_spirv_type_context_t* context, uint32_t* out_type_id);
iree_status_t loom_spirv_emit_type_ptr_physical_storage_buffer_u64(
    loom_spirv_type_context_t* context, uint32_t* out_type_id);
iree_status_t loom_spirv_emit_type_ptr_physical_storage_buffer_scalar(
    loom_spirv_type_context_t* context, loom_spirv_scalar_type_t scalar_type,
    uint32_t* out_type_id);
iree_status_t loom_spirv_emit_type_ptr_workgroup_scalar(
    loom_spirv_type_context_t* context, loom_spirv_scalar_type_t scalar_type,
    uint32_t* out_type_id);
iree_status_t loom_spirv_emit_type_id_for_value_type(
    loom_spirv_type_context_t* context, loom_spirv_value_type_t type,
    uint32_t* out_type_id);

iree_status_t loom_spirv_emit_i32_constant(loom_spirv_type_context_t* context,
                                           int32_t value,
                                           uint32_t* out_constant_id);
iree_status_t loom_spirv_emit_u32_constant(loom_spirv_type_context_t* context,
                                           uint32_t value,
                                           uint32_t* out_constant_id);
iree_status_t loom_spirv_emit_u64_constant(loom_spirv_type_context_t* context,
                                           uint64_t value,
                                           uint32_t* out_constant_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_SPIRV_MODULE_TYPES_H_
