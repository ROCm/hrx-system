// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/spill_lowering.h"

#include "loom/codegen/low/allocation/spill_plan.h"
#include "loom/codegen/low/allocation/storage.h"
#include "loom/codegen/low/allocation_materialization.h"
#include "loom/codegen/low/builder.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/function.h"
#include "loom/error/parameter.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/low/storage.h"
#include "loom/ops/type_registry.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/emit/native/amdgpu/spill_recipe.h"
#include "loom/target/emit/native/amdgpu/storage_layout.h"
#include "loom/target/registers.h"

// SGPR, VGPR and AGPR allocation units are all 32-bit words.
#define LOOM_AMDGPU_SCRATCH_SPILL_UNIT_BYTES 4u

static const loom_low_storage_space_set_t kLoomAmdgpuSpillStorageSpaces =
    LOOM_LOW_STORAGE_SPACE_SET_SCRATCH | LOOM_LOW_STORAGE_SPACE_SET_PRIVATE;

typedef struct loom_amdgpu_spill_emission_context_t {
  // Module receiving emitted recipe operations.
  loom_module_t* module;
  // Target-low descriptor set selected for this function.
  const loom_low_descriptor_set_t* descriptor_set;
  // Attribute name used by scratch packet descriptors.
  loom_string_id_t offset_attr_id;
} loom_amdgpu_spill_emission_context_t;

typedef struct loom_amdgpu_spill_lowering_context_t {
  // State shared by authored and directly planned recipe emission.
  loom_amdgpu_spill_emission_context_t emission;
  // Low function owning the spill traffic being rewritten.
  loom_op_t* function_op;
  // Built fixed-segment layout for the function storage reservations.
  loom_amdgpu_storage_layout_t storage_layout;
  // Optional structured diagnostic emitter for user-authored spill traffic.
  iree_diagnostic_emitter_t emitter;
  // Mutable result receiving emitted diagnostic counts.
  loom_amdgpu_spill_lowering_result_t* result;
  // Scratch arena backing transient lowering state and result lists.
  iree_arena_allocator_t* scratch_arena;
  // Value IDs that the lowered traffic requires in registers.
  loom_value_id_t* required_register_value_ids;
  // Number of entries in |required_register_value_ids|.
  iree_host_size_t required_register_value_count;
  // Capacity of |required_register_value_ids|.
  iree_host_size_t required_register_value_capacity;
  // Register values created as private setup for one reload consumer.
  loom_value_id_t* per_user_placement_value_ids;
  // Number of entries in |per_user_placement_value_ids|.
  iree_host_size_t per_user_placement_value_count;
  // Capacity of |per_user_placement_value_ids|.
  iree_host_size_t per_user_placement_value_capacity;
  // VGPR values defined while spill lowering has widened EXEC.
  loom_value_id_t* expanded_execution_value_ids;
  // Number of entries in |expanded_execution_value_ids|.
  iree_host_size_t expanded_execution_value_count;
  // Capacity of |expanded_execution_value_ids|.
  iree_host_size_t expanded_execution_value_capacity;
} loom_amdgpu_spill_lowering_context_t;

static iree_status_t loom_amdgpu_spill_lowering_initialize_emission_context(
    loom_module_t* module, const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_spill_emission_context_t* context) {
  context->module = module;
  context->descriptor_set = descriptor_set;
  return loom_module_intern_string(module, IREE_SV("offset"),
                                   &context->offset_attr_id);
}

static iree_status_t loom_amdgpu_spill_lowering_append_value(
    loom_amdgpu_spill_lowering_context_t* context, loom_value_id_t value_id,
    loom_value_id_t** values, iree_host_size_t* count,
    iree_host_size_t* capacity) {
  const iree_host_size_t minimum_capacity = *count + 1;
  if (minimum_capacity > *capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        context->scratch_arena, *count, iree_max(minimum_capacity, 8u),
        sizeof(**values), capacity, (void**)values));
  }
  (*values)[(*count)++] = value_id;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_spill_lowering_record_required_register_value(
    loom_amdgpu_spill_lowering_context_t* context, loom_value_id_t value_id) {
  return loom_amdgpu_spill_lowering_append_value(
      context, value_id, &context->required_register_value_ids,
      &context->required_register_value_count,
      &context->required_register_value_capacity);
}

static iree_status_t loom_amdgpu_spill_lowering_record_expanded_execution_value(
    loom_amdgpu_spill_lowering_context_t* context, loom_value_id_t value_id) {
  return loom_amdgpu_spill_lowering_append_value(
      context, value_id, &context->expanded_execution_value_ids,
      &context->expanded_execution_value_count,
      &context->expanded_execution_value_capacity);
}

static iree_status_t loom_amdgpu_spill_lowering_record_per_user_placement_value(
    loom_amdgpu_spill_lowering_context_t* context, loom_value_id_t value_id) {
  return loom_amdgpu_spill_lowering_append_value(
      context, value_id, &context->per_user_placement_value_ids,
      &context->per_user_placement_value_count,
      &context->per_user_placement_value_capacity);
}

static iree_status_t loom_amdgpu_spill_lowering_record_private_result_placement(
    loom_amdgpu_spill_lowering_context_t* context, loom_value_id_t value_id) {
  const loom_value_t* value =
      loom_module_value(context->emission.module, value_id);
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (defining_op == NULL || defining_op->result_count != 1 ||
      value->use_count == 0 || loom_value_has_attribute_uses(value) ||
      loom_module_value_has_type_uses(context->emission.module, value_id)) {
    return iree_ok_status();
  }
  const loom_use_t* uses = loom_value_uses(value);
  const loom_op_t* user = loom_use_user_op(uses[0]);
  if (user == NULL) {
    return iree_ok_status();
  }
  for (uint32_t i = 1; i < value->use_count; ++i) {
    if (loom_use_user_op(uses[i]) != user) {
      return iree_ok_status();
    }
  }
  return loom_amdgpu_spill_lowering_record_per_user_placement_value(context,
                                                                    value_id);
}

static const loom_low_descriptor_t* loom_amdgpu_spill_lowering_descriptor_ref(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref) {
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_descriptor_ref_descriptor(descriptor_set, descriptor_ref);
  IREE_ASSERT(descriptor != NULL,
              "generated descriptor set must satisfy spill lowering refs");
  return descriptor;
}

static void loom_amdgpu_spill_lowering_resolve_storage_reference(
    const loom_amdgpu_spill_lowering_context_t* context,
    loom_value_id_t storage_value_id,
    loom_amdgpu_storage_layout_reference_t* out_reference) {
  loom_amdgpu_storage_layout_lookup_reference(&context->storage_layout,
                                              storage_value_id, out_reference);
}

static bool loom_amdgpu_spill_lowering_storage_space_supported(
    const loom_amdgpu_storage_layout_reference_t* storage_reference) {
  return loom_low_storage_space_set_contains(
      kLoomAmdgpuSpillStorageSpaces, storage_reference->reservation.space);
}

static iree_status_t loom_amdgpu_spill_lowering_emit_unsupported_storage_space(
    const loom_amdgpu_spill_lowering_context_t* context, const loom_op_t* op,
    loom_value_id_t storage_value_id,
    const loom_amdgpu_storage_layout_reference_t* storage_reference) {
  const iree_string_view_t storage_space =
      loom_low_storage_type_space_name(storage_reference->reservation.space);
  if (context->emitter.fn == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU spill lowering cannot lower storage space '%.*s'",
        (int)storage_space.size, storage_space.data);
  }

  iree_string_view_t supported_storage_space_names[LOOM_STORAGE_SPACE_COUNT_];
  const iree_host_size_t supported_storage_space_count =
      loom_low_storage_space_set_names(
          kLoomAmdgpuSpillStorageSpaces,
          IREE_ARRAYSIZE(supported_storage_space_names),
          supported_storage_space_names);
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_function_name(
          context->emission.module, context->function_op)),
      loom_param_string(loom_op_name(context->emission.module, op)),
      loom_param_string(loom_low_diagnostic_value_name(context->emission.module,
                                                       storage_value_id)),
      loom_param_string(storage_space),
      loom_param_string_list(supported_storage_space_names,
                             supported_storage_space_count),
  };
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_AMDGPU_037,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  IREE_RETURN_IF_ERROR(iree_diagnostic_emit(context->emitter, &emission));
  ++context->result->error_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_spill_lowering_emit_out_of_bounds_access(
    const loom_amdgpu_spill_lowering_context_t* context, const loom_op_t* op,
    loom_value_id_t storage_value_id, int64_t access_byte_offset,
    uint64_t access_byte_length,
    const loom_amdgpu_storage_layout_reference_t* storage_reference) {
  if (context->emitter.fn == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU spill lowering cannot lower out-of-bounds spill access");
  }
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_function_name(
          context->emission.module, context->function_op)),
      loom_param_string(loom_op_name(context->emission.module, op)),
      loom_param_string(loom_low_diagnostic_value_name(context->emission.module,
                                                       storage_value_id)),
      loom_param_i64(access_byte_offset),
      loom_param_u64(access_byte_length),
      loom_param_u64(storage_reference->byte_length),
  };
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_AMDGPU_039,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  IREE_RETURN_IF_ERROR(iree_diagnostic_emit(context->emitter, &emission));
  ++context->result->error_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_spill_lowering_validate_access_range(
    const loom_amdgpu_spill_lowering_context_t* context, const loom_op_t* op,
    loom_value_id_t storage_value_id,
    const loom_amdgpu_storage_layout_reference_t* storage_reference,
    int64_t operation_offset, uint64_t access_byte_length,
    bool* out_supported) {
  *out_supported = false;
  if (operation_offset < 0) {
    return loom_amdgpu_spill_lowering_emit_out_of_bounds_access(
        context, op, storage_value_id, operation_offset, access_byte_length,
        storage_reference);
  }
  const uint64_t access_offset = (uint64_t)operation_offset;
  if (access_offset > UINT64_MAX - access_byte_length) {
    return loom_amdgpu_spill_lowering_emit_out_of_bounds_access(
        context, op, storage_value_id, (int64_t)access_offset,
        access_byte_length, storage_reference);
  }
  const uint64_t access_end = access_offset + access_byte_length;
  if (access_end > storage_reference->byte_length) {
    return loom_amdgpu_spill_lowering_emit_out_of_bounds_access(
        context, op, storage_value_id, (int64_t)access_offset,
        access_byte_length, storage_reference);
  }
  *out_supported = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_spill_lowering_make_offset_attr(
    const loom_amdgpu_spill_emission_context_t* context, int64_t offset,
    loom_named_attr_t* out_attr) {
  *out_attr = (loom_named_attr_t){
      .name_id = context->offset_attr_id,
      .value = loom_attr_i64(offset),
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_spill_lowering_emit_unsupported_register_type(
    const loom_amdgpu_spill_lowering_context_t* context, const loom_op_t* op,
    loom_value_id_t value_id, loom_type_t type) {
  if (context->emitter.fn == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU spill lowering cannot lower spill value type");
  }
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_function_name(
          context->emission.module, context->function_op)),
      loom_param_string(loom_op_name(context->emission.module, op)),
      loom_param_string(
          loom_low_diagnostic_value_name(context->emission.module, value_id)),
      loom_param_type(type),
  };
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_AMDGPU_038,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  IREE_RETURN_IF_ERROR(iree_diagnostic_emit(context->emitter, &emission));
  ++context->result->error_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_spill_lowering_build_register_convert(
    const loom_amdgpu_spill_emission_context_t* context,
    loom_builder_t* builder, loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_value_id_t source, loom_type_t result_type,
    loom_location_id_t location, loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_spill_lowering_descriptor_ref(context->descriptor_set,
                                                descriptor_ref);
  const loom_value_id_t operands[] = {source};
  const loom_type_t result_types[] = {result_type};
  loom_op_t* convert_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, context->descriptor_set, descriptor,
      /*access_flags=*/0, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(NULL, 0), result_types,
      IREE_ARRAYSIZE(result_types), /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &convert_op));
  *out_value = loom_low_op_results(convert_op).values[0];
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_spill_lowering_build_exec_read(
    const loom_amdgpu_spill_emission_context_t* context,
    loom_builder_t* builder, loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_type_t result_type, loom_location_id_t location,
    loom_value_id_t* out_exec) {
  *out_exec = LOOM_VALUE_ID_INVALID;
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_spill_lowering_descriptor_ref(context->descriptor_set,
                                                descriptor_ref);
  loom_op_t* read_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, context->descriptor_set, descriptor,
      /*access_flags=*/0,
      /*operands=*/NULL, /*operand_count=*/0, loom_named_attr_slice_empty(),
      &result_type, 1, /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &read_op));
  *out_exec = loom_low_op_results(read_op).values[0];
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_spill_lowering_build_exec_write(
    const loom_amdgpu_spill_emission_context_t* context,
    loom_builder_t* builder, loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_value_id_t exec, loom_location_id_t location) {
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_spill_lowering_descriptor_ref(context->descriptor_set,
                                                descriptor_ref);
  loom_op_t* write_op = NULL;
  return loom_low_build_resolved_descriptor_op(
      builder, context->descriptor_set, descriptor,
      /*access_flags=*/0, &exec, 1, loom_named_attr_slice_empty(),
      /*result_types=*/NULL, /*result_count=*/0, /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &write_op);
}

static iree_status_t loom_amdgpu_spill_lowering_build_full_exec_write(
    const loom_amdgpu_spill_emission_context_t* context,
    loom_builder_t* builder, loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_location_id_t location) {
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_spill_lowering_descriptor_ref(context->descriptor_set,
                                                descriptor_ref);
  loom_op_t* write_op = NULL;
  return loom_low_build_resolved_descriptor_op(
      builder, context->descriptor_set, descriptor,
      /*access_flags=*/0,
      /*operands=*/NULL, /*operand_count=*/0, loom_named_attr_slice_empty(),
      /*result_types=*/NULL,
      /*result_count=*/0, /*tied_results=*/NULL, /*tied_result_count=*/0,
      location, &write_op);
}

static iree_status_t loom_amdgpu_spill_lowering_build_slice(
    loom_builder_t* builder, loom_value_id_t source, uint32_t chunk_start,
    loom_type_t result_type, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  loom_op_t* slice_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_slice_build(builder, source, chunk_start,
                                            result_type, location, &slice_op));
  *out_value = loom_low_slice_result(slice_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_spill_lowering_build_storage_address(
    loom_builder_t* builder, loom_value_id_t storage,
    uint64_t storage_byte_offset, loom_type_t result_type,
    loom_location_id_t location, loom_value_id_t* out_address) {
  *out_address = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_LE(storage_byte_offset, INT64_MAX);
  loom_op_t* address_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_storage_address_build(
      builder, storage, (int64_t)storage_byte_offset, result_type, location,
      &address_op));
  *out_address = loom_low_storage_address_result(address_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_spill_lowering_store_chunk(
    const loom_amdgpu_spill_emission_context_t* context,
    loom_builder_t* builder, loom_value_id_t address, loom_value_id_t value,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    const loom_amdgpu_spill_recipe_chunk_t* chunk,
    loom_location_id_t location) {
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_spill_lowering_descriptor_ref(context->descriptor_set,
                                                descriptor_ref);
  IREE_ASSERT(descriptor != NULL,
              "planned spill descriptor must exist in the descriptor set");
  const bool has_address = address != LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_EQ(
      has_address,
      iree_all_bits_set(chunk->flags,
                        LOOM_AMDGPU_SPILL_RECIPE_CHUNK_FLAG_STORAGE_ADDRESS));
  const loom_value_id_t operands[2] = {has_address ? address : value, value};
  const iree_host_size_t operand_count = has_address ? 2 : 1;
  loom_named_attr_t attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_make_offset_attr(
      context, chunk->packet_byte_offset, &attr));
  loom_op_t* store_op = NULL;
  return loom_low_build_resolved_descriptor_op(
      builder, context->descriptor_set, descriptor,
      /*access_flags=*/0, operands, operand_count,
      loom_make_named_attr_slice(&attr, 1),
      /*result_types=*/NULL,
      /*result_count=*/0, /*tied_results=*/NULL, /*tied_result_count=*/0,
      location, &store_op);
}

static iree_status_t loom_amdgpu_spill_lowering_load_chunk(
    const loom_amdgpu_spill_emission_context_t* context,
    loom_builder_t* builder, loom_value_id_t address, loom_type_t result_type,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    const loom_amdgpu_spill_recipe_chunk_t* chunk, loom_location_id_t location,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_spill_lowering_descriptor_ref(context->descriptor_set,
                                                descriptor_ref);
  IREE_ASSERT(descriptor != NULL,
              "planned reload descriptor must exist in the descriptor set");
  const bool has_address = address != LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_EQ(
      has_address,
      iree_all_bits_set(chunk->flags,
                        LOOM_AMDGPU_SPILL_RECIPE_CHUNK_FLAG_STORAGE_ADDRESS));
  const loom_value_id_t operands[1] = {address};
  const iree_host_size_t operand_count = has_address ? 1 : 0;
  loom_named_attr_t attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_make_offset_attr(
      context, chunk->packet_byte_offset, &attr));
  const loom_type_t result_types[] = {result_type};
  loom_op_t* load_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      builder, context->descriptor_set, descriptor,
      /*access_flags=*/0, operands, operand_count,
      loom_make_named_attr_slice(&attr, 1), result_types,
      IREE_ARRAYSIZE(result_types), /*tied_results=*/NULL,
      /*tied_result_count=*/0, location, &load_op));
  *out_value = loom_low_op_results(load_op).values[0];
  return iree_ok_status();
}

typedef struct loom_amdgpu_spill_recipe_emitter_t {
  // State shared by authored and directly planned recipe emission.
  loom_amdgpu_spill_emission_context_t* context;
  // Builder receiving emitted recipe operations.
  loom_builder_t* builder;
  // Compact recipe being interpreted.
  const loom_amdgpu_spill_recipe_t* recipe;
  // Storage handle referenced by address steps.
  loom_value_id_t storage;
  // Register value consumed by a store recipe.
  loom_value_id_t source;
  // Source location attached to every emitted operation.
  loom_location_id_t location;
  // Emitted SSA values and placement facts in dense recipe result order.
  loom_low_spill_materialized_value_t* emitted_values;
  // Number of entries allocated in |emitted_values|.
  iree_host_size_t value_capacity;
  // Number of initialized entries in |emitted_values|.
  iree_host_size_t emitted_value_count;
  // Completed reload chunks retained for a terminal concat.
  loom_value_id_t* loaded_chunks;
  // Number of initialized entries in |loaded_chunks|.
  iree_host_size_t loaded_chunk_count;
  // SSA value replacing the generic low.reload result.
  loom_value_id_t exported_result;
} loom_amdgpu_spill_recipe_emitter_t;

static loom_value_id_t loom_amdgpu_spill_recipe_emitter_value(
    const loom_amdgpu_spill_recipe_emitter_t* emitter,
    loom_amdgpu_spill_recipe_value_index_t value_index) {
  if (value_index == LOOM_AMDGPU_SPILL_RECIPE_VALUE_SOURCE) {
    IREE_ASSERT_NE(emitter->source, LOOM_VALUE_ID_INVALID);
    return emitter->source;
  }
  IREE_ASSERT_LT(value_index, emitter->emitted_value_count);
  return emitter->emitted_values[value_index].value_id;
}

static iree_status_t loom_amdgpu_spill_recipe_emitter_record_result(
    loom_amdgpu_spill_recipe_emitter_t* emitter,
    const loom_amdgpu_spill_recipe_step_t* step, loom_value_id_t value_id) {
  IREE_ASSERT_NE(step->result, LOOM_AMDGPU_SPILL_RECIPE_VALUE_NONE);
  IREE_ASSERT_EQ(step->result, emitter->emitted_value_count);
  IREE_ASSERT_LT(emitter->emitted_value_count, emitter->value_capacity);
  loom_low_spill_materialized_value_flags_t flags = 0;
  if (iree_any_bit_set(
          step->result_flags,
          LOOM_AMDGPU_SPILL_RECIPE_STEP_RESULT_FLAG_EXPANDED_EXECUTION)) {
    flags |= LOOM_LOW_SPILL_MATERIALIZED_VALUE_FLAG_EXPANDED_EXECUTION;
  }
  if (iree_any_bit_set(
          step->result_flags,
          LOOM_AMDGPU_SPILL_RECIPE_STEP_RESULT_FLAG_PRIVATE_TO_CONSUMER)) {
    flags |= LOOM_LOW_SPILL_MATERIALIZED_VALUE_FLAG_PRIVATE_TO_CONSUMER;
  }
  emitter->emitted_values[emitter->emitted_value_count++] =
      (loom_low_spill_materialized_value_t){
          .value_id = value_id,
          .flags = flags,
      };
  if (iree_any_bit_set(
          step->result_flags,
          LOOM_AMDGPU_SPILL_RECIPE_STEP_RESULT_FLAG_RELOAD_CHUNK)) {
    IREE_ASSERT_LT(emitter->loaded_chunk_count, emitter->recipe->chunk_count);
    emitter->loaded_chunks[emitter->loaded_chunk_count++] = value_id;
  }
  if (iree_any_bit_set(step->result_flags,
                       LOOM_AMDGPU_SPILL_RECIPE_STEP_RESULT_FLAG_EXPORTED)) {
    IREE_ASSERT_EQ(emitter->exported_result, LOOM_VALUE_ID_INVALID);
    emitter->exported_result = value_id;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_spill_recipe_emit_step(
    void* user_data, const loom_amdgpu_spill_recipe_step_t* step) {
  loom_amdgpu_spill_recipe_emitter_t* emitter =
      (loom_amdgpu_spill_recipe_emitter_t*)user_data;
  loom_amdgpu_spill_emission_context_t* context = emitter->context;
  loom_builder_t* builder = emitter->builder;
  loom_value_id_t result = LOOM_VALUE_ID_INVALID;
  const loom_type_t result_type =
      loom_amdgpu_spill_recipe_step_result_type(emitter->recipe, step);
  switch (step->kind) {
    case LOOM_AMDGPU_SPILL_RECIPE_STEP_EXEC_READ: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_build_exec_read(
          context, builder, step->descriptor_ref, result_type,
          emitter->location, &result));
      break;
    }
    case LOOM_AMDGPU_SPILL_RECIPE_STEP_EXEC_FULL: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_build_full_exec_write(
          context, builder, step->descriptor_ref, emitter->location));
      break;
    }
    case LOOM_AMDGPU_SPILL_RECIPE_STEP_SLICE: {
      IREE_ASSERT(step->chunk != NULL);
      IREE_ASSERT_EQ(step->operand_count, 1u);
      const loom_value_id_t source =
          loom_amdgpu_spill_recipe_emitter_value(emitter, step->operands[0]);
      IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_build_slice(
          builder, source, step->chunk->unit_start, result_type,
          emitter->location, &result));
      break;
    }
    case LOOM_AMDGPU_SPILL_RECIPE_STEP_SGPR_TO_VGPR: {
      IREE_ASSERT_EQ(step->operand_count, 1u);
      IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_build_register_convert(
          context, builder, step->descriptor_ref,
          loom_amdgpu_spill_recipe_emitter_value(emitter, step->operands[0]),
          result_type, emitter->location, &result));
      break;
    }
    case LOOM_AMDGPU_SPILL_RECIPE_STEP_STORAGE_ADDRESS:
      IREE_ASSERT(step->chunk != NULL);
      IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_build_storage_address(
          builder, emitter->storage, step->chunk->storage_byte_offset,
          result_type, emitter->location, &result));
      break;
    case LOOM_AMDGPU_SPILL_RECIPE_STEP_STORE: {
      IREE_ASSERT(step->chunk != NULL);
      IREE_ASSERT(step->operand_count == 1 || step->operand_count == 2);
      const bool has_address = step->operand_count == 2;
      const loom_value_id_t address =
          has_address ? loom_amdgpu_spill_recipe_emitter_value(
                            emitter, step->operands[0])
                      : LOOM_VALUE_ID_INVALID;
      const loom_value_id_t value = loom_amdgpu_spill_recipe_emitter_value(
          emitter, step->operands[has_address ? 1 : 0]);
      IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_store_chunk(
          context, builder, address, value, step->descriptor_ref, step->chunk,
          emitter->location));
      break;
    }
    case LOOM_AMDGPU_SPILL_RECIPE_STEP_LOAD: {
      IREE_ASSERT(step->chunk != NULL);
      IREE_ASSERT_LE(step->operand_count, 1u);
      const loom_value_id_t address =
          step->operand_count == 1 ? loom_amdgpu_spill_recipe_emitter_value(
                                         emitter, step->operands[0])
                                   : LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_load_chunk(
          context, builder, address, result_type, step->descriptor_ref,
          step->chunk, emitter->location, &result));
      break;
    }
    case LOOM_AMDGPU_SPILL_RECIPE_STEP_VGPR_TO_SGPR: {
      IREE_ASSERT_EQ(step->operand_count, 1u);
      IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_build_register_convert(
          context, builder, step->descriptor_ref,
          loom_amdgpu_spill_recipe_emitter_value(emitter, step->operands[0]),
          result_type, emitter->location, &result));
      break;
    }
    case LOOM_AMDGPU_SPILL_RECIPE_STEP_EXEC_RESTORE: {
      IREE_ASSERT_EQ(step->operand_count, 1u);
      IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_build_exec_write(
          context, builder, step->descriptor_ref,
          loom_amdgpu_spill_recipe_emitter_value(emitter, step->operands[0]),
          emitter->location));
      break;
    }
    case LOOM_AMDGPU_SPILL_RECIPE_STEP_CONCAT: {
      IREE_ASSERT_EQ(emitter->loaded_chunk_count, emitter->recipe->chunk_count);
      loom_op_t* concat_op = NULL;
      IREE_RETURN_IF_ERROR(loom_low_concat_build(
          builder, emitter->loaded_chunks, emitter->loaded_chunk_count,
          result_type, emitter->location, &concat_op));
      result = loom_low_concat_result(concat_op);
      break;
    }
    default:
      IREE_ASSERT_UNREACHABLE("invalid AMDGPU spill recipe step");
  }
  if (step->result == LOOM_AMDGPU_SPILL_RECIPE_VALUE_NONE) {
    IREE_ASSERT_EQ(result, LOOM_VALUE_ID_INVALID);
    return iree_ok_status();
  }
  IREE_ASSERT_NE(result, LOOM_VALUE_ID_INVALID);
  return loom_amdgpu_spill_recipe_emitter_record_result(emitter, step, result);
}

static iree_status_t loom_amdgpu_spill_lowering_emit_recipe(
    loom_amdgpu_spill_emission_context_t* context, loom_builder_t* builder,
    const loom_amdgpu_spill_recipe_t* recipe, loom_value_id_t storage,
    loom_value_id_t source, loom_location_id_t location,
    iree_arena_allocator_t* arena,
    loom_low_spill_materialized_action_t* out_action) {
  *out_action = (loom_low_spill_materialized_action_t){
      .value_id = LOOM_VALUE_ID_INVALID,
  };
  const iree_host_size_t value_capacity =
      (iree_host_size_t)recipe->chunk_count * 3u + 2u;
  loom_low_spill_materialized_value_t* emitted_values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, value_capacity, sizeof(*emitted_values), (void**)&emitted_values));
  loom_value_id_t* loaded_chunks = NULL;
  if (recipe->kind == LOOM_AMDGPU_SPILL_RECIPE_KIND_RELOAD) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, recipe->chunk_count,
                                                   sizeof(*loaded_chunks),
                                                   (void**)&loaded_chunks));
  }
  loom_amdgpu_spill_recipe_emitter_t emitter = {
      .context = context,
      .builder = builder,
      .recipe = recipe,
      .storage = storage,
      .source = source,
      .location = location,
      .emitted_values = emitted_values,
      .value_capacity = value_capacity,
      .loaded_chunks = loaded_chunks,
      .exported_result = LOOM_VALUE_ID_INVALID,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_recipe_visit(
      recipe, loom_amdgpu_spill_recipe_emit_step, &emitter));
  IREE_ASSERT_EQ(emitter.loaded_chunk_count,
                 recipe->kind == LOOM_AMDGPU_SPILL_RECIPE_KIND_RELOAD
                     ? recipe->chunk_count
                     : 0u);
  IREE_ASSERT_EQ(recipe->kind == LOOM_AMDGPU_SPILL_RECIPE_KIND_RELOAD,
                 emitter.exported_result != LOOM_VALUE_ID_INVALID);
  *out_action = (loom_low_spill_materialized_action_t){
      .value_id = emitter.exported_result,
      .emitted_values = emitter.emitted_values,
      .emitted_value_count = emitter.emitted_value_count,
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_spill_lowering_record_action_values(
    loom_amdgpu_spill_lowering_context_t* context,
    const loom_low_spill_materialized_action_t* action) {
  for (iree_host_size_t i = 0; i < action->emitted_value_count; ++i) {
    const loom_low_spill_materialized_value_t value = action->emitted_values[i];
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_spill_lowering_record_required_register_value(
            context, value.value_id));
    if (iree_any_bit_set(
            value.flags,
            LOOM_LOW_SPILL_MATERIALIZED_VALUE_FLAG_EXPANDED_EXECUTION)) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_spill_lowering_record_expanded_execution_value(
              context, value.value_id));
    }
    if (iree_any_bit_set(
            value.flags,
            LOOM_LOW_SPILL_MATERIALIZED_VALUE_FLAG_PRIVATE_TO_CONSUMER)) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_spill_lowering_record_per_user_placement_value(
              context, value.value_id));
    }
  }
  return iree_ok_status();
}

typedef struct loom_amdgpu_spill_materialization_traffic_plan_t {
  // Shared state required to emit each selected recipe.
  loom_amdgpu_spill_emission_context_t emission_context;
  // Recipes indexed by retained execution-order instruction.
  const loom_amdgpu_spill_recipe_t* recipes;
  // Number of entries in |recipes|.
  iree_host_size_t recipe_count;
} loom_amdgpu_spill_materialization_traffic_plan_t;

static iree_status_t loom_amdgpu_emit_planned_spill_materialization_traffic(
    void* user_data, iree_host_size_t instruction_index,
    loom_builder_t* builder, loom_value_id_t storage_value_id,
    loom_value_id_t source_value_id, loom_location_id_t location,
    iree_arena_allocator_t* arena,
    loom_low_spill_materialized_action_t* out_action) {
  loom_amdgpu_spill_materialization_traffic_plan_t* plan =
      (loom_amdgpu_spill_materialization_traffic_plan_t*)user_data;
  IREE_ASSERT_LT(instruction_index, plan->recipe_count);
  const loom_amdgpu_spill_recipe_t* recipe = &plan->recipes[instruction_index];
  IREE_ASSERT_NE(recipe->register_kind,
                 LOOM_AMDGPU_SPILL_REGISTER_KIND_UNSUPPORTED);
  return loom_amdgpu_spill_lowering_emit_recipe(
      &plan->emission_context, builder, recipe, storage_value_id,
      source_value_id, location, arena, out_action);
}

iree_status_t loom_amdgpu_plan_spill_materialization_traffic(
    void* user_data, const loom_low_allocation_table_t* table,
    const loom_low_allocation_spill_plan_t* spill_plans,
    iree_host_size_t spill_plan_count,
    const loom_low_spill_materialization_plan_t* materialization_plan,
    iree_arena_allocator_t* arena,
    loom_low_spill_materialization_traffic_plan_t* out_plan) {
  const loom_amdgpu_storage_layout_segment_sizes_t* segment_prefix = user_data;
  *out_plan = (loom_low_spill_materialization_traffic_plan_t){0};

  loom_amdgpu_spill_materialization_traffic_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*plan), (void**)&plan));
  *plan = (loom_amdgpu_spill_materialization_traffic_plan_t){0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_initialize_emission_context(
      table->module, table->target.descriptor_set, &plan->emission_context));

  loom_amdgpu_storage_layout_segment_sizes_t segment_sizes = *segment_prefix;
  for (iree_host_size_t i = 0; i < table->storage_layout.record_count; ++i) {
    loom_amdgpu_storage_layout_reservation_t projected_reservation = {0};
    IREE_RETURN_IF_ERROR(loom_amdgpu_storage_layout_project_reservation(
        &table->storage_layout.records[i].reservation, &segment_sizes,
        &projected_reservation));
  }

  loom_amdgpu_storage_layout_reservation_t* storage_reservations = NULL;
  if (spill_plan_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, spill_plan_count, sizeof(*storage_reservations),
        (void**)&storage_reservations));
  }
  for (iree_host_size_t i = 0; i < spill_plan_count; ++i) {
    if (!loom_low_spill_materialization_record_has_traffic(
            &materialization_plan->records[i])) {
      continue;
    }
    if (materialization_plan->slot_representatives != NULL) {
      const uint32_t representative =
          materialization_plan->slot_representatives[i];
      if (representative != i) {
        IREE_ASSERT_LT(representative, i);
        storage_reservations[i] = storage_reservations[representative];
        continue;
      }
    }
    const loom_low_allocation_spill_plan_t* spill_plan = &spill_plans[i];
    const loom_low_storage_layout_reservation_t source_reservation = {
        .space = loom_low_allocation_storage_space_for_spill_slot(
            spill_plan->slot_space),
        .byte_size = spill_plan->byte_size,
        .byte_alignment = spill_plan->byte_alignment,
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_storage_layout_project_reservation(
        &source_reservation, &segment_sizes, &storage_reservations[i]));
  }

  loom_amdgpu_spill_recipe_t* recipes = NULL;
  if (materialization_plan->instruction_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, materialization_plan->instruction_count, sizeof(*recipes),
        (void**)&recipes));
  }
  for (iree_host_size_t i = 0; i < materialization_plan->instruction_count;
       ++i) {
    const uint32_t action_index =
        loom_low_spill_materialization_instruction_action_index(
            &materialization_plan->instructions[i]);
    loom_low_spill_materialization_traffic_t traffic = {0};
    const bool has_traffic = loom_low_spill_materialization_plan_query_traffic(
        table->module, spill_plans, materialization_plan, action_index,
        &traffic);
    IREE_ASSERT(has_traffic);
    const loom_low_spill_materialization_action_t* action =
        &materialization_plan->actions[action_index];
    IREE_ASSERT_LT(action->spill_plan_index, spill_plan_count);
    const loom_amdgpu_spill_recipe_kind_t recipe_kind =
        traffic.kind == LOOM_LOW_SPILL_MATERIALIZATION_TRAFFIC_STORE
            ? LOOM_AMDGPU_SPILL_RECIPE_KIND_STORE
            : LOOM_AMDGPU_SPILL_RECIPE_KIND_RELOAD;
    const loom_amdgpu_storage_layout_reservation_t reservation =
        storage_reservations[action->spill_plan_index];
    const loom_amdgpu_storage_layout_reference_t storage_reference = {
        .reservation = reservation,
        .byte_offset = 0,
        .byte_length = reservation.byte_size,
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_spill_recipe_build(
        table->target.descriptor_set, recipe_kind, traffic.value_type,
        &storage_reference,
        loom_low_spill_materialization_traffic_byte_offset(
            &traffic, spill_plans[action->spill_plan_index].byte_size),
        arena, &recipes[i]));
    // Every AMDGPU register class outside SGPR, VGPR, and AGPR is unspillable.
    // Allocation-generated traffic must therefore have a complete recipe.
    IREE_ASSERT_NE(recipes[i].register_kind,
                   LOOM_AMDGPU_SPILL_REGISTER_KIND_UNSUPPORTED);
  }
  plan->recipes = recipes;
  plan->recipe_count = materialization_plan->instruction_count;
  *out_plan = (loom_low_spill_materialization_traffic_plan_t){
      .user_data = plan,
      .emit = loom_amdgpu_emit_planned_spill_materialization_traffic,
  };
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_spill_lowering_rewrite_spill(
    loom_amdgpu_spill_lowering_context_t* context, loom_rewriter_t* rewriter,
    loom_op_t* op, iree_arena_allocator_t* recipe_arena) {
  const loom_value_id_t value = loom_low_storage_store_value(op);
  const loom_type_t value_type =
      loom_module_value_type(context->emission.module, value);
  const loom_amdgpu_spill_register_kind_t register_kind =
      loom_amdgpu_spill_recipe_register_kind(context->emission.descriptor_set,
                                             value_type);
  if (register_kind == LOOM_AMDGPU_SPILL_REGISTER_KIND_UNSUPPORTED) {
    return loom_amdgpu_spill_lowering_emit_unsupported_register_type(
        context, op, value, value_type);
  }
  const uint32_t unit_count = loom_low_register_type_unit_count(value_type);

  const loom_value_id_t storage = loom_low_storage_store_storage(op);
  loom_amdgpu_storage_layout_reference_t storage_reference = {0};
  loom_amdgpu_spill_lowering_resolve_storage_reference(context, storage,
                                                       &storage_reference);
  if (!loom_amdgpu_spill_lowering_storage_space_supported(&storage_reference)) {
    return loom_amdgpu_spill_lowering_emit_unsupported_storage_space(
        context, op, storage, &storage_reference);
  }
  const uint64_t access_byte_length =
      (uint64_t)unit_count * LOOM_AMDGPU_SCRATCH_SPILL_UNIT_BYTES;
  bool access_supported = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_validate_access_range(
      context, op, storage, &storage_reference,
      loom_low_storage_transfer_offset(op), access_byte_length,
      &access_supported));
  if (!access_supported) {
    return iree_ok_status();
  }
  loom_amdgpu_spill_recipe_t recipe = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_recipe_build(
      context->emission.descriptor_set, LOOM_AMDGPU_SPILL_RECIPE_KIND_STORE,
      value_type, &storage_reference, loom_low_storage_transfer_offset(op),
      recipe_arena, &recipe));

  loom_builder_set_before(&rewriter->builder, op);
  // A per-lane definition only assigns the lanes active at its program point.
  // Preserve the other lanes already accumulated in their lane-private spill
  // slots by leaving EXEC unchanged. SGPR values describe one wave-wide value
  // and still require every lane's scratch slot to receive the same snapshot.
  loom_low_spill_materialized_action_t materialized_action = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_emit_recipe(
      &context->emission, &rewriter->builder, &recipe, storage, value,
      op->location, recipe_arena, &materialized_action));
  IREE_ASSERT_EQ(materialized_action.value_id, LOOM_VALUE_ID_INVALID);
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_record_action_values(
      context, &materialized_action));
  return loom_rewriter_erase(rewriter, op);
}

static iree_status_t loom_amdgpu_spill_lowering_rewrite_reload(
    loom_amdgpu_spill_lowering_context_t* context, loom_rewriter_t* rewriter,
    loom_op_t* op, iree_arena_allocator_t* recipe_arena) {
  const loom_value_id_t result = loom_low_storage_load_result(op);
  const loom_type_t result_type =
      loom_module_value_type(context->emission.module, result);
  const loom_amdgpu_spill_register_kind_t register_kind =
      loom_amdgpu_spill_recipe_register_kind(context->emission.descriptor_set,
                                             result_type);
  if (register_kind == LOOM_AMDGPU_SPILL_REGISTER_KIND_UNSUPPORTED) {
    return loom_amdgpu_spill_lowering_emit_unsupported_register_type(
        context, op, result, result_type);
  }
  const uint32_t unit_count = loom_low_register_type_unit_count(result_type);

  const loom_value_id_t storage = loom_low_storage_load_storage(op);
  loom_amdgpu_storage_layout_reference_t storage_reference = {0};
  loom_amdgpu_spill_lowering_resolve_storage_reference(context, storage,
                                                       &storage_reference);
  if (!loom_amdgpu_spill_lowering_storage_space_supported(&storage_reference)) {
    return loom_amdgpu_spill_lowering_emit_unsupported_storage_space(
        context, op, storage, &storage_reference);
  }
  const uint64_t access_byte_length =
      (uint64_t)unit_count * LOOM_AMDGPU_SCRATCH_SPILL_UNIT_BYTES;
  bool access_supported = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_validate_access_range(
      context, op, storage, &storage_reference,
      loom_low_storage_transfer_offset(op), access_byte_length,
      &access_supported));
  if (!access_supported) {
    return iree_ok_status();
  }
  loom_amdgpu_spill_recipe_t recipe = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_recipe_build(
      context->emission.descriptor_set, LOOM_AMDGPU_SPILL_RECIPE_KIND_RELOAD,
      result_type, &storage_reference, loom_low_storage_transfer_offset(op),
      recipe_arena, &recipe));

  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  // A per-lane reload defines only the lanes active at its program point,
  // matching other vector memory instructions. This also lets a backedge reload
  // share its destination with the prior loop state without clobbering retired
  // lanes. SGPR reloads still gather a wave-wide scalar snapshot through a
  // temporary VGPR and therefore require full EXEC.
  loom_low_spill_materialized_action_t materialized_action = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_emit_recipe(
      &context->emission, &rewriter->builder, &recipe, storage,
      LOOM_VALUE_ID_INVALID, op->location, recipe_arena, &materialized_action));
  const loom_value_id_t replacement = materialized_action.value_id;
  IREE_ASSERT_NE(replacement, LOOM_VALUE_ID_INVALID);
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_record_action_values(
      context, &materialized_action));
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  return loom_amdgpu_spill_lowering_record_private_result_placement(
      context, replacement);
}

static iree_status_t loom_amdgpu_spill_lowering_collect_ops(
    loom_region_t* body, iree_arena_allocator_t* arena, loom_op_t*** out_ops,
    iree_host_size_t* out_op_count) {
  *out_ops = NULL;
  *out_op_count = 0;
  iree_host_size_t op_capacity = 0;
  loom_block_t* block = NULL;
  loom_op_t* op = NULL;
  loom_region_for_each_block(body, block) {
    loom_block_for_each_op(block, op) {
      if (!loom_low_storage_store_isa(op) && !loom_low_storage_load_isa(op)) {
        continue;
      }
      const iree_host_size_t minimum_capacity = *out_op_count + 1;
      if (minimum_capacity > op_capacity) {
        IREE_RETURN_IF_ERROR(iree_arena_grow_array(
            arena, *out_op_count, iree_max(minimum_capacity, 8u),
            sizeof(**out_ops), &op_capacity, (void**)out_ops));
      }
      (*out_ops)[(*out_op_count)++] = op;
    }
  }
  return iree_ok_status();
}

// Spill lowering precedes schedule construction. Build its private storage
// projection only after spill discovery proves that rewriting is required.
static iree_status_t loom_amdgpu_spill_lowering_build_storage_layout(
    const loom_module_t* module, const loom_region_t* body,
    loom_amdgpu_storage_layout_segment_sizes_t segment_prefix,
    iree_arena_allocator_t* arena,
    loom_amdgpu_storage_layout_t* out_storage_layout) {
  loom_low_storage_layout_builder_t builder;
  loom_low_storage_layout_builder_initialize(&builder);
  const loom_block_t* block = NULL;
  const loom_op_t* op = NULL;
  loom_region_for_each_block(body, block) {
    loom_block_for_each_op(block, op) {
      if (!loom_low_storage_reserve_isa(op) && !loom_low_storage_view_isa(op)) {
        continue;
      }
      IREE_RETURN_IF_ERROR(
          loom_low_storage_layout_builder_append(module, op, arena, &builder));
    }
  }
  loom_low_storage_layout_t source_layout;
  IREE_RETURN_IF_ERROR(
      loom_low_storage_layout_builder_finish(&builder, arena, &source_layout));
  return loom_amdgpu_storage_layout_build(&source_layout, segment_prefix, arena,
                                          out_storage_layout);
}

static iree_status_t loom_amdgpu_spill_lowering_initialize_context(
    loom_module_t* module, loom_op_t* function_op, loom_region_t* body,
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_storage_layout_segment_sizes_t segment_prefix,
    iree_diagnostic_emitter_t emitter,
    loom_amdgpu_spill_lowering_result_t* result,
    iree_arena_allocator_t* scratch_arena,
    loom_amdgpu_spill_lowering_context_t* out_context) {
  *out_context = (loom_amdgpu_spill_lowering_context_t){
      .function_op = function_op,
      .emitter = emitter,
      .result = result,
      .scratch_arena = scratch_arena,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_initialize_emission_context(
      module, descriptor_set, &out_context->emission));
  return loom_amdgpu_spill_lowering_build_storage_layout(
      module, body, segment_prefix, scratch_arena,
      &out_context->storage_layout);
}

iree_status_t loom_amdgpu_lower_spill_traffic(
    loom_module_t* module, loom_op_t* function_op,
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_storage_layout_segment_sizes_t segment_prefix,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* scratch_arena,
    loom_amdgpu_spill_lowering_result_t* out_result) {
  if (module == NULL || function_op == NULL || descriptor_set == NULL ||
      out_result == NULL || scratch_arena == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU spill lowering requires module, function, descriptor set, "
        "result, and scratch arena");
  }
  *out_result = (loom_amdgpu_spill_lowering_result_t){0};
  loom_region_t* body = loom_low_function_body(function_op);
  if (body == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU spill lowering requires a low function body");
  }

  loom_op_t** ops = NULL;
  iree_host_size_t op_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_collect_ops(
      body, scratch_arena, &ops, &op_count));
  if (op_count == 0) {
    return iree_ok_status();
  }

  loom_amdgpu_spill_lowering_context_t context = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_spill_lowering_initialize_context(
      module, function_op, body, descriptor_set, segment_prefix, emitter,
      out_result, scratch_arena, &context));

  loom_rewriter_t rewriter = {0};
  loom_rewriter_initialize(&rewriter, module, scratch_arena);
  iree_arena_allocator_t recipe_arena;
  iree_arena_initialize(scratch_arena->block_pool, &recipe_arena);
  // Generated packet helpers cannot recursively spill. The frame loop retains
  // already-spilled producers separately; an existing source of an authored or
  // CFG-edge spill can still have other uses that benefit from spilling.
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < op_count && iree_status_is_ok(status); ++i) {
    iree_arena_reset(&recipe_arena);
    if (loom_low_storage_store_isa(ops[i])) {
      status = loom_amdgpu_spill_lowering_rewrite_spill(&context, &rewriter,
                                                        ops[i], &recipe_arena);
    } else if (loom_low_storage_load_isa(ops[i])) {
      status = loom_amdgpu_spill_lowering_rewrite_reload(&context, &rewriter,
                                                         ops[i], &recipe_arena);
    }
  }
  iree_arena_deinitialize(&recipe_arena);
  loom_rewriter_deinitialize(&rewriter);
  if (iree_status_is_ok(status)) {
    out_result->lowered_traffic = true;
    out_result->required_register_value_ids =
        context.required_register_value_ids;
    out_result->required_register_value_count =
        context.required_register_value_count;
    out_result->expanded_execution_value_ids =
        context.expanded_execution_value_ids;
    out_result->expanded_execution_value_count =
        context.expanded_execution_value_count;
    out_result->per_user_placement_value_ids =
        context.per_user_placement_value_ids;
    out_result->per_user_placement_value_count =
        context.per_user_placement_value_count;
  }
  return status;
}
