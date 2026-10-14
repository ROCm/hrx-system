// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/abi/task/parameter_layout.h"

#include "iree/base/alignment.h"
#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ir/module.h"

static uint8_t loom_task_parameter_size(loom_type_t type,
                                        uint32_t* out_alignment) {
  const loom_scalar_type_t element_type = loom_type_element_type(type);
  const uint32_t element_bits = element_type == LOOM_SCALAR_TYPE_INDEX ||
                                        element_type == LOOM_SCALAR_TYPE_OFFSET
                                    ? 64
                                    : loom_scalar_type_bitwidth(element_type);
  const uint32_t element_bytes = (element_bits + 7) / 8;
  uint64_t element_count = 1;
  if (!(loom_type_is_scalar(type) ||
        (loom_type_is_vector(type) &&
         loom_type_static_element_count(type, &element_count))) ||
      !element_bytes || !element_count ||
      element_count > UINT8_MAX / element_bytes) {
    return 0;
  }
  *out_alignment = iree_min(element_bytes, 8);
  return (uint8_t)(element_count * element_bytes);
}

static iree_status_t loom_task_parameter_layout_reject(
    const loom_op_t* source_op, iree_diagnostic_emitter_t emitter,
    iree_string_view_t constraint) {
  const loom_diagnostic_param_t params[] = {loom_param_string(constraint)};
  return iree_diagnostic_emit(emitter,
                              &(loom_diagnostic_emission_t){
                                  .op = source_op,
                                  .error = LOOM_ERR_BACKEND_052,
                                  .params = params,
                                  .param_count = IREE_ARRAYSIZE(params),
                              });
}

static iree_status_t loom_task_parameter_layout_reject_parameter(
    const loom_op_t* source_op, iree_diagnostic_emitter_t emitter,
    uint16_t index, loom_type_t type, iree_string_view_t constraint) {
  const loom_diagnostic_param_t params[] = {loom_param_i64(index),
                                            loom_param_type(type),
                                            loom_param_string(constraint)};
  return iree_diagnostic_emit(emitter,
                              &(loom_diagnostic_emission_t){
                                  .op = source_op,
                                  .error = LOOM_ERR_BACKEND_053,
                                  .params = params,
                                  .param_count = IREE_ARRAYSIZE(params),
                              });
}

// A signature carries logical types in source order; offsets explicitly place
// each value in either the binding table or the byte-addressed constant table.
// This contract remains meaningful after dead parameter uses disappear.
iree_status_t loom_task_parameter_layout_parse(
    const loom_module_t* module, const loom_op_t* source_op,
    loom_named_attr_slice_t layout, iree_diagnostic_emitter_t emitter,
    iree_arena_allocator_t* arena, bool* out_accepted,
    loom_task_parameter_layout_t* out_abi) {
  *out_accepted = false;
  *out_abi = (loom_task_parameter_layout_t){0};
  const loom_func_type_data_t* signature = NULL;
  const loom_attribute_t* offsets = NULL;
  for (uint16_t i = 0; i < layout.count; ++i) {
    const loom_named_attr_t* field = &layout.entries[i];
    const iree_string_view_t key =
        loom_string_table_get(&module->strings, field->name_id);
    if (iree_string_view_equal(key, IREE_SV("signature")) &&
        field->value.kind == LOOM_ATTR_TYPE) {
      const loom_type_t type =
          loom_type_table_get(&module->types, field->value.type_id);
      if (loom_type_is_function(type)) {
        signature = loom_type_func_data(type);
      }
    } else if (iree_string_view_equal(key, IREE_SV("offsets")) &&
               field->value.kind == LOOM_ATTR_I64_ARRAY) {
      offsets = &field->value;
    } else {
      const loom_diagnostic_param_t params[] = {
          loom_param_string(key),
          loom_param_string(IREE_SV("'signature' with a function type or "
                                    "'offsets' with an integer array")),
      };
      return iree_diagnostic_emit(emitter,
                                  &(loom_diagnostic_emission_t){
                                      .op = source_op,
                                      .error = LOOM_ERR_BACKEND_054,
                                      .params = params,
                                      .param_count = IREE_ARRAYSIZE(params),
                                  });
    }
  }
  if (!signature || signature->result_count || !offsets ||
      offsets->count != signature->arg_count) {
    return loom_task_parameter_layout_reject(
        source_op, emitter,
        IREE_SV("a void logical signature and one offset "
                "per parameter"));
  }
  iree_hal_executable_dispatch_parameter_v0_t* rows = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, signature->arg_count,
                                                 sizeof(*rows), (void**)&rows));
  loom_task_parameter_layout_t abi = {
      .attributes = {.workgroup_size_x = 1,
                     .workgroup_size_y = 1,
                     .workgroup_size_z = 1,
                     .parameter_count = signature->arg_count},
      .parameters = rows,
  };
  bool accepted = true;
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0;
       i < signature->arg_count && accepted && iree_status_is_ok(status); ++i) {
    const loom_type_t type = signature->types[i];
    const int64_t offset = offsets->i64_array[i];
    rows[i] = (iree_hal_executable_dispatch_parameter_v0_t){.name = UINT16_MAX};
    if (loom_type_is_buffer(type)) {
      if (offset < 0 || offset >= IREE_HAL_EXECUTABLE_MAX_BINDING_COUNT) {
        accepted = false;
        status = loom_task_parameter_layout_reject_parameter(
            source_op, emitter, i, type,
            IREE_SV("a nonnegative binding ordinal within the dispatch binding "
                    "table"));
      } else {
        rows[i].type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_BINDING;
        rows[i].offset = (uint16_t)offset;
        abi.attributes.binding_count =
            (uint8_t)iree_max(abi.attributes.binding_count, offset + 1);
      }
    } else {
      uint32_t alignment = 0;
      const uint8_t size = loom_task_parameter_size(type, &alignment);
      if (!size || offset < 0 ||
          offset >
              (int64_t)IREE_HAL_EXECUTABLE_MAX_CONSTANT_BYTE_LENGTH - size) {
        accepted = false;
        status = loom_task_parameter_layout_reject_parameter(
            source_op, emitter, i, type,
            IREE_SV(
                "a byte-addressable scalar or static vector with a "
                "nonnegative offset fitting the dispatch constant segment"));
      } else {
        rows[i].type = IREE_HAL_EXECUTABLE_DISPATCH_PARAM_TYPE_V0_CONSTANT;
        rows[i].size = size;
        rows[i].offset = (uint16_t)offset;
        abi.attributes.constant_byte_length =
            iree_max(abi.attributes.constant_byte_length,
                     (uint32_t)rows[i].offset + rows[i].size);
      }
    }
  }
  if (iree_status_is_ok(status) && accepted) {
    *out_accepted = true;
    *out_abi = abi;
  }
  return status;
}

iree_status_t loom_task_parameter_layout_build(
    loom_module_t* module, const loom_op_t* source_op, const loom_type_t* types,
    uint16_t count, iree_diagnostic_emitter_t emitter,
    iree_arena_allocator_t* scratch_arena, bool* out_accepted,
    loom_named_attr_slice_t* out_layout) {
  *out_accepted = false;
  *out_layout = loom_named_attr_slice_empty();
  int64_t* offsets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, count, sizeof(*offsets), (void**)&offsets));
  uint32_t binding_count = 0;
  uint32_t constant_bytes = 0;
  bool accepted = true;
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < count && accepted && iree_status_is_ok(status);
       ++i) {
    if (loom_type_is_buffer(types[i])) {
      offsets[i] = binding_count++;
    } else {
      uint32_t alignment = 0;
      const uint8_t size = loom_task_parameter_size(types[i], &alignment);
      if (!size) {
        accepted = false;
        status = loom_task_parameter_layout_reject_parameter(
            source_op, emitter, i, types[i],
            IREE_SV("a buffer, byte-addressable scalar, or static vector"));
      } else {
        constant_bytes = (uint32_t)iree_host_align(constant_bytes, alignment);
        offsets[i] = constant_bytes;
        constant_bytes += size;
      }
    }
  }
  if (!iree_status_is_ok(status) || !accepted) {
    return status;
  }
  if (binding_count > IREE_HAL_EXECUTABLE_MAX_BINDING_COUNT ||
      constant_bytes > IREE_HAL_EXECUTABLE_MAX_CONSTANT_BYTE_LENGTH) {
    return loom_task_parameter_layout_reject(
        source_op, emitter,
        IREE_SV("a parameter layout fitting the dispatch binding and constant "
                "limits"));
  }
  loom_type_t signature;
  IREE_RETURN_IF_ERROR(loom_module_intern_function_type(module, types, count,
                                                        NULL, 0, &signature));
  loom_named_attr_t fields[2];
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, IREE_SV("offsets"),
                                                 &fields[0].name_id));
  fields[0].value = loom_attr_i64_array(offsets, count);
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, IREE_SV("signature"),
                                                 &fields[1].name_id));
  fields[1].value =
      loom_attr_type(loom_module_lookup_type_id(module, signature));
  loom_attribute_t layout;
  IREE_RETURN_IF_ERROR(loom_module_make_canonical_attr_dict(
      module, loom_make_named_attr_slice(fields, 2), &layout));
  *out_layout = loom_attr_as_dict(layout);
  *out_accepted = true;
  return iree_ok_status();
}
