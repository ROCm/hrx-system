// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/hal/kernel_abi_layout.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/function.h"
#include "loom/codegen/low/representation_binding.h"
#include "loom/error/parameter.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/registers.h"

static iree_string_view_t loom_amdgpu_hal_kernel_abi_module_string(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return loom_string_table_get(&module->strings, string_id);
}

static loom_type_t loom_amdgpu_hal_kernel_abi_type_attr(
    const loom_module_t* module, loom_type_id_t type_id) {
  if (type_id == LOOM_TYPE_ID_INVALID || type_id >= module->types.count) {
    return loom_type_none();
  }
  return loom_type_table_get(&module->types, type_id);
}

static uint32_t loom_amdgpu_hal_kernel_abi_direct_arg_size_from_type(
    loom_type_t type) {
  const uint32_t unit_count = loom_low_register_type_unit_count(type);
  return unit_count * LOOM_AMDGPU_HAL_KERNEL_ABI_DIRECT_SCALAR_KERNARG_SIZE;
}

static iree_status_t loom_amdgpu_hal_kernel_abi_format_resource_name(
    uint32_t binding_index, iree_arena_allocator_t* arena,
    iree_string_view_t* out_name) {
  char scratch[32];
  int length =
      iree_snprintf(scratch, sizeof(scratch), "binding%" PRIu32, binding_index);
  IREE_ASSERT(length >= 0 && (iree_host_size_t)length < sizeof(scratch));
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, (iree_host_size_t)length, (void**)&storage));
  memcpy(storage, scratch, (iree_host_size_t)length);
  *out_name = iree_make_string_view(storage, (iree_host_size_t)length);
  return iree_ok_status();
}

static iree_string_view_t loom_amdgpu_hal_kernel_abi_value_name(
    const loom_module_t* module, loom_value_id_t value_id) {
  if (value_id >= module->values.count) {
    return iree_string_view_empty();
  }
  const loom_string_id_t name_id = loom_module_value(module, value_id)->name_id;
  return loom_amdgpu_hal_kernel_abi_module_string(module, name_id);
}

static const loom_attribute_t* loom_amdgpu_hal_kernel_abi_find_layout_attr(
    loom_named_attr_slice_t attrs, loom_string_id_t key_id) {
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    if (attrs.entries[i].name_id == key_id) {
      return &attrs.entries[i].value;
    }
  }
  return NULL;
}

static void loom_amdgpu_hal_kernel_abi_direct_arg_from_block_arg(
    const loom_module_t* module, const loom_block_t* entry_block,
    uint16_t argument_index, uint32_t parameter_index, uint32_t kernarg_offset,
    loom_amdgpu_hal_kernarg_direct_arg_t* out_direct_arg) {
  const loom_value_id_t arg_id = loom_block_arg_id(entry_block, argument_index);
  const loom_type_t abi_type = loom_module_value_type(module, arg_id);
  const uint32_t kernarg_size =
      loom_amdgpu_hal_kernel_abi_direct_arg_size_from_type(abi_type);
  *out_direct_arg = (loom_amdgpu_hal_kernarg_direct_arg_t){
      .arg_id = arg_id,
      .name = loom_amdgpu_hal_kernel_abi_value_name(module, arg_id),
      .parameter_index = parameter_index,
      .argument_index = argument_index,
      .kernarg_offset = kernarg_offset,
      .kernarg_size = kernarg_size,
      .payload_size = kernarg_size,
      .kernarg_alignment =
          LOOM_AMDGPU_HAL_KERNEL_ABI_DIRECT_SCALAR_KERNARG_ALIGNMENT,
      .abi_type = abi_type,
  };
  return;
}

typedef struct loom_amdgpu_hal_kernel_abi_layout_attr_keys_t {
  // Interned constant count field name.
  loom_string_id_t constant_count;
  // Interned direct arg count field name.
  loom_string_id_t direct_arg_count;
  // Interned direct arg names field name.
  loom_string_id_t direct_arg_names;
  // Interned direct arg offsets field name.
  loom_string_id_t direct_arg_offsets;
  // Interned direct arg parameter indices field name.
  loom_string_id_t direct_arg_parameter_indices;
  // Interned direct arg payload sizes field name.
  loom_string_id_t direct_arg_payload_sizes;
  // Interned direct arg sizes field name.
  loom_string_id_t direct_arg_sizes;
  // Interned parameter count field name.
  loom_string_id_t parameter_count;
  // Interned resource count field name.
  loom_string_id_t resource_count;
  // Interned resource offsets field name.
  loom_string_id_t resource_offsets;
  // Interned resource parameter indices field name.
  loom_string_id_t resource_parameter_indices;
  // Interned uses kernarg segment ptr field name.
  loom_string_id_t uses_kernarg_segment_ptr;
} loom_amdgpu_hal_kernel_abi_layout_attr_keys_t;

static iree_status_t loom_amdgpu_hal_kernel_abi_intern_layout_attr_keys(
    loom_module_t* module,
    loom_amdgpu_hal_kernel_abi_layout_attr_keys_t* out_keys) {
  *out_keys = (loom_amdgpu_hal_kernel_abi_layout_attr_keys_t){0};
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("constant_count"), &out_keys->constant_count));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("direct_arg_count"), &out_keys->direct_arg_count));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("direct_arg_names"), &out_keys->direct_arg_names));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("direct_arg_offsets"), &out_keys->direct_arg_offsets));
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(module, IREE_SV("direct_arg_parameter_indices"),
                                &out_keys->direct_arg_parameter_indices));
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(module, IREE_SV("direct_arg_payload_sizes"),
                                &out_keys->direct_arg_payload_sizes));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("direct_arg_sizes"), &out_keys->direct_arg_sizes));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("parameter_count"), &out_keys->parameter_count));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("resource_count"), &out_keys->resource_count));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, IREE_SV("resource_offsets"), &out_keys->resource_offsets));
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(module, IREE_SV("resource_parameter_indices"),
                                &out_keys->resource_parameter_indices));
  return loom_module_intern_string(module, IREE_SV("uses_kernarg_segment_ptr"),
                                   &out_keys->uses_kernarg_segment_ptr);
}

static iree_status_t loom_amdgpu_hal_kernel_abi_format_layout_entry_key(
    loom_module_t* module, iree_string_view_t prefix, iree_host_size_t ordinal,
    loom_string_id_t* out_key_id) {
  char scratch[64];
  int length = iree_snprintf(scratch, sizeof(scratch), "%.*s%" PRIhsz,
                             (int)prefix.size, prefix.data, ordinal);
  IREE_ASSERT(length >= 0 && (iree_host_size_t)length < sizeof(scratch));
  return loom_module_intern_string(
      module, iree_make_string_view(scratch, (iree_host_size_t)length),
      out_key_id);
}

static iree_status_t loom_amdgpu_hal_kernel_abi_make_string_attr(
    loom_module_t* module, iree_string_view_t value,
    loom_attribute_t* out_attr) {
  loom_string_id_t string_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, value, &string_id));
  *out_attr = loom_attr_string(string_id);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_hal_kernel_abi_make_direct_arg_names_attr(
    loom_module_t* module, const loom_amdgpu_hal_kernel_abi_layout_t* layout,
    iree_arena_allocator_t* scratch_arena, loom_attribute_t* out_attr) {
  loom_named_attr_t* entries = NULL;
  if (layout->direct_arg_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(scratch_arena, layout->direct_arg_count,
                                  sizeof(*entries), (void**)&entries));
    memset(entries, 0, layout->direct_arg_count * sizeof(*entries));
  }
  for (iree_host_size_t i = 0; i < layout->direct_arg_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_format_layout_entry_key(
        module, IREE_SV("arg"), i, &entries[i].name_id));
    IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_make_string_attr(
        module, layout->direct_args[i].name, &entries[i].value));
  }
  return loom_module_make_canonical_attr_dict(
      module, loom_make_named_attr_slice(entries, layout->direct_arg_count),
      out_attr);
}

static iree_status_t loom_amdgpu_hal_kernel_abi_make_direct_arg_sizes_attr(
    const loom_amdgpu_hal_kernel_abi_layout_t* layout,
    iree_arena_allocator_t* scratch_arena, loom_attribute_t* out_attr) {
  IREE_ASSERT_LE(layout->direct_arg_count, (iree_host_size_t)UINT16_MAX);
  int64_t* entries = NULL;
  if (layout->direct_arg_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(scratch_arena, layout->direct_arg_count,
                                  sizeof(*entries), (void**)&entries));
  }
  for (iree_host_size_t i = 0; i < layout->direct_arg_count; ++i) {
    entries[i] = layout->direct_args[i].kernarg_size;
  }
  *out_attr = loom_attr_i64_array(entries, (uint16_t)layout->direct_arg_count);
  return iree_ok_status();
}

static bool loom_amdgpu_hal_kernel_abi_has_distinct_direct_arg_payload_sizes(
    const loom_amdgpu_hal_kernel_abi_layout_t* layout) {
  for (iree_host_size_t i = 0; i < layout->direct_arg_count; ++i) {
    if (layout->direct_args[i].payload_size !=
        layout->direct_args[i].kernarg_size) {
      return true;
    }
  }
  return false;
}

static iree_status_t
loom_amdgpu_hal_kernel_abi_make_direct_arg_payload_sizes_attr(
    const loom_amdgpu_hal_kernel_abi_layout_t* layout,
    iree_arena_allocator_t* scratch_arena, loom_attribute_t* out_attr) {
  IREE_ASSERT_LE(layout->direct_arg_count, (iree_host_size_t)UINT16_MAX);
  int64_t* entries = NULL;
  if (layout->direct_arg_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(scratch_arena, layout->direct_arg_count,
                                  sizeof(*entries), (void**)&entries));
  }
  for (iree_host_size_t i = 0; i < layout->direct_arg_count; ++i) {
    entries[i] = layout->direct_args[i].payload_size;
  }
  *out_attr = loom_attr_i64_array(entries, (uint16_t)layout->direct_arg_count);
  return iree_ok_status();
}

typedef enum loom_amdgpu_hal_kernel_abi_layout_array_kind_e {
  LOOM_AMDGPU_HAL_KERNEL_ABI_LAYOUT_ARRAY_RESOURCE_PARAMETER_INDICES = 0,
  LOOM_AMDGPU_HAL_KERNEL_ABI_LAYOUT_ARRAY_RESOURCE_OFFSETS = 1,
  LOOM_AMDGPU_HAL_KERNEL_ABI_LAYOUT_ARRAY_DIRECT_ARG_PARAMETER_INDICES = 2,
  LOOM_AMDGPU_HAL_KERNEL_ABI_LAYOUT_ARRAY_DIRECT_ARG_OFFSETS = 3,
} loom_amdgpu_hal_kernel_abi_layout_array_kind_t;

static iree_host_size_t loom_amdgpu_hal_kernel_abi_layout_array_count(
    const loom_amdgpu_hal_kernel_abi_layout_t* layout,
    loom_amdgpu_hal_kernel_abi_layout_array_kind_t kind) {
  switch (kind) {
    case LOOM_AMDGPU_HAL_KERNEL_ABI_LAYOUT_ARRAY_RESOURCE_PARAMETER_INDICES:
    case LOOM_AMDGPU_HAL_KERNEL_ABI_LAYOUT_ARRAY_RESOURCE_OFFSETS:
      return layout->resource_count;
    case LOOM_AMDGPU_HAL_KERNEL_ABI_LAYOUT_ARRAY_DIRECT_ARG_PARAMETER_INDICES:
    case LOOM_AMDGPU_HAL_KERNEL_ABI_LAYOUT_ARRAY_DIRECT_ARG_OFFSETS:
      return layout->direct_arg_count;
  }
  IREE_ASSERT_UNREACHABLE("unknown AMDGPU HAL ABI layout array kind");
  IREE_BUILTIN_UNREACHABLE();
}

static uint32_t loom_amdgpu_hal_kernel_abi_layout_array_value(
    const loom_amdgpu_hal_kernel_abi_layout_t* layout,
    loom_amdgpu_hal_kernel_abi_layout_array_kind_t kind,
    iree_host_size_t index) {
  switch (kind) {
    case LOOM_AMDGPU_HAL_KERNEL_ABI_LAYOUT_ARRAY_RESOURCE_PARAMETER_INDICES:
      return layout->resources[index].parameter_index;
    case LOOM_AMDGPU_HAL_KERNEL_ABI_LAYOUT_ARRAY_RESOURCE_OFFSETS:
      return layout->resources[index].kernarg_offset;
    case LOOM_AMDGPU_HAL_KERNEL_ABI_LAYOUT_ARRAY_DIRECT_ARG_PARAMETER_INDICES:
      return layout->direct_args[index].parameter_index;
    case LOOM_AMDGPU_HAL_KERNEL_ABI_LAYOUT_ARRAY_DIRECT_ARG_OFFSETS:
      return layout->direct_args[index].kernarg_offset;
  }
  IREE_ASSERT_UNREACHABLE("unknown AMDGPU HAL ABI layout array kind");
  IREE_BUILTIN_UNREACHABLE();
}

static iree_status_t loom_amdgpu_hal_kernel_abi_make_layout_u32_array_attr(
    const loom_amdgpu_hal_kernel_abi_layout_t* layout,
    loom_amdgpu_hal_kernel_abi_layout_array_kind_t kind,
    iree_arena_allocator_t* scratch_arena, loom_attribute_t* out_attr) {
  const iree_host_size_t count =
      loom_amdgpu_hal_kernel_abi_layout_array_count(layout, kind);
  IREE_ASSERT_LE(count, (iree_host_size_t)UINT16_MAX);
  int64_t* entries = NULL;
  if (count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, count, sizeof(*entries), (void**)&entries));
  }
  for (iree_host_size_t i = 0; i < count; ++i) {
    entries[i] = loom_amdgpu_hal_kernel_abi_layout_array_value(layout, kind, i);
  }
  *out_attr = loom_attr_i64_array(entries, (uint16_t)count);
  return iree_ok_status();
}

bool loom_amdgpu_hal_kernel_abi_has_layout_attr(const loom_op_t* function_op) {
  return loom_low_kernel_def_isa(function_op) &&
         loom_low_kernel_def_has_abi_layout(function_op);
}

iree_status_t loom_amdgpu_hal_kernel_abi_make_layout_attr(
    loom_module_t* module, const loom_amdgpu_hal_kernel_abi_layout_t* layout,
    iree_arena_allocator_t* scratch_arena, loom_attribute_t* out_attr) {
  loom_amdgpu_hal_kernel_abi_layout_attr_keys_t keys = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_hal_kernel_abi_intern_layout_attr_keys(module, &keys));

  loom_attribute_t direct_arg_names_attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_make_direct_arg_names_attr(
      module, layout, scratch_arena, &direct_arg_names_attr));
  loom_attribute_t direct_arg_offsets_attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_make_layout_u32_array_attr(
      layout, LOOM_AMDGPU_HAL_KERNEL_ABI_LAYOUT_ARRAY_DIRECT_ARG_OFFSETS,
      scratch_arena, &direct_arg_offsets_attr));
  loom_attribute_t direct_arg_parameter_indices_attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_make_layout_u32_array_attr(
      layout,
      LOOM_AMDGPU_HAL_KERNEL_ABI_LAYOUT_ARRAY_DIRECT_ARG_PARAMETER_INDICES,
      scratch_arena, &direct_arg_parameter_indices_attr));
  loom_attribute_t direct_arg_sizes_attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_make_direct_arg_sizes_attr(
      layout, scratch_arena, &direct_arg_sizes_attr));
  const bool has_distinct_direct_arg_payload_sizes =
      loom_amdgpu_hal_kernel_abi_has_distinct_direct_arg_payload_sizes(layout);
  loom_attribute_t direct_arg_payload_sizes_attr = {0};
  if (has_distinct_direct_arg_payload_sizes) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_hal_kernel_abi_make_direct_arg_payload_sizes_attr(
            layout, scratch_arena, &direct_arg_payload_sizes_attr));
  }
  loom_attribute_t resource_offsets_attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_make_layout_u32_array_attr(
      layout, LOOM_AMDGPU_HAL_KERNEL_ABI_LAYOUT_ARRAY_RESOURCE_OFFSETS,
      scratch_arena, &resource_offsets_attr));
  loom_attribute_t resource_parameter_indices_attr = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_make_layout_u32_array_attr(
      layout,
      LOOM_AMDGPU_HAL_KERNEL_ABI_LAYOUT_ARRAY_RESOURCE_PARAMETER_INDICES,
      scratch_arena, &resource_parameter_indices_attr));

  loom_named_attr_t entries[12] = {0};
  iree_host_size_t entry_count = 0;
  entries[entry_count++] = (loom_named_attr_t){
      .name_id = keys.constant_count,
      .value = loom_attr_i64(layout->constant_count),
  };
  entries[entry_count++] = (loom_named_attr_t){
      .name_id = keys.direct_arg_count,
      .value = loom_attr_i64(layout->direct_arg_count),
  };
  entries[entry_count++] = (loom_named_attr_t){
      .name_id = keys.direct_arg_names,
      .value = direct_arg_names_attr,
  };
  entries[entry_count++] = (loom_named_attr_t){
      .name_id = keys.direct_arg_offsets,
      .value = direct_arg_offsets_attr,
  };
  entries[entry_count++] = (loom_named_attr_t){
      .name_id = keys.direct_arg_parameter_indices,
      .value = direct_arg_parameter_indices_attr,
  };
  entries[entry_count++] = (loom_named_attr_t){
      .name_id = keys.direct_arg_sizes,
      .value = direct_arg_sizes_attr,
  };
  entries[entry_count++] = (loom_named_attr_t){
      .name_id = keys.parameter_count,
      .value = loom_attr_i64(layout->parameter_count),
  };
  entries[entry_count++] = (loom_named_attr_t){
      .name_id = keys.resource_count,
      .value = loom_attr_i64(layout->resource_count),
  };
  entries[entry_count++] = (loom_named_attr_t){
      .name_id = keys.resource_offsets,
      .value = resource_offsets_attr,
  };
  entries[entry_count++] = (loom_named_attr_t){
      .name_id = keys.resource_parameter_indices,
      .value = resource_parameter_indices_attr,
  };
  entries[entry_count++] = (loom_named_attr_t){
      .name_id = keys.uses_kernarg_segment_ptr,
      .value = loom_attr_bool(layout->uses_kernarg_segment_ptr),
  };
  if (has_distinct_direct_arg_payload_sizes) {
    entries[entry_count++] = (loom_named_attr_t){
        .name_id = keys.direct_arg_payload_sizes,
        .value = direct_arg_payload_sizes_attr,
    };
  }
  return loom_module_make_canonical_attr_dict(
      module, loom_make_named_attr_slice(entries, entry_count), out_attr);
}

static iree_status_t loom_amdgpu_hal_kernel_abi_resource_from_import(
    const loom_module_t* module, const loom_op_t* resource_op,
    iree_arena_allocator_t* arena,
    loom_amdgpu_hal_kernarg_resource_t* out_resource) {
  *out_resource = (loom_amdgpu_hal_kernarg_resource_t){0};

  iree_string_view_t resource_name = iree_string_view_empty();
  int64_t binding_index = loom_low_resource_index(resource_op);
  IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_format_resource_name(
      (uint32_t)binding_index, arena, &resource_name));

  uint64_t kernarg_offset =
      (uint64_t)binding_index *
      LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_SIZE;

  *out_resource = (loom_amdgpu_hal_kernarg_resource_t){
      .resource_op = resource_op,
      .name = resource_name,
      .binding_index = (uint32_t)binding_index,
      .parameter_index = (uint32_t)binding_index,
      .kernarg_offset = (uint32_t)kernarg_offset,
      .kernarg_size = LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_SIZE,
      .kernarg_alignment =
          LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_ALIGNMENT,
      .source_type = loom_amdgpu_hal_kernel_abi_type_attr(
          module, loom_low_resource_source_type(resource_op)),
      .abi_type =
          loom_module_value_type(module, loom_low_resource_result(resource_op)),
  };
  return iree_ok_status();
}

iree_status_t loom_amdgpu_hal_kernel_abi_layout_build(
    const loom_module_t* module, const loom_op_t* function_op,
    iree_arena_allocator_t* arena,
    loom_amdgpu_hal_kernel_abi_layout_t* out_layout) {
  const loom_region_t* body = loom_low_function_const_body(function_op);
  const loom_block_t* entry_block = body != NULL && body->block_count != 0
                                        ? loom_region_const_entry_block(body)
                                        : NULL;
  const iree_host_size_t direct_arg_count =
      entry_block != NULL ? entry_block->arg_count : 0;
  bool direct_arg_used = false;
  if (entry_block != NULL) {
    for (uint16_t i = 0; i < entry_block->arg_count; ++i) {
      const loom_value_id_t arg_id = loom_block_arg_id(entry_block, i);
      const loom_value_t* arg_value = loom_module_value(module, arg_id);
      if (arg_value->use_count != 0 ||
          loom_module_value_has_type_uses(module, arg_id)) {
        direct_arg_used = true;
        break;
      }
    }
  }
  iree_host_size_t resource_count = 0;
  bool resource_used = false;
  if (body != NULL) {
    for (uint16_t block_index = 0; block_index < body->block_count;
         ++block_index) {
      const loom_block_t* block = loom_region_const_block(body, block_index);
      const loom_op_t* resource_op = NULL;
      loom_block_for_each_op(block, resource_op) {
        if (!loom_low_resource_isa(resource_op)) {
          continue;
        }
        ++resource_count;
        const loom_value_id_t result = loom_low_resource_result(resource_op);
        const loom_value_t* result_value = loom_module_value(module, result);
        resource_used |= result_value->use_count != 0 ||
                         loom_module_value_has_type_uses(module, result);
      }
    }
  }
  if (resource_count > LOOM_AMDGPU_HAL_KERNEL_ABI_MAX_RESOURCE_COUNT) {
    IREE_ASSERT_UNREACHABLE("verified AMDGPU HAL ABI resource count");
    IREE_BUILTIN_UNREACHABLE();
  }
  const uint64_t kernarg_resource_bytes =
      (uint64_t)resource_count *
      LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_SIZE;
  if (direct_arg_count > UINT32_MAX) {
    IREE_ASSERT_UNREACHABLE("verified AMDGPU HAL ABI direct argument count");
    IREE_BUILTIN_UNREACHABLE();
  }

  loom_amdgpu_hal_kernarg_resource_t* resources = NULL;
  if (resource_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, resource_count, sizeof(*resources), (void**)&resources));
    memset(resources, 0, resource_count * sizeof(*resources));
  }

  if (body != NULL) {
    for (uint16_t block_index = 0; block_index < body->block_count;
         ++block_index) {
      const loom_block_t* block = loom_region_const_block(body, block_index);
      const loom_op_t* resource_op = NULL;
      loom_block_for_each_op(block, resource_op) {
        if (!loom_low_resource_isa(resource_op)) {
          continue;
        }

        const int64_t binding_index = loom_low_resource_index(resource_op);
        if (binding_index < 0 || (uint64_t)binding_index >= resource_count) {
          IREE_ASSERT_UNREACHABLE(
              "verified AMDGPU HAL ABI resource binding index");
          IREE_BUILTIN_UNREACHABLE();
        }
        const iree_host_size_t slot_index = (iree_host_size_t)binding_index;
        if (resources[slot_index].resource_op != NULL) {
          IREE_ASSERT_UNREACHABLE(
              "verified AMDGPU HAL ABI resource binding index");
          IREE_BUILTIN_UNREACHABLE();
        }

        loom_amdgpu_hal_kernarg_resource_t resource = {0};
        IREE_RETURN_IF_ERROR(loom_amdgpu_hal_kernel_abi_resource_from_import(
            module, resource_op, arena, &resource));
        resources[slot_index] = resource;
      }
    }
  }

  for (iree_host_size_t i = 0; i < resource_count; ++i) {
    if (resources[i].resource_op == NULL) {
      IREE_ASSERT_UNREACHABLE("verified AMDGPU HAL ABI resource binding set");
      IREE_BUILTIN_UNREACHABLE();
    }
  }

  loom_amdgpu_hal_kernarg_direct_arg_t* direct_args = NULL;
  uint64_t kernarg_direct_args_end = kernarg_resource_bytes;
  uint64_t constant_count = 0;
  if (direct_arg_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, direct_arg_count, sizeof(*direct_args), (void**)&direct_args));
    memset(direct_args, 0, direct_arg_count * sizeof(*direct_args));
    for (uint16_t i = 0; i < entry_block->arg_count; ++i) {
      if (kernarg_direct_args_end > UINT32_MAX) {
        IREE_ASSERT_UNREACHABLE(
            "verified AMDGPU HAL ABI direct argument extent");
        IREE_BUILTIN_UNREACHABLE();
      }
      const uint32_t parameter_index = (uint32_t)(resource_count + i);
      loom_amdgpu_hal_kernel_abi_direct_arg_from_block_arg(
          module, entry_block, i, parameter_index,
          (uint32_t)kernarg_direct_args_end, &direct_args[i]);
      kernarg_direct_args_end += direct_args[i].kernarg_size;
      constant_count += direct_args[i].kernarg_size /
                        LOOM_AMDGPU_HAL_KERNEL_ABI_DIRECT_SCALAR_KERNARG_SIZE;
    }
  }
  const uint64_t kernarg_segment_size = iree_align_uint64(
      kernarg_direct_args_end,
      LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_ALIGNMENT);
  if (kernarg_segment_size > UINT32_MAX || constant_count > UINT32_MAX) {
    IREE_ASSERT_UNREACHABLE("verified AMDGPU HAL ABI direct argument extent");
    IREE_BUILTIN_UNREACHABLE();
  }

  *out_layout = (loom_amdgpu_hal_kernel_abi_layout_t){
      .function_op = function_op,
      .parameter_count = (uint32_t)(resource_count + direct_arg_count),
      .kernarg_segment_size = (uint32_t)kernarg_segment_size,
      .kernarg_segment_alignment =
          LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_ALIGNMENT,
      .uses_kernarg_segment_ptr = resource_used || direct_arg_used,
      .constant_count = (uint32_t)constant_count,
      .resources = resources,
      .resource_count = resource_count,
      .direct_args = direct_args,
      .direct_arg_count = direct_arg_count,
  };

  return iree_ok_status();
}

// Dictionary admission retains source context until the final field is decoded.
// The status carries only allocation and diagnostic delivery failures.
typedef struct loom_amdgpu_hal_layout_decoder_t {
  // Module owning dictionary keys, strings, and source operations.
  const loom_module_t* module;
  // Kernel declaring this parameter layout.
  const loom_op_t* function_op;
  // Diagnostic destination, which may have no installed callback.
  iree_diagnostic_emitter_t emitter;
  // Failure delivering the first rejected-field diagnostic.
  iree_status_t status;
} loom_amdgpu_hal_layout_decoder_t;

static bool loom_amdgpu_hal_layout_reject(
    loom_amdgpu_hal_layout_decoder_t* decoder, iree_string_view_t field_name,
    iree_string_view_t constraint) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(field_name),
      loom_param_string(constraint),
  };
  const loom_diagnostic_emission_t emission = {
      .op = decoder->function_op,
      .error = LOOM_ERR_AMDGPU_059,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  decoder->status = iree_diagnostic_emit(decoder->emitter, &emission);
  return false;
}

static const loom_attribute_t* loom_amdgpu_hal_layout_find(
    const loom_amdgpu_hal_layout_decoder_t* decoder,
    loom_named_attr_slice_t attrs, iree_string_view_t name) {
  const loom_string_id_t key = loom_module_lookup_string(decoder->module, name);
  return loom_amdgpu_hal_kernel_abi_find_layout_attr(attrs, key);
}

static const loom_attribute_t* loom_amdgpu_hal_layout_field(
    loom_amdgpu_hal_layout_decoder_t* decoder, loom_named_attr_slice_t attrs,
    iree_string_view_t name, loom_attr_kind_t kind) {
  const loom_attribute_t* attr =
      loom_amdgpu_hal_layout_find(decoder, attrs, name);
  if (attr == NULL || attr->kind != kind) {
    iree_string_view_t constraint = iree_string_view_empty();
    switch (kind) {
      case LOOM_ATTR_I64:
        constraint = IREE_SV("an integer field");
        break;
      case LOOM_ATTR_BOOL:
        constraint = IREE_SV("a boolean field");
        break;
      case LOOM_ATTR_DICT:
        constraint = IREE_SV("a dictionary field");
        break;
      case LOOM_ATTR_STRING:
        constraint = IREE_SV("a string field");
        break;
      default:
        IREE_BUILTIN_UNREACHABLE();
    }
    loom_amdgpu_hal_layout_reject(decoder, name, constraint);
    return NULL;
  }
  return attr;
}

static bool loom_amdgpu_hal_layout_u32(
    loom_amdgpu_hal_layout_decoder_t* decoder, iree_string_view_t name,
    int64_t value, uint32_t* out_value) {
  if (value < 0 || value > UINT32_MAX) {
    return loom_amdgpu_hal_layout_reject(
        decoder, name, IREE_SV("an integer in the range [0, 4294967295]"));
  }
  *out_value = (uint32_t)value;
  return true;
}

static bool loom_amdgpu_hal_layout_count(
    loom_amdgpu_hal_layout_decoder_t* decoder, loom_named_attr_slice_t attrs,
    iree_string_view_t name, uint32_t* out_value) {
  const loom_attribute_t* attr =
      loom_amdgpu_hal_layout_field(decoder, attrs, name, LOOM_ATTR_I64);
  return attr && loom_amdgpu_hal_layout_u32(decoder, name,
                                            loom_attr_as_i64(*attr), out_value);
}

static bool loom_amdgpu_hal_layout_array(
    loom_amdgpu_hal_layout_decoder_t* decoder, loom_named_attr_slice_t attrs,
    iree_string_view_t name, uint32_t count, bool optional,
    const int64_t** out_values) {
  const loom_attribute_t* attr =
      loom_amdgpu_hal_layout_find(decoder, attrs, name);
  *out_values = NULL;
  if (attr == NULL && optional) {
    return true;
  }
  if (attr == NULL || attr->kind != LOOM_ATTR_I64_ARRAY ||
      attr->count != count) {
    return loom_amdgpu_hal_layout_reject(
        decoder, name,
        IREE_SV(
            "an integer array with the declared argument or resource count"));
  }
  *out_values = attr->i64_array;
  return true;
}

static bool loom_amdgpu_hal_layout_parameter(
    loom_amdgpu_hal_layout_decoder_t* decoder, iree_string_view_t name,
    int64_t value, const uint32_t* parameter_slots, uint32_t parameter_count,
    uint32_t* out_value) {
  if (value < 0 || (uint64_t)value >= parameter_count) {
    return loom_amdgpu_hal_layout_reject(
        decoder, name, IREE_SV("a parameter index within parameter_count"));
  }
  if (parameter_slots[value]) {
    return loom_amdgpu_hal_layout_reject(
        decoder, name,
        IREE_SV("distinct parameter indices across resources and direct "
                "arguments"));
  }
  *out_value = (uint32_t)value;
  return true;
}

iree_status_t loom_amdgpu_hal_kernel_abi_layout_from_attr(
    const loom_module_t* module, const loom_op_t* function_op,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* arena,
    bool* out_accepted, loom_amdgpu_hal_kernel_abi_layout_t* out_layout) {
  *out_accepted = false;
  *out_layout = (loom_amdgpu_hal_kernel_abi_layout_t){0};
  loom_amdgpu_hal_layout_decoder_t decoder = {
      .module = module,
      .function_op = function_op,
      .emitter = emitter,
      .status = iree_ok_status(),
  };
  const loom_named_attr_slice_t attrs =
      loom_low_kernel_def_abi_layout(function_op);
  uint32_t parameter_count = 0, resource_count = 0;
  uint32_t direct_arg_count = 0, constant_count = 0;
  if (!loom_amdgpu_hal_layout_count(&decoder, attrs, IREE_SV("parameter_count"),
                                    &parameter_count) ||
      !loom_amdgpu_hal_layout_count(&decoder, attrs, IREE_SV("resource_count"),
                                    &resource_count) ||
      !loom_amdgpu_hal_layout_count(
          &decoder, attrs, IREE_SV("direct_arg_count"), &direct_arg_count) ||
      !loom_amdgpu_hal_layout_count(&decoder, attrs, IREE_SV("constant_count"),
                                    &constant_count)) {
    return decoder.status;
  }
  if (resource_count > LOOM_AMDGPU_HAL_KERNEL_ABI_MAX_RESOURCE_COUNT ||
      direct_arg_count > UINT16_MAX) {
    loom_amdgpu_hal_layout_reject(
        &decoder,
        resource_count > LOOM_AMDGPU_HAL_KERNEL_ABI_MAX_RESOURCE_COUNT
            ? IREE_SV("resource_count")
            : IREE_SV("direct_arg_count"),
        IREE_SV("a count no greater than 65535"));
    return decoder.status;
  }
  if (parameter_count != resource_count + direct_arg_count) {
    loom_amdgpu_hal_layout_reject(
        &decoder, IREE_SV("parameter_count"),
        IREE_SV("the sum of resource_count and direct_arg_count"));
    return decoder.status;
  }
  const loom_region_t* body = loom_low_function_const_body(function_op);
  const uint16_t entry_arg_count =
      body && body->block_count ? loom_region_const_entry_block(body)->arg_count
                                : 0;
  if (entry_arg_count != 0 && entry_arg_count != direct_arg_count) {
    loom_amdgpu_hal_layout_reject(
        &decoder, IREE_SV("direct_arg_count"),
        IREE_SV("the number of unmaterialized entry block arguments"));
    return decoder.status;
  }

  const loom_attribute_t* uses_kernarg = loom_amdgpu_hal_layout_field(
      &decoder, attrs, IREE_SV("uses_kernarg_segment_ptr"), LOOM_ATTR_BOOL);
  if (!uses_kernarg) {
    return decoder.status;
  }
  const loom_attribute_t* names_attr = loom_amdgpu_hal_layout_field(
      &decoder, attrs, IREE_SV("direct_arg_names"), LOOM_ATTR_DICT);
  if (!names_attr) {
    return decoder.status;
  }
  const loom_named_attr_slice_t names = loom_attr_as_dict(*names_attr);
  if (names.count != direct_arg_count) {
    loom_amdgpu_hal_layout_reject(&decoder, IREE_SV("direct_arg_names"),
                                  IREE_SV("one name for each direct argument"));
    return decoder.status;
  }

  const int64_t* resource_parameters = NULL;
  const int64_t* resource_offsets = NULL;
  const int64_t* direct_parameters = NULL;
  const int64_t* direct_offsets = NULL;
  const int64_t* direct_sizes = NULL;
  const int64_t* payload_sizes = NULL;
  if (!loom_amdgpu_hal_layout_array(
          &decoder, attrs, IREE_SV("resource_parameter_indices"),
          resource_count, false, &resource_parameters) ||
      !loom_amdgpu_hal_layout_array(&decoder, attrs,
                                    IREE_SV("resource_offsets"), resource_count,
                                    false, &resource_offsets) ||
      !loom_amdgpu_hal_layout_array(
          &decoder, attrs, IREE_SV("direct_arg_parameter_indices"),
          direct_arg_count, false, &direct_parameters) ||
      !loom_amdgpu_hal_layout_array(&decoder, attrs,
                                    IREE_SV("direct_arg_offsets"),
                                    direct_arg_count, false, &direct_offsets) ||
      !loom_amdgpu_hal_layout_array(&decoder, attrs,
                                    IREE_SV("direct_arg_sizes"),
                                    direct_arg_count, false, &direct_sizes) ||
      !loom_amdgpu_hal_layout_array(&decoder, attrs,
                                    IREE_SV("direct_arg_payload_sizes"),
                                    direct_arg_count, true, &payload_sizes)) {
    return decoder.status;
  }
  // One-based row ordinals, with direct arguments following resource rows.
  // Zero marks an unclaimed slot until the complete parameter layout is read.
  uint32_t* parameter_slots = NULL;
  if (parameter_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, parameter_count,
                                                   sizeof(*parameter_slots),
                                                   (void**)&parameter_slots));
    memset(parameter_slots, 0, parameter_count * sizeof(*parameter_slots));
  }
  loom_amdgpu_hal_kernarg_resource_t* resources = NULL;
  if (resource_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, resource_count, sizeof(*resources), (void**)&resources));
  }
  loom_amdgpu_hal_kernarg_direct_arg_t* direct_args = NULL;
  if (direct_arg_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, direct_arg_count, sizeof(*direct_args), (void**)&direct_args));
  }

  uint64_t segment_end = 0, decoded_constant_count = 0;
  for (uint32_t i = 0; i < resource_count; ++i) {
    uint32_t parameter_index = 0, offset = 0;
    if (!loom_amdgpu_hal_layout_parameter(
            &decoder, IREE_SV("resource_parameter_indices"),
            resource_parameters[i], parameter_slots, parameter_count,
            &parameter_index) ||
        !loom_amdgpu_hal_layout_u32(&decoder, IREE_SV("resource_offsets"),
                                    resource_offsets[i], &offset)) {
      return decoder.status;
    }
    if (offset % LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_ALIGNMENT) {
      loom_amdgpu_hal_layout_reject(&decoder, IREE_SV("resource_offsets"),
                                    IREE_SV("8-byte aligned resource offsets"));
      return decoder.status;
    }
    iree_string_view_t name = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_hal_kernel_abi_format_resource_name(i, arena, &name));
    resources[i] = (loom_amdgpu_hal_kernarg_resource_t){
        .name = name,
        .binding_index = i,
        .parameter_index = parameter_index,
        .kernarg_offset = offset,
        .kernarg_size = LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_SIZE,
        .kernarg_alignment =
            LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_ALIGNMENT,
        .source_type = loom_type_none(),
        .abi_type = loom_type_none(),
    };
    parameter_slots[parameter_index] = i + 1;
    segment_end =
        iree_max(segment_end, (uint64_t)offset + resources[i].kernarg_size);
  }
  for (uint32_t i = 0; i < direct_arg_count; ++i) {
    char key[32];
    int length = iree_snprintf(key, sizeof(key), "arg%" PRIu32, i);
    const loom_attribute_t* name = loom_amdgpu_hal_layout_field(
        &decoder, names, iree_make_string_view(key, (iree_host_size_t)length),
        LOOM_ATTR_STRING);
    if (!name) {
      return decoder.status;
    }
    uint32_t size = 0, payload_size = 0, parameter_index = 0, offset = 0;
    if (!loom_amdgpu_hal_layout_u32(&decoder, IREE_SV("direct_arg_sizes"),
                                    direct_sizes[i], &size)) {
      return decoder.status;
    }
    if (size == 0 ||
        size % LOOM_AMDGPU_HAL_KERNEL_ABI_DIRECT_SCALAR_KERNARG_SIZE ||
        size > LOOM_AMDGPU_HAL_KERNEL_ABI_DIRECT_ARG_MAX_UNIT_COUNT *
                   LOOM_AMDGPU_HAL_KERNEL_ABI_DIRECT_SCALAR_KERNARG_SIZE) {
      loom_amdgpu_hal_layout_reject(
          &decoder, IREE_SV("direct_arg_sizes"),
          IREE_SV(
              "a positive whole number of constant words, at most 128 bytes"));
      return decoder.status;
    }
    payload_size = size;
    if (payload_sizes && !loom_amdgpu_hal_layout_u32(
                             &decoder, IREE_SV("direct_arg_payload_sizes"),
                             payload_sizes[i], &payload_size)) {
      return decoder.status;
    }
    if (payload_size == 0 || payload_size > size) {
      loom_amdgpu_hal_layout_reject(
          &decoder, IREE_SV("direct_arg_payload_sizes"),
          IREE_SV(
              "a positive payload size no greater than its native storage"));
      return decoder.status;
    }
    if (!loom_amdgpu_hal_layout_parameter(
            &decoder, IREE_SV("direct_arg_parameter_indices"),
            direct_parameters[i], parameter_slots, parameter_count,
            &parameter_index) ||
        !loom_amdgpu_hal_layout_u32(&decoder, IREE_SV("direct_arg_offsets"),
                                    direct_offsets[i], &offset)) {
      return decoder.status;
    }
    if (offset % LOOM_AMDGPU_HAL_KERNEL_ABI_DIRECT_SCALAR_KERNARG_ALIGNMENT) {
      loom_amdgpu_hal_layout_reject(
          &decoder, IREE_SV("direct_arg_offsets"),
          IREE_SV("4-byte aligned direct argument offsets"));
      return decoder.status;
    }
    direct_args[i] = (loom_amdgpu_hal_kernarg_direct_arg_t){
        .arg_id = LOOM_VALUE_ID_INVALID,
        .name = loom_amdgpu_hal_kernel_abi_module_string(
            module, loom_attr_as_string_id(*name)),
        .parameter_index = parameter_index,
        .argument_index = (uint16_t)i,
        .kernarg_offset = offset,
        .kernarg_size = size,
        .payload_size = payload_size,
        .kernarg_alignment =
            LOOM_AMDGPU_HAL_KERNEL_ABI_DIRECT_SCALAR_KERNARG_ALIGNMENT,
        .abi_type = loom_type_none(),
    };
    parameter_slots[parameter_index] = resource_count + i + 1;
    segment_end = iree_max(segment_end, (uint64_t)offset + size);
    decoded_constant_count +=
        size / LOOM_AMDGPU_HAL_KERNEL_ABI_DIRECT_SCALAR_KERNARG_SIZE;
  }
  if (decoded_constant_count != constant_count) {
    loom_amdgpu_hal_layout_reject(
        &decoder, IREE_SV("constant_count"),
        IREE_SV("the total number of direct argument storage words"));
    return decoder.status;
  }
  uint64_t previous_end = 0;
  for (uint32_t i = 0; i < parameter_count; ++i) {
    const uint32_t row = parameter_slots[i] - 1;
    const bool is_resource = row < resource_count;
    const uint32_t offset =
        is_resource ? resources[row].kernarg_offset
                    : direct_args[row - resource_count].kernarg_offset;
    const uint32_t size = is_resource
                              ? resources[row].kernarg_size
                              : direct_args[row - resource_count].kernarg_size;
    if (offset < previous_end) {
      loom_amdgpu_hal_layout_reject(
          &decoder,
          is_resource ? IREE_SV("resource_offsets")
                      : IREE_SV("direct_arg_offsets"),
          IREE_SV("disjoint argument storage in parameter order"));
      return decoder.status;
    }
    previous_end = (uint64_t)offset + size;
  }
  const uint64_t segment_size = iree_align_uint64(
      segment_end, LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_ALIGNMENT);
  if (segment_size > UINT32_MAX) {
    loom_amdgpu_hal_layout_reject(
        &decoder, IREE_SV("kernarg_segment_size"),
        IREE_SV("an aligned kernarg extent no greater than 4294967295 bytes"));
    return decoder.status;
  }
  *out_layout = (loom_amdgpu_hal_kernel_abi_layout_t){
      .function_op = function_op,
      .parameter_count = parameter_count,
      .kernarg_segment_size = (uint32_t)segment_size,
      .kernarg_segment_alignment =
          LOOM_AMDGPU_HAL_KERNEL_ABI_GLOBAL_BUFFER_KERNARG_ALIGNMENT,
      .uses_kernarg_segment_ptr = loom_attr_as_bool(*uses_kernarg),
      .constant_count = constant_count,
      .resources = resources,
      .resource_count = resource_count,
      .direct_args = direct_args,
      .direct_arg_count = direct_arg_count,
  };
  *out_accepted = true;
  return iree_ok_status();
}
