// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/parameter/inputs.h"

#include "iree/vm/buffer.h"
#include "iree/vm/reflection.h"

typedef struct loom_serve_weight_source_t {
  // Retained canonical keys, metadata, and backing checkpoint files.
  iree_io_parameter_index_t* index;
  // Retained imported-file cache shared with other users of this source.
  iree_io_parameter_provider_t* provider;
} loom_serve_weight_source_t;

struct loom_serve_weight_inputs_t {
  // Allocator owning source records and the immutable captured range table.
  iree_allocator_t host_allocator;
  // Number of policy-local input sources.
  uint32_t source_count;
  // Retained selected sources in policy-local order, following this object.
  // Unused peers in a request catalog do not stay live through this plan.
  loom_serve_weight_source_t* sources;
  // Number of initialized ranges, addressable by the compact recipe ordinals.
  uint32_t count;
  // Allocated range capacity, grown only during cold source capture.
  iree_host_size_t capacity;
  // Owned ordered input ranges. Final spans retain ordinals across growth.
  loom_serve_weight_input_t* values;
  // Largest captured input extent, before lane alignment.
  iree_device_size_t staging_length;
  // Maximum requested input alignment across all recipes.
  iree_device_size_t staging_alignment;
};

iree_status_t loom_serve_weight_inputs_allocate(
    const loom_serve_preparation_parameter_t* parameter,
    loom_serve_parameter_sources_t* sources, iree_allocator_t host_allocator,
    loom_serve_weight_inputs_t** out_inputs) {
  *out_inputs = NULL;
  loom_serve_weight_inputs_t* inputs = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_struct_array(
      host_allocator, sizeof(*inputs), parameter->source_count,
      sizeof(*inputs->sources), (void**)&inputs));
  inputs->host_allocator = host_allocator;
  inputs->staging_alignment = 1;
  inputs->source_count = parameter->source_count;
  inputs->sources = (loom_serve_weight_source_t*)(inputs + 1);
  for (uint32_t i = 0; i < parameter->source_count; ++i) {
    inputs->sources[i].index = loom_serve_parameter_sources_index(
        sources, parameter->source_ordinals[i]);
    iree_io_parameter_index_retain(inputs->sources[i].index);
    inputs->sources[i].provider = loom_serve_parameter_sources_provider(
        sources, parameter->source_ordinals[i]);
    iree_io_parameter_provider_retain(inputs->sources[i].provider);
  }
  *out_inputs = inputs;
  return iree_ok_status();
}

void loom_serve_weight_inputs_free(loom_serve_weight_inputs_t* inputs) {
  if (!inputs) {
    return;
  }
  for (uint32_t i = 0; i < inputs->source_count; ++i) {
    iree_io_parameter_provider_release(inputs->sources[i].provider);
    iree_io_parameter_index_release(inputs->sources[i].index);
  }
  iree_allocator_free(inputs->host_allocator, inputs->values);
  iree_allocator_free(inputs->host_allocator, inputs);
}

iree_io_parameter_provider_t* loom_serve_weight_inputs_provider(
    const loom_serve_weight_inputs_t* inputs, uint32_t source) {
  return inputs->sources[source].provider;
}

uint32_t loom_serve_weight_inputs_source_count(
    const loom_serve_weight_inputs_t* inputs) {
  return inputs->source_count;
}

iree_status_t loom_serve_weight_inputs_validate_direct(
    const loom_serve_weight_inputs_t* inputs, iree_string_view_t key,
    iree_device_size_t byte_length) {
  const iree_io_parameter_index_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(
      iree_io_parameter_index_lookup(inputs->sources[0].index, key, &entry));
  if (byte_length > entry->length) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "weight '%.*s' requires %" PRIu64
        " bytes from checkpoint source zero, but its entry has %" PRIu64,
        (int)key.size, key.data, (uint64_t)byte_length, entry->length);
  }
  return iree_ok_status();
}

const loom_serve_weight_input_t* loom_serve_weight_inputs_at(
    const loom_serve_weight_inputs_t* inputs, uint32_t ordinal) {
  return &inputs->values[ordinal];
}

iree_device_size_t loom_serve_weight_inputs_staging_length(
    const loom_serve_weight_inputs_t* inputs) {
  return iree_device_align(inputs->staging_length, inputs->staging_alignment);
}

iree_device_size_t loom_serve_weight_inputs_staging_alignment(
    const loom_serve_weight_inputs_t* inputs) {
  return inputs->staging_alignment;
}

typedef struct loom_serve_weight_inputs_module_t {
  // Native VM module prefix.
  iree_vm_module_t base;
  // Allocator owning only this module, not the borrowed input plan.
  iree_allocator_t host_allocator;
  // Descriptor borrowing static callable and export tables.
  iree_vm_module_descriptor_t descriptor;
  // Canonical VM buffer reference type from the enclosing environment.
  iree_vm_ref_types_t types;
  // Borrowed source indexes and range storage, retained by the weights plan.
  loom_serve_weight_inputs_t* inputs;
  // Capture is available only during a serialized weight policy invocation.
  bool active;
  // Current tensor's partial range and aligned staging extent.
  loom_serve_weight_input_recipe_t recipe;
} loom_serve_weight_inputs_module_t;

void loom_serve_weight_inputs_module_begin(iree_vm_module_t* base) {
  loom_serve_weight_inputs_module_t* module =
      (loom_serve_weight_inputs_module_t*)base;
  module->active = true;
  module->recipe =
      (loom_serve_weight_input_recipe_t){.first = module->inputs->count};
}

loom_serve_weight_input_recipe_t loom_serve_weight_inputs_module_take(
    iree_vm_module_t* base) {
  loom_serve_weight_inputs_module_t* module =
      (loom_serve_weight_inputs_module_t*)base;
  module->active = false;
  module->inputs->staging_length =
      iree_max(module->inputs->staging_length, module->recipe.byte_length);
  return module->recipe;
}

static iree_status_t loom_serve_weight_inputs_lookup(
    loom_serve_weight_inputs_module_t* module,
    const iree_vm_call_packet_t* call,
    const iree_io_parameter_index_entry_t** out_entry) {
  const uint32_t source = (uint32_t)iree_vm_call_value_argument_load(call, 0);
  if (source >= module->inputs->source_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "weight source %u is not declared", source);
  }
  iree_vm_ref_t ref = iree_vm_ref_null();
  iree_vm_call_ref_argument_load_borrow(call, 0, &ref);
  if (!ref.object) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "weight input key must not be null");
  }
  iree_vm_buffer_t* buffer = (iree_vm_buffer_t*)ref.object;
  iree_const_byte_span_t bytes = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_buffer_map_read(
      buffer, 0, iree_vm_buffer_length(buffer), &bytes));
  return iree_io_parameter_index_lookup(
      module->inputs->sources[source].index,
      iree_make_string_view((const char*)bytes.data, bytes.data_length),
      out_entry);
}

static iree_status_t loom_serve_weight_inputs_capture(
    loom_serve_weight_inputs_module_t* module,
    const iree_vm_call_packet_t* call,
    const iree_io_parameter_index_entry_t* entry) {
  loom_serve_weight_inputs_t* inputs = module->inputs;
  const int64_t offset = (int64_t)iree_vm_call_value_argument_load(call, 1);
  const int64_t length = (int64_t)iree_vm_call_value_argument_load(call, 2);
  const int64_t alignment = (int64_t)iree_vm_call_value_argument_load(call, 3);
  if (offset < 0 || length <= 0 || (uint64_t)offset > entry->length ||
      (uint64_t)length > entry->length - (uint64_t)offset) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "weight '%.*s' input range offset %" PRId64 " length %" PRId64
        " exceeds checkpoint tensor length %" PRIu64,
        (int)entry->key.size, entry->key.data, offset, length, entry->length);
  }
  if (alignment <= 0 || (alignment & (alignment - 1)) ||
      module->recipe.byte_length > INT64_MAX - (uint64_t)(alignment - 1)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "weight input requires addressable power-of-two alignment");
  }
  const iree_device_size_t target =
      iree_device_align(module->recipe.byte_length, alignment);
  // Leave every captured extent addressable after final lane alignment as well.
  const uint64_t maximum_alignment =
      iree_max(inputs->staging_alignment, (uint64_t)alignment);
  if ((uint64_t)length > INT64_MAX - target ||
      target + length > INT64_MAX - (maximum_alignment - 1) ||
      inputs->staging_length > INT64_MAX - (maximum_alignment - 1) ||
      inputs->count == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "weight input staging extent is not addressable");
  }
  if (inputs->count == inputs->capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        inputs->host_allocator, (iree_host_size_t)inputs->count + 1,
        sizeof(*inputs->values), &inputs->capacity, (void**)&inputs->values));
  }
  inputs->values[inputs->count++] = (loom_serve_weight_input_t){
      .entry = entry,
      .span = {.parameter_offset = (uint64_t)offset,
               .buffer_offset = target,
               .length = (uint64_t)length},
      .source = (uint32_t)iree_vm_call_value_argument_load(call, 0),
  };
  ++module->recipe.count;
  module->recipe.byte_length = target + length;
  inputs->staging_alignment = maximum_alignment;
  iree_vm_call_value_result_store(call, 0, target);
  return iree_ok_status();
}

static iree_status_t loom_serve_weight_inputs_module_start(
    iree_vm_module_t* base,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_execution_outcome_t* out_outcome) {
  loom_serve_weight_inputs_module_t* module =
      (loom_serve_weight_inputs_module_t*)base;
  if (!module->active) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "weight inputs require an active policy capture");
  }
  const iree_vm_call_packet_t* call = &params->call;
  if (params->function_ordinal == 0) {
    iree_vm_call_value_result_store(call, 0, module->inputs->source_count);
  } else {
    const iree_io_parameter_index_entry_t* entry = NULL;
    IREE_RETURN_IF_ERROR(loom_serve_weight_inputs_lookup(module, call, &entry));
    if (params->function_ordinal == 1) {
      if (entry->length > INT64_MAX) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "weight input tensor length is not addressable");
      }
      iree_vm_buffer_t* metadata = NULL;
      IREE_RETURN_IF_ERROR(
          iree_vm_buffer_wrap(IREE_VM_BUFFER_ACCESS_FLAG_READ,
                              iree_make_byte_span((void*)entry->metadata.data,
                                                  entry->metadata.data_length),
                              iree_vm_buffer_release_callback_null(),
                              module->host_allocator, &metadata));
      iree_vm_ref_t result =
          iree_vm_buffer_ref_from_ptr_move(&module->types, &metadata);
      iree_vm_call_ref_result_store_move(call, 0, &result);
      iree_vm_call_value_result_store(call, 0, entry->length);
    } else {
      IREE_RETURN_IF_ERROR(
          loom_serve_weight_inputs_capture(module, call, entry));
    }
  }
  *out_outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  return iree_ok_status();
}

static const iree_vm_module_signature_type_t inputs_count_results[] = {
    {IREE_VM_SCALAR_TYPE_I32, 0}};
static const iree_vm_module_signature_type_t inputs_parameter_arguments[] = {
    {IREE_VM_SCALAR_TYPE_I32, 0}, {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0}};
static const iree_vm_module_signature_type_t inputs_parameter_results[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0}, {IREE_VM_SCALAR_TYPE_I64, 0}};
static const iree_vm_module_signature_type_t inputs_input_arguments[] = {
    {IREE_VM_SCALAR_TYPE_I32, 0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
    {IREE_VM_SCALAR_TYPE_I64, 0},
    {IREE_VM_SCALAR_TYPE_I64, 0},
    {IREE_VM_SCALAR_TYPE_I64, 0}};
static const iree_vm_module_signature_type_t inputs_input_results[] = {
    {IREE_VM_SCALAR_TYPE_I64, 0}};
static const iree_vm_module_callable_type_declaration_t inputs_callables[] = {
    {.signature = {.results = {inputs_count_results, 1, 1, 0, 0}}},
    {.signature = {.arguments = {inputs_parameter_arguments, 2, 1, 1, 0},
                   .results = {inputs_parameter_results, 2, 1, 1, 0}}},
    {.signature = {.arguments = {inputs_input_arguments, 5, 4, 1, 0},
                   .results = {inputs_input_results, 1, 1, 0, 0}}},
};
static const iree_vm_module_export_declaration_t inputs_exports[] = {
    {.export_name = IREE_SVL("count"),
     .callable_type_ordinal = 0,
     .function_ordinal = 0},
    {.export_name = IREE_SVL("input"),
     .callable_type_ordinal = 2,
     .function_ordinal = 2},
    {.export_name = IREE_SVL("parameter"),
     .callable_type_ordinal = 1,
     .function_ordinal = 1},
};

static void loom_serve_weight_inputs_module_destroy(iree_vm_module_t* base) {
  loom_serve_weight_inputs_module_t* module =
      (loom_serve_weight_inputs_module_t*)base;
  iree_allocator_free(module->host_allocator, module);
}
static void loom_serve_weight_inputs_query_export(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_export_declaration_t* out_value) {
  *out_value = inputs_exports[ordinal];
}
static void loom_serve_weight_inputs_query_callable(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_callable_type_declaration_t* out_value) {
  *out_value = inputs_callables[ordinal];
}
static void loom_serve_weight_inputs_query_import_group(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_import_group_t* out_value) {
  IREE_CHECK_UNREACHABLE("weights module has no imports");
}
static void loom_serve_weight_inputs_query_import(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_import_declaration_t* out_value) {
  IREE_CHECK_UNREACHABLE("weights module has no imports");
}
static const iree_vm_module_vtable_t inputs_module_vtable = {
    .structure_size = sizeof(iree_vm_module_vtable_t),
    .abi_version = IREE_VM_MODULE_ABI_VERSION_0,
    .destroy = loom_serve_weight_inputs_module_destroy,
    .function_start = loom_serve_weight_inputs_module_start,
    .function_resume = iree_vm_module_function_resume_unreachable,
    .query_import_group = loom_serve_weight_inputs_query_import_group,
    .query_import = loom_serve_weight_inputs_query_import,
    .query_export = loom_serve_weight_inputs_query_export,
    .query_callable_type = loom_serve_weight_inputs_query_callable,
    .query_presentation = iree_vm_module_query_presentation_none,
    .metadata_by_ordinal = iree_vm_module_metadata_by_ordinal_none,
};

iree_status_t loom_serve_weight_inputs_module_create(
    loom_serve_weight_inputs_t* inputs, iree_vm_environment_t* environment,
    iree_allocator_t host_allocator, iree_vm_module_t** out_module) {
  *out_module = NULL;
  iree_vm_ref_types_t types = {0};
  IREE_RETURN_IF_ERROR(iree_vm_ref_types_resolve(
      iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm")),
      &types));
  loom_serve_weight_inputs_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*module), (void**)&module));
  module->host_allocator = host_allocator;
  module->types = types;
  module->inputs = inputs;
  module->descriptor = (iree_vm_module_descriptor_t){
      .name = IREE_SVL("weights"),
      .flags = IREE_VM_MODULE_FLAG_LINKABLE,
      .ref_types = {&module->types.buffer, 1},
      .counts = {.function_count = 3,
                 .callable_type_count = 3,
                 .export_count = 3,
                 .callable_fields = {.value_count = 8, .ref_count = 3}},
  };
  iree_status_t status = iree_vm_module_initialize(
      &inputs_module_vtable, &module->descriptor, &module->base);
  if (iree_status_is_ok(status)) {
    *out_module = &module->base;
  } else {
    iree_allocator_free(host_allocator, module);
  }
  return status;
}
