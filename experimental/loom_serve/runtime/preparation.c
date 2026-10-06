// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/runtime/preparation.h"

#include <inttypes.h>
#include <stdio.h>

#include "experimental/loom_serve/runtime/program.h"
#include "iree/base/internal/path.h"
#include "iree/vm/buffer.h"
#include "iree/vm/reflection.h"
#include "iree/vm/sync.h"

typedef struct preparation_stage_t {
  // Public declaration; its array pointers refer to the owned storage below.
  loom_serve_preparation_stage_t declaration;
  // Allocated compiler-binding capacity.
  iree_host_size_t config_capacity;
  // Owned compiler-binding storage, mutable only during bootstrap.
  loomc_config_binding_t* config;
  // Allocated parameter-declaration capacity.
  iree_host_size_t parameter_capacity;
  // Owned parameter declarations, mutable only during bootstrap.
  loom_serve_preparation_parameter_t* parameters;
} preparation_stage_t;

struct loom_serve_preparation_t {
  // Allocator owning the catalog and every copied string.
  iree_allocator_t allocator;
  // Number of initialized stages, including partial declarations on failure.
  iree_host_size_t stage_count;
  // Allocated stage capacity.
  iree_host_size_t stage_capacity;
  // Owned stage declarations.
  preparation_stage_t* stages;
};

typedef struct preparation_module_t {
  // Native module prefix.
  iree_vm_module_t base;
  // Owned descriptor borrowing the static declaration tables.
  iree_vm_module_descriptor_t descriptor;
  // Canonical VM buffer type borrowed from the environment.
  iree_vm_ref_type_t buffer_type;
  // Borrowed mutable catalog, live through the sole bootstrap invocation.
  loom_serve_preparation_t* preparation;
} preparation_module_t;

void loom_serve_preparation_destroy(loom_serve_preparation_t* preparation) {
  if (!preparation) {
    return;
  }
  const iree_allocator_t allocator = preparation->allocator;
  for (iree_host_size_t i = 0; i < preparation->stage_count; ++i) {
    preparation_stage_t* stage = &preparation->stages[i];
    iree_allocator_free(allocator, (void*)stage->declaration.root.data);
    for (iree_host_size_t j = 0; j < stage->declaration.config.binding_count;
         ++j) {
      iree_allocator_free(allocator, (void*)stage->config[j].key.data);
      iree_allocator_free(allocator, (void*)stage->config[j].value.data);
    }
    for (iree_host_size_t j = 0; j < stage->declaration.parameter_count; ++j) {
      iree_allocator_free(allocator, (void*)stage->parameters[j].path.data);
      iree_allocator_free(allocator, (void*)stage->parameters[j].policy.data);
    }
    iree_allocator_free(allocator, stage->config);
    iree_allocator_free(allocator, stage->parameters);
  }
  iree_allocator_free(allocator, preparation->stages);
  iree_allocator_free(allocator, preparation);
}

static iree_status_t preparation_copy_string(iree_string_view_t value,
                                             iree_allocator_t allocator,
                                             iree_string_view_t* out_value) {
  char* data = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_clone(
      allocator, iree_make_const_byte_span(value.data, value.size),
      (void**)&data));
  *out_value = iree_make_string_view(data, value.size);
  return iree_ok_status();
}

static iree_status_t preparation_string_argument(
    const iree_vm_call_packet_t* call, iree_host_size_t ordinal,
    iree_string_view_t* out_value) {
  iree_vm_ref_t ref = iree_vm_ref_null();
  iree_vm_call_ref_argument_load_borrow(call, ordinal, &ref);
  if (!ref.object) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "preparation string must not be null");
  }
  iree_vm_buffer_t* buffer = (iree_vm_buffer_t*)ref.object;
  iree_const_byte_span_t bytes = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_vm_buffer_map_read(
      buffer, 0, iree_vm_buffer_length(buffer), &bytes));
  *out_value =
      iree_make_string_view((const char*)bytes.data, bytes.data_length);
  return iree_ok_status();
}

static iree_status_t preparation_add_stage(
    loom_serve_preparation_t* preparation, const iree_vm_call_packet_t* call) {
  iree_string_view_t root = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(preparation_string_argument(call, 0, &root));
  if (!root.size || preparation->stage_count == INT32_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "preparation requires a named, addressable stage");
  }
  if (preparation->stage_count == preparation->stage_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        preparation->allocator, preparation->stage_count + 1,
        sizeof(*preparation->stages), &preparation->stage_capacity,
        (void**)&preparation->stages));
  }
  const iree_host_size_t index = preparation->stage_count++;
  preparation_stage_t* stage = &preparation->stages[index];
  *stage = (preparation_stage_t){0};
  stage->declaration.tag = (int64_t)iree_vm_call_value_argument_load(call, 0);
  stage->declaration.config.flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED;
  IREE_RETURN_IF_ERROR(preparation_copy_string(root, preparation->allocator,
                                               &stage->declaration.root));
  iree_vm_call_value_result_store(call, 0, index);
  return iree_ok_status();
}

static iree_status_t preparation_add_config(
    loom_serve_preparation_t* preparation, preparation_stage_t* stage,
    iree_string_view_t key, iree_string_view_t value) {
  if (!key.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "preparation configuration key is empty");
  }
  if (stage->declaration.config.binding_count == stage->config_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        preparation->allocator, stage->declaration.config.binding_count + 1,
        sizeof(*stage->config), &stage->config_capacity,
        (void**)&stage->config));
  }
  stage->declaration.config.bindings = stage->config;
  loomc_config_binding_t* binding =
      &stage->config[stage->declaration.config.binding_count++];
  *binding = (loomc_config_binding_t){0};
  iree_string_view_t copied = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      preparation_copy_string(key, preparation->allocator, &copied));
  binding->key = (loomc_string_view_t){copied.data, copied.size};
  IREE_RETURN_IF_ERROR(
      preparation_copy_string(value, preparation->allocator, &copied));
  binding->value = (loomc_string_view_t){copied.data, copied.size};
  return iree_ok_status();
}

static iree_status_t preparation_add_parameter(
    loom_serve_preparation_t* preparation, preparation_stage_t* stage,
    const iree_vm_call_packet_t* call) {
  iree_string_view_t directory = iree_string_view_empty();
  iree_string_view_t path = iree_string_view_empty();
  iree_string_view_t policy = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(preparation_string_argument(call, 0, &directory));
  IREE_RETURN_IF_ERROR(preparation_string_argument(call, 1, &path));
  IREE_RETURN_IF_ERROR(preparation_string_argument(call, 2, &policy));
  const uint32_t binding = (uint32_t)iree_vm_call_value_argument_load(call, 1);
  if ((!directory.size && !path.size) || !policy.size ||
      binding == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "preparation parameter requires paths and binding");
  }
  for (iree_host_size_t i = 0; i < stage->declaration.parameter_count; ++i) {
    if (stage->parameters[i].binding == binding) {
      return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                              "parameter binding %u declared twice", binding);
    }
  }
  if (stage->declaration.parameter_count == stage->parameter_capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        preparation->allocator, stage->declaration.parameter_count + 1,
        sizeof(*stage->parameters), &stage->parameter_capacity,
        (void**)&stage->parameters));
  }
  stage->declaration.parameters = stage->parameters;
  loom_serve_preparation_parameter_t* parameter =
      &stage->parameters[stage->declaration.parameter_count++];
  *parameter = (loom_serve_preparation_parameter_t){.binding = binding};
  char* joined = NULL;
  IREE_RETURN_IF_ERROR(
      iree_file_path_join(directory, path, preparation->allocator, &joined));
  parameter->path = iree_make_cstring_view(joined);
  return preparation_copy_string(policy, preparation->allocator,
                                 &parameter->policy);
}

static iree_status_t preparation_module_start(
    iree_vm_module_t* base,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_execution_outcome_t* out_outcome) {
  loom_serve_preparation_t* preparation =
      ((preparation_module_t*)base)->preparation;
  const iree_vm_call_packet_t* call = &params->call;
  iree_status_t status = iree_ok_status();
  if (params->function_ordinal == 0) {
    status = preparation_add_stage(preparation, call);
  } else {
    const uint32_t index = (uint32_t)iree_vm_call_value_argument_load(call, 0);
    if (index >= preparation->stage_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "preparation stage %u is not declared", index);
    }
    preparation_stage_t* stage = &preparation->stages[index];
    if (params->function_ordinal == 3) {
      status = preparation_add_parameter(preparation, stage, call);
    } else {
      iree_string_view_t key = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(preparation_string_argument(call, 0, &key));
      iree_string_view_t value = iree_string_view_empty();
      char digits[32];
      if (params->function_ordinal == 1) {
        const int length =
            snprintf(digits, sizeof(digits), "%" PRId64,
                     (int64_t)iree_vm_call_value_argument_load(call, 1));
        value = iree_make_string_view(digits, length);
      } else {
        IREE_RETURN_IF_ERROR(preparation_string_argument(call, 1, &value));
      }
      status = preparation_add_config(preparation, stage, key, value);
    }
  }
  if (iree_status_is_ok(status)) {
    *out_outcome = IREE_VM_EXECUTION_OUTCOME_COMPLETED;
  }
  return status;
}

static const iree_vm_module_signature_type_t preparation_stage_arguments[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0}, {IREE_VM_SCALAR_TYPE_I64, 0}};
static const iree_vm_module_signature_type_t preparation_stage_results[] = {
    {IREE_VM_SCALAR_TYPE_I32, 0}};
static const iree_vm_module_signature_type_t
    preparation_config_i64_arguments[] = {
        {IREE_VM_SCALAR_TYPE_I32, 0},
        {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
        {IREE_VM_SCALAR_TYPE_I64, 0}};
static const iree_vm_module_signature_type_t preparation_config_arguments[] = {
    {IREE_VM_SCALAR_TYPE_I32, 0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0}};
static const iree_vm_module_signature_type_t preparation_parameter_arguments[] =
    {{IREE_VM_SCALAR_TYPE_I32, 0},
     {IREE_VM_SCALAR_TYPE_I32, 0},
     {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
     {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
     {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0}};

static const iree_vm_module_callable_type_declaration_t
    preparation_callables[] = {
        {.signature = {.arguments = {preparation_stage_arguments, 2, 1, 1, 0},
                       .results = {preparation_stage_results, 1, 1, 0, 0}}},
        {.signature = {.arguments = {preparation_config_i64_arguments, 3, 2, 1,
                                     0}}},
        {.signature = {.arguments = {preparation_config_arguments, 3, 1, 2,
                                     0}}},
        {.signature = {.arguments = {preparation_parameter_arguments, 5, 2, 3,
                                     0}}},
};

static const iree_vm_module_export_declaration_t preparation_exports[] = {
    {.export_name = IREE_SVL("config"),
     .callable_type_ordinal = 2,
     .function_ordinal = 2},
    {.export_name = IREE_SVL("config_i64"),
     .callable_type_ordinal = 1,
     .function_ordinal = 1},
    {.export_name = IREE_SVL("parameter"),
     .callable_type_ordinal = 3,
     .function_ordinal = 3},
    {.export_name = IREE_SVL("stage"),
     .callable_type_ordinal = 0,
     .function_ordinal = 0},
};

static void preparation_module_destroy(iree_vm_module_t* base) {
  preparation_module_t* module = (preparation_module_t*)base;
  iree_allocator_free(module->preparation->allocator, module);
}

static void preparation_module_query_export(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_export_declaration_t* out_value) {
  *out_value = preparation_exports[ordinal];
}

static void preparation_module_query_callable(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_callable_type_declaration_t* out_value) {
  *out_value = preparation_callables[ordinal];
}

static void preparation_module_query_import_group(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_import_group_t* out_value) {
  IREE_CHECK_UNREACHABLE("preparation module has no imports");
}

static void preparation_module_query_import(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_import_declaration_t* out_value) {
  IREE_CHECK_UNREACHABLE("preparation module has no imports");
}

static const iree_vm_module_vtable_t preparation_module_vtable = {
    .structure_size = sizeof(iree_vm_module_vtable_t),
    .abi_version = IREE_VM_MODULE_ABI_VERSION_0,
    .destroy = preparation_module_destroy,
    .function_start = preparation_module_start,
    .function_resume = iree_vm_module_function_resume_unreachable,
    .query_import_group = preparation_module_query_import_group,
    .query_import = preparation_module_query_import,
    .query_export = preparation_module_query_export,
    .query_callable_type = preparation_module_query_callable,
    .query_presentation = iree_vm_module_query_presentation_none,
    .metadata_by_ordinal = iree_vm_module_metadata_by_ordinal_none,
};

iree_status_t loom_serve_preparation_create(
    iree_vm_environment_t* environment, iree_string_view_t source_path,
    iree_string_view_t entry, iree_vm_variant_span_t arguments,
    iree_vm_variant_span_t results, iree_vm_module_span_t libraries,
    loom_serve_preparation_t** out_preparation,
    iree_allocator_t host_allocator) {
  *out_preparation = NULL;
  loom_serve_preparation_t* preparation = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*preparation), (void**)&preparation));
  preparation->allocator = host_allocator;
  preparation_module_t* module = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*module), (void**)&module);
  iree_vm_module_t* library = NULL;
  if (iree_status_is_ok(status)) {
    module->preparation = preparation;
    iree_vm_ref_types_t types = {0};
    status = iree_vm_ref_types_resolve(
        iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm")),
        &types);
    module->buffer_type = types.buffer;
    module->descriptor = (iree_vm_module_descriptor_t){
        .name = IREE_SVL("prepare"),
        .flags = IREE_VM_MODULE_FLAG_LINKABLE,
        .ref_types = {&module->buffer_type, 1},
        .counts = {.function_count = 4,
                   .callable_type_count = 4,
                   .export_count = 4,
                   .callable_fields = {.value_count = 7, .ref_count = 7}},
    };
    if (iree_status_is_ok(status)) {
      status = iree_vm_module_initialize(&preparation_module_vtable,
                                         &module->descriptor, &module->base);
    }
    if (iree_status_is_ok(status)) {
      library = &module->base;
    }
  }
  loom_serve_program_t* program = NULL;
  iree_vm_module_t** linked = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, libraries.count + 1,
                                         sizeof(*linked), (void**)&linked);
  }
  if (iree_status_is_ok(status)) {
    linked[0] = library;
    for (iree_host_size_t i = 0; i < libraries.count; ++i) {
      linked[i + 1] = libraries.data[i];
    }
    status = loom_serve_program_create(
        environment, source_path, 1, &entry,
        (iree_vm_module_span_t){linked, libraries.count + 1}, host_allocator,
        &program);
  }
  iree_allocator_free(host_allocator, linked);
  if (iree_status_is_ok(status)) {
    iree_vm_function_t function = {0};
    status =
        iree_vm_process_lookup_function(loom_serve_program_process(program),
                                        IREE_SV("model"), entry, &function);
    if (iree_status_is_ok(status)) {
      status = iree_vm_invoke(loom_serve_program_invocation(program), function,
                              arguments, results);
    }
  }
  loom_serve_program_destroy(program);
  if (library) {
    iree_vm_module_release(library);
  } else {
    iree_allocator_free(host_allocator, module);
  }
  if (iree_status_is_ok(status) && !preparation->stage_count) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "model preparation declared no stages");
  }
  if (iree_status_is_ok(status)) {
    *out_preparation = preparation;
  } else {
    loom_serve_preparation_destroy(preparation);
  }
  return status;
}

iree_host_size_t loom_serve_preparation_stage_count(
    const loom_serve_preparation_t* preparation) {
  return preparation->stage_count;
}

const loom_serve_preparation_stage_t* loom_serve_preparation_stage(
    const loom_serve_preparation_t* preparation, iree_host_size_t index) {
  return &preparation->stages[index].declaration;
}
