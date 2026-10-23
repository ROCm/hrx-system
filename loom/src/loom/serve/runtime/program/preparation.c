// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/program/preparation.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "iree/base/internal/path.h"
#include "iree/vm/buffer.h"
#include "iree/vm/reflection.h"
#include "iree/vm/sync.h"
#include "loom/serve/runtime/program/parameter_layout.h"
#include "loom/serve/runtime/program/program.h"
#include "loomc/iree.h"

typedef struct loom_serve_preparation_stage_storage_t {
  // Public declaration; its array pointers refer to the owned storage below.
  loom_serve_preparation_stage_t declaration;
  // Allocated compiler-binding capacity.
  iree_host_size_t config_capacity;
  // Owned compiler-binding storage, mutable only during declaration capture.
  loomc_config_binding_t* config;
  // Allocated parameter-declaration capacity.
  iree_host_size_t parameter_capacity;
  // Owned parameter declarations, mutable only during declaration capture.
  loom_serve_preparation_parameter_t* parameters;
  // Allocated policy-input capacity for each parameter declaration.
  iree_host_size_t* input_capacities;
} loom_serve_preparation_stage_storage_t;

struct loom_serve_preparation_t {
  // Allocator owning the catalog and every copied string.
  iree_allocator_t host_allocator;
  // Retained model facts supplying immutable defaults and named sources.
  loom_model_card_t* card;
  // Borrowed service-owned component contract retained through card.
  const loom_model_component_t* component;
  // Requested owned bytes, accumulated where each allocation is established.
  iree_host_size_t storage_size;
  // Owned source-selected host entry, empty without a forecast declaration.
  iree_string_view_t memory_forecast;
  // Number of initialized stages, including partial declarations on failure.
  iree_host_size_t stage_count;
  // Allocated stage capacity.
  iree_host_size_t stage_capacity;
  // Owned stage declarations.
  loom_serve_preparation_stage_storage_t* stages;
  // Borrowed immutable cold source prefix, outliving this capture.
  const loom_serve_preparation_t* base;
  // Source ordinal of the first entry owned by this capture.
  uint32_t first_source;
  // Number of owned sources, including partial declarations on failure.
  uint32_t source_count;
  // Allocated source capacity.
  iree_host_size_t source_capacity;
  // Owned ordered shard lists, independent of resident placement.
  loom_model_file_list_t* sources;
  // Allocated shard-list capacity for each owned source.
  iree_host_size_t* path_capacities;
};

typedef struct loom_serve_preparation_module_t {
  // Native module prefix.
  iree_vm_module_t base;
  // Allocator owning the module and each independent declaration capture.
  iree_allocator_t host_allocator;
  // Owned descriptor borrowing the static declaration tables.
  iree_vm_module_descriptor_t descriptor;
  // Canonical VM buffer type borrowed from the environment.
  iree_vm_ref_type_t buffer_type;
  // Retained model facts for named source declarations and stage config.
  loom_model_card_t* card;
  // Borrowed service-owned component contract retained through card.
  const loom_model_component_t* component;
  // One serialized source invocation's declaration capability.
  struct {
    // Only an explicitly scoped invocation may declare or configure stages.
    bool active;
    // Source-visible ordinal of the first stage declared in this capture.
    uint32_t first_stage;
    // Immutable cold source prefix borrowed through the captured declarations.
    const loom_serve_preparation_t* base;
    // The capture owner explicitly authorizes checkpoint files and policies.
    loom_serve_preparation_flags_t flags;
    // Owned partial declarations, transferred by take or freed at destruction.
    loom_serve_preparation_t* preparation;
  } capture;
} loom_serve_preparation_module_t;

void loom_serve_preparation_module_begin(iree_vm_module_t* base,
                                         uint32_t first_stage,
                                         const loom_serve_preparation_t* prefix,
                                         loom_serve_preparation_flags_t flags) {
  loom_serve_preparation_module_t* module =
      (loom_serve_preparation_module_t*)base;
  module->capture.active = true;
  module->capture.first_stage = first_stage;
  module->capture.base = prefix;
  module->capture.flags = flags;
}

loom_serve_preparation_t* loom_serve_preparation_module_take(
    iree_vm_module_t* base) {
  loom_serve_preparation_module_t* module =
      (loom_serve_preparation_module_t*)base;
  loom_serve_preparation_t* preparation = module->capture.preparation;
  // Shard arrays may grow after a binding refers to them. Freeze their views
  // once, before publishing this capture to policy preparation or execution.
  for (iree_host_size_t i = 0; preparation && i < preparation->stage_count;
       ++i) {
    loom_serve_preparation_stage_storage_t* stage = &preparation->stages[i];
    for (iree_host_size_t j = 0; j < stage->declaration.parameter_count; ++j) {
      loom_serve_preparation_parameter_t* parameter = &stage->parameters[j];
      loom_model_file_list_t* sources =
          (loom_model_file_list_t*)parameter->sources;
      for (uint32_t s = 0; s < parameter->source_count; ++s) {
        sources[s] = loom_serve_preparation_source(
            preparation, parameter->source_ordinals[s]);
      }
    }
  }
  module->capture.preparation = NULL;
  module->capture.active = false;
  return preparation;
}

void loom_serve_preparation_free(loom_serve_preparation_t* preparation) {
  if (!preparation) {
    return;
  }
  const iree_allocator_t host_allocator = preparation->host_allocator;
  iree_allocator_free(host_allocator, (void*)preparation->memory_forecast.data);
  for (iree_host_size_t i = 0; i < preparation->stage_count; ++i) {
    loom_serve_preparation_stage_storage_t* stage = &preparation->stages[i];
    iree_allocator_free(host_allocator, (void*)stage->declaration.root.data);
    for (iree_host_size_t j = 0; j < stage->declaration.config.binding_count;
         ++j) {
      iree_allocator_free(host_allocator, (void*)stage->config[j].key.data);
      iree_allocator_free(host_allocator, (void*)stage->config[j].value.data);
    }
    for (iree_host_size_t j = 0; j < stage->declaration.parameter_count; ++j) {
      const loom_serve_preparation_parameter_t* parameter =
          &stage->parameters[j];
      iree_allocator_free(host_allocator, (void*)parameter->sources);
      iree_allocator_free(host_allocator, (void*)parameter->source_ordinals);
      iree_allocator_free(host_allocator,
                          (void*)stage->parameters[j].policy.data);
    }
    iree_allocator_free(host_allocator, stage->config);
    iree_allocator_free(host_allocator, stage->parameters);
    iree_allocator_free(host_allocator, stage->input_capacities);
  }
  for (uint32_t i = 0; i < preparation->source_count; ++i) {
    const loom_model_file_list_t paths = preparation->sources[i];
    for (iree_host_size_t p = 0; p < paths.count; ++p) {
      iree_allocator_free(host_allocator, (void*)paths.values[p].path.data);
      iree_allocator_free(host_allocator, (void*)paths.values[p].format.data);
    }
    iree_allocator_free(host_allocator, (void*)paths.values);
  }
  iree_allocator_free(host_allocator, preparation->sources);
  iree_allocator_free(host_allocator, preparation->path_capacities);
  iree_allocator_free(host_allocator, preparation->stages);
  loom_model_card_release(preparation->card);
  iree_allocator_free(host_allocator, preparation);
}

static iree_status_t loom_serve_preparation_grow_array(
    loom_serve_preparation_t* preparation, iree_host_size_t minimum_capacity,
    iree_host_size_t element_size, iree_host_size_t* capacity, void** values) {
  const iree_host_size_t previous_capacity = *capacity;
  IREE_RETURN_IF_ERROR(iree_allocator_grow_array(preparation->host_allocator,
                                                 minimum_capacity, element_size,
                                                 capacity, values));
  preparation->storage_size += (*capacity - previous_capacity) * element_size;
  return iree_ok_status();
}

static iree_status_t loom_serve_preparation_copy_string(
    loom_serve_preparation_t* preparation, iree_string_view_t value,
    iree_string_view_t* out_value) {
  char* data = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_clone(
      preparation->host_allocator,
      iree_make_const_byte_span(value.data, value.size), (void**)&data));
  preparation->storage_size += value.size;
  *out_value = iree_make_string_view(data, value.size);
  return iree_ok_status();
}

static iree_status_t loom_serve_preparation_string_argument(
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
  if (bytes.data_length && memchr(bytes.data, 0, bytes.data_length)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "preparation strings must not contain NUL bytes");
  }
  *out_value =
      iree_make_string_view((const char*)bytes.data, bytes.data_length);
  return iree_ok_status();
}

static iree_status_t loom_serve_preparation_add_stage(
    loom_serve_preparation_t* preparation, const iree_vm_call_packet_t* call,
    uint32_t first_stage, loom_serve_preparation_stage_kind_t kind) {
  iree_string_view_t root = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_serve_preparation_string_argument(call, 0, &root));
  if (!root.size || preparation->stage_count == INT32_MAX - first_stage) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "preparation requires a named, addressable stage");
  }
  if (preparation->stage_count == preparation->stage_capacity) {
    IREE_RETURN_IF_ERROR(loom_serve_preparation_grow_array(
        preparation, preparation->stage_count + 1, sizeof(*preparation->stages),
        &preparation->stage_capacity, (void**)&preparation->stages));
  }
  const iree_host_size_t index = preparation->stage_count++;
  loom_serve_preparation_stage_storage_t* stage = &preparation->stages[index];
  *stage = (loom_serve_preparation_stage_storage_t){0};
  stage->declaration.kind = kind;
  stage->declaration.tag = (int64_t)iree_vm_call_value_argument_load(call, 0);
  stage->declaration.config.flags = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED;
  if (preparation->card) {
    stage->declaration.config.json_object = loomc_string_view_from_iree(
        loom_model_card_selection(preparation->card)->config);
  }
  IREE_RETURN_IF_ERROR(loom_serve_preparation_copy_string(
      preparation, root, &stage->declaration.root));
  iree_vm_call_value_result_store(call, 0, first_stage + index);
  return iree_ok_status();
}

static iree_status_t loom_serve_preparation_add_config(
    loom_serve_preparation_t* preparation,
    loom_serve_preparation_stage_storage_t* stage, iree_string_view_t key,
    iree_string_view_t value) {
  if (!key.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "preparation configuration key is empty");
  }
  if (stage->declaration.config.binding_count == stage->config_capacity) {
    IREE_RETURN_IF_ERROR(loom_serve_preparation_grow_array(
        preparation, stage->declaration.config.binding_count + 1,
        sizeof(*stage->config), &stage->config_capacity,
        (void**)&stage->config));
  }
  stage->declaration.config.bindings = stage->config;
  loomc_config_binding_t* binding =
      &stage->config[stage->declaration.config.binding_count++];
  *binding = (loomc_config_binding_t){0};
  iree_string_view_t copied = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      loom_serve_preparation_copy_string(preparation, key, &copied));
  binding->key = (loomc_string_view_t){copied.data, copied.size};
  IREE_RETURN_IF_ERROR(
      loom_serve_preparation_copy_string(preparation, value, &copied));
  binding->value = (loomc_string_view_t){copied.data, copied.size};
  return iree_ok_status();
}

static iree_status_t loom_serve_preparation_bind_parameter(
    loom_serve_preparation_t* preparation,
    loom_serve_preparation_stage_storage_t* stage, uint32_t binding,
    uint32_t input, uint32_t source, iree_string_view_t policy) {
  if (stage->declaration.kind != LOOM_SERVE_PREPARATION_STAGE_COMMAND) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter placement requires a command reflection contract");
  }
  if (binding > INT32_MAX || input >= INT32_MAX) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "preparation parameter requires representable "
                            "binding and input ordinals");
  }
  if (source >= loom_serve_preparation_source_count(preparation)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter source %u is not declared", source);
  }
  iree_host_size_t index = 0;
  for (; index < stage->declaration.parameter_count; ++index) {
    if (stage->parameters[index].binding == binding) {
      break;
    }
  }
  if (index == stage->declaration.parameter_count) {
    if (index == stage->parameter_capacity) {
      const iree_host_size_t previous_capacity = stage->parameter_capacity;
      IREE_RETURN_IF_ERROR(loom_serve_preparation_grow_array(
          preparation, index + 1, sizeof(*stage->parameters),
          &stage->parameter_capacity, (void**)&stage->parameters));
      stage->declaration.parameters = stage->parameters;
      IREE_RETURN_IF_ERROR(iree_allocator_realloc_array(
          preparation->host_allocator, stage->parameter_capacity,
          sizeof(*stage->input_capacities), (void**)&stage->input_capacities));
      preparation->storage_size +=
          (stage->parameter_capacity - previous_capacity) *
          sizeof(*stage->input_capacities);
    }
    stage->parameters[index] =
        (loom_serve_preparation_parameter_t){.binding = binding};
    stage->input_capacities[index] = 0;
    ++stage->declaration.parameter_count;
    IREE_RETURN_IF_ERROR(loom_serve_preparation_copy_string(
        preparation, policy, &stage->parameters[index].policy));
  } else if (!iree_string_view_equal(policy, stage->parameters[index].policy)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter binding %u requires one preparation policy", binding);
  }
  loom_serve_preparation_parameter_t* parameter = &stage->parameters[index];
  iree_host_size_t* capacity = &stage->input_capacities[index];
  if (input != parameter->source_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter inputs must be declared once, densely from zero");
  }
  loom_model_file_list_t* sources = (loom_model_file_list_t*)parameter->sources;
  if (parameter->source_count == *capacity) {
    const iree_host_size_t previous_capacity = *capacity;
    IREE_RETURN_IF_ERROR(loom_serve_preparation_grow_array(
        preparation, parameter->source_count + 1, sizeof(*sources), capacity,
        (void**)&sources));
    parameter->sources = sources;
    IREE_RETURN_IF_ERROR(
        iree_allocator_realloc_array(preparation->host_allocator, *capacity,
                                     sizeof(*parameter->source_ordinals),
                                     (void**)&parameter->source_ordinals));
    preparation->storage_size +=
        (*capacity - previous_capacity) * sizeof(*parameter->source_ordinals);
  }
  sources[input] = (loom_model_file_list_t){0};
  ((uint32_t*)parameter->source_ordinals)[input] = source;
  ++parameter->source_count;
  return iree_ok_status();
}

static iree_status_t loom_serve_preparation_append_source(
    loom_serve_preparation_t* preparation, uint32_t ordinal,
    iree_string_view_t directory, loom_model_file_t file) {
  if (ordinal == loom_serve_preparation_source_count(preparation)) {
    if (ordinal == INT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "parameter source ordinal is not representable");
    }
    if (preparation->source_count == preparation->source_capacity) {
      const iree_host_size_t previous_capacity = preparation->source_capacity;
      IREE_RETURN_IF_ERROR(loom_serve_preparation_grow_array(
          preparation, preparation->source_count + 1,
          sizeof(*preparation->sources), &preparation->source_capacity,
          (void**)&preparation->sources));
      IREE_RETURN_IF_ERROR(iree_allocator_realloc_array(
          preparation->host_allocator, preparation->source_capacity,
          sizeof(*preparation->path_capacities),
          (void**)&preparation->path_capacities));
      preparation->storage_size +=
          (preparation->source_capacity - previous_capacity) *
          sizeof(*preparation->path_capacities);
    }
    preparation->sources[preparation->source_count] =
        (loom_model_file_list_t){0};
    preparation->path_capacities[preparation->source_count++] = 0;
  }
  const uint32_t local = ordinal - preparation->first_source;
  loom_model_file_list_t* files = &preparation->sources[local];
  iree_host_size_t* capacity = &preparation->path_capacities[local];
  loom_model_file_t* paths = (loom_model_file_t*)files->values;
  if (files->count == *capacity) {
    IREE_RETURN_IF_ERROR(loom_serve_preparation_grow_array(
        preparation, files->count + 1, sizeof(*paths), capacity,
        (void**)&paths));
    files->values = paths;
  }
  char* joined = NULL;
  IREE_RETURN_IF_ERROR(iree_file_path_join(
      directory, file.path, preparation->host_allocator, &joined));
  loom_model_file_t* copied = &paths[files->count++];
  *copied = (loom_model_file_t){.path = iree_make_cstring_view(joined)};
  preparation->storage_size += copied->path.size + 1;
  if (file.format.size) {
    IREE_RETURN_IF_ERROR(loom_serve_preparation_copy_string(
        preparation, file.format, &copied->format));
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_preparation_add_parameter(
    loom_serve_preparation_t* preparation,
    loom_serve_preparation_stage_storage_t* stage,
    const iree_vm_call_packet_t* call) {
  iree_string_view_t policy = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(
      loom_serve_preparation_string_argument(call, 0, &policy));
  const uint32_t binding = (uint32_t)iree_vm_call_value_argument_load(call, 1);
  const uint32_t input = (uint32_t)iree_vm_call_value_argument_load(call, 2);
  const uint32_t source = (uint32_t)iree_vm_call_value_argument_load(call, 3);
  if (!policy.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter requires a preparation policy");
  }
  char* policy_path = NULL;
  if (preparation->card) {
    IREE_RETURN_IF_ERROR(iree_file_path_join(
        loom_model_card_selection(preparation->card)->directory, policy,
        preparation->host_allocator, &policy_path));
    policy = iree_make_cstring_view(policy_path);
  }
  iree_status_t status = loom_serve_preparation_bind_parameter(
      preparation, stage, binding, input, source, policy);
  iree_allocator_free(preparation->host_allocator, policy_path);
  return status;
}

static iree_status_t loom_serve_preparation_add_source(
    loom_serve_preparation_t* preparation, const iree_vm_call_packet_t* call,
    bool append) {
  if (preparation->card) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "card-backed parameters require source_scope");
  }
  iree_string_view_t directory = iree_string_view_empty();
  loom_model_file_t file = {0};
  IREE_RETURN_IF_ERROR(
      loom_serve_preparation_string_argument(call, 0, &directory));
  IREE_RETURN_IF_ERROR(
      loom_serve_preparation_string_argument(call, 1, &file.path));
  if (!directory.size && !file.path.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter source requires a path");
  }
  uint32_t ordinal = loom_serve_preparation_source_count(preparation);
  if (append) {
    ordinal = (uint32_t)iree_vm_call_value_argument_load(call, 0);
    if (ordinal < preparation->first_source ||
        ordinal >= loom_serve_preparation_source_count(preparation)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "source %u is not declared in this capture",
                              ordinal);
    }
  }
  IREE_RETURN_IF_ERROR(loom_serve_preparation_append_source(
      preparation, ordinal, directory, file));
  if (!append) {
    iree_vm_call_value_result_store(call, 0, ordinal);
  }
  return iree_ok_status();
}

static iree_status_t loom_serve_preparation_add_scope(
    loom_serve_preparation_t* preparation, const iree_vm_call_packet_t* call) {
  iree_string_view_t scope = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_serve_preparation_string_argument(call, 0, &scope));
  if (!preparation->card) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "source_scope requires a selected model card");
  }
  const loom_model_parameter_t* parameter = loom_model_selection_parameter(
      loom_model_card_selection(preparation->card), scope);
  if (!parameter) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "unknown parameter scope '%.*s'", (int)scope.size,
                            scope.data);
  }
  const uint32_t ordinal = loom_serve_preparation_source_count(preparation);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < parameter->files.count && iree_status_is_ok(status); ++i) {
    status = loom_serve_preparation_append_source(preparation, ordinal,
                                                  iree_string_view_empty(),
                                                  parameter->files.values[i]);
  }
  if (iree_status_is_ok(status)) {
    iree_vm_call_value_result_store(call, 0, ordinal);
  }
  return status;
}

bool loom_serve_preparation_parameter_equal(
    const loom_serve_preparation_parameter_t* lhs,
    const loom_serve_preparation_parameter_t* rhs) {
  if (lhs->source_count != rhs->source_count ||
      !iree_string_view_equal(lhs->policy, rhs->policy)) {
    return false;
  }
  for (uint32_t s = 0; s < lhs->source_count; ++s) {
    const loom_model_file_list_t a = lhs->sources[s];
    const loom_model_file_list_t b = rhs->sources[s];
    if (a.count != b.count) {
      return false;
    }
    for (iree_host_size_t i = 0; i < a.count; ++i) {
      if (!iree_string_view_equal(a.values[i].path, b.values[i].path) ||
          !iree_string_view_equal(a.values[i].format, b.values[i].format)) {
        return false;
      }
    }
  }
  return true;
}

static iree_status_t loom_serve_preparation_set_workspace_limit(
    loom_serve_preparation_stage_storage_t* stage,
    const iree_vm_call_packet_t* call) {
  if (stage->declaration.kind != LOOM_SERVE_PREPARATION_STAGE_COMMAND) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "workspace limits require a command reflection contract");
  }
  const int64_t limit = (int64_t)iree_vm_call_value_argument_load(call, 1);
  if (limit <= 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "workspace limit must be positive");
  }
  if (stage->declaration.workspace_limit_bytes) {
    return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                            "workspace limit declared twice");
  }
  stage->declaration.workspace_limit_bytes = (uint64_t)limit;
  return iree_ok_status();
}

static iree_status_t loom_serve_preparation_module_start(
    iree_vm_module_t* base,
    const iree_vm_module_function_start_params_t* params,
    iree_vm_execution_outcome_t* out_outcome) {
  loom_serve_preparation_module_t* module =
      (loom_serve_preparation_module_t*)base;
  if (!module->capture.active) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "stage declarations require an active preparation");
  }
  loom_serve_preparation_t* preparation = module->capture.preparation;
  const iree_vm_call_packet_t* call = &params->call;
  iree_status_t status = iree_ok_status();
  if ((params->function_ordinal == 3 || params->function_ordinal == 6 ||
       params->function_ordinal == 7 || params->function_ordinal == 9) &&
      !iree_any_bit_set(module->capture.flags,
                        LOOM_SERVE_PREPARATION_FLAG_PARAMETERS)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter declarations are unavailable in this preparation capture");
  }
  if (params->function_ordinal == 8 &&
      !iree_any_bit_set(module->capture.flags,
                        LOOM_SERVE_PREPARATION_FLAG_MEMORY_FORECAST)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "memory forecast declarations are unavailable in this preparation "
        "capture");
  }
  if (params->function_ordinal == 0 || params->function_ordinal == 4 ||
      params->function_ordinal >= 6) {
    if (!preparation) {
      IREE_RETURN_IF_ERROR(iree_allocator_malloc(
          module->host_allocator, sizeof(*preparation), (void**)&preparation));
      preparation->host_allocator = module->host_allocator;
      preparation->card = module->card;
      preparation->component = module->component;
      loom_model_card_retain(preparation->card);
      preparation->storage_size = sizeof(*preparation);
      preparation->base = module->capture.base;
      preparation->first_source =
          loom_serve_preparation_source_count(module->capture.base);
      module->capture.preparation = preparation;
    }
    if (params->function_ordinal == 8) {
      iree_string_view_t root = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(
          loom_serve_preparation_string_argument(call, 0, &root));
      if (iree_string_view_is_empty(root)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "memory forecast root must not be empty");
      }
      if (!iree_string_view_is_empty(preparation->memory_forecast)) {
        return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                                "memory forecast declared twice");
      }
      status = loom_serve_preparation_copy_string(
          preparation, root, &preparation->memory_forecast);
    } else if (params->function_ordinal == 9) {
      status = loom_serve_preparation_add_scope(preparation, call);
    } else if (params->function_ordinal >= 6) {
      status = loom_serve_preparation_add_source(preparation, call,
                                                 params->function_ordinal == 7);
    } else {
      status = loom_serve_preparation_add_stage(
          preparation, call, module->capture.first_stage,
          params->function_ordinal == 4 ? LOOM_SERVE_PREPARATION_STAGE_DISPATCH
                                        : LOOM_SERVE_PREPARATION_STAGE_COMMAND);
    }
  } else {
    const uint32_t index = (uint32_t)iree_vm_call_value_argument_load(call, 0);
    if (!preparation || index < module->capture.first_stage ||
        index - module->capture.first_stage >= preparation->stage_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "preparation stage %u is not declared", index);
    }
    loom_serve_preparation_stage_storage_t* stage =
        &preparation->stages[index - module->capture.first_stage];
    if (params->function_ordinal == 3) {
      status = loom_serve_preparation_add_parameter(preparation, stage, call);
    } else if (params->function_ordinal == 5) {
      status = loom_serve_preparation_set_workspace_limit(stage, call);
    } else {
      iree_string_view_t key = iree_string_view_empty();
      IREE_RETURN_IF_ERROR(
          loom_serve_preparation_string_argument(call, 0, &key));
      iree_string_view_t value = iree_string_view_empty();
      char digits[32];
      if (params->function_ordinal == 1) {
        const int length =
            snprintf(digits, sizeof(digits), "%" PRId64,
                     (int64_t)iree_vm_call_value_argument_load(call, 1));
        value = iree_make_string_view(digits, length);
      } else {
        IREE_RETURN_IF_ERROR(
            loom_serve_preparation_string_argument(call, 1, &value));
      }
      status =
          loom_serve_preparation_add_config(preparation, stage, key, value);
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
    preparation_memory_forecast_arguments[] = {
        {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0}};
static const iree_vm_module_signature_type_t
    preparation_workspace_limit_arguments[] = {{IREE_VM_SCALAR_TYPE_I32, 0},
                                               {IREE_VM_SCALAR_TYPE_I64, 0}};
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
     {IREE_VM_SCALAR_TYPE_I32, 0},
     {IREE_VM_SCALAR_TYPE_I32, 0},
     {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0}};
static const iree_vm_module_signature_type_t preparation_source_arguments[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0},
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0}};

static const iree_vm_module_signature_type_t preparation_scope_arguments[] = {
    {IREE_VM_MODULE_SIGNATURE_TYPE_KIND_REF, 0}};

static const iree_vm_module_callable_type_declaration_t
    preparation_callables[] = {
        {.signature = {.arguments = {preparation_memory_forecast_arguments, 1,
                                     0, 1, 0}}},
        {.signature = {.arguments = {preparation_workspace_limit_arguments, 2,
                                     2, 0, 0}}},
        {.signature = {.arguments = {preparation_stage_arguments, 2, 1, 1, 0},
                       .results = {preparation_stage_results, 1, 1, 0, 0}}},
        {.signature = {.arguments = {preparation_source_arguments, 2, 0, 2, 0},
                       .results = {preparation_stage_results, 1, 1, 0, 0}}},
        {.signature = {.arguments = {preparation_config_i64_arguments, 3, 2, 1,
                                     0}}},
        {.signature = {.arguments = {preparation_config_arguments, 3, 1, 2,
                                     0}}},
        {.signature = {.arguments = {preparation_parameter_arguments, 5, 4, 1,
                                     0}}},
        {.signature = {.arguments = {preparation_scope_arguments, 1, 0, 1, 0},
                       .results = {preparation_stage_results, 1, 1, 0, 0}}},
};

static const iree_vm_module_export_declaration_t preparation_exports[] = {
    {.export_name = IREE_SVL("config"),
     .callable_type_ordinal = 5,
     .function_ordinal = 2},
    {.export_name = IREE_SVL("config_i64"),
     .callable_type_ordinal = 4,
     .function_ordinal = 1},
    {.export_name = IREE_SVL("dispatch"),
     .callable_type_ordinal = 2,
     .function_ordinal = 4},
    {.export_name = IREE_SVL("memory_forecast"),
     .callable_type_ordinal = 0,
     .function_ordinal = 8},
    {.export_name = IREE_SVL("parameter"),
     .callable_type_ordinal = 6,
     .function_ordinal = 3},
    {.export_name = IREE_SVL("source"),
     .callable_type_ordinal = 3,
     .function_ordinal = 6},
    {.export_name = IREE_SVL("source_append"),
     .callable_type_ordinal = 5,
     .function_ordinal = 7},
    {.export_name = IREE_SVL("source_scope"),
     .callable_type_ordinal = 7,
     .function_ordinal = 9},
    {.export_name = IREE_SVL("stage"),
     .callable_type_ordinal = 2,
     .function_ordinal = 0},
    {.export_name = IREE_SVL("workspace_limit"),
     .callable_type_ordinal = 1,
     .function_ordinal = 5},
};

static void loom_serve_preparation_module_destroy(iree_vm_module_t* base) {
  loom_serve_preparation_module_t* module =
      (loom_serve_preparation_module_t*)base;
  loom_serve_preparation_free(module->capture.preparation);
  loom_model_card_release(module->card);
  iree_allocator_free(module->host_allocator, module);
}

static void loom_serve_preparation_module_query_export(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_export_declaration_t* out_value) {
  *out_value = preparation_exports[ordinal];
}

static void loom_serve_preparation_module_query_callable(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_callable_type_declaration_t* out_value) {
  *out_value = preparation_callables[ordinal];
}

static void loom_serve_preparation_module_query_import_group(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_import_group_t* out_value) {
  IREE_CHECK_UNREACHABLE("preparation module has no imports");
}

static void loom_serve_preparation_module_query_import(
    const iree_vm_module_t* base, iree_host_size_t ordinal,
    iree_vm_module_import_declaration_t* out_value) {
  IREE_CHECK_UNREACHABLE("preparation module has no imports");
}

static const iree_vm_module_vtable_t preparation_module_vtable = {
    .structure_size = sizeof(iree_vm_module_vtable_t),
    .abi_version = IREE_VM_MODULE_ABI_VERSION_0,
    .destroy = loom_serve_preparation_module_destroy,
    .function_start = loom_serve_preparation_module_start,
    .function_resume = iree_vm_module_function_resume_unreachable,
    .query_import_group = loom_serve_preparation_module_query_import_group,
    .query_import = loom_serve_preparation_module_query_import,
    .query_export = loom_serve_preparation_module_query_export,
    .query_callable_type = loom_serve_preparation_module_query_callable,
    .query_presentation = iree_vm_module_query_presentation_none,
    .metadata_by_ordinal = iree_vm_module_metadata_by_ordinal_none,
};

iree_status_t loom_serve_preparation_module_create(
    iree_vm_environment_t* environment, loom_model_card_t* card,
    const loom_model_component_t* component, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module) {
  *out_module = NULL;
  iree_vm_ref_types_t types = {0};
  IREE_RETURN_IF_ERROR(iree_vm_ref_types_resolve(
      iree_vm_environment_lookup_ref_type_table(environment, IREE_SV("vm")),
      &types));
  loom_serve_preparation_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*module), (void**)&module));
  module->host_allocator = host_allocator;
  module->card = card;
  module->component = component;
  loom_model_card_retain(card);
  module->buffer_type = types.buffer;
  module->descriptor = (iree_vm_module_descriptor_t){
      .name = IREE_SVL("prepare"),
      .flags = IREE_VM_MODULE_FLAG_LINKABLE,
      .ref_types = {&module->buffer_type, 1},
      .counts = {.function_count = 10,
                 .callable_type_count = 8,
                 .export_count = 10,
                 .callable_fields = {.value_count = 13, .ref_count = 9}},
  };
  iree_status_t status = iree_vm_module_initialize(
      &preparation_module_vtable, &module->descriptor, &module->base);
  if (iree_status_is_ok(status)) {
    *out_module = &module->base;
  } else {
    loom_model_card_release(module->card);
    iree_allocator_free(host_allocator, module);
  }
  return status;
}

iree_status_t loom_serve_preparation_allocate(
    loomc_target_environment_t* target_environment,
    iree_vm_environment_t* environment, const loom_serve_sources_t* sources,
    iree_string_view_t source_path, iree_string_view_t entry,
    loom_serve_preparation_flags_t flags, iree_vm_variant_span_t arguments,
    iree_vm_variant_span_t results, iree_vm_module_span_t libraries,
    loom_serve_compilation_report_t* report, iree_allocator_t host_allocator,
    loom_serve_preparation_t** out_preparation) {
  *out_preparation = NULL;
  iree_vm_module_t* library = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_preparation_module_create(
      environment, sources->card, sources->component, host_allocator,
      &library));
  loom_serve_preparation_module_begin(library, 0, NULL, flags);
  loom_serve_program_t* program = NULL;
  iree_vm_module_t** linked = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, libraries.count + 1, sizeof(*linked), (void**)&linked);
  if (iree_status_is_ok(status)) {
    linked[0] = library;
    for (iree_host_size_t i = 0; i < libraries.count; ++i) {
      linked[i + 1] = libraries.data[i];
    }
    status = loom_serve_program_allocate_service(
        target_environment, environment, sources, source_path, 1, &entry,
        (iree_vm_module_span_t){linked, libraries.count + 1}, report,
        host_allocator, &program);
  }
  iree_allocator_free(host_allocator, linked);
  if (iree_status_is_ok(status)) {
    iree_vm_function_t function = {0};
    status = loom_serve_program_lookup(program, entry, &function);
    if (iree_status_is_ok(status)) {
      status = iree_vm_invoke(loom_serve_program_invocation(program), function,
                              arguments, results);
    }
  }
  loom_serve_program_free(program);
  loom_serve_preparation_t* preparation =
      loom_serve_preparation_module_take(library);
  iree_vm_module_release(library);
  if (iree_status_is_ok(status) &&
      (!preparation || !preparation->stage_count)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "model preparation declared no stages");
  }
  if (iree_status_is_ok(status)) {
    *out_preparation = preparation;
  } else {
    loom_serve_preparation_free(preparation);
  }
  return status;
}

iree_host_size_t loom_serve_preparation_stage_count(
    const loom_serve_preparation_t* preparation) {
  return preparation->stage_count;
}

loom_model_card_t* loom_serve_preparation_card(
    const loom_serve_preparation_t* preparation) {
  return preparation->card;
}

const loom_model_component_t* loom_serve_preparation_component(
    const loom_serve_preparation_t* preparation) {
  return preparation->component;
}

uint32_t loom_serve_preparation_source_count(
    const loom_serve_preparation_t* preparation) {
  return preparation ? preparation->first_source + preparation->source_count
                     : 0;
}

loom_model_file_list_t loom_serve_preparation_source(
    const loom_serve_preparation_t* preparation, uint32_t ordinal) {
  if (ordinal < preparation->first_source) {
    return loom_serve_preparation_source(preparation->base, ordinal);
  }
  return preparation->sources[ordinal - preparation->first_source];
}

uint32_t loom_serve_preparation_local_sources(
    const loom_serve_preparation_t* preparation,
    const loom_model_file_list_t** out_sources) {
  *out_sources = preparation ? preparation->sources : NULL;
  return preparation ? preparation->source_count : 0;
}

iree_host_size_t loom_serve_preparation_storage_size(
    const loom_serve_preparation_t* preparation) {
  return preparation ? preparation->storage_size : 0;
}

iree_string_view_t loom_serve_preparation_memory_forecast(
    const loom_serve_preparation_t* preparation) {
  return preparation->memory_forecast;
}

static bool loom_serve_preparation_stage_code_equal(
    const loom_serve_preparation_stage_t* a,
    const loom_serve_preparation_stage_t* b) {
  if (a->kind != b->kind || !iree_string_view_equal(a->root, b->root) ||
      a->config.flags != b->config.flags ||
      a->config.binding_count != b->config.binding_count ||
      !iree_string_view_equal(
          iree_string_view_from_loomc(a->config.json_object),
          iree_string_view_from_loomc(b->config.json_object))) {
    return false;
  }
  for (iree_host_size_t i = 0; i < a->config.binding_count; ++i) {
    const loomc_config_binding_t* x = &a->config.bindings[i];
    const loomc_config_binding_t* y = &b->config.bindings[i];
    if (x->key.size != y->key.size || x->value.size != y->value.size ||
        (x->key.size && memcmp(x->key.data, y->key.data, x->key.size)) ||
        (x->value.size &&
         memcmp(x->value.data, y->value.data, x->value.size))) {
      return false;
    }
  }
  return true;
}

bool loom_serve_preparation_code_equal(const loom_serve_preparation_t* lhs,
                                       const loom_serve_preparation_t* rhs) {
  if (lhs == rhs) {
    return true;
  }
  if (!lhs || !rhs || lhs->stage_count != rhs->stage_count) {
    return false;
  }
  for (iree_host_size_t i = 0; i < lhs->stage_count; ++i) {
    if (!loom_serve_preparation_stage_code_equal(&lhs->stages[i].declaration,
                                                 &rhs->stages[i].declaration)) {
      return false;
    }
  }
  return true;
}

bool loom_serve_preparation_equal(const loom_serve_preparation_t* lhs,
                                  const loom_serve_preparation_t* rhs) {
  if (lhs == rhs) {
    return true;
  }
  if (!lhs || !rhs || lhs->stage_count != rhs->stage_count ||
      loom_serve_preparation_source_count(lhs) !=
          loom_serve_preparation_source_count(rhs) ||
      !iree_string_view_equal(lhs->memory_forecast, rhs->memory_forecast)) {
    return false;
  }
  for (uint32_t s = 0; s < loom_serve_preparation_source_count(lhs); ++s) {
    const loom_model_file_list_t a = loom_serve_preparation_source(lhs, s);
    const loom_model_file_list_t b = loom_serve_preparation_source(rhs, s);
    if (a.count != b.count) {
      return false;
    }
    for (iree_host_size_t i = 0; i < a.count; ++i) {
      if (!iree_string_view_equal(a.values[i].path, b.values[i].path) ||
          !iree_string_view_equal(a.values[i].format, b.values[i].format)) {
        return false;
      }
    }
  }
  for (iree_host_size_t i = 0; i < lhs->stage_count; ++i) {
    const loom_serve_preparation_stage_t* a = &lhs->stages[i].declaration;
    const loom_serve_preparation_stage_t* b = &rhs->stages[i].declaration;
    if (!loom_serve_preparation_stage_code_equal(a, b) || a->tag != b->tag ||
        a->workspace_limit_bytes != b->workspace_limit_bytes ||
        a->parameter_count != b->parameter_count) {
      return false;
    }
    for (iree_host_size_t j = 0; j < a->parameter_count; ++j) {
      if (a->parameters[j].binding != b->parameters[j].binding ||
          !loom_serve_preparation_parameter_equal(&a->parameters[j],
                                                  &b->parameters[j])) {
        return false;
      }
    }
  }
  return true;
}

const loom_serve_preparation_stage_t* loom_serve_preparation_stage(
    const loom_serve_preparation_t* preparation, iree_host_size_t index) {
  return &preparation->stages[index].declaration;
}

iree_status_t loom_serve_preparation_stage_check_workspace(
    const loom_serve_preparation_stage_t* stage,
    uint64_t required_byte_length) {
  if (stage->workspace_limit_bytes &&
      required_byte_length > stage->workspace_limit_bytes) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "stage '%.*s' requires %" PRIu64
        " bytes of transient workspace, exceeding its %" PRIu64
        "-byte source limit",
        (int)stage->root.size, stage->root.data, required_byte_length,
        stage->workspace_limit_bytes);
  }
  return iree_ok_status();
}

iree_status_t loom_serve_preparation_stage_check_parameters(
    const loom_serve_preparation_stage_t* stage,
    const loom_cmd_program_t* program,
    const loom_serve_preparation_stage_t* canonical_stage,
    const loom_cmd_program_t* canonical_program) {
  const uint32_t roots =
      canonical_program ? canonical_program->requirements.fixed_buffer_count
                        : 0;
  if (stage->parameter_count != roots ||
      (program ? program->requirements.fixed_buffer_count : 0) != roots ||
      (program ? program->parameter_roots.count : 0) != roots) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "stage '%.*s' checkpoint domains differ from the shared layout",
        (int)stage->root.size, stage->root.data);
  }
  for (uint32_t root = 0; root < roots; ++root) {
    const loom_serve_preparation_parameter_t* parameter =
        &stage->parameters[root];
    const loom_serve_preparation_parameter_t* shared =
        &canonical_stage->parameters[root];
    const loom_cmd_program_parameter_root_t rhs =
        loom_cmd_program_parameter_root_at(program, root);
    if (parameter->binding != root ||
        rhs.fixed_buffer_index != parameter->binding ||
        !loom_serve_preparation_parameter_equal(parameter, shared)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "stage '%.*s' checkpoint domain %u differs from the shared layout",
          (int)stage->root.size, stage->root.data, root);
    }
  }
  if (!program || !canonical_program || program == canonical_program) {
    return iree_ok_status();
  }
  return iree_status_annotate_f(
      loom_serve_parameter_layout_check(canonical_program, program),
      "stage '%.*s' differs from the shared layout", (int)stage->root.size,
      stage->root.data);
}
