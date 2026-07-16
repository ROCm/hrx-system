// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/ideogram4/lora_bake.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "experimental/id4/pipeline/parameter_layout.h"
#include "iree/base/internal/atomics.h"

enum {
  ID4_IDEOGRAM4_LORA_BAKE_CONFIG_COUNT = 3,
  ID4_IDEOGRAM4_LORA_BAKE_CONFIG_VALUE_CAPACITY = 16,
  ID4_IDEOGRAM4_LORA_BAKE_TILE_SIZE = 16,
};

typedef struct id4_ideogram4_lora_bake_kernel_t {
  // Exported function selected as the Loom link root.
  iree_string_view_t function_name;
} id4_ideogram4_lora_bake_kernel_t;

static const iree_string_view_t id4_ideogram4_lora_bake_module_path =
    IREE_SVL("ideogram4/lora_bake_bf16");

static const struct {
  id4_ideogram4_lora_bake_kernel_t pack_down;
  id4_ideogram4_lora_bake_kernel_t decode_dense;
  id4_ideogram4_lora_bake_kernel_t decode_compact_rhs;
  id4_ideogram4_lora_bake_kernel_t update;
  id4_ideogram4_lora_bake_kernel_t update_wmma;
  id4_ideogram4_lora_bake_kernel_t scale;
  id4_ideogram4_lora_bake_kernel_t encode_dense;
  id4_ideogram4_lora_bake_kernel_t encode_compact_rhs;
} id4_ideogram4_lora_bake_kernels = {
    .pack_down =
        {
            .function_name = IREE_SVL("id4_ideogram4_lora_bake_pack_down_bf16"),
        },
    .decode_dense =
        {
            .function_name =
                IREE_SVL("id4_ideogram4_lora_bake_decode_dense_fp8_f32"),
        },
    .decode_compact_rhs =
        {
            .function_name = IREE_SVL("id4_ideogram4_lora_bake_decode_fp8_f32"),
        },
    .update =
        {
            .function_name =
                IREE_SVL("id4_ideogram4_lora_bake_update_bf16_f32"),
        },
    .update_wmma =
        {
            .function_name =
                IREE_SVL("id4_ideogram4_lora_bake_update_bf16_f32_wmma"),
        },
    .scale =
        {
            .function_name = IREE_SVL("id4_ideogram4_lora_bake_scale_f32_fp8"),
        },
    .encode_dense =
        {
            .function_name =
                IREE_SVL("id4_ideogram4_lora_bake_encode_f32_dense_fp8"),
        },
    .encode_compact_rhs =
        {
            .function_name = IREE_SVL("id4_ideogram4_lora_bake_encode_f32_fp8"),
        },
};

static const iree_string_view_t id4_ideogram4_lora_bake_config_keys[] = {
    IREE_SVL("id4.ideogram4.lora_bake.output_row_count"),
    IREE_SVL("id4.ideogram4.lora_bake.input_size"),
    IREE_SVL("id4.ideogram4.lora_bake.rank"),
};

typedef struct id4_ideogram4_lora_bake_config_t {
  // Loom config bindings supplied to one specialization.
  id4_pipeline_kernel_config_binding_t
      bindings[ID4_IDEOGRAM4_LORA_BAKE_CONFIG_COUNT];
  // Decimal strings backing bindings for the duration of preparation.
  char values[ID4_IDEOGRAM4_LORA_BAKE_CONFIG_COUNT]
             [ID4_IDEOGRAM4_LORA_BAKE_CONFIG_VALUE_CAPACITY];
} id4_ideogram4_lora_bake_config_t;

typedef struct id4_ideogram4_lora_bake_specialization_t {
  // Static kernel descriptor identifying the operation family.
  const id4_ideogram4_lora_bake_kernel_t* kernel;
  // Number of output rows processed by this specialization.
  uint32_t output_row_count;
  // Linear input feature count processed by this specialization.
  uint32_t input_size;
  // Adapter rank processed by this specialization.
  uint32_t rank;
  // Prepared executable wrapper retained until all dispatches are submitted.
  id4_pipeline_kernel_executable_t* executable;
  // Resolved exported HAL function.
  iree_hal_executable_function_t function;
} id4_ideogram4_lora_bake_specialization_t;

typedef struct id4_ideogram4_lora_bake_specialization_list_t {
  // Maximum number of records in values.
  iree_host_size_t capacity;
  // Number of initialized records in values.
  iree_host_size_t count;
  // Prepared specialization records.
  id4_ideogram4_lora_bake_specialization_t* values;
  // Host allocator owning values.
  iree_allocator_t host_allocator;
} id4_ideogram4_lora_bake_specialization_list_t;

struct id4_ideogram4_lora_bake_prepared_t {
  // Reference count for shared prepared-program ownership.
  iree_atomic_ref_count_t ref_count;
  // Host allocator owning this object and its specialization list.
  iree_allocator_t host_allocator;
  // Immutable bounded bake schedule retained by this object.
  const id4_ideogram4_lora_bake_plan_t* plan;
  // Exact base pipeline plan retained transitively by |plan|.
  const id4_pipeline_plan_t* base_plan;
  // Immutable LoRA topology retained transitively by |plan|.
  id4_ideogram4_lora_topology_t* topology;
  // Planned placement used before the patchable domain allocation exists.
  const id4_pipeline_device_placement_t* planned_placement;
  // Planned device retained transitively by |base_plan|.
  iree_hal_device_t* planned_device;
  // Shared working-set range receiving issue-time adapter strengths.
  id4_ideogram4_lora_bake_parameter_range_t strength_range;
  // Prepared Loom specializations required by |plan|.
  id4_ideogram4_lora_bake_specialization_list_t specializations;
};

// Two-semaphore timeline that preserves the last accepted edge when a new
// submission is rejected. Alternating semaphores prevents a rejected
// submission from poisoning the edge required to clean up prior work.
typedef struct id4_ideogram4_lora_bake_chain_t {
  // Timeline semaphores alternated by accepted submissions.
  iree_hal_semaphore_t* semaphores[2];
  // Last proposed payload on each semaphore.
  uint64_t payload_values[2];
  // Index of the last successfully submitted signal edge.
  uint32_t current_index;
  // Whether current_index identifies an accepted edge.
  bool has_current;
} id4_ideogram4_lora_bake_chain_t;

typedef struct id4_ideogram4_lora_bake_submit_context_t {
  // Caller options borrowed for the submit call.
  const id4_ideogram4_lora_bake_submit_options_t* options;
  // Prepared immutable bake program borrowed for the submit call.
  const id4_ideogram4_lora_bake_prepared_t* prepared;
  // Actual placement of the queue-allocated patchable domain.
  iree_hal_buffer_placement_t domain_placement;
  // Actual placement of the queue-allocated working storage.
  iree_hal_buffer_placement_t working_placement;
} id4_ideogram4_lora_bake_submit_context_t;

static iree_status_t id4_ideogram4_lora_bake_validate_semaphore_list(
    iree_hal_semaphore_list_t list, iree_string_view_t name) {
  if (list.count == 0) return iree_ok_status();
  if (!list.semaphores || !list.payload_values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%.*s semaphore list is incomplete", (int)name.size,
                            name.data);
  }
  for (iree_host_size_t i = 0; i < list.count; ++i) {
    if (!list.semaphores[i]) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "%.*s semaphore %" PRIhsz " is NULL",
                              (int)name.size, name.data, i);
    }
  }
  return iree_ok_status();
}

static bool id4_ideogram4_lora_bake_semaphore_list_is_well_formed(
    iree_hal_semaphore_list_t list) {
  if (list.count == 0 || !list.semaphores || !list.payload_values) return false;
  for (iree_host_size_t i = 0; i < list.count; ++i) {
    if (!list.semaphores[i]) return false;
  }
  return true;
}

static void id4_ideogram4_lora_bake_fail_caller_signal(
    const id4_ideogram4_lora_bake_submit_options_t* options,
    iree_status_code_t status_code) {
  if (!options || !id4_ideogram4_lora_bake_semaphore_list_is_well_formed(
                      options->signal_semaphore_list)) {
    return;
  }
  iree_hal_semaphore_list_fail(options->signal_semaphore_list,
                               iree_status_from_code(status_code));
}

static iree_status_t id4_ideogram4_lora_bake_validate_prepare_options(
    const id4_ideogram4_lora_bake_prepare_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LoRA bake prepare options are required");
  }
  if (options->structure_size != sizeof(*options)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LoRA bake prepare options size is invalid");
  }
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "LoRA bake prepare extension structures are not supported");
  }
  if (!options->plan || !options->kernel_library || !options->kernel_cache ||
      !options->executable_cache) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LoRA bake plan, kernel library, kernel cache, and executable cache "
        "are required");
  }
  return id4_pipeline_diagnostics_validate_sink(
      options->diagnostics_sink, IREE_SV("Ideogram 4 LoRA bake prepare"));
}

static iree_status_t id4_ideogram4_lora_bake_initialize_prepared_layout(
    id4_ideogram4_lora_bake_prepared_t* prepared) {
  prepared->base_plan = id4_ideogram4_lora_bake_plan_base_plan(prepared->plan);
  prepared->topology = id4_ideogram4_lora_bake_plan_topology(prepared->plan);
  const iree_host_size_t slab_index =
      id4_ideogram4_lora_bake_plan_patchable_slab_index(prepared->plan);
  const id4_pipeline_parameter_slab_plan_t* patchable_slab =
      id4_pipeline_plan_parameter_slab_at(prepared->base_plan, slab_index);
  if (!patchable_slab ||
      patchable_slab->byte_length !=
          id4_ideogram4_lora_bake_plan_patchable_slab_byte_length(
              prepared->plan)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LoRA bake patchable slab no longer matches the base plan");
  }
  prepared->planned_placement = id4_pipeline_plan_placement_at(
      prepared->base_plan, patchable_slab->placement_id);
  iree_hal_device_group_t* device_group =
      id4_pipeline_plan_device_group(prepared->base_plan);
  prepared->planned_device =
      prepared->planned_placement
          ? iree_hal_device_group_device_at(
                device_group, prepared->planned_placement->device_index)
          : NULL;
  if (!prepared->planned_placement || !prepared->planned_device) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LoRA bake patchable slab references an invalid device placement");
  }

  const iree_host_size_t target_count =
      id4_ideogram4_lora_bake_plan_target_count(prepared->plan);
  if (target_count == 0 ||
      target_count !=
          id4_ideogram4_lora_topology_target_count(prepared->topology)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LoRA bake plan and topology target counts do not match");
  }
  iree_device_size_t strength_byte_length = 0;
  if (!iree_device_size_checked_mul(
          id4_ideogram4_lora_topology_adapter_count(prepared->topology),
          sizeof(float), &strength_byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "LoRA bake strength storage size overflows");
  }
  for (iree_host_size_t i = 0; i < target_count; ++i) {
    const id4_ideogram4_lora_bake_target_t* target =
        id4_ideogram4_lora_bake_plan_target_at(prepared->plan, i);
    if (!target || target->working_set.strengths.offset != 0 ||
        target->working_set.strengths.length != strength_byte_length) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "LoRA bake targets do not share one strength storage range");
    }
  }
  prepared->strength_range =
      id4_ideogram4_lora_bake_plan_target_at(prepared->plan, 0)
          ->working_set.strengths;
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_lora_bake_validate_submit_options(
    const id4_ideogram4_lora_bake_submit_options_t* options,
    id4_ideogram4_lora_bake_submit_context_t* out_context) {
  memset(out_context, 0, sizeof(*out_context));
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LoRA bake submit options are required");
  }
  if (options->structure_size != sizeof(*options)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LoRA bake submit options size is invalid");
  }
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "LoRA bake submit extension structures are not supported");
  }
  if (!options->prepared || !options->adapter_provider) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "prepared LoRA bake program and adapter provider are required");
  }
  const id4_pipeline_parameter_source_t* base_parameter_source =
      &options->base_parameter_source;
  if (base_parameter_source->kind !=
          ID4_PIPELINE_PARAMETER_SOURCE_KIND_EXECUTION_LAYOUT ||
      !base_parameter_source->storage.execution_layout.index ||
      !base_parameter_source->storage.execution_layout.provider ||
      iree_string_view_is_empty(
          base_parameter_source->storage.execution_layout.scope) ||
      !iree_io_parameter_provider_query_support(
          base_parameter_source->storage.execution_layout.provider,
          base_parameter_source->storage.execution_layout.scope)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LoRA bake requires a supported execution-layout base source");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_diagnostics_validate_sink(
      options->diagnostics_sink, IREE_SV("Ideogram 4 LoRA bake")));
  IREE_RETURN_IF_ERROR(id4_ideogram4_lora_bake_validate_semaphore_list(
      options->wait_semaphore_list, IREE_SV("LoRA bake wait")));
  IREE_RETURN_IF_ERROR(id4_ideogram4_lora_bake_validate_semaphore_list(
      options->signal_semaphore_list, IREE_SV("LoRA bake signal")));
  if (options->signal_semaphore_list.count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LoRA bake signal semaphore list is empty");
  }
  if (iree_all_bits_set(options->working_set_alloca_flags,
                        IREE_HAL_ALLOCA_FLAG_INDETERMINATE_LIFETIME)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LoRA bake working storage requires a determinate lifetime");
  }

  const id4_ideogram4_lora_bake_prepared_t* prepared = options->prepared;
  IREE_RETURN_IF_ERROR(id4_pipeline_parameter_layout_validate_index(
      prepared->base_plan,
      base_parameter_source->storage.execution_layout.index));

  const iree_host_size_t adapter_count =
      id4_ideogram4_lora_topology_adapter_count(prepared->topology);
  if (options->strength_count != adapter_count || !options->strength_values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LoRA bake requires exactly %" PRIhsz
                            " strength values",
                            adapter_count);
  }
  for (iree_host_size_t i = 0; i < options->strength_count; ++i) {
    if (!isfinite(options->strength_values[i])) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "LoRA bake strength %" PRIhsz " is not finite",
                              i);
    }
  }
  for (iree_host_size_t i = 0; i < adapter_count; ++i) {
    const iree_string_view_t source_scope =
        id4_ideogram4_lora_topology_adapter_source_scope(prepared->topology, i);
    if (iree_string_view_is_empty(source_scope) ||
        !iree_io_parameter_provider_query_support(options->adapter_provider,
                                                  source_scope)) {
      return iree_make_status(
          IREE_STATUS_NOT_FOUND,
          "LoRA bake provider does not support adapter scope '%.*s'",
          (int)source_scope.size, source_scope.data);
    }
  }

  *out_context = (id4_ideogram4_lora_bake_submit_context_t){
      .options = options,
      .prepared = prepared,
  };
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_lora_bake_format_config_value(
    uint32_t value, char* buffer, iree_host_size_t buffer_capacity,
    iree_string_view_t* out_value) {
  const int length = snprintf(buffer, buffer_capacity, "%" PRIu32, value);
  if (length < 0 || (iree_host_size_t)length >= buffer_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "LoRA bake config value is out of range");
  }
  *out_value = iree_make_string_view(buffer, (iree_host_size_t)length);
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_lora_bake_make_config(
    uint32_t output_row_count, uint32_t input_size, uint32_t rank,
    id4_ideogram4_lora_bake_config_t* out_config) {
  memset(out_config, 0, sizeof(*out_config));
  const uint32_t values[] = {output_row_count, input_size, rank};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(values); ++i) {
    iree_string_view_t value = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(id4_ideogram4_lora_bake_format_config_value(
        values[i], out_config->values[i], IREE_ARRAYSIZE(out_config->values[i]),
        &value));
    out_config->bindings[i] = id4_pipeline_make_kernel_config_binding(
        id4_ideogram4_lora_bake_config_keys[i], value);
  }
  return iree_ok_status();
}

static void id4_ideogram4_lora_bake_specialization_list_deinitialize(
    id4_ideogram4_lora_bake_specialization_list_t* list) {
  if (!list) return;
  for (iree_host_size_t i = 0; i < list->count; ++i) {
    id4_pipeline_kernel_executable_release(list->values[i].executable);
  }
  iree_allocator_free(list->host_allocator, list->values);
  memset(list, 0, sizeof(*list));
}

static const id4_ideogram4_lora_bake_specialization_t*
id4_ideogram4_lora_bake_specialization_find(
    const id4_ideogram4_lora_bake_specialization_list_t* list,
    const id4_ideogram4_lora_bake_kernel_t* kernel, uint32_t output_row_count,
    uint32_t input_size, uint32_t rank) {
  for (iree_host_size_t i = 0; i < list->count; ++i) {
    const id4_ideogram4_lora_bake_specialization_t* specialization =
        &list->values[i];
    if (specialization->kernel == kernel &&
        specialization->output_row_count == output_row_count &&
        specialization->input_size == input_size &&
        specialization->rank == rank) {
      return specialization;
    }
  }
  return NULL;
}

static const id4_ideogram4_lora_bake_kernel_t*
id4_ideogram4_lora_bake_decode_kernel(
    id4_ideogram4_lora_bake_weight_layout_t layout) {
  switch (layout) {
    case ID4_IDEOGRAM4_LORA_BAKE_WEIGHT_LAYOUT_DENSE_ROW_MAJOR:
      return &id4_ideogram4_lora_bake_kernels.decode_dense;
    case ID4_IDEOGRAM4_LORA_BAKE_WEIGHT_LAYOUT_COMPACT_RHS_TILE:
      return &id4_ideogram4_lora_bake_kernels.decode_compact_rhs;
    default:
      return NULL;
  }
}

static const id4_ideogram4_lora_bake_kernel_t*
id4_ideogram4_lora_bake_encode_kernel(
    id4_ideogram4_lora_bake_weight_layout_t layout) {
  switch (layout) {
    case ID4_IDEOGRAM4_LORA_BAKE_WEIGHT_LAYOUT_DENSE_ROW_MAJOR:
      return &id4_ideogram4_lora_bake_kernels.encode_dense;
    case ID4_IDEOGRAM4_LORA_BAKE_WEIGHT_LAYOUT_COMPACT_RHS_TILE:
      return &id4_ideogram4_lora_bake_kernels.encode_compact_rhs;
    default:
      return NULL;
  }
}

static iree_status_t id4_ideogram4_lora_bake_specialization_add(
    const id4_ideogram4_lora_bake_prepare_options_t* options,
    const id4_ideogram4_lora_bake_prepared_t* prepared,
    const id4_ideogram4_lora_bake_kernel_t* kernel, uint32_t output_row_count,
    uint32_t input_size, uint32_t rank,
    id4_ideogram4_lora_bake_specialization_list_t* list) {
  if (id4_ideogram4_lora_bake_specialization_find(
          list, kernel, output_row_count, input_size, rank)) {
    return iree_ok_status();
  }
  if (list->count >= list->capacity) {
    IREE_RETURN_IF_ERROR(iree_allocator_grow_array(
        list->host_allocator, iree_max((iree_host_size_t)16, list->count + 1),
        sizeof(list->values[0]), &list->capacity, (void**)&list->values));
  }

  const id4_pipeline_kernel_module_t* module = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_kernel_library_lookup(
      options->kernel_library, id4_ideogram4_lora_bake_module_path, &module));
  id4_ideogram4_lora_bake_config_t config;
  IREE_RETURN_IF_ERROR(id4_ideogram4_lora_bake_make_config(
      output_row_count, input_size, rank, &config));
  const id4_pipeline_kernel_cache_prepare_options_t prepare_options = {
      .structure_size = sizeof(prepare_options),
      .executable_cache = options->executable_cache,
      .queue_affinity = prepared->planned_placement->queue_affinity,
      .caching_mode = options->executable_caching_mode,
      .source_identifier = module->source_identifier,
      .source_contents = module->source_contents,
      .module_path = module->module_path,
      .function_name = kernel->function_name,
      .config_binding_count = IREE_ARRAYSIZE(config.bindings),
      .config_bindings = config.bindings,
      .diagnostic_artifact_flags = options->kernel_diagnostic_artifact_flags,
      .diagnostics_sink = options->diagnostics_sink,
  };

  id4_pipeline_kernel_executable_t* executable = NULL;
  IREE_RETURN_IF_ERROR(id4_pipeline_kernel_cache_prepare_executable(
      options->kernel_cache, &prepare_options, &executable));
  iree_hal_executable_function_t function =
      iree_hal_executable_function_invalid();
  iree_status_t status = iree_hal_executable_lookup_function_by_name(
      id4_pipeline_kernel_executable_hal_executable(executable),
      kernel->function_name, &function);
  if (iree_status_is_ok(status)) {
    list->values[list->count++] = (id4_ideogram4_lora_bake_specialization_t){
        .kernel = kernel,
        .output_row_count = output_row_count,
        .input_size = input_size,
        .rank = rank,
        .executable = executable,
        .function = function,
    };
    executable = NULL;
  }
  id4_pipeline_kernel_executable_release(executable);
  return status;
}

static iree_status_t id4_ideogram4_lora_bake_prepare_specializations(
    const id4_ideogram4_lora_bake_prepare_options_t* options,
    const id4_ideogram4_lora_bake_prepared_t* prepared,
    iree_allocator_t host_allocator,
    id4_ideogram4_lora_bake_specialization_list_t* out_list) {
  memset(out_list, 0, sizeof(*out_list));
  out_list->host_allocator = host_allocator;

  iree_status_t status = iree_ok_status();
  const iree_host_size_t target_count =
      id4_ideogram4_lora_bake_plan_target_count(prepared->plan);
  for (iree_host_size_t i = 0; i < target_count && iree_status_is_ok(status);
       ++i) {
    const id4_ideogram4_lora_bake_target_t* target =
        id4_ideogram4_lora_bake_plan_target_at(prepared->plan, i);
    const id4_ideogram4_dit_lora_target_t* source_target =
        id4_ideogram4_lora_topology_target_at(prepared->topology, i);
    for (iree_host_size_t j = 0;
         j < source_target->segment_count && iree_status_is_ok(status); ++j) {
      status = id4_ideogram4_lora_bake_specialization_add(
          options, prepared, &id4_ideogram4_lora_bake_kernels.pack_down,
          ID4_IDEOGRAM4_LORA_BAKE_TILE_SIZE, target->input_size,
          source_target->segments[j].rank, out_list);
    }

    const uint32_t full_row_count = target->output_rows_per_window;
    const uint32_t final_row_count =
        target->output_size - (target->window_count - 1) * full_row_count;
    const uint32_t row_counts[] = {full_row_count, final_row_count};
    const id4_ideogram4_lora_bake_kernel_t* decode_kernel =
        id4_ideogram4_lora_bake_decode_kernel(target->weight_layout);
    const id4_ideogram4_lora_bake_kernel_t* encode_kernel =
        id4_ideogram4_lora_bake_encode_kernel(target->weight_layout);
    if (!decode_kernel || !encode_kernel) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "LoRA bake target `%.*s` has invalid weight layout %u",
          (int)target->base_parameter_key.size, target->base_parameter_key.data,
          (unsigned)target->weight_layout);
    }
    for (iree_host_size_t j = 0;
         j < IREE_ARRAYSIZE(row_counts) && iree_status_is_ok(status); ++j) {
      const uint32_t row_count = row_counts[j];
      if (j > 0 && row_count == row_counts[0]) continue;
      status = id4_ideogram4_lora_bake_specialization_add(
          options, prepared, decode_kernel, row_count, target->input_size,
          /*rank=*/1, out_list);
      if (iree_status_is_ok(status)) {
        status = id4_ideogram4_lora_bake_specialization_add(
            options, prepared, &id4_ideogram4_lora_bake_kernels.scale,
            row_count, target->input_size, /*rank=*/1, out_list);
      }
      if (iree_status_is_ok(status)) {
        status = id4_ideogram4_lora_bake_specialization_add(
            options, prepared, encode_kernel, row_count, target->input_size,
            /*rank=*/1, out_list);
      }
      for (iree_host_size_t k = 0;
           k < source_target->segment_count && iree_status_is_ok(status); ++k) {
        const uint32_t rank = source_target->segments[k].rank;
        const id4_ideogram4_lora_bake_kernel_t* update_kernel =
            rank % ID4_IDEOGRAM4_LORA_BAKE_TILE_SIZE == 0
                ? &id4_ideogram4_lora_bake_kernels.update_wmma
                : &id4_ideogram4_lora_bake_kernels.update;
        status = id4_ideogram4_lora_bake_specialization_add(
            options, prepared, update_kernel, row_count, target->input_size,
            rank, out_list);
      }
    }
  }
  if (!iree_status_is_ok(status)) {
    id4_ideogram4_lora_bake_specialization_list_deinitialize(out_list);
  }
  return status;
}

static void id4_ideogram4_lora_bake_prepared_destroy(
    id4_ideogram4_lora_bake_prepared_t* prepared) {
  iree_allocator_t host_allocator = prepared->host_allocator;
  id4_ideogram4_lora_bake_specialization_list_deinitialize(
      &prepared->specializations);
  id4_ideogram4_lora_bake_plan_release(
      (id4_ideogram4_lora_bake_plan_t*)prepared->plan);
  iree_allocator_free(host_allocator, prepared);
}

iree_status_t id4_ideogram4_lora_bake_prepare(
    const id4_ideogram4_lora_bake_prepare_options_t* options,
    iree_allocator_t host_allocator,
    id4_ideogram4_lora_bake_prepared_t** out_prepared) {
  IREE_ASSERT_ARGUMENT(out_prepared);
  *out_prepared = NULL;
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_lora_bake_validate_prepare_options(options));

  id4_ideogram4_lora_bake_prepared_t* prepared = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*prepared),
                                             (void**)&prepared));
  memset(prepared, 0, sizeof(*prepared));
  iree_atomic_ref_count_init(&prepared->ref_count);
  prepared->host_allocator = host_allocator;
  prepared->plan = options->plan;
  id4_ideogram4_lora_bake_plan_retain(
      (id4_ideogram4_lora_bake_plan_t*)prepared->plan);

  iree_status_t status =
      id4_ideogram4_lora_bake_initialize_prepared_layout(prepared);
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_lora_bake_prepare_specializations(
        options, prepared, host_allocator, &prepared->specializations);
  }
  if (iree_status_is_ok(status)) {
    *out_prepared = prepared;
  } else {
    id4_ideogram4_lora_bake_prepared_destroy(prepared);
  }
  return status;
}

void id4_ideogram4_lora_bake_prepared_retain(
    id4_ideogram4_lora_bake_prepared_t* prepared) {
  if (!prepared) return;
  iree_atomic_ref_count_inc(&prepared->ref_count);
}

void id4_ideogram4_lora_bake_prepared_release(
    id4_ideogram4_lora_bake_prepared_t* prepared) {
  if (prepared && iree_atomic_ref_count_dec(&prepared->ref_count) == 1) {
    id4_ideogram4_lora_bake_prepared_destroy(prepared);
  }
}

static iree_status_t id4_ideogram4_lora_bake_chain_initialize(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    id4_ideogram4_lora_bake_chain_t* out_chain) {
  memset(out_chain, 0, sizeof(*out_chain));
  iree_status_t status = iree_hal_semaphore_create(device, queue_affinity, 0,
                                                   IREE_HAL_SEMAPHORE_FLAG_NONE,
                                                   &out_chain->semaphores[0]);
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(device, queue_affinity, 0,
                                       IREE_HAL_SEMAPHORE_FLAG_NONE,
                                       &out_chain->semaphores[1]);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_release(out_chain->semaphores[1]);
    iree_hal_semaphore_release(out_chain->semaphores[0]);
    memset(out_chain, 0, sizeof(*out_chain));
  }
  return status;
}

static void id4_ideogram4_lora_bake_chain_deinitialize(
    id4_ideogram4_lora_bake_chain_t* chain) {
  if (!chain) return;
  iree_hal_semaphore_release(chain->semaphores[1]);
  iree_hal_semaphore_release(chain->semaphores[0]);
  memset(chain, 0, sizeof(*chain));
}

static iree_hal_semaphore_list_t id4_ideogram4_lora_bake_chain_current(
    id4_ideogram4_lora_bake_chain_t* chain) {
  if (!chain->has_current) return iree_hal_semaphore_list_empty();
  return (iree_hal_semaphore_list_t){
      .count = 1,
      .semaphores = &chain->semaphores[chain->current_index],
      .payload_values = &chain->payload_values[chain->current_index],
  };
}

static iree_status_t id4_ideogram4_lora_bake_chain_next(
    id4_ideogram4_lora_bake_chain_t* chain, uint32_t* out_index,
    iree_hal_semaphore_list_t* out_signal_list) {
  const uint32_t index = chain->has_current ? chain->current_index ^ 1u : 0u;
  if (chain->payload_values[index] == UINT64_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "LoRA bake timeline payload overflow");
  }
  ++chain->payload_values[index];
  *out_index = index;
  *out_signal_list = (iree_hal_semaphore_list_t){
      .count = 1,
      .semaphores = &chain->semaphores[index],
      .payload_values = &chain->payload_values[index],
  };
  return iree_ok_status();
}

static void id4_ideogram4_lora_bake_chain_commit(
    id4_ideogram4_lora_bake_chain_t* chain, uint32_t index) {
  chain->current_index = index;
  chain->has_current = true;
}

static iree_status_t id4_ideogram4_lora_bake_query_buffer_placement(
    iree_hal_buffer_t* buffer, iree_string_view_t buffer_name,
    iree_hal_buffer_placement_t* out_placement) {
  *out_placement = iree_hal_buffer_placement_undefined();
  const iree_hal_buffer_placement_t placement =
      iree_hal_buffer_allocation_placement(buffer);
  if (iree_hal_buffer_placement_is_undefined(placement) ||
      iree_hal_queue_affinity_is_empty(placement.queue_affinity)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "%.*s has no queue allocation placement",
                            (int)buffer_name.size, buffer_name.data);
  }
  *out_placement = placement;
  return iree_ok_status();
}

static iree_hal_semaphore_list_t id4_ideogram4_lora_bake_join_two_edges(
    iree_hal_semaphore_list_t first, iree_hal_semaphore_list_t second,
    iree_hal_semaphore_t* semaphores[2], uint64_t payload_values[2]) {
  semaphores[0] = first.semaphores[0];
  semaphores[1] = second.semaphores[0];
  payload_values[0] = first.payload_values[0];
  payload_values[1] = second.payload_values[0];
  return (iree_hal_semaphore_list_t){
      .count = 2,
      .semaphores = semaphores,
      .payload_values = payload_values,
  };
}

static iree_status_t id4_ideogram4_lora_bake_submit_dispatch(
    const id4_ideogram4_lora_bake_submit_context_t* context,
    const id4_ideogram4_lora_bake_specialization_t* specialization,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_buffer_ref_list_t bindings,
    id4_ideogram4_lora_bake_chain_t* chain) {
  if (!specialization) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "LoRA bake dispatch specialization was not prepared");
  }
  uint32_t next_index = 0;
  iree_hal_semaphore_list_t signal_semaphore_list;
  IREE_RETURN_IF_ERROR(id4_ideogram4_lora_bake_chain_next(
      chain, &next_index, &signal_semaphore_list));
  iree_status_t status = iree_hal_device_queue_dispatch(
      context->domain_placement.device,
      context->domain_placement.queue_affinity, wait_semaphore_list,
      signal_semaphore_list,
      id4_pipeline_kernel_executable_hal_executable(specialization->executable),
      specialization->function,
      id4_pipeline_kernel_executable_dispatch_config(
          specialization->executable),
      iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE);
  if (iree_status_is_ok(status)) {
    id4_ideogram4_lora_bake_chain_commit(chain, next_index);
  }
  return status;
}

static iree_status_t id4_ideogram4_lora_bake_submit_parameter_read(
    const id4_ideogram4_lora_bake_submit_context_t* context,
    iree_string_view_t source_scope, iree_string_view_t source_key,
    uint64_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    id4_ideogram4_lora_bake_chain_t* chain) {
  uint32_t next_index = 0;
  iree_hal_semaphore_list_t signal_semaphore_list;
  IREE_RETURN_IF_ERROR(id4_ideogram4_lora_bake_chain_next(
      chain, &next_index, &signal_semaphore_list));
  iree_status_t status = iree_io_parameter_provider_read(
      context->options->adapter_provider, context->working_placement.device,
      context->working_placement.queue_affinity,
      id4_ideogram4_lora_bake_chain_current(chain), signal_semaphore_list,
      source_scope, source_key, source_offset, target_buffer, target_offset,
      length);
  if (iree_status_is_ok(status)) {
    id4_ideogram4_lora_bake_chain_commit(chain, next_index);
  }
  return status;
}

static iree_status_t id4_ideogram4_lora_bake_checked_byte_length(
    uint32_t dim0, uint32_t dim1, iree_device_size_t element_size,
    iree_device_size_t* out_byte_length) {
  iree_device_size_t element_count = 0;
  if (!iree_device_size_checked_mul(dim0, dim1, &element_count) ||
      !iree_device_size_checked_mul(element_count, element_size,
                                    out_byte_length)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "LoRA bake tensor byte length overflows");
  }
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_lora_bake_weight_window(
    const id4_ideogram4_lora_bake_target_t* target, uint32_t row_offset,
    uint32_t row_count, iree_device_size_t* out_weight_offset,
    iree_device_size_t* out_weight_length, iree_device_size_t* out_scale_offset,
    iree_device_size_t* out_scale_length) {
  iree_device_size_t row_tile_offset =
      row_offset / ID4_IDEOGRAM4_LORA_BAKE_TILE_SIZE;
  iree_device_size_t input_tile_count =
      target->input_size / ID4_IDEOGRAM4_LORA_BAKE_TILE_SIZE;
  iree_device_size_t tile_offset = 0;
  iree_device_size_t byte_offset = 0;
  iree_device_size_t scale_byte_offset = 0;
  if (!iree_device_size_checked_mul(row_tile_offset, input_tile_count,
                                    &tile_offset) ||
      !iree_device_size_checked_mul(
          tile_offset,
          ID4_IDEOGRAM4_LORA_BAKE_TILE_SIZE * ID4_IDEOGRAM4_LORA_BAKE_TILE_SIZE,
          &byte_offset) ||
      !iree_device_size_checked_add(target->weight_range.offset, byte_offset,
                                    out_weight_offset) ||
      !iree_device_size_checked_mul(row_offset, sizeof(float),
                                    &scale_byte_offset) ||
      !iree_device_size_checked_add(target->scale_range.offset,
                                    scale_byte_offset, out_scale_offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "LoRA bake target window offset overflows");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_lora_bake_checked_byte_length(
      row_count, target->input_size, /*element_size=*/1, out_weight_length));
  IREE_RETURN_IF_ERROR(id4_ideogram4_lora_bake_checked_byte_length(
      row_count, /*dim1=*/1, sizeof(float), out_scale_length));
  return iree_ok_status();
}

static iree_status_t id4_ideogram4_lora_bake_segment_lengths(
    const id4_ideogram4_lora_bake_target_t* target, uint32_t rank,
    iree_device_size_t* out_source_length,
    iree_device_size_t* out_packed_length) {
  iree_device_size_t padded_rank = 0;
  if (!iree_device_size_checked_align(rank, ID4_IDEOGRAM4_LORA_BAKE_TILE_SIZE,
                                      &padded_rank) ||
      padded_rank > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "LoRA bake padded rank overflows");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_lora_bake_checked_byte_length(
      rank, target->input_size, sizeof(uint16_t), out_source_length));
  return id4_ideogram4_lora_bake_checked_byte_length(
      (uint32_t)padded_rank, target->input_size, sizeof(uint16_t),
      out_packed_length);
}

static iree_status_t id4_ideogram4_lora_bake_submit_target(
    const id4_ideogram4_lora_bake_submit_context_t* context,
    const id4_ideogram4_lora_bake_specialization_list_t* specializations,
    const id4_ideogram4_lora_bake_target_t* target,
    const id4_ideogram4_dit_lora_target_t* source_target,
    iree_hal_buffer_t* working_buffer, iree_hal_buffer_t* target_buffer,
    iree_hal_semaphore_list_t domain_ready_list, bool* inout_domain_joined,
    id4_ideogram4_lora_bake_chain_t* work_chain) {
  iree_device_size_t packed_down_offset = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < source_target->segment_count && iree_status_is_ok(status); ++i) {
    const id4_ideogram4_dit_lora_segment_t* segment =
        &source_target->segments[i];
    iree_device_size_t source_length = 0;
    iree_device_size_t packed_length = 0;
    status = id4_ideogram4_lora_bake_segment_lengths(
        target, segment->rank, &source_length, &packed_length);
    if (iree_status_is_ok(status)) {
      status = id4_ideogram4_lora_bake_submit_parameter_read(
          context, segment->source_scope, segment->down_parameter_key,
          /*source_offset=*/0, working_buffer,
          target->working_set.down_source.offset, source_length, work_chain);
    }
    if (iree_status_is_ok(status)) {
      const id4_ideogram4_lora_bake_specialization_t* specialization =
          id4_ideogram4_lora_bake_specialization_find(
              specializations, &id4_ideogram4_lora_bake_kernels.pack_down,
              ID4_IDEOGRAM4_LORA_BAKE_TILE_SIZE, target->input_size,
              segment->rank);
      const iree_hal_buffer_ref_t bindings[] = {
          iree_hal_make_buffer_ref(working_buffer,
                                   target->working_set.down_source.offset,
                                   source_length),
          iree_hal_make_buffer_ref(
              working_buffer,
              target->working_set.down.offset + packed_down_offset,
              packed_length),
      };
      status = id4_ideogram4_lora_bake_submit_dispatch(
          context, specialization,
          id4_ideogram4_lora_bake_chain_current(work_chain),
          (iree_hal_buffer_ref_list_t){IREE_ARRAYSIZE(bindings), bindings},
          work_chain);
    }
    if (iree_status_is_ok(status) &&
        !iree_device_size_checked_add(packed_down_offset, packed_length,
                                      &packed_down_offset)) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "LoRA bake packed Down offset overflows");
    }
  }
  if (iree_status_is_ok(status) &&
      packed_down_offset != target->working_set.down.length) {
    status = iree_make_status(
        IREE_STATUS_INTERNAL,
        "LoRA bake packed Down layout does not match its plan");
  }

  for (uint32_t row_offset = 0;
       row_offset < target->output_size && iree_status_is_ok(status);
       row_offset += target->output_rows_per_window) {
    const uint32_t row_count = iree_min(target->output_rows_per_window,
                                        target->output_size - row_offset);
    iree_device_size_t weight_offset = 0;
    iree_device_size_t weight_length = 0;
    iree_device_size_t scale_offset = 0;
    iree_device_size_t scale_length = 0;
    iree_device_size_t effective_length = 0;
    status = id4_ideogram4_lora_bake_weight_window(
        target, row_offset, row_count, &weight_offset, &weight_length,
        &scale_offset, &scale_length);
    if (iree_status_is_ok(status)) {
      status = id4_ideogram4_lora_bake_checked_byte_length(
          row_count, target->input_size, sizeof(float), &effective_length);
    }
    if (iree_status_is_ok(status)) {
      const id4_ideogram4_lora_bake_specialization_t* specialization =
          id4_ideogram4_lora_bake_specialization_find(
              specializations,
              id4_ideogram4_lora_bake_decode_kernel(target->weight_layout),
              row_count, target->input_size, /*rank=*/1);
      const iree_hal_buffer_ref_t bindings[] = {
          iree_hal_make_buffer_ref(target_buffer, weight_offset, weight_length),
          iree_hal_make_buffer_ref(target_buffer, scale_offset, scale_length),
          iree_hal_make_buffer_ref(working_buffer,
                                   target->working_set.effective_weight.offset,
                                   effective_length),
      };
      iree_hal_semaphore_list_t decode_wait_list =
          id4_ideogram4_lora_bake_chain_current(work_chain);
      iree_hal_semaphore_t* join_semaphores[2];
      uint64_t join_payload_values[2];
      if (!*inout_domain_joined) {
        decode_wait_list = id4_ideogram4_lora_bake_join_two_edges(
            domain_ready_list, decode_wait_list, join_semaphores,
            join_payload_values);
      }
      status = id4_ideogram4_lora_bake_submit_dispatch(
          context, specialization, decode_wait_list,
          (iree_hal_buffer_ref_list_t){IREE_ARRAYSIZE(bindings), bindings},
          work_chain);
      if (iree_status_is_ok(status)) *inout_domain_joined = true;
    }

    packed_down_offset = 0;
    for (iree_host_size_t i = 0;
         i < source_target->segment_count && iree_status_is_ok(status); ++i) {
      const id4_ideogram4_dit_lora_segment_t* segment =
          &source_target->segments[i];
      iree_device_size_t source_length = 0;
      iree_device_size_t packed_length = 0;
      iree_device_size_t up_length = 0;
      iree_device_size_t up_source_offset = 0;
      status = id4_ideogram4_lora_bake_segment_lengths(
          target, segment->rank, &source_length, &packed_length);
      if (iree_status_is_ok(status)) {
        status = id4_ideogram4_lora_bake_checked_byte_length(
            row_count, segment->rank, sizeof(uint16_t), &up_length);
      }
      if (iree_status_is_ok(status) &&
          (!iree_device_size_checked_mul(row_offset, segment->rank,
                                         &up_source_offset) ||
           !iree_device_size_checked_mul(up_source_offset, sizeof(uint16_t),
                                         &up_source_offset))) {
        status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "LoRA bake Up source offset overflows");
      }
      if (iree_status_is_ok(status)) {
        status = id4_ideogram4_lora_bake_submit_parameter_read(
            context, segment->source_scope, segment->up_parameter_key,
            up_source_offset, working_buffer, target->working_set.up.offset,
            up_length, work_chain);
      }
      if (iree_status_is_ok(status)) {
        const id4_ideogram4_lora_bake_kernel_t* update_kernel =
            segment->rank % ID4_IDEOGRAM4_LORA_BAKE_TILE_SIZE == 0
                ? &id4_ideogram4_lora_bake_kernels.update_wmma
                : &id4_ideogram4_lora_bake_kernels.update;
        const id4_ideogram4_lora_bake_specialization_t* specialization =
            id4_ideogram4_lora_bake_specialization_find(
                specializations, update_kernel, row_count, target->input_size,
                segment->rank);
        iree_device_size_t strength_offset = 0;
        if (!iree_device_size_checked_mul(segment->adapter_ordinal,
                                          sizeof(float), &strength_offset) ||
            !iree_device_size_checked_add(target->working_set.strengths.offset,
                                          strength_offset, &strength_offset)) {
          status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                    "LoRA bake strength offset overflows");
        } else {
          const iree_hal_buffer_ref_t bindings[] = {
              iree_hal_make_buffer_ref(
                  working_buffer,
                  target->working_set.down.offset + packed_down_offset,
                  packed_length),
              iree_hal_make_buffer_ref(
                  working_buffer, target->working_set.up.offset, up_length),
              iree_hal_make_buffer_ref(working_buffer, strength_offset,
                                       sizeof(float)),
              iree_hal_make_buffer_ref(
                  working_buffer, target->working_set.effective_weight.offset,
                  effective_length),
          };
          status = id4_ideogram4_lora_bake_submit_dispatch(
              context, specialization,
              id4_ideogram4_lora_bake_chain_current(work_chain),
              (iree_hal_buffer_ref_list_t){IREE_ARRAYSIZE(bindings), bindings},
              work_chain);
        }
      }
      if (iree_status_is_ok(status) &&
          !iree_device_size_checked_add(packed_down_offset, packed_length,
                                        &packed_down_offset)) {
        status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "LoRA bake packed Down offset overflows");
      }
    }

    if (iree_status_is_ok(status)) {
      const id4_ideogram4_lora_bake_specialization_t* specialization =
          id4_ideogram4_lora_bake_specialization_find(
              specializations, &id4_ideogram4_lora_bake_kernels.scale,
              row_count, target->input_size, /*rank=*/1);
      const iree_hal_buffer_ref_t bindings[] = {
          iree_hal_make_buffer_ref(working_buffer,
                                   target->working_set.effective_weight.offset,
                                   effective_length),
          iree_hal_make_buffer_ref(target_buffer, scale_offset, scale_length),
      };
      status = id4_ideogram4_lora_bake_submit_dispatch(
          context, specialization,
          id4_ideogram4_lora_bake_chain_current(work_chain),
          (iree_hal_buffer_ref_list_t){IREE_ARRAYSIZE(bindings), bindings},
          work_chain);
    }
    if (iree_status_is_ok(status)) {
      const id4_ideogram4_lora_bake_specialization_t* specialization =
          id4_ideogram4_lora_bake_specialization_find(
              specializations,
              id4_ideogram4_lora_bake_encode_kernel(target->weight_layout),
              row_count, target->input_size, /*rank=*/1);
      const iree_hal_buffer_ref_t bindings[] = {
          iree_hal_make_buffer_ref(working_buffer,
                                   target->working_set.effective_weight.offset,
                                   effective_length),
          iree_hal_make_buffer_ref(target_buffer, scale_offset, scale_length),
          iree_hal_make_buffer_ref(target_buffer, weight_offset, weight_length),
      };
      status = id4_ideogram4_lora_bake_submit_dispatch(
          context, specialization,
          id4_ideogram4_lora_bake_chain_current(work_chain),
          (iree_hal_buffer_ref_list_t){IREE_ARRAYSIZE(bindings), bindings},
          work_chain);
    }
  }
  return status;
}

static iree_status_t id4_ideogram4_lora_bake_wait_for_list(
    iree_hal_semaphore_list_t list) {
  if (list.count == 0) return iree_ok_status();
  return iree_hal_semaphore_list_wait(list, iree_infinite_timeout(),
                                      IREE_ASYNC_WAIT_FLAG_NONE);
}

static iree_status_t id4_ideogram4_lora_bake_cleanup_failed_submission(
    id4_ideogram4_lora_bake_chain_t* domain_chain,
    id4_ideogram4_lora_bake_chain_t* work_chain,
    iree_hal_buffer_t* working_buffer,
    id4_pipeline_parameter_materialization_t* materialization,
    bool* out_materialization_aborted) {
  *out_materialization_aborted = false;
  if (!working_buffer && !materialization) return iree_ok_status();

  iree_hal_semaphore_list_t domain_edge =
      id4_ideogram4_lora_bake_chain_current(domain_chain);
  iree_hal_semaphore_list_t work_edge =
      id4_ideogram4_lora_bake_chain_current(work_chain);
  iree_hal_semaphore_list_t terminal_edge = domain_edge;
  iree_hal_semaphore_t* join_semaphores[2];
  uint64_t join_payload_values[2];
  if (domain_edge.count != 0 && work_edge.count != 0) {
    terminal_edge = id4_ideogram4_lora_bake_join_two_edges(
        domain_edge, work_edge, join_semaphores, join_payload_values);
  } else if (work_edge.count != 0) {
    terminal_edge = work_edge;
  }
  if (terminal_edge.count == 0) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "LoRA bake owns failed-submission storage without a terminal edge");
  }

  if (materialization) {
    id4_pipeline_diagnostics_sink_t ignore_sink;
    id4_pipeline_diagnostics_sink_initialize_ignore(&ignore_sink);
    iree_status_t status = id4_pipeline_parameter_materialization_abort(
        materialization, terminal_edge, &ignore_sink);
    *out_materialization_aborted = true;
    return status;
  }
  return id4_ideogram4_lora_bake_wait_for_list(terminal_edge);
}

iree_status_t id4_ideogram4_lora_bake_submit(
    const id4_ideogram4_lora_bake_submit_options_t* options,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_materialization_t** out_materialization) {
  IREE_ASSERT_ARGUMENT(out_materialization);
  *out_materialization = NULL;

  id4_ideogram4_lora_bake_submit_context_t context;
  iree_status_t status =
      id4_ideogram4_lora_bake_validate_submit_options(options, &context);
  if (!iree_status_is_ok(status)) {
    id4_ideogram4_lora_bake_fail_caller_signal(options,
                                               iree_status_code(status));
    return status;
  }

  id4_ideogram4_lora_bake_chain_t domain_chain;
  id4_ideogram4_lora_bake_chain_t work_chain;
  memset(&domain_chain, 0, sizeof(domain_chain));
  memset(&work_chain, 0, sizeof(work_chain));
  id4_pipeline_parameter_materialization_t* materialization = NULL;
  id4_pipeline_parameter_materialization_target_t materialization_target;
  memset(&materialization_target, 0, sizeof(materialization_target));
  iree_hal_buffer_t* working_buffer = NULL;

  status = id4_ideogram4_lora_bake_chain_initialize(
      context.prepared->planned_device,
      context.prepared->planned_placement->queue_affinity, &domain_chain);

  uint32_t next_index = 0;
  iree_hal_semaphore_list_t next_signal_list;
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_lora_bake_chain_next(&domain_chain, &next_index,
                                                &next_signal_list);
  }
  if (iree_status_is_ok(status)) {
    const id4_pipeline_parameter_materialization_acquire_options_t
        acquire_options = {
            .structure_size = sizeof(acquire_options),
            .plan = context.prepared->base_plan,
            .target_slab_index =
                id4_ideogram4_lora_bake_plan_patchable_slab_index(
                    context.prepared->plan),
            .allocation_pool = options->allocation_pool,
            .alloca_flags = options->materialization_alloca_flags,
            .wait_semaphore_list = options->wait_semaphore_list,
            .signal_semaphore_list = next_signal_list,
            .diagnostics_sink = options->diagnostics_sink,
        };
    status = id4_pipeline_parameter_materialization_acquire(
        &acquire_options, host_allocator, &materialization);
    if (iree_status_is_ok(status)) {
      id4_ideogram4_lora_bake_chain_commit(&domain_chain, next_index);
    }
  }

  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_materialization_query_target(
        materialization, &materialization_target);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_lora_bake_query_buffer_placement(
        materialization_target.target_buffer, IREE_SV("LoRA patchable domain"),
        &context.domain_placement);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_lora_bake_chain_initialize(
        context.domain_placement.device,
        context.domain_placement.queue_affinity, &work_chain);
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_lora_bake_chain_next(&work_chain, &next_index,
                                                &next_signal_list);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_params_t working_params = {0};
    working_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    working_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    working_params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET |
                           IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
    working_params.queue_affinity = context.domain_placement.queue_affinity;
    working_params.min_alignment = ID4_IDEOGRAM4_LORA_BAKE_TILE_SIZE;
    status = iree_hal_device_queue_alloca(
        context.domain_placement.device,
        context.domain_placement.queue_affinity, options->wait_semaphore_list,
        next_signal_list, options->allocation_pool, working_params,
        id4_ideogram4_lora_bake_plan_working_set_high_water_mark(
            context.prepared->plan),
        options->working_set_alloca_flags, &working_buffer);
    if (iree_status_is_ok(status)) {
      id4_ideogram4_lora_bake_chain_commit(&work_chain, next_index);
    }
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_lora_bake_query_buffer_placement(
        working_buffer, IREE_SV("LoRA bake working storage"),
        &context.working_placement);
  }
  if (iree_status_is_ok(status) &&
      (context.working_placement.device != context.domain_placement.device ||
       !iree_all_bits_set(context.working_placement.queue_affinity,
                          context.domain_placement.queue_affinity))) {
    status = iree_make_status(
        IREE_STATUS_INTERNAL,
        "LoRA bake working storage is not available to the domain queue");
  }
  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_lora_bake_chain_next(&domain_chain, &next_index,
                                                &next_signal_list);
  }
  if (iree_status_is_ok(status)) {
    const id4_pipeline_parameter_source_t* base_parameter_source =
        &options->base_parameter_source;
    const id4_pipeline_parameter_layout_load_options_t load_options = {
        .structure_size = sizeof(load_options),
        .index = base_parameter_source->storage.execution_layout.index,
        .provider = base_parameter_source->storage.execution_layout.provider,
        .scope = base_parameter_source->storage.execution_layout.scope,
        .wait_semaphore_list = materialization_target.readiness_semaphore_list,
        .signal_semaphore_list = next_signal_list,
        .diagnostics_sink = options->diagnostics_sink,
    };
    status = id4_pipeline_parameter_layout_gather_slab(
        context.prepared->base_plan, &load_options,
        materialization_target.slab_index, materialization_target.target_buffer,
        host_allocator);
    if (iree_status_is_ok(status)) {
      id4_ideogram4_lora_bake_chain_commit(&domain_chain, next_index);
    }
  }

  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_lora_bake_chain_next(&work_chain, &next_index,
                                                &next_signal_list);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_update(
        context.working_placement.device,
        context.working_placement.queue_affinity,
        id4_ideogram4_lora_bake_chain_current(&work_chain), next_signal_list,
        options->strength_values, /*source_offset=*/0, working_buffer,
        context.prepared->strength_range.offset,
        context.prepared->strength_range.length, IREE_HAL_UPDATE_FLAG_NONE);
    if (iree_status_is_ok(status)) {
      id4_ideogram4_lora_bake_chain_commit(&work_chain, next_index);
    }
  }

  bool domain_joined = false;
  const iree_host_size_t target_count =
      id4_ideogram4_lora_bake_plan_target_count(context.prepared->plan);
  for (iree_host_size_t i = 0; i < target_count && iree_status_is_ok(status);
       ++i) {
    const id4_ideogram4_lora_bake_target_t* target =
        id4_ideogram4_lora_bake_plan_target_at(context.prepared->plan, i);
    const id4_ideogram4_dit_lora_target_t* source_target =
        id4_ideogram4_lora_topology_target_at(context.prepared->topology, i);
    status = id4_ideogram4_lora_bake_submit_target(
        &context, &context.prepared->specializations, target, source_target,
        working_buffer, materialization_target.target_buffer,
        id4_ideogram4_lora_bake_chain_current(&domain_chain), &domain_joined,
        &work_chain);
  }

  if (iree_status_is_ok(status)) {
    status = id4_ideogram4_lora_bake_chain_next(&work_chain, &next_index,
                                                &next_signal_list);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_queue_dealloca(
        context.working_placement.device,
        context.working_placement.queue_affinity,
        id4_ideogram4_lora_bake_chain_current(&work_chain), next_signal_list,
        working_buffer, options->working_set_dealloca_flags);
    if (iree_status_is_ok(status)) {
      id4_ideogram4_lora_bake_chain_commit(&work_chain, next_index);
    }
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_materialization_publish(
        materialization, id4_ideogram4_lora_bake_chain_current(&work_chain),
        options->signal_semaphore_list, options->diagnostics_sink);
  }

  if (iree_status_is_ok(status)) {
    *out_materialization = materialization;
    materialization = NULL;
  } else {
    bool materialization_aborted = false;
    iree_status_t cleanup_status =
        id4_ideogram4_lora_bake_cleanup_failed_submission(
            &domain_chain, &work_chain, working_buffer, materialization,
            &materialization_aborted);
    status = iree_status_join(status, cleanup_status);
    if (materialization_aborted) {
      id4_pipeline_parameter_materialization_release(materialization);
      materialization = NULL;
    }
    id4_ideogram4_lora_bake_fail_caller_signal(options,
                                               iree_status_code(status));
  }

  iree_hal_buffer_release(working_buffer);
  id4_ideogram4_lora_bake_chain_deinitialize(&work_chain);
  id4_ideogram4_lora_bake_chain_deinitialize(&domain_chain);
  return status;
}
