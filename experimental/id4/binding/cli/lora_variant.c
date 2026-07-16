// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/binding/cli/lora_variant.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "experimental/id4/ideogram4/lora_bake.h"
#include "experimental/id4/ideogram4/lora_bake_plan.h"
#include "experimental/id4/pipeline/parameter_layout.h"
#include "experimental/id4/pipeline/parameter_materialization.h"
#include "experimental/id4/pipeline/parameter_materialization_group.h"

typedef struct id4_cli_lora_variant_edge_t {
  // One-shot timeline semaphore owned while the stack edge is in scope.
  iree_hal_semaphore_t* semaphore;
  // Single payload value signaled on |semaphore|.
  uint64_t payload_value;
} id4_cli_lora_variant_edge_t;

struct id4_cli_lora_variant_t {
  // Host allocator owning the variant.
  iree_allocator_t host_allocator;
  // Complete parameter-domain owner and lifecycle state.
  id4_pipeline_parameter_materialization_group_t* materialization_group;
};

static iree_status_t id4_cli_lora_variant_validate_options(
    const id4_cli_lora_variant_create_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LoRA variant options are required");
  }
  if (options->structure_size < sizeof(*options)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LoRA variant options are too small");
  }
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "LoRA variant extension structures are not supported");
  }
  if (!options->conditioned_dit_plan || !options->lora_set ||
      !options->kernel_cache || !options->executable_cache ||
      !options->kernel_library) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LoRA variant plan, adapters, kernel caches, and kernel library are "
        "required");
  }
  if (id4_cli_lora_set_adapter_count(options->lora_set) == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LoRA variant requires at least one adapter");
  }
  if (options->working_set_byte_capacity == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LoRA variant working-set capacity is required");
  }
  const id4_pipeline_parameter_source_t* base_source =
      &options->base_parameter_source;
  if (base_source->kind !=
          ID4_PIPELINE_PARAMETER_SOURCE_KIND_EXECUTION_LAYOUT ||
      !base_source->storage.execution_layout.index ||
      !base_source->storage.execution_layout.provider ||
      iree_string_view_is_empty(base_source->storage.execution_layout.scope) ||
      !iree_io_parameter_provider_query_support(
          base_source->storage.execution_layout.provider,
          base_source->storage.execution_layout.scope)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "LoRA variant requires a complete execution-layout base source");
  }
  return id4_pipeline_diagnostics_validate_sink(options->diagnostics_sink,
                                                IREE_SV("LoRA variant"));
}

static iree_hal_semaphore_list_t id4_cli_lora_variant_edge_list(
    id4_cli_lora_variant_edge_t* edge) {
  return edge && edge->semaphore
             ? (iree_hal_semaphore_list_t){
                   .count = 1,
                   .semaphores = &edge->semaphore,
                   .payload_values = &edge->payload_value,
               }
             : iree_hal_semaphore_list_empty();
}

static iree_status_t id4_cli_lora_variant_edge_initialize(
    iree_hal_buffer_placement_t placement,
    id4_cli_lora_variant_edge_t* out_edge) {
  memset(out_edge, 0, sizeof(*out_edge));
  out_edge->payload_value = 1;
  return iree_hal_semaphore_create(
      placement.device, placement.queue_affinity, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &out_edge->semaphore);
}

static void id4_cli_lora_variant_edge_deinitialize(
    id4_cli_lora_variant_edge_t* edge) {
  if (!edge) return;
  iree_hal_semaphore_release(edge->semaphore);
  memset(edge, 0, sizeof(*edge));
}

static iree_status_t id4_cli_lora_variant_query_planned_placement(
    const id4_pipeline_plan_t* plan, iree_host_size_t slab_index,
    iree_hal_buffer_placement_t* out_placement) {
  *out_placement = iree_hal_buffer_placement_undefined();
  id4_pipeline_parameter_slab_load_t load;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_plan_parameter_slab_load_at(plan, slab_index, &load));
  if (!load.device || iree_hal_queue_affinity_is_empty(load.queue_affinity)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LoRA variant slab %" PRIhsz
                            " has no planned device placement",
                            slab_index);
  }
  out_placement->device = load.device;
  out_placement->queue_affinity = load.queue_affinity;
  return iree_ok_status();
}

static iree_status_t id4_cli_lora_variant_query_buffer_placement(
    iree_hal_buffer_t* buffer, iree_hal_buffer_placement_t* out_placement) {
  *out_placement = iree_hal_buffer_placement_undefined();
  const iree_hal_buffer_placement_t placement =
      iree_hal_buffer_allocation_placement(buffer);
  if (iree_hal_buffer_placement_is_undefined(placement) ||
      iree_hal_queue_affinity_is_empty(placement.queue_affinity)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "queue-allocated LoRA variant slab has no device placement");
  }
  *out_placement = placement;
  return iree_ok_status();
}

static iree_status_t id4_cli_lora_variant_create_unmodified_materialization(
    const id4_cli_lora_variant_create_options_t* options,
    iree_host_size_t slab_index, iree_allocator_t host_allocator,
    id4_pipeline_parameter_materialization_t** out_materialization) {
  *out_materialization = NULL;
  id4_cli_lora_variant_edge_t acquisition_edge;
  id4_cli_lora_variant_edge_t gather_edge;
  id4_cli_lora_variant_edge_t publication_edge;
  memset(&acquisition_edge, 0, sizeof(acquisition_edge));
  memset(&gather_edge, 0, sizeof(gather_edge));
  memset(&publication_edge, 0, sizeof(publication_edge));
  id4_pipeline_parameter_materialization_t* materialization = NULL;
  iree_hal_semaphore_list_t terminal_wait_list =
      iree_hal_semaphore_list_empty();

  iree_hal_buffer_placement_t planned_placement =
      iree_hal_buffer_placement_undefined();
  iree_status_t status = id4_cli_lora_variant_query_planned_placement(
      options->conditioned_dit_plan, slab_index, &planned_placement);
  if (iree_status_is_ok(status)) {
    status = id4_cli_lora_variant_edge_initialize(planned_placement,
                                                  &acquisition_edge);
  }
  if (iree_status_is_ok(status)) {
    const id4_pipeline_parameter_materialization_acquire_options_t
        acquire_options = {
            .structure_size = sizeof(acquire_options),
            .plan = options->conditioned_dit_plan,
            .target_slab_index = slab_index,
            .alloca_flags = IREE_HAL_ALLOCA_FLAG_NONE,
            .wait_semaphore_list = iree_hal_semaphore_list_empty(),
            .signal_semaphore_list =
                id4_cli_lora_variant_edge_list(&acquisition_edge),
            .diagnostics_sink = options->diagnostics_sink,
        };
    status = id4_pipeline_parameter_materialization_acquire(
        &acquire_options, host_allocator, &materialization);
    if (iree_status_is_ok(status)) {
      terminal_wait_list = id4_cli_lora_variant_edge_list(&acquisition_edge);
    }
  }

  id4_pipeline_parameter_materialization_target_t target;
  memset(&target, 0, sizeof(target));
  iree_hal_buffer_placement_t placement = iree_hal_buffer_placement_undefined();
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_materialization_query_target(
        materialization, &target);
  }
  if (iree_status_is_ok(status)) {
    status = id4_cli_lora_variant_query_buffer_placement(target.target_buffer,
                                                         &placement);
  }
  if (iree_status_is_ok(status)) {
    status = id4_cli_lora_variant_edge_initialize(placement, &gather_edge);
  }
  if (iree_status_is_ok(status)) {
    const id4_pipeline_parameter_source_t* base_source =
        &options->base_parameter_source;
    const id4_pipeline_parameter_layout_load_options_t load_options = {
        .structure_size = sizeof(load_options),
        .index = base_source->storage.execution_layout.index,
        .provider = base_source->storage.execution_layout.provider,
        .scope = base_source->storage.execution_layout.scope,
        .wait_semaphore_list = target.readiness_semaphore_list,
        .signal_semaphore_list = id4_cli_lora_variant_edge_list(&gather_edge),
        .diagnostics_sink = options->diagnostics_sink,
    };
    status = id4_pipeline_parameter_layout_gather_slab(
        options->conditioned_dit_plan, &load_options, target.slab_index,
        target.target_buffer, host_allocator);
    if (iree_status_is_ok(status)) {
      terminal_wait_list = id4_cli_lora_variant_edge_list(&gather_edge);
    }
  }

  if (iree_status_is_ok(status)) {
    status = id4_cli_lora_variant_edge_initialize(placement, &publication_edge);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_materialization_publish(
        materialization, terminal_wait_list,
        id4_cli_lora_variant_edge_list(&publication_edge),
        options->diagnostics_sink);
  }

  if (iree_status_is_ok(status)) {
    *out_materialization = materialization;
    materialization = NULL;
  } else if (materialization) {
    iree_status_t abort_status = id4_pipeline_parameter_materialization_abort(
        materialization, terminal_wait_list, options->diagnostics_sink);
    const bool abort_succeeded = iree_status_is_ok(abort_status);
    status = iree_status_join(status, abort_status);
    if (abort_succeeded) {
      id4_pipeline_parameter_materialization_release(materialization);
      materialization = NULL;
    }
  }

  id4_cli_lora_variant_edge_deinitialize(&publication_edge);
  id4_cli_lora_variant_edge_deinitialize(&gather_edge);
  id4_cli_lora_variant_edge_deinitialize(&acquisition_edge);
  return status;
}

static iree_status_t id4_cli_lora_variant_adopt_materialization(
    id4_pipeline_parameter_materialization_group_t* group,
    id4_pipeline_parameter_materialization_t** inout_materialization,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  iree_status_t status = id4_pipeline_parameter_materialization_group_adopt(
      group, inout_materialization);
  if (!iree_status_is_ok(status) && inout_materialization &&
      *inout_materialization) {
    id4_pipeline_parameter_materialization_binding_t published;
    iree_status_t cleanup_status =
        id4_pipeline_parameter_materialization_query_binding(
            *inout_materialization, &published);
    if (iree_status_is_ok(cleanup_status)) {
      cleanup_status = id4_pipeline_parameter_materialization_retire_and_wait(
          *inout_materialization, published.readiness_semaphore_list,
          IREE_HAL_DEALLOCA_FLAG_NONE, diagnostics_sink);
    }
    if (iree_status_is_ok(cleanup_status)) {
      id4_pipeline_parameter_materialization_release(*inout_materialization);
      *inout_materialization = NULL;
    }
    status = iree_status_join(status, cleanup_status);
  }
  return status;
}

static void id4_cli_lora_variant_free(id4_cli_lora_variant_t* variant) {
  if (!variant) return;
  const iree_allocator_t host_allocator = variant->host_allocator;
  id4_pipeline_parameter_materialization_group_release(
      variant->materialization_group);
  iree_allocator_free(host_allocator, variant);
}

iree_status_t id4_cli_lora_variant_create(
    const id4_cli_lora_variant_create_options_t* options,
    iree_allocator_t host_allocator, id4_cli_lora_variant_t** out_variant) {
  IREE_ASSERT_ARGUMENT(out_variant);
  *out_variant = NULL;
  IREE_RETURN_IF_ERROR(id4_cli_lora_variant_validate_options(options));

  id4_cli_lora_variant_t* variant = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*variant), (void**)&variant);
  if (iree_status_is_ok(status)) {
    memset(variant, 0, sizeof(*variant));
    variant->host_allocator = host_allocator;
    status = id4_pipeline_parameter_materialization_group_create(
        options->conditioned_dit_plan, host_allocator,
        &variant->materialization_group);
  }

  id4_ideogram4_lora_bake_plan_t* bake_plan = NULL;
  if (iree_status_is_ok(status)) {
    id4_ideogram4_lora_bake_plan_create_options_t plan_options;
    memset(&plan_options, 0, sizeof(plan_options));
    plan_options.structure_size = sizeof(plan_options);
    plan_options.base_plan = options->conditioned_dit_plan;
    plan_options.topology = id4_cli_lora_set_topology(options->lora_set);
    plan_options.working_set_byte_capacity = options->working_set_byte_capacity;
    status = id4_ideogram4_lora_bake_plan_create(&plan_options, host_allocator,
                                                 &bake_plan);
  }

  id4_ideogram4_lora_bake_prepared_t* prepared_bake = NULL;
  if (iree_status_is_ok(status)) {
    id4_ideogram4_lora_bake_prepare_options_t prepare_options;
    memset(&prepare_options, 0, sizeof(prepare_options));
    prepare_options.structure_size = sizeof(prepare_options);
    prepare_options.plan = bake_plan;
    prepare_options.kernel_library = options->kernel_library;
    prepare_options.kernel_cache = options->kernel_cache;
    prepare_options.executable_cache = options->executable_cache;
    prepare_options.executable_caching_mode =
        IREE_HAL_EXECUTABLE_CACHING_MODE_NONE;
    prepare_options.diagnostics_sink = options->diagnostics_sink;
    status = id4_ideogram4_lora_bake_prepare(&prepare_options, host_allocator,
                                             &prepared_bake);
  }

  iree_io_parameter_provider_t* adapter_provider = NULL;
  if (iree_status_is_ok(status)) {
    const id4_pipeline_parameter_source_t* base_source =
        &options->base_parameter_source;
    status = id4_cli_lora_set_create_conditioned_provider(
        options->lora_set, base_source->storage.execution_layout.scope,
        base_source->storage.execution_layout.provider, host_allocator,
        &adapter_provider);
  }

  const iree_host_size_t slab_count =
      id4_pipeline_plan_parameter_slab_count(options->conditioned_dit_plan);
  const iree_host_size_t patchable_slab_index =
      bake_plan ? id4_ideogram4_lora_bake_plan_patchable_slab_index(bake_plan)
                : IREE_HOST_SIZE_MAX;
  for (iree_host_size_t i = 0; i < slab_count && iree_status_is_ok(status);
       ++i) {
    if (i == patchable_slab_index) continue;
    id4_pipeline_parameter_materialization_t* materialization = NULL;
    status = id4_cli_lora_variant_create_unmodified_materialization(
        options, i, host_allocator, &materialization);
    if (iree_status_is_ok(status)) {
      status = id4_cli_lora_variant_adopt_materialization(
          variant->materialization_group, &materialization,
          options->diagnostics_sink);
    }
  }

  id4_cli_lora_variant_edge_t bake_publication_edge;
  memset(&bake_publication_edge, 0, sizeof(bake_publication_edge));
  iree_hal_buffer_placement_t planned_placement =
      iree_hal_buffer_placement_undefined();
  if (iree_status_is_ok(status)) {
    status = id4_cli_lora_variant_query_planned_placement(
        options->conditioned_dit_plan, patchable_slab_index,
        &planned_placement);
  }
  if (iree_status_is_ok(status)) {
    status = id4_cli_lora_variant_edge_initialize(planned_placement,
                                                  &bake_publication_edge);
  }
  id4_pipeline_parameter_materialization_t* patchable_materialization = NULL;
  if (iree_status_is_ok(status)) {
    id4_ideogram4_lora_bake_submit_options_t submit_options;
    memset(&submit_options, 0, sizeof(submit_options));
    submit_options.structure_size = sizeof(submit_options);
    submit_options.prepared = prepared_bake;
    submit_options.base_parameter_source = options->base_parameter_source;
    submit_options.adapter_provider = adapter_provider;
    submit_options.strength_count =
        id4_cli_lora_set_adapter_count(options->lora_set);
    submit_options.strength_values =
        id4_cli_lora_set_strengths(options->lora_set);
    submit_options.materialization_alloca_flags = IREE_HAL_ALLOCA_FLAG_NONE;
    submit_options.working_set_alloca_flags = IREE_HAL_ALLOCA_FLAG_NONE;
    submit_options.working_set_dealloca_flags = IREE_HAL_DEALLOCA_FLAG_NONE;
    submit_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
    submit_options.signal_semaphore_list =
        id4_cli_lora_variant_edge_list(&bake_publication_edge);
    submit_options.diagnostics_sink = options->diagnostics_sink;
    status = id4_ideogram4_lora_bake_submit(&submit_options, host_allocator,
                                            &patchable_materialization);
  }
  if (iree_status_is_ok(status)) {
    status = id4_cli_lora_variant_adopt_materialization(
        variant->materialization_group, &patchable_materialization,
        options->diagnostics_sink);
  }
  id4_cli_lora_variant_edge_deinitialize(&bake_publication_edge);

  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_materialization_group_finalize(
        variant->materialization_group);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_materialization_group_wait_ready(
        variant->materialization_group);
  }
  if (iree_status_is_ok(status)) {
    fprintf(
        stdout,
        "Baked LoRA variant: adapters=%" PRIhsz " targets=%" PRIhsz
        " patchable=%" PRIu64 "MiB working_set=%" PRIu64 "MiB\n",
        id4_cli_lora_set_adapter_count(options->lora_set),
        id4_ideogram4_lora_bake_plan_target_count(bake_plan),
        (uint64_t)((id4_ideogram4_lora_bake_plan_patchable_slab_byte_length(
                        bake_plan) +
                    (1024 * 1024 - 1)) /
                   (1024 * 1024)),
        (uint64_t)((id4_ideogram4_lora_bake_plan_working_set_high_water_mark(
                        bake_plan) +
                    (1024 * 1024 - 1)) /
                   (1024 * 1024)));
  }

  iree_io_parameter_provider_release(adapter_provider);
  id4_ideogram4_lora_bake_prepared_release(prepared_bake);
  id4_ideogram4_lora_bake_plan_release(bake_plan);
  if (iree_status_is_ok(status)) {
    *out_variant = variant;
  } else if (variant && variant->materialization_group) {
    iree_status_t retire_status =
        id4_pipeline_parameter_materialization_group_retire(
            variant->materialization_group, iree_hal_semaphore_list_empty(),
            options->diagnostics_sink);
    const bool retire_succeeded = iree_status_is_ok(retire_status);
    status = iree_status_join(status, retire_status);
    if (retire_succeeded) {
      id4_cli_lora_variant_free(variant);
    }
  } else {
    iree_allocator_free(host_allocator, variant);
  }
  return status;
}

id4_pipeline_parameter_slab_set_t* id4_cli_lora_variant_parameter_slabs(
    const id4_cli_lora_variant_t* variant) {
  return variant ? id4_pipeline_parameter_materialization_group_parameter_slabs(
                       variant->materialization_group)
                 : NULL;
}

iree_status_t id4_cli_lora_variant_retire(
    id4_cli_lora_variant_t* variant,
    iree_hal_semaphore_list_t last_use_wait_list,
    id4_pipeline_diagnostics_sink_t* diagnostics_sink) {
  if (!variant) return iree_ok_status();
  return id4_pipeline_parameter_materialization_group_retire(
      variant->materialization_group, last_use_wait_list, diagnostics_sink);
}

void id4_cli_lora_variant_release(id4_cli_lora_variant_t* variant) {
  id4_cli_lora_variant_free(variant);
}
