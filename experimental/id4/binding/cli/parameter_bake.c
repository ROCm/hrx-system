// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/binding/cli/parameter_bake.h"

#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "experimental/id4/pipeline/parameter_layout.h"
#include "experimental/id4/pipeline/parameter_slab.h"
#include "experimental/id4/pipeline/plan.h"
#include "experimental/id4/stages/vae_parameters.h"
#include "experimental/id4/tooling/filesystem.h"
#include "iree/io/file_handle.h"
#include "iree/io/formats/irpa/irpa_builder.h"
#include "iree/io/parameter_index.h"
#include "iree/io/parameter_index_provider.h"
#include "iree/io/stream.h"

typedef iree_status_t(
    IREE_API_PTR* id4_cli_parameter_bake_provider_create_fn_t)(
    const id4_cli_parameter_bake_options_t* options,
    const id4_pipeline_plan_t* plan,
    iree_io_parameter_provider_t* source_provider,
    iree_allocator_t host_allocator,
    iree_io_parameter_provider_t** out_provider);

typedef struct id4_cli_parameter_bake_stage_descriptor_t {
  // Generation-plan stage key selecting the coarse stage plan.
  iree_string_view_t stage_key;
  // Stable provider scope assigned to the baked archive.
  iree_string_view_t archive_scope;
  // Final archive file name relative to the output directory.
  iree_string_view_t file_name;
  // Temporary archive file name relative to the output directory.
  iree_string_view_t partial_file_name;
  // Byte offset of the checkpoint source in the generation source catalog.
  iree_host_size_t source_offset;
  // Coarse stage mask bit selecting this archive.
  id4_ideogram4_generation_resident_stage_mask_t stage_mask;
  // Creates the plan-visible provider used to materialize this stage.
  id4_cli_parameter_bake_provider_create_fn_t provider_create;
} id4_cli_parameter_bake_stage_descriptor_t;

static iree_status_t id4_cli_parameter_bake_retain_source_provider(
    const id4_cli_parameter_bake_options_t* options,
    const id4_pipeline_plan_t* plan,
    iree_io_parameter_provider_t* source_provider,
    iree_allocator_t host_allocator,
    iree_io_parameter_provider_t** out_provider) {
  (void)options;
  (void)plan;
  (void)host_allocator;
  iree_io_parameter_provider_retain(source_provider);
  *out_provider = source_provider;
  return iree_ok_status();
}

static iree_status_t id4_cli_parameter_bake_create_vae_provider(
    const id4_cli_parameter_bake_options_t* options,
    const id4_pipeline_plan_t* plan,
    iree_io_parameter_provider_t* source_provider,
    iree_allocator_t host_allocator,
    iree_io_parameter_provider_t** out_provider) {
  id4_vae_parameter_provider_create_options_t provider_options;
  memset(&provider_options, 0, sizeof(provider_options));
  provider_options.structure_size = sizeof(provider_options);
  provider_options.source_provider = source_provider;
  provider_options.plan = plan;
  provider_options.kernel_library = options->kernel_library;
  provider_options.kernel_cache = options->runtime_context->kernel_cache;
  provider_options.executable_cache =
      options->runtime_context->executable_cache;
  provider_options.diagnostics_sink = options->diagnostics_sink;
  return id4_vae_parameter_provider_create(&provider_options, host_allocator,
                                           out_provider);
}

static const id4_cli_parameter_bake_stage_descriptor_t
    id4_cli_parameter_bake_stage_descriptors[] = {
        {
            IREE_SVL("qwen"),
            IREE_SVL("qwen"),
            IREE_SVL("qwen.irpa"),
            IREE_SVL("qwen.irpa.partial"),
            offsetof(id4_ideogram4_generation_parameter_sources_t, qwen),
            ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_QWEN,
            id4_cli_parameter_bake_retain_source_provider,
        },
        {
            IREE_SVL("dit_conditioned"),
            IREE_SVL("dit_conditioned"),
            IREE_SVL("dit_conditioned.irpa"),
            IREE_SVL("dit_conditioned.irpa.partial"),
            offsetof(id4_ideogram4_generation_parameter_sources_t,
                     dit_conditioned),
            ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_DIT_CONDITIONED,
            id4_cli_parameter_bake_retain_source_provider,
        },
        {
            IREE_SVL("dit_unconditioned"),
            IREE_SVL("dit_unconditioned"),
            IREE_SVL("dit_unconditioned.irpa"),
            IREE_SVL("dit_unconditioned.irpa.partial"),
            offsetof(id4_ideogram4_generation_parameter_sources_t,
                     dit_unconditioned),
            ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_DIT_UNCONDITIONED,
            id4_cli_parameter_bake_retain_source_provider,
        },
        {
            IREE_SVL("decode"),
            IREE_SVL("vae"),
            IREE_SVL("vae.irpa"),
            IREE_SVL("vae.irpa.partial"),
            offsetof(id4_ideogram4_generation_parameter_sources_t, vae),
            ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_DECODE,
            id4_cli_parameter_bake_create_vae_provider,
        },
};

typedef struct id4_cli_parameter_archive_t {
  // Final archive path published after successful population.
  iree_string_view_t final_path;
  // Temporary archive path populated before publication.
  iree_string_view_t partial_path;
  // Writable file backing the archive index.
  iree_io_file_handle_t* file_handle;
  // Writable parameter index referencing |file_handle|.
  iree_io_parameter_index_t* index;
  // Provider exposing |index| under the descriptor archive scope.
  iree_io_parameter_provider_t* provider;
} id4_cli_parameter_archive_t;

static const id4_pipeline_parameter_source_t*
id4_cli_parameter_bake_stage_source(
    const id4_ideogram4_generation_parameter_sources_t* sources,
    const id4_cli_parameter_bake_stage_descriptor_t* descriptor) {
  return (const id4_pipeline_parameter_source_t*)((const uint8_t*)sources +
                                                  descriptor->source_offset);
}

static const id4_cli_parameter_bake_stage_descriptor_t*
id4_cli_parameter_bake_find_stage_descriptor(iree_string_view_t stage_key) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(id4_cli_parameter_bake_stage_descriptors); ++i) {
    const id4_cli_parameter_bake_stage_descriptor_t* descriptor =
        &id4_cli_parameter_bake_stage_descriptors[i];
    if (iree_string_view_equal(descriptor->stage_key, stage_key)) {
      return descriptor;
    }
  }
  return NULL;
}

static void id4_cli_parameter_archive_close(
    id4_cli_parameter_archive_t* archive) {
  iree_io_parameter_provider_release(archive->provider);
  archive->provider = NULL;
  iree_io_parameter_index_release(archive->index);
  archive->index = NULL;
  iree_io_file_handle_release(archive->file_handle);
  archive->file_handle = NULL;
}

static void id4_cli_parameter_archive_deinitialize(
    id4_cli_parameter_archive_t* archive, iree_allocator_t host_allocator) {
  id4_cli_parameter_archive_close(archive);
  id4_tooling_free_path(&archive->partial_path, host_allocator);
  id4_tooling_free_path(&archive->final_path, host_allocator);
}

static iree_status_t id4_cli_parameter_archive_create(
    const id4_pipeline_plan_t* plan,
    const id4_cli_parameter_bake_stage_descriptor_t* descriptor,
    iree_string_view_t output_directory, iree_allocator_t host_allocator,
    id4_cli_parameter_archive_t* out_archive) {
  memset(out_archive, 0, sizeof(*out_archive));
  iree_io_parameter_archive_builder_t builder;
  memset(&builder, 0, sizeof(builder));
  bool builder_initialized = false;
  iree_io_stream_t* stream = NULL;

  iree_status_t status =
      id4_tooling_format_child_path(output_directory, descriptor->file_name,
                                    host_allocator, &out_archive->final_path);
  if (iree_status_is_ok(status)) {
    status = id4_tooling_format_child_path(
        output_directory, descriptor->partial_file_name, host_allocator,
        &out_archive->partial_path);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_io_parameter_archive_builder_initialize(host_allocator, &builder);
    builder_initialized = iree_status_is_ok(status);
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_layout_add_archive_entries(plan, &builder);
  }
  if (iree_status_is_ok(status)) {
    status = iree_io_file_handle_create(
        IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_WRITE |
            IREE_IO_FILE_MODE_RANDOM_ACCESS | IREE_IO_FILE_MODE_ASYNC,
        out_archive->partial_path,
        iree_io_parameter_archive_builder_total_size(&builder), host_allocator,
        &out_archive->file_handle);
  }
  if (iree_status_is_ok(status)) {
    status = iree_io_stream_open(
        IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE,
        out_archive->file_handle, /*file_offset=*/0, host_allocator, &stream);
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_io_parameter_index_create(host_allocator, &out_archive->index);
  }
  if (iree_status_is_ok(status)) {
    status = iree_io_parameter_archive_builder_write(
        &builder, out_archive->file_handle, /*file_offset=*/0, stream,
        out_archive->index);
  }
  if (iree_status_is_ok(status)) {
    status =
        id4_pipeline_parameter_layout_validate_index(plan, out_archive->index);
  }
  if (iree_status_is_ok(status)) {
    status = iree_io_parameter_index_provider_create(
        descriptor->archive_scope, out_archive->index,
        IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS,
        host_allocator, &out_archive->provider);
  }
  iree_io_stream_release(stream);
  if (builder_initialized) {
    iree_io_parameter_archive_builder_deinitialize(&builder);
  }
  if (!iree_status_is_ok(status)) {
    id4_cli_parameter_archive_deinitialize(out_archive, host_allocator);
  }
  return status;
}

static iree_hal_semaphore_list_t id4_cli_parameter_bake_one_semaphore(
    iree_hal_semaphore_t** semaphore, uint64_t* payload_value) {
  return (iree_hal_semaphore_list_t){
      // One timeline semaphore edge.
      .count = 1,
      // Timeline semaphore storage.
      .semaphores = semaphore,
      // Timeline payload storage.
      .payload_values = payload_value,
  };
}

static uint64_t id4_cli_parameter_bake_ceil_mib(
    iree_device_size_t byte_length) {
  return (uint64_t)((byte_length + (1024 * 1024 - 1)) / (1024 * 1024));
}

static iree_status_t id4_cli_parameter_bake_stage(
    const id4_cli_parameter_bake_options_t* options,
    const id4_cli_parameter_bake_stage_descriptor_t* descriptor,
    const id4_pipeline_plan_t* plan,
    const id4_pipeline_parameter_source_t* source,
    iree_allocator_t host_allocator) {
  if (source->kind != ID4_PIPELINE_PARAMETER_SOURCE_KIND_CHECKPOINT ||
      !source->storage.checkpoint.provider) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter bake stage %.*s requires a checkpoint source",
        (int)descriptor->stage_key.size, descriptor->stage_key.data);
  }
  const id4_pipeline_parameter_slab_plan_t* first_slab =
      id4_pipeline_plan_parameter_slab_at(plan, /*index=*/0);
  const id4_pipeline_device_placement_t* placement =
      first_slab
          ? id4_pipeline_plan_placement_at(plan, first_slab->placement_id)
          : NULL;
  iree_hal_device_t* device =
      placement
          ? iree_hal_device_group_device_at(
                id4_pipeline_plan_device_group(plan), placement->device_index)
          : NULL;
  if (!first_slab || !placement || !device) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "parameter bake stage %.*s has no complete device placement",
        (int)descriptor->stage_key.size, descriptor->stage_key.data);
  }
  const iree_host_size_t slab_count =
      id4_pipeline_plan_parameter_slab_count(plan);
  for (iree_host_size_t i = 1; i < slab_count; ++i) {
    const id4_pipeline_parameter_slab_plan_t* slab =
        id4_pipeline_plan_parameter_slab_at(plan, i);
    const id4_pipeline_device_placement_t* slab_placement =
        slab ? id4_pipeline_plan_placement_at(plan, slab->placement_id) : NULL;
    iree_hal_device_t* slab_device =
        slab_placement ? iree_hal_device_group_device_at(
                             id4_pipeline_plan_device_group(plan),
                             slab_placement->device_index)
                       : NULL;
    if (!slab || !slab_placement || slab_device != device ||
        slab_placement->queue_affinity != placement->queue_affinity) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "one parameter archive cannot span stage device queues");
    }
  }

  id4_pipeline_parameter_layout_statistics_t statistics;
  memset(&statistics, 0, sizeof(statistics));
  IREE_RETURN_IF_ERROR(
      id4_pipeline_parameter_layout_query_statistics(plan, &statistics));
  id4_cli_parameter_archive_t archive;
  memset(&archive, 0, sizeof(archive));
  iree_io_parameter_provider_t* materialization_provider = NULL;
  iree_hal_semaphore_t* load_semaphore = NULL;
  iree_hal_semaphore_t* completion_semaphore = NULL;
  id4_pipeline_parameter_slab_set_t* parameter_slabs = NULL;
  bool load_submitted = false;
  bool population_submitted = false;

  fprintf(stdout,
          "Baking parameter layout: stage=%.*s source_entries=%" PRIhsz
          " source=%" PRIu64 "MiB execution_entries=%" PRIhsz
          " execution=%" PRIu64 "MiB path=%.*s\n",
          (int)descriptor->stage_key.size, descriptor->stage_key.data,
          statistics.source_entry_count,
          id4_cli_parameter_bake_ceil_mib(statistics.source_byte_length),
          statistics.execution_entry_count,
          id4_cli_parameter_bake_ceil_mib(statistics.execution_byte_length),
          (int)descriptor->file_name.size, descriptor->file_name.data);

  iree_status_t status = descriptor->provider_create(
      options, plan, source->storage.checkpoint.provider, host_allocator,
      &materialization_provider);
  if (iree_status_is_ok(status)) {
    status = id4_cli_parameter_archive_create(
        plan, descriptor, options->output_directory, host_allocator, &archive);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        device, placement->queue_affinity, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &load_semaphore);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_create(
        device, placement->queue_affinity, /*initial_value=*/0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &completion_semaphore);
  }
  uint64_t load_payload_value = 1;
  iree_hal_semaphore_list_t load_signal_list =
      id4_cli_parameter_bake_one_semaphore(&load_semaphore,
                                           &load_payload_value);
  if (iree_status_is_ok(status)) {
    id4_pipeline_parameter_slab_set_load_options_t load_options;
    memset(&load_options, 0, sizeof(load_options));
    load_options.structure_size = sizeof(load_options);
    load_options.provider = materialization_provider;
    load_options.kernel_library = options->kernel_library;
    load_options.kernel_cache = options->runtime_context->kernel_cache;
    load_options.executable_cache = options->runtime_context->executable_cache;
    load_options.command_buffer_mode =
        options->runtime_context->command_buffer_mode;
    load_options.encoder_staging_memory_type =
        IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    load_options.encoder_staging_chunk_byte_capacity =
        options->encoder_staging_chunk_byte_capacity;
    load_options.diagnostic_artifact_flags = options->diagnostic_artifact_flags;
    load_options.wait_semaphore_list = iree_hal_semaphore_list_empty();
    load_options.signal_semaphore_list = load_signal_list;
    load_options.diagnostics_sink = options->diagnostics_sink;
    status = id4_pipeline_plan_load_parameter_slabs(
        plan, &load_options, host_allocator, &parameter_slabs);
    load_submitted = iree_status_is_ok(status);
  }
  uint64_t completion_payload_value = 1;
  iree_hal_semaphore_list_t completion_signal_list =
      id4_cli_parameter_bake_one_semaphore(&completion_semaphore,
                                           &completion_payload_value);
  if (iree_status_is_ok(status)) {
    id4_pipeline_parameter_layout_populate_options_t populate_options;
    memset(&populate_options, 0, sizeof(populate_options));
    populate_options.structure_size = sizeof(populate_options);
    populate_options.source_provider = materialization_provider;
    populate_options.target_index = archive.index;
    populate_options.target_provider = archive.provider;
    populate_options.target_scope = descriptor->archive_scope;
    populate_options.parameter_slabs = parameter_slabs;
    populate_options.staging_chunk_byte_capacity =
        options->archive_staging_chunk_byte_capacity;
    populate_options.wait_semaphore_list = load_signal_list;
    populate_options.signal_semaphore_list = completion_signal_list;
    populate_options.diagnostics_sink = options->diagnostics_sink;
    status = id4_pipeline_parameter_layout_populate(plan, &populate_options,
                                                    host_allocator);
    population_submitted = iree_status_is_ok(status);
  }
  if (population_submitted) {
    status = iree_status_join(
        status, iree_hal_semaphore_wait(
                    completion_semaphore, completion_payload_value,
                    iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  } else if (load_submitted) {
    status = iree_status_join(
        status, iree_hal_semaphore_wait(load_semaphore, load_payload_value,
                                        iree_infinite_timeout(),
                                        IREE_ASYNC_WAIT_FLAG_NONE));
  }
  if (load_submitted) {
    status = iree_status_join(
        status, id4_pipeline_parameter_slab_set_check_load_group_failures(
                    parameter_slabs, id4_pipeline_plan_stage_name(plan),
                    options->diagnostics_sink));
  }
  if (iree_status_is_ok(status)) {
    status = iree_io_file_handle_flush(archive.file_handle);
  }
  id4_pipeline_parameter_slab_set_release(parameter_slabs);
  iree_io_parameter_provider_release(materialization_provider);
  iree_hal_semaphore_release(completion_semaphore);
  iree_hal_semaphore_release(load_semaphore);
  id4_cli_parameter_archive_close(&archive);
  if (iree_status_is_ok(status)) {
    status = id4_tooling_replace_file(archive.partial_path, archive.final_path,
                                      host_allocator);
  }
  if (iree_status_is_ok(status)) {
    fprintf(stdout, "Baked parameter layout: stage=%.*s scope=%.*s path=%.*s\n",
            (int)descriptor->stage_key.size, descriptor->stage_key.data,
            (int)descriptor->archive_scope.size, descriptor->archive_scope.data,
            (int)archive.final_path.size, archive.final_path.data);
  }
  id4_cli_parameter_archive_deinitialize(&archive, host_allocator);
  return status;
}

static iree_status_t id4_cli_parameter_bake_validate_options(
    const id4_cli_parameter_bake_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter bake options are required");
  }
  if (options->structure_size < sizeof(*options)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter bake options are too small");
  }
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "parameter bake extension structures are not supported");
  }
  if (!options->generation_plan || !options->parameter_sources ||
      !options->runtime_context || !options->kernel_library) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter bake plan, sources, runtime, and kernel "
                            "library are required");
  }
  if (iree_string_view_is_empty(options->output_directory)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter bake output directory is required");
  }
  if (options->encoder_staging_chunk_byte_capacity == 0 ||
      options->archive_staging_chunk_byte_capacity == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter bake staging capacities must be nonzero");
  }
  id4_ideogram4_generation_resident_stage_mask_t supported_stage_mask =
      ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_NONE;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(id4_cli_parameter_bake_stage_descriptors); ++i) {
    supported_stage_mask |=
        id4_cli_parameter_bake_stage_descriptors[i].stage_mask;
  }
  if ((options->stage_mask & supported_stage_mask) == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter bake stage mask selects no parameter-bearing stages");
  }
  return id4_pipeline_diagnostics_validate_sink(options->diagnostics_sink,
                                                IREE_SV("parameter bake"));
}

iree_status_t id4_cli_bake_parameter_layouts(
    const id4_cli_parameter_bake_options_t* options,
    iree_allocator_t host_allocator) {
  IREE_RETURN_IF_ERROR(id4_cli_parameter_bake_validate_options(options));
  IREE_RETURN_IF_ERROR(
      id4_tooling_ensure_directory(options->output_directory, host_allocator));

  id4_ideogram4_generation_resident_stage_mask_t baked_stage_mask =
      ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_NONE;
  const iree_host_size_t stage_count =
      id4_ideogram4_generation_plan_stage_count(options->generation_plan);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < stage_count && iree_status_is_ok(status);
       ++i) {
    iree_string_view_t stage_key = iree_string_view_empty();
    const id4_pipeline_plan_t* stage_plan = NULL;
    status = id4_ideogram4_generation_plan_stage_at(options->generation_plan, i,
                                                    &stage_key, &stage_plan);
    if (!iree_status_is_ok(status) ||
        id4_pipeline_plan_parameter_slab_count(stage_plan) == 0) {
      continue;
    }
    const id4_cli_parameter_bake_stage_descriptor_t* descriptor =
        id4_cli_parameter_bake_find_stage_descriptor(stage_key);
    if (!descriptor) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "parameter-bearing generation stage %.*s has no bake descriptor",
          (int)stage_key.size, stage_key.data);
      break;
    }
    if (!iree_any_bit_set(options->stage_mask, descriptor->stage_mask)) {
      continue;
    }
    const id4_pipeline_parameter_source_t* source =
        id4_cli_parameter_bake_stage_source(options->parameter_sources,
                                            descriptor);
    status = id4_cli_parameter_bake_stage(options, descriptor, stage_plan,
                                          source, host_allocator);
    if (iree_status_is_ok(status)) {
      baked_stage_mask |= descriptor->stage_mask;
    }
  }
  const id4_ideogram4_generation_resident_stage_mask_t requested_stage_mask =
      options->stage_mask &
      (ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_QWEN |
       ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_DIT_CONDITIONED |
       ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_DIT_UNCONDITIONED |
       ID4_IDEOGRAM4_GENERATION_RESIDENT_STAGE_DECODE);
  if (iree_status_is_ok(status) && baked_stage_mask != requested_stage_mask) {
    status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "generation plan did not provide every selected parameter bake stage");
  }
  return status;
}
