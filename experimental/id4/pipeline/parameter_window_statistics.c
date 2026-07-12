// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/parameter_window_statistics.h"

#include <inttypes.h>
#include <string.h>

#include "experimental/id4/pipeline/parameter_window.h"

typedef struct id4_pipeline_parameter_region_window_statistics_t {
  // Exact compact target allocation bytes for one region window.
  iree_device_size_t target_byte_length;
  // Provider source transfer bytes required to populate one region window.
  iree_device_size_t source_transfer_byte_length;
  // Number of load groups submitted for one region window.
  iree_host_size_t load_group_count;
  // Number of encoded load steps submitted for one region window.
  iree_host_size_t encode_load_step_count;
} id4_pipeline_parameter_region_window_statistics_t;

typedef struct id4_pipeline_parameter_slab_window_statistics_t {
  // Stage-owned encoder staging allocation for this target slab.
  iree_device_size_t staging_byte_length;
  // First semantic region whose compact window submits an encoded load.
  iree_host_size_t first_encode_region;
} id4_pipeline_parameter_slab_window_statistics_t;

static iree_status_t id4_pipeline_parameter_window_statistics_add_device_size(
    iree_device_size_t addend, iree_string_view_t field_name,
    iree_device_size_t* inout_value) {
  if (!iree_device_size_checked_add(*inout_value, addend, inout_value)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter window %.*s byte length overflows",
                            (int)field_name.size, field_name.data);
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_window_statistics_add_host_size(
    iree_host_size_t addend, iree_string_view_t field_name,
    iree_host_size_t* inout_value) {
  if (!iree_host_size_checked_add(*inout_value, addend, inout_value)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter window %.*s count overflows",
                            (int)field_name.size, field_name.data);
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_parameter_window_statistics_request_index(
    const id4_pipeline_parameter_load_step_t* step,
    iree_host_size_t request_ordinal, iree_host_size_t* out_request_index) {
  *out_request_index = IREE_HOST_SIZE_MAX;
  if (request_ordinal >= step->request_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter load step request ordinal %" PRIhsz
                            " exceeds request count %" PRIhsz,
                            request_ordinal, step->request_count);
  }
  if (step->request_indices) {
    *out_request_index = step->request_indices[request_ordinal];
    return iree_ok_status();
  }
  if (!iree_host_size_checked_add(step->request_offset, request_ordinal,
                                  out_request_index)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "parameter load step request index overflows");
  }
  return iree_ok_status();
}

static iree_status_t
id4_pipeline_parameter_window_schedule_load_group_target_statistics(
    const id4_pipeline_parameter_window_t* window,
    const id4_pipeline_parameter_window_schedule_t* schedule,
    const id4_pipeline_parameter_load_group_t* group,
    iree_host_size_t compact_group_index,
    id4_pipeline_parameter_window_statistics_t* statistics,
    iree_device_size_t* out_target_byte_length) {
  *out_target_byte_length = 0;
  const iree_host_size_t load_count =
      id4_pipeline_parameter_window_schedule_load_count(schedule);
  const id4_pipeline_parameter_slab_load_t* loads =
      id4_pipeline_parameter_window_schedule_loads(schedule);
  const iree_host_size_t load_step_count =
      id4_pipeline_parameter_window_schedule_load_step_count(schedule);
  const id4_pipeline_parameter_load_step_t* load_steps =
      id4_pipeline_parameter_window_schedule_load_steps(schedule);
  const iree_host_size_t original_group_index =
      id4_pipeline_parameter_window_schedule_original_load_group_at(
          schedule, compact_group_index);
  const iree_host_size_t step_limit = group->step_offset + group->step_count;
  if (step_limit > load_step_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "compact parameter load group step range exceeds "
                            "the schedule");
  }
  for (iree_host_size_t step_index = group->step_offset;
       step_index < step_limit; ++step_index) {
    const id4_pipeline_parameter_load_step_t* step = &load_steps[step_index];
    if (step->target_slab_index >= load_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "compact parameter load step %" PRIhsz
                              " target slab %" PRIhsz
                              " exceeds compact slab count %" PRIhsz,
                              step_index, step->target_slab_index, load_count);
    }
    const id4_pipeline_parameter_slab_load_t* load =
        &loads[step->target_slab_index];
    const id4_pipeline_parameter_window_slab_t* window_slab =
        id4_pipeline_parameter_window_slab_at(window, step->target_slab_index);
    if (!load->request_table || !window_slab) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "compact parameter slab %" PRIhsz " is missing",
                              step->target_slab_index);
    }
    for (iree_host_size_t request_ordinal = 0;
         request_ordinal < step->request_count; ++request_ordinal) {
      iree_host_size_t request_index = IREE_HOST_SIZE_MAX;
      IREE_RETURN_IF_ERROR(
          id4_pipeline_parameter_window_statistics_request_index(
              step, request_ordinal, &request_index));
      if (request_index >= load->request_table->count) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "parameter window request %" PRIhsz
                                " is outside slab %" PRIhsz,
                                request_index, step->target_slab_index);
      }
      const iree_device_size_t request_byte_length =
          load->request_table->values[request_index].span.length;
      IREE_RETURN_IF_ERROR(
          id4_pipeline_parameter_window_statistics_add_device_size(
              request_byte_length, IREE_SV("group target"),
              out_target_byte_length));
      if (request_byte_length >
          statistics->largest_request_target_byte_length) {
        if (request_index >= window_slab->request_count) {
          return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "compact parameter request %" PRIhsz
                                  " exceeds window slab request count %" PRIhsz,
                                  request_index, window_slab->request_count);
        }
        const id4_pipeline_parameter_window_request_t* window_request =
            id4_pipeline_parameter_window_request_at(
                window, window_slab->request_offset + request_index);
        if (!window_request) {
          return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "compact parameter request %" PRIhsz
                                  " is missing from its window",
                                  request_index);
        }
        statistics->largest_request_target_byte_length = request_byte_length;
        statistics->largest_request_index =
            window_request->global_request_index;
        statistics->largest_request_load_group_index = original_group_index;
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t
id4_pipeline_parameter_window_schedule_encode_group_statistics(
    const id4_pipeline_parameter_window_schedule_t* schedule,
    const id4_pipeline_parameter_load_group_t* group,
    iree_device_size_t staging_chunk_byte_capacity,
    id4_pipeline_parameter_encode_statistics_t* out_statistics) {
  const iree_host_size_t load_count =
      id4_pipeline_parameter_window_schedule_load_count(schedule);
  const id4_pipeline_parameter_slab_load_t* loads =
      id4_pipeline_parameter_window_schedule_loads(schedule);
  const iree_host_size_t load_step_count =
      id4_pipeline_parameter_window_schedule_load_step_count(schedule);
  const id4_pipeline_parameter_load_step_t* load_steps =
      id4_pipeline_parameter_window_schedule_load_steps(schedule);
  if (group->step_count == 0 || group->step_offset >= load_step_count ||
      group->target_slab_index >= load_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "compact encoded parameter load group is outside its schedule");
  }
  const id4_pipeline_parameter_slab_load_t* load =
      &loads[group->target_slab_index];
  return id4_pipeline_parameter_encode_query_statistics(
      load->request_table, group->step_count, &load_steps[group->step_offset],
      staging_chunk_byte_capacity, out_statistics);
}

static iree_status_t id4_pipeline_parameter_window_region_statistics(
    const id4_pipeline_plan_t* plan,
    const id4_pipeline_parameter_window_t* window,
    iree_host_size_t region_index,
    iree_device_size_t staging_chunk_byte_capacity,
    id4_pipeline_parameter_window_statistics_t* statistics,
    iree_host_size_t slab_statistics_count,
    id4_pipeline_parameter_slab_window_statistics_t* slab_statistics,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_region_window_statistics_t* out_region) {
  memset(out_region, 0, sizeof(*out_region));
  const iree_host_size_t window_slab_count =
      id4_pipeline_parameter_window_slab_count(window);
  for (iree_host_size_t i = 0; i < window_slab_count; ++i) {
    const id4_pipeline_parameter_window_slab_t* slab =
        id4_pipeline_parameter_window_slab_at(window, i);
    IREE_RETURN_IF_ERROR(
        id4_pipeline_parameter_window_statistics_add_device_size(
            slab->byte_length, IREE_SV("region target"),
            &out_region->target_byte_length));
  }

  id4_pipeline_parameter_window_schedule_create_options_t schedule_options;
  memset(&schedule_options, 0, sizeof(schedule_options));
  schedule_options.structure_size = sizeof(schedule_options);
  schedule_options.plan = (id4_pipeline_plan_t*)plan;
  schedule_options.window = window;
  id4_pipeline_parameter_window_schedule_t* schedule = NULL;
  iree_status_t status = id4_pipeline_parameter_window_schedule_create(
      &schedule_options, host_allocator, &schedule);
  const iree_host_size_t load_step_count =
      id4_pipeline_parameter_window_schedule_load_step_count(schedule);
  const id4_pipeline_parameter_load_step_t* load_steps =
      id4_pipeline_parameter_window_schedule_load_steps(schedule);
  if (iree_status_is_ok(status)) {
    out_region->load_group_count =
        id4_pipeline_parameter_window_schedule_load_group_count(schedule);
  }
  for (iree_host_size_t compact_group_index = 0;
       compact_group_index < out_region->load_group_count &&
       iree_status_is_ok(status);
       ++compact_group_index) {
    id4_pipeline_parameter_load_group_t group;
    status = id4_pipeline_parameter_load_group_at(load_step_count, load_steps,
                                                  compact_group_index, &group);
    if (!iree_status_is_ok(status)) break;
    const iree_host_size_t original_group_index =
        id4_pipeline_parameter_window_schedule_original_load_group_at(
            schedule, compact_group_index);
    iree_device_size_t group_target_byte_length = 0;
    status =
        id4_pipeline_parameter_window_schedule_load_group_target_statistics(
            window, schedule, &group, compact_group_index, statistics,
            &group_target_byte_length);
    if (!iree_status_is_ok(status)) break;
    if (group_target_byte_length >
        statistics->largest_load_group_target_byte_length) {
      statistics->largest_load_group_target_byte_length =
          group_target_byte_length;
      statistics->largest_load_group_index = original_group_index;
    }

    switch (group.kind) {
      case ID4_PIPELINE_PARAMETER_LOAD_GROUP_KIND_GATHER:
        status = id4_pipeline_parameter_window_statistics_add_device_size(
            group_target_byte_length, IREE_SV("region source transfer"),
            &out_region->source_transfer_byte_length);
        break;
      case ID4_PIPELINE_PARAMETER_LOAD_GROUP_KIND_ENCODE: {
        id4_pipeline_parameter_encode_statistics_t encode_statistics;
        status = id4_pipeline_parameter_window_schedule_encode_group_statistics(
            schedule, &group, staging_chunk_byte_capacity, &encode_statistics);
        if (iree_status_is_ok(status)) {
          status = id4_pipeline_parameter_window_statistics_add_device_size(
              encode_statistics.source_byte_length,
              IREE_SV("region source transfer"),
              &out_region->source_transfer_byte_length);
        }
        if (iree_status_is_ok(status)) {
          status = id4_pipeline_parameter_window_statistics_add_host_size(
              group.step_count, IREE_SV("region encoded load step"),
              &out_region->encode_load_step_count);
        }
        if (!iree_status_is_ok(status)) break;
        const id4_pipeline_parameter_window_slab_t* window_slab =
            id4_pipeline_parameter_window_slab_at(window,
                                                  group.target_slab_index);
        if (!window_slab) {
          status =
              iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                               "compact encoded load group target slab %" PRIhsz
                               " exceeds window slab count %" PRIhsz,
                               group.target_slab_index, window_slab_count);
          break;
        }
        const iree_host_size_t original_slab_index =
            window_slab->original_slab_index;
        if (original_slab_index >= slab_statistics_count) {
          status = iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "compact encoded load group original slab %" PRIhsz
              " exceeds plan slab count %" PRIhsz,
              original_slab_index, slab_statistics_count);
          break;
        }
        id4_pipeline_parameter_slab_window_statistics_t* slab =
            &slab_statistics[original_slab_index];
        slab->staging_byte_length =
            iree_max(slab->staging_byte_length,
                     encode_statistics.staging_total_byte_length);
        slab->first_encode_region =
            iree_min(slab->first_encode_region, region_index);
        break;
      }
      default:
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "compact parameter load group %" PRIhsz
                                  " has invalid kind %u",
                                  compact_group_index, group.kind);
        break;
    }
  }

  id4_pipeline_parameter_window_schedule_release(schedule);
  return status;
}

static iree_status_t id4_pipeline_parameter_window_finalize_slab_statistics(
    iree_host_size_t slab_statistics_count,
    const id4_pipeline_parameter_slab_window_statistics_t* slab_statistics,
    iree_device_size_t* out_staging_byte_length) {
  *out_staging_byte_length = 0;
  for (iree_host_size_t i = 0; i < slab_statistics_count; ++i) {
    IREE_RETURN_IF_ERROR(
        id4_pipeline_parameter_window_statistics_add_device_size(
            slab_statistics[i].staging_byte_length, IREE_SV("encoder staging"),
            out_staging_byte_length));
  }
  return iree_ok_status();
}

iree_status_t id4_pipeline_parameter_window_query_resource_statistics(
    const id4_pipeline_plan_t* plan,
    const id4_pipeline_parameter_window_t* window,
    id4_pipeline_parameter_window_source_kind_t source_kind,
    iree_device_size_t encoder_staging_chunk_byte_capacity,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_window_resource_statistics_t* out_statistics) {
  IREE_ASSERT_ARGUMENT(out_statistics);
  memset(out_statistics, 0, sizeof(*out_statistics));
  if (!plan || !window) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter window resource statistics require a plan and window");
  }
  if (source_kind ==
      ID4_PIPELINE_PARAMETER_WINDOW_SOURCE_KIND_EXECUTION_LAYOUT) {
    id4_pipeline_parameter_window_resource_statistics_t statistics;
    memset(&statistics, 0, sizeof(statistics));
    statistics.load_group_count =
        id4_pipeline_parameter_window_slab_count(window);
    for (iree_host_size_t i = 0; i < statistics.load_group_count; ++i) {
      const id4_pipeline_parameter_window_slab_t* slab =
          id4_pipeline_parameter_window_slab_at(window, i);
      if (!slab || !iree_device_size_checked_add(
                       statistics.target_byte_length, slab->byte_length,
                       &statistics.target_byte_length)) {
        return iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "execution-layout parameter window byte length overflows");
      }
    }
    statistics.source_transfer_byte_length = statistics.target_byte_length;
    *out_statistics = statistics;
    return iree_ok_status();
  }
  if (source_kind != ID4_PIPELINE_PARAMETER_WINDOW_SOURCE_KIND_CHECKPOINT ||
      encoder_staging_chunk_byte_capacity == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "checkpoint parameter window resource statistics require an encoder "
        "staging capacity");
  }

  const iree_host_size_t parameter_slab_count =
      id4_pipeline_plan_parameter_slab_count(plan);
  id4_pipeline_parameter_slab_window_statistics_t* slab_statistics = NULL;
  iree_status_t status = iree_ok_status();
  if (parameter_slab_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, parameter_slab_count,
                                         sizeof(slab_statistics[0]),
                                         (void**)&slab_statistics);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < parameter_slab_count; ++i) {
      memset(&slab_statistics[i], 0, sizeof(slab_statistics[i]));
      slab_statistics[i].first_encode_region = IREE_HOST_SIZE_MAX;
    }
  }

  id4_pipeline_parameter_window_statistics_t aggregate_statistics;
  memset(&aggregate_statistics, 0, sizeof(aggregate_statistics));
  aggregate_statistics.largest_load_group_index = IREE_HOST_SIZE_MAX;
  aggregate_statistics.largest_request_index = IREE_HOST_SIZE_MAX;
  aggregate_statistics.largest_request_load_group_index = IREE_HOST_SIZE_MAX;
  id4_pipeline_parameter_region_window_statistics_t region_statistics;
  memset(&region_statistics, 0, sizeof(region_statistics));
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_window_region_statistics(
        plan, window, /*region_index=*/0, encoder_staging_chunk_byte_capacity,
        &aggregate_statistics, parameter_slab_count, slab_statistics,
        host_allocator, &region_statistics);
  }

  id4_pipeline_parameter_window_resource_statistics_t statistics;
  memset(&statistics, 0, sizeof(statistics));
  if (iree_status_is_ok(status)) {
    statistics.target_byte_length = region_statistics.target_byte_length;
    statistics.source_transfer_byte_length =
        region_statistics.source_transfer_byte_length;
    statistics.load_group_count = region_statistics.load_group_count;
    statistics.encode_load_step_count =
        region_statistics.encode_load_step_count;
    status = id4_pipeline_parameter_window_finalize_slab_statistics(
        parameter_slab_count, slab_statistics,
        &statistics.encoder_staging_byte_length);
  }
  if (iree_status_is_ok(status)) {
    *out_statistics = statistics;
  }
  iree_allocator_free(host_allocator, slab_statistics);
  return status;
}

static iree_status_t id4_pipeline_parameter_window_statistics_validate_options(
    const id4_pipeline_parameter_window_statistics_options_t* options) {
  if (!options || options->structure_size < sizeof(*options)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter window statistics options are invalid");
  }
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "parameter window statistics extension structures are not supported");
  }
  if (!options->plan || options->concurrent_window_count == 0 ||
      options->encoder_staging_chunk_byte_capacity == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "parameter window statistics require a plan, concurrent windows, and "
        "encoder staging capacity");
  }
  return iree_ok_status();
}

iree_status_t id4_pipeline_parameter_window_query_statistics(
    const id4_pipeline_parameter_window_statistics_options_t* options,
    iree_allocator_t host_allocator,
    id4_pipeline_parameter_window_statistics_t* out_statistics) {
  IREE_ASSERT_ARGUMENT(out_statistics);
  memset(out_statistics, 0, sizeof(*out_statistics));
  out_statistics->largest_load_group_index = IREE_HOST_SIZE_MAX;
  out_statistics->largest_request_index = IREE_HOST_SIZE_MAX;
  out_statistics->largest_request_load_group_index = IREE_HOST_SIZE_MAX;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_parameter_window_statistics_validate_options(options));

  id4_pipeline_parameter_window_statistics_t statistics;
  memset(&statistics, 0, sizeof(statistics));
  statistics.largest_load_group_index = IREE_HOST_SIZE_MAX;
  statistics.largest_request_index = IREE_HOST_SIZE_MAX;
  statistics.largest_request_load_group_index = IREE_HOST_SIZE_MAX;
  const iree_host_size_t region_count =
      id4_pipeline_plan_region_count(options->plan);
  statistics.concurrent_window_count =
      iree_min(region_count, options->concurrent_window_count);
  statistics.window_count = region_count;
  const iree_host_size_t parameter_slab_count =
      id4_pipeline_plan_parameter_slab_count(options->plan);
  for (iree_host_size_t i = 0; i < parameter_slab_count; ++i) {
    const id4_pipeline_parameter_slab_plan_t* slab =
        id4_pipeline_plan_parameter_slab_at(options->plan, i);
    IREE_RETURN_IF_ERROR(
        id4_pipeline_parameter_window_statistics_add_device_size(
            slab->byte_length, IREE_SV("full slab target"),
            &statistics.full_slab_target_byte_length));
  }

  id4_pipeline_parameter_region_window_statistics_t* region_statistics = NULL;
  id4_pipeline_parameter_slab_window_statistics_t* slab_statistics = NULL;
  iree_status_t status = iree_ok_status();
  if (region_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, region_count,
                                         sizeof(region_statistics[0]),
                                         (void**)&region_statistics);
  }
  if (iree_status_is_ok(status) && parameter_slab_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, parameter_slab_count,
                                         sizeof(slab_statistics[0]),
                                         (void**)&slab_statistics);
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < parameter_slab_count; ++i) {
      memset(&slab_statistics[i], 0, sizeof(slab_statistics[i]));
      slab_statistics[i].first_encode_region = IREE_HOST_SIZE_MAX;
    }
  }
  for (iree_host_size_t region_index = 0;
       region_index < region_count && iree_status_is_ok(status);
       ++region_index) {
    id4_pipeline_parameter_window_t* window = NULL;
    status = id4_pipeline_parameter_window_create_for_region(
        options->plan, region_index, host_allocator, &window);
    if (iree_status_is_ok(status)) {
      status = id4_pipeline_parameter_window_region_statistics(
          options->plan, window, region_index,
          options->encoder_staging_chunk_byte_capacity, &statistics,
          parameter_slab_count, slab_statistics, host_allocator,
          &region_statistics[region_index]);
    }
    id4_pipeline_parameter_window_release(window);
    if (!iree_status_is_ok(status)) break;
    const id4_pipeline_parameter_region_window_statistics_t* region =
        &region_statistics[region_index];
    status = id4_pipeline_parameter_window_statistics_add_device_size(
        region->target_byte_length, IREE_SV("total target"),
        &statistics.total_target_byte_length);
    if (iree_status_is_ok(status)) {
      status = id4_pipeline_parameter_window_statistics_add_device_size(
          region->source_transfer_byte_length, IREE_SV("total source transfer"),
          &statistics.total_source_transfer_byte_length);
    }
    if (iree_status_is_ok(status)) {
      status = id4_pipeline_parameter_window_statistics_add_host_size(
          region->load_group_count, IREE_SV("total load group"),
          &statistics.total_load_group_count);
    }
    if (iree_status_is_ok(status)) {
      status = id4_pipeline_parameter_window_statistics_add_host_size(
          region->encode_load_step_count, IREE_SV("total encoded load step"),
          &statistics.total_encode_load_step_count);
    }
  }
  if (iree_status_is_ok(status)) {
    status = id4_pipeline_parameter_window_finalize_slab_statistics(
        parameter_slab_count, slab_statistics,
        &statistics.encoder_staging_byte_length);
  }

  for (iree_host_size_t window_offset = 0;
       window_offset < region_count && iree_status_is_ok(status);
       ++window_offset) {
    id4_pipeline_parameter_region_window_statistics_t concurrent;
    memset(&concurrent, 0, sizeof(concurrent));
    const iree_host_size_t window_limit = iree_min(
        region_count, window_offset + statistics.concurrent_window_count);
    for (iree_host_size_t region_index = window_offset;
         region_index < window_limit && iree_status_is_ok(status);
         ++region_index) {
      const id4_pipeline_parameter_region_window_statistics_t* region =
          &region_statistics[region_index];
      status = id4_pipeline_parameter_window_statistics_add_device_size(
          region->target_byte_length, IREE_SV("concurrent target"),
          &concurrent.target_byte_length);
      if (iree_status_is_ok(status)) {
        status = id4_pipeline_parameter_window_statistics_add_device_size(
            region->source_transfer_byte_length,
            IREE_SV("concurrent source transfer"),
            &concurrent.source_transfer_byte_length);
      }
      if (iree_status_is_ok(status)) {
        status = id4_pipeline_parameter_window_statistics_add_host_size(
            region->load_group_count, IREE_SV("concurrent load group"),
            &concurrent.load_group_count);
      }
      if (iree_status_is_ok(status)) {
        status = id4_pipeline_parameter_window_statistics_add_host_size(
            region->encode_load_step_count,
            IREE_SV("concurrent encoded load step"),
            &concurrent.encode_load_step_count);
      }
    }
    iree_device_size_t active_staging_byte_length = 0;
    const iree_host_size_t prefetched_region_limit = iree_min(
        region_count, window_offset + statistics.concurrent_window_count);
    for (iree_host_size_t slab_index = 0;
         slab_index < parameter_slab_count && iree_status_is_ok(status);
         ++slab_index) {
      const id4_pipeline_parameter_slab_window_statistics_t* slab =
          &slab_statistics[slab_index];
      if (slab->first_encode_region >= prefetched_region_limit) continue;
      status = id4_pipeline_parameter_window_statistics_add_device_size(
          slab->staging_byte_length, IREE_SV("active encoder staging"),
          &active_staging_byte_length);
    }
    iree_device_size_t live_byte_length = concurrent.target_byte_length;
    if (iree_status_is_ok(status)) {
      status = id4_pipeline_parameter_window_statistics_add_device_size(
          active_staging_byte_length, IREE_SV("concurrent live"),
          &live_byte_length);
    }
    if (!iree_status_is_ok(status)) break;
    statistics.peak_target_byte_length = iree_max(
        statistics.peak_target_byte_length, concurrent.target_byte_length);
    statistics.peak_live_byte_length =
        iree_max(statistics.peak_live_byte_length, live_byte_length);
    statistics.peak_source_transfer_byte_length =
        iree_max(statistics.peak_source_transfer_byte_length,
                 concurrent.source_transfer_byte_length);
    statistics.peak_load_group_count =
        iree_max(statistics.peak_load_group_count, concurrent.load_group_count);
    statistics.peak_encode_load_step_count =
        iree_max(statistics.peak_encode_load_step_count,
                 concurrent.encode_load_step_count);
  }

  if (iree_status_is_ok(status)) {
    *out_statistics = statistics;
  }
  iree_allocator_free(host_allocator, slab_statistics);
  iree_allocator_free(host_allocator, region_statistics);
  return status;
}
