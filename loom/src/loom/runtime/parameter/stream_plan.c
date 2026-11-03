// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/parameter/stream_plan.h"

#include <stdlib.h>

static iree_device_size_t loom_runtime_weight_read_window_size(
    const loom_runtime_virtual_buffer_t* reservation) {
  const iree_device_size_t slab_size =
      loom_runtime_virtual_buffer_mapping_size(reservation);
  return iree_min(slab_size, 64 * 1024 * 1024);
}

// Counts fragments before allocation, then scatters them into each group's
// assigned slice. Each source range visits only the windows it intersects.
static iree_host_size_t loom_runtime_weight_plan_fragments(
    iree_host_size_t span_count, const loom_runtime_weight_span_t* spans,
    const loom_runtime_weight_storage_t* storages,
    const loom_runtime_weight_inputs_t* inputs, bool elastic,
    loom_runtime_weight_stream_plan_t* plan) {
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < span_count; ++i) {
    const loom_runtime_weight_span_t* span = &spans[i];
    if (span->inputs.count && span->prepare) {
      continue;
    }
    for (uint32_t p = 0; p < iree_max(1, span->inputs.count); ++p) {
      loom_runtime_weight_fragment_t fragment = {
          .key = span->parameter.key,
          .range = {.buffer_offset = span->parameter.byte_offset,
                    .length = span->parameter.byte_length}};
      if (span->inputs.count) {
        const loom_runtime_weight_input_t* input =
            loom_runtime_weight_inputs_at(inputs, span->inputs.first + p);
        fragment = (loom_runtime_weight_fragment_t){.key = input->entry->key,
                                                  .range = input->span,
                                                  .source = input->source};
        fragment.range.buffer_offset += span->parameter.byte_offset;
      }
      iree_host_size_t begin = plan->spans[i].begin;
      iree_host_size_t end = plan->spans[i].end;
      if (elastic && span->inputs.count) {
        const iree_device_size_t window_size =
            loom_runtime_weight_read_window_size(
                storages[span->storage_index].reservation);
        const iree_host_size_t base =
            begin - span->parameter.byte_offset / window_size;
        begin = base + fragment.range.buffer_offset / window_size;
        end = base +
              (fragment.range.buffer_offset + fragment.range.length - 1) /
                  window_size +
              1;
      }
      for (iree_host_size_t j = begin; j < end; ++j) {
        loom_runtime_weight_group_t* group = &plan->groups[j];
        group->storage_index = span->storage_index;
        if (group->fragments) {
          loom_runtime_weight_fragment_t clipped = fragment;
          if (elastic) {
            const iree_device_size_t first =
                iree_max(fragment.range.buffer_offset, group->backing.offset);
            const iree_device_size_t last =
                iree_min(fragment.range.buffer_offset + fragment.range.length,
                         group->backing.offset + group->backing.length);
            clipped.range.parameter_offset +=
                first - fragment.range.buffer_offset;
            clipped.range.buffer_offset = first;
            clipped.range.length = last - first;
          }
          group->fragments[group->count] = clipped;
        }
        ++group->count;
        ++count;
      }
    }
  }
  return count;
}

static int loom_runtime_weight_compare_fragments(const void* lhs,
                                               const void* rhs) {
  const loom_runtime_weight_fragment_t* a = lhs;
  const loom_runtime_weight_fragment_t* b = rhs;
  if (a->source != b->source) {
    return a->source < b->source ? -1 : 1;
  }
  return (a->range.buffer_offset > b->range.buffer_offset) -
         (a->range.buffer_offset < b->range.buffer_offset);
}

// Builds only cold activation metadata. Elastic groups are read windows
// within physical slabs; fixed groups preserve canonical tensors and individual
// preparers. Tensor-to-group intervals are recorded here and consumed without
// rescanning.
iree_status_t loom_runtime_weight_stream_plan_initialize(
    iree_host_size_t span_count, const loom_runtime_weight_span_t* spans,
    iree_host_size_t storage_count, const loom_runtime_weight_storage_t* storages,
    const loom_runtime_weight_inputs_t* inputs, bool elastic,
    iree_allocator_t allocator, loom_runtime_weight_stream_plan_t* plan) {
  *plan = (loom_runtime_weight_stream_plan_t){0};
  iree_host_size_t* storage_groups = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(allocator, storage_count,
                                                   sizeof(*storage_groups),
                                                   (void**)&storage_groups));
  iree_status_t status = iree_allocator_malloc_array(
      allocator, span_count, sizeof(*plan->spans), (void**)&plan->spans);
  if (iree_status_is_ok(status)) {
    if (elastic) {
      for (iree_host_size_t i = 0; i < storage_count; ++i) {
        storage_groups[i] = plan->group_count;
        const loom_runtime_virtual_buffer_t* reservation =
            storages[i].reservation;
        const iree_device_size_t length = iree_hal_buffer_byte_length(
            loom_runtime_virtual_buffer_handle(reservation));
        plan->group_count +=
            (length - 1) / loom_runtime_weight_read_window_size(reservation) + 1;
        plan->backing_count +=
            loom_runtime_virtual_buffer_slab_count(reservation);
      }
    }
    for (iree_host_size_t i = 0; i < span_count; ++i) {
      const loom_runtime_weight_span_t* span = &spans[i];
      plan->has_preparation |= span->prepare != NULL;
      if (elastic) {
        const loom_runtime_virtual_buffer_t* reservation =
            storages[span->storage_index].reservation;
        const iree_device_size_t window_size =
            loom_runtime_weight_read_window_size(reservation);
        const iree_host_size_t base = storage_groups[span->storage_index];
        plan->spans[i] = (loom_runtime_weight_tensor_plan_t){
            base + span->parameter.byte_offset / window_size,
            base + (span->parameter.byte_offset + span->parameter.byte_length +
                    window_size - 1) /
                       window_size};
        if (!span->parameter.byte_length) {
          // Empty tensors still resolve their checkpoint key. At the end of a
          // root, the zero-byte read belongs to its final mapped window.
          plan->spans[i].begin =
              base +
              iree_min(span->parameter.byte_offset / window_size,
                       (iree_hal_buffer_byte_length(
                            loom_runtime_virtual_buffer_handle(reservation)) -
                        1) /
                           window_size);
          plan->spans[i].end = plan->spans[i].begin + 1;
        }
      } else {
        if (!i || span->prepare || spans[i - 1].prepare ||
            span->storage_index != spans[i - 1].storage_index) {
          ++plan->group_count;
        }
        plan->spans[i] = (loom_runtime_weight_tensor_plan_t){
            plan->group_count - 1, plan->group_count};
      }
    }
    status = iree_allocator_malloc_array(allocator, plan->group_count,
                                         sizeof(*plan->groups),
                                         (void**)&plan->groups);
  }
  iree_host_size_t fragment_count = 0;
  if (iree_status_is_ok(status)) {
    if (elastic) {
      for (iree_host_size_t i = 0; i < storage_count; ++i) {
        loom_runtime_virtual_buffer_t* reservation = storages[i].reservation;
        const iree_device_size_t window_size =
            loom_runtime_weight_read_window_size(reservation);
        const iree_device_size_t length = iree_hal_buffer_byte_length(
            loom_runtime_virtual_buffer_handle(reservation));
        const iree_host_size_t count = (length - 1) / window_size + 1;
        const iree_host_size_t windows_per_slab =
            loom_runtime_virtual_buffer_mapping_size(reservation) / window_size;
        for (iree_host_size_t j = 0; j < count; ++j) {
          loom_runtime_weight_group_t* group =
              &plan->groups[storage_groups[i] + j];
          group->storage_index = i;
          group->backing = (loom_runtime_memory_range_t){
              reservation, j * window_size,
              iree_min(window_size, length - j * window_size)};
          group->mapping_group =
              storage_groups[i] + j / windows_per_slab * windows_per_slab;
        }
      }
    }
    fragment_count = loom_runtime_weight_plan_fragments(
        span_count, spans, storages, inputs, elastic, plan);
    if (fragment_count) {
      status = iree_allocator_malloc_array(allocator, fragment_count,
                                           sizeof(*plan->fragments),
                                           (void**)&plan->fragments);
    }
  }
  if (iree_status_is_ok(status)) {
    iree_host_size_t cursor = 0;
    for (iree_host_size_t i = 0; i < plan->group_count; ++i) {
      plan->groups[i].fragments =
          plan->fragments ? plan->fragments + cursor : NULL;
      cursor += plan->groups[i].count;
      plan->groups[i].count = 0;
    }
    loom_runtime_weight_plan_fragments(span_count, spans, storages, inputs,
                                     elastic, plan);
    for (iree_host_size_t i = 0; i < plan->group_count; ++i) {
      loom_runtime_weight_group_t* group = &plan->groups[i];
      if (group->count > 1) {
        qsort(group->fragments, group->count, sizeof(*group->fragments),
              loom_runtime_weight_compare_fragments);
      }
      group->read_begin = plan->read_count;
      for (iree_host_size_t j = 0; j < group->count; ++j) {
        plan->read_count +=
            !j || group->fragments[j].source != group->fragments[j - 1].source;
      }
      group->read_end = plan->read_count;
    }
    if (plan->read_count) {
      status = iree_allocator_malloc_array(allocator, plan->read_count,
                                           sizeof(*plan->reads),
                                           (void**)&plan->reads);
    }
  }
  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < plan->group_count; ++i) {
      const loom_runtime_weight_group_t* group = &plan->groups[i];
      iree_host_size_t ordinal = group->read_begin;
      for (iree_host_size_t j = 0; j < group->count; ++j) {
        if (j && group->fragments[j].source != group->fragments[j - 1].source) {
          ++ordinal;
        }
        loom_runtime_weight_direct_read_t* read = &plan->reads[ordinal];
        if (!read->count) {
          read->fragments = &group->fragments[j];
        }
        ++read->count;
        read->byte_length += group->fragments[j].range.length;
      }
    }
  }
  iree_allocator_free(allocator, storage_groups);
  return status;
}

void loom_runtime_weight_stream_plan_deinitialize(
    loom_runtime_weight_stream_plan_t* plan, iree_allocator_t allocator) {
  for (iree_host_size_t i = 0; plan->groups && i < plan->group_count; ++i) {
    iree_hal_semaphore_release(plan->groups[i].mapped);
  }
  iree_allocator_free(allocator, plan->reads);
  iree_allocator_free(allocator, plan->fragments);
  iree_allocator_free(allocator, plan->spans);
  iree_allocator_free(allocator, plan->groups);
}
