// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/clobber_index.h"

#include <string.h>

#include "loom/codegen/low/descriptors.h"

// One transient physical write before location grouping removes repeated keys.
struct loom_low_allocation_clobber_event_t {
  // Descriptor-defined shared storage identity.
  uint32_t storage_key : 31;
  // Whether a definition is permitted exactly at |point|.
  uint32_t permits_definition : 1;
  // Atomic location within |storage_key|.
  uint32_t location;
  // Program point overwritten by the implicit write.
  uint32_t point;
};

static_assert(sizeof(loom_low_allocation_clobber_event_t) == 12,
              "transient physical clobber events must remain compact");

// Build metadata for one descriptor-defined shared storage namespace.
typedef struct loom_low_allocation_clobber_namespace_t {
  // First prefix offset for this namespace in the final index.
  uint32_t location_range_start;
  // One past the largest clobbered location, or zero when unused.
  uint32_t location_count;
} loom_low_allocation_clobber_namespace_t;

void loom_low_allocation_clobber_builder_initialize(
    iree_arena_allocator_t* scratch_arena,
    loom_low_allocation_clobber_builder_t* out_builder) {
  IREE_ASSERT_ARGUMENT(scratch_arena);
  IREE_ASSERT_ARGUMENT(out_builder);
  *out_builder = (loom_low_allocation_clobber_builder_t){
      .scratch_arena = scratch_arena,
  };
}

iree_status_t loom_low_allocation_clobber_builder_record(
    loom_low_allocation_clobber_builder_t* builder, uint32_t storage_key,
    uint32_t location, uint32_t point, bool permits_definition) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_LE(storage_key, UINT32_C(0x1FFFF));
  IREE_ASSERT_LE(location, UINT16_MAX);
  IREE_ASSERT(builder->event_count == 0 || point >= builder->last_point,
              "physical clobbers must follow canonical program-point order");
  if (builder->event_count == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "physical clobber count exceeds u32 range");
  }
  if (builder->event_count == builder->event_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        builder->scratch_arena, builder->event_count, builder->event_count + 1u,
        sizeof(*builder->events), &builder->event_capacity,
        (void**)&builder->events));
  }
  builder->events[builder->event_count++] =
      (loom_low_allocation_clobber_event_t){
          .storage_key = storage_key,
          .permits_definition = permits_definition,
          .location = location,
          .point = point,
      };
  builder->last_point = point;
  builder->has_permitted_definitions |= permits_definition;
  return iree_ok_status();
}

static iree_host_size_t loom_low_allocation_clobber_namespace_ordinal(
    uint32_t storage_key, uint16_t maximum_alias_set_id) {
  if (storage_key <= maximum_alias_set_id) {
    return storage_key;
  }
  IREE_ASSERT_GE(storage_key, UINT32_C(0x10000));
  return (iree_host_size_t)maximum_alias_set_id + 1u +
         (storage_key - UINT32_C(0x10000));
}

iree_status_t loom_low_allocation_clobber_builder_build(
    loom_low_allocation_clobber_builder_t* builder,
    const loom_low_descriptor_set_t* descriptor_set,
    iree_arena_allocator_t* arena,
    loom_low_allocation_clobber_index_t* out_index) {
  IREE_ASSERT_ARGUMENT(builder);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_index);
  *out_index = (loom_low_allocation_clobber_index_t){0};
  if (builder->event_count == 0) {
    return iree_ok_status();
  }
  IREE_ASSERT_ARGUMENT(descriptor_set);

  uint16_t maximum_alias_set_id = 0;
  for (uint32_t i = 0; i < descriptor_set->reg_class_count; ++i) {
    maximum_alias_set_id = iree_max(
        maximum_alias_set_id, descriptor_set->reg_classes[i].alias_set_id);
  }
  iree_host_size_t namespace_count = 0;
  if (!iree_host_size_checked_add((iree_host_size_t)maximum_alias_set_id + 1u,
                                  descriptor_set->reg_class_count,
                                  &namespace_count)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "physical clobber namespace count exceeds host size");
  }
  loom_low_allocation_clobber_namespace_t* namespaces = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(builder->scratch_arena, namespace_count,
                                sizeof(*namespaces), (void**)&namespaces));
  memset(namespaces, 0, namespace_count * sizeof(*namespaces));

  for (iree_host_size_t i = 0; i < builder->event_count; ++i) {
    const loom_low_allocation_clobber_event_t* event = &builder->events[i];
    const iree_host_size_t namespace_ordinal =
        loom_low_allocation_clobber_namespace_ordinal(event->storage_key,
                                                      maximum_alias_set_id);
    IREE_ASSERT_LT(namespace_ordinal, namespace_count);
    namespaces[namespace_ordinal].location_count = iree_max(
        namespaces[namespace_ordinal].location_count, event->location + 1u);
  }

  iree_host_size_t location_range_count = 0;
  for (iree_host_size_t i = 0; i < namespace_count; ++i) {
    loom_low_allocation_clobber_namespace_t* clobber_namespace = &namespaces[i];
    if (clobber_namespace->location_count == 0) {
      continue;
    }
    const iree_host_size_t range_count =
        (iree_host_size_t)clobber_namespace->location_count + 1u;
    if (!iree_host_size_checked_add(location_range_count, range_count,
                                    &location_range_count) ||
        location_range_count > UINT32_MAX) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "physical clobber location index exceeds u32 range");
    }
  }

  uint32_t* points = NULL;
  uint32_t* location_range_starts = NULL;
  loom_low_allocation_clobber_storage_t* storage_by_reg_class = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, builder->event_count, sizeof(*points), (void**)&points));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, location_range_count, sizeof(*location_range_starts),
      (void**)&location_range_starts));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, descriptor_set->reg_class_count, sizeof(*storage_by_reg_class),
      (void**)&storage_by_reg_class));
  memset(location_range_starts, 0,
         location_range_count * sizeof(*location_range_starts));
  memset(storage_by_reg_class, 0,
         descriptor_set->reg_class_count * sizeof(*storage_by_reg_class));

  iree_bitmap_t permitted_definitions = {
      .bit_count = builder->event_count,
  };
  if (builder->has_permitted_definitions) {
    const iree_host_size_t word_count =
        iree_bitmap_calculate_words(builder->event_count);
    uint64_t* words = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, word_count, sizeof(*words), (void**)&words));
    memset(words, 0, word_count * sizeof(*words));
    permitted_definitions.words = words;
  }

  uint32_t next_location_range = 0;
  for (iree_host_size_t i = 0; i < namespace_count; ++i) {
    loom_low_allocation_clobber_namespace_t* clobber_namespace = &namespaces[i];
    if (clobber_namespace->location_count == 0) {
      continue;
    }
    clobber_namespace->location_range_start = next_location_range;
    next_location_range += clobber_namespace->location_count + 1u;
  }
  IREE_ASSERT_EQ(next_location_range, location_range_count);

  for (iree_host_size_t i = 0; i < builder->event_count; ++i) {
    const loom_low_allocation_clobber_event_t* event = &builder->events[i];
    const iree_host_size_t namespace_ordinal =
        loom_low_allocation_clobber_namespace_ordinal(event->storage_key,
                                                      maximum_alias_set_id);
    const loom_low_allocation_clobber_namespace_t* clobber_namespace =
        &namespaces[namespace_ordinal];
    ++location_range_starts[clobber_namespace->location_range_start +
                            event->location + 1u];
  }

  uint32_t next_point = 0;
  for (iree_host_size_t i = 0; i < namespace_count; ++i) {
    const loom_low_allocation_clobber_namespace_t* clobber_namespace =
        &namespaces[i];
    if (clobber_namespace->location_count == 0) {
      continue;
    }
    for (uint32_t j = 0; j <= clobber_namespace->location_count; ++j) {
      const uint32_t range_index = clobber_namespace->location_range_start + j;
      const uint32_t count = location_range_starts[range_index];
      next_point += count;
      location_range_starts[range_index] = next_point;
    }
  }
  IREE_ASSERT_EQ(next_point, builder->event_count);

  uint32_t* location_cursors = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->scratch_arena, location_range_count, sizeof(*location_cursors),
      (void**)&location_cursors));
  memcpy(location_cursors, location_range_starts,
         location_range_count * sizeof(*location_cursors));
  for (iree_host_size_t i = 0; i < builder->event_count; ++i) {
    const loom_low_allocation_clobber_event_t* event = &builder->events[i];
    const iree_host_size_t namespace_ordinal =
        loom_low_allocation_clobber_namespace_ordinal(event->storage_key,
                                                      maximum_alias_set_id);
    const loom_low_allocation_clobber_namespace_t* clobber_namespace =
        &namespaces[namespace_ordinal];
    const uint32_t range_index =
        clobber_namespace->location_range_start + event->location;
    const uint32_t point_index = location_cursors[range_index]++;
    IREE_ASSERT(point_index == location_range_starts[range_index] ||
                    points[point_index - 1u] <= event->point,
                "location clobbers must retain program-point order");
    points[point_index] = event->point;
    if (event->permits_definition) {
      iree_bitmap_set(permitted_definitions, point_index);
    }
  }

  for (uint32_t i = 0; i < descriptor_set->reg_class_count; ++i) {
    const uint32_t storage_key =
        loom_low_reg_class_storage_key(descriptor_set, (uint16_t)i);
    const iree_host_size_t namespace_ordinal =
        loom_low_allocation_clobber_namespace_ordinal(storage_key,
                                                      maximum_alias_set_id);
    const loom_low_allocation_clobber_namespace_t* clobber_namespace =
        &namespaces[namespace_ordinal];
    if (clobber_namespace->location_count != 0) {
      storage_by_reg_class[i] = (loom_low_allocation_clobber_storage_t){
          .location_range_start = clobber_namespace->location_range_start,
          .location_count = clobber_namespace->location_count,
      };
    }
  }

  *out_index = (loom_low_allocation_clobber_index_t){
      .points = points,
      .permitted_definitions = permitted_definitions,
      .location_range_starts = location_range_starts,
      .storage_by_reg_class = storage_by_reg_class,
  };
  return iree_ok_status();
}
