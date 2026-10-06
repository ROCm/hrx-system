// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/write_interference.h"

#include <string.h>

#include "iree/base/bitmap.h"
#include "loom/codegen/low/allocation/write_interference_flow.h"
#include "loom/codegen/low/read_retention.h"
#include "loom/ir/module.h"
#include "loom/target/registers.h"
#include "loom/util/index_set.h"
#include "loom/util/segmented_storage.h"

typedef struct loom_low_write_range_t {
  // First element in the owning array.
  uint32_t start;
  // Number of elements in the range.
  uint32_t count;
} loom_low_write_range_t;

typedef struct loom_low_write_value_t {
  // Required tied-storage origin in the local ordinal domain.
  loom_value_ordinal_t origin;
  // Fixed physical base, or UINT32_MAX when unconstrained.
  uint32_t fixed_base;
  // Collection metadata retires before the query index is constructed.
  union {
    // Register extents used by operand collection and completion
    // classification.
    struct {
      // Width of the value's linear register range.
      uint32_t width;
      // First retained-read unit bit, or UINT32_MAX if never retained.
      uint32_t retained_start;
    } construction;
    // Indexed write constraints and copy equations mentioning this origin.
    loom_low_write_range_t constraints;
  };
} loom_low_write_value_t;

enum loom_low_write_access_flag_bits_e {
  LOOM_LOW_WRITE_ACCESS_RETAIN = 1u << 0,
  LOOM_LOW_WRITE_ACCESS_WRITE = 1u << 1,
  LOOM_LOW_WRITE_ACCESS_INTERFERE = 1u << 2,
};

typedef struct loom_low_write_access_t {
  // Storage origin read or written by the event.
  loom_value_ordinal_t value;
  // Structural source origin, or INVALID for an unconditional access.
  loom_value_ordinal_t source;
  // First accessed unit in |value|.
  uint32_t offset;
  // First copied unit in |source|.
  uint32_t source_offset;
  // Number of accessed units.
  uint32_t count;
  // Next access at this point, or UINT32_MAX.
  uint32_t next;
  // Read, overwrite and hazardous-write effects.
  uint8_t flags;
} loom_low_write_access_t;

typedef struct loom_low_write_event_t {
  // First access at this write point, or UINT32_MAX.
  uint32_t access;
  // Shared immutable snapshot before this point's writes, after its read reset.
  loom_low_write_range_t retained;
  // A replacing read clears the incoming retained domain before writes.
  bool reset;
} loom_low_write_event_t;

typedef struct loom_low_write_retained_t {
  // Canonical storage origin whose read is retained.
  loom_value_ordinal_t value;
  // First retained unit within the origin.
  uint32_t offset;
  // Number of consecutive retained units.
  uint32_t count;
} loom_low_write_retained_t;

// Fresh retained ranges are nonempty. A zero retained count denotes a proved
// copy equation while its source remains present, or completed interference
// once the source is also retired. Component incidence needs only destination
// and retained origins after classification; those identities remain live.
typedef struct loom_low_write_constraint_t {
  // Canonical origin receiving the physical write.
  loom_value_ordinal_t destination;
  // First written unit within the destination.
  uint32_t destination_offset;
  // Retained input range, or count zero after equation/completion
  // classification.
  loom_low_write_retained_t retained;
  // Copy source, or INVALID for an unconditional or completed write.
  loom_value_ordinal_t source;
  // Write extent before completion, interpreted by source presence.
  union {
    // Number of units in an unconditional instruction write.
    uint32_t count;
    // Source unit for a copy, which always writes exactly one unit.
    uint32_t source_offset;
  } units;
} loom_low_write_constraint_t;

static uint32_t loom_low_write_constraint_write_count(
    const loom_low_write_constraint_t* row) {
  return row->source == LOOM_VALUE_ORDINAL_INVALID ? row->units.count : 1;
}

static bool loom_low_write_constraint_is_complete(
    const loom_low_write_constraint_t* row) {
  return row->source == LOOM_VALUE_ORDINAL_INVALID && row->retained.count == 0;
}

static void loom_low_write_constraint_complete(
    loom_low_write_constraint_t* row) {
  row->retained.count = 0;
  row->source = LOOM_VALUE_ORDINAL_INVALID;
}

// The constant index split keeps payloads bounded and rows directly
// addressable. Full-table scans resolve each segment once and advance through
// its populated prefix without repeating the directory lookup for each row.
#define LOOM_LOW_WRITE_CONSTRAINT_SEGMENT_SHIFT 7u
#define LOOM_LOW_WRITE_CONSTRAINTS_PER_SEGMENT \
  (1u << LOOM_LOW_WRITE_CONSTRAINT_SEGMENT_SHIFT)
#define LOOM_LOW_WRITE_CONSTRAINT_SEGMENT_MASK \
  (LOOM_LOW_WRITE_CONSTRAINTS_PER_SEGMENT - 1u)

static_assert((UINT32_MAX >> LOOM_LOW_WRITE_CONSTRAINT_SEGMENT_SHIFT) <
                  LOOM_SEGMENTED_STORAGE_MAX_SEGMENT_COUNT,
              "constraint segments must cover the full row index domain");

struct loom_low_allocation_write_interference_t {
  // Static local target read/write semantics.
  const loom_low_read_retention_t* rule;
  // Resolved linear physical register bank.
  uint16_t register_class;
  // Resetting register classes indexed by descriptor-local class ID.
  uint8_t* reset_classes;
  // Canonical storage and endpoint indices by local value ordinal.
  loom_low_write_value_t* values;
  // Number of entries in |values|.
  iree_host_size_t value_count;
  // Collected instruction effects and retained ranges by write point.
  loom_low_write_event_t* events;
  // Number of entries in |events|.
  uint32_t point_count;
  // Collected operand and structural accesses.
  loom_low_write_access_t* accesses;
  // Number of initialized accesses.
  iree_host_size_t access_count;
  // Logical capacity of |accesses|, at most UINT32_MAX.
  iree_host_size_t access_capacity;
  // Number of bits in the compact retained allocation-unit domain, a subset
  // of the producer-bounded unit-liveness domain.
  uint32_t retained_unit_count;
  // Immutable retained-range snapshots shared by events with unchanged state.
  loom_low_write_retained_t* retained;
  // Number of initialized retained ranges.
  iree_host_size_t retained_count;
  // Logical capacity of |retained|, at most UINT32_MAX.
  iree_host_size_t retained_capacity;
  // Stable bounded segments of final conditional physical-write exclusions.
  loom_segmented_storage_t constraints;
  // Exact number of rows in the finalized constraint table.
  iree_host_size_t constraint_count;
  // Immutable row addresses grouped by any participating storage origin.
  // The stable segments share this table's complete decision lifetime.
  const loom_low_write_constraint_t** indexed_constraints;
  // Reusable candidate-local equality propagation workspace.
  uint32_t* inferred_bases;
  // Origins reached by candidate-local zero-copy implications.
  loom_value_ordinal_t* inferred_origins;
};

static loom_low_write_constraint_t* loom_low_write_constraint_at(
    loom_low_allocation_write_interference_t* table, uint32_t index) {
  loom_low_write_constraint_t* rows =
      (loom_low_write_constraint_t*)loom_segmented_storage_segment(
          &table->constraints,
          index >> LOOM_LOW_WRITE_CONSTRAINT_SEGMENT_SHIFT);
  return &rows[index & LOOM_LOW_WRITE_CONSTRAINT_SEGMENT_MASK];
}

iree_status_t loom_low_allocation_write_interference_create(
    const loom_low_resolved_target_t* target,
    const loom_low_placement_table_t* placement,
    const loom_liveness_analysis_t* liveness, iree_arena_allocator_t* arena,
    loom_low_allocation_write_interference_t** out_interference) {
  *out_interference = NULL;
  const loom_low_read_retention_t* rule =
      target->target_facts != NULL ? target->target_facts->read_retention
                                   : NULL;
  if (rule == NULL ||
      target->target_facts->storage.snapshot.subgroup_size !=
          rule->subgroup_size ||
      liveness->operation_count == 0) {
    return iree_ok_status();
  }
  const uint32_t end_point =
      liveness->blocks[liveness->block_count - 1].end_point;
  if (end_point == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "retained-write events exceed u32 index capacity");
  }
  loom_low_allocation_write_interference_t* table = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*table), (void**)&table));
  *table = (loom_low_allocation_write_interference_t){
      .rule = rule,
      .register_class = LOOM_LOW_REGISTER_CLASS_ID_INVALID,
      .value_count = liveness->value_count,
      .point_count = end_point + 1,
  };
  const loom_low_descriptor_set_t* descriptors = target->descriptor_set;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, descriptors->reg_class_count, sizeof(*table->reset_classes),
      (void**)&table->reset_classes));
  memset(table->reset_classes, 0, descriptors->reg_class_count);
  for (uint16_t c = 0; c < descriptors->reg_class_count; ++c) {
    const iree_string_view_t name = loom_low_descriptor_set_string(
        descriptors, descriptors->reg_classes[c].name_string_ref);
    if (iree_string_view_equal(name, rule->register_class)) {
      table->register_class = c;
    }
    for (uint16_t r = 0; r < rule->reset_register_class_count; ++r) {
      table->reset_classes[c] |=
          iree_string_view_equal(name, rule->reset_register_classes[r]);
    }
  }
  IREE_ASSERT_NE(
      table->register_class, LOOM_LOW_REGISTER_CLASS_ID_INVALID,
      "target read-retention bank must belong to its representation");
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, table->value_count,
                                                 sizeof(*table->values),
                                                 (void**)&table->values));
  for (uint32_t v = 0; v < table->value_count; ++v) {
    const loom_liveness_interval_t* interval =
        loom_liveness_interval_for_value_ordinal(liveness, v);
    table->values[v] = (loom_low_write_value_t){
        .origin = placement->tied_storage_origins_by_value_ordinal != NULL
                      ? placement->tied_storage_origins_by_value_ordinal[v]
                      : v,
        .fixed_base = UINT32_MAX,
        .construction =
            {
                .width = interval != NULL ? interval->unit_count : 0,
                .retained_start = UINT32_MAX,
            },
    };
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, table->point_count,
                                                 sizeof(*table->events),
                                                 (void**)&table->events));
  for (uint32_t p = 0; p < table->point_count; ++p) {
    table->events[p] = (loom_low_write_event_t){.access = UINT32_MAX};
  }
  *out_interference = table;
  return iree_ok_status();
}

// Called only for full construction arrays. Limiting logical capacity makes
// the next growth fail before an append can use UINT32_MAX as a row index.
static iree_status_t loom_low_write_grow_array(iree_arena_allocator_t* arena,
                                               iree_host_size_t count,
                                               iree_host_size_t element_size,
                                               iree_host_size_t* inout_capacity,
                                               void** inout_values) {
  if (count == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "retained-write rows exceed u32 index capacity");
  }
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      arena, count, count + 1, element_size, inout_capacity, inout_values));
  *inout_capacity = iree_min(*inout_capacity, UINT32_MAX);
  return iree_ok_status();
}

static iree_status_t loom_low_write_append_access(
    loom_low_allocation_write_interference_t* table, uint32_t point,
    loom_low_write_access_t access, iree_arena_allocator_t* arena) {
  if (table->access_count == table->access_capacity) {
    IREE_RETURN_IF_ERROR(loom_low_write_grow_array(
        arena, table->access_count, sizeof(*table->accesses),
        &table->access_capacity, (void**)&table->accesses));
  }
  access.next = table->events[point].access;
  table->events[point].access = (uint32_t)table->access_count;
  table->accesses[table->access_count++] = access;
  return iree_ok_status();
}

iree_status_t loom_low_allocation_write_interference_note_operand(
    loom_low_allocation_write_interference_t* table,
    const loom_local_value_domain_t* value_domain,
    const loom_low_descriptor_set_t* descriptors,
    const loom_low_descriptor_t* descriptor, const loom_op_t* op,
    uint16_t operand_index, uint32_t point, iree_arena_allocator_t* arena) {
  if (table == NULL) {
    return iree_ok_status();
  }
  const loom_low_operand_t* operand =
      &descriptors->operands[descriptor->operand_start + operand_index];
  const loom_low_instruction_class_flags_t classes =
      loom_low_descriptor_set_descriptor_view(descriptors, descriptor)
          ->instruction_class_flags;
  const bool reader = iree_any_bit_set(classes, table->rule->reader_classes);
  if (operand->source_value_index == LOOM_LOW_ID_NONE) {
    if (reader &&
        iree_any_bit_set(operand->flags, LOOM_LOW_OPERAND_FLAG_STATE_READ)) {
      const uint16_t register_class =
          descriptors->reg_class_alts[operand->reg_class_alt_start]
              .reg_class_id;
      table->events[point].reset |= table->reset_classes[register_class] != 0;
    }
    return iree_ok_status();
  }
  const bool result = operand_index < descriptor->result_count;
  if (!result && !loom_low_descriptor_operand_maps_to_packet_operand(
                     descriptors, descriptor, operand_index)) {
    return iree_ok_status();
  }
  const uint16_t end =
      !result &&
              iree_any_bit_set(operand->flags, LOOM_LOW_OPERAND_FLAG_VARIADIC)
          ? op->operand_count
          : operand->source_value_index + 1;
  for (uint16_t i = operand->source_value_index; i < end; ++i) {
    const loom_value_id_t value =
        result ? loom_op_const_results(op)[i] : loom_op_const_operands(op)[i];
    const loom_type_t type =
        loom_module_value_type(value_domain->module, value);
    if (!loom_low_type_is_register(type)) {
      continue;
    }
    const uint16_t register_class = loom_low_register_type_class_id(type);
    if (reader && !result) {
      table->events[point].reset |= table->reset_classes[register_class] != 0;
    }
    if (register_class != table->register_class) {
      continue;
    }
    const loom_value_ordinal_t ordinal =
        loom_local_value_domain_ordinal(value_domain, value);
    const loom_value_ordinal_t origin = table->values[ordinal].origin;
    uint8_t flags = 0;
    if (result) {
      flags = LOOM_LOW_WRITE_ACCESS_WRITE;
      if (iree_any_bit_set(classes, table->rule->writer_classes)) {
        flags |= LOOM_LOW_WRITE_ACCESS_INTERFERE;
      }
    } else if (reader && operand->role == table->rule->retained_operand_role) {
      flags = LOOM_LOW_WRITE_ACCESS_RETAIN;
      if (table->values[origin].construction.retained_start == UINT32_MAX) {
        table->values[origin].construction.retained_start =
            table->retained_unit_count;
        table->retained_unit_count += table->values[origin].construction.width;
      }
    }
    if (flags != 0) {
      IREE_RETURN_IF_ERROR(loom_low_write_append_access(
          table, point,
          (loom_low_write_access_t){
              .value = origin,
              .source = LOOM_VALUE_ORDINAL_INVALID,
              .count = table->values[ordinal].construction.width,
              .flags = flags},
          arena));
    }
  }
  return iree_ok_status();
}

void loom_low_allocation_write_interference_note_fixed(
    loom_low_allocation_write_interference_t* table,
    loom_value_ordinal_t ordinal,
    const loom_low_allocation_assignment_t* assignment) {
  if (table != NULL &&
      loom_low_allocation_assignment_is_physical_register_class(
          assignment, table->register_class)) {
    table->values[table->values[ordinal].origin].fixed_base =
        assignment->location_base;
  }
}

static bool loom_low_write_known_delta(
    const loom_low_allocation_write_interference_t* table,
    loom_value_ordinal_t destination, loom_value_ordinal_t source,
    int64_t* out_delta) {
  if (destination == source) {
    *out_delta = 0;
    return true;
  }
  const uint32_t destination_base = table->values[destination].fixed_base;
  const uint32_t source_base = table->values[source].fixed_base;
  if (destination_base == UINT32_MAX || source_base == UINT32_MAX) {
    return false;
  }
  *out_delta = (int64_t)destination_base - source_base;
  return true;
}

// Known writes cut only the overwritten units. For ALU writes this boundary
// promises the physical planner's post-write dependency; memory results carry
// their ordinary asynchronous completion dependency before reuse.
// Returns whether a previously materialized snapshot must be rebuilt.
static bool loom_low_write_transfer(
    const loom_low_allocation_write_interference_t* table, uint32_t point,
    const loom_low_write_retained_t* units, iree_bitmap_t active) {
  bool invalidated = false;
  const loom_low_write_event_t* event = &table->events[point];
  for (uint32_t a = event->access; a != UINT32_MAX;
       a = table->accesses[a].next) {
    const loom_low_write_access_t* access = &table->accesses[a];
    if (!iree_any_bit_set(access->flags, LOOM_LOW_WRITE_ACCESS_WRITE)) {
      continue;
    }
    if (access->source != LOOM_VALUE_ORDINAL_INVALID) {
      int64_t delta = 0;
      if (!loom_low_write_known_delta(table, access->value, access->source,
                                      &delta) ||
          delta + access->offset == access->source_offset) {
        continue;
      }
    }
    for (iree_host_size_t bit = iree_bitmap_find_first_set(active, 0);
         bit < active.bit_count;
         bit = iree_bitmap_find_first_set(active, bit + 1)) {
      const loom_low_write_retained_t* unit = &units[bit];
      int64_t delta = 0;
      if (loom_low_write_known_delta(table, access->value, unit->value,
                                     &delta) &&
          unit->offset >= delta + access->offset &&
          unit->offset < delta + access->offset + access->count) {
        iree_bitmap_reset(active, bit);
        invalidated = true;
      }
    }
  }
  for (uint32_t a = event->access; a != UINT32_MAX;
       a = table->accesses[a].next) {
    const loom_low_write_access_t* access = &table->accesses[a];
    if (iree_any_bit_set(access->flags, LOOM_LOW_WRITE_ACCESS_RETAIN)) {
      iree_bitmap_set_span(
          active, table->values[access->value].construction.retained_start,
          access->count);
      invalidated = true;
    }
  }
  return invalidated;
}

static iree_status_t loom_low_write_snapshot(
    loom_low_allocation_write_interference_t* table,
    const loom_low_write_retained_t* units, iree_bitmap_t active,
    iree_arena_allocator_t* arena, loom_low_write_range_t* out_snapshot) {
  *out_snapshot =
      (loom_low_write_range_t){.start = (uint32_t)table->retained_count};
  for (iree_host_size_t bit = iree_bitmap_find_first_set(active, 0);
       bit < active.bit_count;) {
    loom_low_write_retained_t range = units[bit++];
    while (bit < active.bit_count && iree_bitmap_test(active, bit) &&
           units[bit].value == range.value &&
           units[bit].offset == range.offset + range.count) {
      ++range.count;
      ++bit;
    }
    if (table->retained_count == table->retained_capacity) {
      IREE_RETURN_IF_ERROR(loom_low_write_grow_array(
          arena, table->retained_count, sizeof(*table->retained),
          &table->retained_capacity, (void**)&table->retained));
    }
    table->retained[table->retained_count++] = range;
    ++out_snapshot->count;
    if (bit < active.bit_count) {
      bit = iree_bitmap_find_first_set(active, bit);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_write_count_constraints(
    loom_low_allocation_write_interference_t* table, uint32_t point) {
  const loom_low_write_event_t* event = &table->events[point];
  if (event->retained.count == 0) {
    return iree_ok_status();
  }
  for (uint32_t a = event->access; a != UINT32_MAX;
       a = table->accesses[a].next) {
    const loom_low_write_access_t* access = &table->accesses[a];
    if (iree_any_bit_set(access->flags, LOOM_LOW_WRITE_ACCESS_INTERFERE)) {
      const uint32_t write_count =
          access->source != LOOM_VALUE_ORDINAL_INVALID ? access->count : 1;
      const uint64_t row_count = (uint64_t)event->retained.count * write_count;
      if (row_count > UINT32_MAX - table->constraint_count) {
        return iree_make_status(
            IREE_STATUS_RESOURCE_EXHAUSTED,
            "retained-write table exceeds u32 index capacity");
      }
      table->constraint_count += row_count;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_write_allocate_constraints(
    loom_low_allocation_write_interference_t* table,
    iree_arena_allocator_t* arena) {
  if (table->constraint_count == 0) {
    return iree_ok_status();
  }
  // A smaller table has only one segment, so its payload can be exact without
  // changing the constant logical index split used during construction.
  const uint32_t row_count =
      iree_min(table->constraint_count, LOOM_LOW_WRITE_CONSTRAINTS_PER_SEGMENT);
  loom_segmented_storage_initialize(
      row_count * sizeof(loom_low_write_constraint_t),
      iree_alignof(loom_low_write_constraint_t), &table->constraints);
  const uint32_t segment_count = ((table->constraint_count - 1) >>
                                  LOOM_LOW_WRITE_CONSTRAINT_SEGMENT_SHIFT) +
                                 1;
  for (uint32_t segment_index = 0; segment_index < segment_count;
       ++segment_index) {
    void* segment = NULL;
    IREE_RETURN_IF_ERROR(
        loom_segmented_storage_append(&table->constraints, arena, &segment));
  }
  return iree_ok_status();
}

static uint32_t loom_low_write_record_point(
    loom_low_allocation_write_interference_t* table, uint32_t point,
    uint32_t next_constraint) {
  const loom_low_write_event_t* event = &table->events[point];
  for (uint32_t r = 0; r < event->retained.count; ++r) {
    const loom_low_write_retained_t range =
        table->retained[event->retained.start + r];
    for (uint32_t a = event->access; a != UINT32_MAX;
         a = table->accesses[a].next) {
      const loom_low_write_access_t* access = &table->accesses[a];
      if (!iree_any_bit_set(access->flags, LOOM_LOW_WRITE_ACCESS_INTERFERE)) {
        continue;
      }
      const bool copy = access->source != LOOM_VALUE_ORDINAL_INVALID;
      const uint32_t row_count = copy ? access->count : 1;
      for (uint32_t unit = 0; unit < row_count; ++unit) {
        loom_low_write_constraint_t* row =
            loom_low_write_constraint_at(table, next_constraint++);
        *row = (loom_low_write_constraint_t){
            .destination = access->value,
            .destination_offset = access->offset + unit,
            .retained = range,
            .source = access->source,
        };
        if (copy) {
          row->units.source_offset = access->source_offset + unit;
        } else {
          row->units.count = access->count;
        }
      }
    }
  }
  return next_constraint;
}

// Weighted identities record base(value) - base(parent). Equations arise only
// from fixed bindings or a structural copy forced to alias its retained input;
// candidate-search failure never contributes an equation.
typedef struct loom_low_write_identity_t {
  // Canonical parent storage origin.
  loom_value_ordinal_t parent;
  // Number of members when this entry is a root.
  uint32_t size;
  // Signed location difference from the parent.
  int64_t offset;
} loom_low_write_identity_t;

static loom_low_write_identity_t loom_low_write_identity_root(
    loom_low_write_identity_t* identities, loom_value_ordinal_t value) {
  loom_low_write_identity_t result = identities[value];
  while (identities[result.parent].parent != result.parent) {
    result.offset += identities[result.parent].offset;
    result.parent = identities[result.parent].parent;
  }
  int64_t remaining = result.offset;
  while (identities[value].parent != value) {
    const loom_low_write_identity_t previous = identities[value];
    identities[value].parent = result.parent;
    identities[value].offset = remaining;
    remaining -= previous.offset;
    value = previous.parent;
  }
  return result;
}

static bool loom_low_write_ranges_overlap(int64_t difference,
                                          uint32_t destination_count,
                                          uint32_t retained_count) {
  return difference < retained_count && difference + destination_count > 0;
}

typedef struct loom_low_write_copy_unit_t {
  // Canonical storage origin containing this unit.
  loom_value_ordinal_t value;
  // Unit position within its origin.
  uint32_t offset;
  // Forwarded source unit index, or UINT32_MAX at an identity boundary.
  uint32_t source;
  // Write point of the forwarding operation.
  uint32_t point;
} loom_low_write_copy_unit_t;

// A forced overwrite cannot become a no-op if its source ultimately forwards
// another known unit while the overwritten mask unit is retained throughout
// the chain. Classify that physical sink before deriving zero-copy equations;
// otherwise the contradiction can be blamed on an earlier avoidable copy.
static iree_status_t loom_low_write_classify_copy_chains(
    loom_low_allocation_write_interference_t* table,
    const loom_liveness_analysis_t* liveness,
    const loom_low_placement_table_t* placement, uint32_t* inout_seed_count,
    iree_arena_allocator_t* scratch_arena) {
  uint32_t* starts = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, table->value_count, sizeof(*starts), (void**)&starts));
  uint32_t unit_count = 0;
  for (uint32_t v = 0; v < table->value_count; ++v) {
    starts[v] = UINT32_MAX;
    const loom_liveness_interval_t* interval =
        loom_liveness_interval_for_value_ordinal(liveness, v);
    if (interval != NULL &&
        interval->value_class.type_kind == LOOM_TYPE_REGISTER &&
        interval->value_class.register_class_id == table->register_class &&
        table->values[v].origin == v) {
      starts[v] = unit_count;
      unit_count += table->values[v].construction.width;
    }
  }
  loom_low_write_copy_unit_t* units = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, unit_count, sizeof(*units), (void**)&units));
  for (uint32_t v = 0; v < table->value_count; ++v) {
    if (starts[v] == UINT32_MAX) {
      continue;
    }
    for (uint32_t u = 0; u < table->values[v].construction.width; ++u) {
      units[starts[v] + u] = (loom_low_write_copy_unit_t){
          .value = v, .offset = u, .source = UINT32_MAX};
    }
  }
  for (uint32_t r = 0; r < placement->relation_count; ++r) {
    const loom_low_placement_relation_t* relation = &placement->relations[r];
    if (relation->cause < LOOM_LOW_PLACEMENT_CAUSE_LOW_COPY ||
        relation->cause > LOOM_LOW_PLACEMENT_CAUSE_LOW_CONCAT ||
        starts[relation->result_ordinal] == UINT32_MAX ||
        starts[relation->source_ordinal] == UINT32_MAX) {
      continue;
    }
    for (uint32_t u = 0; u < relation->unit_count; ++u) {
      loom_low_write_copy_unit_t* unit =
          &units[starts[relation->result_ordinal] +
                 relation->result_unit_offset + u];
      unit->source =
          starts[relation->source_ordinal] + relation->source_unit_offset + u;
      unit->point = relation->write_point;
    }
  }
  for (uint32_t c = 0; c < table->constraint_count;) {
    const uint32_t segment_end =
        c + iree_min(table->constraint_count - c,
                     LOOM_LOW_WRITE_CONSTRAINTS_PER_SEGMENT);
    loom_low_write_constraint_t* row = loom_low_write_constraint_at(table, c);
    for (; c < segment_end; ++c, ++row) {
      int64_t delta = 0;
      if (row->source == LOOM_VALUE_ORDINAL_INVALID ||
          !loom_low_write_known_delta(table, row->destination,
                                      row->retained.value, &delta) ||
          !loom_low_write_ranges_overlap(
              delta + row->destination_offset - row->retained.offset, 1,
              row->retained.count)) {
        continue;
      }
      const int64_t position = delta + row->destination_offset;
      uint32_t current = starts[row->source] + row->units.source_offset;
      while (current != UINT32_MAX) {
        const loom_low_write_copy_unit_t* unit = &units[current];
        if (loom_low_write_known_delta(table, unit->value, row->retained.value,
                                       &delta)) {
          if (delta + unit->offset != position) {
            loom_low_write_constraint_complete(row);
            --*inout_seed_count;
          }
          break;
        }
        if (unit->source == UINT32_MAX) {
          break;
        }
        const loom_low_write_range_t retained =
            table->events[unit->point].retained;
        bool active = false;
        for (uint32_t i = 0; i < retained.count; ++i) {
          const loom_low_write_retained_t* range =
              &table->retained[retained.start + i];
          active |= range->value == row->retained.value &&
                    position >= range->offset &&
                    position < range->offset + range->count;
        }
        if (!active) {
          break;
        }
        current = unit->source;
      }
    }
  }
  return iree_ok_status();
}

// Only a forced copy can connect previously independent storage identities.
// Without that seed, direct fixed/same-origin overwrites are the complete
// result.
static uint32_t loom_low_write_seed_completion(
    loom_low_allocation_write_interference_t* table) {
  uint32_t seed_count = 0;
  for (uint32_t c = 0; c < table->constraint_count;) {
    const uint32_t segment_end =
        c + iree_min(table->constraint_count - c,
                     LOOM_LOW_WRITE_CONSTRAINTS_PER_SEGMENT);
    loom_low_write_constraint_t* row = loom_low_write_constraint_at(table, c);
    for (; c < segment_end; ++c, ++row) {
      int64_t delta = 0;
      if (!loom_low_write_known_delta(table, row->destination,
                                      row->retained.value, &delta) ||
          !loom_low_write_ranges_overlap(
              delta + row->destination_offset - row->retained.offset,
              loom_low_write_constraint_write_count(row),
              row->retained.count)) {
        continue;
      }
      if (row->source == LOOM_VALUE_ORDINAL_INVALID) {
        loom_low_write_constraint_complete(row);
      } else {
        ++seed_count;
      }
    }
  }
  return seed_count;
}

typedef struct loom_low_write_completion_t {
  // Constraint owner; retired value metadata holds endpoint incidence ranges.
  loom_low_allocation_write_interference_t* table;
  // Weighted union forest, with component sizes at roots.
  loom_low_write_identity_t* identities;
  // Next member in a root-headed component list, or UINT32_MAX.
  uint32_t* next_members;
  // Last member of each root's component list.
  uint32_t* tails;
  // Initially disconnected copy rows indexed at destination and retained value.
  uint32_t* incidences;
  // Hierarchical pending-row bits, selected in cyclic source order.
  struct {
    // Leaf membership followed by summaries of nonempty words.
    uint64_t* words;
    // Fixed row-domain membership and summary layout.
    loom_index_set_layout_t layout;
    // Last dequeued row, whose pending bit has been cleared.
    uint32_t cursor;
  } pending;
} loom_low_write_completion_t;

// Only the current and following scan rounds can be pending. Selecting in
// cyclic source order preserves the original closure's witness order. Each row
// becomes connected once, so the last cursor's bit remains clear after it is
// retired.
static uint32_t loom_low_write_completion_dequeue(
    loom_low_write_completion_t* completion) {
  const uint32_t key = loom_index_set_select(&completion->pending.layout,
                                             completion->pending.words,
                                             completion->pending.cursor);
  if (key != LOOM_INDEX_SET_NONE) {
    loom_index_set_erase(&completion->pending.layout, completion->pending.words,
                         key);
    completion->pending.cursor = key;
  }
  return key;
}

static iree_status_t loom_low_write_completion_initialize(
    loom_low_allocation_write_interference_t* table,
    iree_arena_allocator_t* scratch_arena,
    loom_low_write_completion_t* completion) {
  *completion = (loom_low_write_completion_t){
      .table = table,
      .next_members = table->inferred_origins,
      .tails = table->inferred_bases,
  };
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, table->value_count, sizeof(*completion->identities),
      (void**)&completion->identities));
  loom_low_write_identity_t* identities = completion->identities;
  loom_value_ordinal_t anchor = LOOM_VALUE_ORDINAL_INVALID;
  for (uint32_t v = 0; v < table->value_count; ++v) {
    identities[v] = (loom_low_write_identity_t){.parent = v, .size = 1};
    completion->next_members[v] = UINT32_MAX;
    completion->tails[v] = v;
    table->values[v].constraints = (loom_low_write_range_t){0};
    if (table->values[v].fixed_base != UINT32_MAX) {
      if (anchor == LOOM_VALUE_ORDINAL_INVALID) {
        anchor = v;
      } else {
        identities[v].parent = anchor;
        identities[v].offset = (int64_t)table->values[v].fixed_base -
                               table->values[anchor].fixed_base;
        ++identities[anchor].size;
        completion->next_members[completion->tails[anchor]] = v;
        completion->tails[anchor] = v;
      }
    }
  }
  for (uint32_t c = 0; c < table->constraint_count;) {
    const uint32_t segment_end =
        c + iree_min(table->constraint_count - c,
                     LOOM_LOW_WRITE_CONSTRAINTS_PER_SEGMENT);
    const loom_low_write_constraint_t* row =
        loom_low_write_constraint_at(table, c);
    for (; c < segment_end; ++c, ++row) {
      if (row->source == LOOM_VALUE_ORDINAL_INVALID ||
          row->retained.count == 0) {
        continue;
      }
      if (identities[row->destination].parent !=
          identities[row->retained.value].parent) {
        ++table->values[row->destination].constraints.count;
        ++table->values[row->retained.value].constraints.count;
      }
    }
  }
  uint64_t incidence_count = 0;
  for (uint32_t v = 0; v < table->value_count; ++v) {
    loom_low_write_range_t* range = &table->values[v].constraints;
    if (range->count > UINT32_MAX - incidence_count) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "retained-write table exceeds u32 index capacity");
    }
    range->start = (uint32_t)incidence_count;
    incidence_count += range->count;
    range->count = 0;
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, incidence_count, sizeof(*completion->incidences),
      (void**)&completion->incidences));
  completion->pending.layout =
      loom_index_set_calculate_layout((uint32_t)table->constraint_count);
  const uint32_t word_count = completion->pending.layout.word_count;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, word_count, sizeof(*completion->pending.words),
      (void**)&completion->pending.words));
  memset(completion->pending.words, 0,
         word_count * sizeof(*completion->pending.words));
  for (uint32_t c = 0; c < table->constraint_count;) {
    const uint32_t segment_end =
        c + iree_min(table->constraint_count - c,
                     LOOM_LOW_WRITE_CONSTRAINTS_PER_SEGMENT);
    const loom_low_write_constraint_t* row =
        loom_low_write_constraint_at(table, c);
    for (; c < segment_end; ++c, ++row) {
      if (row->source == LOOM_VALUE_ORDINAL_INVALID ||
          row->retained.count == 0) {
        continue;
      }
      const loom_low_write_identity_t destination =
          identities[row->destination];
      const loom_low_write_identity_t retained =
          identities[row->retained.value];
      if (destination.parent == retained.parent) {
        if (loom_low_write_ranges_overlap(
                destination.offset + row->destination_offset - retained.offset -
                    row->retained.offset,
                1, row->retained.count)) {
          loom_index_set_insert(&completion->pending.layout,
                                completion->pending.words, c);
        }
      } else {
        loom_low_write_range_t* destination_range =
            &table->values[row->destination].constraints;
        completion->incidences[destination_range->start +
                               destination_range->count++] = c;
        loom_low_write_range_t* retained_range =
            &table->values[row->retained.value].constraints;
        completion
            ->incidences[retained_range->start + retained_range->count++] = c;
      }
    }
  }
  return iree_ok_status();
}

static void loom_low_write_completion_merge(
    loom_low_write_completion_t* completion,
    loom_low_write_identity_t destination, loom_low_write_identity_t source,
    int64_t required) {
  loom_low_allocation_write_interference_t* table = completion->table;
  loom_low_write_identity_t* identities = completion->identities;
  uint32_t loser = destination.parent;
  uint32_t winner = source.parent;
  int64_t offset = required + source.offset - destination.offset;
  if (identities[loser].size > identities[winner].size) {
    loser = source.parent;
    winner = destination.parent;
    offset = -offset;
  }
  // Only relationships crossing this component boundary become connected now.
  // Visit the smaller list before linking its root, so already-connected rows
  // are never enqueued again. Each member participates at most log2(V) times.
  for (uint32_t v = loser; v != UINT32_MAX; v = completion->next_members[v]) {
    const loom_low_write_range_t range = table->values[v].constraints;
    for (uint32_t i = 0; i < range.count; ++i) {
      const uint32_t c = completion->incidences[range.start + i];
      const loom_low_write_constraint_t* row =
          loom_low_write_constraint_at(table, c);
      const uint32_t other =
          row->destination == v ? row->retained.value : row->destination;
      if (loom_low_write_identity_root(identities, other).parent == winner) {
        loom_index_set_insert(&completion->pending.layout,
                              completion->pending.words, c);
      }
    }
  }
  identities[loser].parent = winner;
  identities[loser].offset = offset;
  identities[winner].size += identities[loser].size;
  completion->next_members[completion->tails[winner]] = loser;
  completion->tails[winner] = completion->tails[loser];
}

static iree_status_t loom_low_write_classify_completion(
    loom_low_allocation_write_interference_t* table,
    const loom_liveness_analysis_t* liveness,
    const loom_low_placement_table_t* placement,
    iree_arena_allocator_t* scratch_arena) {
  uint32_t seed_count = loom_low_write_seed_completion(table);
  if (seed_count == 0) {
    return iree_ok_status();
  }
  const iree_arena_checkpoint_t chains_checkpoint =
      iree_arena_checkpoint_save(scratch_arena);
  IREE_RETURN_IF_ERROR(loom_low_write_classify_copy_chains(
      table, liveness, placement, &seed_count, scratch_arena));
  iree_arena_checkpoint_restore(&chains_checkpoint);
  if (seed_count == 0) {
    return iree_ok_status();
  }

  loom_low_write_completion_t completion;
  // Completion indexing precedes equation classification, so every remaining
  // nonempty retained range still describes an ordinary write.
  IREE_RETURN_IF_ERROR(
      loom_low_write_completion_initialize(table, scratch_arena, &completion));
  loom_low_write_identity_t* identities = completion.identities;
  for (uint32_t cursor = loom_low_write_completion_dequeue(&completion);
       cursor != UINT32_MAX;
       cursor = loom_low_write_completion_dequeue(&completion)) {
    loom_low_write_constraint_t* row =
        loom_low_write_constraint_at(table, cursor);
    const loom_low_write_identity_t destination =
        loom_low_write_identity_root(identities, row->destination);
    const loom_low_write_identity_t retained =
        loom_low_write_identity_root(identities, row->retained.value);
    if (!loom_low_write_ranges_overlap(
            destination.offset + row->destination_offset - retained.offset -
                row->retained.offset,
            1, row->retained.count)) {
      continue;
    }
    const loom_low_write_identity_t source =
        loom_low_write_identity_root(identities, row->source);
    const int64_t required =
        (int64_t)row->units.source_offset - row->destination_offset;
    if (destination.parent == source.parent) {
      if (destination.offset - source.offset != required) {
        loom_low_write_constraint_complete(row);
      }
    } else {
      row->retained.count = 0;
      loom_low_write_completion_merge(&completion, destination, source,
                                      required);
    }
  }
  for (uint32_t c = 0; c < table->constraint_count;) {
    const uint32_t segment_end =
        c + iree_min(table->constraint_count - c,
                     LOOM_LOW_WRITE_CONSTRAINTS_PER_SEGMENT);
    loom_low_write_constraint_t* row = loom_low_write_constraint_at(table, c);
    for (; c < segment_end; ++c, ++row) {
      if (row->source != LOOM_VALUE_ORDINAL_INVALID ||
          row->retained.count == 0) {
        continue;
      }
      const loom_low_write_identity_t destination =
          loom_low_write_identity_root(identities, row->destination);
      const loom_low_write_identity_t retained =
          loom_low_write_identity_root(identities, row->retained.value);
      if (destination.parent == retained.parent &&
          loom_low_write_ranges_overlap(
              destination.offset + row->destination_offset - retained.offset -
                  row->retained.offset,
              row->units.count, row->retained.count)) {
        loom_low_write_constraint_complete(row);
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_write_index_constraints(
    loom_low_allocation_write_interference_t* table,
    iree_arena_allocator_t* arena) {
  for (uint32_t v = 0; v < table->value_count; ++v) {
    table->values[v].constraints = (loom_low_write_range_t){0};
  }
  for (uint32_t c = 0; c < table->constraint_count;) {
    const uint32_t segment_end =
        c + iree_min(table->constraint_count - c,
                     LOOM_LOW_WRITE_CONSTRAINTS_PER_SEGMENT);
    const loom_low_write_constraint_t* row =
        loom_low_write_constraint_at(table, c);
    for (; c < segment_end; ++c, ++row) {
      if (loom_low_write_constraint_is_complete(row)) {
        continue;
      }
      if (row->retained.count == 0) {
        ++table->values[row->destination].constraints.count;
        ++table->values[row->source].constraints.count;
      } else {
        ++table->values[row->destination].constraints.count;
        if (row->retained.value != row->destination) {
          ++table->values[row->retained.value].constraints.count;
        }
        if (row->source != LOOM_VALUE_ORDINAL_INVALID &&
            row->source != row->destination &&
            row->source != row->retained.value) {
          ++table->values[row->source].constraints.count;
        }
      }
    }
  }
  uint64_t count = 0;
  for (uint32_t v = 0; v < table->value_count; ++v) {
    loom_low_write_range_t* range = &table->values[v].constraints;
    range->start = (uint32_t)count;
    count += range->count;
    range->count = 0;
  }
  if (count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "retained-write table exceeds u32 index capacity");
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, count, sizeof(*table->indexed_constraints),
      (void**)&table->indexed_constraints));
  for (uint32_t c = 0; c < table->constraint_count;) {
    const uint32_t segment_end =
        c + iree_min(table->constraint_count - c,
                     LOOM_LOW_WRITE_CONSTRAINTS_PER_SEGMENT);
    const loom_low_write_constraint_t* row =
        loom_low_write_constraint_at(table, c);
    for (; c < segment_end; ++c, ++row) {
      if (loom_low_write_constraint_is_complete(row)) {
        continue;
      }
      const loom_value_ordinal_t values[] = {
          row->destination,
          row->retained.count == 0 ? row->source : row->retained.value,
          row->retained.count == 0 ? LOOM_VALUE_ORDINAL_INVALID : row->source,
      };
      for (uint32_t i = 0; i < IREE_ARRAYSIZE(values); ++i) {
        if (values[i] == LOOM_VALUE_ORDINAL_INVALID ||
            (i > 0 && values[i] == values[0]) ||
            (i > 1 && values[i] == values[1])) {
          continue;
        }
        loom_low_write_range_t* range = &table->values[values[i]].constraints;
        table->indexed_constraints[range->start + range->count++] = row;
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_write_finalize_impl(
    loom_low_allocation_write_interference_t* table,
    const loom_liveness_analysis_t* liveness, const loom_cfg_graph_t* cfg_graph,
    const loom_low_placement_table_t* placement, iree_arena_allocator_t* arena,
    iree_arena_allocator_t* scratch_arena) {
  for (iree_host_size_t r = 0; r < placement->relation_count; ++r) {
    const loom_low_placement_relation_t* relation = &placement->relations[r];
    if (relation->cause == LOOM_LOW_PLACEMENT_CAUSE_TIED_RESULT ||
        relation->cause >= LOOM_LOW_PLACEMENT_CAUSE_SCHEDULE_PAIR_AFFINITY) {
      continue;
    }
    const loom_liveness_interval_t* interval =
        loom_liveness_interval_for_value_ordinal(liveness,
                                                 relation->result_ordinal);
    if (interval->value_class.type_kind != LOOM_TYPE_REGISTER ||
        interval->value_class.register_class_id != table->register_class) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_write_append_access(
        table, relation->write_point,
        (loom_low_write_access_t){
            .value = table->values[relation->result_ordinal].origin,
            .source = table->values[relation->source_ordinal].origin,
            .offset = relation->result_unit_offset,
            .source_offset = relation->source_unit_offset,
            .count = relation->unit_count,
            .flags =
                LOOM_LOW_WRITE_ACCESS_WRITE | LOOM_LOW_WRITE_ACCESS_INTERFERE,
        },
        arena));
  }
  const iree_arena_checkpoint_t construction_checkpoint =
      iree_arena_checkpoint_save(scratch_arena);
  loom_low_write_retained_t* units = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(scratch_arena, table->retained_unit_count,
                                sizeof(*units), (void**)&units));
  for (uint32_t v = 0; v < table->value_count; ++v) {
    const loom_low_write_value_t* value = &table->values[v];
    if (value->construction.retained_start == UINT32_MAX) {
      continue;
    }
    for (uint32_t u = 0; u < value->construction.width; ++u) {
      units[value->construction.retained_start + u] =
          (loom_low_write_retained_t){v, u, 1};
    }
  }
  loom_low_write_flow_t flow;
  IREE_RETURN_IF_ERROR(loom_low_write_flow_build(
      liveness, cfg_graph, table->point_count, scratch_arena, &flow));
  const iree_host_size_t word_count =
      iree_bitmap_calculate_words(table->retained_unit_count);
  const iree_host_size_t byte_count = word_count * sizeof(uint64_t);
  uint64_t* incoming = NULL;
  uint64_t* active_words = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, flow.block_count, byte_count, (void**)&incoming));
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(scratch_arena, byte_count, (void**)&active_words));
  memset(incoming, 0, flow.block_count * byte_count);
  iree_bitmap_t active = {.bit_count = table->retained_unit_count,
                          .words = active_words};
  // All transfers are monotone gen/kill maps. Only changed incoming sets
  // require another evaluation. Seed every span: local reads can generate
  // retained state even without incoming bits or reachable predecessors.
  uint32_t* pending_next = flow.worklist;
  for (uint32_t b = 0; b < flow.block_count; ++b) {
    pending_next[b] = b + 1;
  }
  uint32_t pending_head = 0;
  uint32_t pending_tail = flow.block_count - 1;
  pending_next[pending_tail] = UINT32_MAX;
  while (pending_head != UINT32_MAX) {
    const uint32_t b = pending_head;
    pending_head = pending_next[b];
    // A self-link marks absence; queued links name a distinct next span or
    // UINT32_MAX at the tail. Removing membership before transfer permits
    // a self-loop to enqueue this span again without a separate bitmap.
    pending_next[b] = b;
    const loom_low_write_flow_block_t* block = &flow.blocks[b];
    memcpy(active_words, incoming + b * word_count, byte_count);
    for (uint32_t p = block->begin; p < block->end; ++p) {
      if (table->events[p].reset) {
        iree_bitmap_reset_all(active);
      }
      loom_low_write_transfer(table, p, units, active);
    }
    for (uint32_t s = 0; s < block->successor_count; ++s) {
      const uint32_t successor = flow.successors[block->successor_start + s];
      uint64_t* next = incoming + successor * word_count;
      bool changed = false;
      for (iree_host_size_t w = 0; w < word_count; ++w) {
        const uint64_t added = active_words[w] & ~next[w];
        next[w] |= added;
        changed |= added != 0;
      }
      if (changed && pending_next[successor] == successor) {
        if (pending_head == UINT32_MAX) {
          pending_head = successor;
        } else {
          pending_next[pending_tail] = successor;
        }
        pending_tail = successor;
        pending_next[successor] = UINT32_MAX;
      }
    }
  }
  for (uint32_t b = 0; b < flow.block_count; ++b) {
    const loom_low_write_flow_block_t* block = &flow.blocks[b];
    memcpy(active_words, incoming + b * word_count, byte_count);
    loom_low_write_range_t snapshot = {0};
    bool snapshot_invalidated = true;
    for (uint32_t p = block->begin; p < block->end; ++p) {
      if (table->events[p].reset) {
        iree_bitmap_reset_all(active);
        snapshot = (loom_low_write_range_t){0};
        snapshot_invalidated = false;
      }
      // The transfer owns every mutation of |active|. Preserve its snapshot
      // until that producer invalidates it instead of comparing or rebuilding
      // identical ranges at every point. A reset establishes the empty set.
      if (snapshot_invalidated) {
        IREE_RETURN_IF_ERROR(
            loom_low_write_snapshot(table, units, active, arena, &snapshot));
      }
      table->events[p].retained = snapshot;
      IREE_RETURN_IF_ERROR(loom_low_write_count_constraints(table, p));
      snapshot_invalidated = loom_low_write_transfer(table, p, units, active);
    }
  }
  // Events own their immutable snapshots in the decision arena. The flow graph,
  // bitmaps and unit decoder no longer participate in constraint construction.
  iree_arena_checkpoint_restore(&construction_checkpoint);
  // Immutable snapshots and collected access extents determine the exact row
  // count. All segments are allocated before row publication; no payload moves
  // during construction or subsequent classification and candidate queries.
  IREE_RETURN_IF_ERROR(loom_low_write_allocate_constraints(table, arena));
  uint32_t next_constraint = 0;
  for (uint32_t p = 1; p < table->point_count; ++p) {
    next_constraint = loom_low_write_record_point(table, p, next_constraint);
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, table->value_count, sizeof(*table->inferred_bases),
      (void**)&table->inferred_bases));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, table->value_count, sizeof(*table->inferred_origins),
      (void**)&table->inferred_origins));
  // Inference starts after classification. Its two value-sized arrays first
  // hold component member links and tails, then become query workspace.
  IREE_RETURN_IF_ERROR(loom_low_write_classify_completion(
      table, liveness, placement, scratch_arena));
  // Retired ranges and sources retain the closure result; no workspace escapes.
  iree_arena_checkpoint_restore(&construction_checkpoint);
  IREE_RETURN_IF_ERROR(loom_low_write_index_constraints(table, arena));
  memset(table->inferred_bases, 0xFF,
         table->value_count * sizeof(*table->inferred_bases));
  return iree_ok_status();
}

iree_status_t loom_low_allocation_write_interference_finalize(
    loom_low_allocation_write_interference_t* table,
    const loom_liveness_analysis_t* liveness, const loom_cfg_graph_t* cfg_graph,
    const loom_low_placement_table_t* placement, iree_arena_allocator_t* arena,
    iree_arena_allocator_t* scratch_arena) {
  if (table == NULL || table->retained_unit_count == 0) {
    return iree_ok_status();
  }
  const iree_arena_checkpoint_t scratch_checkpoint =
      iree_arena_checkpoint_save(scratch_arena);
  const iree_status_t status = loom_low_write_finalize_impl(
      table, liveness, cfg_graph, placement, arena, scratch_arena);
  iree_arena_checkpoint_restore(&scratch_checkpoint);
  return status;
}

// Preserve the stored uint32_t base and unresolved sentinel without conversion.
// A provisional spill occupies the next value outside that domain.
#define LOOM_LOW_WRITE_LOCATION_UNKNOWN ((int64_t)UINT32_MAX)
#define LOOM_LOW_WRITE_LOCATION_SPILLED ((int64_t)UINT32_MAX + 1)

static int64_t loom_low_write_location(
    const loom_low_allocation_write_interference_t* table,
    const loom_low_allocation_assignment_map_t* assignments,
    loom_value_ordinal_t value, loom_value_ordinal_t candidate, uint32_t base,
    const loom_low_allocation_write_proposal_t* proposal) {
  if (value == candidate) {
    return base;
  }
  if (proposal != NULL && proposal->bases[value] != UINT32_MAX) {
    return proposal->bases[value];
  }
  if (table->inferred_bases[value] != UINT32_MAX) {
    return table->inferred_bases[value];
  }
  // Constraint endpoints and retained ranges already name valid local values.
  // The assignment owner publishes an index only after its record exists.
  const uint32_t assignment_index =
      assignments->assignment_indices_by_value_ordinal[value];
  if (assignment_index != UINT32_MAX) {
    const loom_low_allocation_assignment_t* assignment =
        &assignments->assignments[assignment_index];
    if (loom_low_allocation_assignment_is_physical_register_class(
            assignment, table->register_class)) {
      return assignment->location_base;
    }
    if (assignment->location_kind == LOOM_LOW_ALLOCATION_LOCATION_SPILL_SLOT) {
      return LOOM_LOW_WRITE_LOCATION_SPILLED;
    }
  }
  return table->values[value].fixed_base;
}

// Records a new candidate-local implication after its prior location was
// resolved as unknown. Known locations are checked by the caller, so each
// origin enters the queue only once.
static bool loom_low_write_infer_conflicts(
    loom_low_allocation_write_interference_t* table,
    loom_value_ordinal_t origin, int64_t required_base,
    uint32_t* pending_count) {
  if (required_base < 0 || required_base >= UINT32_MAX) {
    return true;
  }
  table->inferred_bases[origin] = (uint32_t)required_base;
  table->inferred_origins[(*pending_count)++] = origin;
  return false;
}

static loom_value_ordinal_t loom_low_write_origin_conflicting_read(
    loom_low_allocation_write_interference_t* table,
    const loom_low_allocation_assignment_map_t* assignments,
    loom_value_ordinal_t origin, uint32_t base,
    const loom_low_allocation_write_proposal_t* proposal) {
  table->inferred_origins[0] = origin;
  table->inferred_bases[origin] = base;
  uint32_t pending_count = 1;
  bool conflicts = false;
  loom_value_ordinal_t retained_origin = LOOM_VALUE_ORDINAL_INVALID;
  for (uint32_t pending = 0; pending < pending_count && !conflicts; ++pending) {
    const loom_low_write_range_t range =
        table->values[table->inferred_origins[pending]].constraints;
    for (uint32_t i = 0; i < range.count && !conflicts; ++i) {
      const loom_low_write_constraint_t* row =
          table->indexed_constraints[range.start + i];
      const int64_t destination = loom_low_write_location(
          table, assignments, row->destination, origin, base, proposal);
      if (row->retained.count != 0) {
        if (destination >= LOOM_LOW_WRITE_LOCATION_UNKNOWN) {
          continue;
        }
        const int64_t retained = loom_low_write_location(
            table, assignments, row->retained.value, origin, base, proposal);
        if (retained >= LOOM_LOW_WRITE_LOCATION_UNKNOWN ||
            !loom_low_write_ranges_overlap(
                destination + row->destination_offset - retained -
                    row->retained.offset,
                loom_low_write_constraint_write_count(row),
                row->retained.count)) {
          continue;
        }
        if (row->source == LOOM_VALUE_ORDINAL_INVALID) {
          conflicts = true;
          retained_origin = row->retained.value;
          continue;
        }
      } else if (destination == LOOM_LOW_WRITE_LOCATION_SPILLED) {
        continue;
      }
      const int64_t source = loom_low_write_location(
          table, assignments, row->source, origin, base, proposal);
      // Spill plans replace these lifetimes and rebuild allocation before
      // emission. A spilled range is not an unresolved physical endpoint.
      if (source == LOOM_LOW_WRITE_LOCATION_SPILLED) {
        continue;
      }
      if (destination < LOOM_LOW_WRITE_LOCATION_UNKNOWN) {
        const int64_t required_source =
            destination + row->destination_offset - row->units.source_offset;
        conflicts =
            source < LOOM_LOW_WRITE_LOCATION_UNKNOWN
                ? source != required_source
                : loom_low_write_infer_conflicts(
                      table, row->source, required_source, &pending_count);
      } else {
        // Only an equation reaches here. The pending origin is one endpoint,
        // so an unknown destination implies a known source.
        conflicts = loom_low_write_infer_conflicts(
            table, row->destination,
            source + row->units.source_offset - row->destination_offset,
            &pending_count);
      }
      if (conflicts) {
        retained_origin = row->retained.value;
      }
    }
  }
  for (uint32_t i = 0; i < pending_count; ++i) {
    table->inferred_bases[table->inferred_origins[i]] = UINT32_MAX;
  }
  return retained_origin;
}

static loom_value_ordinal_t loom_low_write_assignment_origin(
    const loom_low_allocation_write_interference_t* table,
    const loom_low_allocation_assignment_map_t* assignments,
    const loom_low_allocation_assignment_t* assignment) {
  const loom_value_ordinal_t ordinal = loom_module_value_ordinal_scratch_lookup(
      assignments->module, assignment->value_id);
  return table->values[ordinal].origin;
}

loom_value_ordinal_t loom_low_allocation_write_interference_conflicting_read(
    loom_low_allocation_write_interference_t* table,
    const loom_low_allocation_assignment_map_t* assignments,
    const loom_low_allocation_assignment_t* candidate) {
  if (table == NULL || table->constraint_count == 0 ||
      !loom_low_allocation_assignment_is_physical_register_class(
          candidate, table->register_class)) {
    return LOOM_VALUE_ORDINAL_INVALID;
  }
  const loom_value_ordinal_t origin =
      loom_low_write_assignment_origin(table, assignments, candidate);
  return loom_low_write_origin_conflicting_read(table, assignments, origin,
                                                candidate->location_base, NULL);
}

iree_status_t loom_low_allocation_write_proposal_initialize(
    const loom_low_allocation_write_interference_t* table,
    iree_arena_allocator_t* arena,
    loom_low_allocation_write_proposal_t* out_proposal) {
  *out_proposal = (loom_low_allocation_write_proposal_t){0};
  if (table == NULL || table->constraint_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, table->value_count,
                                                 sizeof(*out_proposal->bases),
                                                 (void**)&out_proposal->bases));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, table->value_count, sizeof(*out_proposal->origins),
      (void**)&out_proposal->origins));
  memset(out_proposal->bases, 0xFF,
         table->value_count * sizeof(*out_proposal->bases));
  return iree_ok_status();
}

void loom_low_allocation_write_proposal_reset(
    loom_low_allocation_write_proposal_t* proposal) {
  for (iree_host_size_t i = 0; i < proposal->count; ++i) {
    proposal->bases[proposal->origins[i]] = UINT32_MAX;
  }
  proposal->count = 0;
}

void loom_low_allocation_write_proposal_add(
    const loom_low_allocation_write_interference_t* table,
    const loom_low_allocation_assignment_map_t* assignments,
    const loom_low_allocation_assignment_t* assignment, uint32_t base,
    loom_low_allocation_write_proposal_t* proposal) {
  if (proposal->bases == NULL ||
      !loom_low_allocation_assignment_is_physical_register_class(
          assignment, table->register_class)) {
    return;
  }
  const loom_value_ordinal_t origin =
      loom_low_write_assignment_origin(table, assignments, assignment);
  if (proposal->bases[origin] == UINT32_MAX) {
    proposal->origins[proposal->count++] = origin;
  }
  proposal->bases[origin] = base;
}

bool loom_low_allocation_write_proposal_conflicts(
    loom_low_allocation_write_interference_t* table,
    const loom_low_allocation_assignment_map_t* assignments,
    const loom_low_allocation_write_proposal_t* proposal) {
  for (iree_host_size_t i = 0; i < proposal->count; ++i) {
    const loom_value_ordinal_t origin = proposal->origins[i];
    if (loom_low_write_origin_conflicting_read(
            table, assignments, origin, proposal->bases[origin], proposal) !=
        LOOM_VALUE_ORDINAL_INVALID) {
      return true;
    }
  }
  return false;
}

bool loom_low_allocation_write_interference_temporary_conflicts(
    const loom_low_allocation_write_interference_t* table,
    const loom_low_allocation_assignment_map_t* assignments,
    uint32_t write_point, const loom_low_move_location_t* temporary) {
  if (table == NULL ||
      temporary->descriptor_reg_class_id != table->register_class ||
      temporary->location_kind !=
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
    return false;
  }
  const loom_low_write_range_t range = table->events[write_point].retained;
  for (uint32_t i = 0; i < range.count; ++i) {
    const loom_low_write_retained_t* retained =
        &table->retained[range.start + i];
    // Move planning consumes final spill-free assignments. Retained origins
    // already have physical locations; no speculative inference participates.
    const loom_low_allocation_assignment_t* assignment =
        &assignments
             ->assignments[assignments->assignment_indices_by_value_ordinal
                               [retained->value]];
    const int64_t offset = (int64_t)temporary->location -
                           assignment->location_base - retained->offset;
    if (offset >= 0 && offset < retained->count) {
      return true;
    }
  }
  return false;
}
