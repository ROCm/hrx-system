// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/fact_cast.h"

#include "loom/ir/structural_hash.h"
#include "loom/util/fact_table.h"

typedef struct loom_value_fact_integer_cast_entry_t {
  // Conversion result, or INVALID for an unoccupied hash slot.
  loom_value_id_t value;
  // Current complete publication; zero widths mark a withdrawn proof.
  loom_value_fact_integer_cast_t cast;
} loom_value_fact_integer_cast_entry_t;

struct loom_value_fact_integer_casts_t {
  // Open-addressed entries retained for this populated fact scope.
  loom_value_fact_integer_cast_entry_t* entries;
  // Power-of-two entry capacity, allocated only after the first publication.
  iree_host_size_t capacity;
  // Occupied keys, including reusable withdrawn publications.
  iree_host_size_t count;
};

static loom_value_fact_integer_cast_entry_t*
loom_value_fact_integer_casts_find_slot(
    loom_value_fact_integer_cast_entry_t* entries, iree_host_size_t capacity,
    loom_value_id_t value) {
  const iree_host_size_t mask = capacity - 1;
  iree_host_size_t slot = loom_structural_hash_finalize(value) & mask;
  while (entries[slot].value != LOOM_VALUE_ID_INVALID &&
         entries[slot].value != value) {
    slot = (slot + 1) & mask;
  }
  return &entries[slot];
}

static loom_value_fact_integer_cast_entry_t*
loom_value_fact_integer_casts_lookup(
    const loom_value_fact_integer_casts_t* casts, loom_value_id_t value) {
  if (!casts) {
    return NULL;
  }
  loom_value_fact_integer_cast_entry_t* entry =
      loom_value_fact_integer_casts_find_slot(casts->entries, casts->capacity,
                                              value);
  return entry->value == value ? entry : NULL;
}

loom_value_fact_integer_cast_t loom_value_fact_table_query_integer_cast(
    const loom_value_fact_table_t* table, loom_value_id_t value) {
  loom_value_fact_table_record_read(table, value);
  const loom_value_fact_integer_cast_entry_t* entry =
      loom_value_fact_integer_casts_lookup(table ? table->integer_casts : NULL,
                                           value);
  return entry ? entry->cast : (loom_value_fact_integer_cast_t){0};
}

iree_status_t loom_value_fact_table_for_each_integer_cast(
    const loom_value_fact_table_t* table,
    iree_status_t (*callback)(void* user_data, loom_value_id_t value,
                              loom_value_fact_integer_cast_t cast),
    void* user_data) {
  const loom_value_fact_integer_casts_t* casts = table->integer_casts;
  if (!casts) {
    return iree_ok_status();
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < casts->capacity && iree_status_is_ok(status);
       ++i) {
    const loom_value_fact_integer_cast_entry_t* entry = &casts->entries[i];
    if (entry->cast.immediate.bit_count) {
      status = callback(user_data, entry->value, entry->cast);
    }
  }
  return status;
}

static bool loom_value_fact_cast_equal(loom_integer_cast_t lhs,
                                       loom_integer_cast_t rhs) {
  return lhs.bit_count == rhs.bit_count &&
         (!lhs.bit_count ||
          (lhs.source_value_id == rhs.source_value_id &&
           loom_integer_cast_same_interpretation(&lhs, &rhs)));
}

bool loom_value_fact_integer_cast_equal(loom_value_fact_integer_cast_t lhs,
                                        loom_value_fact_integer_cast_t rhs) {
  return loom_value_fact_cast_equal(lhs.immediate, rhs.immediate) &&
         loom_value_fact_cast_equal(lhs.composed, rhs.composed);
}

iree_status_t loom_value_fact_table_define_integer_cast(
    loom_value_fact_table_t* table, loom_value_id_t value,
    loom_value_fact_integer_cast_t cast) {
  loom_value_fact_integer_cast_entry_t* entry =
      loom_value_fact_integer_casts_lookup(table->integer_casts, value);
  const loom_value_fact_integer_cast_t previous =
      entry ? entry->cast : (loom_value_fact_integer_cast_t){0};
  const bool changed = !loom_value_fact_integer_cast_equal(previous, cast);
  if (changed && !entry) {
    IREE_RETURN_IF_ERROR(loom_value_fact_table_touch(table, value));
    if (!table->integer_casts) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate(table->transient_arena,
                                               sizeof(*table->integer_casts),
                                               (void**)&table->integer_casts));
      *table->integer_casts = (loom_value_fact_integer_casts_t){0};
    }
    loom_value_fact_integer_casts_t* casts = table->integer_casts;
    if (casts->count >= casts->capacity - casts->capacity / 4) {
      iree_host_size_t capacity = 8;
      if (casts->capacity &&
          !iree_host_size_checked_mul(casts->capacity, 2, &capacity)) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "integer cast index capacity overflow");
      }
      loom_value_fact_integer_cast_entry_t* entries = NULL;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(table->transient_arena,
                                                     capacity, sizeof(*entries),
                                                     (void**)&entries));
      for (iree_host_size_t i = 0; i < capacity; ++i) {
        entries[i] = (loom_value_fact_integer_cast_entry_t){
            .value = LOOM_VALUE_ID_INVALID,
        };
      }
      for (iree_host_size_t i = 0; i < casts->capacity; ++i) {
        if (casts->entries[i].value != LOOM_VALUE_ID_INVALID) {
          *loom_value_fact_integer_casts_find_slot(
              entries, capacity, casts->entries[i].value) = casts->entries[i];
        }
      }
      casts->entries = entries;
      casts->capacity = capacity;
    }
    entry = loom_value_fact_integer_casts_find_slot(casts->entries,
                                                    casts->capacity, value);
    entry->value = value;
    ++casts->count;
  }
  if (changed) {
    entry->cast = cast;
  }
  loom_value_fact_table_record_origin(
      table, value, LOOM_VALUE_FACT_ORIGIN_INTEGER_CAST, changed);
  return iree_ok_status();
}

iree_status_t loom_value_fact_table_define_operand_cast(
    loom_fact_context_t* context, uint16_t result_ordinal,
    uint16_t operand_ordinal, loom_integer_cast_kind_t kind,
    uint8_t bit_count) {
  const loom_value_fact_origin_publication_t* publication =
      context->publication;
  if (!publication) {
    return iree_ok_status();
  }
  loom_value_fact_table_t* table = context->table;
  const loom_value_id_t source = loom_value_fact_table_query_numeric_identity(
      table, publication->numeric_operands
                 ? publication->numeric_operands[operand_ordinal]
                 : loom_op_const_operands(publication->op)[operand_ordinal]);
  const loom_integer_cast_t immediate = {
      .source_value_id = source,
      .bit_count = bit_count,
      .kind = kind,
  };
  const loom_value_fact_integer_cast_t input =
      loom_value_fact_table_query_integer_cast(table, source);
  const loom_value_fact_integer_cast_t cast = {
      .immediate = immediate,
      .composed =
          input.composed.bit_count
              ? loom_integer_cast_compose(input.composed, kind, bit_count)
              : immediate,
  };
  return loom_value_fact_table_define_integer_cast(
      table, loom_op_const_results(publication->op)[result_ordinal], cast);
}

bool loom_value_fact_table_query_integer_cast_from(
    const loom_value_fact_table_t* table, loom_value_id_t value,
    loom_value_id_t source, loom_integer_cast_t* out_cast) {
  if (!table->integer_casts) {
    return false;
  }
  value = loom_value_fact_table_query_numeric_identity(table, value);
  source = loom_value_fact_table_query_numeric_identity(table, source);
  const loom_value_fact_integer_cast_t cast =
      loom_value_fact_table_query_integer_cast(table, value);
  if (!cast.immediate.bit_count) {
    return false;
  }
  if (loom_value_fact_table_query_numeric_identity(
          table, cast.immediate.source_value_id) == source) {
    *out_cast = cast.immediate;
  } else if (loom_value_fact_table_query_numeric_identity(
                 table, cast.composed.source_value_id) == source) {
    *out_cast = cast.composed;
  } else {
    const loom_value_fact_integer_cast_t source_cast =
        loom_value_fact_table_query_integer_cast(table, source);
    if (!source_cast.composed.bit_count ||
        cast.composed.bit_count > source_cast.composed.bit_count ||
        loom_value_fact_table_query_numeric_identity(
            table, cast.composed.source_value_id) !=
            loom_value_fact_table_query_numeric_identity(
                table, source_cast.composed.source_value_id)) {
      return false;
    }
    *out_cast = cast.composed;
  }
  out_cast->source_value_id = source;
  return true;
}

void loom_value_fact_table_clear_integer_cast(loom_value_fact_table_t* table,
                                              loom_value_id_t value) {
  loom_value_fact_integer_cast_entry_t* entry =
      loom_value_fact_integer_casts_lookup(table->integer_casts, value);
  if (entry && entry->cast.immediate.bit_count) {
    entry->cast = (loom_value_fact_integer_cast_t){0};
    loom_value_fact_table_record_origin(
        table, value, LOOM_VALUE_FACT_ORIGIN_INTEGER_CAST, true);
  }
}
