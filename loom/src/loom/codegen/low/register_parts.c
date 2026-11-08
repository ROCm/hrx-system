// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/register_parts.h"

#include <string.h>

#include "loom/codegen/low/representation_binding.h"
#include "loom/target/registers.h"
#include "loom/util/adaptive_sort.h"

typedef enum loom_low_register_part_resolution_e {
  LOOM_LOW_REGISTER_PART_UNVISITED = 0,
  LOOM_LOW_REGISTER_PART_ACTIVE,
  LOOM_LOW_REGISTER_PART_RESOLVED,
} loom_low_register_part_resolution_t;

struct loom_low_register_part_continuation_t {
  // Result whose written mask is extended by its tied source.
  loom_value_id_t result;
  // Value supplying the preserved register parts.
  loom_value_id_t source;
  // State in the nonrecursive dependency traversal.
  loom_low_register_part_resolution_t resolution;
};

iree_status_t loom_low_register_parts_continue(loom_low_register_parts_t* parts,
                                               loom_value_id_t result,
                                               loom_value_id_t source,
                                               uint32_t written_mask,
                                               uint32_t full_mask) {
  uint32_t mask = written_mask | loom_low_register_parts_mask(parts, source);
  loom_low_register_parts_define(parts, result, mask);
  if ((mask & full_mask) == full_mask) {
    return iree_ok_status();
  }
  if (parts->continuations.count == parts->continuations.capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        parts->arena, parts->continuations.count,
        parts->continuations.count + 1, sizeof(*parts->continuations.values),
        &parts->continuations.capacity, (void**)&parts->continuations.values));
  }
  parts->continuations.values[parts->continuations.count++] =
      (loom_low_register_part_continuation_t){.result = result,
                                              .source = source};
  return iree_ok_status();
}

iree_status_t loom_low_register_parts_require(
    loom_low_register_parts_t* parts,
    const loom_low_register_part_requirement_t* requirement) {
  uint32_t mask = loom_low_register_parts_mask(parts, requirement->value);
  if ((mask & requirement->mask) == requirement->mask) {
    return iree_ok_status();
  }
  if (parts->requirements.count == parts->requirements.capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        parts->arena, parts->requirements.count, parts->requirements.count + 1,
        sizeof(*parts->requirements.values), &parts->requirements.capacity,
        (void**)&parts->requirements.values));
  }
  parts->requirements.values[parts->requirements.count++] = *requirement;
  return iree_ok_status();
}

static bool loom_low_register_part_result_less(
    const loom_low_register_part_continuation_t* lhs,
    const loom_low_register_part_continuation_t* rhs) {
  return lhs->result < rhs->result;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_low_register_parts_sort,
                          loom_low_register_part_continuation_t,
                          loom_low_register_part_result_less)

// Binary search is paid once per incomplete continuation, never per use.
static iree_host_size_t loom_low_register_parts_find_source(
    const loom_low_register_parts_t* parts, loom_value_id_t source) {
  iree_host_size_t begin = 0;
  iree_host_size_t end = parts->continuations.count;
  while (begin < end) {
    iree_host_size_t middle = begin + (end - begin) / 2;
    if (parts->continuations.values[middle].result < source) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  return begin < parts->continuations.count &&
                 parts->continuations.values[begin].result == source
             ? begin
             : parts->continuations.count;
}

iree_status_t loom_low_register_parts_resolve(
    loom_low_register_parts_t* parts) {
  const iree_host_size_t count = parts->continuations.count;
  if (count == 0) {
    return iree_ok_status();
  }
  loom_low_register_parts_sort(parts->continuations.values, count);
  iree_host_size_t* stack = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      parts->arena, count, sizeof(*stack), (void**)&stack));
  for (iree_host_size_t root = 0; root < count; ++root) {
    if (parts->continuations.values[root].resolution ==
        LOOM_LOW_REGISTER_PART_RESOLVED) {
      continue;
    }
    iree_host_size_t stack_count = 0;
    iree_host_size_t current = root;
    while (current < count && parts->continuations.values[current].resolution ==
                                  LOOM_LOW_REGISTER_PART_UNVISITED) {
      loom_low_register_part_continuation_t* continuation =
          &parts->continuations.values[current];
      continuation->resolution = LOOM_LOW_REGISTER_PART_ACTIVE;
      stack[stack_count++] = current;
      current =
          loom_low_register_parts_find_source(parts, continuation->source);
    }
    if (current < count && parts->continuations.values[current].resolution ==
                               LOOM_LOW_REGISTER_PART_ACTIVE) {
      // A cycle can occur in unreachable IR. Only its own masks belong in the
      // cycle union; incoming continuations can add parts for their results.
      uint32_t mask = 0;
      iree_host_size_t cycle_start = stack_count;
      do {
        --cycle_start;
        mask |= loom_low_register_parts_mask(
            parts, parts->continuations.values[stack[cycle_start]].result);
      } while (stack[cycle_start] != current);
      while (stack_count > cycle_start) {
        loom_low_register_part_continuation_t* continuation =
            &parts->continuations.values[stack[--stack_count]];
        loom_low_register_parts_define(parts, continuation->result, mask);
        continuation->resolution = LOOM_LOW_REGISTER_PART_RESOLVED;
      }
    }
    while (stack_count > 0) {
      loom_low_register_part_continuation_t* continuation =
          &parts->continuations.values[stack[--stack_count]];
      uint32_t mask =
          loom_low_register_parts_mask(parts, continuation->result) |
          loom_low_register_parts_mask(parts, continuation->source);
      loom_low_register_parts_define(parts, continuation->result, mask);
      continuation->resolution = LOOM_LOW_REGISTER_PART_RESOLVED;
    }
  }
  return iree_ok_status();
}

iree_status_t loom_low_register_parts_analyze(
    const loom_local_value_domain_t* value_domain,
    const loom_low_descriptor_set_t* descriptor_set,
    iree_arena_allocator_t* arena, uint32_t** out_masks) {
  *out_masks = NULL;
  loom_low_register_parts_t parts = {
      .value_domain = value_domain,
      .arena = arena,
  };
  const loom_module_t* module = value_domain->module;
  for (loom_value_ordinal_t ordinal = 0; ordinal < value_domain->value_count;
       ++ordinal) {
    if (parts.local_masks != NULL) {
      parts.local_masks[ordinal] = UINT32_MAX;
    }
    const loom_value_id_t value_id = value_domain->value_ids[ordinal];
    const loom_type_t type = loom_module_value_type(module, value_id);
    if (!loom_type_is_register(type)) {
      continue;
    }
    const uint16_t class_id = loom_low_register_type_class_id(type);
    const uint32_t full_mask =
        descriptor_set->reg_classes[class_id].full_register_part_mask;
    if (full_mask == 0) {
      continue;
    }
    const loom_value_t* value = loom_module_value(module, value_id);
    if (loom_value_is_block_arg(value)) {
      continue;
    }
    const loom_op_t* op = loom_value_def_op(value);
    loom_low_descriptor_packet_t packet;
    loom_low_descriptor_packet_initialize(descriptor_set, op, &packet);
    if (packet.descriptor == NULL) {
      continue;
    }
    const uint16_t result_index = loom_value_def_index(value);
    const loom_low_operand_t* result = loom_low_descriptor_operand(
        descriptor_set, packet.descriptor, result_index);
    const uint32_t written_mask =
        loom_low_register_parts_operand_mask(descriptor_set, result, class_id);
    if (written_mask == full_mask) {
      continue;
    }
    const loom_tied_result_t* tie = loom_op_find_tied_result(op, result_index);
    const loom_value_id_t source =
        tie != NULL ? loom_op_const_operands(op)[tie->operand_index]
                    : LOOM_VALUE_ID_INVALID;
    if (parts.local_masks == NULL) {
      // All preceding values are complete until the first partial definition.
      if (tie != NULL &&
          loom_local_value_domain_ordinal(value_domain, source) < ordinal) {
        continue;
      }
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, value_domain->value_count, sizeof(*parts.local_masks),
          (void**)&parts.local_masks));
      memset(parts.local_masks, 0,
             value_domain->value_count * sizeof(*parts.local_masks));
      memset(parts.local_masks, 0xFF, ordinal * sizeof(*parts.local_masks));
    }
    if (tie != NULL) {
      IREE_RETURN_IF_ERROR(loom_low_register_parts_continue(
          &parts, value_id, source, written_mask, full_mask));
    } else {
      loom_low_register_parts_define(&parts, value_id, written_mask);
    }
  }
  IREE_RETURN_IF_ERROR(loom_low_register_parts_resolve(&parts));
  *out_masks = parts.local_masks;
  return iree_ok_status();
}
