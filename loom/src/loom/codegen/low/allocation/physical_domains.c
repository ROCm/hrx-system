// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation/physical_domains.h"

#include <string.h>

#include "iree/base/bitmap.h"
#include "iree/base/internal/math.h"
#include "loom/codegen/low/allocation/live_range.h"
#include "loom/util/adaptive_sort.h"

typedef struct loom_low_physical_component_t {
  // Union/find representative among scalar components.
  uint32_t parent;
  // Next member of the circular affinity component.
  uint32_t next;
  // Original liveness interval index for this member.
  uint32_t interval_index;
  // First storage point of this member's own interval.
  uint32_t interval_start;
  // Exclusive final storage point of this member's own interval.
  uint32_t interval_end;
  // First storage point of the component's conservative horizon.
  uint32_t start;
  // Exclusive final storage point of that horizon.
  uint32_t end;
  // Union rank, independent of the selected representative's candidate set.
  uint8_t rank;
  // Atomic storage demand of this value's required tied component. Only its
  // origin contributes; the origin's unit lifetime already retains every
  // member.
  uint16_t demand_units;
  // Candidate-set identity preserved by merges, then compacted to the index
  // of its distinct effective domain before the lifetime sweeps.
  iree_host_size_t domain;
  // Immutable canonical-physical-ID bits, shared until an intersection needs
  // a new set. Every set has the same byte length within this allocation.
  const uint64_t* registers;
} loom_low_physical_component_t;

typedef struct loom_low_physical_domain_t {
  // Immutable canonical-physical-ID bits for one distinct effective domain.
  const uint64_t* registers;
  // Atomic storage covered by every candidate in the effective domain.
  uint64_t* atomic_units;
  // Union of physical candidates appearing in strictly smaller atomic storage
  // domains.
  uint64_t* narrower;
} loom_low_physical_domain_t;

typedef struct loom_low_physical_candidate_unit_t {
  // Global atomic storage unit occupied by the candidate.
  uint16_t atomic_unit;
  // Semantic ordinal of the candidate in its register class.
  uint16_t candidate_ordinal;
} loom_low_physical_candidate_unit_t;

static bool loom_low_physical_bits_test(const uint64_t* words, uint32_t bit) {
  return (words[bit / 64] & (UINT64_C(1) << (bit % 64))) != 0;
}

static void loom_low_physical_bits_set(uint64_t* words, uint32_t bit) {
  words[bit / 64] |= UINT64_C(1) << (bit % 64);
}

static bool loom_low_physical_candidate_unit_less(
    const loom_low_physical_candidate_unit_t* lhs,
    const loom_low_physical_candidate_unit_t* rhs) {
  if (lhs->atomic_unit != rhs->atomic_unit) {
    return lhs->atomic_unit < rhs->atomic_unit;
  }
  return lhs->candidate_ordinal < rhs->candidate_ordinal;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_low_physical_candidate_unit_sort,
                          loom_low_physical_candidate_unit_t,
                          loom_low_physical_candidate_unit_less)

static bool loom_low_physical_point_less(const uint32_t* lhs,
                                         const uint32_t* rhs) {
  return *lhs < *rhs;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_low_physical_point_sort, uint32_t,
                          loom_low_physical_point_less)

static uint32_t loom_low_physical_component_find(
    loom_low_physical_component_t* components, uint32_t index) {
  uint32_t root = index;
  while (components[root].parent != root) {
    root = components[root].parent;
  }
  while (components[index].parent != index) {
    const uint32_t next = components[index].parent;
    components[index].parent = root;
    index = next;
  }
  return root;
}

static bool loom_low_physical_component_domain_less(
    loom_low_physical_component_t* const* lhs,
    loom_low_physical_component_t* const* rhs) {
  return (*lhs)->domain < (*rhs)->domain;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_low_physical_component_domain_sort,
                          loom_low_physical_component_t*,
                          loom_low_physical_component_domain_less)

static bool loom_low_physical_candidate_unit_find(
    const loom_low_physical_candidate_unit_t* units,
    iree_host_size_t unit_count, uint16_t atomic_unit,
    uint16_t* out_candidate_ordinal) {
  iree_host_size_t begin = 0;
  iree_host_size_t end = unit_count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    if (units[middle].atomic_unit < atomic_unit) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  if (begin == unit_count || units[begin].atomic_unit != atomic_unit) {
    return false;
  }
  *out_candidate_ordinal = units[begin].candidate_ordinal;
  return true;
}

static void loom_low_physical_register_mark_candidates(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t physical_register_id,
    const loom_low_physical_candidate_unit_t* candidate_units,
    iree_host_size_t candidate_unit_count, uint64_t* words) {
  const loom_low_physical_register_t* physical_register =
      &descriptor_set->physical_registers[physical_register_id];
  const uint16_t* atomic_units =
      descriptor_set->physical_register_atomic_units +
      physical_register->atomic_unit_start;
  for (uint16_t i = 0; i < physical_register->atomic_unit_count; ++i) {
    uint16_t candidate_ordinal = 0;
    if (loom_low_physical_candidate_unit_find(
            candidate_units, candidate_unit_count, atomic_units[i],
            &candidate_ordinal)) {
      loom_low_physical_bits_set(words, candidate_ordinal);
    }
  }
}

static iree_host_size_t loom_low_physical_point_lower_bound(
    const uint32_t* points, iree_host_size_t point_count, uint32_t point) {
  iree_host_size_t begin = 0;
  iree_host_size_t end = point_count;
  while (begin < end) {
    const iree_host_size_t middle = begin + (end - begin) / 2;
    if (points[middle] < point) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  return begin;
}

static uint64_t loom_low_physical_demand_query(const uint64_t* tree,
                                               iree_host_size_t tree_base,
                                               iree_host_size_t begin,
                                               iree_host_size_t end) {
  uint64_t maximum = 0;
  begin += tree_base;
  end += tree_base;
  while (begin < end) {
    if (begin & 1) {
      maximum = iree_max(maximum, tree[begin]);
      ++begin;
    }
    if (end & 1) {
      --end;
      maximum = iree_max(maximum, tree[end]);
    }
    begin /= 2;
    end /= 2;
  }
  return maximum;
}

typedef enum loom_low_physical_demand_scope_e {
  LOOM_LOW_PHYSICAL_DEMAND_SCOPE_REGISTER_CLASS_DOMAIN = 0,
  LOOM_LOW_PHYSICAL_DEMAND_SCOPE_EFFECTIVE_DOMAIN_AND_DESCENDANTS = 1,
  LOOM_LOW_PHYSICAL_DEMAND_SCOPE_REGISTER_CLASS_DOMAIN_AND_DESCENDANTS = 2,
} loom_low_physical_demand_scope_t;

static void loom_low_physical_demand_tree_build(
    const loom_liveness_analysis_t* liveness,
    loom_low_physical_component_t* components, uint32_t component_count,
    const iree_host_size_t* register_class_domains, const uint8_t* subsets,
    iree_host_size_t domain_count, iree_host_size_t domain,
    loom_low_physical_demand_scope_t scope, const uint32_t* demand_points,
    iree_host_size_t point_count, iree_host_size_t tree_base,
    iree_host_size_t tree_count, uint64_t* demand_starts, uint64_t* demand_ends,
    uint64_t* demand_tree) {
  memset(demand_starts, 0, point_count * sizeof(*demand_starts));
  memset(demand_ends, 0, point_count * sizeof(*demand_ends));
  memset(demand_tree, 0, tree_count * sizeof(*demand_tree));
  for (uint32_t i = 0; i < component_count; ++i) {
    const uint16_t units = components[i].demand_units;
    if (units == 0) {
      continue;
    }
    const uint32_t root = loom_low_physical_component_find(components, i);
    const uint16_t class_id = liveness->intervals[components[i].interval_index]
                                  .value_class.register_class_id;
    const iree_host_size_t member_domain =
        scope == LOOM_LOW_PHYSICAL_DEMAND_SCOPE_EFFECTIVE_DOMAIN_AND_DESCENDANTS
            ? components[root].domain
            : register_class_domains[class_id];
    const bool included =
        member_domain == domain ||
        (scope != LOOM_LOW_PHYSICAL_DEMAND_SCOPE_REGISTER_CLASS_DOMAIN &&
         subsets[member_domain * domain_count + domain]);
    if (!included) {
      continue;
    }
    const iree_host_size_t start = loom_low_physical_point_lower_bound(
        demand_points, point_count, components[i].interval_start);
    const iree_host_size_t end = loom_low_physical_point_lower_bound(
        demand_points, point_count, components[i].interval_end);
    demand_starts[start] += units;
    demand_ends[end] += units;
  }
  // There is at most one component per uint32 value ordinal and each component
  // contributes at most a uint16 count. The total demand is therefore below
  // 2^48 and is representable throughout this sweep.
  uint64_t live_units = 0;
  for (iree_host_size_t i = 0; i < point_count; ++i) {
    IREE_ASSERT_GE(live_units, demand_ends[i]);
    live_units -= demand_ends[i];
    live_units += demand_starts[i];
    demand_tree[tree_base + i] = live_units;
  }
  for (iree_host_size_t i = tree_base - 1; i > 0; --i) {
    demand_tree[i] = iree_max(demand_tree[2 * i], demand_tree[2 * i + 1]);
  }
}

static bool loom_low_physical_interval_is_scalar(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_interval_t* interval) {
  return interval->unit_count == 1 &&
         interval->value_class.type_kind == LOOM_TYPE_REGISTER &&
         loom_low_reg_class_uses_explicit_physical_registers(
             &descriptor_set
                  ->reg_classes[interval->value_class.register_class_id]);
}

static iree_status_t loom_low_physical_components_merge(
    const loom_low_placement_table_t* placement,
    const uint32_t* indices_by_ordinal, iree_host_size_t word_count,
    iree_arena_allocator_t* arena, loom_low_physical_component_t* components,
    iree_host_size_t next_domain) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < placement->relation_count && iree_status_is_ok(status); ++i) {
    const loom_low_placement_relation_t* relation = &placement->relations[i];
    if (relation->kind != LOOM_LOW_PLACEMENT_RELATION_SAME_STORAGE ||
        !loom_low_placement_relation_can_alias(relation) ||
        relation->unit_count != 1 || relation->source_unit_offset != 0 ||
        relation->result_unit_offset != 0) {
      continue;
    }
    uint32_t source = indices_by_ordinal[relation->source_ordinal];
    uint32_t result = indices_by_ordinal[relation->result_ordinal];
    if (source == UINT32_MAX || result == UINT32_MAX) {
      continue;
    }
    source = loom_low_physical_component_find(components, source);
    result = loom_low_physical_component_find(components, result);
    if (source == result) {
      continue;
    }
    const uint64_t* source_words = components[source].registers;
    const uint64_t* result_words = components[result].registers;
    uint64_t any = 0;
    uint64_t source_only = 0;
    uint64_t result_only = 0;
    for (iree_host_size_t w = 0; w < word_count; ++w) {
      any |= source_words[w] & result_words[w];
      source_only |= source_words[w] & ~result_words[w];
      result_only |= result_words[w] & ~source_words[w];
    }
    if (!any) {
      continue;
    }
    const uint64_t* common = source_only == 0 ? source_words : result_words;
    iree_host_size_t common_domain = source_only == 0
                                         ? components[source].domain
                                         : components[result].domain;
    if (source_only != 0 && result_only != 0) {
      uint64_t* intersection = NULL;
      status = iree_arena_allocate_array(
          arena, word_count, sizeof(*intersection), (void**)&intersection);
      if (!iree_status_is_ok(status)) {
        continue;
      }
      for (iree_host_size_t w = 0; w < word_count; ++w) {
        intersection[w] = source_words[w] & result_words[w];
      }
      common = intersection;
      common_domain = next_domain++;
    }
    if (components[source].rank < components[result].rank) {
      const uint32_t temporary = source;
      source = result;
      result = temporary;
    }
    components[result].parent = source;
    components[source].rank +=
        components[source].rank == components[result].rank;
    components[source].registers = common;
    components[source].domain = common_domain;
    components[source].start =
        iree_min(components[source].start, components[result].start);
    components[source].end =
        iree_max(components[source].end, components[result].end);
    const uint32_t next = components[source].next;
    components[source].next = components[result].next;
    components[result].next = next;
  }
  return status;
}

static iree_status_t loom_low_physical_domains_build_preferences(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_low_placement_table_t* placement, iree_host_size_t scalar_count,
    iree_arena_allocator_t* arena, uint32_t* offsets, uint64_t* penalty_words,
    uint64_t* affinity_words, uint64_t* reservation_words) {
  const iree_host_size_t word_count =
      iree_bitmap_calculate_words(descriptor_set->physical_register_count);
  const iree_host_size_t atomic_word_count =
      iree_bitmap_calculate_words(descriptor_set->physical_register_unit_count);
  uint64_t** class_words = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, descriptor_set->reg_class_count,
                                sizeof(*class_words), (void**)&class_words));
  memset(class_words, 0,
         descriptor_set->reg_class_count * sizeof(*class_words));
  uint64_t** class_atomic_words = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, descriptor_set->reg_class_count, sizeof(*class_atomic_words),
      (void**)&class_atomic_words));
  memset(class_atomic_words, 0,
         descriptor_set->reg_class_count * sizeof(*class_atomic_words));
  loom_low_physical_candidate_unit_t** candidate_units_by_class = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, descriptor_set->reg_class_count, sizeof(*candidate_units_by_class),
      (void**)&candidate_units_by_class));
  memset(candidate_units_by_class, 0,
         descriptor_set->reg_class_count * sizeof(*candidate_units_by_class));
  iree_host_size_t* candidate_unit_counts_by_class = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, descriptor_set->reg_class_count,
                                sizeof(*candidate_unit_counts_by_class),
                                (void**)&candidate_unit_counts_by_class));
  memset(candidate_unit_counts_by_class, 0,
         descriptor_set->reg_class_count *
             sizeof(*candidate_unit_counts_by_class));
  uint32_t* indices_by_ordinal = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, liveness->value_count,
                                                 sizeof(*indices_by_ordinal),
                                                 (void**)&indices_by_ordinal));
  memset(indices_by_ordinal, 0xFF,
         liveness->value_count * sizeof(*indices_by_ordinal));
  loom_low_physical_component_t* components = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, scalar_count, sizeof(*components), (void**)&components));
  loom_low_physical_component_t** order = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, scalar_count, sizeof(*order), (void**)&order));
  uint16_t* used_class_ids = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, scalar_count, sizeof(*used_class_ids), (void**)&used_class_ids));
  iree_host_size_t used_class_count = 0;
  bool has_overlapping_domains = false;
  uint32_t component_count = 0;
  uint32_t offset = 0;
  for (loom_value_ordinal_t v = 0; v < liveness->value_count; ++v) {
    const uint32_t index = liveness->value_interval_indices[v];
    if (index == UINT32_MAX) {
      continue;
    }
    const loom_liveness_interval_t* interval = &liveness->intervals[index];
    if (!loom_low_physical_interval_is_scalar(descriptor_set, interval)) {
      continue;
    }
    const uint16_t class_id = interval->value_class.register_class_id;
    if (!class_words[class_id]) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, word_count, sizeof(uint64_t), (void**)&class_words[class_id]));
      memset(class_words[class_id], 0, word_count * sizeof(uint64_t));
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate_array(arena, atomic_word_count, sizeof(uint64_t),
                                    (void**)&class_atomic_words[class_id]));
      memset(class_atomic_words[class_id], 0,
             atomic_word_count * sizeof(uint64_t));
      const loom_low_reg_class_t* reg_class =
          &descriptor_set->reg_classes[class_id];
      // Both factors are uint16, so their product fits every supported
      // iree_host_size_t representation.
      const iree_host_size_t candidate_unit_count =
          (iree_host_size_t)reg_class->allocatable_count *
          reg_class->physical_atomic_unit_count;
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, candidate_unit_count,
          sizeof(*candidate_units_by_class[class_id]),
          (void**)&candidate_units_by_class[class_id]));
      candidate_unit_counts_by_class[class_id] = candidate_unit_count;
      iree_host_size_t candidate_unit_index = 0;
      for (uint16_t r = 0; r < reg_class->allocatable_count; ++r) {
        const uint32_t physical_register_id =
            loom_low_descriptor_set_physical_register_candidate(descriptor_set,
                                                                class_id, r);
        loom_low_physical_bits_set(class_words[class_id], physical_register_id);
        const loom_low_physical_register_t* physical_register =
            &descriptor_set->physical_registers[physical_register_id];
        const uint16_t* atomic_units =
            descriptor_set->physical_register_atomic_units +
            physical_register->atomic_unit_start;
        for (uint16_t j = 0; j < physical_register->atomic_unit_count; ++j) {
          loom_low_physical_bits_set(class_atomic_words[class_id],
                                     atomic_units[j]);
          candidate_units_by_class[class_id][candidate_unit_index++] =
              (loom_low_physical_candidate_unit_t){
                  .atomic_unit = atomic_units[j],
                  .candidate_ordinal = r,
              };
        }
      }
      IREE_ASSERT_EQ(candidate_unit_index, candidate_unit_count);
      loom_low_physical_candidate_unit_sort(candidate_units_by_class[class_id],
                                            candidate_unit_count);
      for (iree_host_size_t c = 0;
           c < used_class_count && !has_overlapping_domains; ++c) {
        uint64_t shared = 0;
        uint64_t difference = 0;
        const uint16_t previous_class_id = used_class_ids[c];
        for (iree_host_size_t w = 0; w < atomic_word_count; ++w) {
          shared |= class_atomic_words[class_id][w] &
                    class_atomic_words[previous_class_id][w];
          difference |= class_atomic_words[class_id][w] ^
                        class_atomic_words[previous_class_id][w];
        }
        has_overlapping_domains = shared != 0 && difference != 0;
      }
      used_class_ids[used_class_count++] = class_id;
    }
    const uint32_t point =
        loom_low_allocation_unit_liveness_point_start_for_value_ordinal(
            unit_liveness, liveness, v);
    const uint32_t start = point != UINT32_MAX
                               ? unit_liveness->start_points[point]
                               : interval->start_point;
    const uint32_t end =
        point != UINT32_MAX
            ? unit_liveness->end_points[point]
            : loom_low_allocation_live_range_interval_storage_end_point(
                  interval);
    indices_by_ordinal[v] = component_count;
    components[component_count] = (loom_low_physical_component_t){
        .parent = component_count,
        .next = component_count,
        .interval_index = index,
        .interval_start = start,
        .interval_end = end,
        .start = start,
        .end = end,
        .demand_units =
            placement->tied_storage_origins_by_value_ordinal == NULL ||
                    placement->tied_storage_origins_by_value_ordinal[v] == v
                ? descriptor_set->reg_classes[class_id]
                      .physical_atomic_unit_count
                : 0,
        .domain = class_id,
        .registers = class_words[class_id],
    };
    ++component_count;
    offsets[index] = offset;
    offset += iree_bitmap_calculate_words(
        descriptor_set->reg_classes[class_id].allocatable_count);
  }
  // Equal or disjoint classes cannot form a narrower affinity intersection or
  // constrain one another. Their candidate preferences are uniformly inert.
  if (!has_overlapping_domains) {
    memset(offsets, 0xFF, liveness->interval_count * sizeof(*offsets));
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_low_physical_components_merge(
      placement, indices_by_ordinal, word_count, arena, components,
      descriptor_set->reg_class_count));
  iree_host_size_t root_count = 0;
  for (uint32_t i = 0; i < component_count; ++i) {
    const uint32_t root = loom_low_physical_component_find(components, i);
    if (root == i) {
      order[root_count++] = &components[i];
    }
    const uint32_t interval_index = components[i].interval_index;
    const uint16_t class_id =
        liveness->intervals[interval_index].value_class.register_class_id;
    const loom_low_reg_class_t* reg_class =
        &descriptor_set->reg_classes[class_id];
    const uint64_t* common = components[root].registers;
    if (common == class_words[class_id]) {
      continue;
    }
    for (uint16_t r = 0; r < reg_class->allocatable_count; ++r) {
      if (!loom_low_physical_bits_test(
              common, loom_low_descriptor_set_physical_register_candidate(
                          descriptor_set, class_id, r))) {
        loom_low_physical_bits_set(penalty_words + offsets[interval_index], r);
        loom_low_physical_bits_set(affinity_words + offsets[interval_index], r);
      }
    }
  }
  // Intern equal effective affinity domains and original register-class
  // domains. Affinity may prefer a narrower common location, but every member
  // can fall back to its class domain when that location is unavailable.
  loom_low_physical_component_domain_sort(order, root_count);
  iree_host_size_t domain_capacity = 0;
  for (iree_host_size_t i = 0; i < root_count; ++i) {
    if (i == 0 || order[i - 1]->domain != order[i]->domain) {
      ++domain_capacity;
    }
  }
  if (!iree_host_size_checked_add(domain_capacity, used_class_count,
                                  &domain_capacity)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "physical domain count overflow");
  }
  loom_low_physical_domain_t* domains = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, domain_capacity, sizeof(*domains), (void**)&domains));
  memset(domains, 0, domain_capacity * sizeof(*domains));
  iree_host_size_t domain_count = 0;
  iree_host_size_t previous_identity = IREE_HOST_SIZE_MAX;
  iree_host_size_t domain_index = 0;
  for (iree_host_size_t i = 0; i < root_count; ++i) {
    if (order[i]->domain != previous_identity) {
      previous_identity = order[i]->domain;
      domain_index = 0;
      while (domain_index < domain_count &&
             memcmp(domains[domain_index].registers, order[i]->registers,
                    word_count * sizeof(uint64_t)) != 0) {
        ++domain_index;
      }
      if (domain_index == domain_count) {
        domains[domain_count++].registers = order[i]->registers;
      }
    }
    order[i]->domain = domain_index;
  }
  iree_host_size_t* register_class_domains = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, descriptor_set->reg_class_count, sizeof(*register_class_domains),
      (void**)&register_class_domains));
  for (uint16_t class_id = 0; class_id < descriptor_set->reg_class_count;
       ++class_id) {
    register_class_domains[class_id] = IREE_HOST_SIZE_MAX;
  }
  for (iree_host_size_t i = 0; i < used_class_count; ++i) {
    const uint16_t class_id = used_class_ids[i];
    domain_index = 0;
    while (domain_index < domain_count &&
           memcmp(domains[domain_index].registers, class_words[class_id],
                  word_count * sizeof(uint64_t)) != 0) {
      ++domain_index;
    }
    if (domain_index == domain_count) {
      domains[domain_count++].registers = class_words[class_id];
    }
    register_class_domains[class_id] = domain_index;
  }
  uint64_t* domain_atomic_units = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, domain_count, atomic_word_count * sizeof(*domain_atomic_units),
      (void**)&domain_atomic_units));
  memset(domain_atomic_units, 0,
         domain_count * atomic_word_count * sizeof(*domain_atomic_units));
  for (iree_host_size_t d = 0; d < domain_count; ++d) {
    domains[d].atomic_units = domain_atomic_units + d * atomic_word_count;
    for (uint32_t r = 0; r < descriptor_set->physical_register_count; ++r) {
      if (!loom_low_physical_bits_test(domains[d].registers, r)) {
        continue;
      }
      const loom_low_physical_register_t* physical_register =
          &descriptor_set->physical_registers[r];
      const uint16_t* atomic_units =
          descriptor_set->physical_register_atomic_units +
          physical_register->atomic_unit_start;
      for (uint16_t j = 0; j < physical_register->atomic_unit_count; ++j) {
        loom_low_physical_bits_set(domains[d].atomic_units, atomic_units[j]);
      }
    }
  }

  // Physical names are representation choices, not storage identities. Merge
  // effective domains with equal atomic coverage before computing capacity so
  // simultaneous values from differently named classes contribute to the same
  // demand calendar.
  loom_low_physical_domain_t* atomic_domains = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, domain_count, sizeof(*atomic_domains), (void**)&atomic_domains));
  memset(atomic_domains, 0, domain_count * sizeof(*atomic_domains));
  uint64_t* atomic_domain_registers = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, domain_count, word_count * sizeof(*atomic_domain_registers),
      (void**)&atomic_domain_registers));
  memset(atomic_domain_registers, 0,
         domain_count * word_count * sizeof(*atomic_domain_registers));
  iree_host_size_t* atomic_domain_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, domain_count, sizeof(*atomic_domain_indices),
      (void**)&atomic_domain_indices));
  iree_host_size_t atomic_domain_count = 0;
  for (iree_host_size_t d = 0; d < domain_count; ++d) {
    iree_host_size_t atomic_domain_index = 0;
    while (atomic_domain_index < atomic_domain_count &&
           memcmp(atomic_domains[atomic_domain_index].atomic_units,
                  domains[d].atomic_units,
                  atomic_word_count * sizeof(uint64_t)) != 0) {
      ++atomic_domain_index;
    }
    if (atomic_domain_index == atomic_domain_count) {
      atomic_domains[atomic_domain_index].atomic_units =
          domains[d].atomic_units;
      atomic_domains[atomic_domain_index].registers =
          atomic_domain_registers + atomic_domain_index * word_count;
      ++atomic_domain_count;
    }
    atomic_domain_indices[d] = atomic_domain_index;
    uint64_t* registers =
        atomic_domain_registers + atomic_domain_index * word_count;
    for (iree_host_size_t w = 0; w < word_count; ++w) {
      registers[w] |= domains[d].registers[w];
    }
  }
  for (iree_host_size_t i = 0; i < root_count; ++i) {
    order[i]->domain = atomic_domain_indices[order[i]->domain];
  }
  for (iree_host_size_t i = 0; i < used_class_count; ++i) {
    const uint16_t class_id = used_class_ids[i];
    register_class_domains[class_id] =
        atomic_domain_indices[register_class_domains[class_id]];
  }
  domains = atomic_domains;
  domain_count = atomic_domain_count;

  uint64_t* narrower = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, domain_count, word_count * sizeof(*narrower), (void**)&narrower));
  memset(narrower, 0, domain_count * word_count * sizeof(*narrower));
  for (iree_host_size_t d = 0; d < domain_count; ++d) {
    domains[d].narrower = narrower + d * word_count;
  }
  uint8_t* subsets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, domain_count, domain_count * sizeof(*subsets), (void**)&subsets));
  // Only distinct domains are compared. Candidate sets from one explicit
  // class are disjoint internally; strict atomic subsets are the domains that
  // compete with a broader class for only part of its storage.
  for (iree_host_size_t a = 0; a < domain_count; ++a) {
    for (iree_host_size_t b = 0; b < domain_count; ++b) {
      uint64_t outside = 0;
      uint64_t missing = 0;
      for (iree_host_size_t w = 0; w < atomic_word_count; ++w) {
        outside |= domains[a].atomic_units[w] & ~domains[b].atomic_units[w];
        missing |= domains[b].atomic_units[w] & ~domains[a].atomic_units[w];
      }
      const bool subset = outside == 0 && missing != 0;
      subsets[a * domain_count + b] = subset;
      if (!subset) {
        continue;
      }
      for (iree_host_size_t w = 0; w < word_count; ++w) {
        domains[b].narrower[w] |= domains[a].registers[w];
      }
    }
  }
  uint64_t* narrower_words = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, offset, sizeof(*narrower_words), (void**)&narrower_words));
  memset(narrower_words, 0, offset * sizeof(*narrower_words));
  uint64_t* overlapping_words = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, offset, sizeof(*overlapping_words), (void**)&overlapping_words));
  memset(overlapping_words, 0, offset * sizeof(*overlapping_words));
  for (iree_host_size_t i = 0; i < root_count; ++i) {
    const loom_low_physical_component_t* component = order[i];
    const loom_low_physical_domain_t* domain = &domains[component->domain];
    uint32_t member = component->parent;
    do {
      const uint32_t interval_index = components[member].interval_index;
      const uint16_t reg_class_id =
          liveness->intervals[interval_index].value_class.register_class_id;
      for (uint32_t r = 0; r < descriptor_set->physical_register_count; ++r) {
        if (!loom_low_physical_bits_test(domain->narrower, r)) {
          continue;
        }
        loom_low_physical_register_mark_candidates(
            descriptor_set, r, candidate_units_by_class[reg_class_id],
            candidate_unit_counts_by_class[reg_class_id],
            narrower_words + offsets[interval_index]);
      }
      member = components[member].next;
    } while (member != component->parent);
  }

  const iree_host_size_t class_word_count =
      iree_bitmap_calculate_words(descriptor_set->reg_class_count);
  uint64_t* domain_classes = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, domain_count, class_word_count * sizeof(*domain_classes),
      (void**)&domain_classes));
  memset(domain_classes, 0,
         domain_count * class_word_count * sizeof(*domain_classes));
  bool has_narrowed_affinity = false;
  for (uint32_t i = 0; i < component_count; ++i) {
    const uint32_t root = loom_low_physical_component_find(components, i);
    const uint16_t class_id = liveness->intervals[components[i].interval_index]
                                  .value_class.register_class_id;
    loom_low_physical_bits_set(
        domain_classes + components[root].domain * class_word_count, class_id);
    loom_low_physical_bits_set(
        domain_classes + register_class_domains[class_id] * class_word_count,
        class_id);
    has_narrowed_affinity |=
        components[root].domain != register_class_domains[class_id];
  }

  iree_host_size_t point_capacity = 0;
  if (!iree_host_size_checked_mul(component_count, 2, &point_capacity)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "physical demand point count overflow");
  }
  uint32_t* demand_points = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, point_capacity, sizeof(*demand_points), (void**)&demand_points));
  for (uint32_t i = 0; i < component_count; ++i) {
    demand_points[2 * i] = components[i].interval_start;
    demand_points[2 * i + 1] = components[i].interval_end;
  }
  loom_low_physical_point_sort(demand_points, point_capacity);
  iree_host_size_t point_count = 0;
  for (iree_host_size_t i = 0; i < point_capacity; ++i) {
    if (point_count == 0 ||
        demand_points[point_count - 1] != demand_points[i]) {
      demand_points[point_count++] = demand_points[i];
    }
  }
  iree_host_size_t tree_base = 1;
  while (tree_base < point_count) {
    if (tree_base > IREE_HOST_SIZE_MAX / 2) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "physical demand tree size overflow");
    }
    tree_base *= 2;
  }
  iree_host_size_t tree_count = 0;
  if (!iree_host_size_checked_mul(tree_base, 2, &tree_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "physical demand tree size overflow");
  }
  uint64_t* demand_starts = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, point_count, sizeof(*demand_starts), (void**)&demand_starts));
  uint64_t* demand_ends = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, point_count, sizeof(*demand_ends), (void**)&demand_ends));
  uint64_t* demand_tree = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, tree_count, sizeof(*demand_tree), (void**)&demand_tree));

  // Build one atomic-storage lifetime calendar at a time. Descendant domains
  // contribute their atomic width to every ancestor calendar, making each
  // range maximum the Hall demand that must remain available in that domain.
  // Affinity calendars preserve preferred transfers. Actual class calendars
  // additionally reserve overlapping narrow storage when a member falls back
  // outside its affinity; its own nonoverlapping copy does not reserve itself.
  for (iree_host_size_t a = 0; a < domain_count; ++a) {
    bool has_broader_domain = false;
    for (iree_host_size_t b = 0; b < domain_count; ++b) {
      has_broader_domain |= subsets[a * domain_count + b];
    }
    if (!has_broader_domain) {
      continue;
    }
    const uint32_t scope_count = has_narrowed_affinity ? 2 : 1;
    for (uint32_t scope_index = 0; scope_index < scope_count; ++scope_index) {
      const bool reserve_fallback = scope_index != 0;
      loom_low_physical_demand_tree_build(
          liveness, components, component_count, register_class_domains,
          subsets, domain_count, a,
          reserve_fallback
              ? LOOM_LOW_PHYSICAL_DEMAND_SCOPE_REGISTER_CLASS_DOMAIN_AND_DESCENDANTS
              : LOOM_LOW_PHYSICAL_DEMAND_SCOPE_EFFECTIVE_DOMAIN_AND_DESCENDANTS,
          demand_points, point_count, tree_base, tree_count, demand_starts,
          demand_ends, demand_tree);
      const iree_host_size_t query_count =
          reserve_fallback ? component_count : root_count;
      for (iree_host_size_t i = 0; i < query_count; ++i) {
        const loom_low_physical_component_t* component =
            reserve_fallback ? &components[i] : order[i];
        const uint32_t interval_index = component->interval_index;
        const uint16_t broad_class_id =
            liveness->intervals[interval_index].value_class.register_class_id;
        const iree_host_size_t broad_domain =
            reserve_fallback ? register_class_domains[broad_class_id]
                             : component->domain;
        if (!subsets[a * domain_count + broad_domain]) {
          continue;
        }
        const uint32_t start =
            reserve_fallback ? component->interval_start : component->start;
        const uint32_t stop =
            reserve_fallback ? component->interval_end : component->end;
        const iree_host_size_t begin = loom_low_physical_point_lower_bound(
            demand_points, point_count, start);
        const iree_host_size_t end = loom_low_physical_point_lower_bound(
            demand_points, point_count, stop);
        const uint64_t peak_units =
            loom_low_physical_demand_query(demand_tree, tree_base, begin, end);
        if (peak_units == 0) {
          continue;
        }
        uint32_t member = reserve_fallback ? (uint32_t)i : component->parent;
        do {
          const uint32_t member_interval = components[member].interval_index;
          const uint16_t member_class = liveness->intervals[member_interval]
                                            .value_class.register_class_id;
          for (uint16_t narrow_class_id = 0;
               narrow_class_id < descriptor_set->reg_class_count;
               ++narrow_class_id) {
            if (!loom_low_physical_bits_test(
                    domain_classes + a * class_word_count, narrow_class_id)) {
              continue;
            }
            const loom_low_reg_class_t* narrow_class =
                &descriptor_set->reg_classes[narrow_class_id];
            const uint16_t candidate_units =
                narrow_class->physical_atomic_unit_count;
            const uint64_t required_candidates =
                peak_units / candidate_units +
                (peak_units % candidate_units != 0);
            const uint16_t reserve_count = (uint16_t)iree_min(
                required_candidates, (uint64_t)narrow_class->allocatable_count);
            uint16_t selected_count = 0;
            for (uint16_t c = 0; c < narrow_class->allocatable_count; ++c) {
              const uint16_t candidate_ordinal =
                  descriptor_set->physical_register_allocation_ordinals
                      [narrow_class->physical_register_candidate_start + c];
              const uint32_t physical_register_id =
                  loom_low_descriptor_set_physical_register_candidate(
                      descriptor_set, narrow_class_id, candidate_ordinal);
              if (!loom_low_physical_bits_test(domains[a].registers,
                                               physical_register_id)) {
                continue;
              }
              loom_low_physical_register_mark_candidates(
                  descriptor_set, physical_register_id,
                  candidate_units_by_class[member_class],
                  candidate_unit_counts_by_class[member_class],
                  overlapping_words + offsets[member_interval]);
              if (++selected_count == reserve_count) {
                break;
              }
            }
          }
          member = components[member].next;
        } while (!reserve_fallback && member != component->parent);
      }
    }
  }

  // Copy affinity is a preference: a member whose common candidate is
  // unavailable falls back to its original register class and materializes the
  // copy. Count broad pressure by those class domains so affinity cannot hide
  // fallback demand. Reuse idle narrower storage only when that demand exceeds
  // the candidates outside every narrower domain.
  bool* prefer_narrower_by_interval = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, liveness->interval_count, sizeof(*prefer_narrower_by_interval),
      (void**)&prefer_narrower_by_interval));
  memset(prefer_narrower_by_interval, 0,
         liveness->interval_count * sizeof(*prefer_narrower_by_interval));
  for (iree_host_size_t b = 0; b < domain_count; ++b) {
    bool has_narrower_domain = false;
    for (iree_host_size_t a = 0; a < domain_count; ++a) {
      has_narrower_domain |= subsets[a * domain_count + b];
    }
    if (!has_narrower_domain) {
      continue;
    }
    loom_low_physical_demand_tree_build(
        liveness, components, component_count, register_class_domains, subsets,
        domain_count, b, LOOM_LOW_PHYSICAL_DEMAND_SCOPE_REGISTER_CLASS_DOMAIN,
        demand_points, point_count, tree_base, tree_count, demand_starts,
        demand_ends, demand_tree);
    for (iree_host_size_t i = 0; i < root_count; ++i) {
      const loom_low_physical_component_t* component = order[i];
      if (component->domain != b) {
        continue;
      }
      const iree_host_size_t begin = loom_low_physical_point_lower_bound(
          demand_points, point_count, component->start);
      const iree_host_size_t end = loom_low_physical_point_lower_bound(
          demand_points, point_count, component->end);
      const uint64_t peak_units =
          loom_low_physical_demand_query(demand_tree, tree_base, begin, end);
      uint32_t member = component->parent;
      do {
        const uint32_t interval_index = components[member].interval_index;
        const uint16_t class_id =
            liveness->intervals[interval_index].value_class.register_class_id;
        const loom_low_reg_class_t* reg_class =
            &descriptor_set->reg_classes[class_id];
        uint16_t general_candidate_count = 0;
        for (uint16_t candidate = 0; candidate < reg_class->allocatable_count;
             ++candidate) {
          const uint64_t bit = UINT64_C(1) << (candidate % 64);
          const iree_host_size_t word =
              offsets[interval_index] + candidate / 64;
          const bool narrower_candidate = (narrower_words[word] & bit) != 0;
          const bool outside_affinity = (affinity_words[word] & bit) != 0;
          general_candidate_count += !narrower_candidate && !outside_affinity;
        }
        const uint64_t general_units = (uint64_t)general_candidate_count *
                                       reg_class->physical_atomic_unit_count;
        prefer_narrower_by_interval[interval_index] =
            peak_units > general_units;
        member = components[member].next;
      } while (member != component->parent);
    }
  }

  // Capacity-bound broad values first reuse idle narrower storage, then
  // general-only storage, and only then storage reserved by concurrent narrow
  // demand. Other broad values retain ordinary allocation order while still
  // preserving the reserved capacity.
  for (uint32_t i = 0; i < component_count; ++i) {
    const uint32_t index = components[i].interval_index;
    const uint16_t class_id =
        liveness->intervals[index].value_class.register_class_id;
    const uint16_t count =
        descriptor_set->reg_classes[class_id].allocatable_count;
    const iree_host_size_t interval_word_count =
        iree_bitmap_calculate_words(count);
    bool has_idle_narrower_candidate = false;
    for (iree_host_size_t w = 0; w < interval_word_count; ++w) {
      has_idle_narrower_candidate |=
          (narrower_words[offsets[index] + w] &
           ~overlapping_words[offsets[index] + w]) != 0;
    }
    for (iree_host_size_t w = 0; w < interval_word_count; ++w) {
      const uint32_t remaining = count - (uint32_t)(w * 64);
      const uint64_t valid_bits =
          remaining < 64 ? (UINT64_C(1) << remaining) - 1 : UINT64_MAX;
      const uint64_t narrower_candidates =
          narrower_words[offsets[index] + w] & valid_bits;
      const uint64_t reserved_narrower =
          overlapping_words[offsets[index] + w] & valid_bits;
      if (has_idle_narrower_candidate && prefer_narrower_by_interval[index]) {
        penalty_words[offsets[index] + w] |=
            (~narrower_candidates | reserved_narrower) & valid_bits;
        reservation_words[offsets[index] + w] |= reserved_narrower;
      } else {
        reservation_words[offsets[index] + w] |= reserved_narrower;
      }
    }
    const uint32_t first_rank =
        (uint32_t)(penalty_words[offsets[index]] & 1) +
        (uint32_t)(reservation_words[offsets[index]] & 1);
    const bool first_breaks_affinity =
        (affinity_words[offsets[index]] & 1) != 0;
    const bool first_reserved = (reservation_words[offsets[index]] & 1) != 0;
    bool uniform = true;
    for (uint16_t candidate = 1; candidate < count; ++candidate) {
      const uint64_t bit = UINT64_C(1) << (candidate % 64);
      const iree_host_size_t word = offsets[index] + candidate / 64;
      const uint32_t rank = (uint32_t)((penalty_words[word] & bit) != 0) +
                            (uint32_t)((reservation_words[word] & bit) != 0);
      uniform &= rank == first_rank &&
                 ((affinity_words[word] & bit) != 0) == first_breaks_affinity &&
                 ((reservation_words[word] & bit) != 0) == first_reserved;
    }
    if (uniform) {
      offsets[index] = UINT32_MAX;
    }
  }
  return iree_ok_status();
}

loom_low_allocation_physical_domain_row_t
loom_low_allocation_physical_domains_for_interval(
    const loom_low_allocation_physical_domains_t* domains,
    const loom_liveness_analysis_t* liveness,
    const loom_liveness_interval_t* interval) {
  if (!domains || !domains->offsets) {
    return (loom_low_allocation_physical_domain_row_t){0};
  }
  const uint32_t offset = domains->offsets[interval - liveness->intervals];
  if (offset == UINT32_MAX) {
    return (loom_low_allocation_physical_domain_row_t){0};
  }
  return (loom_low_allocation_physical_domain_row_t){
      .penalty_words = domains->penalty_words + offset,
      .affinity_words = domains->affinity_words + offset,
      .reservation_words = domains->reservation_words + offset,
  };
}

iree_status_t loom_low_allocation_physical_domains_build(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_low_placement_table_t* placement, iree_arena_allocator_t* arena,
    loom_low_allocation_physical_domains_t* out_domains) {
  *out_domains = (loom_low_allocation_physical_domains_t){0};
  if (descriptor_set->physical_register_count == 0) {
    return iree_ok_status();
  }
  iree_host_size_t scalar_count = 0;
  iree_host_size_t word_count = 0;
  for (iree_host_size_t i = 0; i < liveness->interval_count; ++i) {
    const loom_liveness_interval_t* interval = &liveness->intervals[i];
    if (!loom_low_physical_interval_is_scalar(descriptor_set, interval)) {
      continue;
    }
    ++scalar_count;
    word_count += iree_bitmap_calculate_words(
        descriptor_set->reg_classes[interval->value_class.register_class_id]
            .allocatable_count);
  }
  if (scalar_count == 0) {
    return iree_ok_status();
  }
  if (word_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "physical preference word offsets exceed uint32_t");
  }
  uint32_t* offsets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, liveness->interval_count, sizeof(*offsets), (void**)&offsets));
  memset(offsets, 0xFF, liveness->interval_count * sizeof(*offsets));
  uint64_t* penalty_words = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, word_count, sizeof(*penalty_words), (void**)&penalty_words));
  memset(penalty_words, 0, word_count * sizeof(*penalty_words));
  uint64_t* affinity_words = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, word_count, sizeof(*affinity_words), (void**)&affinity_words));
  memset(affinity_words, 0, word_count * sizeof(*affinity_words));
  uint64_t* reservation_words = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, word_count,
                                                 sizeof(*reservation_words),
                                                 (void**)&reservation_words));
  memset(reservation_words, 0, word_count * sizeof(*reservation_words));
  const iree_arena_checkpoint_t checkpoint = iree_arena_checkpoint_save(arena);
  iree_status_t status = loom_low_physical_domains_build_preferences(
      descriptor_set, liveness, unit_liveness, placement, scalar_count, arena,
      offsets, penalty_words, affinity_words, reservation_words);
  iree_arena_checkpoint_restore(&checkpoint);
  if (iree_status_is_ok(status)) {
    *out_domains = (loom_low_allocation_physical_domains_t){
        .offsets = offsets,
        .penalty_words = penalty_words,
        .affinity_words = affinity_words,
        .reservation_words = reservation_words,
    };
  }
  return status;
}
