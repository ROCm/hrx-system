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
  // First storage point of the component's conservative horizon.
  uint32_t start;
  // Exclusive final storage point of that horizon.
  uint32_t end;
  // Union rank, independent of the selected representative's candidate set.
  uint8_t rank;
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
  // Union of candidates appearing in strictly smaller effective domains.
  uint64_t* narrower;
  // Point-table slots queried by members of this domain.
  struct {
    // First slot in the point and query tables.
    iree_host_size_t start;
    // Number of slots queried by this domain.
    iree_host_size_t count;
  } queries;
  // Point-table slots updated by members of this domain.
  struct {
    // First entry in the indirect action table.
    iree_host_size_t start;
    // Number of point-table updates performed by this domain.
    iree_host_size_t count;
  } actions;
} loom_low_physical_domain_t;

static bool loom_low_physical_bits_test(const uint64_t* words, uint32_t bit) {
  return (words[bit / 64] & (UINT64_C(1) << (bit % 64))) != 0;
}

static void loom_low_physical_bits_set(uint64_t* words, uint32_t bit) {
  words[bit / 64] |= UINT64_C(1) << (bit % 64);
}

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

static bool loom_low_physical_component_lifetime_less(
    loom_low_physical_component_t* const* lhs,
    loom_low_physical_component_t* const* rhs) {
  if ((*lhs)->start != (*rhs)->start) {
    return (*lhs)->start < (*rhs)->start;
  }
  if ((*lhs)->end != (*rhs)->end) {
    return (*lhs)->end < (*rhs)->end;
  }
  return (*lhs)->interval_index < (*rhs)->interval_index;
}

LOOM_DEFINE_ADAPTIVE_SORT(loom_low_physical_component_lifetime_sort,
                          loom_low_physical_component_t*,
                          loom_low_physical_component_lifetime_less)

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

static void loom_low_physical_domain_sweep(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness,
    const loom_low_physical_component_t* components,
    loom_low_physical_component_t* const* order, iree_host_size_t count,
    const loom_low_physical_domain_t* domains, const uint32_t* queries,
    const iree_host_size_t* actions, uint32_t* points, bool backwards,
    const uint32_t* offsets, uint64_t* words) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    const loom_low_physical_component_t* component =
        order[backwards ? count - 1 - i : i];
    const loom_low_physical_domain_t* domain = &domains[component->domain];
    for (iree_host_size_t q = domain->queries.start;
         q < domain->queries.start + domain->queries.count; ++q) {
      const bool overlaps =
          backwards ? points[q] < component->end : points[q] > component->start;
      if (!overlaps) {
        continue;
      }
      uint32_t member = component->parent;
      do {
        const uint32_t interval_index = components[member].interval_index;
        const loom_low_reg_class_t* reg_class =
            &descriptor_set->reg_classes[liveness->intervals[interval_index]
                                             .value_class.register_class_id];
        // The component intersection proves membership in every member class.
        const uint16_t ordinal =
            descriptor_set->physical_register_candidate_ordinals
                [reg_class->candidate_lookup.ordinal_start + queries[q] -
                 reg_class->candidate_lookup.register_base];
        loom_low_physical_bits_set(words + offsets[interval_index], ordinal);
        member = components[member].next;
      } while (member != component->parent);
    }
    for (iree_host_size_t a = domain->actions.start;
         a < domain->actions.start + domain->actions.count; ++a) {
      uint32_t* point = &points[actions[a]];
      *point = backwards ? component->start : iree_max(*point, component->end);
    }
  }
}

static iree_status_t loom_low_physical_domains_build_preferences(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_liveness_analysis_t* liveness,
    const loom_low_allocation_unit_liveness_t* unit_liveness,
    const loom_low_placement_table_t* placement, iree_host_size_t scalar_count,
    iree_arena_allocator_t* arena, uint32_t* offsets, uint64_t* words) {
  const iree_host_size_t word_count =
      iree_bitmap_calculate_words(descriptor_set->physical_register_count);
  uint64_t** class_words = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, descriptor_set->reg_class_count,
                                sizeof(*class_words), (void**)&class_words));
  memset(class_words, 0,
         descriptor_set->reg_class_count * sizeof(*class_words));
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
  // Before affinity merging, this array indexes one representative per used
  // class. It is then reused for root ordering during domain construction.
  iree_host_size_t used_class_count = 0;
  bool has_overlapping_domains = false;
  uint32_t component_count = 0;
  uint32_t offset = 0;
  iree_status_t status = iree_ok_status();
  for (loom_value_ordinal_t v = 0;
       v < liveness->value_count && iree_status_is_ok(status); ++v) {
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
      status = iree_arena_allocate_array(arena, word_count, sizeof(uint64_t),
                                         (void**)&class_words[class_id]);
      if (!iree_status_is_ok(status)) {
        continue;
      }
      memset(class_words[class_id], 0, word_count * sizeof(uint64_t));
      const loom_low_reg_class_t* reg_class =
          &descriptor_set->reg_classes[class_id];
      for (uint16_t r = 0; r < reg_class->allocatable_count; ++r) {
        loom_low_physical_bits_set(
            class_words[class_id],
            loom_low_descriptor_set_physical_register_candidate(descriptor_set,
                                                                class_id, r));
      }
      for (iree_host_size_t c = 0;
           c < used_class_count && !has_overlapping_domains; ++c) {
        uint64_t shared = 0;
        uint64_t difference = 0;
        for (iree_host_size_t w = 0; w < word_count; ++w) {
          shared |= class_words[class_id][w] & order[c]->registers[w];
          difference |= class_words[class_id][w] ^ order[c]->registers[w];
        }
        has_overlapping_domains = shared != 0 && difference != 0;
      }
      order[used_class_count++] = &components[component_count];
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
        .start = start,
        .end = end,
        .domain = class_id,
        .registers = class_words[class_id],
    };
    ++component_count;
    offsets[index] = offset;
    offset += iree_bitmap_calculate_words(
        descriptor_set->reg_classes[class_id].allocatable_count);
  }
  IREE_RETURN_IF_ERROR(status);
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
        loom_low_physical_bits_set(words + offsets[interval_index], r);
      }
    }
  }
  // Intern equal effective domains once, independently of affinity-component
  // identity. Distinct dynamic values retain separate interference/lifetimes.
  loom_low_physical_component_domain_sort(order, root_count);
  iree_host_size_t domain_capacity = 0;
  for (iree_host_size_t i = 0; i < root_count; ++i) {
    if (i == 0 || order[i - 1]->domain != order[i]->domain) {
      ++domain_capacity;
    }
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
  // Only distinct domains are compared. Intervals never perform pairwise
  // lifetime or candidate-set searches during either sweep.
  for (iree_host_size_t a = 0; a < domain_count; ++a) {
    for (iree_host_size_t b = 0; b < domain_count; ++b) {
      uint64_t outside = 0;
      for (iree_host_size_t w = 0; w < word_count; ++w) {
        outside |= domains[a].registers[w] & ~domains[b].registers[w];
      }
      const bool subset = a != b && outside == 0;
      subsets[a * domain_count + b] = subset;
      if (!subset) {
        continue;
      }
      for (iree_host_size_t w = 0; w < word_count; ++w) {
        domains[b].narrower[w] |= domains[a].registers[w];
        domains[a].actions.count +=
            iree_math_count_ones_u64(domains[a].registers[w]);
      }
    }
  }
  iree_host_size_t query_count = 0;
  iree_host_size_t action_count = 0;
  for (iree_host_size_t d = 0; d < domain_count; ++d) {
    domains[d].queries.start = query_count;
    for (iree_host_size_t w = 0; w < word_count; ++w) {
      domains[d].queries.count +=
          iree_math_count_ones_u64(domains[d].narrower[w]);
    }
    query_count += domains[d].queries.count;
    domains[d].actions.start = action_count;
    action_count += domains[d].actions.count;
  }
  uint32_t* queries = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, query_count, sizeof(*queries), (void**)&queries));
  iree_host_size_t* actions = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, action_count, sizeof(*actions), (void**)&actions));
  uint32_t* points = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, query_count, sizeof(*points), (void**)&points));
  for (iree_host_size_t d = 0; d < domain_count; ++d) {
    iree_host_size_t q = domains[d].queries.start;
    for (uint32_t r = 0; r < descriptor_set->physical_register_count; ++r) {
      if (loom_low_physical_bits_test(domains[d].narrower, r)) {
        queries[q++] = r;
      }
    }
  }
  for (iree_host_size_t a = 0; a < domain_count; ++a) {
    iree_host_size_t cursor = domains[a].actions.start;
    for (iree_host_size_t b = 0; b < domain_count; ++b) {
      if (!subsets[a * domain_count + b]) {
        continue;
      }
      for (iree_host_size_t q = domains[b].queries.start;
           q < domains[b].queries.start + domains[b].queries.count; ++q) {
        if (loom_low_physical_bits_test(domains[a].registers, queries[q])) {
          actions[cursor++] = q;
        }
      }
    }
  }
  if (query_count != 0) {
    iree_host_size_t relevant_count = 0;
    for (iree_host_size_t i = 0; i < root_count; ++i) {
      const loom_low_physical_domain_t* domain = &domains[order[i]->domain];
      if (domain->queries.count != 0 || domain->actions.count != 0) {
        order[relevant_count++] = order[i];
      }
    }
    loom_low_physical_component_lifetime_sort(order, relevant_count);
    memset(points, 0, query_count * sizeof(*points));
    loom_low_physical_domain_sweep(descriptor_set, liveness, components, order,
                                   relevant_count, domains, queries, actions,
                                   points, false, offsets, words);
    memset(points, 0xFF, query_count * sizeof(*points));
    loom_low_physical_domain_sweep(descriptor_set, liveness, components, order,
                                   relevant_count, domains, queries, actions,
                                   points, true, offsets, words);
  }
  // Uniform rows carry no preference. In particular, an entirely discouraged
  // class must preserve the ordinary first-candidate search termination.
  for (uint32_t i = 0; i < component_count; ++i) {
    const uint32_t index = components[i].interval_index;
    const uint16_t class_id =
        liveness->intervals[index].value_class.register_class_id;
    const uint16_t count =
        descriptor_set->reg_classes[class_id].allocatable_count;
    uint32_t set_count = 0;
    for (iree_host_size_t w = 0; w < iree_bitmap_calculate_words(count); ++w) {
      set_count += iree_math_count_ones_u64(words[offsets[index] + w]);
    }
    if (set_count == 0 || set_count == count) {
      offsets[index] = UINT32_MAX;
    }
  }
  return iree_ok_status();
}

const uint64_t* loom_low_allocation_physical_domains_for_interval(
    const loom_low_allocation_physical_domains_t* domains,
    const loom_liveness_analysis_t* liveness,
    const loom_liveness_interval_t* interval) {
  if (!domains || !domains->offsets) {
    return NULL;
  }
  const uint32_t offset = domains->offsets[interval - liveness->intervals];
  return offset == UINT32_MAX ? NULL : domains->words + offset;
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
  uint64_t* words = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, word_count, sizeof(*words), (void**)&words));
  memset(words, 0, word_count * sizeof(*words));
  const iree_arena_checkpoint_t checkpoint = iree_arena_checkpoint_save(arena);
  iree_status_t status = loom_low_physical_domains_build_preferences(
      descriptor_set, liveness, unit_liveness, placement, scalar_count, arena,
      offsets, words);
  iree_arena_checkpoint_restore(&checkpoint);
  if (iree_status_is_ok(status)) {
    *out_domains = (loom_low_allocation_physical_domains_t){
        .offsets = offsets,
        .words = words,
    };
  }
  return status;
}
