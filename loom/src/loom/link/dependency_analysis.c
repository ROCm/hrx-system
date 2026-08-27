// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/dependency_analysis.h"

#include <string.h>

#include "loom/link/func_contract_projection.h"
#include "loom/rewrite/remap.h"

typedef struct loom_link_dependency_builder_t {
  // Frozen index being analyzed.
  const loom_link_module_index_t* index;
  // Caller arena retaining the returned report.
  iree_arena_allocator_t* output_arena;
  // Invocation-local arena retaining dense analysis scratch.
  iree_arena_allocator_t* scratch_arena;
  // Bitset indexed by provider ordinal for declared direct libraries.
  uint64_t* direct_provider_bits;
  // Bitset indexed by provider ordinal for direct libraries with a use.
  uint64_t* used_provider_bits;
  // Bitset indexed by symbol ordinal for unique exact requirements.
  uint64_t* exact_requirement_bits;
  // Requirement ordinal indexed by exact target symbol ordinal.
  iree_host_size_t* exact_requirement_ordinals;
  // Demand occurrence counts indexed by template family ordinal.
  iree_host_size_t* template_demand_counts;
  // First source symbol indexed by template family ordinal.
  iree_host_size_t* template_first_source_symbols;
  // First source root indexed by template family ordinal.
  uint8_t* template_first_source_roots;
  // Mutable report under construction.
  loom_link_dependency_analysis_t analysis;
} loom_link_dependency_builder_t;

static bool loom_link_dependency_bit_test(const uint64_t* bits,
                                          iree_host_size_t ordinal) {
  return bits != NULL &&
         (bits[ordinal >> 6] & (UINT64_C(1) << (ordinal & 63u))) != 0;
}

static bool loom_link_dependency_bit_test_and_set(uint64_t* bits,
                                                  iree_host_size_t ordinal) {
  const uint64_t mask = UINT64_C(1) << (ordinal & 63u);
  uint64_t* word = &bits[ordinal >> 6];
  const bool was_set = (*word & mask) != 0;
  *word |= mask;
  return was_set;
}

static iree_status_t loom_link_dependency_allocate_bits(
    iree_arena_allocator_t* arena, iree_host_size_t bit_count,
    uint64_t** out_bits) {
  *out_bits = NULL;
  const iree_host_size_t word_count =
      bit_count / 64u + (bit_count % 64u != 0 ? 1u : 0u);
  if (word_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, word_count, sizeof(**out_bits), (void**)out_bits));
  memset(*out_bits, 0, word_count * sizeof(**out_bits));
  return iree_ok_status();
}

static bool loom_link_dependency_symbol_is_exact_requirement(
    const loom_link_module_index_t* index,
    const loom_link_module_index_symbol_t* symbol) {
  const loom_link_module_index_provider_t* provider =
      loom_link_module_index_symbol_provider(index, symbol);
  IREE_ASSERT(provider);
  if (provider->role != LOOM_LINK_PROVIDER_ROLE_INPUT ||
      symbol->identity != LOOM_LINK_SYMBOL_IDENTITY_GLOBAL ||
      !iree_any_bit_set(symbol->flags, LOOM_LINK_SYMBOL_FLAG_DECLARATION) ||
      iree_any_bit_set(symbol->flags, LOOM_LINK_SYMBOL_FLAG_IMPORT |
                                          LOOM_LINK_SYMBOL_FLAG_CONFIG) ||
      symbol->template_family_ordinal !=
          LOOM_LINK_TEMPLATE_FAMILY_ORDINAL_INVALID ||
      !iree_any_bit_set(symbol->facets.schema.interfaces,
                        LOOM_SYMBOL_INTERFACE_FUNC_LIKE) ||
      iree_any_bit_set(symbol->facets.schema.interfaces,
                       LOOM_SYMBOL_INTERFACE_TARGET)) {
    return false;
  }
  return true;
}

static void loom_link_dependency_mark_exact_occurrence(
    loom_link_dependency_builder_t* builder,
    const loom_link_module_index_symbol_t* target,
    iree_host_size_t* requirement_count) {
  if (!loom_link_dependency_bit_test_and_set(builder->exact_requirement_bits,
                                             target->ordinal)) {
    ++*requirement_count;
  }
  ++builder->analysis.exact_occurrence_count;
}

static void loom_link_dependency_visit_input_exact_occurrences(
    loom_link_dependency_builder_t* builder, bool populate,
    iree_host_size_t* requirement_count) {
  const iree_host_size_t module_count =
      loom_link_module_index_module_count(builder->index);
  for (iree_host_size_t module_ordinal = 0; module_ordinal < module_count;
       ++module_ordinal) {
    const loom_link_module_index_module_t* module =
        loom_link_module_index_module_at(builder->index, module_ordinal);
    const loom_link_module_index_provider_t* provider =
        loom_link_module_index_provider_at(builder->index,
                                           module->provider_ordinal);
    IREE_ASSERT(module);
    IREE_ASSERT(provider);
    if (provider->role != LOOM_LINK_PROVIDER_ROLE_INPUT) {
      continue;
    }

    for (uint32_t i = 0; i < module->dependencies.root_count; ++i) {
      const uint32_t target_symbol_id = module->dependencies.values[i];
      const loom_link_module_index_symbol_t* target =
          loom_link_module_index_symbol_at(
              builder->index, module->symbol_start_ordinal + target_symbol_id);
      IREE_ASSERT(target);
      if (!loom_link_dependency_symbol_is_exact_requirement(builder->index,
                                                            target)) {
        continue;
      }
      if (!populate) {
        loom_link_dependency_mark_exact_occurrence(builder, target,
                                                   requirement_count);
        continue;
      }
      loom_link_dependency_requirement_t* requirement =
          (loom_link_dependency_requirement_t*)
              builder->analysis.requirements.values +
          builder->exact_requirement_ordinals[target->ordinal];
      if (requirement->occurrence_count++ == 0 && !requirement->exported) {
        requirement->first_source_symbol_ordinal =
            LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
        requirement->first_source_root_region_index_plus_one = 0;
      }
      requirement->target_interfaces |=
          module->dependencies.target_interfaces[i];
    }

    for (iree_host_size_t local_symbol_ordinal = 0;
         local_symbol_ordinal < module->symbol_count; ++local_symbol_ordinal) {
      const loom_link_module_index_symbol_t* source =
          loom_link_module_index_symbol_at(
              builder->index,
              module->symbol_start_ordinal + local_symbol_ordinal);
      IREE_ASSERT(source);
      const uint32_t first = source->dependencies.first;
      for (uint32_t i = 0; i < source->dependencies.count; ++i) {
        const iree_host_size_t occurrence_ordinal = first + i;
        const uint32_t target_symbol_id =
            module->dependencies.values[occurrence_ordinal];
        const loom_link_module_index_symbol_t* target =
            loom_link_module_index_symbol_at(
                builder->index,
                module->symbol_start_ordinal + target_symbol_id);
        IREE_ASSERT(target);
        if (!loom_link_dependency_symbol_is_exact_requirement(builder->index,
                                                              target)) {
          continue;
        }
        if (!populate) {
          loom_link_dependency_mark_exact_occurrence(builder, target,
                                                     requirement_count);
          continue;
        }
        loom_link_dependency_requirement_t* requirement =
            (loom_link_dependency_requirement_t*)
                builder->analysis.requirements.values +
            builder->exact_requirement_ordinals[target->ordinal];
        if (requirement->occurrence_count++ == 0 && !requirement->exported) {
          requirement->first_source_symbol_ordinal = source->ordinal;
          requirement->first_source_root_region_index_plus_one =
              module->dependencies
                  .source_root_region_indices_plus_one[occurrence_ordinal];
        }
        requirement->target_interfaces |=
            module->dependencies.target_interfaces[occurrence_ordinal];
      }
    }
  }
}

static void loom_link_dependency_mark_input_exports(
    loom_link_dependency_builder_t* builder,
    iree_host_size_t* requirement_count) {
  const loom_link_module_index_symbol_ordinal_list_t input_exports =
      loom_link_module_index_input_exports(builder->index);
  for (iree_host_size_t i = 0; i < input_exports.count; ++i) {
    const loom_link_module_index_symbol_t* symbol =
        loom_link_module_index_symbol_at(builder->index,
                                         input_exports.values[i]);
    IREE_ASSERT(symbol);
    if (!loom_link_dependency_symbol_is_exact_requirement(builder->index,
                                                          symbol)) {
      continue;
    }
    if (!loom_link_dependency_bit_test_and_set(builder->exact_requirement_bits,
                                               symbol->ordinal)) {
      ++*requirement_count;
    }
  }
}

static iree_status_t loom_link_dependency_collect_template_demands(
    loom_link_dependency_builder_t* builder,
    iree_host_size_t* out_requirement_count) {
  *out_requirement_count = 0;
  const iree_host_size_t family_count =
      loom_link_module_index_template_family_count(builder->index);
  if (family_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(builder->scratch_arena, family_count,
                                  sizeof(*builder->template_demand_counts),
                                  (void**)&builder->template_demand_counts));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->scratch_arena, family_count,
        sizeof(*builder->template_first_source_symbols),
        (void**)&builder->template_first_source_symbols));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->scratch_arena, family_count,
        sizeof(*builder->template_first_source_roots),
        (void**)&builder->template_first_source_roots));
    memset(builder->template_demand_counts, 0,
           family_count * sizeof(*builder->template_demand_counts));
    for (iree_host_size_t i = 0; i < family_count; ++i) {
      builder->template_first_source_symbols[i] =
          LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
      builder->template_first_source_roots[i] = 0;
    }
  }

  const iree_host_size_t module_count =
      loom_link_module_index_module_count(builder->index);
  for (iree_host_size_t module_ordinal = 0; module_ordinal < module_count;
       ++module_ordinal) {
    const loom_link_module_index_module_t* module =
        loom_link_module_index_module_at(builder->index, module_ordinal);
    const loom_link_module_index_provider_t* provider =
        loom_link_module_index_provider_at(builder->index,
                                           module->provider_ordinal);
    IREE_ASSERT(module);
    IREE_ASSERT(provider);
    if (provider->role != LOOM_LINK_PROVIDER_ROLE_INPUT) {
      continue;
    }
    for (iree_host_size_t local_symbol_ordinal = 0;
         local_symbol_ordinal < module->symbol_count; ++local_symbol_ordinal) {
      const loom_link_module_index_symbol_t* source =
          loom_link_module_index_symbol_at(
              builder->index,
              module->symbol_start_ordinal + local_symbol_ordinal);
      IREE_ASSERT(source);
      const uint32_t first = source->template_demands.first;
      for (uint32_t i = 0; i < source->template_demands.count; ++i) {
        const iree_host_size_t occurrence_ordinal = first + i;
        const loom_link_template_family_ordinal_t family_ordinal =
            module->template_demands.values[occurrence_ordinal];
        IREE_ASSERT_LT(family_ordinal, family_count);
        iree_host_size_t* demand_count =
            &builder->template_demand_counts[family_ordinal];
        if ((*demand_count)++ == 0) {
          ++*out_requirement_count;
          builder->template_first_source_symbols[family_ordinal] =
              source->ordinal;
          builder->template_first_source_roots[family_ordinal] =
              module->template_demands
                  .source_root_region_indices_plus_one[occurrence_ordinal];
        }
        ++builder->analysis.template_demand_occurrence_count;
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_link_dependency_prepare_direct_providers(
    loom_link_dependency_builder_t* builder,
    const loom_link_dependency_analysis_options_t* options) {
  const iree_host_size_t provider_count =
      loom_link_module_index_provider_count(builder->index);
  IREE_RETURN_IF_ERROR(loom_link_dependency_allocate_bits(
      builder->scratch_arena, provider_count, &builder->direct_provider_bits));
  IREE_RETURN_IF_ERROR(loom_link_dependency_allocate_bits(
      builder->scratch_arena, provider_count, &builder->used_provider_bits));
  const iree_host_size_t direct_count =
      options ? options->direct_provider_count : 0;
  const iree_host_size_t* direct_ordinals =
      options ? options->direct_provider_ordinals : NULL;
  if (direct_count != 0 && direct_ordinals == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "direct provider count is non-zero but ordinals are NULL");
  }
  iree_host_size_t* copied_ordinals = NULL;
  if (direct_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->output_arena, direct_count, sizeof(*copied_ordinals),
        (void**)&copied_ordinals));
  }
  for (iree_host_size_t i = 0; i < direct_count; ++i) {
    const iree_host_size_t provider_ordinal = direct_ordinals[i];
    const loom_link_module_index_provider_t* provider =
        loom_link_module_index_provider_at(builder->index, provider_ordinal);
    if (provider == NULL) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "direct provider ordinal is out of range");
    }
    if (provider->role != LOOM_LINK_PROVIDER_ROLE_LIBRARY) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "direct provider is not a library");
    }
    if (loom_link_dependency_bit_test_and_set(builder->direct_provider_bits,
                                              provider_ordinal)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "direct provider ordinal is duplicated");
    }
    copied_ordinals[i] = provider_ordinal;
  }
  builder->analysis.direct_providers.values = copied_ordinals;
  builder->analysis.direct_providers.count = direct_count;
  return iree_ok_status();
}

static loom_link_dependency_candidate_origin_t
loom_link_dependency_candidate_origin(
    const loom_link_dependency_builder_t* builder,
    const loom_link_module_index_provider_t* provider) {
  if (provider->role == LOOM_LINK_PROVIDER_ROLE_INPUT) {
    return LOOM_LINK_DEPENDENCY_CANDIDATE_INPUT;
  }
  return loom_link_dependency_bit_test(builder->direct_provider_bits,
                                       provider->ordinal)
             ? LOOM_LINK_DEPENDENCY_CANDIDATE_DIRECT_LIBRARY
             : LOOM_LINK_DEPENDENCY_CANDIDATE_TRANSITIVE_LIBRARY;
}

static bool loom_link_dependency_exact_candidate(
    const loom_link_dependency_builder_t* builder,
    const loom_link_module_index_symbol_t* requirement,
    const loom_link_module_index_symbol_t* candidate) {
  if (candidate == requirement ||
      candidate->template_family_ordinal !=
          LOOM_LINK_TEMPLATE_FAMILY_ORDINAL_INVALID ||
      iree_any_bit_set(candidate->flags, LOOM_LINK_SYMBOL_FLAG_IMPORT |
                                             LOOM_LINK_SYMBOL_FLAG_CONFIG)) {
    return false;
  }
  const loom_link_module_index_provider_t* provider =
      loom_link_module_index_symbol_provider(builder->index, candidate);
  IREE_ASSERT(provider);
  if (provider->role == LOOM_LINK_PROVIDER_ROLE_INPUT) {
    return iree_any_bit_set(candidate->flags,
                            LOOM_LINK_SYMBOL_FLAG_CONCRETE_DEFINITION);
  }
  if (candidate->identity == LOOM_LINK_SYMBOL_IDENTITY_PRIVATE) {
    return iree_any_bit_set(candidate->flags,
                            LOOM_LINK_SYMBOL_FLAG_CONCRETE_DEFINITION);
  }
  return candidate->identity == LOOM_LINK_SYMBOL_IDENTITY_GLOBAL &&
         iree_any_bit_set(candidate->flags, LOOM_LINK_SYMBOL_FLAG_EXPORT) &&
         iree_any_bit_set(candidate->flags,
                          LOOM_LINK_SYMBOL_FLAG_DECLARATION |
                              LOOM_LINK_SYMBOL_FLAG_CONCRETE_DEFINITION);
}

static bool loom_link_dependency_exact_candidate_accessible(
    const loom_link_module_index_symbol_t* candidate,
    loom_link_dependency_candidate_origin_t origin) {
  return origin == LOOM_LINK_DEPENDENCY_CANDIDATE_INPUT ||
         candidate->identity == LOOM_LINK_SYMBOL_IDENTITY_GLOBAL;
}

static bool loom_link_dependency_exact_candidate_is_definition(
    const loom_link_module_index_symbol_t* candidate) {
  return iree_any_bit_set(candidate->flags,
                          LOOM_LINK_SYMBOL_FLAG_CONCRETE_DEFINITION);
}

static iree_host_size_t loom_link_dependency_exact_candidate_count(
    const loom_link_dependency_builder_t* builder,
    const loom_link_module_index_symbol_t* requirement) {
  iree_host_size_t count = 0;
  const loom_link_module_index_symbol_t* candidate =
      loom_link_module_index_lookup_name(builder->index, requirement->name);
  while (candidate != NULL) {
    if (loom_link_dependency_exact_candidate(builder, requirement, candidate)) {
      ++count;
    }
    candidate =
        loom_link_module_index_next_same_name(builder->index, candidate);
  }
  return count;
}

static iree_host_size_t loom_link_dependency_template_candidate_count(
    const loom_link_dependency_builder_t* builder,
    loom_link_template_family_ordinal_t family_ordinal) {
  const loom_link_module_index_template_family_t* family =
      loom_link_module_index_template_family_at(builder->index, family_ordinal);
  IREE_ASSERT(family);
  return family->providers.count;
}

static iree_status_t loom_link_dependency_initialize_requirements(
    loom_link_dependency_builder_t* builder,
    iree_host_size_t exact_requirement_count,
    iree_host_size_t template_requirement_count) {
  const iree_host_size_t symbol_count =
      loom_link_module_index_symbol_count(builder->index);
  if (symbol_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->scratch_arena, symbol_count,
        sizeof(*builder->exact_requirement_ordinals),
        (void**)&builder->exact_requirement_ordinals));
    for (iree_host_size_t i = 0; i < symbol_count; ++i) {
      builder->exact_requirement_ordinals[i] =
          LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
    }
  }
  iree_host_size_t requirement_count = 0;
  if (!iree_host_size_checked_add(exact_requirement_count,
                                  template_requirement_count,
                                  &requirement_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "dependency requirement count overflow");
  }
  loom_link_dependency_requirement_t* requirements = NULL;
  if (requirement_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->output_arena, requirement_count, sizeof(*requirements),
        (void**)&requirements));
    memset(requirements, 0, requirement_count * sizeof(*requirements));
  }

  iree_host_size_t requirement_ordinal = 0;
  for (iree_host_size_t symbol_ordinal = 0; symbol_ordinal < symbol_count;
       ++symbol_ordinal) {
    if (!loom_link_dependency_bit_test(builder->exact_requirement_bits,
                                       symbol_ordinal)) {
      continue;
    }
    IREE_ASSERT(requirement_ordinal < exact_requirement_count);
    builder->exact_requirement_ordinals[symbol_ordinal] = requirement_ordinal;
    const loom_link_module_index_symbol_t* symbol =
        loom_link_module_index_symbol_at(builder->index, symbol_ordinal);
    IREE_ASSERT(symbol);
    const bool exported =
        iree_any_bit_set(symbol->flags, LOOM_LINK_SYMBOL_FLAG_EXPORT);
    requirements[requirement_ordinal++] = (loom_link_dependency_requirement_t){
        .kind = LOOM_LINK_DEPENDENCY_REQUIREMENT_EXACT_SYMBOL,
        .ownership = LOOM_LINK_DEPENDENCY_OWNERSHIP_UNSATISFIED,
        .resolution = LOOM_LINK_DEPENDENCY_RESOLUTION_UNRESOLVED,
        .exported = exported,
        .first_source_symbol_ordinal =
            exported ? symbol_ordinal : LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL,
        .target_interfaces = exported ? symbol->facets.schema.interfaces
                                      : (loom_symbol_interface_flags_t)0,
        .target =
            {
                .symbol_ordinal = symbol_ordinal,
                .template_family_ordinal =
                    LOOM_LINK_TEMPLATE_FAMILY_ORDINAL_INVALID,
            },
    };
  }
  IREE_ASSERT_EQ(requirement_ordinal, exact_requirement_count);

  const iree_host_size_t family_count =
      loom_link_module_index_template_family_count(builder->index);
  for (iree_host_size_t family_ordinal = 0; family_ordinal < family_count;
       ++family_ordinal) {
    if (builder->template_demand_counts[family_ordinal] == 0) {
      continue;
    }
    requirements[requirement_ordinal++] = (loom_link_dependency_requirement_t){
        .kind = LOOM_LINK_DEPENDENCY_REQUIREMENT_TEMPLATE_FAMILY,
        .ownership = LOOM_LINK_DEPENDENCY_OWNERSHIP_OPEN,
        .resolution = LOOM_LINK_DEPENDENCY_RESOLUTION_NOT_APPLICABLE,
        .occurrence_count = builder->template_demand_counts[family_ordinal],
        .first_source_symbol_ordinal =
            builder->template_first_source_symbols[family_ordinal],
        .first_source_root_region_index_plus_one =
            builder->template_first_source_roots[family_ordinal],
        .target =
            {
                .symbol_ordinal = LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL,
                .template_family_ordinal =
                    (loom_link_template_family_ordinal_t)family_ordinal,
            },
    };
  }
  IREE_ASSERT_EQ(requirement_ordinal, requirement_count);
  builder->analysis.requirements.values = requirements;
  builder->analysis.requirements.count = requirement_count;
  return iree_ok_status();
}

static iree_status_t loom_link_dependency_allocate_candidates(
    loom_link_dependency_builder_t* builder) {
  loom_link_dependency_requirement_t* requirements =
      (loom_link_dependency_requirement_t*)
          builder->analysis.requirements.values;
  iree_host_size_t candidate_count = 0;
  for (iree_host_size_t i = 0; i < builder->analysis.requirements.count; ++i) {
    loom_link_dependency_requirement_t* requirement = &requirements[i];
    requirement->candidates.first = candidate_count;
    if (requirement->kind == LOOM_LINK_DEPENDENCY_REQUIREMENT_EXACT_SYMBOL) {
      const loom_link_module_index_symbol_t* target =
          loom_link_module_index_symbol_at(builder->index,
                                           requirement->target.symbol_ordinal);
      requirement->candidates.count =
          loom_link_dependency_exact_candidate_count(builder, target);
    } else {
      requirement->candidates.count =
          loom_link_dependency_template_candidate_count(
              builder, requirement->target.template_family_ordinal);
    }
    if (!iree_host_size_checked_add(
            candidate_count, requirement->candidates.count, &candidate_count)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "dependency candidate count overflow");
    }
  }
  loom_link_dependency_candidate_t* candidates = NULL;
  if (candidate_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(builder->output_arena, candidate_count,
                                  sizeof(*candidates), (void**)&candidates));
    memset(candidates, 0, candidate_count * sizeof(*candidates));
  }
  builder->analysis.candidates.values = candidates;
  builder->analysis.candidates.count = candidate_count;
  return iree_ok_status();
}

static void loom_link_dependency_fill_exact_candidates(
    loom_link_dependency_builder_t* builder,
    loom_link_dependency_requirement_t* requirement) {
  loom_link_dependency_candidate_t* candidates =
      (loom_link_dependency_candidate_t*)builder->analysis.candidates.values;
  const loom_link_module_index_symbol_t* target =
      loom_link_module_index_symbol_at(builder->index,
                                       requirement->target.symbol_ordinal);
  iree_host_size_t position = requirement->candidates.first;
  const loom_link_module_index_symbol_t* candidate =
      loom_link_module_index_lookup_name(builder->index, target->name);
  while (candidate != NULL) {
    if (loom_link_dependency_exact_candidate(builder, target, candidate)) {
      const loom_link_module_index_provider_t* provider =
          loom_link_module_index_symbol_provider(builder->index, candidate);
      candidates[position++] = (loom_link_dependency_candidate_t){
          .symbol_ordinal = candidate->ordinal,
          .provider_ordinal = provider->ordinal,
          .origin = loom_link_dependency_candidate_origin(builder, provider),
      };
    }
    candidate =
        loom_link_module_index_next_same_name(builder->index, candidate);
  }
  IREE_ASSERT_EQ(position,
                 requirement->candidates.first + requirement->candidates.count);
}

static void loom_link_dependency_fill_template_candidates(
    loom_link_dependency_builder_t* builder,
    loom_link_dependency_requirement_t* requirement) {
  loom_link_dependency_candidate_t* candidates =
      (loom_link_dependency_candidate_t*)builder->analysis.candidates.values;
  const loom_link_module_index_template_family_t* family =
      loom_link_module_index_template_family_at(
          builder->index, requirement->target.template_family_ordinal);
  IREE_ASSERT(family);
  iree_host_size_t position = requirement->candidates.first;
  iree_host_size_t symbol_ordinal = family->providers.first_symbol_ordinal;
  while (symbol_ordinal != LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL) {
    const loom_link_module_index_symbol_t* candidate =
        loom_link_module_index_symbol_at(builder->index, symbol_ordinal);
    const loom_link_module_index_provider_t* provider =
        loom_link_module_index_symbol_provider(builder->index, candidate);
    IREE_ASSERT(candidate);
    IREE_ASSERT(provider);
    candidates[position++] = (loom_link_dependency_candidate_t){
        .symbol_ordinal = candidate->ordinal,
        .provider_ordinal = provider->ordinal,
        .origin = loom_link_dependency_candidate_origin(builder, provider),
        .compatible = true,
    };
    symbol_ordinal = candidate->next.template_provider_ordinal;
  }
  IREE_ASSERT_EQ(position,
                 requirement->candidates.first + requirement->candidates.count);
}

static bool loom_link_dependency_candidate_has_func_contract(
    const loom_link_module_index_symbol_t* requirement,
    const loom_link_module_index_symbol_t* candidate) {
  return iree_all_bits_set(candidate->facets.schema.interfaces,
                           requirement->facets.schema.interfaces) &&
         iree_any_bit_set(candidate->facets.schema.interfaces,
                          LOOM_SYMBOL_INTERFACE_FUNC_LIKE);
}

static iree_status_t loom_link_dependency_project_exact_contracts(
    loom_link_dependency_builder_t* builder,
    loom_link_func_contract_projection_t* projection) {
  loom_link_dependency_requirement_t* requirements =
      (loom_link_dependency_requirement_t*)
          builder->analysis.requirements.values;
  const loom_link_dependency_candidate_t* candidates =
      builder->analysis.candidates.values;
  for (iree_host_size_t i = 0; i < builder->analysis.requirements.count; ++i) {
    const loom_link_dependency_requirement_t* requirement = &requirements[i];
    if (requirement->kind != LOOM_LINK_DEPENDENCY_REQUIREMENT_EXACT_SYMBOL) {
      continue;
    }
    const loom_link_module_index_symbol_t* target =
        loom_link_module_index_symbol_at(builder->index,
                                         requirement->target.symbol_ordinal);
    const loom_link_func_contract_t* contract = NULL;
    IREE_RETURN_IF_ERROR(
        loom_link_func_contract_projection_load(projection, target, &contract));
    for (iree_host_size_t j = 0; j < requirement->candidates.count; ++j) {
      const loom_link_module_index_symbol_t* candidate =
          loom_link_module_index_symbol_at(
              builder->index,
              candidates[requirement->candidates.first + j].symbol_ordinal);
      if (loom_link_dependency_candidate_has_func_contract(target, candidate)) {
        IREE_RETURN_IF_ERROR(loom_link_func_contract_projection_load(
            projection, candidate, &contract));
      }
    }
  }
  return iree_ok_status();
}

static void loom_link_dependency_mark_direct_candidate_used(
    loom_link_dependency_builder_t* builder,
    const loom_link_dependency_candidate_t* candidate) {
  if (candidate->origin == LOOM_LINK_DEPENDENCY_CANDIDATE_DIRECT_LIBRARY) {
    loom_link_dependency_bit_test_and_set(builder->used_provider_bits,
                                          candidate->provider_ordinal);
  }
}

static iree_status_t loom_link_dependency_check_exact_candidates(
    loom_link_dependency_builder_t* builder,
    loom_link_func_contract_projection_t* projection, loom_ir_remap_t* remap,
    loom_link_dependency_requirement_t* requirement) {
  loom_link_dependency_candidate_t* candidates =
      (loom_link_dependency_candidate_t*)builder->analysis.candidates.values;
  const loom_link_module_index_symbol_t* target =
      loom_link_module_index_symbol_at(builder->index,
                                       requirement->target.symbol_ordinal);
  const loom_link_func_contract_t* target_contract = NULL;
  IREE_RETURN_IF_ERROR(loom_link_func_contract_projection_load(
      projection, target, &target_contract));

  iree_host_size_t input_count = 0;
  iree_host_size_t input_compatible_count = 0;
  iree_host_size_t direct_count = 0;
  iree_host_size_t direct_compatible_count = 0;
  iree_host_size_t transitive_count = 0;
  iree_host_size_t transitive_compatible_count = 0;
  iree_host_size_t external_compatible_definition_count = 0;
  iree_host_size_t inaccessible_compatible_count = 0;
  for (iree_host_size_t i = 0; i < requirement->candidates.count; ++i) {
    loom_link_dependency_candidate_t* candidate =
        &candidates[requirement->candidates.first + i];
    const loom_link_module_index_symbol_t* candidate_symbol =
        loom_link_module_index_symbol_at(builder->index,
                                         candidate->symbol_ordinal);
    if (loom_link_dependency_candidate_has_func_contract(target,
                                                         candidate_symbol)) {
      const loom_link_func_contract_t* candidate_contract = NULL;
      IREE_RETURN_IF_ERROR(loom_link_func_contract_projection_load(
          projection, candidate_symbol, &candidate_contract));
      IREE_RETURN_IF_ERROR(loom_link_func_contract_check(
          target_contract, candidate_contract, remap, &candidate->mismatch));
      candidate->compatible =
          !loom_link_func_contract_mismatch_present(&candidate->mismatch);
    } else {
      candidate->mismatch = (loom_link_func_contract_mismatch_t){
          .kind = LOOM_LINK_FUNC_CONTRACT_MISMATCH_FIELD,
          .field_name = IREE_SV("symbol interfaces"),
      };
    }

    const bool accessible = loom_link_dependency_exact_candidate_accessible(
        candidate_symbol, candidate->origin);
    if (!accessible) {
      inaccessible_compatible_count += candidate->compatible ? 1 : 0;
      continue;
    }
    loom_link_dependency_mark_direct_candidate_used(builder, candidate);
    if (candidate->origin != LOOM_LINK_DEPENDENCY_CANDIDATE_INPUT &&
        candidate->compatible &&
        loom_link_dependency_exact_candidate_is_definition(candidate_symbol)) {
      ++external_compatible_definition_count;
    }

    switch (candidate->origin) {
      case LOOM_LINK_DEPENDENCY_CANDIDATE_INPUT:
        ++input_count;
        input_compatible_count += candidate->compatible ? 1 : 0;
        break;
      case LOOM_LINK_DEPENDENCY_CANDIDATE_DIRECT_LIBRARY:
        ++direct_count;
        direct_compatible_count += candidate->compatible ? 1 : 0;
        break;
      case LOOM_LINK_DEPENDENCY_CANDIDATE_TRANSITIVE_LIBRARY:
        ++transitive_count;
        transitive_compatible_count += candidate->compatible ? 1 : 0;
        break;
    }
  }

  if (input_count != 0) {
    requirement->ownership = input_compatible_count != 0
                                 ? LOOM_LINK_DEPENDENCY_OWNERSHIP_LOCAL
                                 : LOOM_LINK_DEPENDENCY_OWNERSHIP_INCOMPATIBLE;
  } else if (direct_count != 0) {
    requirement->ownership = direct_compatible_count != 0
                                 ? LOOM_LINK_DEPENDENCY_OWNERSHIP_DIRECT
                                 : LOOM_LINK_DEPENDENCY_OWNERSHIP_INCOMPATIBLE;
  } else if (transitive_count != 0) {
    requirement->ownership = transitive_compatible_count != 0
                                 ? LOOM_LINK_DEPENDENCY_OWNERSHIP_MISSING_DIRECT
                                 : LOOM_LINK_DEPENDENCY_OWNERSHIP_INCOMPATIBLE;
  } else if (inaccessible_compatible_count != 0) {
    requirement->ownership = LOOM_LINK_DEPENDENCY_OWNERSHIP_INACCESSIBLE;
  } else {
    requirement->ownership = LOOM_LINK_DEPENDENCY_OWNERSHIP_UNSATISFIED;
  }

  if (input_count != 0) {
    requirement->resolution =
        input_compatible_count == 1
            ? LOOM_LINK_DEPENDENCY_RESOLUTION_LOCAL
            : (input_compatible_count > 1
                   ? LOOM_LINK_DEPENDENCY_RESOLUTION_AMBIGUOUS
                   : LOOM_LINK_DEPENDENCY_RESOLUTION_INCOMPATIBLE);
  } else if (external_compatible_definition_count == 1) {
    requirement->resolution = LOOM_LINK_DEPENDENCY_RESOLUTION_UNIQUE;
  } else if (external_compatible_definition_count > 1) {
    requirement->resolution = LOOM_LINK_DEPENDENCY_RESOLUTION_AMBIGUOUS;
  } else if (direct_compatible_count != 0 || transitive_compatible_count != 0) {
    requirement->resolution = LOOM_LINK_DEPENDENCY_RESOLUTION_DEFERRED;
  } else if (direct_count != 0 || transitive_count != 0) {
    requirement->resolution = LOOM_LINK_DEPENDENCY_RESOLUTION_INCOMPATIBLE;
  } else {
    requirement->resolution = LOOM_LINK_DEPENDENCY_RESOLUTION_UNRESOLVED;
  }
  return iree_ok_status();
}

static void loom_link_dependency_classify_template_candidates(
    loom_link_dependency_builder_t* builder,
    loom_link_dependency_requirement_t* requirement) {
  const loom_link_dependency_candidate_t* candidates =
      builder->analysis.candidates.values;
  iree_host_size_t input_count = 0;
  iree_host_size_t direct_count = 0;
  iree_host_size_t transitive_count = 0;
  for (iree_host_size_t i = 0; i < requirement->candidates.count; ++i) {
    const loom_link_dependency_candidate_t* candidate =
        &candidates[requirement->candidates.first + i];
    loom_link_dependency_mark_direct_candidate_used(builder, candidate);
    switch (candidate->origin) {
      case LOOM_LINK_DEPENDENCY_CANDIDATE_INPUT:
        ++input_count;
        break;
      case LOOM_LINK_DEPENDENCY_CANDIDATE_DIRECT_LIBRARY:
        ++direct_count;
        break;
      case LOOM_LINK_DEPENDENCY_CANDIDATE_TRANSITIVE_LIBRARY:
        ++transitive_count;
        break;
    }
  }
  if (direct_count != 0) {
    requirement->ownership = LOOM_LINK_DEPENDENCY_OWNERSHIP_DIRECT;
  } else if (input_count != 0) {
    requirement->ownership = LOOM_LINK_DEPENDENCY_OWNERSHIP_LOCAL;
  } else if (transitive_count != 0) {
    requirement->ownership = LOOM_LINK_DEPENDENCY_OWNERSHIP_MISSING_DIRECT;
  } else {
    requirement->ownership = LOOM_LINK_DEPENDENCY_OWNERSHIP_OPEN;
  }
}

static iree_status_t loom_link_dependency_classify_requirements(
    loom_link_dependency_builder_t* builder,
    iree_arena_block_pool_t* block_pool, iree_allocator_t allocator) {
  loom_link_dependency_requirement_t* requirements =
      (loom_link_dependency_requirement_t*)
          builder->analysis.requirements.values;
  for (iree_host_size_t i = 0; i < builder->analysis.requirements.count; ++i) {
    if (requirements[i].kind == LOOM_LINK_DEPENDENCY_REQUIREMENT_EXACT_SYMBOL) {
      loom_link_dependency_fill_exact_candidates(builder, &requirements[i]);
    } else {
      loom_link_dependency_fill_template_candidates(builder, &requirements[i]);
    }
  }

  loom_link_func_contract_projection_t* projection = NULL;
  IREE_RETURN_IF_ERROR(loom_link_func_contract_projection_allocate(
      builder->index, block_pool, allocator, &projection));
  iree_status_t status =
      loom_link_dependency_project_exact_contracts(builder, projection);
  loom_ir_remap_t comparison_remap = {0};
  if (iree_status_is_ok(status)) {
    const loom_ir_remap_options_t remap_options = {
        .value_map_kind = LOOM_IR_REMAP_VALUE_MAP_SOURCE_INDEXED,
    };
    loom_module_t* identity_module =
        loom_link_func_contract_projection_module(projection);
    status = loom_ir_remap_initialize(identity_module, identity_module,
                                      builder->scratch_arena, &remap_options,
                                      &comparison_remap);
  }
  for (iree_host_size_t i = 0;
       i < builder->analysis.requirements.count && iree_status_is_ok(status);
       ++i) {
    if (requirements[i].kind == LOOM_LINK_DEPENDENCY_REQUIREMENT_EXACT_SYMBOL) {
      status = loom_link_dependency_check_exact_candidates(
          builder, projection, &comparison_remap, &requirements[i]);
    } else {
      loom_link_dependency_classify_template_candidates(builder,
                                                        &requirements[i]);
    }
  }
  loom_link_func_contract_projection_free(projection);
  return status;
}

static iree_status_t loom_link_dependency_finish_provider_sets(
    loom_link_dependency_builder_t* builder) {
  const iree_host_size_t direct_count =
      builder->analysis.direct_providers.count;
  iree_host_size_t used_count = 0;
  for (iree_host_size_t i = 0; i < direct_count; ++i) {
    if (loom_link_dependency_bit_test(
            builder->used_provider_bits,
            builder->analysis.direct_providers.values[i])) {
      ++used_count;
    }
  }
  const iree_host_size_t unused_count = direct_count - used_count;
  iree_host_size_t* used = NULL;
  iree_host_size_t* unused = NULL;
  if (used_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->output_arena, used_count, sizeof(*used), (void**)&used));
  }
  if (unused_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->output_arena, unused_count, sizeof(*unused), (void**)&unused));
  }
  iree_host_size_t used_position = 0;
  iree_host_size_t unused_position = 0;
  for (iree_host_size_t i = 0; i < direct_count; ++i) {
    const iree_host_size_t provider_ordinal =
        builder->analysis.direct_providers.values[i];
    if (loom_link_dependency_bit_test(builder->used_provider_bits,
                                      provider_ordinal)) {
      used[used_position++] = provider_ordinal;
    } else {
      unused[unused_position++] = provider_ordinal;
    }
  }
  IREE_ASSERT_EQ(used_position, used_count);
  IREE_ASSERT_EQ(unused_position, unused_count);
  builder->analysis.used_direct_providers.values = used;
  builder->analysis.used_direct_providers.count = used_count;
  builder->analysis.unused_direct_providers.values = unused;
  builder->analysis.unused_direct_providers.count = unused_count;
  return iree_ok_status();
}

iree_status_t loom_link_dependency_analyze(
    const loom_link_module_index_t* index,
    const loom_link_dependency_analysis_options_t* options,
    iree_arena_block_pool_t* block_pool, iree_arena_allocator_t* arena,
    iree_allocator_t allocator, loom_link_dependency_analysis_t* out_analysis) {
  IREE_ASSERT_ARGUMENT(index);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_analysis);
  *out_analysis = (loom_link_dependency_analysis_t){0};

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(block_pool, &scratch_arena);
  loom_link_dependency_builder_t builder = {
      .index = index,
      .output_arena = arena,
      .scratch_arena = &scratch_arena,
      .analysis = {.index = index},
  };
  iree_status_t status =
      loom_link_dependency_prepare_direct_providers(&builder, options);
  iree_host_size_t exact_requirement_count = 0;
  if (iree_status_is_ok(status)) {
    const iree_host_size_t symbol_count =
        loom_link_module_index_symbol_count(index);
    status = loom_link_dependency_allocate_bits(
        &scratch_arena, symbol_count, &builder.exact_requirement_bits);
  }
  if (iree_status_is_ok(status)) {
    loom_link_dependency_visit_input_exact_occurrences(
        &builder, /*populate=*/false, &exact_requirement_count);
    loom_link_dependency_mark_input_exports(&builder, &exact_requirement_count);
  }
  iree_host_size_t template_requirement_count = 0;
  if (iree_status_is_ok(status)) {
    status = loom_link_dependency_collect_template_demands(
        &builder, &template_requirement_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_link_dependency_initialize_requirements(
        &builder, exact_requirement_count, template_requirement_count);
  }
  if (iree_status_is_ok(status)) {
    loom_link_dependency_visit_input_exact_occurrences(
        &builder, /*populate=*/true, &exact_requirement_count);
    status = loom_link_dependency_allocate_candidates(&builder);
  }
  if (iree_status_is_ok(status)) {
    status = loom_link_dependency_classify_requirements(&builder, block_pool,
                                                        allocator);
  }
  if (iree_status_is_ok(status)) {
    status = loom_link_dependency_finish_provider_sets(&builder);
  }
  if (iree_status_is_ok(status)) {
    *out_analysis = builder.analysis;
  }
  iree_arena_deinitialize(&scratch_arena);
  return status;
}
