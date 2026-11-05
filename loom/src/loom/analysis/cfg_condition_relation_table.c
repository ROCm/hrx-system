// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/cfg_condition_relation_table.h"

#include <string.h>

#define LOOM_CFG_CONDITION_RELATION_PAGE_SHIFT 6
#define LOOM_CFG_CONDITION_RELATION_PAGE_WIDTH \
  (UINT32_C(1) << LOOM_CFG_CONDITION_RELATION_PAGE_SHIFT)

typedef struct loom_cfg_condition_relation_publication_view_t {
  // First page occurrence owned by the view.
  uint32_t first_page;

  // Number of page occurrences owned by the view.
  uint32_t page_count;
} loom_cfg_condition_relation_publication_view_t;

typedef struct loom_cfg_condition_relation_publication_page_t {
  // Source view containing the page rows.
  uint32_t view;

  // First source row in the page interval.
  uint32_t row_begin;

  // Source row following the page interval.
  uint32_t row_end;

  // First identical page occurrence in publication order.
  uint32_t canonical_page;

  // Immutable page ordinal assigned during publication.
  uint32_t retained_page;
} loom_cfg_condition_relation_publication_page_t;

static_assert(sizeof(loom_cfg_condition_relation_publication_page_t) == 20,
              "condition page planning records must remain 20 bytes");

static loom_condition_relation_matrix_t*
loom_cfg_condition_relation_publication_page_matrix(
    const loom_cfg_condition_relation_table_builder_t* builder,
    const loom_cfg_condition_relation_publication_page_t* page) {
  return &builder->views[page->view].integer_relations;
}

static uint32_t loom_cfg_condition_relation_publication_page_first_left(
    const loom_cfg_condition_relation_table_builder_t* builder,
    const loom_cfg_condition_relation_publication_page_t* page) {
  const loom_condition_relation_matrix_t* matrix =
      loom_cfg_condition_relation_publication_page_matrix(builder, page);
  const uint32_t left = matrix->rows[page->row_begin].left;
  return (left >> LOOM_CFG_CONDITION_RELATION_PAGE_SHIFT)
         << LOOM_CFG_CONDITION_RELATION_PAGE_SHIFT;
}

static uint32_t loom_cfg_condition_relation_publication_page_live_row_count(
    const loom_cfg_condition_relation_table_builder_t* builder,
    const loom_cfg_condition_relation_publication_page_t* page) {
  const loom_condition_relation_matrix_t* matrix =
      loom_cfg_condition_relation_publication_page_matrix(builder, page);
  uint32_t live_row_count = 0;
  for (uint32_t row = page->row_begin; row < page->row_end; ++row) {
    live_row_count +=
        loom_condition_relation_matrix_row_is_empty(&matrix->rows[row]) ? 0 : 1;
  }
  return live_row_count;
}

static uint64_t loom_cfg_condition_relation_hash_combine(uint64_t hash,
                                                         uint32_t value) {
  hash ^= value;
  hash *= UINT64_C(1099511628211);
  return hash;
}

static uint64_t loom_cfg_condition_relation_publication_page_hash(
    const loom_cfg_condition_relation_table_builder_t* builder,
    const loom_cfg_condition_relation_publication_page_t* page) {
  const loom_condition_relation_matrix_t* matrix =
      loom_cfg_condition_relation_publication_page_matrix(builder, page);
  uint64_t hash = UINT64_C(14695981039346656037);
  hash = loom_cfg_condition_relation_hash_combine(
      hash,
      loom_cfg_condition_relation_publication_page_first_left(builder, page));
  for (uint32_t i = page->row_begin; i < page->row_end; ++i) {
    const loom_condition_relation_matrix_row_t* row = &matrix->rows[i];
    if (loom_condition_relation_matrix_row_is_empty(row)) {
      continue;
    }
    hash = loom_cfg_condition_relation_hash_combine(hash, row->left);
    for (loom_integer_order_outcome_t outcome = 0;
         outcome < LOOM_INTEGER_ORDER_OUTCOME_COUNT; ++outcome) {
      hash = loom_cfg_condition_relation_hash_combine(hash,
                                                      row->excluded[outcome]);
    }
  }
  return hash;
}

static uint32_t loom_cfg_condition_relation_next_live_row(
    const loom_cfg_condition_relation_table_builder_t* builder,
    const loom_cfg_condition_relation_publication_page_t* page,
    uint32_t position) {
  const loom_condition_relation_matrix_t* matrix =
      loom_cfg_condition_relation_publication_page_matrix(builder, page);
  while (position < page->row_end &&
         loom_condition_relation_matrix_row_is_empty(&matrix->rows[position])) {
    ++position;
  }
  return position;
}

static bool loom_cfg_condition_relation_publication_pages_equal(
    const loom_cfg_condition_relation_table_builder_t* builder,
    const loom_cfg_condition_relation_publication_page_t* left,
    const loom_cfg_condition_relation_publication_page_t* right) {
  if (loom_cfg_condition_relation_publication_page_first_left(builder, left) !=
      loom_cfg_condition_relation_publication_page_first_left(builder, right)) {
    return false;
  }
  const loom_condition_relation_matrix_t* left_matrix =
      loom_cfg_condition_relation_publication_page_matrix(builder, left);
  const loom_condition_relation_matrix_t* right_matrix =
      loom_cfg_condition_relation_publication_page_matrix(builder, right);
  uint32_t left_position = left->row_begin;
  uint32_t right_position = right->row_begin;
  while (true) {
    left_position =
        loom_cfg_condition_relation_next_live_row(builder, left, left_position);
    right_position = loom_cfg_condition_relation_next_live_row(builder, right,
                                                               right_position);
    if (left_position == left->row_end || right_position == right->row_end) {
      return left_position == left->row_end && right_position == right->row_end;
    }
    const loom_condition_relation_matrix_row_t* left_row =
        &left_matrix->rows[left_position++];
    const loom_condition_relation_matrix_row_t* right_row =
        &right_matrix->rows[right_position++];
    if (memcmp(left_row, right_row, sizeof(*left_row)) != 0) {
      return false;
    }
  }
}

static uint32_t loom_cfg_condition_relation_matrix_page_count(
    const loom_condition_relation_matrix_t* matrix) {
  uint32_t page_count = 0;
  uint32_t previous_page = UINT32_MAX;
  for (uint32_t i = 0; i < matrix->row_count; ++i) {
    const loom_condition_relation_matrix_row_t* row = &matrix->rows[i];
    if (loom_condition_relation_matrix_row_is_empty(row)) {
      continue;
    }
    const uint32_t page = row->left >> LOOM_CFG_CONDITION_RELATION_PAGE_SHIFT;
    if (page != previous_page) {
      ++page_count;
      previous_page = page;
    }
  }
  return page_count;
}

static void loom_cfg_condition_relation_plan_view_pages(
    loom_cfg_condition_relation_table_builder_t* builder, uint32_t view,
    uint32_t first_page, loom_cfg_condition_relation_publication_page_t* pages,
    loom_cfg_condition_relation_publication_view_t* out_view) {
  loom_condition_relation_matrix_t* matrix =
      &builder->views[view].integer_relations;
  *out_view = (loom_cfg_condition_relation_publication_view_t){
      .first_page = first_page,
  };
  uint32_t previous_page = UINT32_MAX;
  loom_cfg_condition_relation_publication_page_t* current_page = NULL;
  for (uint32_t i = 0; i < matrix->row_count; ++i) {
    const loom_condition_relation_matrix_row_t* row = &matrix->rows[i];
    if (loom_condition_relation_matrix_row_is_empty(row)) {
      continue;
    }
    const uint32_t page = row->left >> LOOM_CFG_CONDITION_RELATION_PAGE_SHIFT;
    if (page != previous_page) {
      current_page = &pages[first_page + out_view->page_count++];
      *current_page = (loom_cfg_condition_relation_publication_page_t){
          .view = view,
          .row_begin = i,
          .canonical_page = UINT32_MAX,
          .retained_page = UINT32_MAX,
      };
      previous_page = page;
    }
    current_page->row_end = i + 1;
  }
}

static iree_status_t loom_cfg_condition_relation_plan_pages(
    loom_cfg_condition_relation_table_builder_t* builder,
    iree_arena_allocator_t* scratch_arena,
    loom_cfg_condition_relation_publication_view_t** out_views,
    loom_cfg_condition_relation_publication_page_t** out_pages,
    uint32_t* out_page_count, uint32_t* out_unique_page_count,
    uint64_t* out_unique_live_row_count) {
  *out_views = NULL;
  *out_pages = NULL;
  *out_page_count = 0;
  *out_unique_page_count = 0;
  *out_unique_live_row_count = 0;

  loom_cfg_condition_relation_publication_view_t* views = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      scratch_arena, builder->view_count, sizeof(*views), (void**)&views));
  uint64_t total_page_count = 0;
  for (uint32_t view = 0; view < builder->view_count; ++view) {
    total_page_count += loom_cfg_condition_relation_matrix_page_count(
        &builder->views[view].integer_relations);
  }
  if (total_page_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "condition relation pages exceed uint32_t");
  }

  loom_cfg_condition_relation_publication_page_t* pages = NULL;
  if (total_page_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, (iree_host_size_t)total_page_count, sizeof(*pages),
        (void**)&pages));
  }
  uint32_t page_count = 0;
  for (uint32_t view = 0; view < builder->view_count; ++view) {
    loom_cfg_condition_relation_plan_view_pages(builder, view, page_count,
                                                pages, &views[view]);
    page_count += views[view].page_count;
  }
  IREE_ASSERT_EQ(page_count, total_page_count);

  uint32_t hash_capacity = 1;
  while (hash_capacity < page_count) {
    if (hash_capacity > UINT32_MAX / 2) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "condition relation page index exceeds uint32_t");
    }
    hash_capacity *= 2;
  }
  if (page_count != 0) {
    if (hash_capacity > UINT32_MAX / 2) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "condition relation page index exceeds uint32_t");
    }
    hash_capacity *= 2;
  }
  uint32_t* hash_slots = NULL;
  if (page_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(scratch_arena, hash_capacity,
                                                   sizeof(*hash_slots),
                                                   (void**)&hash_slots));
    memset(hash_slots, 0xFF, hash_capacity * sizeof(*hash_slots));
  }

  uint32_t unique_page_count = 0;
  uint64_t unique_live_row_count = 0;
  for (uint32_t i = 0; i < page_count; ++i) {
    loom_cfg_condition_relation_publication_page_t* page = &pages[i];
    const uint64_t hash =
        loom_cfg_condition_relation_publication_page_hash(builder, page);
    uint32_t slot = (uint32_t)hash & (hash_capacity - 1);
    while (hash_slots[slot] != UINT32_MAX &&
           !loom_cfg_condition_relation_publication_pages_equal(
               builder, page, &pages[hash_slots[slot]])) {
      slot = (slot + 1) & (hash_capacity - 1);
    }
    if (hash_slots[slot] == UINT32_MAX) {
      hash_slots[slot] = i;
      page->canonical_page = i;
      ++unique_page_count;
      unique_live_row_count +=
          loom_cfg_condition_relation_publication_page_live_row_count(builder,
                                                                      page);
    } else {
      page->canonical_page = hash_slots[slot];
    }
  }

  *out_views = views;
  *out_pages = pages;
  *out_page_count = page_count;
  *out_unique_page_count = unique_page_count;
  *out_unique_live_row_count = unique_live_row_count;
  return iree_ok_status();
}

static iree_status_t loom_cfg_condition_relation_table_publish_domain(
    const loom_cfg_condition_operand_domain_t* source,
    const loom_cfg_condition_relation_anchor_builder_t* derived_anchors,
    iree_arena_allocator_t* arena,
    loom_cfg_condition_operand_domain_t** out_domain) {
  *out_domain = NULL;
  const uint32_t derived_anchor_count =
      loom_cfg_condition_relation_anchor_builder_count(derived_anchors);
  iree_host_size_t derived_anchor_size = 0;
  IREE_RETURN_IF_ERROR(loom_cfg_condition_relation_anchor_builder_storage_size(
      derived_anchors, &derived_anchor_size));
  iree_host_size_t allocation_size = 0;
  if (!iree_host_size_checked_add(sizeof(*source), derived_anchor_size,
                                  &allocation_size)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "condition relation anchors exceed host size");
  }
  loom_cfg_condition_operand_domain_t* domain = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, allocation_size, (void**)&domain));
  *domain = *source;
  if (derived_anchor_count != 0) {
    loom_cfg_condition_relation_anchor_builder_publish(derived_anchors,
                                                       domain + 1);
  }
  if (domain->value_count != 0) {
    loom_value_id_t* values = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, domain->value_count, sizeof(*values), (void**)&values));
    memcpy(values, domain->values, domain->value_count * sizeof(*values));
    domain->values = values;
  }
  if (domain->constant_count != 0) {
    int64_t* constants = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, domain->constant_count, sizeof(*constants), (void**)&constants));
    memcpy(constants, domain->constants,
           domain->constant_count * sizeof(*constants));
    domain->constants = constants;
  }
  *out_domain = domain;
  return iree_ok_status();
}

#if IREE_HAVE_ATTRIBUTE(minsize)
__attribute__((minsize))
#endif
iree_status_t loom_cfg_condition_relation_table_publish(
    loom_cfg_condition_relation_table_builder_t* builder,
    iree_arena_allocator_t* scratch_arena, iree_arena_allocator_t* arena,
    loom_cfg_condition_relation_table_t* out_table) {
  loom_cfg_condition_relation_publication_view_t* publication_views = NULL;
  loom_cfg_condition_relation_publication_page_t* publication_pages = NULL;
  uint32_t page_count = 0;
  uint32_t unique_page_count = 0;
  uint64_t unique_live_row_count = 0;
  IREE_RETURN_IF_ERROR(loom_cfg_condition_relation_plan_pages(
      builder, scratch_arena, &publication_views, &publication_pages,
      &page_count, &unique_page_count, &unique_live_row_count));

  const uint64_t root_count =
      (uint64_t)builder->view_count * 2 +
      unique_live_row_count * LOOM_INTEGER_ORDER_OUTCOME_COUNT +
      loom_cfg_condition_relation_anchor_builder_count(
          builder->derived_anchors);
  if (root_count > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "condition relation roots exceed host size");
  }
  loom_condition_relation_set_id_t* roots = NULL;
  if (root_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(scratch_arena, (iree_host_size_t)root_count,
                                  sizeof(*roots), (void**)&roots));
  }
  iree_host_size_t root_position = 0;
  for (uint32_t view = 0; view < builder->view_count; ++view) {
    for (uint8_t value = 0; value < 2; ++value) {
      roots[root_position++] = builder->views[view].boolean_values[value];
    }
  }
  for (uint32_t page = 0; page < page_count; ++page) {
    loom_cfg_condition_relation_publication_page_t* publication_page =
        &publication_pages[page];
    if (publication_page->canonical_page != page) {
      continue;
    }
    loom_condition_relation_matrix_t* matrix =
        loom_cfg_condition_relation_publication_page_matrix(builder,
                                                            publication_page);
    for (uint32_t row = publication_page->row_begin;
         row < publication_page->row_end; ++row) {
      const loom_condition_relation_matrix_row_t* source_row =
          &matrix->rows[row];
      if (loom_condition_relation_matrix_row_is_empty(source_row)) {
        continue;
      }
      for (loom_integer_order_outcome_t outcome = 0;
           outcome < LOOM_INTEGER_ORDER_OUTCOME_COUNT; ++outcome) {
        roots[root_position++] = source_row->excluded[outcome];
      }
    }
  }
  if (builder->derived_anchors != NULL) {
    const uint32_t entry_count =
        loom_cfg_condition_relation_anchor_builder_count(
            builder->derived_anchors);
    for (uint32_t i = 0; i < entry_count; ++i) {
      roots[root_position++] = loom_cfg_condition_relation_anchor_builder_root(
          builder->derived_anchors, i);
    }
  }
  IREE_ASSERT_EQ(root_position, root_count);
  loom_condition_relation_set_index_t set_index = {0};
  IREE_RETURN_IF_ERROR(loom_condition_relation_set_builder_publish(
      builder->set_builder, roots, (iree_host_size_t)root_count, arena,
      &set_index));

  root_position = 0;
  for (uint32_t view = 0; view < builder->view_count; ++view) {
    for (uint8_t value = 0; value < 2; ++value) {
      builder->views[view].boolean_values[value] = roots[root_position++];
    }
  }
  for (uint32_t page = 0; page < page_count; ++page) {
    loom_cfg_condition_relation_publication_page_t* publication_page =
        &publication_pages[page];
    if (publication_page->canonical_page != page) {
      continue;
    }
    loom_condition_relation_matrix_t* matrix =
        loom_cfg_condition_relation_publication_page_matrix(builder,
                                                            publication_page);
    for (uint32_t row = publication_page->row_begin;
         row < publication_page->row_end; ++row) {
      loom_condition_relation_matrix_row_t* target_row = &matrix->rows[row];
      if (loom_condition_relation_matrix_row_is_empty(target_row)) {
        continue;
      }
      for (loom_integer_order_outcome_t outcome = 0;
           outcome < LOOM_INTEGER_ORDER_OUTCOME_COUNT; ++outcome) {
        target_row->excluded[outcome] = roots[root_position++];
      }
    }
  }
  if (builder->derived_anchors != NULL) {
    const uint32_t entry_count =
        loom_cfg_condition_relation_anchor_builder_count(
            builder->derived_anchors);
    for (uint32_t i = 0; i < entry_count; ++i) {
      loom_cfg_condition_relation_anchor_builder_set_root(
          builder->derived_anchors, i, roots[root_position++]);
    }
  }
  IREE_ASSERT_EQ(root_position, root_count);

  loom_cfg_condition_relation_view_t* views = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, builder->view_count, sizeof(*views), (void**)&views));
  loom_condition_relation_matrix_page_t* retained_pages = NULL;
  if (unique_page_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, unique_page_count,
                                                   sizeof(*retained_pages),
                                                   (void**)&retained_pages));
  }
  const loom_condition_relation_matrix_page_t** retained_page_references = NULL;
  if (page_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, page_count, sizeof(*retained_page_references),
        (void**)&retained_page_references));
  }
  uint32_t next_retained_page = 0;
  for (uint32_t page = 0; page < page_count; ++page) {
    loom_cfg_condition_relation_publication_page_t* publication_page =
        &publication_pages[page];
    if (publication_page->canonical_page == page) {
      publication_page->retained_page = next_retained_page++;
      loom_condition_relation_matrix_page_t* retained_page =
          &retained_pages[publication_page->retained_page];
      const uint32_t first_left =
          loom_cfg_condition_relation_publication_page_first_left(
              builder, publication_page);
      *retained_page = (loom_condition_relation_matrix_page_t){
          .first_left = first_left,
          .last_left = first_left + LOOM_CFG_CONDITION_RELATION_PAGE_WIDTH - 1,
      };
      loom_condition_relation_matrix_t* matrix =
          loom_cfg_condition_relation_publication_page_matrix(builder,
                                                              publication_page);
      loom_condition_relation_matrix_t page_matrix = {
          .rows = &matrix->rows[publication_page->row_begin],
          .row_count = publication_page->row_end - publication_page->row_begin,
      };
      IREE_RETURN_IF_ERROR(loom_condition_relation_matrix_view_publish(
          &page_matrix, arena, &retained_page->contents));
    } else {
      publication_page->retained_page =
          publication_pages[publication_page->canonical_page].retained_page;
      IREE_ASSERT_NE(publication_page->retained_page, UINT32_MAX);
    }
    retained_page_references[page] =
        &retained_pages[publication_page->retained_page];
  }
  IREE_ASSERT_EQ(next_retained_page, unique_page_count);

  for (uint32_t view = 0; view < builder->view_count; ++view) {
    views[view] = (loom_cfg_condition_relation_view_t){
        .boolean_values = {builder->views[view].boolean_values[0],
                           builder->views[view].boolean_values[1]},
    };
    const loom_cfg_condition_relation_publication_view_t* publication_view =
        &publication_views[view];
    if (publication_view->page_count != 0) {
      views[view].integer_relations = (loom_condition_relation_matrix_view_t){
          .entries.pages =
              retained_page_references + publication_view->first_page,
          .entry_count = publication_view->page_count,
          .encoding = LOOM_CONDITION_RELATION_MATRIX_VIEW_PAGES,
      };
    }
  }

  uint32_t* retained_edge_view_indices = NULL;
  if (builder->edge_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, builder->edge_count, sizeof(*retained_edge_view_indices),
        (void**)&retained_edge_view_indices));
    memcpy(retained_edge_view_indices, builder->edge_view_indices,
           (iree_host_size_t)builder->edge_count *
               sizeof(*retained_edge_view_indices));
  }
  loom_cfg_condition_operand_domain_t* operand_domain = NULL;
  IREE_RETURN_IF_ERROR(loom_cfg_condition_relation_table_publish_domain(
      builder->operand_domain, builder->derived_anchors, arena,
      &operand_domain));
  *out_table = (loom_cfg_condition_relation_table_t){
      .operand_domain = operand_domain,
      .set_index = set_index,
      .views = views,
      .edge_view_indices = retained_edge_view_indices,
      .view_count = builder->view_count,
      .block_count = builder->block_count,
      .edge_count = builder->edge_count,
      .derived_anchor_count = loom_cfg_condition_relation_anchor_builder_count(
          builder->derived_anchors),
  };
  return iree_ok_status();
}

const loom_cfg_condition_relation_view_t*
loom_cfg_condition_relation_table_block(
    const loom_cfg_condition_relation_table_t* table, uint16_t block_index) {
  return block_index < table->block_count ? &table->views[block_index] : NULL;
}

const loom_cfg_condition_relation_view_t*
loom_cfg_condition_relation_table_edge(
    const loom_cfg_condition_relation_table_t* table,
    loom_cfg_edge_index_t edge_index) {
  if (edge_index >= table->edge_count) {
    return NULL;
  }
  const uint32_t view = table->edge_view_indices[edge_index];
  return view < table->view_count ? &table->views[view] : NULL;
}

bool loom_cfg_condition_relation_view_query_boolean(
    const loom_cfg_condition_relation_table_t* table,
    const loom_cfg_condition_relation_view_t* view, loom_value_id_t value_id,
    bool* out_value) {
  const loom_cfg_condition_operand_t operand =
      loom_cfg_condition_operand_domain_value(table->operand_domain, value_id);
  if (operand == LOOM_CFG_CONDITION_OPERAND_INVALID) {
    return false;
  }
  const bool known_false = loom_condition_relation_set_index_contains(
      &table->set_index, view->boolean_values[0], operand);
  const bool known_true = loom_condition_relation_set_index_contains(
      &table->set_index, view->boolean_values[1], operand);
  if (known_false == known_true) {
    return false;
  }
  *out_value = known_true;
  return true;
}

loom_integer_order_outcome_bits_t
loom_cfg_condition_relation_view_query_excluded_outcomes(
    const loom_cfg_condition_relation_table_t* table,
    const loom_cfg_condition_relation_view_t* view,
    const loom_value_fact_table_t* fact_table,
    loom_condition_integer_operand_t left,
    loom_condition_integer_operand_t right) {
  loom_cfg_condition_operand_t left_variants[2];
  loom_cfg_condition_operand_t right_variants[2];
  loom_cfg_condition_operand_domain_variants(table->operand_domain, fact_table,
                                             left, left_variants);
  loom_cfg_condition_operand_domain_variants(table->operand_domain, fact_table,
                                             right, right_variants);
  loom_integer_order_outcome_bits_t exclusions = 0;
  for (uint8_t left_index = 0; left_index < 2; ++left_index) {
    if (left_variants[left_index] == LOOM_CFG_CONDITION_OPERAND_INVALID) {
      continue;
    }
    const loom_condition_relation_set_id_t* row =
        loom_condition_relation_matrix_view_find(&view->integer_relations,
                                                 left_variants[left_index]);
    if (row == NULL) {
      continue;
    }
    for (uint8_t right_index = 0; right_index < 2; ++right_index) {
      if (right_variants[right_index] == LOOM_CFG_CONDITION_OPERAND_INVALID) {
        continue;
      }
      for (loom_integer_order_outcome_t outcome = 0;
           outcome < LOOM_INTEGER_ORDER_OUTCOME_COUNT; ++outcome) {
        if (loom_condition_relation_set_index_contains(
                &table->set_index, row[outcome], right_variants[right_index])) {
          exclusions |= (loom_integer_order_outcome_bits_t)(1u << outcome);
        }
      }
    }
  }
  return exclusions;
}

static bool loom_cfg_condition_relation_from_exclusions(
    loom_integer_order_outcome_bits_t exclusions,
    loom_symbolic_integer_relation_t* out_relation) {
  const loom_integer_order_outcome_bits_t outcomes =
      LOOM_INTEGER_ORDER_OUTCOME_BIT_ALL &
      (loom_integer_order_outcome_bits_t)~exclusions;
  return loom_symbolic_integer_relation_from_outcomes(outcomes, out_relation);
}

typedef struct loom_cfg_condition_relation_visit_state_t {
  // Table owning the immutable set index and operand domain.
  const loom_cfg_condition_relation_table_t* table;

  // Three excluded-outcome roots for the anchored row.
  const loom_condition_relation_set_id_t* excluded;

  // Authored left operand represented by the retained row.
  loom_condition_integer_operand_t left;

  // Caller visitor.
  loom_cfg_condition_relation_visit_fn_t visit;

  // Caller visitor state.
  void* user_data;

  // Outcome root currently being enumerated.
  loom_integer_order_outcome_t outcome;
} loom_cfg_condition_relation_visit_state_t;

static bool loom_cfg_condition_relation_visit_member(void* user_data,
                                                     uint32_t right) {
  loom_cfg_condition_relation_visit_state_t* state =
      (loom_cfg_condition_relation_visit_state_t*)user_data;
  for (loom_integer_order_outcome_t outcome = 0; outcome < state->outcome;
       ++outcome) {
    if (loom_condition_relation_set_index_contains(
            &state->table->set_index, state->excluded[outcome], right)) {
      return true;
    }
  }

  loom_integer_order_outcome_bits_t exclusions = 0;
  for (loom_integer_order_outcome_t outcome = 0;
       outcome < LOOM_INTEGER_ORDER_OUTCOME_COUNT; ++outcome) {
    if (loom_condition_relation_set_index_contains(
            &state->table->set_index, state->excluded[outcome], right)) {
      exclusions |= (loom_integer_order_outcome_bits_t)(1u << outcome);
    }
  }
  loom_symbolic_integer_relation_t relation = 0;
  if (!loom_cfg_condition_relation_from_exclusions(exclusions, &relation)) {
    return true;
  }
  const loom_condition_integer_relation_t retained_relation = {
      .relation = relation,
      .left = state->left,
      .right = loom_cfg_condition_operand_domain_expand(
          state->table->operand_domain, right),
  };
  return state->visit(state->user_data, &retained_relation);
}

static bool loom_cfg_condition_relation_visit_operand_rows(
    const loom_cfg_condition_relation_table_t* table,
    const loom_cfg_condition_relation_view_t* view,
    loom_cfg_condition_operand_t operand, loom_condition_integer_operand_t left,
    loom_cfg_condition_relation_visit_fn_t visit, void* user_data) {
  const loom_condition_relation_set_id_t* excluded =
      loom_condition_relation_matrix_view_find(&view->integer_relations,
                                               operand);
  if (excluded == NULL) {
    return true;
  }
  loom_cfg_condition_relation_visit_state_t state = {
      .table = table,
      .excluded = excluded,
      .left = left,
      .visit = visit,
      .user_data = user_data,
  };
  for (loom_integer_order_outcome_t outcome = 0;
       outcome < LOOM_INTEGER_ORDER_OUTCOME_COUNT; ++outcome) {
    state.outcome = outcome;
    if (!loom_condition_relation_set_index_for_each_while(
            &table->set_index, excluded[outcome],
            loom_cfg_condition_relation_visit_member, &state)) {
      return false;
    }
  }
  return true;
}

static loom_condition_relation_set_id_t
loom_cfg_condition_relation_table_lookup_derived_anchors(
    const loom_cfg_condition_relation_table_t* table,
    loom_value_id_t anchor_value_id) {
  if (table->derived_anchor_count == 0) {
    return LOOM_CONDITION_RELATION_SET_EMPTY;
  }
  const void* retained_storage = table->operand_domain + 1;
  const loom_cfg_condition_relation_anchor_index_t* index =
      (const loom_cfg_condition_relation_anchor_index_t*)retained_storage;
  return loom_cfg_condition_relation_anchor_index_lookup(
      index, table->derived_anchor_count, table->operand_domain,
      anchor_value_id);
}

typedef struct loom_cfg_condition_derived_relation_visit_state_t {
  // Table owning the derived incidence and relation operand domain.
  const loom_cfg_condition_relation_table_t* table;

  // View whose relation rows are being queried.
  const loom_cfg_condition_relation_view_t* view;

  // Caller visitor.
  loom_cfg_condition_relation_visit_fn_t visit;

  // Caller visitor state.
  void* user_data;
} loom_cfg_condition_derived_relation_visit_state_t;

static bool loom_cfg_condition_relation_visit_derived_operand(
    void* user_data, uint32_t operand) {
  loom_cfg_condition_derived_relation_visit_state_t* state =
      (loom_cfg_condition_derived_relation_visit_state_t*)user_data;
  return loom_cfg_condition_relation_visit_operand_rows(
      state->table, state->view, operand,
      loom_cfg_condition_operand_domain_expand(state->table->operand_domain,
                                               operand),
      state->visit, state->user_data);
}

bool loom_cfg_condition_relation_view_for_each_while(
    const loom_cfg_condition_relation_table_t* table,
    const loom_cfg_condition_relation_view_t* view,
    const loom_value_fact_table_t* fact_table,
    loom_condition_integer_operand_t anchor,
    loom_cfg_condition_relation_visit_fn_t visit, void* user_data) {
  loom_cfg_condition_operand_t anchor_variants[2];
  loom_cfg_condition_operand_domain_variants(table->operand_domain, fact_table,
                                             anchor, anchor_variants);
  for (uint8_t anchor_index = 0; anchor_index < 2; ++anchor_index) {
    if (anchor_variants[anchor_index] == LOOM_CFG_CONDITION_OPERAND_INVALID) {
      continue;
    }
    if (!loom_cfg_condition_relation_visit_operand_rows(
            table, view, anchor_variants[anchor_index], anchor, visit,
            user_data)) {
      return false;
    }
  }
  if (anchor.kind != LOOM_CONDITION_INTEGER_OPERAND_VALUE) {
    return true;
  }
  const loom_condition_relation_set_id_t derived_operands =
      loom_cfg_condition_relation_table_lookup_derived_anchors(table,
                                                               anchor.value_id);
  loom_cfg_condition_derived_relation_visit_state_t state = {
      .table = table,
      .view = view,
      .visit = visit,
      .user_data = user_data,
  };
  if (!loom_condition_relation_set_index_for_each_while(
          &table->set_index, derived_operands,
          loom_cfg_condition_relation_visit_derived_operand, &state)) {
    return false;
  }
  return true;
}
