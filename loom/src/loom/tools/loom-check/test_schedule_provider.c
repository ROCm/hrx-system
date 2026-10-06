// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-check/test_schedule_provider.h"

#include <inttypes.h>

#include "loom/codegen/low/descriptors.h"
#include "loom/codegen/low/function_model.h"
#include "loom/codegen/low/schedule/dependencies.h"
#include "loom/codegen/low/schedule/run.h"
#include "loom/tools/loom-check/diagnostics.h"
#include "loom/tools/loom-check/low_emit.h"

enum loom_check_test_schedule_option_flag_bits_e {
  LOOM_CHECK_TEST_SCHEDULE_OPTION_FLAG_HAS_KIND = 1u << 0,
  LOOM_CHECK_TEST_SCHEDULE_OPTION_FLAG_HAS_STRATEGY = 1u << 1,
  LOOM_CHECK_TEST_SCHEDULE_OPTION_FLAG_HAS_TIMING = 1u << 2,
};
typedef uint32_t loom_check_test_schedule_option_flags_t;

typedef struct loom_check_test_schedule_dependency_query_t {
  // Dependency kind selected by the RUN line.
  loom_low_schedule_dependency_kind_t kind;
  // Inclusive minimum issue separation selected for that kind.
  int32_t minimum_issue_separation_cycles;
} loom_check_test_schedule_dependency_query_t;

typedef struct loom_check_test_schedule_options_t {
  // Module-local target-low function symbol selected by the RUN line.
  iree_string_view_t function_symbol_name;
  // Comma-separated source nodes whose predecessor sets are requested.
  iree_string_view_t consumer_nodes;
  // Comma-separated source nodes whose scheduled order is requested.
  iree_string_view_t order_nodes;
  // Comma-separated source nodes whose issue cycles are requested.
  iree_string_view_t issue_nodes;
  // Comma-separated source nodes whose selected descriptors are requested.
  iree_string_view_t descriptor_nodes;
  // Producer and consumer source nodes whose dependency timing is requested.
  uint32_t timing_producer_node;
  uint32_t timing_consumer_node;
  // Dependency relation whose predecessor sets are observed.
  loom_check_test_schedule_dependency_query_t dependency_query;
  // Candidate selection strategy used by Low frame construction.
  loom_low_schedule_strategy_t schedule_strategy;
  // Low allocation budget overrides parsed from the RUN line.
  loom_low_allocation_budget_t
      allocation_budgets[LOOM_CHECK_LOW_EMIT_MAX_ALLOCATION_BUDGETS];
  // Number of entries in |allocation_budgets|.
  iree_host_size_t allocation_budget_count;
  // Presence of required or uniquely specified RUN options.
  loom_check_test_schedule_option_flags_t flags;
} loom_check_test_schedule_options_t;

static bool loom_check_test_schedule_matches(
    const loom_check_emit_provider_t* provider,
    iree_string_view_t target_name) {
  (void)provider;
  return iree_string_view_equal(target_name, IREE_SV("low-schedule-query"));
}

static iree_status_t loom_check_test_schedule_parse_dependency_kind(
    iree_string_view_t value,
    loom_check_test_schedule_dependency_query_t* out_query) {
  *out_query = (loom_check_test_schedule_dependency_query_t){
      .minimum_issue_separation_cycles = INT32_MIN,
  };
  if (iree_string_view_equal(value, IREE_SV("ssa"))) {
    out_query->kind = LOOM_LOW_SCHEDULE_DEPENDENCY_SSA;
  } else if (iree_string_view_equal(value, IREE_SV("effect"))) {
    out_query->kind = LOOM_LOW_SCHEDULE_DEPENDENCY_EFFECT;
  } else if (iree_string_view_equal(value, IREE_SV("state"))) {
    out_query->kind = LOOM_LOW_SCHEDULE_DEPENDENCY_STATE;
  } else if (iree_string_view_equal(value, IREE_SV("storage"))) {
    out_query->kind = LOOM_LOW_SCHEDULE_DEPENDENCY_STORAGE;
  } else if (iree_string_view_equal(value, IREE_SV("storage-lifetime"))) {
    out_query->kind = LOOM_LOW_SCHEDULE_DEPENDENCY_STORAGE;
    out_query->minimum_issue_separation_cycles = 1;
  } else if (iree_string_view_equal(value, IREE_SV("order"))) {
    out_query->kind = LOOM_LOW_SCHEDULE_DEPENDENCY_ORDER;
  } else {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low-schedule-query option 'kind' expected 'ssa', 'effect', 'state', "
        "'storage', 'storage-lifetime', or 'order', got '%.*s'",
        (int)value.size, value.data);
  }
  return iree_ok_status();
}

static bool loom_check_test_schedule_consume_node(iree_string_view_t* list,
                                                  uint32_t* out_node) {
  if (iree_string_view_is_empty(*list)) {
    return false;
  }
  iree_string_view_t token = iree_string_view_empty();
  iree_string_view_split(*list, ',', &token, list);
  token = iree_string_view_trim(token);
  const bool parsed = iree_string_view_atoi_uint32(token, out_node);
  IREE_ASSERT(parsed);
  return true;
}

static iree_status_t loom_check_test_schedule_validate_node_list(
    iree_string_view_t option_name, iree_string_view_t value) {
  if (iree_string_view_is_empty(value)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low-schedule-query option '%.*s' requires a source node list",
        (int)option_name.size, option_name.data);
  }
  iree_string_view_t remaining = value;
  iree_host_size_t node_ordinal = 0;
  while (!iree_string_view_is_empty(remaining)) {
    iree_string_view_t token = iree_string_view_empty();
    iree_string_view_split(remaining, ',', &token, &remaining);
    token = iree_string_view_trim(token);
    uint32_t node = 0;
    if (iree_string_view_is_empty(token) ||
        !iree_string_view_atoi_uint32(token, &node)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low-schedule-query option '%.*s' expected comma-separated source "
          "node indices, got '%.*s'",
          (int)option_name.size, option_name.data, (int)value.size, value.data);
    }
    iree_string_view_t peers = value;
    for (iree_host_size_t i = 0; i < node_ordinal; ++i) {
      uint32_t peer = 0;
      loom_check_test_schedule_consume_node(&peers, &peer);
      if (peer == node) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "low-schedule-query option '%.*s' duplicates node %" PRIu32,
            (int)option_name.size, option_name.data, node);
      }
    }
    ++node_ordinal;
  }
  return iree_ok_status();
}

static iree_status_t loom_check_test_schedule_parse_timing_edge(
    iree_string_view_t value, loom_check_test_schedule_options_t* options) {
  iree_string_view_t producer = iree_string_view_empty();
  iree_string_view_t consumer = iree_string_view_empty();
  iree_string_view_split(value, ':', &producer, &consumer);
  producer = iree_string_view_trim(producer);
  consumer = iree_string_view_trim(consumer);
  if (iree_string_view_is_empty(producer) ||
      iree_string_view_is_empty(consumer) ||
      iree_string_view_find_char(consumer, ':', 0) != IREE_STRING_VIEW_NPOS ||
      !iree_string_view_atoi_uint32(producer, &options->timing_producer_node) ||
      !iree_string_view_atoi_uint32(consumer, &options->timing_consumer_node)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low-schedule-query option 'timing' expected <producer>:<consumer>, "
        "got '%.*s'",
        (int)value.size, value.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_check_test_schedule_parse_option(
    iree_string_view_t token, loom_check_test_schedule_options_t* options) {
  iree_string_view_t name = iree_string_view_empty();
  iree_string_view_t value = iree_string_view_empty();
  iree_string_view_split(token, '=', &name, &value);
  name = iree_string_view_trim(name);
  value = iree_string_view_trim(value);
  if (iree_string_view_equal(name, IREE_SV("kind"))) {
    if (iree_any_bit_set(options->flags,
                         LOOM_CHECK_TEST_SCHEDULE_OPTION_FLAG_HAS_KIND)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate low-schedule-query option 'kind'");
    }
    IREE_RETURN_IF_ERROR(loom_check_test_schedule_parse_dependency_kind(
        value, &options->dependency_query));
    options->flags |= LOOM_CHECK_TEST_SCHEDULE_OPTION_FLAG_HAS_KIND;
    return iree_ok_status();
  }
  if (iree_string_view_equal(name, IREE_SV("strategy"))) {
    if (iree_any_bit_set(options->flags,
                         LOOM_CHECK_TEST_SCHEDULE_OPTION_FLAG_HAS_STRATEGY)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate low-schedule-query option 'strategy'");
    }
    IREE_RETURN_IF_ERROR(loom_check_low_emit_parse_schedule_strategy(
        value, IREE_SV("low-schedule-query"), &options->schedule_strategy));
    options->flags |= LOOM_CHECK_TEST_SCHEDULE_OPTION_FLAG_HAS_STRATEGY;
    return iree_ok_status();
  }
  if (iree_string_view_equal(name, IREE_SV("timing"))) {
    if (iree_any_bit_set(options->flags,
                         LOOM_CHECK_TEST_SCHEDULE_OPTION_FLAG_HAS_TIMING)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate low-schedule-query option 'timing'");
    }
    IREE_RETURN_IF_ERROR(
        loom_check_test_schedule_parse_timing_edge(value, options));
    options->flags |= LOOM_CHECK_TEST_SCHEDULE_OPTION_FLAG_HAS_TIMING;
    return iree_ok_status();
  }
  iree_string_view_t* node_list = NULL;
  if (iree_string_view_equal(name, IREE_SV("consumer"))) {
    node_list = &options->consumer_nodes;
  } else if (iree_string_view_equal(name, IREE_SV("order"))) {
    node_list = &options->order_nodes;
  } else if (iree_string_view_equal(name, IREE_SV("issue"))) {
    node_list = &options->issue_nodes;
  } else if (iree_string_view_equal(name, IREE_SV("descriptor"))) {
    node_list = &options->descriptor_nodes;
  } else {
    return loom_check_low_emit_parse_allocation_budget(
        token, IREE_SV("low-schedule-query"), options->allocation_budgets,
        IREE_ARRAYSIZE(options->allocation_budgets),
        &options->allocation_budget_count);
  }
  if (!iree_string_view_is_empty(*node_list)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "duplicate low-schedule-query option '%.*s'",
                            (int)name.size, name.data);
  }
  if (!iree_string_view_equal(name, IREE_SV("consumer")) ||
      !iree_string_view_equal(value, IREE_SV("*"))) {
    IREE_RETURN_IF_ERROR(
        loom_check_test_schedule_validate_node_list(name, value));
  }
  *node_list = value;
  return iree_ok_status();
}

static iree_status_t loom_check_test_schedule_parse_options(
    const loom_check_emit_provider_request_t* request,
    loom_check_test_schedule_options_t* out_options) {
  *out_options = (loom_check_test_schedule_options_t){
      .schedule_strategy = LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY,
  };
  iree_string_view_t symbol_name = iree_string_view_empty();
  iree_string_view_t option_text = iree_string_view_empty();
  iree_string_view_split(request->target_options, ' ', &symbol_name,
                         &option_text);
  symbol_name = iree_string_view_trim(symbol_name);
  option_text = iree_string_view_trim(option_text);
  if (!iree_string_view_starts_with(symbol_name, IREE_SV("@")) ||
      symbol_name.size == 1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low-schedule-query requires a low function symbol name");
  }
  out_options->function_symbol_name =
      iree_string_view_substr(symbol_name, 1, IREE_HOST_SIZE_MAX);

  while (!iree_string_view_is_empty(option_text)) {
    iree_string_view_t token = iree_string_view_empty();
    iree_string_view_split(option_text, ' ', &token, &option_text);
    token = iree_string_view_trim(token);
    if (!iree_string_view_is_empty(token)) {
      IREE_RETURN_IF_ERROR(
          loom_check_test_schedule_parse_option(token, out_options));
    }
    option_text = iree_string_view_trim(option_text);
  }
  const bool has_consumers =
      !iree_string_view_is_empty(out_options->consumer_nodes);
  const bool has_kind = iree_any_bit_set(
      out_options->flags, LOOM_CHECK_TEST_SCHEDULE_OPTION_FLAG_HAS_KIND);
  const bool has_timing = iree_any_bit_set(
      out_options->flags, LOOM_CHECK_TEST_SCHEDULE_OPTION_FLAG_HAS_TIMING);
  if ((has_consumers || has_timing) != has_kind) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low-schedule-query option 'kind' is required with 'consumer' or "
        "'timing', and requires at least one of them");
  }
  if (!has_consumers && !has_timing &&
      iree_string_view_is_empty(out_options->order_nodes) &&
      iree_string_view_is_empty(out_options->issue_nodes) &&
      iree_string_view_is_empty(out_options->descriptor_nodes)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low-schedule-query requires option 'consumer', 'order', 'issue', "
        "'descriptor', or 'timing'");
  }
  return iree_ok_status();
}

static iree_string_view_t loom_check_test_schedule_separation_source_name(
    loom_low_schedule_separation_source_t source) {
  switch (source) {
    case LOOM_LOW_SCHEDULE_SEPARATION_SOURCE_STRUCTURAL:
      return IREE_SV("structural");
    case LOOM_LOW_SCHEDULE_SEPARATION_SOURCE_SCHEDULE_CLASS:
      return IREE_SV("schedule-class");
    case LOOM_LOW_SCHEDULE_SEPARATION_SOURCE_EVENT_PAIR:
      return IREE_SV("event-pair");
    default:
      return IREE_SV("unknown");
  }
}

static bool loom_check_test_schedule_node_list_contains(iree_string_view_t list,
                                                        uint32_t node) {
  uint32_t candidate = 0;
  while (loom_check_test_schedule_consume_node(&list, &candidate)) {
    if (candidate == node) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_check_test_schedule_append_order(
    const loom_low_schedule_table_t* schedule, iree_string_view_t selected,
    iree_string_builder_t* builder) {
  if (iree_string_view_is_empty(selected)) {
    return iree_ok_status();
  }
  iree_host_size_t selected_count = 0;
  iree_string_view_t remaining = selected;
  uint32_t node_index = 0;
  while (loom_check_test_schedule_consume_node(&remaining, &node_index)) {
    if (node_index >= schedule->node_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low-schedule-query order node %" PRIu32
                              " is out of range for %" PRIhsz " nodes",
                              node_index, schedule->node_count);
    }
    ++selected_count;
  }

  iree_host_size_t emitted_count = 0;
  for (uint32_t block_index = 0; block_index < schedule->block_count;
       ++block_index) {
    bool block_selected = false;
    remaining = selected;
    while (loom_check_test_schedule_consume_node(&remaining, &node_index)) {
      if (schedule->nodes[node_index].block_index == block_index) {
        block_selected = true;
        break;
      }
    }
    if (!block_selected) {
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "block %" PRIu32 " order: ", block_index));
    iree_host_size_t block_emitted_count = 0;
    const loom_low_schedule_block_t* block = &schedule->blocks[block_index];
    for (uint32_t i = 0; i < block->scheduled_node_count; ++i) {
      node_index =
          schedule->scheduled_node_indices[block->scheduled_node_start + i];
      if (!loom_check_test_schedule_node_list_contains(selected, node_index)) {
        continue;
      }
      if (block_emitted_count++ != 0) {
        IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ", "));
      }
      IREE_RETURN_IF_ERROR(
          iree_string_builder_append_format(builder, "%" PRIu32, node_index));
    }
    emitted_count += block_emitted_count;
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  }
  IREE_ASSERT_EQ(emitted_count, selected_count);
  return iree_ok_status();
}

static iree_status_t loom_check_test_schedule_append_predecessors(
    const loom_low_schedule_table_t* schedule,
    const loom_check_test_schedule_dependency_query_t* query,
    uint32_t consumer_node, iree_string_builder_t* builder) {
  if (consumer_node >= schedule->node_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low-schedule-query consumer node %" PRIu32
                            " is out of range for %" PRIhsz " nodes",
                            consumer_node, schedule->node_count);
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder, "node %" PRIu32 " <- ", consumer_node));
  iree_host_size_t predecessor_count = 0;
  for (uint32_t producer_node = 0; producer_node < schedule->node_count;
       ++producer_node) {
    bool is_predecessor = false;
    for (iree_host_size_t i = 0; i < schedule->dependencies.count; ++i) {
      const loom_low_schedule_dependency_t* dependency =
          loom_low_schedule_dependency_graph_at(&schedule->dependencies,
                                                (uint32_t)i);
      if (dependency->kind == query->kind &&
          dependency->minimum_issue_separation_cycles >=
              query->minimum_issue_separation_cycles &&
          dependency->producer_node == producer_node &&
          dependency->consumer_node == consumer_node) {
        is_predecessor = true;
        break;
      }
    }
    if (!is_predecessor) {
      continue;
    }
    if (predecessor_count++ != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, ", "));
    }
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_format(builder, "%" PRIu32, producer_node));
  }
  return iree_string_builder_append_cstring(
      builder, predecessor_count == 0 ? "none\n" : "\n");
}

static iree_status_t loom_check_test_schedule_append_issue_cycles(
    const loom_low_schedule_table_t* schedule, iree_string_view_t selected,
    iree_string_builder_t* builder) {
  uint32_t node_index = 0;
  while (loom_check_test_schedule_consume_node(&selected, &node_index)) {
    if (node_index >= schedule->node_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low-schedule-query issue node %" PRIu32
                              " is out of range for %" PRIhsz " nodes",
                              node_index, schedule->node_count);
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "node %" PRIu32 " issue-cycle: %" PRIu32 "\n", node_index,
        schedule->nodes[node_index].issue_cycle));
  }
  return iree_ok_status();
}

static iree_status_t loom_check_test_schedule_append_descriptors(
    const loom_low_schedule_table_t* schedule, iree_string_view_t selected,
    iree_string_builder_t* builder) {
  uint32_t node_index = 0;
  while (loom_check_test_schedule_consume_node(&selected, &node_index)) {
    if (node_index >= schedule->node_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low-schedule-query descriptor node %" PRIu32
                              " is out of range for %" PRIhsz " nodes",
                              node_index, schedule->node_count);
    }
    const loom_low_descriptor_t* descriptor =
        schedule->nodes[node_index].descriptor;
    if (descriptor == NULL) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, "node %" PRIu32 " descriptor: none\n", node_index));
      continue;
    }
    const iree_string_view_t descriptor_key = loom_low_descriptor_set_string(
        schedule->target.descriptor_set, descriptor->key_string_ref);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder, "node %" PRIu32 " descriptor: %.*s\n", node_index,
        (int)descriptor_key.size, descriptor_key.data));
  }
  return iree_ok_status();
}

static bool loom_check_test_schedule_has_predecessor(
    const loom_low_schedule_table_t* schedule,
    const loom_check_test_schedule_dependency_query_t* query,
    uint32_t consumer_node) {
  for (iree_host_size_t i = 0; i < schedule->dependencies.count; ++i) {
    const loom_low_schedule_dependency_t* dependency =
        loom_low_schedule_dependency_graph_at(&schedule->dependencies,
                                              (uint32_t)i);
    if (dependency->kind == query->kind &&
        dependency->minimum_issue_separation_cycles >=
            query->minimum_issue_separation_cycles &&
        dependency->consumer_node == consumer_node) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_check_test_schedule_append_all_predecessors(
    const loom_low_schedule_table_t* schedule,
    const loom_check_test_schedule_dependency_query_t* query,
    iree_string_builder_t* builder) {
  iree_host_size_t consumer_count = 0;
  for (uint32_t consumer_node = 0; consumer_node < schedule->node_count;
       ++consumer_node) {
    if (!loom_check_test_schedule_has_predecessor(schedule, query,
                                                  consumer_node)) {
      continue;
    }
    ++consumer_count;
    IREE_RETURN_IF_ERROR(loom_check_test_schedule_append_predecessors(
        schedule, query, consumer_node, builder));
  }
  return consumer_count == 0
             ? iree_string_builder_append_cstring(builder, "none\n")
             : iree_ok_status();
}

static iree_status_t loom_check_test_schedule_append_timing(
    const loom_low_schedule_table_t* schedule,
    const loom_check_test_schedule_options_t* options,
    iree_string_builder_t* builder) {
  if (!iree_any_bit_set(options->flags,
                        LOOM_CHECK_TEST_SCHEDULE_OPTION_FLAG_HAS_TIMING)) {
    return iree_ok_status();
  }
  if (options->timing_producer_node >= schedule->node_count ||
      options->timing_consumer_node >= schedule->node_count) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "low-schedule-query timing edge %" PRIu32 ":%" PRIu32
        " is out of range for %" PRIhsz " nodes",
        options->timing_producer_node, options->timing_consumer_node,
        schedule->node_count);
  }
  const loom_low_schedule_dependency_t* match = NULL;
  for (iree_host_size_t i = 0; i < schedule->dependencies.count; ++i) {
    const loom_low_schedule_dependency_t* dependency =
        loom_low_schedule_dependency_graph_at(&schedule->dependencies,
                                              (uint32_t)i);
    if (dependency->kind != options->dependency_query.kind ||
        dependency->minimum_issue_separation_cycles <
            options->dependency_query.minimum_issue_separation_cycles ||
        dependency->producer_node != options->timing_producer_node ||
        dependency->consumer_node != options->timing_consumer_node) {
      continue;
    }
    if (match != NULL) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low-schedule-query timing edge %" PRIu32 ":%" PRIu32
          " has multiple dependencies of the selected kind",
          options->timing_producer_node, options->timing_consumer_node);
    }
    match = dependency;
  }
  if (match == NULL) {
    return iree_string_builder_append_format(
        builder, "timing %" PRIu32 " -> %" PRIu32 ": none\n",
        options->timing_producer_node, options->timing_consumer_node);
  }
  const iree_string_view_t separation_source =
      loom_check_test_schedule_separation_source_name(match->separation_source);
  const iree_string_view_t model_quality = loom_low_model_quality_name(
      (loom_low_model_quality_t)match->model_quality);
  return iree_string_builder_append_format(
      builder,
      "timing %" PRIu32 " -> %" PRIu32 ": separation=%" PRId32
      " source=%.*s quality=%.*s\n",
      options->timing_producer_node, options->timing_consumer_node,
      match->minimum_issue_separation_cycles, (int)separation_source.size,
      separation_source.data, (int)model_quality.size, model_quality.data);
}

static iree_status_t loom_check_test_schedule_build(
    const loom_check_emit_provider_request_t* request,
    const loom_check_test_schedule_options_t* options,
    loom_low_schedule_table_t* out_schedule, bool* out_accepted) {
  *out_accepted = false;
  loom_check_diagnostic_emitter_capture_t diagnostic_capture = {
      .diagnostic_collector = request->diagnostic_collector,
      .module = request->module,
      .source_resolver = request->source_resolver,
      .emitter = LOOM_EMITTER_PASS,
  };
  iree_diagnostic_emitter_t emitter = {0};
  if (request->diagnostic_collector != NULL) {
    emitter = (iree_diagnostic_emitter_t){
        .fn = loom_check_diagnostic_emitter_capture_emit,
        .user_data = &diagnostic_capture,
    };
  }
  loom_op_t* low_function = NULL;
  IREE_RETURN_IF_ERROR(loom_check_low_emit_find_low_function_def(
      request->module, options->function_symbol_name, request->test_case,
      request->filename, request->diagnostic_collector, emitter,
      &low_function));
  if (low_function == NULL) {
    return iree_ok_status();
  }

  const loom_low_schedule_options_t schedule_options = {
      .allocation_budgets = options->allocation_budgets,
      .allocation_budget_count = options->allocation_budget_count,
      .emitter = emitter,
      .flags = LOOM_LOW_SCHEDULE_FLAG_RETAIN_LIVENESS |
               LOOM_LOW_SCHEDULE_FLAG_RETAIN_PRESSURE_STEPS,
      .strategy = options->schedule_strategy,
  };
  loom_low_function_model_t model = {0};
  iree_status_t status = loom_low_function_model_initialize(
      request->module, low_function,
      /*function_target_facts=*/NULL, &request->low_registry->registry, emitter,
      /*flags=*/0, request->case_arena, &model);
  if (iree_status_is_ok(status)) {
    status = loom_low_schedule_function(&model, &schedule_options,
                                        request->case_arena, out_schedule);
  }
  if (iree_status_is_ok(status)) {
    *out_accepted = out_schedule->error_count == 0;
  }
  loom_low_function_model_deinitialize(&model);
  return status;
}

static iree_status_t loom_check_test_schedule_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request) {
  (void)provider;
  loom_check_test_schedule_options_t options;
  IREE_RETURN_IF_ERROR(
      loom_check_test_schedule_parse_options(request, &options));

  loom_low_schedule_table_t schedule = {0};
  bool schedule_accepted = false;
  IREE_RETURN_IF_ERROR(loom_check_test_schedule_build(
      request, &options, &schedule, &schedule_accepted));
  if (request->diagnostic_collector != NULL &&
      loom_check_diagnostic_collector_has_error(
          request->diagnostic_collector)) {
    return iree_ok_status();
  }
  if (!schedule_accepted) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_check_test_schedule_append_order(
      &schedule, options.order_nodes, &request->result->actual_output));
  IREE_RETURN_IF_ERROR(loom_check_test_schedule_append_issue_cycles(
      &schedule, options.issue_nodes, &request->result->actual_output));
  IREE_RETURN_IF_ERROR(loom_check_test_schedule_append_descriptors(
      &schedule, options.descriptor_nodes, &request->result->actual_output));
  if (iree_string_view_equal(options.consumer_nodes, IREE_SV("*"))) {
    IREE_RETURN_IF_ERROR(loom_check_test_schedule_append_all_predecessors(
        &schedule, &options.dependency_query, &request->result->actual_output));
  } else {
    iree_string_view_t consumers = options.consumer_nodes;
    uint32_t consumer_node = 0;
    while (loom_check_test_schedule_consume_node(&consumers, &consumer_node)) {
      IREE_RETURN_IF_ERROR(loom_check_test_schedule_append_predecessors(
          &schedule, &options.dependency_query, consumer_node,
          &request->result->actual_output));
    }
  }
  return loom_check_test_schedule_append_timing(
      &schedule, &options, &request->result->actual_output);
}

static iree_status_t loom_check_test_schedule_append_names(
    const loom_check_emit_provider_t* provider,
    iree_string_builder_t* builder) {
  (void)provider;
  return iree_string_builder_append_cstring(builder, "low-schedule-query");
}

const loom_check_emit_provider_t loom_check_test_schedule_provider = {
    .name = IREE_SVL("schedule-query"),
    .match = loom_check_test_schedule_matches,
    .execute = loom_check_test_schedule_execute,
    .append_names = loom_check_test_schedule_append_names,
};
