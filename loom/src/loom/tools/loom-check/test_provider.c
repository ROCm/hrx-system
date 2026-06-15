// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Synthetic provider support for checked-in .loom-test files.

#include "loom/tools/loom-check/test_provider.h"

#include "iree/base/api.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/packet_hazard_plan.h"
#include "loom/codegen/low/packet_hazard_plan_json.h"
#include "loom/codegen/low/packet_progress.h"
#include "loom/ir/ir.h"
#include "loom/target/test/provider.h"
#include "loom/tools/loom-check/diagnostics.h"
#include "loom/tools/loom-check/execute.h"
#include "loom/tools/loom-check/low_emit.h"

enum {
  LOOM_CHECK_TEST_SYNTHETIC_HAZARD_REASON_PHYSICAL_OVERLAP = 1,
  LOOM_CHECK_TEST_SYNTHETIC_HAZARD_REASON_MISSING_DATA = 2,
  LOOM_CHECK_TEST_SYNTHETIC_PROGRESS_CLASS_ISSUE = 1,
  LOOM_CHECK_TEST_SYNTHETIC_HAZARD_ACTION_PADDING = 1,
};

typedef enum loom_check_test_synthetic_hazard_case_e {
  LOOM_CHECK_TEST_SYNTHETIC_HAZARD_CASE_ACTION = 0,
  LOOM_CHECK_TEST_SYNTHETIC_HAZARD_CASE_MISSING = 1,
} loom_check_test_synthetic_hazard_case_t;

typedef struct loom_check_test_synthetic_hazard_options_t {
  // Module-local target-low function symbol selected by the RUN line.
  iree_string_view_t function_symbol_name;
  // Scenario selected by the test fixture.
  loom_check_test_synthetic_hazard_case_t test_case;
  // Candidate selection strategy used by low frame construction.
  loom_low_schedule_strategy_t schedule_strategy;
  // True once a strategy option has been parsed.
  bool has_schedule_strategy_option;
  // Low allocation budget overrides parsed from target options.
  loom_low_allocation_budget_t
      allocation_budgets[LOOM_CHECK_LOW_EMIT_MAX_ALLOCATION_BUDGETS];
  // Number of entries in |allocation_budgets|.
  iree_host_size_t allocation_budget_count;
  // Fixed low allocation requests parsed from target options.
  loom_check_low_emit_fixed_value_spec_t allocation_fixed_value_specs
      [LOOM_CHECK_LOW_EMIT_MAX_ALLOCATION_FIXED_VALUES];
  // Number of entries in |allocation_fixed_value_specs|.
  iree_host_size_t allocation_fixed_value_spec_count;
} loom_check_test_synthetic_hazard_options_t;

typedef struct loom_check_test_synthetic_hazard_context_t {
  // Scenario selected by the test fixture.
  loom_check_test_synthetic_hazard_case_t test_case;
  // Producer packet node found in the scheduled function.
  uint32_t producer_node_index;
  // Consumer packet node found in the scheduled function.
  uint32_t consumer_node_index;
  // SSA result defined by |producer_node_index|.
  loom_value_id_t producer_value_id;
  // SSA result defined by |consumer_node_index|.
  loom_value_id_t consumer_value_id;
} loom_check_test_synthetic_hazard_context_t;

static bool loom_check_test_synthetic_hazard_matches(
    const loom_check_emit_provider_t* provider,
    iree_string_view_t target_name) {
  (void)provider;
  return iree_string_view_equal(target_name,
                                IREE_SV("low-packet-hazard-plan-json"));
}

static iree_status_t loom_check_test_synthetic_hazard_parse_named_option(
    iree_string_view_t name, iree_string_view_t value,
    loom_check_test_synthetic_hazard_options_t* options, bool* out_matched) {
  *out_matched = true;
  if (iree_string_view_equal(name, IREE_SV("case"))) {
    if (iree_string_view_equal(value, IREE_SV("action"))) {
      options->test_case = LOOM_CHECK_TEST_SYNTHETIC_HAZARD_CASE_ACTION;
      return iree_ok_status();
    }
    if (iree_string_view_equal(value, IREE_SV("missing"))) {
      options->test_case = LOOM_CHECK_TEST_SYNTHETIC_HAZARD_CASE_MISSING;
      return iree_ok_status();
    }
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "synthetic hazard option 'case' expected 'action' or 'missing', got "
        "'%.*s'",
        (int)value.size, value.data);
  }
  if (iree_string_view_equal(name, IREE_SV("strategy"))) {
    if (options->has_schedule_strategy_option) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate synthetic hazard option 'strategy'");
    }
    IREE_RETURN_IF_ERROR(loom_check_low_emit_parse_schedule_strategy(
        value, IREE_SV("synthetic hazard"), &options->schedule_strategy));
    options->has_schedule_strategy_option = true;
    return iree_ok_status();
  }
  *out_matched = false;
  return iree_ok_status();
}

static iree_status_t loom_check_test_synthetic_hazard_parse_option(
    iree_string_view_t token,
    loom_check_test_synthetic_hazard_options_t* options) {
  iree_string_view_t name = iree_string_view_empty();
  iree_string_view_t value = iree_string_view_empty();
  iree_string_view_split(token, '=', &name, &value);
  name = iree_string_view_trim(name);
  value = iree_string_view_trim(value);

  bool matched = false;
  IREE_RETURN_IF_ERROR(loom_check_test_synthetic_hazard_parse_named_option(
      name, value, options, &matched));
  if (matched) {
    return iree_ok_status();
  }
  return loom_check_low_emit_parse_allocation_option(
      token, IREE_SV("synthetic hazard"), options->allocation_budgets,
      IREE_ARRAYSIZE(options->allocation_budgets),
      &options->allocation_budget_count, options->allocation_fixed_value_specs,
      IREE_ARRAYSIZE(options->allocation_fixed_value_specs),
      &options->allocation_fixed_value_spec_count);
}

static iree_status_t loom_check_test_synthetic_hazard_parse_emit_options(
    const loom_check_emit_provider_request_t* request,
    loom_check_test_synthetic_hazard_options_t* out_options) {
  *out_options = (loom_check_test_synthetic_hazard_options_t){
      .test_case = LOOM_CHECK_TEST_SYNTHETIC_HAZARD_CASE_ACTION,
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
        "synthetic hazard plan requires a low function symbol name");
  }
  out_options->function_symbol_name =
      iree_string_view_substr(symbol_name, 1, IREE_HOST_SIZE_MAX);

  while (!iree_string_view_is_empty(option_text)) {
    iree_string_view_t token = iree_string_view_empty();
    iree_string_view_t remaining = iree_string_view_empty();
    iree_string_view_split(option_text, ' ', &token, &remaining);
    token = iree_string_view_trim(token);
    if (!iree_string_view_is_empty(token)) {
      IREE_RETURN_IF_ERROR(
          loom_check_test_synthetic_hazard_parse_option(token, out_options));
    }
    option_text = iree_string_view_trim(remaining);
  }
  return iree_ok_status();
}

static iree_status_t loom_check_test_synthetic_hazard_find_pair(
    const loom_low_schedule_table_t* schedule,
    loom_check_test_synthetic_hazard_context_t* context) {
  context->producer_node_index = LOOM_LOW_SCHEDULE_NODE_NONE;
  context->consumer_node_index = LOOM_LOW_SCHEDULE_NODE_NONE;
  context->producer_value_id = LOOM_VALUE_ID_INVALID;
  context->consumer_value_id = LOOM_VALUE_ID_INVALID;
  for (iree_host_size_t packet_index = 0;
       packet_index < loom_low_packet_count(schedule); ++packet_index) {
    uint32_t node_index = LOOM_LOW_SCHEDULE_NODE_NONE;
    IREE_RETURN_IF_ERROR(
        loom_low_packet_node_index_at(schedule, packet_index, &node_index));
    if (node_index >= schedule->node_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "synthetic hazard packet index %" PRIhsz
                              " references out-of-range node %" PRIu32,
                              packet_index, node_index);
    }
    const loom_low_schedule_node_t* node = &schedule->nodes[node_index];
    if (node->kind != LOOM_LOW_SCHEDULE_NODE_DESCRIPTOR ||
        node->op->result_count == 0) {
      continue;
    }
    if (context->producer_node_index == LOOM_LOW_SCHEDULE_NODE_NONE) {
      context->producer_node_index = node_index;
      context->producer_value_id = loom_op_const_results(node->op)[0];
      continue;
    }
    context->consumer_node_index = node_index;
    context->consumer_value_id = loom_op_const_results(node->op)[0];
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "synthetic hazard plan requires two descriptor packets with results");
}

static iree_status_t loom_check_test_synthetic_hazard_emit_missing(
    loom_low_packet_hazard_plan_emit_fn_t emit, void* emit_user_data) {
  const loom_low_packet_hazard_plan_event_t event = {
      .kind = LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_MISSING_TARGET_DATA,
      .reason_id = LOOM_CHECK_TEST_SYNTHETIC_HAZARD_REASON_MISSING_DATA,
      .reason_name = IREE_SV("synthetic.missing-target-data"),
      .producer_node_index = LOOM_LOW_SCHEDULE_NODE_NONE,
      .progress_class_id = LOOM_LOW_PACKET_PROGRESS_CLASS_NONE,
      .target_detail = IREE_SV("synthetic semantic tag unavailable"),
  };
  return emit(emit_user_data, &event);
}

static iree_status_t loom_check_test_synthetic_hazard_query(
    void* user_data, const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_progress_table_t* progress,
    const loom_low_packet_view_t* packet,
    loom_low_packet_hazard_plan_emit_fn_t emit, void* emit_user_data) {
  (void)progress;
  const loom_check_test_synthetic_hazard_context_t* context =
      (const loom_check_test_synthetic_hazard_context_t*)user_data;
  if (context->test_case == LOOM_CHECK_TEST_SYNTHETIC_HAZARD_CASE_MISSING) {
    if (packet->packet_index == 0) {
      return loom_check_test_synthetic_hazard_emit_missing(emit,
                                                           emit_user_data);
    }
    return iree_ok_status();
  }
  if (packet->node_index != context->consumer_node_index) {
    return iree_ok_status();
  }
  if (allocation == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "synthetic hazard physical-overlap predicate requires allocation");
  }

  const loom_low_allocation_assignment_t* producer_assignment =
      loom_low_allocation_try_map_active_value_assignment(
          allocation, context->producer_value_id,
          /*out_assignment_index=*/NULL);
  const loom_low_allocation_assignment_t* consumer_assignment =
      loom_low_allocation_try_map_active_value_assignment(
          allocation, context->consumer_value_id,
          /*out_assignment_index=*/NULL);
  const loom_low_descriptor_set_t* descriptor_set =
      schedule->target.descriptor_set;
  if (!loom_low_allocation_storage_assignment_ranges_overlap(
          descriptor_set, producer_assignment, consumer_assignment)) {
    return iree_ok_status();
  }

  const loom_low_packet_hazard_plan_event_t event = {
      .kind = LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_ACTION,
      .action_id = LOOM_CHECK_TEST_SYNTHETIC_HAZARD_ACTION_PADDING,
      .action_name = IREE_SV("synthetic.padding"),
      .reason_id = LOOM_CHECK_TEST_SYNTHETIC_HAZARD_REASON_PHYSICAL_OVERLAP,
      .reason_name = IREE_SV("synthetic.physical-overlap"),
      .producer_node_index = context->producer_node_index,
      .progress_class_id = LOOM_CHECK_TEST_SYNTHETIC_PROGRESS_CLASS_ISSUE,
      .progress_class_name = IREE_SV("synthetic.issue"),
      .required_progress = 2,
      .observed_progress = 0,
      .residual_progress = 2,
  };
  return emit(emit_user_data, &event);
}

static iree_status_t loom_check_test_synthetic_hazard_progress_query(
    void* user_data, const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_view_t* packet,
    loom_low_packet_progress_emit_fn_t emit, void* emit_user_data) {
  (void)user_data;
  (void)schedule;
  (void)allocation;
  if (packet->descriptor == NULL) {
    return iree_ok_status();
  }
  const loom_low_packet_progress_event_t event = {
      .progress_class_id = LOOM_CHECK_TEST_SYNTHETIC_PROGRESS_CLASS_ISSUE,
      .progress_class_name = IREE_SV("synthetic.issue"),
      .action = LOOM_LOW_PACKET_PROGRESS_ACTION_ADVANCE,
      .units = 1,
  };
  return emit(emit_user_data, &event);
}

static iree_status_t loom_check_test_synthetic_hazard_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request) {
  (void)provider;
  loom_check_test_synthetic_hazard_options_t options;
  IREE_RETURN_IF_ERROR(
      loom_check_test_synthetic_hazard_parse_emit_options(request, &options));

  loom_low_emission_frame_t frame = {0};
  IREE_RETURN_IF_ERROR(loom_check_low_emit_packetize_function(
      request, options.function_symbol_name, options.schedule_strategy,
      options.allocation_budgets, options.allocation_budget_count,
      options.allocation_fixed_value_specs,
      options.allocation_fixed_value_spec_count,
      loom_low_schedule_pair_affinity_list_empty(),
      /*storage_lease_provider=*/NULL, /*spill_free_options=*/NULL, &frame));
  if (request->diagnostic_collector != NULL &&
      request->diagnostic_collector->count != 0) {
    return iree_ok_status();
  }

  loom_check_test_synthetic_hazard_context_t context = {
      .test_case = options.test_case,
  };
  IREE_RETURN_IF_ERROR(
      loom_check_test_synthetic_hazard_find_pair(&frame.schedule, &context));

  loom_low_allocation_value_scratch_t scratch = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_acquire_value_scratch(&frame.allocation, &scratch));
  const loom_low_packet_progress_provider_t progress_provider = {
      .query = loom_check_test_synthetic_hazard_progress_query,
  };
  loom_low_packet_progress_table_t progress = {0};
  const loom_low_packet_hazard_plan_provider_t hazard_provider = {
      .user_data = &context,
      .query = loom_check_test_synthetic_hazard_query,
  };
  loom_low_packet_hazard_plan_t plan = {0};
  iree_status_t status = loom_low_packet_progress_build(
      &frame.schedule, &frame.allocation, &progress_provider,
      request->case_arena, &progress);
  if (iree_status_is_ok(status)) {
    status = loom_low_packet_hazard_plan_build(
        &frame.schedule, &frame.allocation, &progress, &hazard_provider,
        request->case_arena, &plan);
  }
  loom_low_allocation_release_value_scratch(&scratch);
  IREE_RETURN_IF_ERROR(status);
  return loom_low_packet_hazard_plan_format_json(
      &plan, &request->result->actual_output);
}

static iree_status_t loom_check_test_synthetic_hazard_append_names(
    const loom_check_emit_provider_t* provider,
    iree_string_builder_t* builder) {
  (void)provider;
  return iree_string_builder_append_cstring(builder,
                                            "low-packet-hazard-plan-json");
}

static const loom_check_emit_provider_t kLoomCheckTestSyntheticHazardProvider =
    {
        .name = IREE_SVL("synthetic-hazard-plan"),
        .match = loom_check_test_synthetic_hazard_matches,
        .execute = loom_check_test_synthetic_hazard_execute,
        .append_names = loom_check_test_synthetic_hazard_append_names,
};

static const loom_check_emit_provider_t* const kLoomCheckTestEmitProviders[] = {
    &kLoomCheckTestSyntheticHazardProvider,
};

const loom_check_provider_t loom_check_test_provider = {
    .name = IREE_SVL("test"),
    .target_provider = &loom_test_target_provider,
    .emit_providers = kLoomCheckTestEmitProviders,
    .emit_provider_count = IREE_ARRAYSIZE(kLoomCheckTestEmitProviders),
};
