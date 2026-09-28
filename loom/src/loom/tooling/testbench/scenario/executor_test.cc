// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/testbench/scenario/executor.h"

#include <string>

#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/check/ops.h"
#include "loom/ops/test/ops.h"

namespace loom {
namespace {

typedef struct ExecutionTimeline {
  iree_host_size_t clock;
  iree_host_size_t last_prepare_tick;
  iree_host_size_t first_execute_tick;
} ExecutionTimeline;

typedef struct TestProfileState {
  ExecutionTimeline* timeline;
  iree_host_size_t prepare_count;
  iree_host_size_t execute_count;
  iree_host_size_t last_call_count;
  iree_host_size_t benchmark_count;
  iree_host_size_t last_benchmark_call_count;
  iree_host_size_t mismatch_trial_ordinal;
  iree_host_size_t device_event_trial_ordinal;
  iree_hal_device_event_sink_t device_event_sink;
  bool inject_mismatch;
  bool inject_device_event;
  bool device_event_changes_in_isolation;
  bool saw_runtime_value_during_prepare;
  bool saw_incomplete_alias;
  bool saw_shared_replica_storage;
} TestProfileState;

static iree_status_t ExecuteTestProduct(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t call_count, loom_testbench_product_call_t* calls) {
  TestProfileState* state = static_cast<TestProfileState*>(user_data);
  ++state->execute_count;
  state->last_call_count = call_count;
  const iree_host_size_t tick = ++state->timeline->clock;
  if (state->timeline->first_execute_tick == 0) {
    state->timeline->first_execute_tick = tick;
  }

  for (iree_host_size_t call_index = 0; call_index < call_count; ++call_index) {
    loom_testbench_product_call_t* call = &calls[call_index];
    if (state->inject_device_event &&
        call->identity->trial_ordinal == state->device_event_trial_ordinal) {
      iree_hal_device_event_site_t site = iree_hal_device_event_site_default();
      site.site_id = 17;
      site.source_file = IREE_SV("event_subject.loom");
      site.start_line = 23;
      site.start_column = 5;
      iree_hal_device_event_t event = iree_hal_device_event_default();
      event.type = IREE_HAL_DEVICE_EVENT_TYPE_ASAN_REPORT;
      event.severity =
          state->device_event_changes_in_isolation && call_count == 1
              ? IREE_HAL_DEVICE_EVENT_SEVERITY_WARNING
              : IREE_HAL_DEVICE_EVENT_SEVERITY_ERROR;
      event.source.driver_id = IREE_SV("test");
      event.source.device_id = IREE_SV("scenario");
      event.site = &site;
      iree_hal_device_event_sink_publish(state->device_event_sink, &event);
    }
    if (invocation->result_count == 1) {
      if (invocation->input_count != 1 ||
          call->arguments[0].kind != LOOM_TESTBENCH_VALUE_KIND_SCALAR) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "identity test product requires one scalar");
      }
      loom_testbench_value_retain(&call->arguments[0], &call->results[0]);
      continue;
    }
    if (invocation->result_count != 0 || invocation->input_count != 2 ||
        call->arguments[0].kind != LOOM_TESTBENCH_VALUE_KIND_BUFFER ||
        call->arguments[1].kind != LOOM_TESTBENCH_VALUE_KIND_BUFFER) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "update test product requires two buffers");
    }
    const iree_tooling_buffer_binding_t* storage = &call->arguments[0].buffer;
    const iree_tooling_buffer_binding_t* tail = &call->arguments[1].buffer;
    state->saw_incomplete_alias |=
        iree_hal_buffer_test_overlap(storage->buffer, 0, IREE_HAL_WHOLE_BUFFER,
                                     tail->buffer, 0, IREE_HAL_WHOLE_BUFFER) !=
        IREE_HAL_BUFFER_OVERLAP_PARTIAL;
    int32_t value = 0;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_map_read(
        storage->buffer, storage->byte_offset, &value, sizeof(value)));
    value += static_cast<int32_t>(call->identity->trial_ordinal + 1);
    if (state->inject_mismatch &&
        call->identity->trial_ordinal == state->mismatch_trial_ordinal) {
      ++value;
    }
    IREE_RETURN_IF_ERROR(iree_hal_buffer_map_write(
        storage->buffer, storage->byte_offset, &value, sizeof(value)));
  }
  return iree_ok_status();
}

static iree_status_t BenchmarkTestProduct(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    iree_host_size_t call_count, loom_testbench_product_call_t* calls,
    const loom_run_benchmark_options_t* options,
    iree_allocator_t host_allocator, loom_run_benchmark_result_t* out_result) {
  (void)host_allocator;
  TestProfileState* state = static_cast<TestProfileState*>(user_data);
  ++state->benchmark_count;
  state->last_benchmark_call_count = call_count;
  if (call_count != options->batch_size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "test benchmark call count is inconsistent");
  }
  if (invocation->result_count == 0 && invocation->input_count == 2) {
    for (iree_host_size_t call_index = 0; call_index < call_count;
         ++call_index) {
      const iree_tooling_buffer_binding_t* storage =
          &calls[call_index].arguments[0].buffer;
      const iree_tooling_buffer_binding_t* tail =
          &calls[call_index].arguments[1].buffer;
      state->saw_incomplete_alias |=
          iree_hal_buffer_test_overlap(
              storage->buffer, 0, IREE_HAL_WHOLE_BUFFER, tail->buffer, 0,
              IREE_HAL_WHOLE_BUFFER) != IREE_HAL_BUFFER_OVERLAP_PARTIAL;
      if (call_index != 0) {
        const iree_tooling_buffer_binding_t* previous_storage =
            &calls[call_index - 1].arguments[0].buffer;
        state->saw_shared_replica_storage |=
            iree_hal_buffer_test_overlap(previous_storage->buffer, 0,
                                         IREE_HAL_WHOLE_BUFFER, storage->buffer,
                                         0, IREE_HAL_WHOLE_BUFFER) !=
            IREE_HAL_BUFFER_OVERLAP_DISJOINT;
      }
    }
  }
  loom_run_benchmark_result_initialize(out_result);
  out_result->batch_size = options->batch_size;
  return iree_ok_status();
}

static iree_status_t PrepareTestProduct(
    void* user_data, const loom_testbench_invocation_plan_t* invocation,
    const loom_testbench_value_table_t* configuration,
    iree_allocator_t host_allocator,
    loom_testbench_prepared_product_t* out_product) {
  (void)host_allocator;
  TestProfileState* state = static_cast<TestProfileState*>(user_data);
  ++state->prepare_count;
  state->timeline->last_prepare_tick = ++state->timeline->clock;
  for (iree_host_size_t i = 0; i < invocation->workload_count; ++i) {
    state->saw_runtime_value_during_prepare |=
        loom_testbench_value_table_contains(configuration,
                                            invocation->workload_value_ids[i]);
  }
  for (iree_host_size_t i = 0; i < invocation->input_count; ++i) {
    state->saw_runtime_value_during_prepare |=
        loom_testbench_value_table_contains(configuration,
                                            invocation->input_value_ids[i]);
  }
  *out_product = {};
  out_product->execute = ExecuteTestProduct;
  out_product->benchmark = BenchmarkTestProduct;
  out_product->user_data = state;
  return iree_ok_status();
}

struct DeviceEventScenarioRun {
  loom_testbench_device_event_capture_t capture = {};
  loom_testbench_scenario_configuration_values_t configuration = {};
  ExecutionTimeline timeline = {};
  TestProfileState target_state = {};
  TestProfileState oracle_state = {};
  loom_testbench_prepared_scenario_configuration_t prepared = {};
  loom_testbench_scenario_trial_executor_t executor = {};
  loom_testbench_scenario_trial_result_list_t results = {};

  ~DeviceEventScenarioRun() {
    loom_testbench_scenario_trial_executor_deinitialize(&executor);
    loom_testbench_prepared_scenario_configuration_deinitialize(&prepared);
    loom_testbench_scenario_configuration_values_deinitialize(&configuration);
    loom_testbench_device_event_capture_deinitialize(&capture);
  }
};

static iree_status_t InitializeDeviceEventScenarioRun(
    const loom_module_t* module, const loom_testbench_scenario_plan_t* scenario,
    const loom_testbench_value_materializer_options_t* materializer,
    iree_host_size_t device_event_trial_ordinal,
    bool device_event_changes_in_isolation, iree_allocator_t host_allocator,
    DeviceEventScenarioRun* out_run) {
  IREE_RETURN_IF_ERROR(loom_testbench_device_event_capture_initialize(
      /*record_capacity=*/8, host_allocator, &out_run->capture));
  IREE_RETURN_IF_ERROR(loom_testbench_scenario_configuration_values_initialize(
      module, scenario, host_allocator, &out_run->configuration));
  IREE_RETURN_IF_ERROR(loom_testbench_scenario_configuration_values_materialize(
      materializer, loom_testbench_entropy_root(0x1122334455667788ull),
      /*configuration_ordinal=*/0, &out_run->configuration));

  out_run->target_state.timeline = &out_run->timeline;
  out_run->target_state.device_event_trial_ordinal = device_event_trial_ordinal;
  out_run->target_state.device_event_sink =
      loom_testbench_device_event_capture_sink(&out_run->capture);
  out_run->target_state.inject_device_event = true;
  out_run->target_state.device_event_changes_in_isolation =
      device_event_changes_in_isolation;
  out_run->oracle_state.timeline = &out_run->timeline;
  loom_testbench_scenario_execution_options_t execution_options = {};
  loom_testbench_scenario_execution_options_initialize(&execution_options);
  execution_options.target.name = IREE_SV("test-target");
  execution_options.target.prepare = PrepareTestProduct;
  execution_options.target.user_data = &out_run->target_state;
  execution_options.oracle.name = IREE_SV("test-oracle");
  execution_options.oracle.prepare = PrepareTestProduct;
  execution_options.oracle.user_data = &out_run->oracle_state;
  execution_options.device_event_capture = &out_run->capture;
  IREE_RETURN_IF_ERROR(loom_testbench_prepare_scenario_configuration(
      &execution_options, &out_run->configuration,
      LOOM_TESTBENCH_SCENARIO_EXECUTION_MODE_CORRECTNESS, &out_run->prepared));
  IREE_RETURN_IF_ERROR(loom_testbench_scenario_trial_executor_initialize(
      &out_run->prepared, /*trial_index=*/0, materializer,
      scenario->trials[0].trial_count, &out_run->executor));
  return loom_testbench_run_scenario_trial_batch(
      &out_run->executor, /*first_trial_ordinal=*/0,
      scenario->trials[0].trial_count, &out_run->results);
}

class ScenarioExecutorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, host_allocator_, &block_pool_);
    iree_arena_initialize(&block_pool_, &plan_arena_);
    loom_context_initialize(host_allocator_, &context_);
    RegisterDialect(LOOM_DIALECT_CHECK, loom_check_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(iree_hal_allocator_create_heap(
        IREE_SV("scenario_executor"), host_allocator_, host_allocator_,
        &device_allocator_));
  }

  void TearDown() override {
    iree_hal_allocator_release(device_allocator_);
    iree_arena_deinitialize(&plan_arena_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  using DialectVtablesFn = const loom_op_vtable_t* const* (*)(iree_host_size_t *
                                                              out_count);

  void RegisterDialect(loom_dialect_id_t dialect_id, DialectVtablesFn fn) {
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables = fn(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)vtable_count));
  }

  loom_module_t* ParseModule() {
    static const char source[] = R"(
test.func @update(%storage: tensor<4xi32>, %tail: tensor<2xi32>) {
  test.yield
}

test.func @update_oracle(%storage: tensor<4xi32>, %tail: tensor<2xi32>) {
  test.yield
}

test.func @identity(%input: i32) -> (i32) {
  test.yield %input : i32
}

check.scenario public @batched configure[2](%configuration: index, %configuration_entropy: check.entropy) {
  %configuration_stream = check.entropy.fork %configuration_entropy name("configuration") : check.entropy
  %configuration_word = check.entropy.read %configuration_stream[%configuration] : check.entropy -> i64
  check.trial[3](%trial: index, %entropy: check.entropy) {
    %input_stream = check.entropy.fork %entropy name("input") : check.entropy
    %seed = check.entropy.read %input_stream[0] : check.entropy -> i64
    %storage = check.generate.random.uniform seed(%seed) range(-16 to 16) : tensor<4xi32>
    %tail = check.tensor.view %storage offset(8) : tensor<4xi32> -> tensor<2xi32>
    check.compare<@update, @update_oracle>(%storage, %tail) : (tensor<4xi32>, tensor<2xi32>) -> () {
      check.expect.bitwise actual(%storage) expected(%storage) : tensor<4xi32>
      check.expect.bitwise actual(%tail) expected(%tail) : tensor<2xi32>
    }
  }
  check.trial[2](%trial: index, %entropy: check.entropy) {
    %input = check.literal value(11) : i32
    check.invoke<@identity>(%input) : (i32) -> (i32)
  }
  check.return
}
)";
    loom_text_parse_options_t options = {};
    options.max_errors = 20;
    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(loom_text_parse(iree_make_cstring_view(source),
                                   IREE_SV("scenario_executor_test.loom"),
                                   &context_, &block_pool_, &options, &module));
    EXPECT_NE(module, nullptr);
    return module;
  }

  loom_module_t* ParseDeviceEventModule() {
    static const char source[] = R"(
test.func @update(%storage: tensor<4xi32>, %tail: tensor<2xi32>) {
  test.yield
}

check.scenario public @unexpected_device_event {
  check.trial[3](%trial: index, %entropy: check.entropy) {
    %storage = check.generate.iota offset(0) step(1) : tensor<4xi32>
    %tail = check.tensor.view %storage offset(8) : tensor<4xi32> -> tensor<2xi32>
    check.compare<@update>(%storage, %tail) : (tensor<4xi32>, tensor<2xi32>) -> () {
      check.expect.bitwise actual(%storage) expected(%storage) : tensor<4xi32>
    }
  }
  check.return
}

check.scenario public @expected_device_event {
  check.trial[2](%trial: index, %entropy: check.entropy) {
    %storage = check.generate.iota offset(0) step(1) : tensor<4xi32>
    %tail = check.tensor.view %storage offset(8) : tensor<4xi32> -> tensor<2xi32>
    check.compare<@update>(%storage, %tail) : (tensor<4xi32>, tensor<2xi32>) -> () {
      check.expect.bitwise actual(%storage) expected(%storage) : tensor<4xi32>
      check.expect.event<device> {type = "asan_report", count = 1}
    }
  }
  check.return
}
)";
    loom_text_parse_options_t options = {};
    options.max_errors = 20;
    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(loom_text_parse(iree_make_cstring_view(source),
                                   IREE_SV("scenario_device_events.loom"),
                                   &context_, &block_pool_, &options, &module));
    EXPECT_NE(module, nullptr);
    return module;
  }

  loom_testbench_module_plan_t PlanModule(loom_module_t* module) {
    loom_testbench_module_plan_t plan = {};
    IREE_EXPECT_OK(
        loom_testbench_plan_module(module, nullptr, &plan_arena_, &plan));
    return plan;
  }

  loom_testbench_value_materializer_options_t MaterializerOptions() {
    loom_testbench_value_materializer_options_t options = {};
    loom_testbench_value_materializer_options_initialize(&options);
    options.device_allocator = device_allocator_;
    return options;
  }

  static const loom_testbench_value_t* Lookup(
      const loom_testbench_value_table_t* table, loom_value_id_t value_id) {
    const loom_testbench_value_t* value = nullptr;
    IREE_EXPECT_OK(
        loom_testbench_value_table_lookup_borrow(table, value_id, &value));
    return value;
  }

  static std::string WriteResultJson(
      const loom_testbench_scenario_trial_result_t& result) {
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    loom_output_stream_t stream;
    loom_output_stream_for_builder(&builder, &stream);
    IREE_EXPECT_OK(
        loom_testbench_scenario_trial_result_write_json(&result, &stream));
    const iree_string_view_t view = iree_string_builder_view(&builder);
    std::string json(view.data, view.size);
    iree_string_builder_deinitialize(&builder);
    return json;
  }

  iree_allocator_t host_allocator_ = iree_allocator_system();
  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t plan_arena_;
  loom_context_t context_;
  iree_hal_allocator_t* device_allocator_ = nullptr;
};

TEST_F(ScenarioExecutorTest, PreparesBeforeMaterializationAndExecutesBatches) {
  loom_module_t* module = ParseModule();
  ASSERT_NE(module, nullptr);
  loom_testbench_module_plan_t plan = PlanModule(module);
  ASSERT_EQ(plan.issue_count, 0u);
  ASSERT_EQ(plan.scenario_count, 1u);
  const loom_testbench_scenario_plan_t& scenario = plan.scenarios[0];
  EXPECT_NE(scenario.trials[0].action.target.callee_ref.symbol_id,
            scenario.trials[0].action.oracle.callee_ref.symbol_id);

  loom_testbench_value_materializer_options_t materializer =
      MaterializerOptions();
  loom_testbench_scenario_configuration_values_t configuration = {};
  IREE_ASSERT_OK(loom_testbench_scenario_configuration_values_initialize(
      module, &scenario, host_allocator_, &configuration));
  const loom_testbench_entropy_t entropy_root =
      loom_testbench_entropy_root(0x3141592653589793ull);
  IREE_ASSERT_OK(loom_testbench_scenario_configuration_values_materialize(
      &materializer, entropy_root, /*configuration_ordinal=*/1,
      &configuration));

  ExecutionTimeline timeline = {};
  TestProfileState target_state = {};
  target_state.timeline = &timeline;
  TestProfileState oracle_state = {};
  oracle_state.timeline = &timeline;
  loom_testbench_scenario_execution_options_t execution_options = {};
  loom_testbench_scenario_execution_options_initialize(&execution_options);
  execution_options.target.name = IREE_SV("test-target");
  execution_options.target.prepare = PrepareTestProduct;
  execution_options.target.user_data = &target_state;
  execution_options.oracle.name = IREE_SV("test-oracle");
  execution_options.oracle.prepare = PrepareTestProduct;
  execution_options.oracle.user_data = &oracle_state;

  loom_testbench_prepared_scenario_configuration_t prepared = {};
  IREE_ASSERT_OK(loom_testbench_prepare_scenario_configuration(
      &execution_options, &configuration,
      LOOM_TESTBENCH_SCENARIO_EXECUTION_MODE_CORRECTNESS, &prepared));
  EXPECT_EQ(target_state.prepare_count, 2u);
  EXPECT_EQ(oracle_state.prepare_count, 1u);
  EXPECT_EQ(target_state.execute_count, 0u);
  EXPECT_EQ(oracle_state.execute_count, 0u);
  EXPECT_FALSE(target_state.saw_runtime_value_during_prepare);
  EXPECT_FALSE(oracle_state.saw_runtime_value_during_prepare);

  loom_testbench_scenario_trial_executor_t compare_executor = {};
  IREE_ASSERT_OK(loom_testbench_scenario_trial_executor_initialize(
      &prepared, /*trial_index=*/0, &materializer, /*batch_capacity=*/3,
      &compare_executor));
  loom_testbench_scenario_trial_result_list_t compare_results = {};
  IREE_ASSERT_OK(loom_testbench_run_scenario_trial_batch(
      &compare_executor, /*first_trial_ordinal=*/0, /*trial_count=*/3,
      &compare_results));
  ASSERT_EQ(compare_results.count, 3u);
  EXPECT_EQ(target_state.execute_count, 1u);
  EXPECT_EQ(target_state.last_call_count, 3u);
  EXPECT_EQ(oracle_state.execute_count, 1u);
  EXPECT_EQ(oracle_state.last_call_count, 3u);
  EXPECT_GT(timeline.first_execute_tick, timeline.last_prepare_tick);
  EXPECT_FALSE(target_state.saw_incomplete_alias);
  EXPECT_FALSE(oracle_state.saw_incomplete_alias);
  for (iree_host_size_t i = 0; i < compare_results.count; ++i) {
    EXPECT_TRUE(compare_results.values[i].passed);
    EXPECT_EQ(compare_results.values[i].identity.entropy_root.low,
              entropy_root.low);
    EXPECT_EQ(compare_results.values[i].identity.entropy_root.high,
              entropy_root.high);
    EXPECT_EQ(compare_results.values[i].identity.configuration_ordinal, 1u);
    EXPECT_EQ(compare_results.values[i].identity.trial_index, 0u);
    EXPECT_EQ(compare_results.values[i].identity.trial_ordinal, i);
    ASSERT_NE(compare_results.values[i].expectation_report, nullptr);
    EXPECT_EQ(compare_results.values[i].expectation_report->failure_count, 0u);

    const loom_testbench_invocation_plan_t& invocation =
        scenario.trials[0].action.target;
    const loom_testbench_value_t* target_storage =
        Lookup(&compare_executor.trial_values[i].target,
               invocation.input_value_ids[0]);
    const loom_testbench_value_t* oracle_storage =
        Lookup(&compare_executor.trial_values[i].oracle,
               invocation.input_value_ids[0]);
    EXPECT_EQ(iree_hal_buffer_test_overlap(
                  target_storage->buffer.buffer, 0, IREE_HAL_WHOLE_BUFFER,
                  oracle_storage->buffer.buffer, 0, IREE_HAL_WHOLE_BUFFER),
              IREE_HAL_BUFFER_OVERLAP_DISJOINT);
  }

  loom_testbench_scenario_trial_executor_t invoke_executor = {};
  IREE_ASSERT_OK(loom_testbench_scenario_trial_executor_initialize(
      &prepared, /*trial_index=*/1, &materializer, /*batch_capacity=*/2,
      &invoke_executor));
  loom_testbench_scenario_trial_result_list_t invoke_results = {};
  IREE_ASSERT_OK(loom_testbench_run_scenario_trial_batch(
      &invoke_executor, /*first_trial_ordinal=*/0, /*trial_count=*/2,
      &invoke_results));
  ASSERT_EQ(invoke_results.count, 2u);
  EXPECT_EQ(target_state.execute_count, 2u);
  EXPECT_EQ(target_state.last_call_count, 2u);
  EXPECT_EQ(oracle_state.execute_count, 1u);
  for (iree_host_size_t i = 0; i < invoke_results.count; ++i) {
    EXPECT_TRUE(invoke_results.values[i].passed);
    EXPECT_EQ(invoke_results.values[i].identity.trial_index, 1u);
    EXPECT_EQ(invoke_results.values[i].identity.trial_ordinal, i);
    EXPECT_EQ(invoke_results.values[i].expectation_report, nullptr);
  }

  loom_testbench_scenario_trial_executor_deinitialize(&invoke_executor);
  loom_testbench_scenario_trial_executor_deinitialize(&compare_executor);
  loom_testbench_prepared_scenario_configuration_deinitialize(&prepared);
  loom_testbench_scenario_configuration_values_deinitialize(&configuration);
  loom_module_free(module);
}

TEST_F(ScenarioExecutorTest, BenchmarksIndependentTargetOnlyReplicas) {
  loom_module_t* module = ParseModule();
  ASSERT_NE(module, nullptr);
  loom_testbench_module_plan_t plan = PlanModule(module);
  ASSERT_EQ(plan.issue_count, 0u);
  ASSERT_EQ(plan.scenario_count, 1u);
  const loom_testbench_scenario_plan_t& scenario = plan.scenarios[0];
  EXPECT_NE(scenario.trials[0].action.target.callee_ref.symbol_id,
            scenario.trials[0].action.oracle.callee_ref.symbol_id);

  loom_testbench_value_materializer_options_t materializer =
      MaterializerOptions();
  loom_testbench_scenario_configuration_values_t configuration = {};
  IREE_ASSERT_OK(loom_testbench_scenario_configuration_values_initialize(
      module, &scenario, host_allocator_, &configuration));
  const loom_testbench_entropy_t entropy_root =
      loom_testbench_entropy_root(0x0123456789abcdefull);
  IREE_ASSERT_OK(loom_testbench_scenario_configuration_values_materialize(
      &materializer, entropy_root, /*configuration_ordinal=*/1,
      &configuration));

  ExecutionTimeline timeline = {};
  TestProfileState target_state = {};
  target_state.timeline = &timeline;
  TestProfileState oracle_state = {};
  oracle_state.timeline = &timeline;
  loom_testbench_scenario_execution_options_t execution_options = {};
  loom_testbench_scenario_execution_options_initialize(&execution_options);
  execution_options.target.name = IREE_SV("test-target");
  execution_options.target.prepare = PrepareTestProduct;
  execution_options.target.user_data = &target_state;
  execution_options.oracle.name = IREE_SV("test-oracle");
  execution_options.oracle.prepare = PrepareTestProduct;
  execution_options.oracle.user_data = &oracle_state;

  loom_testbench_prepared_scenario_configuration_t prepared = {};
  IREE_ASSERT_OK(loom_testbench_prepare_scenario_configuration(
      &execution_options, &configuration,
      LOOM_TESTBENCH_SCENARIO_EXECUTION_MODE_BENCHMARK, &prepared));
  EXPECT_EQ(prepared.mode, LOOM_TESTBENCH_SCENARIO_EXECUTION_MODE_BENCHMARK);
  EXPECT_EQ(target_state.prepare_count, 2u);
  EXPECT_EQ(oracle_state.prepare_count, 0u);
  EXPECT_EQ(target_state.execute_count, 0u);
  EXPECT_EQ(oracle_state.execute_count, 0u);

  loom_testbench_scenario_trial_executor_t executor = {};
  IREE_ASSERT_OK(loom_testbench_scenario_trial_executor_initialize(
      &prepared, /*trial_index=*/0, &materializer, /*batch_capacity=*/3,
      &executor));
  EXPECT_EQ(executor.results, nullptr);
  EXPECT_EQ(executor.expectation_reports, nullptr);
  EXPECT_EQ(executor.oracle_calls, nullptr);

  loom_run_benchmark_options_t benchmark_options = {};
  loom_run_benchmark_options_initialize(&benchmark_options);
  benchmark_options.batch_size = 3;
  loom_run_benchmark_result_t benchmark_result = {};
  IREE_ASSERT_OK(loom_testbench_benchmark_scenario_trial(
      &executor, /*trial_ordinal=*/2, &benchmark_options, &benchmark_result));
  EXPECT_EQ(benchmark_result.batch_size, 3u);
  EXPECT_EQ(target_state.benchmark_count, 1u);
  EXPECT_EQ(target_state.last_benchmark_call_count, 3u);
  EXPECT_EQ(target_state.execute_count, 0u);
  EXPECT_EQ(oracle_state.benchmark_count, 0u);
  EXPECT_EQ(oracle_state.execute_count, 0u);
  EXPECT_FALSE(target_state.saw_incomplete_alias);
  EXPECT_FALSE(target_state.saw_shared_replica_storage);

  const loom_testbench_invocation_plan_t& invocation =
      scenario.trials[0].action.target;
  const loom_testbench_value_t* previous_storage = nullptr;
  for (iree_host_size_t i = 0; i < benchmark_options.batch_size; ++i) {
    const loom_testbench_scenario_trial_values_t& values =
        executor.trial_values[i];
    EXPECT_FALSE(iree_any_bit_set(
        values.flags, LOOM_TESTBENCH_SCENARIO_VALUE_FLAG_HAS_ORACLE));
    EXPECT_EQ(values.oracle.slot_count, 0u);
    EXPECT_EQ(values.identity.entropy_root.low, entropy_root.low);
    EXPECT_EQ(values.identity.entropy_root.high, entropy_root.high);
    EXPECT_EQ(values.identity.configuration_ordinal, 1u);
    EXPECT_EQ(values.identity.trial_index, 0u);
    EXPECT_EQ(values.identity.trial_ordinal, 2u);

    const loom_testbench_value_t* storage =
        Lookup(&values.target, invocation.input_value_ids[0]);
    const loom_testbench_value_t* tail =
        Lookup(&values.target, invocation.input_value_ids[1]);
    EXPECT_EQ(iree_hal_buffer_test_overlap(
                  storage->buffer.buffer, 0, IREE_HAL_WHOLE_BUFFER,
                  tail->buffer.buffer, 0, IREE_HAL_WHOLE_BUFFER),
              IREE_HAL_BUFFER_OVERLAP_PARTIAL);
    if (previous_storage != nullptr) {
      EXPECT_EQ(iree_hal_buffer_test_overlap(
                    previous_storage->buffer.buffer, 0, IREE_HAL_WHOLE_BUFFER,
                    storage->buffer.buffer, 0, IREE_HAL_WHOLE_BUFFER),
                IREE_HAL_BUFFER_OVERLAP_DISJOINT);
    }
    previous_storage = storage;
  }

  loom_testbench_scenario_trial_executor_deinitialize(&executor);
  loom_testbench_prepared_scenario_configuration_deinitialize(&prepared);
  loom_testbench_scenario_configuration_values_deinitialize(&configuration);
  loom_module_free(module);
}

TEST_F(ScenarioExecutorTest, ReportsAuthoredExpectationAndReplayIdentity) {
  loom_module_t* module = ParseModule();
  ASSERT_NE(module, nullptr);
  loom_testbench_module_plan_t plan = PlanModule(module);
  ASSERT_EQ(plan.issue_count, 0u);
  const loom_testbench_scenario_plan_t& scenario = plan.scenarios[0];

  loom_testbench_value_materializer_options_t materializer =
      MaterializerOptions();
  loom_testbench_scenario_configuration_values_t configuration = {};
  IREE_ASSERT_OK(loom_testbench_scenario_configuration_values_initialize(
      module, &scenario, host_allocator_, &configuration));
  const loom_testbench_entropy_t entropy_root =
      loom_testbench_entropy_root(0x2718281828459045ull);
  IREE_ASSERT_OK(loom_testbench_scenario_configuration_values_materialize(
      &materializer, entropy_root, /*configuration_ordinal=*/0,
      &configuration));

  ExecutionTimeline timeline = {};
  TestProfileState target_state = {};
  target_state.timeline = &timeline;
  TestProfileState oracle_state = {};
  oracle_state.timeline = &timeline;
  oracle_state.mismatch_trial_ordinal = 1;
  oracle_state.inject_mismatch = true;
  loom_testbench_scenario_execution_options_t execution_options = {};
  loom_testbench_scenario_execution_options_initialize(&execution_options);
  execution_options.target.name = IREE_SV("test-target");
  execution_options.target.prepare = PrepareTestProduct;
  execution_options.target.user_data = &target_state;
  execution_options.oracle.name = IREE_SV("test-oracle");
  execution_options.oracle.prepare = PrepareTestProduct;
  execution_options.oracle.user_data = &oracle_state;
  loom_testbench_prepared_scenario_configuration_t prepared = {};
  IREE_ASSERT_OK(loom_testbench_prepare_scenario_configuration(
      &execution_options, &configuration,
      LOOM_TESTBENCH_SCENARIO_EXECUTION_MODE_CORRECTNESS, &prepared));
  loom_testbench_scenario_trial_executor_t executor = {};
  IREE_ASSERT_OK(loom_testbench_scenario_trial_executor_initialize(
      &prepared, /*trial_index=*/0, &materializer, /*batch_capacity=*/3,
      &executor));

  loom_testbench_scenario_trial_result_list_t results = {};
  IREE_ASSERT_OK(loom_testbench_run_scenario_trial_batch(
      &executor, /*first_trial_ordinal=*/0, /*trial_count=*/3, &results));
  ASSERT_EQ(results.count, 3u);
  EXPECT_TRUE(results.values[0].passed);
  EXPECT_FALSE(results.values[1].passed);
  EXPECT_TRUE(results.values[2].passed);
  EXPECT_EQ(results.values[1].identity.entropy_root.low, entropy_root.low);
  EXPECT_EQ(results.values[1].identity.entropy_root.high, entropy_root.high);
  EXPECT_EQ(results.values[1].identity.configuration_ordinal, 0u);
  EXPECT_EQ(results.values[1].identity.trial_index, 0u);
  EXPECT_EQ(results.values[1].identity.trial_ordinal, 1u);
  const loom_testbench_expectation_report_t* report =
      results.values[1].expectation_report;
  ASSERT_NE(report, nullptr);
  ASSERT_EQ(report->failure_count, 1u);
  EXPECT_EQ(report->failures[0].expectation,
            &scenario.trials[0].action.expectations[0]);
  EXPECT_EQ(report->failures[0].expectation->op->location,
            scenario.trials[0].action.expectations[0].op->location);

  loom_testbench_scenario_trial_executor_deinitialize(&executor);
  loom_testbench_prepared_scenario_configuration_deinitialize(&prepared);
  loom_testbench_scenario_configuration_values_deinitialize(&configuration);
  loom_module_free(module);
}

TEST_F(ScenarioExecutorTest, AttributesUnexpectedDeviceErrorsToExactTrial) {
  loom_module_t* module = ParseDeviceEventModule();
  ASSERT_NE(module, nullptr);
  loom_testbench_module_plan_t plan = PlanModule(module);
  ASSERT_EQ(plan.issue_count, 0u);
  ASSERT_EQ(plan.scenario_count, 2u);
  const loom_testbench_scenario_plan_t& scenario = plan.scenarios[0];
  EXPECT_FALSE(scenario.trials[0].action.expects_device_events);

  loom_testbench_value_materializer_options_t materializer =
      MaterializerOptions();
  {
    DeviceEventScenarioRun run;
    IREE_ASSERT_OK(InitializeDeviceEventScenarioRun(
        module, &scenario, &materializer, /*device_event_trial_ordinal=*/1,
        /*device_event_changes_in_isolation=*/false, host_allocator_, &run));

    ASSERT_EQ(run.results.count, 3u);
    EXPECT_EQ(run.target_state.execute_count, 4u);
    EXPECT_EQ(run.target_state.last_call_count, 1u);
    EXPECT_EQ(run.oracle_state.execute_count, 1u);
    EXPECT_EQ(run.oracle_state.last_call_count, 3u);
    EXPECT_TRUE(run.results.values[0].passed);
    EXPECT_FALSE(run.results.values[1].passed);
    EXPECT_TRUE(run.results.values[2].passed);

    const loom_testbench_scenario_trial_result_t& failure =
        run.results.values[1];
    EXPECT_EQ(failure.identity.trial_ordinal, 1u);
    ASSERT_NE(failure.device_events, nullptr);
    ASSERT_EQ(failure.device_events->count, 1u);
    EXPECT_EQ(failure.device_events->dropped_count, 0u);
    EXPECT_EQ(failure.unhandled_device_event_count, 1u);
    EXPECT_EQ(failure.expected_device_events, nullptr);
    EXPECT_EQ(run.results.values[0].device_events->count, 0u);
    EXPECT_EQ(run.results.values[2].device_events->count, 0u);

    const std::string json = WriteResultJson(failure);
    EXPECT_THAT(json, ::testing::HasSubstr("\"trial_ordinal\":1"));
    EXPECT_THAT(json, ::testing::HasSubstr("\"device_events\":"));
    EXPECT_THAT(json, ::testing::HasSubstr("\"type\":\"asan_report\""));
    EXPECT_THAT(json, ::testing::HasSubstr("\"driver\":\"test\""));
    EXPECT_THAT(json, ::testing::HasSubstr("\"device\":\"scenario\""));
    EXPECT_THAT(json,
                ::testing::HasSubstr("\"source_file\":\"event_subject.loom\""));
    EXPECT_THAT(json, ::testing::HasSubstr("\"source_location\":"));
    EXPECT_THAT(json, ::testing::HasSubstr(
                          "\"filename\":\"scenario_device_events.loom\""));
  }
  loom_module_free(module);
}

TEST_F(ScenarioExecutorTest, ConsumesExpectedDeviceErrorsPerTrial) {
  loom_module_t* module = ParseDeviceEventModule();
  ASSERT_NE(module, nullptr);
  loom_testbench_module_plan_t plan = PlanModule(module);
  ASSERT_EQ(plan.issue_count, 0u);
  ASSERT_EQ(plan.scenario_count, 2u);
  const loom_testbench_scenario_plan_t& scenario = plan.scenarios[1];
  EXPECT_TRUE(scenario.trials[0].action.expects_device_events);

  loom_testbench_value_materializer_options_t materializer =
      MaterializerOptions();
  {
    DeviceEventScenarioRun run;
    IREE_ASSERT_OK(InitializeDeviceEventScenarioRun(
        module, &scenario, &materializer, /*device_event_trial_ordinal=*/0,
        /*device_event_changes_in_isolation=*/false, host_allocator_, &run));

    ASSERT_EQ(run.results.count, 2u);
    EXPECT_EQ(run.target_state.execute_count, 2u);
    EXPECT_EQ(run.target_state.last_call_count, 1u);
    EXPECT_EQ(run.oracle_state.execute_count, 1u);
    EXPECT_EQ(run.oracle_state.last_call_count, 2u);

    const loom_testbench_scenario_trial_result_t& expected =
        run.results.values[0];
    EXPECT_TRUE(expected.passed);
    ASSERT_NE(expected.expectation_report, nullptr);
    EXPECT_EQ(expected.expectation_report->expectation_count, 2u);
    EXPECT_EQ(expected.expectation_report->passed_count, 2u);
    EXPECT_EQ(expected.expectation_report->failure_count, 0u);
    ASSERT_NE(expected.device_events, nullptr);
    ASSERT_EQ(expected.device_events->count, 1u);
    ASSERT_NE(expected.expected_device_events, nullptr);
    EXPECT_EQ(expected.expected_device_events[0], 1u);
    EXPECT_EQ(expected.unhandled_device_event_count, 0u);
    EXPECT_THAT(WriteResultJson(expected),
                ::testing::Not(::testing::HasSubstr("\"device_events\":")));

    const loom_testbench_scenario_trial_result_t& absent =
        run.results.values[1];
    EXPECT_FALSE(absent.passed);
    ASSERT_NE(absent.expectation_report, nullptr);
    EXPECT_EQ(absent.expectation_report->expectation_count, 2u);
    EXPECT_EQ(absent.expectation_report->passed_count, 1u);
    ASSERT_EQ(absent.expectation_report->failure_count, 1u);
    EXPECT_EQ(absent.expectation_report->failures[0].kind,
              LOOM_TESTBENCH_EXPECTATION_EVENT);
    EXPECT_EQ(absent.expectation_report->failures[0].expectation,
              &scenario.trials[0].action.expectations[1]);
    EXPECT_EQ(absent.expectation_report->failures[0].expectation->op->location,
              scenario.trials[0].action.expectations[1].op->location);
    const std::string json = WriteResultJson(absent);
    EXPECT_THAT(json, ::testing::HasSubstr("\"kind\":\"event\""));
    EXPECT_THAT(json, ::testing::HasSubstr("\"source_location\":"));
  }
  loom_module_free(module);
}

TEST_F(ScenarioExecutorTest, RejectsChangedBatchEventAttribution) {
  loom_module_t* module = ParseDeviceEventModule();
  ASSERT_NE(module, nullptr);
  loom_testbench_module_plan_t plan = PlanModule(module);
  ASSERT_EQ(plan.issue_count, 0u);
  ASSERT_EQ(plan.scenario_count, 2u);
  const loom_testbench_scenario_plan_t& scenario = plan.scenarios[0];

  loom_testbench_value_materializer_options_t materializer =
      MaterializerOptions();
  {
    DeviceEventScenarioRun run;
    iree::Status status(InitializeDeviceEventScenarioRun(
        module, &scenario, &materializer, /*device_event_trial_ordinal=*/1,
        /*device_event_changes_in_isolation=*/true, host_allocator_, &run));
    EXPECT_THAT(status,
                iree::testing::status::StatusIs(iree::StatusCode::kAborted));
    EXPECT_THAT(status.ToString(),
                ::testing::HasSubstr("refusing to guess event ownership"));
    EXPECT_THAT(status.ToString(), ::testing::HasSubstr("range [0, 3)"));
    EXPECT_THAT(status.ToString(),
                ::testing::HasSubstr("batch 1 captured/0 dropped, replay 1 "
                                     "captured/0 dropped"));
    EXPECT_EQ(run.target_state.execute_count, 4u);
    EXPECT_EQ(run.target_state.last_call_count, 1u);
    EXPECT_EQ(run.oracle_state.execute_count, 0u);
  }
  loom_module_free(module);
}

}  // namespace
}  // namespace loom
