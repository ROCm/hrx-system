// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/testbench/scenario/executor.h"

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"
#include "iree/hal/memory/heap_pool.h"
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
  iree_host_size_t fail_prepare_ordinal;
  iree_host_size_t destroy_count;
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

static void DestroyTestProduct(void* user_data) {
  TestProfileState* state = static_cast<TestProfileState*>(user_data);
  ++state->destroy_count;
}

static const loom_testbench_product_vtable_t kTestProductVtable = {
    .execute = ExecuteTestProduct,
    .benchmark = BenchmarkTestProduct,
};

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
  if (state->prepare_count == state->fail_prepare_ordinal) {
    return iree_make_status(IREE_STATUS_ABORTED,
                            "injected scenario product preparation failure");
  }
  *out_product = {};
  out_product->vtable = &kTestProductVtable;
  out_product->destroy = DestroyTestProduct;
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
      &execution_options, &out_run->configuration, &out_run->prepared));
  IREE_RETURN_IF_ERROR(loom_testbench_scenario_trial_executor_initialize(
      &out_run->prepared, /*trial_index=*/0,
      LOOM_TESTBENCH_SCENARIO_EXECUTION_MODE_CORRECTNESS, materializer,
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
    const iree_hal_pool_host_access_t host_access = {
        .access = IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
        .modes =
            IREE_HAL_MAPPING_MODE_SCOPED | IREE_HAL_MAPPING_MODE_PERSISTENT,
    };
    IREE_ASSERT_OK(iree_hal_heap_pool_create(host_access, host_allocator_,
                                             host_allocator_, &pool_));
  }

  void TearDown() override {
    iree_hal_pool_release(pool_);
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
  check.trial[1](%trial: index, %entropy: check.entropy) {
    %storage = check.generate.iota offset(7) step(1) : tensor<4xi32>
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
    loom_text_parse_options_t options = {.max_errors = 20};
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
    loom_text_parse_options_t options = {.max_errors = 20};
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
    options.target_pool = pool_;
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
  iree_hal_pool_t* pool_ = nullptr;
};

// Records the external profile's ownership callbacks; preparation and reuse
// themselves run through the production executor.
struct PreparationTrace {
  struct Product {
    // Profile trace that outlives the prepared configuration.
    PreparationTrace* trace;
    // Invocation supplied to the owning preparation callback.
    const loom_testbench_invocation_plan_t* invocation;
  };

  explicit PreparationTrace(size_t capacity) : products(capacity) {}

  static iree_status_t Execute(void*, const loom_testbench_invocation_plan_t*,
                               iree_host_size_t,
                               loom_testbench_product_call_t*) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "preparation-only test profile");
  }

  static void Destroy(void* user_data) {
    auto* product = static_cast<Product*>(user_data);
    product->trace->destroyed.push_back(product->invocation);
  }

  static iree_status_t Prepare(
      void* user_data, const loom_testbench_invocation_plan_t* invocation,
      const loom_testbench_value_table_t*, iree_allocator_t,
      loom_testbench_prepared_product_t* out_product) {
    auto* trace = static_cast<PreparationTrace*>(user_data);
    Product* product = &trace->products[trace->prepared.size()];
    *product = {.trace = trace, .invocation = invocation};
    trace->prepared.push_back(invocation);
    static const loom_testbench_product_vtable_t vtable = {.execute = Execute};
    *out_product = {
        .vtable = &vtable,
        .destroy = Destroy,
        .user_data = product,
    };
    // Return a partially created product to exercise cleanup of the failing
    // callback as well as all previously prepared owners.
    if (trace->prepared.size() == trace->fail_at) {
      return iree_make_status(IREE_STATUS_ABORTED,
                              "injected product preparation failure");
    }
    return iree_ok_status();
  }

  // Stable product storage owned by the profile, not by a borrowed trial.
  std::vector<Product> products;
  // Preparation callback order, including a failing callback.
  std::vector<const loom_testbench_invocation_plan_t*> prepared;
  // Destruction callback order.
  std::vector<const loom_testbench_invocation_plan_t*> destroyed;
  // One-based failing preparation ordinal, or zero for success.
  size_t fail_at = 0;
};

TEST_F(ScenarioExecutorTest,
       IndexesCompleteProductKeysWithoutChangingOwnership) {
  loom_module_t* first_module = nullptr;
  loom_module_t* second_module = nullptr;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("first"), &block_pool_,
                                      nullptr, host_allocator_, &first_module));
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("second"),
                                      &block_pool_, nullptr, host_allocator_,
                                      &second_module));
  const loom_value_id_t value_ids[][2] = {{0, 1}, {2, 3}};
  std::array<loom_testbench_invocation_plan_t, 128> keys = {};
  for (size_t i = 0; i < keys.size(); ++i) {
    keys[i] = {
        .kind = LOOM_TESTBENCH_INVOCATION_FUNCTION_CALL,
        .module = first_module,
        .callee_ref = {.module_id = 0, .symbol_id = 1},
        .workload_value_ids = value_ids[0],
        .input_value_ids = value_ids[0],
        .input_count = 1,
        .result_value_ids = value_ids[0],
        .result_count = 1,
    };
    if (i >= 8) {
      keys[i].callee_ref.symbol_id = static_cast<uint16_t>(i + 1);
    }
  }
  keys[1].module = second_module;
  keys[2].kind = LOOM_TESTBENCH_INVOCATION_KERNEL_LAUNCH;
  keys[3].callee_ref.symbol_id = 3;
  keys[4].callee_ref.symbol_id = 2;
  keys[5].workload_count = 1;
  keys[6].input_count = 2;
  keys[7].result_count = 2;

  for (size_t unique_count : {1u, 8u, 16u, 128u}) {
    std::vector<loom_testbench_trial_plan_t> trials(unique_count * 3);
    for (size_t i = 0; i < trials.size(); ++i) {
      auto& trial = trials[i];
      trial.trial_count = 1;
      trial.action.kind = i % 3 == 0 ? LOOM_TESTBENCH_SCENARIO_ACTION_INVOKE
                                     : LOOM_TESTBENCH_SCENARIO_ACTION_COMPARE;
      trial.action.target = keys[unique_count - 1 - i % unique_count];
      // Trial-local value identities do not enter the preparation key.
      trial.action.target.input_value_ids = value_ids[i % 2];
      trial.action.target.result_value_ids = value_ids[(i + 1) % 2];
      trial.action.oracle = trial.action.target;
    }
    const loom_testbench_scenario_plan_t scenario = {
        .name = IREE_SV("product_owners"),
        .configuration_count = 2,
        .trials = trials.data(),
        .trial_count = trials.size(),
    };
    loom_testbench_scenario_configuration_values_t configuration = {
        .scenario_plan = &scenario,
        .flags = LOOM_TESTBENCH_SCENARIO_VALUE_FLAG_MATERIALIZED,
    };
    for (bool prepare_oracles : {false, true}) {
      for (size_t configuration_ordinal = 0; configuration_ordinal < 2;
           ++configuration_ordinal) {
        SCOPED_TRACE(::testing::Message()
                     << "unique=" << unique_count
                     << " oracles=" << prepare_oracles
                     << " configuration=" << configuration_ordinal);
        configuration.configuration_ordinal = configuration_ordinal;
        PreparationTrace trace(trials.size() * 2);
        loom_testbench_scenario_execution_options_t options = {};
        loom_testbench_scenario_execution_options_initialize(&options);
        options.prepare_oracles = prepare_oracles;
        options.target = {IREE_SV("target"), PreparationTrace::Prepare, &trace};
        options.oracle = {IREE_SV("oracle"), PreparationTrace::Prepare, &trace};
        loom_testbench_prepared_scenario_configuration_t prepared = {};
        IREE_ASSERT_OK(loom_testbench_prepare_scenario_configuration(
            &options, &configuration, &prepared));

        std::array<size_t, 128> target_owners;
        std::array<size_t, 128> oracle_owners;
        target_owners.fill(IREE_HOST_SIZE_MAX);
        oracle_owners.fill(IREE_HOST_SIZE_MAX);
        std::vector<const loom_testbench_invocation_plan_t*> expected;
        for (size_t i = 0; i < trials.size(); ++i) {
          const size_t key = i % unique_count;
          if (target_owners[key] == IREE_HOST_SIZE_MAX) {
            target_owners[key] = i;
            expected.push_back(&trials[i].action.target);
          }
          const auto& target = prepared.trials[i].target;
          EXPECT_EQ(target.invocation,
                    &trials[target_owners[key]].action.target);
          EXPECT_EQ(target.user_data,
                    prepared.trials[target_owners[key]].target.user_data);
          EXPECT_EQ(target.destroy != nullptr, target_owners[key] == i);
          if (prepare_oracles &&
              trials[i].action.kind == LOOM_TESTBENCH_SCENARIO_ACTION_COMPARE) {
            if (oracle_owners[key] == IREE_HOST_SIZE_MAX) {
              oracle_owners[key] = i;
              expected.push_back(&trials[i].action.oracle);
            }
            const auto& oracle = prepared.trials[i].oracle;
            EXPECT_EQ(oracle.invocation,
                      &trials[oracle_owners[key]].action.oracle);
            EXPECT_EQ(oracle.user_data,
                      prepared.trials[oracle_owners[key]].oracle.user_data);
            EXPECT_NE(oracle.user_data, target.user_data);
            EXPECT_EQ(oracle.destroy != nullptr, oracle_owners[key] == i);
          } else {
            EXPECT_EQ(prepared.trials[i].oracle.invocation, nullptr);
          }
        }
        EXPECT_EQ(trace.prepared, expected);
        EXPECT_TRUE(trace.destroyed.empty());
        loom_testbench_prepared_scenario_configuration_deinitialize(&prepared);
        std::reverse(expected.begin(), expected.end());
        EXPECT_EQ(trace.destroyed, expected);
      }
    }
  }
  loom_module_free(second_module);
  loom_module_free(first_module);
}

TEST_F(ScenarioExecutorTest, CleansFailedPreparationAndEarlierSharedOwners) {
  std::array<loom_testbench_trial_plan_t, 4> trials = {};
  for (size_t i = 0; i < trials.size(); ++i) {
    trials[i].trial_count = 1;
    trials[i].action.kind = LOOM_TESTBENCH_SCENARIO_ACTION_COMPARE;
    trials[i].action.target.kind = LOOM_TESTBENCH_INVOCATION_FUNCTION_CALL;
    trials[i].action.target.callee_ref.symbol_id = i < 2 ? 2 : 1;
    trials[i].action.oracle = trials[i].action.target;
  }
  const loom_testbench_scenario_plan_t scenario = {
      .name = IREE_SV("failed_product"),
      .configuration_count = 1,
      .trials = trials.data(),
      .trial_count = trials.size(),
  };
  const loom_testbench_scenario_configuration_values_t configuration = {
      .scenario_plan = &scenario,
      .flags = LOOM_TESTBENCH_SCENARIO_VALUE_FLAG_MATERIALIZED,
  };
  for (size_t fail_at = 1; fail_at <= 4; ++fail_at) {
    SCOPED_TRACE(fail_at);
    PreparationTrace trace(trials.size() * 2);
    trace.fail_at = fail_at;
    loom_testbench_scenario_execution_options_t options = {};
    loom_testbench_scenario_execution_options_initialize(&options);
    options.target = {IREE_SV("target"), PreparationTrace::Prepare, &trace};
    options.oracle = {IREE_SV("oracle"), PreparationTrace::Prepare, &trace};
    loom_testbench_prepared_scenario_configuration_t prepared = {};
    IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                          loom_testbench_prepare_scenario_configuration(
                              &options, &configuration, &prepared));
    EXPECT_EQ(prepared.trials, nullptr);
    EXPECT_EQ(prepared.trial_count, 0u);
    EXPECT_EQ(trace.prepared.size(), fail_at);
    auto expected_destroyed = trace.prepared;
    std::reverse(expected_destroyed.begin(), expected_destroyed.end());
    EXPECT_EQ(trace.destroyed, expected_destroyed);
  }
}

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
  TestProfileState target_state = {.timeline = &timeline};
  TestProfileState oracle_state = {.timeline = &timeline};
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
      &execution_options, &configuration, &prepared));
  EXPECT_EQ(target_state.prepare_count, 2u);
  EXPECT_EQ(oracle_state.prepare_count, 1u);
  EXPECT_EQ(prepared.trials[1].target.invocation,
            prepared.trials[0].target.invocation);
  EXPECT_EQ(prepared.trials[1].oracle.invocation,
            prepared.trials[0].oracle.invocation);
  EXPECT_EQ(prepared.trials[1].target.destroy, nullptr);
  EXPECT_EQ(prepared.trials[1].oracle.destroy, nullptr);
  EXPECT_EQ(target_state.execute_count, 0u);
  EXPECT_EQ(oracle_state.execute_count, 0u);
  EXPECT_FALSE(target_state.saw_runtime_value_during_prepare);
  EXPECT_FALSE(oracle_state.saw_runtime_value_during_prepare);

  loom_testbench_scenario_trial_executor_t compare_executor = {};
  IREE_ASSERT_OK(loom_testbench_scenario_trial_executor_initialize(
      &prepared, /*trial_index=*/0,
      LOOM_TESTBENCH_SCENARIO_EXECUTION_MODE_CORRECTNESS, &materializer,
      /*batch_capacity=*/3, &compare_executor));
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
      &prepared, /*trial_index=*/2,
      LOOM_TESTBENCH_SCENARIO_EXECUTION_MODE_CORRECTNESS, &materializer,
      /*batch_capacity=*/2, &invoke_executor));
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
    EXPECT_EQ(invoke_results.values[i].identity.trial_index, 2u);
    EXPECT_EQ(invoke_results.values[i].identity.trial_ordinal, i);
    EXPECT_EQ(invoke_results.values[i].expectation_report, nullptr);
  }

  loom_testbench_scenario_trial_executor_t repeated_executor = {};
  IREE_ASSERT_OK(loom_testbench_scenario_trial_executor_initialize(
      &prepared, /*trial_index=*/1,
      LOOM_TESTBENCH_SCENARIO_EXECUTION_MODE_CORRECTNESS, &materializer,
      /*batch_capacity=*/1, &repeated_executor));
  loom_testbench_scenario_trial_result_list_t repeated_results = {};
  IREE_ASSERT_OK(loom_testbench_run_scenario_trial_batch(
      &repeated_executor, /*first_trial_ordinal=*/0, /*trial_count=*/1,
      &repeated_results));
  ASSERT_EQ(repeated_results.count, 1u);
  EXPECT_TRUE(repeated_results.values[0].passed);
  EXPECT_EQ(repeated_results.values[0].identity.trial_index, 1u);
  EXPECT_EQ(target_state.execute_count, 3u);
  EXPECT_EQ(oracle_state.execute_count, 2u);

  loom_testbench_scenario_trial_executor_deinitialize(&repeated_executor);
  loom_testbench_scenario_trial_executor_deinitialize(&invoke_executor);
  loom_testbench_scenario_trial_executor_deinitialize(&compare_executor);
  loom_testbench_prepared_scenario_configuration_deinitialize(&prepared);
  EXPECT_EQ(target_state.destroy_count, 2u);
  EXPECT_EQ(oracle_state.destroy_count, 1u);
  loom_testbench_scenario_configuration_values_deinitialize(&configuration);
  loom_module_free(module);
}

TEST_F(ScenarioExecutorTest, ReleasesOwnedProductsAfterPreparationFailure) {
  loom_module_t* module = ParseModule();
  ASSERT_NE(module, nullptr);
  loom_testbench_module_plan_t plan = PlanModule(module);
  ASSERT_EQ(plan.issue_count, 0u);
  ASSERT_EQ(plan.scenario_count, 1u);
  const loom_testbench_scenario_plan_t& scenario = plan.scenarios[0];

  loom_testbench_value_materializer_options_t materializer =
      MaterializerOptions();
  loom_testbench_scenario_configuration_values_t configuration = {};
  IREE_ASSERT_OK(loom_testbench_scenario_configuration_values_initialize(
      module, &scenario, host_allocator_, &configuration));
  IREE_ASSERT_OK(loom_testbench_scenario_configuration_values_materialize(
      &materializer, loom_testbench_entropy_root(0),
      /*configuration_ordinal=*/0, &configuration));

  ExecutionTimeline timeline = {};
  TestProfileState target_state = {
      .timeline = &timeline,
      .fail_prepare_ordinal = 2,
  };
  TestProfileState oracle_state = {.timeline = &timeline};
  loom_testbench_scenario_execution_options_t execution_options = {};
  loom_testbench_scenario_execution_options_initialize(&execution_options);
  execution_options.target.name = IREE_SV("test-target");
  execution_options.target.prepare = PrepareTestProduct;
  execution_options.target.user_data = &target_state;
  execution_options.oracle.name = IREE_SV("test-oracle");
  execution_options.oracle.prepare = PrepareTestProduct;
  execution_options.oracle.user_data = &oracle_state;

  loom_testbench_prepared_scenario_configuration_t prepared = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        loom_testbench_prepare_scenario_configuration(
                            &execution_options, &configuration, &prepared));
  EXPECT_EQ(target_state.prepare_count, 2u);
  EXPECT_EQ(target_state.destroy_count, 1u);
  EXPECT_EQ(oracle_state.prepare_count, 1u);
  EXPECT_EQ(oracle_state.destroy_count, 1u);
  EXPECT_EQ(prepared.trials, nullptr);

  loom_testbench_scenario_configuration_values_deinitialize(&configuration);
  loom_module_free(module);
}

TEST_F(ScenarioExecutorTest, ReusesProductsAcrossCorrectnessAndBenchmarkModes) {
  // Only value identities and a typed invocation plan are needed to exercise
  // executor ownership. Compilation is the injected profile's responsibility.
  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("products"),
                                      &block_pool_, nullptr, host_allocator_,
                                      &module));
  loom_string_id_t entropy_name;
  IREE_ASSERT_OK(loom_module_intern_string(module, IREE_SV("check.entropy"),
                                           &entropy_name));
  loom_testbench_trial_plan_t trial = {.trial_count = 3};
  IREE_ASSERT_OK(
      loom_module_define_value(module, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
                               &trial.ordinal_value_id));
  IREE_ASSERT_OK(loom_module_define_value(
      module, loom_type_dialect_opaque(entropy_name), &trial.entropy_value_id));
  loom_value_id_t actual_result, expected_result;
  IREE_ASSERT_OK(loom_module_define_value(
      module, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &actual_result));
  IREE_ASSERT_OK(loom_module_define_value(
      module, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &expected_result));
  trial.action.kind = LOOM_TESTBENCH_SCENARIO_ACTION_COMPARE;
  trial.action.target.module = module;
  trial.action.target.kind = LOOM_TESTBENCH_INVOCATION_FUNCTION_CALL;
  trial.action.target.input_value_ids = &trial.ordinal_value_id;
  trial.action.target.input_count = 1;
  trial.action.target.result_value_ids = &actual_result;
  trial.action.target.result_count = 1;
  trial.action.oracle = trial.action.target;
  trial.action.oracle.result_value_ids = &expected_result;
  loom_testbench_scenario_plan_t scenario = {
      .name = IREE_SV("products"),
      .configuration_count = 1,
      .trials = &trial,
      .trial_count = 1,
  };
  loom_testbench_scenario_configuration_values_t configuration = {};
  IREE_ASSERT_OK(loom_testbench_scenario_configuration_values_initialize(
      module, &scenario, host_allocator_, &configuration));
  auto materializer = MaterializerOptions();
  IREE_ASSERT_OK(loom_testbench_scenario_configuration_values_materialize(
      &materializer, loom_testbench_entropy_root(17), 0, &configuration));
  ExecutionTimeline timeline = {};
  TestProfileState target_state = {.timeline = &timeline};
  TestProfileState oracle_state = {.timeline = &timeline};
  loom_testbench_scenario_execution_options_t options = {};
  loom_testbench_scenario_execution_options_initialize(&options);
  options.target = {IREE_SV("target"), PrepareTestProduct, &target_state};
  options.oracle = {IREE_SV("oracle"), PrepareTestProduct, &oracle_state};
  loom_testbench_prepared_scenario_configuration_t prepared = {};
  IREE_ASSERT_OK(loom_testbench_prepare_scenario_configuration(
      &options, &configuration, &prepared));

  loom_testbench_scenario_trial_executor_t checker = {};
  IREE_ASSERT_OK(loom_testbench_scenario_trial_executor_initialize(
      &prepared, 0, LOOM_TESTBENCH_SCENARIO_EXECUTION_MODE_CORRECTNESS,
      &materializer, 1, &checker));
  ASSERT_NE(checker.oracle_calls, nullptr);
  loom_testbench_scenario_trial_result_list_t results = {};
  IREE_ASSERT_OK(
      loom_testbench_run_scenario_trial_batch(&checker, 2, 1, &results));
  ASSERT_EQ(results.count, 1u);
  EXPECT_TRUE(results.values[0].passed);
  EXPECT_EQ(results.values[0].identity.trial_ordinal, 2u);
  EXPECT_EQ(target_state.execute_count, 1u);
  EXPECT_EQ(oracle_state.execute_count, 1u);
  loom_testbench_scenario_trial_executor_deinitialize(&checker);
  EXPECT_EQ(target_state.destroy_count, 0u);
  EXPECT_EQ(oracle_state.destroy_count, 0u);

  loom_testbench_scenario_trial_executor_t benchmark = {};
  IREE_ASSERT_OK(loom_testbench_scenario_trial_executor_initialize(
      &prepared, 0, LOOM_TESTBENCH_SCENARIO_EXECUTION_MODE_BENCHMARK,
      &materializer, 3, &benchmark));
  EXPECT_EQ(benchmark.oracle_calls, nullptr);
  EXPECT_EQ(benchmark.results, nullptr);
  EXPECT_EQ(benchmark.expectation_reports, nullptr);
  loom_run_benchmark_options_t benchmark_options = {};
  loom_run_benchmark_options_initialize(&benchmark_options);
  benchmark_options.batch_size = 3;
  loom_run_benchmark_result_t benchmark_result = {};
  IREE_ASSERT_OK(loom_testbench_benchmark_scenario_trial(
      &benchmark, 2, &benchmark_options, &benchmark_result));
  EXPECT_EQ(target_state.benchmark_count, 1u);
  EXPECT_EQ(target_state.last_benchmark_call_count, 3u);
  EXPECT_EQ(oracle_state.benchmark_count, 0u);
  EXPECT_EQ(target_state.execute_count, 1u);
  EXPECT_EQ(oracle_state.execute_count, 1u);
  EXPECT_EQ(target_state.prepare_count, 1u);
  EXPECT_EQ(oracle_state.prepare_count, 1u);
  for (iree_host_size_t i = 0; i < benchmark.batch_capacity; ++i) {
    EXPECT_EQ(benchmark.trial_values[i].oracle.slot_count, 0u);
    EXPECT_EQ(benchmark.trial_values[i].identity.trial_ordinal, 2u);
    EXPECT_EQ(Lookup(&benchmark.trial_values[i].target, trial.ordinal_value_id)
                  ->scalar.storage.i64,
              2);
  }
  loom_testbench_scenario_trial_executor_deinitialize(&benchmark);
  loom_testbench_prepared_scenario_configuration_deinitialize(&prepared);
  EXPECT_EQ(target_state.destroy_count, 1u);
  EXPECT_EQ(oracle_state.destroy_count, 1u);

  // An absent oracle is an explicit preparation failure, never a request to
  // compile the target again under its native profile.
  options.oracle = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        loom_testbench_prepare_scenario_configuration(
                            &options, &configuration, &prepared));
  EXPECT_EQ(prepared.trials, nullptr);
  EXPECT_EQ(target_state.destroy_count, 2u);
  EXPECT_EQ(oracle_state.prepare_count, 1u);

  // Target-only preparation neither requires nor calls an oracle profile.
  // Supplying a profile must not accidentally enable its compilation.
  options.prepare_oracles = false;
  for (bool bind_oracle : {false, true}) {
    SCOPED_TRACE(bind_oracle);
    if (bind_oracle) {
      options.oracle = {IREE_SV("oracle"), PrepareTestProduct, &oracle_state};
    }
    IREE_ASSERT_OK(loom_testbench_prepare_scenario_configuration(
        &options, &configuration, &prepared));
    EXPECT_EQ(prepared.trials[0].oracle.vtable, nullptr);
    IREE_ASSERT_OK(loom_testbench_scenario_trial_executor_initialize(
        &prepared, 0, LOOM_TESTBENCH_SCENARIO_EXECUTION_MODE_BENCHMARK,
        &materializer, 3, &benchmark));
    IREE_ASSERT_OK(loom_testbench_benchmark_scenario_trial(
        &benchmark, 2, &benchmark_options, &benchmark_result));
    EXPECT_EQ(benchmark.oracle_calls, nullptr);
    EXPECT_EQ(benchmark.results, nullptr);
    EXPECT_EQ(benchmark.expectation_reports, nullptr);
    for (iree_host_size_t i = 0; i < benchmark.batch_capacity; ++i) {
      EXPECT_EQ(benchmark.trial_values[i].oracle.slot_count, 0u);
    }
    loom_testbench_scenario_trial_executor_deinitialize(&benchmark);
    loom_testbench_prepared_scenario_configuration_deinitialize(&prepared);
    EXPECT_EQ(oracle_state.prepare_count, 1u);
    EXPECT_EQ(oracle_state.execute_count, 1u);
    EXPECT_EQ(oracle_state.destroy_count, 1u);
  }
  EXPECT_EQ(target_state.prepare_count, 4u);
  EXPECT_EQ(target_state.destroy_count, 4u);
  EXPECT_EQ(target_state.execute_count, 1u);
  EXPECT_EQ(target_state.benchmark_count, 3u);
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
  TestProfileState target_state = {.timeline = &timeline};
  TestProfileState oracle_state = {.timeline = &timeline};
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
      &execution_options, &configuration, &prepared));
  EXPECT_EQ(target_state.prepare_count, 2u);
  EXPECT_EQ(oracle_state.prepare_count, 1u);
  EXPECT_EQ(target_state.execute_count, 0u);
  EXPECT_EQ(oracle_state.execute_count, 0u);

  loom_testbench_scenario_trial_executor_t executor = {};
  IREE_ASSERT_OK(loom_testbench_scenario_trial_executor_initialize(
      &prepared, /*trial_index=*/0,
      LOOM_TESTBENCH_SCENARIO_EXECUTION_MODE_BENCHMARK, &materializer,
      /*batch_capacity=*/3, &executor));
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
  EXPECT_EQ(target_state.destroy_count, 2u);
  EXPECT_EQ(oracle_state.destroy_count, 1u);
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
  TestProfileState target_state = {.timeline = &timeline};
  TestProfileState oracle_state = {
      .timeline = &timeline,
      .mismatch_trial_ordinal = 1,
      .inject_mismatch = true,
  };
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
      &execution_options, &configuration, &prepared));
  loom_testbench_scenario_trial_executor_t executor = {};
  IREE_ASSERT_OK(loom_testbench_scenario_trial_executor_initialize(
      &prepared, /*trial_index=*/0,
      LOOM_TESTBENCH_SCENARIO_EXECUTION_MODE_CORRECTNESS, &materializer,
      /*batch_capacity=*/3, &executor));

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
              LOOM_CHECK_EXPECTATION_EVENT);
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
