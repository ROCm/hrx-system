// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/iree-benchmark-loom/snapshot.h"

#include <cstring>
#include <memory>

#include "iree/base/internal/json.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/tools/iree-benchmark-loom/hal_actual.h"
#include "loom/tools/iree-benchmark-loom/launch_evidence.h"
#include "loom/tools/iree-benchmark-loom/report.h"
#include "loomc/compile_report.h"
#include "loomc/iree.h"

namespace loom {
namespace {

static const loom_sanitizer_options_t kNoSanitizer = {};

using ByteSequencePtr = std::unique_ptr<loomc_byte_sequence_t,
                                        decltype(&loomc_byte_sequence_release)>;

static ByteSequencePtr CopyByteSequence(iree_string_view_t source) {
  loomc_byte_sequence_t* sequence = nullptr;
  IREE_EXPECT_OK(iree_status_from_loomc(loomc_byte_sequence_create_copy(
      loomc_make_byte_span(source.data, source.size), loomc_allocator_system(),
      &sequence)));
  return ByteSequencePtr(sequence, loomc_byte_sequence_release);
}

static iree_string_view_t SnapshotJson(
    iree_benchmark_loom_snapshot_sink_t* snapshot,
    iree_string_builder_t* storage) {
  iree_string_builder_reset(storage);
  IREE_EXPECT_OK(
      iree_benchmark_loom_snapshot_sink_append_json(snapshot, storage));
  return iree_string_builder_view(storage);
}

static iree_string_view_t ParseJsonDocument(iree_string_view_t json) {
  iree_string_view_t cursor = json;
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_OK(iree_json_consume_value(&cursor, &value));
  IREE_EXPECT_OK(iree_json_consume_insignificant(&cursor));
  EXPECT_TRUE(iree_string_view_is_empty(cursor));
  return value;
}

static iree_string_view_t LookupObject(iree_string_view_t object,
                                       iree_string_view_t key) {
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_OK(iree_json_lookup_object_value(object, key, &value));
  return value;
}

static iree_string_view_t TryLookupObject(iree_string_view_t object,
                                          iree_string_view_t key) {
  iree_string_view_t value = iree_string_view_empty();
  IREE_EXPECT_OK(iree_json_try_lookup_object_value(object, key, &value));
  return value;
}

static iree_string_view_t FirstArrayElement(iree_string_view_t array) {
  iree_string_view_t element = iree_string_view_empty();
  IREE_EXPECT_OK(iree_json_array_get(array, 0, &element));
  return element;
}

static bool JsonArrayContainsString(iree_string_view_t array,
                                    iree_string_view_t expected) {
  iree_host_size_t count = 0;
  IREE_EXPECT_OK(iree_json_array_length(array, &count));
  for (iree_host_size_t i = 0; i < count; ++i) {
    iree_string_view_t element = iree_string_view_empty();
    IREE_EXPECT_OK(iree_json_array_get(array, i, &element));
    if (iree_string_view_equal(element, expected)) {
      return true;
    }
  }
  return false;
}

TEST(BenchmarkSnapshotSinkTest, OwnsSkippedEvidenceAndMatchesJsonl) {
  for (bool scenario : {false, true}) {
    SCOPED_TRACE(scenario);
    char message[] = "provider is unavailable";
    loom_testbench_requirement_result_t reason = {
        .skipped = true,
        .op_kind = LOOM_TESTBENCH_REQUIREMENT_OP_KIND_REQUIRES,
        .code = LOOM_TESTBENCH_REQUIREMENT_SKIP_CODE_PROVIDER_UNAVAILABLE,
        .provider = IREE_SV("test.provider"),
        .provider_code = IREE_SV("unavailable"),
        .display_message = iree_make_cstring_view(message),
    };
    const iree_benchmark_loom_benchmark_result_t result = {
        .state = IREE_SV("skipped"),
        .skip_reason = &reason,
    };
    const iree_benchmark_loom_benchmark_policy_t policy = {
        .measure = IREE_SV("dispatch_complete"),
    };
    const iree_benchmark_loom_work_item_t work = {};
    const loom_module_t module = {};
    const loom_testbench_case_plan_t case_plan = {.name = IREE_SV("case")};
    const loom_testbench_scenario_plan_t scenario_plan = {
        .name = IREE_SV("scenario"),
    };
    const loom_testbench_benchmark_plan_t benchmark = {
        .name = IREE_SV("bench"),
    };
    const iree_benchmark_loom_run_identity_t run = {.run_id = IREE_SV("run")};
    iree_benchmark_loom_candidate_identity_t candidate = {
        .candidate_id = IREE_SV("c0"),
    };
    iree_benchmark_loom_snapshot_sink_t snapshot = {};
    IREE_ASSERT_OK(iree_benchmark_loom_snapshot_sink_initialize(
        iree_allocator_system(), &snapshot));
    iree_benchmark_loom_event_sink_t sink = {};
    iree_benchmark_loom_snapshot_event_sink_initialize(&snapshot, &sink);
    IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_run(&sink, &run, false,
                                                           &kNoSanitizer));
    for (size_t i = 0; i < 2; ++i) {
      candidate.candidate_index = i;
      IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_benchmark_result(
          &sink, &run, &candidate, &work, &module, &benchmark,
          scenario ? nullptr : &case_plan, scenario ? &scenario_plan : nullptr,
          &policy, &result, 0, 0));
    }
    iree_string_builder_t jsonl;
    iree_string_builder_initialize(iree_allocator_system(), &jsonl);
    IREE_ASSERT_OK(iree_benchmark_loom_append_benchmark_result(
        &run, &candidate, 0, &module, &benchmark,
        scenario ? nullptr : &case_plan, scenario ? &scenario_plan : nullptr,
        &policy, &result, 0, 0, &jsonl));
    // Neither the deferred snapshot nor rendered JSONL borrows provider text.
    std::memset(message, '?', sizeof(message));
    reason = {};
    const iree_benchmark_loom_artifact_bundle_t bundle = {};
    IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_summary(
        &sink, &run, &bundle, /*planned_case_count=*/scenario ? 0 : 1,
        /*planned_benchmark_count=*/2, /*selected_benchmark_count=*/2,
        /*logical_sample_count=*/2, /*work_item_count=*/1,
        /*failure_count=*/0, /*failed_benchmark_count=*/0,
        /*correctness_sample_count=*/0, /*correctness_failed_sample_count=*/0,
        /*dry_run=*/false));
    iree_string_builder_t output;
    iree_string_builder_initialize(iree_allocator_system(), &output);
    auto root = ParseJsonDocument(SnapshotJson(&snapshot, &output));
    auto work_items = LookupObject(root, IREE_SV("work_items"));
    iree_host_size_t count = 0;
    IREE_ASSERT_OK(iree_json_array_length(work_items, &count));
    EXPECT_EQ(count, 1u);
    auto evidence = FirstArrayElement(work_items);
    EXPECT_TRUE(iree_string_view_equal(LookupObject(evidence, IREE_SV("state")),
                                       IREE_SV("skipped")));
    auto skip = LookupObject(evidence, IREE_SV("skip_reason"));
    EXPECT_TRUE(iree_string_view_equal(
        skip, IREE_SV("{\"provider\":\"test.provider\",\"op\":\"requires\","
                      "\"code\":\"provider_unavailable\",\"provider_code\":"
                      "\"unavailable\","
                      "\"display_message\":\"provider is unavailable\"}")));
    auto row = ParseJsonDocument(iree_string_builder_view(&jsonl));
    EXPECT_TRUE(iree_string_view_equal(
        LookupObject(LookupObject(row, IREE_SV("benchmark_result")),
                     IREE_SV("skip_reason")),
        skip));
    IREE_ASSERT_OK(iree_json_array_length(
        LookupObject(root, IREE_SV("benchmarks")), &count));
    EXPECT_EQ(count, 2u);
    EXPECT_TRUE(iree_string_view_is_empty(
        TryLookupObject(evidence, IREE_SV("measurement"))));
    EXPECT_TRUE(iree_string_view_is_empty(
        TryLookupObject(evidence, IREE_SV("failure"))));
    iree_string_builder_deinitialize(&jsonl);
    iree_string_builder_deinitialize(&output);
    iree_benchmark_loom_snapshot_sink_deinitialize(&snapshot);
  }
}

TEST(BenchmarkSnapshotSinkTest, PreservesInterleavedProfileSuppression) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_benchmark_loom_snapshot_sink_t snapshot = {};
  IREE_ASSERT_OK(
      iree_benchmark_loom_snapshot_sink_initialize(allocator, &snapshot));
  iree_benchmark_loom_event_sink_t event_sink = {};
  iree_benchmark_loom_snapshot_event_sink_initialize(&snapshot, &event_sink);
  iree_benchmark_loom_run_identity_t run = {
      .run_id = IREE_SV("run"),
  };
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_run(
      &event_sink, &run, /*dry_run=*/false, &kNoSanitizer));

  iree_benchmark_loom_selected_benchmark_t selection = {
      .identity = {.candidate_id = IREE_SV("candidate")},
      .policy =
          {
              .measure_kind = IREE_BENCHMARK_LOOM_MEASURE_DISPATCH_COMPLETE,
              .measure = IREE_SV("dispatch_complete"),
          },
  };
  loom_testbench_scenario_plan_t scenario_plan = {
      .name = IREE_SV("scenario"),
  };
  selection.scenario_plan = &scenario_plan;
  iree_benchmark_loom_dispatch_comparison_candidate_t candidate = {
      .selection = &selection,
  };
  iree_benchmark_loom_benchmark_result_t result = {
      .executed = true,
      .passed = true,
      .has_benchmark_sample_ordinal = true,
      .benchmark_sample_ordinal = 7,
      .has_scenario_coordinate = true,
      .scenario_coordinate = {2, 3, 5},
      .has_hal_benchmark = true,
      .hal_benchmark =
          {
              .timing =
                  {
                      .batch_size = 1,
                      .measured_batch_count = 1,
                      .measured_operation_count = 1,
                  },
          },
  };
  for (iree_host_size_t i = 0; i < 2; ++i) {
    IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_benchmark_repetition(
        &event_sink, &run, &candidate, &selection.identity, IREE_SV("group"),
        IREE_SV("AB"), i, i, 'A', /*profile_suppressed=*/i != 0, &result));
  }
  iree_benchmark_loom_artifact_bundle_t bundle = {};
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_summary(
      &event_sink, &run, &bundle, /*planned_case_count=*/1,
      /*planned_benchmark_count=*/1, /*selected_benchmark_count=*/1,
      /*logical_sample_count=*/1, /*work_item_count=*/1,
      /*failure_count=*/0, /*failed_benchmark_count=*/0,
      /*correctness_sample_count=*/1, /*correctness_failed_sample_count=*/0,
      /*dry_run=*/false));

  iree_string_builder_t output;
  iree_string_builder_initialize(allocator, &output);
  iree_string_view_t root = ParseJsonDocument(SnapshotJson(&snapshot, &output));
  iree_string_view_t repetitions = LookupObject(root, IREE_SV("repetitions"));
  for (iree_host_size_t i = 0; i < 2; ++i) {
    iree_string_view_t repetition = iree_string_view_empty();
    IREE_ASSERT_OK(iree_json_array_get(repetitions, i, &repetition));
    iree_string_view_t suppressed = TryLookupObject(
        repetition, IREE_SV("profile_suppressed_for_interleave"));
    EXPECT_TRUE(iree_string_view_equal(
        suppressed, i == 0 ? iree_string_view_empty() : IREE_SV("true")));
    EXPECT_TRUE(iree_string_view_equal(
        LookupObject(repetition, IREE_SV("benchmark_sample_index")),
        IREE_SV("7")));
    EXPECT_TRUE(iree_string_view_equal(
        LookupObject(repetition, IREE_SV("configuration_ordinal")),
        IREE_SV("2")));
    EXPECT_TRUE(iree_string_view_equal(
        LookupObject(repetition, IREE_SV("trial_index")), IREE_SV("3")));
    EXPECT_TRUE(iree_string_view_equal(
        LookupObject(repetition, IREE_SV("trial_ordinal")), IREE_SV("5")));
  }
  iree_string_builder_deinitialize(&output);
  iree_benchmark_loom_snapshot_sink_deinitialize(&snapshot);
}

TEST(BenchmarkSnapshotSinkTest, AggregatesDeduplicatedWorkItems) {
  const iree_benchmark_loom_work_item_t work_item = {.work_item_index = 7};
  iree_allocator_t allocator = iree_allocator_system();
  iree_benchmark_loom_snapshot_sink_t snapshot = {};
  IREE_ASSERT_OK(
      iree_benchmark_loom_snapshot_sink_initialize(allocator, &snapshot));
  iree_benchmark_loom_event_sink_t event_sink = {};
  iree_benchmark_loom_snapshot_event_sink_initialize(&snapshot, &event_sink);

  iree_benchmark_loom_run_identity_t run = {
      .run_id = IREE_SV("run"),
      .source = IREE_SV("input.loom"),
      .results_path = IREE_SV("-"),
      .file_output_dir = IREE_SV("/tmp/loom"),
  };
  iree_benchmark_loom_candidate_identity_t candidate0 = {
      .candidate_id = IREE_SV("c0"),
      .candidate_index = 0,
  };
  iree_benchmark_loom_candidate_identity_t candidate1 = {
      .candidate_id = IREE_SV("c1"),
      .candidate_index = 1,
  };
  loom_module_t module = {};
  loom_testbench_benchmark_plan_t benchmark_plan = {
      .name = IREE_SV("kernel_latency"),
  };
  loom_testbench_case_plan_t case_plan = {.name = IREE_SV("kernel_case")};
  iree_benchmark_loom_benchmark_policy_t policy = {
      .measure_kind = IREE_BENCHMARK_LOOM_MEASURE_CASE_END_TO_END,
      .measure = IREE_SV("case_end_to_end"),
  };
  iree_benchmark_loom_benchmark_result_t result = {
      .executed = true,
      .passed = true,
      .has_benchmark_sample_ordinal = true,
      .benchmark_sample_ordinal = 0,
      .samples_per_iteration = 1,
  };
  result.timing.count = 3;
  result.timing.total_ns = 90;
  result.timing.minimum_ns = 20;
  result.timing.maximum_ns = 40;
  result.timing.mean_ns = 30.0;
  result.timing.p50_ns = 30;
  result.timing.p90_ns = 40;
  iree_benchmark_loom_workload_value_t workload_values[] = {
      {.type = LOOM_SCALAR_TYPE_INDEX, .value = 4096},
      {.type = LOOM_SCALAR_TYPE_I32, .value = 513},
  };
  iree_benchmark_loom_launch_record_t launch_record = {
      .sample = {},
      .sequence_step_ordinal = 0,
      .entry = IREE_SV("kernel_entry"),
      .workload_values = workload_values,
      .workload_value_count = IREE_ARRAYSIZE(workload_values),
      .launch_config =
          {
              .type = LOOMC_STRUCTURE_TYPE_LAUNCH_CONFIG,
              .structure_size = sizeof(loomc_launch_config_t),
              .next = nullptr,
              .workgroup_count = {64, 2, 1},
              .workgroup_size = {64, 1, 1},
              .workgroup_cluster_size = {2, 1, 1},
              .subgroup_size = 32,
              .workgroup_storage_bytes = 0,
          },
  };
  iree_benchmark_loom_launch_evidence_t launch_evidence = {
      .host_allocator = {},
      .case_plan = &case_plan,
      .records = &launch_record,
      .record_count = 1,
      .workload_values = workload_values,
      .workload_value_count = IREE_ARRAYSIZE(workload_values),
  };
  result.launch_evidence = &launch_evidence;
  loom_sanitizer_options_t sanitizer = {
      .checks = LOOM_SANITIZER_CHECK_RACE,
      .reporting_mode = LOOM_SANITIZER_REPORTING_MODE_REPORT_ONLY,
  };

  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_run(
      &event_sink, &run, /*dry_run=*/false, &sanitizer));
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_benchmark_result(
      &event_sink, &run, &candidate0, &work_item, &module, &benchmark_plan,
      &case_plan, /*scenario_plan=*/nullptr, &policy, &result,
      /*correctness_sample_count=*/1,
      /*correctness_failed_sample_count=*/0));
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_benchmark_result(
      &event_sink, &run, &candidate1, &work_item, &module, &benchmark_plan,
      &case_plan, /*scenario_plan=*/nullptr, &policy, &result,
      /*correctness_sample_count=*/1,
      /*correctness_failed_sample_count=*/0));
  workload_values[0].value = 1;
  launch_record.launch_config.workgroup_count.x = 1;
  iree_benchmark_loom_artifact_bundle_t bundle = {};
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_summary(
      &event_sink, &run, &bundle, /*planned_case_count=*/1,
      /*planned_benchmark_count=*/2, /*selected_benchmark_count=*/2,
      /*logical_sample_count=*/2, /*work_item_count=*/1,
      /*failure_count=*/0, /*failed_benchmark_count=*/0,
      /*correctness_sample_count=*/1, /*correctness_failed_sample_count=*/0,
      /*dry_run=*/false));

  iree_string_builder_t output;
  iree_string_builder_initialize(allocator, &output);
  iree_string_view_t root = ParseJsonDocument(SnapshotJson(&snapshot, &output));

  iree_string_view_t output_format =
      LookupObject(root, IREE_SV("output_format"));
  EXPECT_TRUE(iree_string_view_equal(output_format, IREE_SV("snapshot")));
  iree_string_view_t sanitizer_json = LookupObject(root, IREE_SV("sanitizer"));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(sanitizer_json, IREE_SV("checks")), IREE_SV("race")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(sanitizer_json, IREE_SV("reporting_mode")),
      IREE_SV("report-only")));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(root, IREE_SV("work_item_count"))));
  iree_string_view_t summary = LookupObject(root, IREE_SV("summary"));
  iree_string_view_t planned = LookupObject(summary, IREE_SV("planned"));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(planned, IREE_SV("work_item_count")), IREE_SV("1")));

  iree_string_view_t work_items = LookupObject(root, IREE_SV("work_items"));
  iree_host_size_t work_item_count = 0;
  IREE_ASSERT_OK(iree_json_array_length(work_items, &work_item_count));
  EXPECT_EQ(work_item_count, 1u);

  iree_string_view_t benchmarks = LookupObject(root, IREE_SV("benchmarks"));
  iree_host_size_t benchmark_count = 0;
  IREE_ASSERT_OK(iree_json_array_length(benchmarks, &benchmark_count));
  EXPECT_EQ(benchmark_count, 2u);

  iree_string_view_t first_work_item = FirstArrayElement(work_items);
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(first_work_item, IREE_SV("benchmark")),
      IREE_SV("kernel_latency")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(first_work_item, IREE_SV("case")), IREE_SV("kernel_case")));
  iree_string_view_t policy_json =
      LookupObject(first_work_item, IREE_SV("policy"));
  EXPECT_TRUE(
      iree_string_view_equal(LookupObject(policy_json, IREE_SV("measure")),
                             IREE_SV("case_end_to_end")));
  iree_string_view_t correctness =
      LookupObject(first_work_item, IREE_SV("correctness"));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(correctness, IREE_SV("sample_count")), IREE_SV("1")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(correctness, IREE_SV("failed_sample_count")), IREE_SV("0")));
  iree_string_view_t measurement =
      LookupObject(first_work_item, IREE_SV("measurement"));
  EXPECT_FALSE(iree_string_view_is_empty(
      LookupObject(measurement, IREE_SV("timing_ns"))));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(first_work_item, IREE_SV("timing_ns"))));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(measurement, IREE_SV("batch_timing_ns"))));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(measurement, IREE_SV("dispatch_timing_ns"))));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(measurement, IREE_SV("operation_timing_ns"))));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(first_work_item, IREE_SV("profile_replay"))));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(first_work_item, IREE_SV("compile_report"))));
  iree_string_view_t launches =
      LookupObject(first_work_item, IREE_SV("launches"));
  iree_string_view_t launch = FirstArrayElement(launches);
  iree_string_view_t workload = LookupObject(launch, IREE_SV("workload"));
  iree_string_view_t first_workload_value = FirstArrayElement(workload);
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(first_workload_value, IREE_SV("type")), IREE_SV("index")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(first_workload_value, IREE_SV("value")), IREE_SV("4096")));
  iree_string_view_t launch_config =
      LookupObject(launch, IREE_SV("launch_config"));
  iree_string_view_t workgroup_count =
      LookupObject(launch_config, IREE_SV("workgroup_count"));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(workgroup_count, IREE_SV("x")), IREE_SV("64")));
  iree_string_view_t workgroup_cluster_size =
      LookupObject(launch_config, IREE_SV("workgroup_cluster_size"));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(workgroup_cluster_size, IREE_SV("x")), IREE_SV("2")));
  EXPECT_TRUE(
      iree_string_view_is_empty(TryLookupObject(root, IREE_SV("failures"))));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(root, IREE_SV("failed_samples"))));

  iree_string_builder_deinitialize(&output);
  iree_benchmark_loom_snapshot_sink_deinitialize(&snapshot);
}

TEST(BenchmarkSnapshotSinkTest, IncludesRequestedProfileSummary) {
  const iree_benchmark_loom_work_item_t work_item = {};
  iree_allocator_t allocator = iree_allocator_system();
  iree_benchmark_loom_snapshot_sink_t snapshot = {};
  IREE_ASSERT_OK(
      iree_benchmark_loom_snapshot_sink_initialize(allocator, &snapshot));
  iree_benchmark_loom_event_sink_t event_sink = {};
  iree_benchmark_loom_snapshot_event_sink_initialize(&snapshot, &event_sink);

  iree_benchmark_loom_run_identity_t run = {
      .run_id = IREE_SV("run"),
      .source = IREE_SV("input.loom"),
      .results_path = IREE_SV("-"),
  };
  iree_benchmark_loom_candidate_identity_t candidate = {
      .candidate_id = IREE_SV("c0"),
  };
  loom_module_t module = {};
  loom_testbench_benchmark_plan_t benchmark_plan = {
      .name = IREE_SV("kernel_latency"),
  };
  loom_testbench_case_plan_t case_plan = {.name = IREE_SV("kernel_case")};
  iree_benchmark_loom_benchmark_policy_t policy = {
      .measure = IREE_SV("dispatch_complete"),
  };
  iree_benchmark_loom_benchmark_result_t result = {
      .executed = true,
      .passed = true,
      .samples_per_iteration = 1,
      .has_hal_benchmark = true,
  };
  result.hal_benchmark.timing.batch_size = 16;
  result.hal_benchmark.timing.measured_batch_count = 3;
  result.hal_benchmark.timing.measured_operation_count = 48;
  result.hal_benchmark.timing.stop_reason =
      LOOM_RUN_BENCHMARK_STOP_REASON_STABLE;
  result.hal_benchmark.timing.operation_timing.count = 3;
  result.hal_benchmark.timing.operation_timing.total_ns = 90;
  result.hal_benchmark.timing.operation_timing.minimum_ns = 20;
  result.hal_benchmark.timing.operation_timing.maximum_ns = 40;
  result.hal_benchmark.timing.operation_timing.mean_ns = 30.0;
  result.hal_benchmark.timing.operation_timing.p50_ns = 30;
  result.hal_benchmark.timing.operation_timing.p90_ns = 40;
  result.hal_benchmark.profile_replay.requested = true;

  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_run(
      &event_sink, &run, /*dry_run=*/false, &kNoSanitizer));
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_benchmark_result(
      &event_sink, &run, &candidate, &work_item, &module, &benchmark_plan,
      &case_plan, /*scenario_plan=*/nullptr, &policy, &result,
      /*correctness_sample_count=*/1,
      /*correctness_failed_sample_count=*/0));
  iree_benchmark_loom_artifact_bundle_t bundle = {};
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_summary(
      &event_sink, &run, &bundle, /*planned_case_count=*/1,
      /*planned_benchmark_count=*/1, /*selected_benchmark_count=*/1,
      /*logical_sample_count=*/1, /*work_item_count=*/1,
      /*failure_count=*/0, /*failed_benchmark_count=*/0,
      /*correctness_sample_count=*/1, /*correctness_failed_sample_count=*/0,
      /*dry_run=*/false));

  iree_string_builder_t output;
  iree_string_builder_initialize(allocator, &output);
  iree_string_view_t root = ParseJsonDocument(SnapshotJson(&snapshot, &output));
  iree_string_view_t work_items = LookupObject(root, IREE_SV("work_items"));
  iree_string_view_t first_work_item = FirstArrayElement(work_items);
  iree_string_view_t profile_replay =
      LookupObject(first_work_item, IREE_SV("profile_replay"));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(profile_replay, IREE_SV("measurement_relationship")),
      IREE_SV("distinct_execution")));
  iree_string_view_t profile = LookupObject(profile_replay, IREE_SV("summary"));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(profile, IREE_SV("requested")), IREE_SV("true")));
  EXPECT_TRUE(iree_string_view_equal(LookupObject(profile, IREE_SV("executed")),
                                     IREE_SV("false")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(profile, IREE_SV("row_count")), IREE_SV("0")));
  iree_host_size_t profile_row_count = 0;
  IREE_ASSERT_OK(iree_json_array_length(LookupObject(profile, IREE_SV("rows")),
                                        &profile_row_count));
  EXPECT_EQ(profile_row_count, 0u);

  iree_string_builder_deinitialize(&output);
  iree_benchmark_loom_snapshot_sink_deinitialize(&snapshot);
}

TEST(BenchmarkSnapshotSinkTest, IncludesHalTimingCountsAndWarnings) {
  const iree_benchmark_loom_work_item_t work_item = {};
  iree_allocator_t allocator = iree_allocator_system();
  iree_benchmark_loom_snapshot_sink_t snapshot = {};
  IREE_ASSERT_OK(
      iree_benchmark_loom_snapshot_sink_initialize(allocator, &snapshot));
  iree_benchmark_loom_event_sink_t event_sink = {};
  iree_benchmark_loom_snapshot_event_sink_initialize(&snapshot, &event_sink);

  iree_benchmark_loom_run_identity_t run = {
      .run_id = IREE_SV("run"),
      .source = IREE_SV("input.loom"),
      .results_path = IREE_SV("-"),
  };
  iree_benchmark_loom_candidate_identity_t candidate = {
      .candidate_id = IREE_SV("c0"),
  };
  loom_module_t module = {};
  loom_testbench_benchmark_plan_t benchmark_plan = {
      .name = IREE_SV("kernel_latency"),
  };
  loom_testbench_case_plan_t case_plan = {.name = IREE_SV("kernel_case")};
  iree_benchmark_loom_benchmark_policy_t policy = {
      .measure = IREE_SV("dispatch_complete"),
  };
  policy.hal_options.timing.stable_p90_to_p50_delta_ppm = 100000;
  iree_benchmark_loom_benchmark_result_t result = {
      .executed = true,
      .passed = true,
      .samples_per_iteration = 1,
      .has_hal_benchmark = true,
  };
  result.hal_benchmark.timing.batch_size = 1;
  result.hal_benchmark.timing.measured_batch_count = 3;
  result.hal_benchmark.timing.measured_operation_count = 3;
  result.hal_benchmark.timing.measured_duration_ns = 900;
  result.hal_benchmark.timing.stop_reason =
      LOOM_RUN_BENCHMARK_STOP_REASON_MAX_BATCH_COUNT;
  result.hal_benchmark.timing.batch_timing.count = 3;
  result.hal_benchmark.timing.batch_timing.total_ns = 900;
  result.hal_benchmark.timing.batch_timing.minimum_ns = 200;
  result.hal_benchmark.timing.batch_timing.maximum_ns = 400;
  result.hal_benchmark.timing.batch_timing.mean_ns = 300.0;
  result.hal_benchmark.timing.batch_timing.p50_ns = 300;
  result.hal_benchmark.timing.batch_timing.p90_ns = 400;
  result.hal_benchmark.timing.batch_timing.p90_to_p50_delta_ppm = 333333;
  result.hal_benchmark.timing.operation_timing =
      result.hal_benchmark.timing.batch_timing;
  result.data_cache.populated = true;
  result.data_cache.dispatches_per_batch = 6;

  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_run(
      &event_sink, &run, /*dry_run=*/false, &kNoSanitizer));
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_benchmark_result(
      &event_sink, &run, &candidate, &work_item, &module, &benchmark_plan,
      &case_plan, /*scenario_plan=*/nullptr, &policy, &result,
      /*correctness_sample_count=*/1,
      /*correctness_failed_sample_count=*/0));
  iree_benchmark_loom_artifact_bundle_t bundle = {};
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_summary(
      &event_sink, &run, &bundle, /*planned_case_count=*/1,
      /*planned_benchmark_count=*/1, /*selected_benchmark_count=*/1,
      /*logical_sample_count=*/1, /*work_item_count=*/1,
      /*failure_count=*/0, /*failed_benchmark_count=*/0,
      /*correctness_sample_count=*/1, /*correctness_failed_sample_count=*/0,
      /*dry_run=*/false));

  iree_string_builder_t output;
  iree_string_builder_initialize(allocator, &output);
  iree_string_view_t root = ParseJsonDocument(SnapshotJson(&snapshot, &output));
  iree_string_view_t work_items = LookupObject(root, IREE_SV("work_items"));
  iree_string_view_t first_work_item = FirstArrayElement(work_items);
  iree_string_view_t measurement =
      LookupObject(first_work_item, IREE_SV("measurement"));
  EXPECT_TRUE(iree_string_view_is_empty(TryLookupObject(
      first_work_item, IREE_SV("logical_operations_per_batch"))));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(first_work_item, IREE_SV("operation_timing_ns"))));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(measurement, IREE_SV("timing_ns"))));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(measurement, IREE_SV("logical_operations_per_batch")),
      IREE_SV("1")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(measurement, IREE_SV("physical_dispatches_per_batch")),
      IREE_SV("6")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(measurement,
                   IREE_SV("physical_dispatches_per_logical_operation")),
      IREE_SV("6")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(measurement, IREE_SV("measured_logical_operation_count")),
      IREE_SV("3")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(measurement, IREE_SV("measured_physical_dispatch_count")),
      IREE_SV("18")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(measurement, IREE_SV("mean_physical_dispatch_duration_ns")),
      IREE_SV("50.000")));
  EXPECT_FALSE(iree_string_view_is_empty(
      LookupObject(measurement, IREE_SV("batch_timing_ns"))));
  EXPECT_FALSE(iree_string_view_is_empty(
      LookupObject(measurement, IREE_SV("operation_timing_ns"))));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(measurement, IREE_SV("dispatch_timing_ns"))));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(measurement, IREE_SV("measured_dispatch_count"))));
  EXPECT_TRUE(iree_string_view_is_empty(
      TryLookupObject(measurement, IREE_SV("measured_operation_count"))));

  iree_string_view_t timing_interpretation =
      LookupObject(measurement, IREE_SV("timing_interpretation"));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(timing_interpretation, IREE_SV("score")),
      IREE_SV("operation_timing_ns")));
  iree_string_view_t warnings =
      LookupObject(timing_interpretation, IREE_SV("warnings"));
  EXPECT_TRUE(
      JsonArrayContainsString(warnings, IREE_SV("short_measured_duration")));
  EXPECT_TRUE(JsonArrayContainsString(
      warnings, IREE_SV("single_logical_operation_batch")));
  EXPECT_TRUE(JsonArrayContainsString(
      warnings, IREE_SV("low_physical_dispatch_sample_count")));
  EXPECT_TRUE(JsonArrayContainsString(
      warnings, IREE_SV("sub_microsecond_logical_operation")));
  EXPECT_TRUE(
      JsonArrayContainsString(warnings, IREE_SV("unstable_p90_to_p50")));

  iree_string_builder_deinitialize(&output);
  iree_benchmark_loom_snapshot_sink_deinitialize(&snapshot);
}

TEST(BenchmarkSnapshotSinkTest, IncludesRequestedCompileReport) {
  const iree_benchmark_loom_work_item_t work_item = {};
  iree_allocator_t allocator = iree_allocator_system();
  iree_benchmark_loom_snapshot_sink_t snapshot = {};
  IREE_ASSERT_OK(
      iree_benchmark_loom_snapshot_sink_initialize(allocator, &snapshot));
  iree_benchmark_loom_event_sink_t event_sink = {};
  iree_benchmark_loom_snapshot_event_sink_initialize(&snapshot, &event_sink);

  const iree_string_view_t compile_report_json = IREE_SV(
      R"({"artifact_kind":"target-artifact","economics":{"memory":{"per_workitem_issued":{"read_bytes":12,"write_bytes":4,"global_load_count":3,"global_store_count":1},"dispatch_issued":{"read_bytes":96,"write_bytes":32,"global_load_count":24,"global_store_count":8}}}})");
  ByteSequencePtr compile_report_contents =
      CopyByteSequence(compile_report_json);
  const loomc_artifact_t compile_report_artifact = {
      .kind = LOOMC_ARTIFACT_KIND_REPORT,
      .format =
          loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_COMPILE_REPORT_JSON),
      .identifier = loomc_make_cstring_view("compile_report"),
      .contents = compile_report_contents.get(),
  };

  iree_benchmark_loom_run_identity_t run = {
      .run_id = IREE_SV("run"),
      .source = IREE_SV("input.loom"),
      .results_path = IREE_SV("-"),
  };
  iree_benchmark_loom_candidate_identity_t candidate = {
      .candidate_id = IREE_SV("c0"),
  };
  loom_module_t module = {};
  loom_testbench_benchmark_plan_t benchmark_plan = {
      .name = IREE_SV("kernel_latency"),
  };
  loom_testbench_case_plan_t case_plan = {.name = IREE_SV("kernel_case")};
  iree_benchmark_loom_benchmark_policy_t policy = {
      .measure = IREE_SV("case_end_to_end"),
  };
  iree_benchmark_loom_benchmark_result_t result = {
      .executed = true,
      .passed = true,
      .samples_per_iteration = 1,
  };
  result.timing.count = 1;
  result.timing.total_ns = 10;
  result.timing.minimum_ns = 10;
  result.timing.maximum_ns = 10;
  result.timing.mean_ns = 10.0;
  result.timing.p50_ns = 10;
  result.timing.p90_ns = 10;
  result.compile_report = &compile_report_artifact;

  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_run(
      &event_sink, &run, /*dry_run=*/false, &kNoSanitizer));
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_benchmark_result(
      &event_sink, &run, &candidate, &work_item, &module, &benchmark_plan,
      &case_plan, /*scenario_plan=*/nullptr, &policy, &result,
      /*correctness_sample_count=*/1,
      /*correctness_failed_sample_count=*/0));
  iree_benchmark_loom_artifact_bundle_t bundle = {};
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_summary(
      &event_sink, &run, &bundle, /*planned_case_count=*/1,
      /*planned_benchmark_count=*/1, /*selected_benchmark_count=*/1,
      /*logical_sample_count=*/1, /*work_item_count=*/1,
      /*failure_count=*/0, /*failed_benchmark_count=*/0,
      /*correctness_sample_count=*/1, /*correctness_failed_sample_count=*/0,
      /*dry_run=*/false));

  iree_string_builder_t output;
  iree_string_builder_initialize(allocator, &output);
  iree_string_view_t root = ParseJsonDocument(SnapshotJson(&snapshot, &output));
  iree_string_view_t work_items = LookupObject(root, IREE_SV("work_items"));
  iree_string_view_t first_work_item = FirstArrayElement(work_items);
  iree_string_view_t compile_report =
      LookupObject(first_work_item, IREE_SV("compile_report"));
  iree_string_view_t artifact_kind =
      LookupObject(compile_report, IREE_SV("artifact_kind"));
  EXPECT_TRUE(
      iree_string_view_equal(artifact_kind, IREE_SV("target-artifact")));
  iree_string_view_t economics =
      LookupObject(compile_report, IREE_SV("economics"));
  iree_string_view_t memory = LookupObject(economics, IREE_SV("memory"));
  iree_string_view_t per_workitem_issued =
      LookupObject(memory, IREE_SV("per_workitem_issued"));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(per_workitem_issued, IREE_SV("read_bytes")), IREE_SV("12")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(per_workitem_issued, IREE_SV("write_bytes")), IREE_SV("4")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(per_workitem_issued, IREE_SV("global_load_count")),
      IREE_SV("3")));
  iree_string_view_t dispatch_issued =
      LookupObject(memory, IREE_SV("dispatch_issued"));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(dispatch_issued, IREE_SV("read_bytes")), IREE_SV("96")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(dispatch_issued, IREE_SV("write_bytes")), IREE_SV("32")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(dispatch_issued, IREE_SV("global_store_count")),
      IREE_SV("8")));

  iree_string_builder_deinitialize(&output);
  iree_benchmark_loom_snapshot_sink_deinitialize(&snapshot);
}

TEST(BenchmarkSnapshotSinkTest, IncludesFailurePayloadsOnFailure) {
  const iree_benchmark_loom_work_item_t work_item = {};
  iree_allocator_t allocator = iree_allocator_system();
  iree_benchmark_loom_snapshot_sink_t snapshot = {};
  IREE_ASSERT_OK(
      iree_benchmark_loom_snapshot_sink_initialize(allocator, &snapshot));
  iree_benchmark_loom_event_sink_t event_sink = {};
  iree_benchmark_loom_snapshot_event_sink_initialize(&snapshot, &event_sink);

  iree_benchmark_loom_run_identity_t run = {
      .run_id = IREE_SV("run"),
      .source = IREE_SV("input.loom"),
      .results_path = IREE_SV("-"),
  };
  iree_benchmark_loom_candidate_identity_t candidate = {
      .candidate_id = IREE_SV("c0"),
  };
  loom_module_t module = {};
  loom_testbench_benchmark_plan_t benchmark_plan = {
      .name = IREE_SV("kernel_latency"),
  };
  loom_testbench_case_plan_t case_plan = {.name = IREE_SV("kernel_case")};
  iree_benchmark_loom_benchmark_policy_t policy = {
      .measure = IREE_SV("dispatch_complete"),
  };
  iree_benchmark_loom_hal_actual_provider_t provider = {};
  provider.execution.invocation_options.function_name =
      IREE_SV("rejected_kernel");
  provider.execution.compile_failure_stage = IREE_SV("compile");
  provider.execution.compile_failure_kind = IREE_SV("diagnostics");
  provider.execution.compile_failure_message =
      IREE_SV("candidate did not lower");
  provider.execution.compile_rejected = true;
  provider.diagnostics.error_count = 1;
  iree_benchmark_loom_benchmark_result_t result = {};
  iree_benchmark_loom_benchmark_result_set_compile_rejection(&provider,
                                                             &result);
  result.diagnostic_json = IREE_SV("{\"message\":\"bad op\"}");

  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_run(
      &event_sink, &run, /*dry_run=*/false, &kNoSanitizer));
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_failure(
      &event_sink, &run, IREE_SV("parse"), IREE_SV("diagnostics"),
      IREE_SV("input module has parse errors"), /*diagnostics=*/NULL));
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_benchmark_result(
      &event_sink, &run, &candidate, &work_item, &module, &benchmark_plan,
      &case_plan, /*scenario_plan=*/nullptr, &policy, &result,
      /*correctness_sample_count=*/0,
      /*correctness_failed_sample_count=*/0));
  iree_benchmark_loom_artifact_bundle_t bundle = {};
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_summary(
      &event_sink, &run, &bundle, /*planned_case_count=*/1,
      /*planned_benchmark_count=*/1, /*selected_benchmark_count=*/1,
      /*logical_sample_count=*/1, /*work_item_count=*/1,
      /*failure_count=*/1, /*failed_benchmark_count=*/1,
      /*correctness_sample_count=*/0, /*correctness_failed_sample_count=*/0,
      /*dry_run=*/false));

  iree_string_builder_t output;
  iree_string_builder_initialize(allocator, &output);
  iree_string_view_t root = ParseJsonDocument(SnapshotJson(&snapshot, &output));
  iree_string_view_t failures = LookupObject(root, IREE_SV("failures"));
  iree_host_size_t failure_count = 0;
  IREE_ASSERT_OK(iree_json_array_length(failures, &failure_count));
  EXPECT_EQ(failure_count, 1u);
  iree_string_view_t work_items = LookupObject(root, IREE_SV("work_items"));
  iree_string_view_t first_work_item = FirstArrayElement(work_items);
  iree_string_view_t failure =
      LookupObject(first_work_item, IREE_SV("failure"));
  iree_string_view_t failure_entry = LookupObject(failure, IREE_SV("entry"));
  EXPECT_TRUE(
      iree_string_view_equal(failure_entry, IREE_SV("rejected_kernel")));
  iree_string_view_t failure_stage = LookupObject(failure, IREE_SV("stage"));
  EXPECT_TRUE(iree_string_view_equal(failure_stage, IREE_SV("compile")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(failure, IREE_SV("diagnostic_error_count")), IREE_SV("1")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(failure, IREE_SV("diagnostic_warning_count")),
      IREE_SV("0")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(failure, IREE_SV("diagnostic_remark_count")), IREE_SV("0")));
  iree_string_view_t diagnostics =
      LookupObject(failure, IREE_SV("diagnostics"));
  iree_host_size_t diagnostic_count = 0;
  IREE_ASSERT_OK(iree_json_array_length(diagnostics, &diagnostic_count));
  EXPECT_EQ(diagnostic_count, 1u);

  iree_string_builder_deinitialize(&output);
  iree_benchmark_loom_snapshot_sink_deinitialize(&snapshot);
}

TEST(BenchmarkSnapshotSinkTest, DryRunReportsPlannedWorkAliases) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_benchmark_loom_snapshot_sink_t snapshot = {};
  IREE_ASSERT_OK(
      iree_benchmark_loom_snapshot_sink_initialize(allocator, &snapshot));
  iree_benchmark_loom_event_sink_t event_sink = {};
  iree_benchmark_loom_snapshot_event_sink_initialize(&snapshot, &event_sink);

  iree_benchmark_loom_run_identity_t run = {
      .run_id = IREE_SV("run"),
      .source = IREE_SV("input.loom"),
      .results_path = IREE_SV("-"),
      .file_output_dir = IREE_SV("/tmp/loom"),
  };
  iree_benchmark_loom_selected_benchmark_t selections[2] = {};
  selections[0].identity.candidate_id = IREE_SV("c0");
  selections[0].identity.candidate_index = 0;
  selections[1].identity.candidate_id = IREE_SV("c1");
  selections[1].identity.candidate_index = 1;
  loom_testbench_benchmark_plan_t benchmark_plans[2] = {};
  benchmark_plans[0].name = IREE_SV("kernel_latency_default");
  benchmark_plans[0].sample_count = 1;
  benchmark_plans[1].name = IREE_SV("kernel_latency_alias");
  benchmark_plans[1].sample_count = 1;
  loom_testbench_case_plan_t case_plan = {.name = IREE_SV("kernel_case")};
  selections[0].benchmark_plan = &benchmark_plans[0];
  selections[0].case_plan = &case_plan;
  selections[0].policy.measure = IREE_SV("dispatch_complete");
  selections[1].benchmark_plan = &benchmark_plans[1];
  selections[1].case_plan = &case_plan;
  selections[1].policy.measure = IREE_SV("dispatch_complete");
  iree_benchmark_loom_logical_sample_t logical_samples[2] = {};
  logical_samples[0].selection_index = 0;
  logical_samples[0].begin_benchmark_sample = 0;
  logical_samples[0].end_benchmark_sample = 1;
  logical_samples[0].work_item_index = 7;
  logical_samples[0].next_alias_index = 1;
  logical_samples[1].selection_index = 1;
  logical_samples[1].begin_benchmark_sample = 0;
  logical_samples[1].end_benchmark_sample = 1;
  logical_samples[1].work_item_index = 7;
  logical_samples[1].next_alias_index =
      IREE_BENCHMARK_LOOM_WORK_PLAN_INDEX_INVALID;
  iree_benchmark_loom_work_item_t work_item = {
      .kind = IREE_BENCHMARK_LOOM_WORK_ITEM_DISPATCH_SAMPLE,
      .work_item_index = 7,
      .representative_selection_index = 0,
      .hal_compile_item_index = 0,
      .begin_benchmark_sample = 0,
      .end_benchmark_sample = 1,
  };
  iree_benchmark_loom_work_plan_t work_plan = {
      .selected_benchmarks = selections,
      .selected_benchmark_count = IREE_ARRAYSIZE(selections),
      .logical_samples = logical_samples,
      .logical_sample_count = IREE_ARRAYSIZE(logical_samples),
      .work_items = &work_item,
      .work_item_count = 1,
  };
  loom_module_t module = {};
  iree_benchmark_loom_artifact_bundle_t bundle = {};

  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_run(
      &event_sink, &run, /*dry_run=*/true, &kNoSanitizer));
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_work_plan(
      &event_sink, &run, &module, &work_plan));
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_summary(
      &event_sink, &run, &bundle, /*planned_case_count=*/1,
      /*planned_benchmark_count=*/2, /*selected_benchmark_count=*/2,
      /*logical_sample_count=*/2, /*work_item_count=*/1,
      /*failure_count=*/0, /*failed_benchmark_count=*/0,
      /*correctness_sample_count=*/0, /*correctness_failed_sample_count=*/0,
      /*dry_run=*/true));

  iree_string_builder_t output;
  iree_string_builder_initialize(allocator, &output);
  iree_string_view_t root = ParseJsonDocument(SnapshotJson(&snapshot, &output));

  bool dry_run = false;
  IREE_ASSERT_OK(iree_json_lookup_bool(root, IREE_SV("dry_run"), &dry_run));
  EXPECT_TRUE(dry_run);
  iree_string_view_t benchmarks = iree_string_view_empty();
  IREE_ASSERT_OK(
      iree_json_lookup_object_value(root, IREE_SV("benchmarks"), &benchmarks));
  iree_host_size_t benchmark_count = 0;
  IREE_ASSERT_OK(iree_json_array_length(benchmarks, &benchmark_count));
  EXPECT_EQ(benchmark_count, 2u);
  iree_string_view_t work_items = iree_string_view_empty();
  IREE_ASSERT_OK(
      iree_json_lookup_object_value(root, IREE_SV("work_items"), &work_items));
  iree_host_size_t work_item_count = 0;
  IREE_ASSERT_OK(iree_json_array_length(work_items, &work_item_count));
  EXPECT_EQ(work_item_count, 1u);

  iree_string_view_t first_work_item = iree_string_view_empty();
  IREE_ASSERT_OK(iree_json_array_get(work_items, 0, &first_work_item));
  iree_string_view_t work_item_index = iree_string_view_empty();
  IREE_ASSERT_OK(iree_json_lookup_object_value(
      first_work_item, IREE_SV("work_item_index"), &work_item_index));
  EXPECT_TRUE(iree_string_view_equal(work_item_index, IREE_SV("7")));
  iree_string_view_t work_item_kind = iree_string_view_empty();
  IREE_ASSERT_OK(iree_json_lookup_object_value(first_work_item, IREE_SV("kind"),
                                               &work_item_kind));
  EXPECT_TRUE(
      iree_string_view_equal(work_item_kind, IREE_SV("dispatch_sample")));
  iree_string_view_t representative_candidate_id = iree_string_view_empty();
  IREE_ASSERT_OK(iree_json_lookup_object_value(
      first_work_item, IREE_SV("representative_candidate_id"),
      &representative_candidate_id));
  EXPECT_TRUE(
      iree_string_view_equal(representative_candidate_id, IREE_SV("c0")));

  iree_string_view_t first_benchmark = iree_string_view_empty();
  IREE_ASSERT_OK(iree_json_array_get(benchmarks, 0, &first_benchmark));
  iree_string_view_t second_benchmark = iree_string_view_empty();
  IREE_ASSERT_OK(iree_json_array_get(benchmarks, 1, &second_benchmark));
  iree_string_view_t first_candidate_id = iree_string_view_empty();
  IREE_ASSERT_OK(iree_json_lookup_object_value(
      first_benchmark, IREE_SV("candidate_id"), &first_candidate_id));
  EXPECT_TRUE(iree_string_view_equal(first_candidate_id, IREE_SV("c0")));
  iree_string_view_t second_candidate_id = iree_string_view_empty();
  IREE_ASSERT_OK(iree_json_lookup_object_value(
      second_benchmark, IREE_SV("candidate_id"), &second_candidate_id));
  EXPECT_TRUE(iree_string_view_equal(second_candidate_id, IREE_SV("c1")));
  iree_string_view_t first_alias_work_item = iree_string_view_empty();
  IREE_ASSERT_OK(iree_json_lookup_object_value(
      first_benchmark, IREE_SV("work_item_index"), &first_alias_work_item));
  EXPECT_TRUE(iree_string_view_equal(first_alias_work_item, IREE_SV("7")));
  iree_string_view_t second_alias_work_item = iree_string_view_empty();
  IREE_ASSERT_OK(iree_json_lookup_object_value(
      second_benchmark, IREE_SV("work_item_index"), &second_alias_work_item));
  EXPECT_TRUE(iree_string_view_equal(second_alias_work_item, IREE_SV("7")));

  iree_string_builder_deinitialize(&output);
  iree_benchmark_loom_snapshot_sink_deinitialize(&snapshot);
}

TEST(BenchmarkSnapshotSinkTest, DryRunReportsScenarioTrialCoordinates) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_benchmark_loom_snapshot_sink_t snapshot = {};
  IREE_ASSERT_OK(
      iree_benchmark_loom_snapshot_sink_initialize(allocator, &snapshot));
  iree_benchmark_loom_event_sink_t event_sink = {};
  iree_benchmark_loom_snapshot_event_sink_initialize(&snapshot, &event_sink);

  iree_benchmark_loom_run_identity_t run = {
      .run_id = IREE_SV("run"),
      .source = IREE_SV("input.loom"),
      .results_path = IREE_SV("-"),
      .file_output_dir = IREE_SV("/tmp/loom"),
  };
  loom_testbench_benchmark_plan_t benchmark_plan = {
      .name = IREE_SV("scenario_throughput"),
      .sample_count = 8,
  };
  loom_testbench_scenario_plan_t scenario_plan = {.name = IREE_SV("scenario")};
  iree_benchmark_loom_selected_benchmark_t selection = {};
  selection.identity.candidate_id = IREE_SV("c0");
  selection.benchmark_plan = &benchmark_plan;
  selection.scenario_plan = &scenario_plan;
  selection.policy.measure = IREE_SV("dispatch_complete");
  const loom_testbench_scenario_sample_coordinate_t coordinate = {1, 2, 3};
  iree_benchmark_loom_logical_sample_t logical_sample = {
      .begin_benchmark_sample = 7,
      .end_benchmark_sample = 8,
      .scenario_coordinate = coordinate,
      .work_item_index = 4,
      .next_alias_index = IREE_BENCHMARK_LOOM_WORK_PLAN_INDEX_INVALID,
  };
  iree_benchmark_loom_work_item_t work_item = {
      .kind = IREE_BENCHMARK_LOOM_WORK_ITEM_SCENARIO_TRIAL,
      .work_item_index = 4,
      .hal_compile_item_index = IREE_BENCHMARK_LOOM_INDEX_INVALID,
      .begin_benchmark_sample = 7,
      .end_benchmark_sample = 8,
      .scenario_coordinate = coordinate,
  };
  iree_benchmark_loom_work_plan_t work_plan = {
      .selected_benchmarks = &selection,
      .selected_benchmark_count = 1,
      .logical_samples = &logical_sample,
      .logical_sample_count = 1,
      .work_items = &work_item,
      .work_item_count = 1,
  };
  loom_module_t module = {};
  iree_benchmark_loom_artifact_bundle_t bundle = {};

  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_run(
      &event_sink, &run, /*dry_run=*/true, &kNoSanitizer));
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_work_plan(
      &event_sink, &run, &module, &work_plan));
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_summary(
      &event_sink, &run, &bundle, /*planned_case_count=*/0,
      /*planned_benchmark_count=*/1, /*selected_benchmark_count=*/1,
      /*logical_sample_count=*/1, /*work_item_count=*/1,
      /*failure_count=*/0, /*failed_benchmark_count=*/0,
      /*correctness_sample_count=*/0, /*correctness_failed_sample_count=*/0,
      /*dry_run=*/true));

  iree_string_builder_t output;
  iree_string_builder_initialize(allocator, &output);
  iree_string_view_t root = ParseJsonDocument(SnapshotJson(&snapshot, &output));
  iree_string_view_t work_items = LookupObject(root, IREE_SV("work_items"));
  iree_string_view_t first_work_item = FirstArrayElement(work_items);
  EXPECT_TRUE(
      iree_string_view_equal(LookupObject(first_work_item, IREE_SV("kind")),
                             IREE_SV("scenario_trial")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(first_work_item, IREE_SV("scenario")), IREE_SV("scenario")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(first_work_item, IREE_SV("configuration_ordinal")),
      IREE_SV("1")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(first_work_item, IREE_SV("trial_index")), IREE_SV("2")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(first_work_item, IREE_SV("trial_ordinal")), IREE_SV("3")));

  iree_string_builder_deinitialize(&output);
  iree_benchmark_loom_snapshot_sink_deinitialize(&snapshot);
}

TEST(BenchmarkSnapshotSinkTest, ReportsExecutedScenarioAliases) {
  iree_benchmark_loom_work_item_t physical_work[] = {
      {.work_item_index = 4, .representative_selection_index = 1},
      {.work_item_index = 2, .representative_selection_index = 0},
  };
  iree_allocator_t allocator = iree_allocator_system();
  iree_benchmark_loom_snapshot_sink_t snapshot = {};
  IREE_ASSERT_OK(
      iree_benchmark_loom_snapshot_sink_initialize(allocator, &snapshot));
  iree_benchmark_loom_event_sink_t event_sink = {};
  iree_benchmark_loom_snapshot_event_sink_initialize(&snapshot, &event_sink);

  iree_benchmark_loom_run_identity_t run = {.run_id = IREE_SV("run")};
  loom_module_t module = {};
  const loom_testbench_benchmark_plan_t benchmarks[] = {
      {.name = IREE_SV("first")},
      {.name = IREE_SV("second")},
      {.name = IREE_SV("alias")},
  };
  loom_testbench_scenario_plan_t scenario_plan = {.name = IREE_SV("scenario")};
  const iree_benchmark_loom_candidate_identity_t candidates[] = {
      {.candidate_id = IREE_SV("c0"), .candidate_index = 0},
      {.candidate_id = IREE_SV("c1"), .candidate_index = 1},
      {.candidate_id = IREE_SV("c2"), .candidate_index = 2},
  };
  iree_benchmark_loom_benchmark_policy_t policy = {
      .measure = IREE_SV("dispatch_complete"),
  };
  iree_benchmark_loom_benchmark_result_t result = {
      .executed = true,
      .passed = true,
      .has_benchmark_sample_ordinal = true,
      .benchmark_sample_ordinal = 17,
      .has_scenario_coordinate = true,
      .scenario_coordinate = {2, 3, 5},
      .samples_per_iteration = 1,
  };

  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_run(
      &event_sink, &run, /*dry_run=*/false, &kNoSanitizer));
  // Physical and selection indexes differ. Aliases need not arrive adjacent
  // to their representative, nor physical items in ascending index order.
  const size_t selections[] = {1, 0, 2, 2};
  for (size_t i = 0; i < IREE_ARRAYSIZE(selections); ++i) {
    const size_t selection = selections[i];
    result.benchmark_sample_ordinal = 17 + i % 2;
    result.scenario_coordinate.trial_ordinal = 5 + i % 2;
    IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_benchmark_result(
        &event_sink, &run, &candidates[selection], &physical_work[i % 2],
        &module, &benchmarks[selection], /*case_plan=*/nullptr, &scenario_plan,
        &policy, &result, /*correctness_sample_count=*/0,
        /*correctness_failed_sample_count=*/0));
  }
  // The snapshot owns its rendered evidence, not these borrowed work items.
  physical_work[0].work_item_index = 99;
  physical_work[1].work_item_index = 98;
  iree_benchmark_loom_artifact_bundle_t bundle = {};
  IREE_ASSERT_OK(iree_benchmark_loom_event_sink_emit_summary(
      &event_sink, &run, &bundle, /*planned_case_count=*/0,
      /*planned_benchmark_count=*/3, /*selected_benchmark_count=*/3,
      /*logical_sample_count=*/4, /*work_item_count=*/2,
      /*failure_count=*/0, /*failed_benchmark_count=*/0,
      /*correctness_sample_count=*/0, /*correctness_failed_sample_count=*/0,
      /*dry_run=*/false));

  iree_string_builder_t output;
  iree_string_builder_initialize(allocator, &output);
  const iree_string_view_t root =
      ParseJsonDocument(SnapshotJson(&snapshot, &output));
  const iree_string_view_t work_items =
      LookupObject(root, IREE_SV("work_items"));
  iree_host_size_t work_item_count = 0;
  IREE_ASSERT_OK(iree_json_array_length(work_items, &work_item_count));
  EXPECT_EQ(work_item_count, 2u);
  const iree_string_view_t work_item = FirstArrayElement(work_items);
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(work_item, IREE_SV("scenario")), IREE_SV("scenario")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(work_item, IREE_SV("benchmark_sample_index")),
      IREE_SV("17")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(work_item, IREE_SV("configuration_ordinal")), IREE_SV("2")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(work_item, IREE_SV("trial_index")), IREE_SV("3")));
  EXPECT_TRUE(iree_string_view_equal(
      LookupObject(work_item, IREE_SV("trial_ordinal")), IREE_SV("5")));
  const iree_string_view_t benchmark_rows =
      LookupObject(root, IREE_SV("benchmarks"));
  iree_host_size_t benchmark_count = 0;
  IREE_ASSERT_OK(iree_json_array_length(benchmark_rows, &benchmark_count));
  EXPECT_EQ(benchmark_count, 4u);
  for (size_t i = 0; i < IREE_ARRAYSIZE(selections); ++i) {
    iree_string_view_t row = iree_string_view_empty();
    IREE_ASSERT_OK(iree_json_array_get(benchmark_rows, i, &row));
    EXPECT_TRUE(
        iree_string_view_equal(LookupObject(row, IREE_SV("candidate_id")),
                               candidates[selections[i]].candidate_id));
    EXPECT_TRUE(
        iree_string_view_equal(LookupObject(row, IREE_SV("work_item_index")),
                               i % 2 == 0 ? IREE_SV("4") : IREE_SV("2")));
    EXPECT_TRUE(iree_string_view_is_empty(
        TryLookupObject(row, IREE_SV("measurement"))));
    if (i < work_item_count) {
      IREE_ASSERT_OK(iree_json_array_get(work_items, i, &row));
      EXPECT_TRUE(
          iree_string_view_equal(LookupObject(row, IREE_SV("work_item_index")),
                                 i == 0 ? IREE_SV("4") : IREE_SV("2")));
      EXPECT_TRUE(
          iree_string_view_equal(LookupObject(row, IREE_SV("benchmark")),
                                 benchmarks[selections[i]].name));
      EXPECT_FALSE(
          iree_string_view_is_empty(LookupObject(row, IREE_SV("measurement"))));
    }
  }

  iree_string_builder_deinitialize(&output);
  iree_benchmark_loom_snapshot_sink_deinitialize(&snapshot);
}

}  // namespace
}  // namespace loom
