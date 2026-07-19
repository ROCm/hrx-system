// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal/candidate.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ops/func/ops.h"
#include "loom/target/low_descriptor_registry_core_test.h"

namespace loom {
namespace {

using DialectVtablesFn = const loom_op_vtable_t* const* (*)(iree_host_size_t*);

iree_status_t RegisterDialect(loom_context_t* context, uint8_t dialect_id,
                              DialectVtablesFn dialect_vtables_fn) {
  iree_host_size_t count = 0;
  const loom_op_vtable_t* const* vtables = dialect_vtables_fn(&count);
  return loom_context_register_dialect(context, dialect_id, vtables,
                                       (uint16_t)count);
}

iree_status_t RegisterContext(void* user_data, loom_context_t* context) {
  (void)user_data;
  return RegisterDialect(context, LOOM_DIALECT_FUNC, loom_func_dialect_vtables);
}

iree_status_t InitializeLowDescriptorRegistry(
    void* user_data, loom_target_low_descriptor_registry_t* out_registry) {
  (void)user_data;
  loom_target_core_test_low_descriptor_registry_initialize(out_registry);
  return iree_ok_status();
}

constexpr char kHalSource[] =
    "func.def @empty() {\n"
    "  func.return\n"
    "}\n";

int kFakeHalTarget = 0;
int kFakeHalRuntime = 0;
bool g_fake_hal_emit_was_called = false;
bool g_fake_hal_expect_module_target = false;
loom_target_compile_report_t* g_fake_hal_emit_report = nullptr;
const loom_target_pipeline_options_t* g_fake_hal_emit_target_pipeline_options =
    nullptr;
enum class FakeHalEmitOutcome {
  kArtifact,
  kNoArtifact,
  kFailure,
};
FakeHalEmitOutcome g_fake_hal_emit_outcome = FakeHalEmitOutcome::kArtifact;
int g_fake_hal_deinitialize_target_count = 0;
int g_fake_hal_deinitialize_artifact_count = 0;
const uint8_t kFakeHalExecutableData[] = {0x7F, 'E', 'L', 'F'};
const uint8_t kFakeHalTargetArtifactData[] = {'h', 's', 'a', 'c', 'o'};
static const loom_target_snapshot_t kFakeSnapshot = {
    /*.name=*/IREE_SVL("fake-snapshot"),
};
static const loom_target_export_plan_t kFakeExportPlan = {
    /*.name=*/IREE_SVL("fake-export"),
    /*.export_symbol=*/{},
    /*.abi_kind=*/LOOM_TARGET_ABI_HAL_KERNEL,
};
static const loom_target_bundle_t kFakeTargetBundle = {
    /*.name=*/IREE_SVL("fake-bundle"),
    /*.snapshot=*/&kFakeSnapshot,
    /*.export_plan=*/&kFakeExportPlan,
};

struct FakeArtifactAnalyzerState {
  int call_count = 0;
  bool reject = false;
  loom_target_artifact_format_t artifact_format =
      LOOM_TARGET_ARTIFACT_FORMAT_UNKNOWN;
  iree_const_byte_span_t artifact_data = {};
  iree_string_view_t target_key = {};
  const loom_target_bundle_t* target_bundle = nullptr;
  loom_diagnostic_sink_t diagnostic_sink = {};
  uint32_t max_errors = 0;
  bool report_present = false;
};

iree_status_t FakeAnalyzeArtifact(
    void* user_data, const loom_run_artifact_analysis_request_t* request) {
  auto* state = static_cast<FakeArtifactAnalyzerState*>(user_data);
  ++state->call_count;
  state->artifact_format = request->artifact_format;
  state->artifact_data = request->artifact_data;
  state->target_key = request->target_key;
  state->target_bundle = request->target_bundle;
  state->diagnostic_sink = request->diagnostic_sink;
  state->max_errors = request->max_errors;
  state->report_present = request->report != nullptr;
  if (request->report != nullptr) {
    const loom_target_compile_report_artifact_analysis_t analysis = {
        /*.analyzer_name=*/IREE_SVL("fake-artifact-analyzer"),
        /*.analyzer_abi_version=*/7,
        /*.outcome=*/
        state->reject
            ? LOOM_TARGET_COMPILE_REPORT_ARTIFACT_ANALYSIS_OUTCOME_REJECTED
            : LOOM_TARGET_COMPILE_REPORT_ARTIFACT_ANALYSIS_OUTCOME_CLEAN,
        /*.artifact_target=*/request->target_key,
        /*.instruction_count=*/5,
        /*.memory_event_count=*/2,
        /*.function_count=*/1,
        /*.analyzed_function_count=*/1,
        /*.finding_count=*/state->reject ? 1u : 0u,
        /*.reported_finding_count=*/state->reject ? 1u : 0u,
        /*.findings_truncated=*/false,
        /*.stopped_early=*/false,
        /*.complete=*/true,
        /*.passed=*/!state->reject,
    };
    loom_target_compile_report_record_artifact_analysis(request->report,
                                                        &analysis);
  }
  if (state->reject) {
    return iree_make_status(IREE_STATUS_ABORTED,
                            "fake final-artifact rejection");
  }
  return iree_ok_status();
}

loom_run_artifact_analyzer_t FakeArtifactAnalyzer(
    FakeArtifactAnalyzerState* state) {
  return (loom_run_artifact_analyzer_t){
      /*.fn=*/FakeAnalyzeArtifact,
      /*.user_data=*/state,
  };
}

const loom_run_hal_runtime_t* FakeHalRuntime() {
  return reinterpret_cast<const loom_run_hal_runtime_t*>(&kFakeHalRuntime);
}

iree_status_t FakeHalSelectDeviceTarget(
    const loom_run_hal_artifact_provider_t* provider,
    const loom_run_hal_runtime_t* runtime, iree_allocator_t allocator,
    loom_run_hal_device_target_t* out_target) {
  (void)provider;
  (void)runtime;
  (void)allocator;
  *out_target = (loom_run_hal_device_target_t){
      /*.hal_target=*/nullptr,
      /*.data=*/&kFakeHalTarget,
      /*.target_storage=*/{},
      /*.target_bundle=*/&kFakeTargetBundle,
      /*.target_key=*/IREE_SVL("fake-hal"),
  };
  return iree_ok_status();
}

void FakeHalDeinitializeDeviceTarget(
    const loom_run_hal_artifact_provider_t* provider,
    loom_run_hal_device_target_t* target, iree_allocator_t allocator) {
  (void)provider;
  (void)allocator;
  ++g_fake_hal_deinitialize_target_count;
  *target = {};
}

iree_status_t FakeHalEmitArtifact(
    const loom_run_hal_artifact_provider_t* provider, loom_module_t* module,
    const loom_run_hal_device_target_t* target,
    loom_diagnostic_sink_t diagnostic_sink,
    loom_source_resolver_t source_resolver, uint32_t max_errors,
    const loom_target_pipeline_options_t* target_pipeline_options,
    loom_run_candidate_artifact_flags_t artifact_flags,
    const loom_run_candidate_artifact_manifest_options_t* artifact_manifest,
    loom_target_compile_report_t* report, iree_allocator_t allocator,
    bool* out_emitted, loom_run_hal_artifact_t* out_artifact) {
  (void)provider;
  (void)module;
  (void)diagnostic_sink;
  (void)source_resolver;
  (void)max_errors;
  (void)artifact_flags;
  (void)artifact_manifest;
  (void)allocator;
  g_fake_hal_emit_was_called = true;
  g_fake_hal_emit_report = report;
  g_fake_hal_emit_target_pipeline_options = target_pipeline_options;
  *out_emitted = false;
  if (g_fake_hal_expect_module_target && target->data != NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected module target emission");
  }
  if (!g_fake_hal_expect_module_target && target->data != &kFakeHalTarget) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unexpected fake HAL target payload");
  }
  if (g_fake_hal_emit_outcome == FakeHalEmitOutcome::kFailure) {
    return iree_make_status(IREE_STATUS_ABORTED,
                            "fake HAL provider emission failure");
  }
  if (g_fake_hal_emit_outcome == FakeHalEmitOutcome::kNoArtifact) {
    return iree_ok_status();
  }
  *out_artifact = (loom_run_hal_artifact_t){
      /*.hal_target=*/nullptr,
      /*.target_key=*/IREE_SVL("fake-hal-target"),
      /*.target_bundle=*/{},
      /*.target_artifact_format=*/LOOM_TARGET_ARTIFACT_FORMAT_ELF,
      /*.target_artifact_data=*/
      iree_make_const_byte_span(kFakeHalTargetArtifactData,
                                sizeof(kFakeHalTargetArtifactData)),
      /*.target_listing_format=*/{},
      /*.target_listing_data=*/{},
      /*.sidecars=*/{},
      /*.sidecar_count=*/{},
      /*.executable_data=*/
      iree_make_const_byte_span(kFakeHalExecutableData,
                                sizeof(kFakeHalExecutableData)),
      /*.storage=*/&kFakeHalTarget,
  };
  *out_emitted = true;
  return iree_ok_status();
}

void FakeHalDeinitializeArtifact(
    const loom_run_hal_artifact_provider_t* provider,
    loom_run_hal_artifact_t* artifact, iree_allocator_t allocator) {
  (void)provider;
  (void)allocator;
  ++g_fake_hal_deinitialize_artifact_count;
  *artifact = {};
}

const loom_run_hal_artifact_provider_t kFakeHalArtifactProvider = {
    /*.name=*/IREE_SVL("fake-hal"),
    /*.hal_driver_name=*/IREE_SVL("fake"),
    /*.target_family_name=*/IREE_SVL("fake"),
    /*.default_pipeline_options=*/{},
    /*.select_device_target=*/FakeHalSelectDeviceTarget,
    /*.select_target_key=*/{},
    /*.deinitialize_device_target=*/FakeHalDeinitializeDeviceTarget,
    /*.emit_artifact=*/FakeHalEmitArtifact,
    /*.deinitialize_artifact=*/FakeHalDeinitializeArtifact,
};

class HalCandidateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_fake_hal_emit_was_called = false;
    g_fake_hal_expect_module_target = false;
    g_fake_hal_emit_report = nullptr;
    g_fake_hal_emit_target_pipeline_options = nullptr;
    g_fake_hal_emit_outcome = FakeHalEmitOutcome::kArtifact;
    g_fake_hal_deinitialize_target_count = 0;
    g_fake_hal_deinitialize_artifact_count = 0;
    loom_run_session_options_t options = {};
    loom_run_session_options_initialize(&options);
    options.register_context = (loom_run_register_context_callback_t){
        /*.fn=*/RegisterContext,
    };
    options.initialize_low_descriptor_registry =
        (loom_run_initialize_low_descriptor_registry_callback_t){
            /*.fn=*/InitializeLowDescriptorRegistry,
        };
    IREE_ASSERT_OK(loom_run_session_initialize(&options, &session_));
  }

  void TearDown() override { loom_run_session_deinitialize(&session_); }

  iree_status_t Parse(iree_string_view_t source,
                      loom_run_module_t* out_module) {
    loom_run_module_parse_options_t options = {};
    loom_run_module_parse_options_initialize(&options);
    options.filename = IREE_SV("hal_candidate_test.loom");
    options.source = source;
    return loom_run_module_parse(&session_, &options, out_module);
  }

  void InitializeCompileOptions(
      loom_run_module_t* run_module,
      loom_run_candidate_compile_options_t* out_options) {
    loom_run_candidate_compile_options_initialize(out_options);
    out_options->source_resolver = loom_run_module_source_resolver(run_module);
  }

  loom_run_session_t session_ = {};
};

TEST_F(HalCandidateTest, CompileHalExecutableCandidate) {
  loom_run_module_t run_module = {};
  IREE_ASSERT_OK(Parse(IREE_SV(kHalSource), &run_module));

  loom_run_candidate_compile_options_t options = {};
  InitializeCompileOptions(&run_module, &options);
  loom_target_compile_report_t report = {};
  options.report = &report;
  FakeArtifactAnalyzerState analyzer_state;
  const loom_run_artifact_analyzer_t analyzer =
      FakeArtifactAnalyzer(&analyzer_state);
  options.artifact_analyzer = &analyzer;

  loom_run_hal_candidate_t candidate = {};
  g_fake_hal_emit_was_called = false;
  g_fake_hal_expect_module_target = false;
  g_fake_hal_emit_report = nullptr;
  IREE_ASSERT_OK(loom_run_hal_candidate_compile(
      &kFakeHalArtifactProvider, FakeHalRuntime(), &run_module, &options,
      iree_allocator_system(), &candidate));
  EXPECT_TRUE(g_fake_hal_emit_was_called);
  EXPECT_EQ(g_fake_hal_emit_report, &candidate.compile_report);
  EXPECT_EQ(g_fake_hal_emit_target_pipeline_options,
            &options.target_pipeline_options);
  EXPECT_EQ(candidate.provider, &kFakeHalArtifactProvider);
  EXPECT_EQ(candidate.device_target.data, &kFakeHalTarget);
  EXPECT_EQ(candidate.device_target.target_bundle, &kFakeTargetBundle);
  EXPECT_EQ(candidate.artifact.target_bundle, &kFakeTargetBundle);
  EXPECT_TRUE(iree_string_view_equal(candidate.artifact.target_key,
                                     IREE_SV("fake-hal-target")));
  EXPECT_EQ(candidate.artifact.target_artifact_format,
            LOOM_TARGET_ARTIFACT_FORMAT_ELF);
  EXPECT_EQ(candidate.artifact.target_artifact_data.data,
            kFakeHalTargetArtifactData);
  EXPECT_EQ(candidate.artifact.target_artifact_data.data_length,
            sizeof(kFakeHalTargetArtifactData));
  EXPECT_EQ(candidate.artifact.executable_data.data, kFakeHalExecutableData);
  EXPECT_EQ(candidate.artifact.executable_data.data_length,
            sizeof(kFakeHalExecutableData));
  EXPECT_EQ(analyzer_state.call_count, 1);
  EXPECT_EQ(analyzer_state.artifact_format, LOOM_TARGET_ARTIFACT_FORMAT_ELF);
  EXPECT_EQ(analyzer_state.artifact_data.data, kFakeHalTargetArtifactData);
  EXPECT_EQ(analyzer_state.artifact_data.data_length,
            sizeof(kFakeHalTargetArtifactData));
  EXPECT_TRUE(iree_string_view_equal(analyzer_state.target_key,
                                     IREE_SV("fake-hal-target")));
  EXPECT_EQ(analyzer_state.target_bundle, &kFakeTargetBundle);
  EXPECT_EQ(analyzer_state.diagnostic_sink.fn, options.diagnostic_sink.fn);
  EXPECT_EQ(analyzer_state.diagnostic_sink.user_data,
            options.diagnostic_sink.user_data);
  EXPECT_EQ(analyzer_state.max_errors, options.max_errors);
  EXPECT_TRUE(analyzer_state.report_present);
  EXPECT_EQ(candidate.compile_report.artifact_kind,
            LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_EXECUTABLE);
  EXPECT_EQ(candidate.compile_report.status_code, IREE_STATUS_OK);
  EXPECT_TRUE(iree_string_view_equal(candidate.compile_report.backend_name,
                                     IREE_SV("fake-hal")));
  EXPECT_TRUE(iree_string_view_equal(candidate.compile_report.target_key,
                                     IREE_SV("fake-hal")));
  EXPECT_TRUE(iree_string_view_equal(candidate.compile_report.artifact_format,
                                     IREE_SV("elf")));
  EXPECT_EQ(candidate.compile_report.artifact_size,
            sizeof(kFakeHalExecutableData));
  EXPECT_EQ(report.artifact_size, candidate.compile_report.artifact_size);
  EXPECT_TRUE(
      iree_any_bit_set(report.detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_ARTIFACT_ANALYSIS));
  EXPECT_TRUE(iree_string_view_equal(report.artifact_analysis.analyzer_name,
                                     IREE_SV("fake-artifact-analyzer")));
  EXPECT_EQ(report.artifact_analysis.analyzer_abi_version, 7u);
  EXPECT_EQ(report.artifact_analysis.outcome,
            LOOM_TARGET_COMPILE_REPORT_ARTIFACT_ANALYSIS_OUTCOME_CLEAN);
  EXPECT_TRUE(iree_string_view_equal(report.artifact_analysis.artifact_target,
                                     IREE_SV("fake-hal-target")));
  EXPECT_EQ(report.artifact_analysis.instruction_count, 5u);
  EXPECT_EQ(report.artifact_analysis.memory_event_count, 2u);
  EXPECT_EQ(report.artifact_analysis.function_count, 1u);
  EXPECT_EQ(report.artifact_analysis.analyzed_function_count, 1u);
  EXPECT_EQ(report.artifact_analysis.finding_count, 0u);
  EXPECT_EQ(report.artifact_analysis.reported_finding_count, 0u);
  EXPECT_TRUE(report.artifact_analysis.complete);
  EXPECT_TRUE(report.artifact_analysis.passed);

  loom_run_hal_candidate_deinitialize(&candidate);
  EXPECT_EQ(candidate.provider, nullptr);
  EXPECT_EQ(report.artifact_analysis.outcome,
            LOOM_TARGET_COMPILE_REPORT_ARTIFACT_ANALYSIS_OUTCOME_CLEAN);
  loom_target_compile_report_deinitialize(&report);
  loom_run_module_deinitialize(&run_module);
}

TEST_F(HalCandidateTest, CompileHalExecutableCandidateWithoutReport) {
  loom_run_module_t run_module = {};
  IREE_ASSERT_OK(Parse(IREE_SV(kHalSource), &run_module));

  loom_run_candidate_compile_options_t options = {};
  InitializeCompileOptions(&run_module, &options);

  loom_run_hal_candidate_t candidate = {};
  g_fake_hal_emit_was_called = false;
  g_fake_hal_expect_module_target = false;
  g_fake_hal_emit_report = nullptr;
  IREE_ASSERT_OK(loom_run_hal_candidate_compile(
      &kFakeHalArtifactProvider, FakeHalRuntime(), &run_module, &options,
      iree_allocator_system(), &candidate));
  EXPECT_TRUE(g_fake_hal_emit_was_called);
  EXPECT_EQ(g_fake_hal_emit_report, nullptr);
  EXPECT_EQ(candidate.compile_report.detail_flags,
            LOOM_TARGET_COMPILE_REPORT_DETAIL_NONE);
  EXPECT_EQ(candidate.compile_report.artifact_size, 0u);

  loom_run_hal_candidate_deinitialize(&candidate);
  loom_run_module_deinitialize(&run_module);
}

TEST_F(HalCandidateTest, EmitsModuleTargetCandidate) {
  loom_run_module_t run_module = {};
  IREE_ASSERT_OK(Parse(IREE_SV(kHalSource), &run_module));

  loom_run_candidate_compile_options_t options = {};
  InitializeCompileOptions(&run_module, &options);
  loom_target_compile_report_t report = {};
  options.report = &report;
  FakeArtifactAnalyzerState analyzer_state;
  const loom_run_artifact_analyzer_t analyzer =
      FakeArtifactAnalyzer(&analyzer_state);
  options.artifact_analyzer = &analyzer;

  loom_run_hal_candidate_t candidate = {};
  g_fake_hal_emit_was_called = false;
  g_fake_hal_expect_module_target = true;
  g_fake_hal_emit_report = nullptr;
  IREE_ASSERT_OK(loom_run_hal_candidate_emit_module_target(
      &kFakeHalArtifactProvider, &run_module, &options, iree_allocator_system(),
      &candidate));
  EXPECT_TRUE(g_fake_hal_emit_was_called);
  EXPECT_EQ(g_fake_hal_emit_report, &candidate.compile_report);
  EXPECT_EQ(candidate.provider, &kFakeHalArtifactProvider);
  EXPECT_EQ(candidate.device_target.data, nullptr);
  EXPECT_EQ(candidate.device_target.target_bundle, nullptr);
  EXPECT_EQ(candidate.artifact.target_bundle, nullptr);
  EXPECT_EQ(analyzer_state.call_count, 1);
  EXPECT_EQ(analyzer_state.artifact_data.data, kFakeHalTargetArtifactData);
  EXPECT_EQ(analyzer_state.target_bundle, nullptr);
  EXPECT_TRUE(analyzer_state.report_present);
  EXPECT_TRUE(iree_string_view_equal(candidate.artifact.target_key,
                                     IREE_SV("fake-hal-target")));
  EXPECT_TRUE(iree_string_view_is_empty(candidate.compile_report.target_key));
  EXPECT_TRUE(iree_string_view_equal(candidate.compile_report.artifact_format,
                                     IREE_SV("elf")));
  EXPECT_EQ(report.artifact_size, sizeof(kFakeHalExecutableData));

  g_fake_hal_expect_module_target = false;
  loom_run_hal_candidate_deinitialize(&candidate);
  loom_target_compile_report_deinitialize(&report);
  loom_run_module_deinitialize(&run_module);
}

TEST_F(HalCandidateTest, EmitsExplicitTargetCandidate) {
  loom_run_module_t run_module = {};
  IREE_ASSERT_OK(Parse(IREE_SV(kHalSource), &run_module));

  loom_run_candidate_compile_options_t options = {};
  InitializeCompileOptions(&run_module, &options);
  FakeArtifactAnalyzerState analyzer_state;
  const loom_run_hal_device_target_t target = {
      /*.hal_target=*/nullptr,
      /*.data=*/&kFakeHalTarget,
      /*.target_storage=*/{},
      /*.target_bundle=*/&kFakeTargetBundle,
      /*.target_key=*/IREE_SVL("explicit-fake-hal"),
  };

  loom_run_hal_candidate_t candidate = {};
  {
    const loom_run_artifact_analyzer_t analyzer =
        FakeArtifactAnalyzer(&analyzer_state);
    options.artifact_analyzer = &analyzer;
    IREE_ASSERT_OK(loom_run_hal_candidate_emit_target(
        &kFakeHalArtifactProvider, &target, &run_module, &options,
        iree_allocator_system(), &candidate));
    EXPECT_TRUE(g_fake_hal_emit_was_called);
    EXPECT_FALSE(candidate.owns_device_target);
    EXPECT_EQ(candidate.artifact.target_bundle, &kFakeTargetBundle);
    EXPECT_EQ(analyzer_state.call_count, 1);
    EXPECT_EQ(analyzer_state.artifact_format, LOOM_TARGET_ARTIFACT_FORMAT_ELF);
    EXPECT_EQ(analyzer_state.artifact_data.data, kFakeHalTargetArtifactData);
    EXPECT_EQ(analyzer_state.target_bundle, &kFakeTargetBundle);
    EXPECT_FALSE(analyzer_state.report_present);
  }

  loom_run_hal_candidate_deinitialize(&candidate);
  EXPECT_EQ(analyzer_state.call_count, 1);
  EXPECT_EQ(g_fake_hal_deinitialize_target_count, 0);
  EXPECT_EQ(g_fake_hal_deinitialize_artifact_count, 1);
  loom_run_module_deinitialize(&run_module);
}

TEST_F(HalCandidateTest, SkipsAnalyzerWhenProviderEmitsNoArtifact) {
  loom_run_module_t run_module = {};
  IREE_ASSERT_OK(Parse(IREE_SV(kHalSource), &run_module));

  loom_run_candidate_compile_options_t options = {};
  InitializeCompileOptions(&run_module, &options);
  FakeArtifactAnalyzerState analyzer_state;
  const loom_run_artifact_analyzer_t analyzer =
      FakeArtifactAnalyzer(&analyzer_state);
  options.artifact_analyzer = &analyzer;
  g_fake_hal_emit_outcome = FakeHalEmitOutcome::kNoArtifact;

  loom_run_hal_candidate_t candidate = {};
  IREE_ASSERT_OK(loom_run_hal_candidate_compile(
      &kFakeHalArtifactProvider, FakeHalRuntime(), &run_module, &options,
      iree_allocator_system(), &candidate));
  EXPECT_TRUE(g_fake_hal_emit_was_called);
  EXPECT_FALSE(candidate.compiled);
  EXPECT_EQ(analyzer_state.call_count, 0);

  loom_run_hal_candidate_deinitialize(&candidate);
  loom_run_module_deinitialize(&run_module);
}

TEST_F(HalCandidateTest, SkipsAnalyzerWhenProviderFails) {
  loom_run_module_t run_module = {};
  IREE_ASSERT_OK(Parse(IREE_SV(kHalSource), &run_module));

  loom_run_candidate_compile_options_t options = {};
  InitializeCompileOptions(&run_module, &options);
  loom_target_compile_report_t report = {};
  options.report = &report;
  FakeArtifactAnalyzerState analyzer_state;
  const loom_run_artifact_analyzer_t analyzer =
      FakeArtifactAnalyzer(&analyzer_state);
  options.artifact_analyzer = &analyzer;
  g_fake_hal_emit_outcome = FakeHalEmitOutcome::kFailure;

  loom_run_hal_candidate_t candidate = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      loom_run_hal_candidate_compile(&kFakeHalArtifactProvider,
                                     FakeHalRuntime(), &run_module, &options,
                                     iree_allocator_system(), &candidate));
  EXPECT_EQ(analyzer_state.call_count, 0);
  EXPECT_EQ(candidate.provider, nullptr);
  EXPECT_EQ(report.status_code, IREE_STATUS_ABORTED);
  EXPECT_EQ(g_fake_hal_deinitialize_target_count, 1);
  EXPECT_EQ(g_fake_hal_deinitialize_artifact_count, 1);

  loom_target_compile_report_deinitialize(&report);
  loom_run_module_deinitialize(&run_module);
}

TEST_F(HalCandidateTest, AnalyzerRejectionDiscardsCandidate) {
  loom_run_module_t run_module = {};
  IREE_ASSERT_OK(Parse(IREE_SV(kHalSource), &run_module));

  loom_run_candidate_compile_options_t options = {};
  InitializeCompileOptions(&run_module, &options);
  loom_target_compile_report_t report = {};
  options.report = &report;
  FakeArtifactAnalyzerState analyzer_state;
  analyzer_state.reject = true;
  const loom_run_artifact_analyzer_t analyzer =
      FakeArtifactAnalyzer(&analyzer_state);
  options.artifact_analyzer = &analyzer;

  loom_run_hal_candidate_t candidate = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      loom_run_hal_candidate_compile(&kFakeHalArtifactProvider,
                                     FakeHalRuntime(), &run_module, &options,
                                     iree_allocator_system(), &candidate));
  EXPECT_EQ(analyzer_state.call_count, 1);
  EXPECT_EQ(analyzer_state.artifact_data.data, kFakeHalTargetArtifactData);
  EXPECT_EQ(analyzer_state.target_bundle, &kFakeTargetBundle);
  EXPECT_TRUE(analyzer_state.report_present);
  EXPECT_EQ(candidate.provider, nullptr);
  EXPECT_FALSE(candidate.compiled);
  EXPECT_EQ(candidate.artifact.storage, nullptr);
  EXPECT_EQ(report.status_code, IREE_STATUS_ABORTED);
  EXPECT_EQ(report.artifact_size, sizeof(kFakeHalExecutableData));
  EXPECT_TRUE(
      iree_any_bit_set(report.detail_flags,
                       LOOM_TARGET_COMPILE_REPORT_DETAIL_ARTIFACT_ANALYSIS));
  EXPECT_EQ(report.artifact_analysis.outcome,
            LOOM_TARGET_COMPILE_REPORT_ARTIFACT_ANALYSIS_OUTCOME_REJECTED);
  EXPECT_EQ(report.artifact_analysis.finding_count, 1u);
  EXPECT_EQ(report.artifact_analysis.reported_finding_count, 1u);
  EXPECT_TRUE(report.artifact_analysis.complete);
  EXPECT_FALSE(report.artifact_analysis.passed);
  EXPECT_EQ(g_fake_hal_deinitialize_target_count, 1);
  EXPECT_EQ(g_fake_hal_deinitialize_artifact_count, 1);

  loom_target_compile_report_deinitialize(&report);
  loom_run_module_deinitialize(&run_module);
}

TEST_F(HalCandidateTest, RejectsAnalyzerWithoutCallback) {
  loom_run_module_t run_module = {};
  IREE_ASSERT_OK(Parse(IREE_SV(kHalSource), &run_module));

  loom_run_candidate_compile_options_t options = {};
  InitializeCompileOptions(&run_module, &options);
  const loom_run_artifact_analyzer_t analyzer = {};
  options.artifact_analyzer = &analyzer;

  loom_run_hal_candidate_t candidate = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_run_hal_candidate_compile(&kFakeHalArtifactProvider,
                                     FakeHalRuntime(), &run_module, &options,
                                     iree_allocator_system(), &candidate));
  EXPECT_EQ(candidate.provider, nullptr);
  EXPECT_EQ(g_fake_hal_deinitialize_target_count, 1);
  EXPECT_EQ(g_fake_hal_deinitialize_artifact_count, 1);

  loom_run_module_deinitialize(&run_module);
}

TEST_F(HalCandidateTest, CompileHalRequiresHooks) {
  loom_run_module_t run_module = {};
  IREE_ASSERT_OK(Parse(IREE_SV(kHalSource), &run_module));

  loom_run_candidate_compile_options_t options = {};
  InitializeCompileOptions(&run_module, &options);
  loom_target_compile_report_t report = {};
  options.report = &report;

  const loom_run_hal_artifact_provider_t provider = {
      /*.name=*/IREE_SVL("missing-hooks"),
  };
  loom_run_hal_candidate_t candidate = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_run_hal_candidate_compile(
                            &provider, FakeHalRuntime(), &run_module, &options,
                            iree_allocator_system(), &candidate));
  EXPECT_EQ(candidate.provider, nullptr);
  EXPECT_EQ(report.artifact_kind,
            LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_EXECUTABLE);
  EXPECT_EQ(report.status_code, IREE_STATUS_INVALID_ARGUMENT);

  loom_run_module_deinitialize(&run_module);
}

}  // namespace
}  // namespace loom
