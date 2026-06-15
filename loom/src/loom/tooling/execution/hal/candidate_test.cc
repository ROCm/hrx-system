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
      /*.data=*/&kFakeHalTarget,
      /*.target_storage=*/{},
      /*.target_bundle=*/&kFakeTargetBundle,
      /*.target_key=*/IREE_SVL("fake-hal"),
  };
  return iree_ok_status();
}

iree_status_t FakeHalEmitArtifact(
    const loom_run_hal_artifact_provider_t* provider, loom_module_t* module,
    const loom_run_hal_device_target_t* target,
    loom_diagnostic_sink_t diagnostic_sink,
    loom_source_resolver_t source_resolver, uint32_t max_errors,
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
  *out_emitted = false;
  if (g_fake_hal_expect_module_target && target->data != NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected module target emission");
  }
  if (!g_fake_hal_expect_module_target && target->data != &kFakeHalTarget) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unexpected fake HAL target payload");
  }
  *out_artifact = (loom_run_hal_artifact_t){
      /*.executable_format=*/IREE_SVL("fake-hal-format"),
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
  *artifact = {};
}

const loom_run_hal_artifact_provider_t kFakeHalArtifactProvider = {
    /*.name=*/IREE_SVL("fake-hal"),
    /*.hal_driver_name=*/IREE_SVL("fake"),
    /*.target_family_name=*/IREE_SVL("fake"),
    /*.default_pipeline_options=*/{},
    /*.select_device_target=*/FakeHalSelectDeviceTarget,
    /*.select_target_key=*/{},
    /*.deinitialize_device_target=*/{},
    /*.emit_artifact=*/FakeHalEmitArtifact,
    /*.deinitialize_artifact=*/FakeHalDeinitializeArtifact,
};

class HalCandidateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_fake_hal_emit_was_called = false;
    g_fake_hal_expect_module_target = false;
    g_fake_hal_emit_report = nullptr;
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

  loom_run_hal_candidate_t candidate = {};
  g_fake_hal_emit_was_called = false;
  g_fake_hal_expect_module_target = false;
  g_fake_hal_emit_report = nullptr;
  IREE_ASSERT_OK(loom_run_hal_candidate_compile(
      &kFakeHalArtifactProvider, FakeHalRuntime(), &run_module, &options,
      iree_allocator_system(), &candidate));
  EXPECT_TRUE(g_fake_hal_emit_was_called);
  EXPECT_EQ(g_fake_hal_emit_report, &candidate.compile_report);
  EXPECT_EQ(candidate.provider, &kFakeHalArtifactProvider);
  EXPECT_EQ(candidate.device_target.data, &kFakeHalTarget);
  EXPECT_EQ(candidate.device_target.target_bundle, &kFakeTargetBundle);
  EXPECT_EQ(candidate.artifact.target_bundle, &kFakeTargetBundle);
  EXPECT_TRUE(iree_string_view_equal(candidate.artifact.executable_format,
                                     IREE_SV("fake-hal-format")));
  EXPECT_EQ(candidate.artifact.target_artifact_format,
            LOOM_TARGET_ARTIFACT_FORMAT_ELF);
  EXPECT_EQ(candidate.artifact.target_artifact_data.data,
            kFakeHalTargetArtifactData);
  EXPECT_EQ(candidate.artifact.target_artifact_data.data_length,
            sizeof(kFakeHalTargetArtifactData));
  EXPECT_EQ(candidate.artifact.executable_data.data, kFakeHalExecutableData);
  EXPECT_EQ(candidate.artifact.executable_data.data_length,
            sizeof(kFakeHalExecutableData));
  EXPECT_EQ(candidate.compile_report.artifact_kind,
            LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_EXECUTABLE);
  EXPECT_EQ(candidate.compile_report.status_code, IREE_STATUS_OK);
  EXPECT_TRUE(iree_string_view_equal(candidate.compile_report.backend_name,
                                     IREE_SV("fake-hal")));
  EXPECT_TRUE(iree_string_view_equal(candidate.compile_report.target_key,
                                     IREE_SV("fake-hal")));
  EXPECT_TRUE(iree_string_view_equal(candidate.compile_report.executable_format,
                                     IREE_SV("fake-hal-format")));
  EXPECT_EQ(candidate.compile_report.artifact_size,
            sizeof(kFakeHalExecutableData));
  EXPECT_EQ(report.artifact_size, candidate.compile_report.artifact_size);

  loom_run_hal_candidate_deinitialize(&candidate);
  EXPECT_EQ(candidate.provider, nullptr);
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
  EXPECT_TRUE(iree_string_view_equal(candidate.artifact.executable_format,
                                     IREE_SV("fake-hal-format")));
  EXPECT_TRUE(iree_string_view_is_empty(candidate.compile_report.target_key));
  EXPECT_TRUE(iree_string_view_equal(candidate.compile_report.executable_format,
                                     IREE_SV("fake-hal-format")));
  EXPECT_EQ(report.artifact_size, sizeof(kFakeHalExecutableData));

  g_fake_hal_expect_module_target = false;
  loom_run_hal_candidate_deinitialize(&candidate);
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
