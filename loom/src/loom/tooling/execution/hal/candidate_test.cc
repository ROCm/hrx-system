// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal/candidate.h"

#include <cstring>

#include "iree/base/byte_sequence.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ops/func/ops.h"
#include "loom/target/low_descriptor_registry_core_test.h"
#include "loom/target/profile.h"
#include "loom/testing/byte_sequence.h"

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

int kFakeHalRuntime = 0;
bool g_fake_hal_emit_was_called = false;
const loom_target_facts_t* g_fake_hal_selected_target_requirement = nullptr;
loom_target_compile_report_t* g_fake_hal_emit_report = nullptr;
const loom_function_version_list_t* g_fake_hal_emit_function_versions = nullptr;
uint32_t g_fake_hal_emit_source_to_low_max_errors = 0;
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
static const loom_target_profile_type_t kFakeTargetProfileType = {
    /*.name=*/IREE_SVL("fake"),
};
static const loom_target_profile_t kFakeTargetProfile = {
    /*.type=*/&kFakeTargetProfileType,
    /*.target_bundle=*/&kFakeTargetBundle,
};
static const loom_target_facts_t kFakeTargetRequirement = {};

typedef struct fake_hal_artifact_storage_t {
  // Immutable target-native artifact contents.
  iree_byte_sequence_t* target_artifact;
  // Immutable HAL executable contents.
  iree_byte_sequence_t* executable;
} fake_hal_artifact_storage_t;

iree_status_t CreateFakeHalArtifactSequence(
    iree_const_byte_span_t source, iree_allocator_t allocator,
    iree_byte_sequence_t** out_sequence) {
  *out_sequence = nullptr;
  void* data = nullptr;
  IREE_RETURN_IF_ERROR(iree_allocator_clone(allocator, source, &data));
  iree_byte_span_t contents = iree_make_byte_span(data, source.data_length);
  iree_status_t status = iree_byte_sequence_create_from_span_move(
      &contents, allocator, out_sequence);
  iree_allocator_free(allocator, contents.data);
  return status;
}

const loom_run_hal_runtime_t* FakeHalRuntime() {
  return reinterpret_cast<const loom_run_hal_runtime_t*>(&kFakeHalRuntime);
}

iree_status_t FakeHalSelectDeviceTarget(const loom_device_provider_t* provider,
                                        const loom_run_hal_runtime_t* runtime,
                                        iree_allocator_t allocator,
                                        loom_device_target_t* out_target) {
  (void)provider;
  (void)runtime;
  (void)allocator;
  *out_target = (loom_device_target_t){
      /*.executable_target=*/nullptr,
      /*.artifact_target=*/
      {
          /*.target_profile=*/&kFakeTargetProfile,
          /*.target_key=*/IREE_SVL("fake-hal"),
      },
  };
  return iree_ok_status();
}

iree_status_t FakeHalSelectCompatibleDeviceTarget(
    const loom_device_provider_t* provider,
    const loom_run_hal_runtime_t* runtime,
    const loom_target_facts_t* target_requirement, iree_allocator_t allocator,
    loom_device_target_t* out_target) {
  g_fake_hal_selected_target_requirement = target_requirement;
  return FakeHalSelectDeviceTarget(provider, runtime, allocator, out_target);
}

iree_status_t FakeHalEmitArtifact(const loom_artifact_provider_t* provider,
                                  loom_module_t* module,
                                  const loom_artifact_target_t* target,
                                  const loom_compile_options_t* options,
                                  iree_allocator_t allocator, bool* out_emitted,
                                  loom_artifact_t* out_artifact) {
  (void)provider;
  (void)module;
  g_fake_hal_emit_was_called = true;
  g_fake_hal_emit_report = options->report;
  g_fake_hal_emit_function_versions = options->function_versions;
  g_fake_hal_emit_source_to_low_max_errors =
      options->target_pipeline_options.source_to_low_max_errors;
  *out_emitted = false;
  if (target->target_profile != &kFakeTargetProfile) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unexpected fake HAL target profile");
  }
  fake_hal_artifact_storage_t* storage = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*storage), (void**)&storage));
  *storage = {};
  iree_status_t status = CreateFakeHalArtifactSequence(
      iree_make_const_byte_span(kFakeHalTargetArtifactData,
                                sizeof(kFakeHalTargetArtifactData)),
      allocator, &storage->target_artifact);
  if (iree_status_is_ok(status)) {
    status = CreateFakeHalArtifactSequence(
        iree_make_const_byte_span(kFakeHalExecutableData,
                                  sizeof(kFakeHalExecutableData)),
        allocator, &storage->executable);
  }
  if (!iree_status_is_ok(status)) {
    iree_byte_sequence_release(storage->target_artifact);
    iree_byte_sequence_release(storage->executable);
    iree_allocator_free(allocator, storage);
    return status;
  }
  *out_artifact = (loom_artifact_t){
      /*.target_key=*/IREE_SVL("fake-hal-target"),
      /*.target_bundle=*/&kFakeTargetBundle,
      /*.target_artifact_format=*/LOOM_TARGET_ARTIFACT_FORMAT_ELF,
      /*.target_artifact_data=*/storage->target_artifact,
      /*.target_listing_format=*/{},
      /*.target_listing_data=*/{},
      /*.sidecars=*/{},
      /*.sidecar_count=*/{},
      /*.executable_data=*/storage->executable,
      /*.storage=*/storage,
  };
  *out_emitted = true;
  return iree_ok_status();
}

void FakeHalDeinitializeArtifact(const loom_artifact_provider_t* provider,
                                 loom_artifact_t* artifact,
                                 iree_allocator_t allocator) {
  (void)provider;
  auto* storage = static_cast<fake_hal_artifact_storage_t*>(artifact->storage);
  if (storage != nullptr) {
    iree_byte_sequence_release(storage->target_artifact);
    iree_byte_sequence_release(storage->executable);
    iree_allocator_free(allocator, storage);
  }
  *artifact = {};
}

const loom_artifact_provider_t kFakeArtifactProvider = {
    /*.name=*/IREE_SVL("fake-hal"),
    /*.target_family_name=*/IREE_SVL("fake"),
    /*.artifact_kind=*/LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_EXECUTABLE,
    /*.default_pipeline_options=*/{},
    /*.select_target=*/{},
    /*.deinitialize_target=*/{},
    /*.emit_artifact=*/FakeHalEmitArtifact,
    /*.deinitialize_artifact=*/FakeHalDeinitializeArtifact,
};

const loom_device_provider_t kFakeDeviceProvider = {
    /*.artifact_provider=*/&kFakeArtifactProvider,
    /*.driver_name=*/IREE_SVL("fake"),
    /*.select_target=*/FakeHalSelectDeviceTarget,
    /*.select_compatible_target=*/FakeHalSelectCompatibleDeviceTarget,
    /*.deinitialize_target=*/{},
};

class HalCandidateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_fake_hal_emit_was_called = false;
    g_fake_hal_selected_target_requirement = nullptr;
    g_fake_hal_emit_report = nullptr;
    g_fake_hal_emit_function_versions = nullptr;
    g_fake_hal_emit_source_to_low_max_errors = 0;
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

  void InitializeCompileOptions(loom_run_module_t* run_module,
                                loom_compile_options_t* out_options) {
    loom_compile_options_initialize(out_options);
    out_options->source_resolver = loom_run_module_source_resolver(run_module);
  }

  loom_run_session_t session_ = {};
};

TEST_F(HalCandidateTest, CompileHalExecutableCandidate) {
  loom_run_module_t run_module = {};
  IREE_ASSERT_OK(Parse(IREE_SV(kHalSource), &run_module));

  loom_compile_options_t options = {};
  InitializeCompileOptions(&run_module, &options);
  const loom_function_version_list_t function_versions = {};
  options.function_versions = &function_versions;
  options.target_pipeline_options.source_to_low_max_errors = 73;
  loom_target_compile_report_t report = {};
  options.report = &report;

  loom_run_hal_candidate_t candidate = {};
  g_fake_hal_emit_was_called = false;
  g_fake_hal_emit_report = nullptr;
  IREE_ASSERT_OK(loom_run_hal_candidate_compile(
      &kFakeDeviceProvider, FakeHalRuntime(), &run_module,
      &kFakeTargetRequirement, &options, iree_allocator_system(), &candidate));
  EXPECT_EQ(g_fake_hal_selected_target_requirement, &kFakeTargetRequirement);
  EXPECT_TRUE(g_fake_hal_emit_was_called);
  EXPECT_EQ(g_fake_hal_emit_report,
            &candidate.artifact_candidate.compile_report);
  EXPECT_EQ(g_fake_hal_emit_function_versions, &function_versions);
  EXPECT_EQ(g_fake_hal_emit_source_to_low_max_errors, 73u);
  EXPECT_EQ(candidate.provider, &kFakeDeviceProvider);
  EXPECT_EQ(candidate.device_target.artifact_target.target_profile,
            &kFakeTargetProfile);
  EXPECT_EQ(loom_device_target_bundle(&candidate.device_target),
            &kFakeTargetBundle);
  EXPECT_EQ(candidate.device_artifact.executable_target, nullptr);
  EXPECT_EQ(candidate.device_artifact.artifact,
            &candidate.artifact_candidate.artifact);
  const loom_artifact_t& artifact = candidate.artifact_candidate.artifact;
  EXPECT_EQ(artifact.target_bundle, &kFakeTargetBundle);
  EXPECT_TRUE(
      iree_string_view_equal(artifact.target_key, IREE_SV("fake-hal-target")));
  EXPECT_EQ(artifact.target_artifact_format, LOOM_TARGET_ARTIFACT_FORMAT_ELF);
  ASSERT_NE(artifact.target_artifact_data, nullptr);
  testing::ByteSequenceClone target_artifact(iree_allocator_system());
  IREE_ASSERT_OK(target_artifact.Clone(artifact.target_artifact_data));
  EXPECT_EQ(target_artifact.contents().data_length,
            sizeof(kFakeHalTargetArtifactData));
  EXPECT_EQ(memcmp(target_artifact.contents().data, kFakeHalTargetArtifactData,
                   sizeof(kFakeHalTargetArtifactData)),
            0);
  ASSERT_NE(artifact.executable_data, nullptr);
  testing::ByteSequenceClone executable(iree_allocator_system());
  IREE_ASSERT_OK(executable.Clone(artifact.executable_data));
  EXPECT_EQ(executable.contents().data_length, sizeof(kFakeHalExecutableData));
  EXPECT_EQ(memcmp(executable.contents().data, kFakeHalExecutableData,
                   sizeof(kFakeHalExecutableData)),
            0);
  EXPECT_EQ(candidate.artifact_candidate.compile_report.artifact_kind,
            LOOM_TARGET_COMPILE_ARTIFACT_KIND_HAL_EXECUTABLE);
  EXPECT_EQ(candidate.artifact_candidate.compile_report.status_code,
            IREE_STATUS_OK);
  EXPECT_TRUE(iree_string_view_equal(
      candidate.artifact_candidate.compile_report.backend_name,
      IREE_SV("fake-hal")));
  EXPECT_TRUE(iree_string_view_equal(
      candidate.artifact_candidate.compile_report.target_key,
      IREE_SV("fake-hal-target")));
  EXPECT_TRUE(iree_string_view_equal(
      candidate.artifact_candidate.compile_report.artifact_format,
      IREE_SV("elf")));
  EXPECT_EQ(candidate.artifact_candidate.compile_report.artifact_size,
            sizeof(kFakeHalExecutableData));
  EXPECT_EQ(report.artifact_size,
            candidate.artifact_candidate.compile_report.artifact_size);

  loom_run_hal_candidate_deinitialize(&candidate);
  EXPECT_EQ(candidate.provider, nullptr);
  loom_run_module_deinitialize(&run_module);
}

TEST_F(HalCandidateTest, CompileHalExecutableCandidateWithoutReport) {
  loom_run_module_t run_module = {};
  IREE_ASSERT_OK(Parse(IREE_SV(kHalSource), &run_module));

  loom_compile_options_t options = {};
  InitializeCompileOptions(&run_module, &options);

  loom_run_hal_candidate_t candidate = {};
  g_fake_hal_emit_was_called = false;
  g_fake_hal_emit_report = nullptr;
  IREE_ASSERT_OK(loom_run_hal_candidate_compile(
      &kFakeDeviceProvider, FakeHalRuntime(), &run_module,
      /*target_requirement=*/nullptr, &options, iree_allocator_system(),
      &candidate));
  EXPECT_EQ(g_fake_hal_selected_target_requirement, nullptr);
  EXPECT_TRUE(g_fake_hal_emit_was_called);
  EXPECT_EQ(g_fake_hal_emit_report, nullptr);
  EXPECT_EQ(candidate.artifact_candidate.compile_report.detail_flags,
            LOOM_TARGET_COMPILE_REPORT_DETAIL_NONE);
  EXPECT_EQ(candidate.artifact_candidate.compile_report.artifact_size, 0u);

  loom_run_hal_candidate_deinitialize(&candidate);
  loom_run_module_deinitialize(&run_module);
}

TEST_F(HalCandidateTest, CompileHalRequiresHooks) {
  loom_run_module_t run_module = {};
  IREE_ASSERT_OK(Parse(IREE_SV(kHalSource), &run_module));

  loom_compile_options_t options = {};
  InitializeCompileOptions(&run_module, &options);

  const loom_device_provider_t provider = {
      /*.artifact_provider=*/&kFakeArtifactProvider,
  };
  loom_run_hal_candidate_t candidate = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_run_hal_candidate_compile(&provider, FakeHalRuntime(), &run_module,
                                     /*target_requirement=*/nullptr, &options,
                                     iree_allocator_system(), &candidate));
  EXPECT_EQ(candidate.provider, nullptr);

  loom_run_module_deinitialize(&run_module);
}

}  // namespace
}  // namespace loom
