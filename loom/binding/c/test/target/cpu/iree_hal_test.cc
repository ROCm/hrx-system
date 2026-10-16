// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/cpu/iree_hal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/base/internal/cpu.h"
#include "iree/base/internal/math.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/task/device_spec.h"
#include "iree/hal/drivers/task/executable/elf/elf_module.h"
#include "iree/hal/utils/device_spec_builder.h"
#include "iree/testing/gtest.h"
#include "loom/target/arch/x86/facts.h"
#include "loom/target/profile.h"
#include "loomc/artifact.h"
#include "loomc/compile.h"
#include "loomc/context.h"
#include "loomc/diagnostic.h"
#include "loomc/interop.h"
#include "loomc/module.h"
#include "loomc/result.h"
#include "loomc/source.h"
#include "loomc/target/configured.h"
#include "loomc/workspace.h"
#include "test/util.h"

namespace {

using loomc::testing::HandlePtr;

using DeviceSpecPtr =
    HandlePtr<iree_hal_device_spec_t, iree_hal_device_spec_release>;
using CompilerPtr = HandlePtr<loomc_compiler_t, loomc_compiler_release>;
using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using TargetEnvironmentPtr =
    HandlePtr<loomc_target_environment_t, loomc_target_environment_release>;
using TargetProfilePtr =
    HandlePtr<loomc_target_profile_t, loomc_target_profile_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;

typedef struct FakeHalDevice {
  // HAL resource header used by device vtable dispatch.
  iree_hal_resource_t resource;

  // Immutable device facts borrowed from the test.
  const iree_hal_device_spec_t* device_spec;
} FakeHalDevice;

std::string ToString(loomc_string_view_t value) {
  return value.data ? std::string(value.data, value.size) : std::string();
}

std::string ToString(const loomc_byte_sequence_t* value) {
  loomc_byte_span_t contents = loomc_byte_span_empty();
  LOOMC_EXPECT_OK(
      loomc_byte_sequence_clone(value, loomc_allocator_system(), &contents));
  std::string result(reinterpret_cast<const char*>(contents.data),
                     contents.data_length);
  loomc_allocator_free(loomc_allocator_system(), (void*)contents.data);
  return result;
}

static const iree_hal_device_spec_t* FakeHalDeviceSpec(
    iree_hal_device_t* base_device) {
  FakeHalDevice* device = reinterpret_cast<FakeHalDevice*>(base_device);
  return device->device_spec;
}

static iree_hal_device_vtable_t MakeFakeHalDeviceVtable() {
  iree_hal_device_vtable_t vtable = {.device_spec = FakeHalDeviceSpec};
  return vtable;
}

static const iree_hal_device_vtable_t kFakeHalDeviceVtable =
    MakeFakeHalDeviceVtable();

void InitializeFakeDevice(const iree_hal_device_spec_t* device_spec,
                          FakeHalDevice* out_device) {
  out_device->device_spec = device_spec;
  iree_hal_resource_initialize(&kFakeHalDeviceVtable, &out_device->resource);
}

iree_status_t CreateCpuDeviceSpec(bool include_cpu_facet,
                                  bool include_loader_target,
                                  DeviceSpecPtr* out_device_spec,
                                  const iree_cpu_data_t* cpu_data = nullptr) {
  out_device_spec->reset();
  iree_hal_device_spec_builder_t builder;
  iree_hal_device_spec_builder_initialize(iree_allocator_system(), &builder);
  iree_status_t status = iree_ok_status();

  std::vector<uint8_t> cpu_payload;
  if (include_cpu_facet) {
    const iree_hal_cpu_device_spec_t cpu_spec = {
        .cpu_data = cpu_data != nullptr
            ? *cpu_data
            : iree_cpu_data_t{
                  .architecture = IREE_CPU_ARCHITECTURE_X86_64,
              },
        .flags = IREE_HAL_CPU_DEVICE_SPEC_FLAG_NONE,
    };
    cpu_payload.resize(iree_hal_cpu_device_spec_payload_size());
    status = iree_hal_cpu_device_spec_encode(
        &cpu_spec, iree_make_byte_span(cpu_payload.data(), cpu_payload.size()));
    if (iree_status_is_ok(status)) {
      const iree_hal_device_spec_facet_t facet = {
          .schema_id =
              iree_make_cstring_view(IREE_HAL_CPU_DEVICE_SPEC_SCHEMA_ID),
          .schema_version = IREE_HAL_CPU_DEVICE_SPEC_SCHEMA_VERSION,
          .payload =
              iree_make_const_byte_span(cpu_payload.data(), cpu_payload.size()),
      };
      status = iree_hal_device_spec_builder_add_facet(&builder, &facet);
    }
  }

  if (iree_status_is_ok(status) && include_loader_target) {
    const iree_hal_executable_target_t target = {
        .family = IREE_SV("cpu"),
        .target_key = IREE_SV("x86_64"),
        .kind = IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC,
        .priority = 0,
        .physical_device_affinity = 1,
        .flags = IREE_HAL_EXECUTABLE_TARGET_FLAG_NONE,
    };
    status =
        iree_hal_device_spec_builder_add_executable_target(&builder, &target);
  }

  iree_hal_device_spec_t* device_spec = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_spec_builder_finalize(&builder, &device_spec);
  }
  iree_hal_device_spec_builder_deinitialize(&builder);
  if (iree_status_is_ok(status)) {
    out_device_spec->reset(device_spec);
  }
  return status;
}

TargetEnvironmentPtr CreateConfiguredTargetEnvironment() {
  loomc_target_environment_t* target_environment = nullptr;
  LOOMC_EXPECT_OK(loomc_target_environment_create_configured(
      loomc_allocator_system(), &target_environment));
  return TargetEnvironmentPtr(target_environment);
}

ContextPtr CreateConfiguredContext(
    loomc_target_environment_t* target_environment) {
  const loomc_context_target_options_t target_options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
      .structure_size = sizeof(target_options),
      .next = nullptr,
      .target_environment = target_environment,
  };
  const loomc_context_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
      .structure_size = sizeof(options),
      .next = &target_options,
  };
  loomc_context_t* context = nullptr;
  LOOMC_EXPECT_OK(
      loomc_context_create(&options, loomc_allocator_system(), &context));
  return ContextPtr(context);
}

WorkspacePtr CreateWorkspace() {
  loomc_workspace_t* workspace = nullptr;
  LOOMC_EXPECT_OK(
      loomc_workspace_create(nullptr, loomc_allocator_system(), &workspace));
  return WorkspacePtr(workspace);
}

CompilerPtr CreateCompiler(loomc_context_t* context) {
  loomc_compiler_t* compiler = nullptr;
  LOOMC_EXPECT_OK(loomc_compiler_create(context, nullptr,
                                        loomc_allocator_system(), &compiler));
  return CompilerPtr(compiler);
}

ModulePtr ParseModule(loomc_context_t* context, loomc_workspace_t* workspace,
                      const char* contents) {
  const loomc_source_options_t source_options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .structure_size = sizeof(source_options),
      .next = nullptr,
      .format = LOOMC_SOURCE_FORMAT_TEXT,
      .identifier = loomc_make_cstring_view("cpu_target.loom"),
      .contents = loomc_make_byte_span(contents, std::strlen(contents)),
      .storage = LOOMC_SOURCE_STORAGE_COPY,
  };
  loomc_source_t* source = nullptr;
  LOOMC_EXPECT_OK(
      loomc_source_create(&source_options, loomc_allocator_system(), &source));
  SourcePtr source_ptr(source);

  loomc_module_t* module = nullptr;
  loomc_result_t* result = nullptr;
  LOOMC_EXPECT_OK(loomc_module_deserialize_from_source(
      context, workspace, source_ptr.get(), nullptr, loomc_allocator_system(),
      &module, &result));
  ResultPtr result_ptr(result);
  EXPECT_TRUE(loomc_result_succeeded(result_ptr.get()));
  return ModulePtr(module);
}

const loomc_artifact_t* FindArtifact(const loomc_result_t* result,
                                     loomc_artifact_kind_t kind,
                                     const char* format) {
  for (loomc_host_size_t i = 0; i < loomc_result_artifact_count(result); ++i) {
    const loomc_artifact_t* artifact = loomc_result_artifact_at(result, i);
    if (artifact != nullptr && artifact->kind == kind &&
        ToString(artifact->format) == format) {
      return artifact;
    }
  }
  return nullptr;
}

TargetProfilePtr SelectCpuTarget(
    loomc_target_environment_t* target_environment, FakeHalDevice* device,
    loomc_result_t** out_result,
    const iree_hal_executable_target_t** out_executable_target = nullptr,
    loomc_target_profile_t* requested_profile = nullptr) {
  const loomc_iree_hal_target_provider_t* providers[] = {
      loomc_cpu_iree_hal_target_provider(),
  };
  const loomc_iree_hal_target_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_IREE_HAL_TARGET_OPTIONS,
      .structure_size = sizeof(options),
      .next = nullptr,
      .identifier = loomc_make_cstring_view("fake-cpu"),
      .device = reinterpret_cast<iree_hal_device_t*>(device),
      .physical_device_affinity = 1,
      .target_profile = requested_profile,
      .providers = providers,
      .provider_count = IREE_ARRAYSIZE(providers),
  };
  loomc_iree_hal_target_selection_t selection = {};
  LOOMC_EXPECT_OK(loomc_target_select_iree_hal(target_environment, &options,
                                               loomc_allocator_system(),
                                               &selection, out_result));
  if (out_executable_target != nullptr) {
    *out_executable_target = selection.executable_target;
  }
  return TargetProfilePtr(selection.target_profile);
}

TEST(LoomcCpuIreeHalTargetTest, PreservesExplicitNativeProfileIdentity) {
  DeviceSpecPtr device_spec;
  IREE_ASSERT_OK(CreateCpuDeviceSpec(/*include_cpu_facet=*/true,
                                     /*include_loader_target=*/true,
                                     &device_spec));
  FakeHalDevice device = {};
  InitializeFakeDevice(device_spec.get(), &device);
  TargetEnvironmentPtr target_environment = CreateConfiguredTargetEnvironment();
  loomc_target_profile_t* selected_profile = nullptr;
  LOOMC_ASSERT_OK(loomc_target_profile_select(
      target_environment.get(), loomc_make_cstring_view("x86:scalar"),
      loomc_allocator_system(), &selected_profile));
  TargetProfilePtr requested_profile(selected_profile);

  loomc_result_t* result = nullptr;
  TargetProfilePtr profile = SelectCpuTarget(
      target_environment.get(), &device, &result,
      /*out_executable_target=*/nullptr, requested_profile.get());
  ResultPtr result_ptr(result);

  ASSERT_NE(result_ptr.get(), nullptr);
  EXPECT_TRUE(loomc_result_succeeded(result_ptr.get()));
  EXPECT_EQ(profile.get(), requested_profile.get());
}

TEST(LoomcCpuIreeHalTargetTest, SelectsNativeProfileAndLoaderTogether) {
  DeviceSpecPtr device_spec;
  IREE_ASSERT_OK(CreateCpuDeviceSpec(/*include_cpu_facet=*/true,
                                     /*include_loader_target=*/true,
                                     &device_spec));
  FakeHalDevice device = {};
  InitializeFakeDevice(device_spec.get(), &device);
  TargetEnvironmentPtr target_environment = CreateConfiguredTargetEnvironment();
  loomc_result_t* result = nullptr;
  const iree_hal_executable_target_t* executable_target = nullptr;
  TargetProfilePtr profile = SelectCpuTarget(target_environment.get(), &device,
                                             &result, &executable_target);
  ResultPtr result_ptr(result);

  ASSERT_NE(result_ptr.get(), nullptr);
  EXPECT_TRUE(loomc_result_succeeded(result_ptr.get()));
  ASSERT_NE(profile.get(), nullptr);
  ASSERT_NE(executable_target, nullptr);
  EXPECT_TRUE(
      iree_string_view_equal(executable_target->family, IREE_SV("cpu")));
  EXPECT_TRUE(
      iree_string_view_equal(executable_target->target_key, IREE_SV("x86_64")));
  EXPECT_EQ(executable_target->kind, IREE_HAL_EXECUTABLE_TARGET_KIND_GENERIC);

  const loom_target_profile_t* native_profile =
      loomc_target_profile_get_interop_view(profile.get());
  ASSERT_NE(native_profile, nullptr);
  ASSERT_NE(native_profile->target_bundle, nullptr);
  ASSERT_NE(native_profile->target_bundle->snapshot, nullptr);
  EXPECT_EQ(native_profile->target_bundle->snapshot->default_pointer_bitwidth,
            64u);
}

TEST(LoomcCpuIreeHalTargetTest, SelectsStrongestProfileFromDeviceFacts) {
  const iree_cpu_data_t cpu_data = {
      .architecture = IREE_CPU_ARCHITECTURE_X86_64,
      .fields = {IREE_CPU_DATA0_X86_64_AVX | IREE_CPU_DATA0_X86_64_FMA |
                 IREE_CPU_DATA0_X86_64_AVX2 | IREE_CPU_DATA0_X86_64_AVX512F |
                 IREE_CPU_DATA0_X86_64_AVX512VL |
                 IREE_CPU_DATA0_X86_64_AVX512DQ |
                 IREE_CPU_DATA0_X86_64_AVX512BW},
  };
  DeviceSpecPtr device_spec;
  IREE_ASSERT_OK(CreateCpuDeviceSpec(/*include_cpu_facet=*/true,
                                     /*include_loader_target=*/true,
                                     &device_spec, &cpu_data));
  FakeHalDevice device = {};
  InitializeFakeDevice(device_spec.get(), &device);
  TargetEnvironmentPtr target_environment = CreateConfiguredTargetEnvironment();
  loomc_result_t* result = nullptr;
  TargetProfilePtr profile =
      SelectCpuTarget(target_environment.get(), &device, &result);
  ResultPtr result_ptr(result);

  ASSERT_NE(result_ptr.get(), nullptr);
  EXPECT_TRUE(loomc_result_succeeded(result_ptr.get()));
  ASSERT_NE(profile.get(), nullptr);
  const loom_target_profile_t* native_profile =
      loomc_target_profile_get_interop_view(profile.get());
  ASSERT_NE(native_profile, nullptr);
  ASSERT_NE(native_profile->target_bundle, nullptr);
  EXPECT_TRUE(iree_string_view_equal(native_profile->target_bundle->name,
                                     IREE_SV("x86-avx512")));
}

TEST(LoomcCpuIreeHalTargetTest, CompilesSerializedAvxVnniInt8ProfileToObject) {
  const iree_cpu_data_t cpu_data = {
      .architecture = IREE_CPU_ARCHITECTURE_X86_64,
      .fields = {IREE_CPU_DATA0_X86_64_AVX | IREE_CPU_DATA0_X86_64_FMA |
                     IREE_CPU_DATA0_X86_64_AVX2 |
                     IREE_CPU_DATA0_X86_64_AVXVNNIINT8,
                 2, 3, 4, 5, 6, 7, 8},
  };
  DeviceSpecPtr device_spec;
  IREE_ASSERT_OK(CreateCpuDeviceSpec(/*include_cpu_facet=*/true,
                                     /*include_loader_target=*/true,
                                     &device_spec, &cpu_data));
  FakeHalDevice device = {};
  InitializeFakeDevice(device_spec.get(), &device);
  TargetEnvironmentPtr target_environment = CreateConfiguredTargetEnvironment();
  loomc_result_t* selection_result = nullptr;
  TargetProfilePtr profile =
      SelectCpuTarget(target_environment.get(), &device, &selection_result);
  ResultPtr selection_result_ptr(selection_result);
  ASSERT_NE(selection_result_ptr.get(), nullptr);
  ASSERT_TRUE(loomc_result_succeeded(selection_result_ptr.get()));
  ASSERT_NE(profile.get(), nullptr);

  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);
  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool, &arena);
  loom_target_facts_t* base_facts = nullptr;
  IREE_ASSERT_OK(loom_target_profile_project_facts(
      loomc_target_profile_get_interop_view(profile.get()), &arena,
      &base_facts));
  const loom_x86_target_facts_t* facts = loom_x86_target_facts_cast(base_facts);
  ASSERT_NE(facts, nullptr);
  EXPECT_EQ(facts->cpu_data.architecture, cpu_data.architecture);
  for (iree_host_size_t i = 0; i < IREE_CPU_DATA_FIELD_COUNT; ++i) {
    EXPECT_EQ(facts->cpu_data.fields[i], cpu_data.fields[i]);
  }
  EXPECT_TRUE(
      iree_string_view_equal(facts->base.storage.config.contract_set_key,
                             IREE_SV("x86.avx2_features.core")));
  EXPECT_NE(facts->base.storage.config.contract_feature_bits, 0u);
  EXPECT_TRUE(loom_target_facts_field_is_explicit(
      &facts->base, LOOM_TARGET_FACT_FIELD_CONTRACT_SET_KEY));
  EXPECT_TRUE(loom_target_facts_field_is_explicit(
      &facts->base, LOOM_TARGET_FACT_FIELD_CONTRACT_FEATURE_BITS));
  iree_arena_deinitialize(&arena);
  iree_arena_block_pool_deinitialize(&block_pool);

  ContextPtr context = CreateConfiguredContext(target_environment.get());
  WorkspacePtr workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  ModulePtr module = ParseModule(context.get(), workspace.get(), R"(
func.def public @dot4i_s8s8_256(%lhs: vector<32xi8>, %rhs: vector<32xi8>, %acc: vector<8xi32>) -> (vector<8xi32>) {
  %dot = vector.dot4i<s8s8> %lhs, %rhs, %acc : vector<32xi8>, vector<32xi8>, vector<8xi32>
  func.return %dot : vector<8xi32>
}
)");
  const loomc_string_view_t root = loomc_make_cstring_view("dot4i_s8s8_256");
  const loomc_emit_options_t emit_options = {
      .type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      .structure_size = sizeof(emit_options),
      .next = nullptr,
      .artifact_format = loomc_make_cstring_view("x86-elf"),
      .identifier = loomc_make_cstring_view("dot4i_s8s8_256.o"),
      .artifact_flags = LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
  };
  const loomc_compile_artifact_options_t compile_options = {
      .type = LOOMC_STRUCTURE_TYPE_COMPILE_ARTIFACT_OPTIONS,
      .structure_size = sizeof(compile_options),
      .next = nullptr,
      .roots = &root,
      .root_count = 1,
      .excluded_roots = nullptr,
      .excluded_root_count = 0,
      .target_profile = profile.get(),
      .config = nullptr,
      .emit_options = &emit_options,
      .artifact_flags = 0,
  };
  loomc_result_t* compile_result = nullptr;
  LOOMC_ASSERT_OK(loomc_compile_artifact(
      compiler.get(), workspace.get(), /*pass_program=*/nullptr, module.get(),
      &compile_options, loomc_allocator_system(), &compile_result));
  ResultPtr compile_result_ptr(compile_result);
  ASSERT_NE(compile_result_ptr.get(), nullptr);
  if (!loomc_result_succeeded(compile_result_ptr.get()) &&
      loomc_result_diagnostic_count(compile_result_ptr.get()) != 0) {
    ADD_FAILURE() << ToString(
        loomc_result_diagnostic_at(compile_result_ptr.get(), 0)->message);
  }
  ASSERT_TRUE(loomc_result_succeeded(compile_result_ptr.get()));
  const loomc_artifact_t* object = FindArtifact(
      compile_result_ptr.get(), LOOMC_ARTIFACT_KIND_EXECUTABLE, "x86-elf");
  ASSERT_NE(object, nullptr);
  static constexpr uint8_t kElfMagic[] = {0x7F, 'E', 'L', 'F'};
  const std::string object_contents = ToString(object->contents);
  ASSERT_GE(object_contents.size(), sizeof(kElfMagic));
  EXPECT_EQ(std::memcmp(object_contents.data(), kElfMagic, sizeof(kElfMagic)),
            0);
}

TEST(LoomcCpuIreeHalTargetTest, CompilesBfloatBroadcastWithoutNeConvert) {
  const iree_cpu_data_t cpu_data = {
      .architecture = IREE_CPU_ARCHITECTURE_X86_64,
      .fields = {IREE_CPU_DATA0_X86_64_AVX | IREE_CPU_DATA0_X86_64_FMA |
                 IREE_CPU_DATA0_X86_64_AVX2},
  };
  DeviceSpecPtr device_spec;
  IREE_ASSERT_OK(CreateCpuDeviceSpec(true, true, &device_spec, &cpu_data));
  FakeHalDevice device = {};
  InitializeFakeDevice(device_spec.get(), &device);
  TargetEnvironmentPtr target_environment = CreateConfiguredTargetEnvironment();
  loomc_result_t* selection_result = nullptr;
  TargetProfilePtr profile =
      SelectCpuTarget(target_environment.get(), &device, &selection_result);
  ResultPtr selection_result_ptr(selection_result);
  ASSERT_TRUE(loomc_result_succeeded(selection_result_ptr.get()));
  ASSERT_NE(profile.get(), nullptr);
  ContextPtr context = CreateConfiguredContext(target_environment.get());
  WorkspacePtr workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());
  for (int lanes : {4, 8}) {
    SCOPED_TRACE(lanes);
    const std::string vector_type = "vector<" + std::to_string(lanes) + "xf32>";
    const std::string source =
        "func.def public @broadcast(%input: buffer) -> (" + vector_type +
        ") {\n"
        "  %base = index.constant 0 : offset\n"
        "  %source = buffer.view %input[%base] : buffer -> view<1xbf16>\n"
        "  %narrow = view.load %source[0] : view<1xbf16> -> bf16\n"
        "  %wide = scalar.extf<daz> %narrow : bf16 to f32\n"
        "  %result = vector.splat %wide : " +
        vector_type +
        "\n"
        "  func.return %result : " +
        vector_type + "\n}\n";
    ModulePtr module =
        ParseModule(context.get(), workspace.get(), source.c_str());
    const loomc_string_view_t root = loomc_make_cstring_view("broadcast");
    const loomc_emit_options_t emit_options = {
        .type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
        .structure_size = sizeof(emit_options),
        .artifact_format = loomc_make_cstring_view("x86-elf"),
        .artifact_flags = LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
    };
    const loomc_compile_artifact_options_t compile_options = {
        .type = LOOMC_STRUCTURE_TYPE_COMPILE_ARTIFACT_OPTIONS,
        .structure_size = sizeof(compile_options),
        .roots = &root,
        .root_count = 1,
        .target_profile = profile.get(),
        .emit_options = &emit_options,
    };
    loomc_result_t* compile_result = nullptr;
    LOOMC_ASSERT_OK(loomc_compile_artifact(
        compiler.get(), workspace.get(), nullptr, module.get(),
        &compile_options, loomc_allocator_system(), &compile_result));
    ResultPtr result(compile_result);
    if (!loomc_result_succeeded(result.get())) {
      for (loomc_host_size_t i = 0;
           i < loomc_result_diagnostic_count(result.get()); ++i) {
        ADD_FAILURE() << ToString(
            loomc_result_diagnostic_at(result.get(), i)->message);
      }
    }
    ASSERT_TRUE(loomc_result_succeeded(result.get()));
    ASSERT_NE(
        FindArtifact(result.get(), LOOMC_ARTIFACT_KIND_EXECUTABLE, "x86-elf"),
        nullptr);
  }
}

class LegacyHalfConversionTest : public ::testing::TestWithParam<int> {};

TEST_P(LegacyHalfConversionTest,
       DeviceFactsCompileAndExecuteStorageConversions) {
  const bool avx512 = GetParam() == 2;
  const bool f16c = GetParam() == 1;
  const iree_cpu_data_t cpu_data = {
      .architecture = IREE_CPU_ARCHITECTURE_X86_64,
      .fields = {IREE_CPU_DATA0_X86_64_AVX | IREE_CPU_DATA0_X86_64_FMA |
                 IREE_CPU_DATA0_X86_64_AVX2 |
                 (avx512 ? IREE_CPU_DATA0_X86_64_AVX512F |
                               IREE_CPU_DATA0_X86_64_AVX512DQ |
                               IREE_CPU_DATA0_X86_64_AVX512BW |
                               IREE_CPU_DATA0_X86_64_AVX512VL
                         : (f16c ? IREE_CPU_DATA0_X86_64_F16C : 0))},
  };
  DeviceSpecPtr device_spec;
  IREE_ASSERT_OK(CreateCpuDeviceSpec(true, true, &device_spec, &cpu_data));
  FakeHalDevice device = {};
  InitializeFakeDevice(device_spec.get(), &device);
  TargetEnvironmentPtr target_environment = CreateConfiguredTargetEnvironment();
  loomc_result_t* selection_result = nullptr;
  TargetProfilePtr profile =
      SelectCpuTarget(target_environment.get(), &device, &selection_result);
  ResultPtr selection_result_ptr(selection_result);
  ASSERT_TRUE(loomc_result_succeeded(selection_result_ptr.get()));
  ASSERT_NE(profile.get(), nullptr);
  ContextPtr context = CreateConfiguredContext(target_environment.get());
  WorkspacePtr workspace = CreateWorkspace();
  CompilerPtr compiler = CreateCompiler(context.get());

  struct Conversion {
    // Source storage element type.
    const char* input;
    // Destination storage element type.
    const char* output;
    // Source conversion operation name.
    const char* operation;
  };
  constexpr Conversion kConversions[] = {
      {"f16", "f32", "extf"},
      {"f16", "f64", "extf"},
      {"f32", "f16", "fptrunc"},
      {"f32", "bf16", "fptrunc"},
  };
  iree_cpu_data_t host_cpu_data;
  iree_cpu_query_data(iree_allocator_system(), &host_cpu_data);
  const bool can_execute =
      iree_cpu_data_satisfies_features(&host_cpu_data, &cpu_data);
  for (const auto& conversion : kConversions) {
    const bool narrow = conversion.input == std::string("f32");
    const bool bfloat = conversion.output == std::string("bf16");
    // Signed zeros, subnormal/normal boundaries, ordinary values, the largest
    // finite value, infinity, and signaling/quiet NaNs, with both signs.
    constexpr uint16_t kHalfBits[] = {
        0x0000, 0x0001, 0x0002, 0x03ff, 0x0400, 0x0401, 0x3555, 0x3bff,
        0x3c00, 0x3c01, 0x4000, 0x7bff, 0x7c00, 0x7c01, 0x7e00, 0x7fff,
        0x8000, 0x8001, 0x8002, 0x83ff, 0x8400, 0x8401, 0xb555, 0xbbff,
        0xbc00, 0xbc01, 0xc000, 0xfbff, 0xfc00, 0xfc01, 0xfe00, 0xffff,
    };
    std::vector<uint32_t> float_bits;
    if (narrow) {
      const auto append_signed = [&float_bits](uint32_t bits) {
        float_bits.push_back(bits);
        float_bits.push_back(bits | 0x80000000u);
      };
      for (uint32_t bits : {0u, 1u, 0x007fffffu, 0x00800000u, 0x3eaaaaabu,
                            0x3f800000u, 0x40000000u, 0x7f7fffffu, 0x7f800000u,
                            0x7f800001u, 0x7fc00000u, 0x7fffffffu}) {
        append_signed(bits);
      }
      // Probe both neighbors of ties at zero, the normal boundary, even/odd
      // significands near one, an exponent carry, and overflow.
      constexpr uint32_t kHalfMidpoints[] = {
          0x33000000, 0x387fe000, 0x3f801000,
          0x3f803000, 0x3ffff000, 0x477ff000,
      };
      constexpr uint32_t kBfloatMidpoints[] = {
          0x00008000, 0x007f8000, 0x3f808000,
          0x3f818000, 0x3fff8000, 0x7f7f8000,
      };
      for (uint32_t midpoint : bfloat ? kBfloatMidpoints : kHalfMidpoints) {
        for (int delta : {-1, 0, 1}) {
          append_signed(midpoint + delta);
        }
      }
    }
    // Scalar carriers, a partial/native register, and multiple native packets
    // exercise the device-selected execution paths. Exact lowering goldens and
    // contract-family tests cover the individual native packet widths.
    for (int lanes : {1, 4, 32}) {
      SCOPED_TRACE(std::string(conversion.input) + " -> " + conversion.output +
                   " lanes=" + std::to_string(lanes));
      const auto value_type = [lanes](const char* element) {
        return lanes == 1
                   ? std::string(element)
                   : "vector<" + std::to_string(lanes) + "x" + element + ">";
      };
      const std::string input_type = value_type(conversion.input);
      const std::string output_type = value_type(conversion.output);
      const std::string input_view =
          "view<" + std::to_string(lanes) + "x" + conversion.input + ">";
      const std::string output_view =
          "view<" + std::to_string(lanes) + "x" + conversion.output + ">";
      const std::string memory = lanes == 1 ? "view" : "vector";
      // The production ELF call bridge preserves the System V convention on
      // Windows as well as hosts whose native convention already matches it.
      const std::string source =
          "func.def public @convert(%input: buffer, %output: buffer, "
          "%reserved: buffer) -> (i32) {\n"
          "  %base = index.constant 0 : offset\n"
          "  %source = buffer.view %input[%base] : buffer -> " +
          input_view +
          "\n"
          "  %destination = buffer.view %output[%base] : buffer -> " +
          output_view +
          "\n"
          "  %value = " +
          memory + ".load %source[0] : " + input_view + " -> " + input_type +
          "\n"
          "  %converted = " +
          (lanes == 1 ? "scalar." : "vector.") + conversion.operation +
          " %value : " + input_type + " to " + output_type +
          "\n"
          "  " +
          memory + ".store %converted, %destination[0] : " + output_type +
          ", " + output_view +
          "\n"
          "  %status = scalar.constant 0 : i32\n"
          "  func.return %status : i32\n}\n";
      ModulePtr module =
          ParseModule(context.get(), workspace.get(), source.c_str());
      const loomc_string_view_t root = loomc_make_cstring_view("convert");
      const loomc_emit_options_t emit_options = {
          .type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
          .structure_size = sizeof(emit_options),
          .artifact_format = loomc_make_cstring_view("x86-elf-shared"),
          .artifact_flags = LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
      };
      const loomc_compile_artifact_options_t compile_options = {
          .type = LOOMC_STRUCTURE_TYPE_COMPILE_ARTIFACT_OPTIONS,
          .structure_size = sizeof(compile_options),
          .roots = &root,
          .root_count = 1,
          .target_profile = profile.get(),
          .emit_options = &emit_options,
      };
      loomc_result_t* compile_result = nullptr;
      LOOMC_ASSERT_OK(loomc_compile_artifact(
          compiler.get(), workspace.get(), nullptr, module.get(),
          &compile_options, loomc_allocator_system(), &compile_result));
      ResultPtr result(compile_result);
      if (!avx512 && !f16c && !narrow) {
        ASSERT_FALSE(loomc_result_succeeded(result.get()));
        ASSERT_GT(loomc_result_diagnostic_count(result.get()), 0u);
        EXPECT_EQ(ToString(loomc_result_diagnostic_at(result.get(), 0)->code),
                  "TARGET/046");
        continue;
      }
      if (!loomc_result_succeeded(result.get())) {
        for (loomc_host_size_t i = 0;
             i < loomc_result_diagnostic_count(result.get()); ++i) {
          ADD_FAILURE() << ToString(
              loomc_result_diagnostic_at(result.get(), i)->message);
        }
      }
      ASSERT_TRUE(loomc_result_succeeded(result.get()));
      const loomc_artifact_t* image = FindArtifact(
          result.get(), LOOMC_ARTIFACT_KIND_EXECUTABLE, "x86-elf-shared");
      ASSERT_NE(image, nullptr);
      if (!can_execute) {
        continue;
      }
      const std::string contents = ToString(image->contents);
      iree_elf_module_t native_module = {};
      IREE_ASSERT_OK(iree_elf_module_initialize_from_memory(
          iree_make_const_byte_span(contents.data(), contents.size()),
          iree_allocator_system(), &native_module));
      auto unload = [](iree_elf_module_t* module) {
        iree_elf_module_deinitialize(module);
      };
      std::unique_ptr<iree_elf_module_t, decltype(unload)> native_guard(
          &native_module, unload);
      void* address = nullptr;
      IREE_ASSERT_OK(
          iree_elf_module_lookup_export(&native_module, "convert", &address));

      const bool to_double = conversion.output == std::string("f64");
      const size_t output_bytes = lanes * (narrow ? 2 : (to_double ? 8 : 4));
      const size_t count =
          narrow ? float_bits.size() : IREE_ARRAYSIZE(kHalfBits);
      for (size_t position = 0; position < count; position += lanes) {
        std::array<uint8_t, 288> input;
        std::array<uint8_t, 288> output;
        input.fill(0xa7);
        output.fill(0xa7);
        std::array<double, 32> expected_wide;
        std::array<uint16_t, 32> expected_half;
        for (int lane = 0; lane < lanes; ++lane) {
          const size_t index = (position + lane) % count;
          if (narrow) {
            const uint32_t bits = float_bits[index];
            std::memcpy(input.data() + 16 + lane * 4, &bits, 4);
            float value;
            std::memcpy(&value, &bits, 4);
            expected_half[lane] = bfloat ? iree_math_f32_to_bf16(value)
                                         : iree_math_f32_to_f16(value);
            if (bfloat && (bits & 0x7fffffffu) > 0x7f800000u) {
              expected_half[lane] = (bits >> 16) | 0x0040u;
            }
          } else {
            const uint16_t bits = kHalfBits[index];
            std::memcpy(input.data() + 16 + lane * 2, &bits, 2);
            expected_wide[lane] = iree_math_f16_to_f32(bits);
          }
        }
        const auto before = input;
        ASSERT_EQ(iree_elf_call_i_ppp(address, input.data() + 16,
                                      output.data() + 16, nullptr),
                  0);
        ASSERT_EQ(input, before);
        const auto is_guard = [](uint8_t byte) { return byte == 0xa7; };
        ASSERT_TRUE(std::all_of(output.begin(), output.begin() + 16, is_guard));
        ASSERT_TRUE(std::all_of(output.begin() + 16 + output_bytes,
                                output.end(), is_guard));
        for (int lane = 0; lane < lanes; ++lane) {
          if (narrow) {
            uint16_t actual;
            std::memcpy(&actual, output.data() + 16 + lane * 2, 2);
            const uint16_t infinity = bfloat ? 0x7f80 : 0x7c00;
            if (!bfloat && (expected_half[lane] & 0x7fff) > infinity) {
              ASSERT_GT(actual & 0x7fff, infinity);
            } else {
              ASSERT_EQ(actual, expected_half[lane]) << position + lane;
            }
          } else {
            double actual;
            if (to_double) {
              std::memcpy(&actual, output.data() + 16 + lane * 8, 8);
            } else {
              float value;
              std::memcpy(&value, output.data() + 16 + lane * 4, 4);
              actual = value;
            }
            if (std::isnan(expected_wide[lane])) {
              ASSERT_TRUE(std::isnan(actual));
            } else {
              ASSERT_EQ(actual, expected_wide[lane]) << position + lane;
              ASSERT_EQ(std::signbit(actual),
                        std::signbit(expected_wide[lane]));
            }
          }
        }
      }
    }
  }
  if (!can_execute) {
    GTEST_SKIP() << "Compiled the profile; host lacks its execution features";
  }
}

INSTANTIATE_TEST_SUITE_P(BaseAvx2F16cAndAvx512, LegacyHalfConversionTest,
                         ::testing::Values(0, 1, 2));

TEST(LoomcCpuIreeHalTargetTest, ReportsMissingLoaderAsTargetDiagnostic) {
  DeviceSpecPtr device_spec;
  IREE_ASSERT_OK(CreateCpuDeviceSpec(/*include_cpu_facet=*/true,
                                     /*include_loader_target=*/false,
                                     &device_spec));
  FakeHalDevice device = {};
  InitializeFakeDevice(device_spec.get(), &device);
  TargetEnvironmentPtr target_environment = CreateConfiguredTargetEnvironment();
  loomc_result_t* result = nullptr;
  TargetProfilePtr profile =
      SelectCpuTarget(target_environment.get(), &device, &result);
  ResultPtr result_ptr(result);

  EXPECT_EQ(profile.get(), nullptr);
  ASSERT_NE(result_ptr.get(), nullptr);
  EXPECT_FALSE(loomc_result_succeeded(result_ptr.get()));
  ASSERT_GE(loomc_result_diagnostic_count(result_ptr.get()), 1u);
  const loomc_diagnostic_t* diagnostic =
      loomc_result_diagnostic_at(result_ptr.get(), 0);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(ToString(diagnostic->code), "CPU/IREE_HAL");
  EXPECT_THAT(ToString(diagnostic->message),
              ::testing::HasSubstr("no compatible native CPU loader"));
}

TEST(LoomcCpuIreeHalTargetTest, DoesNotClaimDevicesWithoutCpuFacts) {
  DeviceSpecPtr device_spec;
  IREE_ASSERT_OK(CreateCpuDeviceSpec(/*include_cpu_facet=*/false,
                                     /*include_loader_target=*/true,
                                     &device_spec));
  FakeHalDevice device = {};
  InitializeFakeDevice(device_spec.get(), &device);
  TargetEnvironmentPtr target_environment = CreateConfiguredTargetEnvironment();
  loomc_result_t* result = nullptr;
  TargetProfilePtr profile =
      SelectCpuTarget(target_environment.get(), &device, &result);
  ResultPtr result_ptr(result);

  EXPECT_EQ(profile.get(), nullptr);
  ASSERT_NE(result_ptr.get(), nullptr);
  EXPECT_FALSE(loomc_result_succeeded(result_ptr.get()));
  ASSERT_GE(loomc_result_diagnostic_count(result_ptr.get()), 1u);
  const loomc_diagnostic_t* diagnostic =
      loomc_result_diagnostic_at(result_ptr.get(), 0);
  ASSERT_NE(diagnostic, nullptr);
  EXPECT_EQ(ToString(diagnostic->code), "IREE_HAL/TARGET");
}

}  // namespace
