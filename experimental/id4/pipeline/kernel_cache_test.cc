// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/kernel_cache.h"

#include <cstring>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/resource.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static const char kConfiguredSmokeKernelSource[] = R"(
config.decl @id4.smoke.workgroups_x : %value: index where [range(%value, 1, 16)]
config.decl @id4.smoke.workgroup_size_x : %value: index where [range(%value, 1, 256)]

kernel.def export("id4_smoke_configured") @id4_smoke_configured() {
  %workgroups_x = config.get @id4.smoke.workgroups_x : index
  %workgroup_size_x = config.get @id4.smoke.workgroup_size_x : index
  %one = index.constant 1 : index
  kernel.launch.config workgroups(%workgroups_x, %one, %one) workgroup_size(%workgroup_size_x, %one, %one) : index
} launch() {
  kernel.return
}
)";

static std::string ToString(iree_string_view_t value) {
  return value.data ? std::string(value.data, value.size) : std::string();
}

static std::string ToString(iree_const_byte_span_t value) {
  return value.data ? std::string(reinterpret_cast<const char*>(value.data),
                                  value.data_length)
                    : std::string();
}

typedef struct id4_pipeline_kernel_cache_test_diagnostics_t {
  // Number of kernel diagnostic events observed.
  iree_host_size_t kernel_event_count;
  // Number of Loom result diagnostics observed.
  iree_host_size_t loom_diagnostic_count;
  // Event keys observed in order.
  std::vector<std::string> keys;
} id4_pipeline_kernel_cache_test_diagnostics_t;

static iree_status_t CaptureDiagnostics(
    void* user_data, const id4_pipeline_diagnostic_event_t* event) {
  auto* diagnostics =
      reinterpret_cast<id4_pipeline_kernel_cache_test_diagnostics_t*>(
          user_data);
  if (event->kind == ID4_PIPELINE_DIAGNOSTIC_EVENT_KIND_KERNEL) {
    ++diagnostics->kernel_event_count;
  }
  if (event->kernel && event->kernel->diagnostic_index != IREE_HOST_SIZE_MAX) {
    ++diagnostics->loom_diagnostic_count;
  }
  diagnostics->keys.push_back(ToString(event->key));
  return iree_ok_status();
}

static id4_pipeline_diagnostics_sink_t MakeDiagnosticsSink(
    id4_pipeline_kernel_cache_test_diagnostics_t* diagnostics) {
  return (id4_pipeline_diagnostics_sink_t){
      /*.emit=*/CaptureDiagnostics,
      /*.user_data=*/diagnostics,
  };
}

typedef struct id4_pipeline_kernel_cache_test_executable_t {
  // HAL resource header.
  iree_hal_resource_t resource;
  // Host allocator used for executable storage.
  iree_allocator_t host_allocator;
} id4_pipeline_kernel_cache_test_executable_t;

extern const iree_hal_executable_vtable_t
    id4_pipeline_kernel_cache_test_executable_vtable;

static id4_pipeline_kernel_cache_test_executable_t* TestExecutableCast(
    iree_hal_executable_t* base_executable) {
  IREE_HAL_ASSERT_TYPE(base_executable,
                       &id4_pipeline_kernel_cache_test_executable_vtable);
  return reinterpret_cast<id4_pipeline_kernel_cache_test_executable_t*>(
      base_executable);
}

static iree_status_t TestExecutableCreate(
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  *out_executable = nullptr;
  id4_pipeline_kernel_cache_test_executable_t* executable = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*executable),
                            reinterpret_cast<void**>(&executable)));
  std::memset(executable, 0, sizeof(*executable));
  iree_hal_resource_initialize(
      &id4_pipeline_kernel_cache_test_executable_vtable, &executable->resource);
  executable->host_allocator = host_allocator;
  *out_executable = reinterpret_cast<iree_hal_executable_t*>(executable);
  return iree_ok_status();
}

static void TestExecutableDestroy(iree_hal_executable_t* base_executable) {
  id4_pipeline_kernel_cache_test_executable_t* executable =
      TestExecutableCast(base_executable);
  iree_allocator_t host_allocator = executable->host_allocator;
  iree_allocator_free(host_allocator, executable);
}

static iree_host_size_t TestExecutableFunctionCount(
    iree_hal_executable_t* base_executable) {
  return 0;
}

static iree_status_t TestExecutableFunctionInfo(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function,
    iree_hal_executable_function_info_t* out_info) {
  return iree_make_status(IREE_STATUS_OUT_OF_RANGE);
}

static iree_status_t TestExecutableFunctionParameters(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function, iree_host_size_t capacity,
    iree_hal_executable_function_parameter_t* out_parameters) {
  return iree_make_status(IREE_STATUS_OUT_OF_RANGE);
}

static iree_status_t TestExecutableLookupFunctionByName(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    iree_hal_executable_function_t* out_function) {
  *out_function = iree_hal_executable_function_invalid();
  return iree_make_status(IREE_STATUS_NOT_FOUND);
}

static iree_status_t TestExecutableTryLookupGlobalByName(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    bool* out_found, iree_hal_executable_global_t* out_global) {
  *out_found = false;
  *out_global = iree_hal_executable_global_invalid();
  return iree_ok_status();
}

static iree_status_t TestExecutableGlobalInfo(
    iree_hal_executable_t* base_executable, iree_hal_executable_global_t global,
    iree_hal_executable_global_info_t* out_info) {
  return iree_make_status(IREE_STATUS_OUT_OF_RANGE);
}

static iree_status_t TestExecutableGlobalBuffer(
    iree_hal_executable_t* base_executable, iree_hal_executable_global_t global,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_buffer_t** out_buffer) {
  *out_buffer = nullptr;
  return iree_make_status(IREE_STATUS_OUT_OF_RANGE);
}

const iree_hal_executable_vtable_t
    id4_pipeline_kernel_cache_test_executable_vtable = {
        /*.destroy=*/TestExecutableDestroy,
        /*.function_count=*/TestExecutableFunctionCount,
        /*.function_info=*/TestExecutableFunctionInfo,
        /*.function_parameters=*/TestExecutableFunctionParameters,
        /*.lookup_function_by_name=*/TestExecutableLookupFunctionByName,
        /*.try_lookup_global_by_name=*/TestExecutableTryLookupGlobalByName,
        /*.global_info=*/TestExecutableGlobalInfo,
        /*.global_buffer=*/TestExecutableGlobalBuffer,
};

typedef struct id4_pipeline_kernel_cache_test_executable_cache_t {
  // HAL resource header.
  iree_hal_resource_t resource;
  // Host allocator used for cache storage.
  iree_allocator_t host_allocator;
  // Number of infer-format calls observed.
  iree_host_size_t infer_count;
  // Number of can-prepare-format calls observed.
  iree_host_size_t can_prepare_count;
  // Number of prepare-executable calls observed.
  iree_host_size_t prepare_count;
  // Queue affinity from the most recent prepare call.
  iree_hal_queue_affinity_t last_queue_affinity;
  // Caching mode from the most recent prepare call.
  iree_hal_executable_caching_mode_t last_caching_mode;
  // Executable byte length from the most recent prepare call.
  iree_host_size_t last_executable_byte_length;
  // Byte length of |last_executable_format|.
  iree_host_size_t last_executable_format_length;
  // Executable format from the most recent prepare call.
  char last_executable_format[128];
} id4_pipeline_kernel_cache_test_executable_cache_t;

extern const iree_hal_executable_cache_vtable_t
    id4_pipeline_kernel_cache_test_executable_cache_vtable;

static id4_pipeline_kernel_cache_test_executable_cache_t* TestCacheCast(
    iree_hal_executable_cache_t* base_executable_cache) {
  IREE_HAL_ASSERT_TYPE(base_executable_cache,
                       &id4_pipeline_kernel_cache_test_executable_cache_vtable);
  return reinterpret_cast<id4_pipeline_kernel_cache_test_executable_cache_t*>(
      base_executable_cache);
}

static iree_status_t TestCacheCreate(
    iree_allocator_t host_allocator,
    id4_pipeline_kernel_cache_test_executable_cache_t** out_executable_cache) {
  *out_executable_cache = nullptr;
  id4_pipeline_kernel_cache_test_executable_cache_t* executable_cache = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*executable_cache),
                            reinterpret_cast<void**>(&executable_cache)));
  std::memset(executable_cache, 0, sizeof(*executable_cache));
  iree_hal_resource_initialize(
      &id4_pipeline_kernel_cache_test_executable_cache_vtable,
      &executable_cache->resource);
  executable_cache->host_allocator = host_allocator;
  *out_executable_cache = executable_cache;
  return iree_ok_status();
}

static void TestCacheDestroy(
    iree_hal_executable_cache_t* base_executable_cache) {
  id4_pipeline_kernel_cache_test_executable_cache_t* executable_cache =
      TestCacheCast(base_executable_cache);
  iree_allocator_t host_allocator = executable_cache->host_allocator;
  iree_allocator_free(host_allocator, executable_cache);
}

static iree_status_t TestCacheInferFormat(
    iree_hal_executable_cache_t* base_executable_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_inferred_size) {
  id4_pipeline_kernel_cache_test_executable_cache_t* executable_cache =
      TestCacheCast(base_executable_cache);
  ++executable_cache->infer_count;
  if (executable_data.data_length < 4 || executable_data.data[0] != 0x7f ||
      executable_data.data[1] != 'E' || executable_data.data[2] != 'L' ||
      executable_data.data[3] != 'F') {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected an ELF executable");
  }
  const iree_string_view_t format = IREE_SV("gfx1100");
  if (format.size >= executable_format_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "executable format buffer too small");
  }
  std::memcpy(executable_format, format.data, format.size);
  executable_format[format.size] = 0;
  *out_inferred_size = executable_data.data_length;
  return iree_ok_status();
}

static bool TestCacheCanPrepareFormat(
    iree_hal_executable_cache_t* base_executable_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format) {
  id4_pipeline_kernel_cache_test_executable_cache_t* executable_cache =
      TestCacheCast(base_executable_cache);
  ++executable_cache->can_prepare_count;
  return iree_string_view_equal(executable_format, IREE_SV("gfx1100"));
}

static iree_status_t TestCachePrepareExecutable(
    iree_hal_executable_cache_t* base_executable_cache,
    const iree_hal_executable_params_t* executable_params,
    iree_hal_executable_t** out_executable) {
  id4_pipeline_kernel_cache_test_executable_cache_t* executable_cache =
      TestCacheCast(base_executable_cache);
  ++executable_cache->prepare_count;
  executable_cache->last_queue_affinity = executable_params->queue_affinity;
  executable_cache->last_caching_mode = executable_params->caching_mode;
  executable_cache->last_executable_byte_length =
      executable_params->executable_data.data_length;
  if (executable_params->executable_format.size >=
      sizeof(executable_cache->last_executable_format)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "recorded executable format is too long");
  }
  executable_cache->last_executable_format_length =
      executable_params->executable_format.size;
  std::memcpy(executable_cache->last_executable_format,
              executable_params->executable_format.data,
              executable_params->executable_format.size);
  executable_cache
      ->last_executable_format[executable_params->executable_format.size] = 0;
  if (executable_params->executable_data.data_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "executable data is empty");
  }
  return TestExecutableCreate(executable_cache->host_allocator, out_executable);
}

const iree_hal_executable_cache_vtable_t
    id4_pipeline_kernel_cache_test_executable_cache_vtable = {
        /*.destroy=*/TestCacheDestroy,
        /*.infer_format=*/TestCacheInferFormat,
        /*.can_prepare_format=*/TestCacheCanPrepareFormat,
        /*.prepare_executable=*/TestCachePrepareExecutable,
};

class KernelCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    id4_pipeline_kernel_cache_create_options_t options = {
        /*.structure_size=*/sizeof(options),
        /*.next=*/nullptr,
        /*.amdgpu_processor=*/IREE_SV("gfx1100"),
    };
    IREE_ASSERT_OK(id4_pipeline_kernel_cache_create(
        &options, iree_allocator_system(), &kernel_cache_));
  }

  void TearDown() override { id4_pipeline_kernel_cache_release(kernel_cache_); }

  id4_pipeline_kernel_cache_t* kernel_cache_ = nullptr;
};

TEST_F(KernelCacheTest, PrepareConfiguredKernelToHalExecutable) {
  id4_pipeline_kernel_cache_test_executable_cache_t* executable_cache = nullptr;
  IREE_ASSERT_OK(TestCacheCreate(iree_allocator_system(), &executable_cache));

  id4_pipeline_kernel_config_binding_t config_bindings[] = {
      {
          /*.key=*/IREE_SV("id4.smoke.workgroups_x"),
          /*.value=*/IREE_SV("2"),
      },
      {
          /*.key=*/IREE_SV("id4.smoke.workgroup_size_x"),
          /*.value=*/IREE_SV("64"),
      },
  };
  id4_pipeline_kernel_cache_test_diagnostics_t diagnostics;
  id4_pipeline_diagnostics_sink_t diagnostics_sink =
      MakeDiagnosticsSink(&diagnostics);
  id4_pipeline_kernel_cache_prepare_options_t options = {
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.executable_cache=*/
      reinterpret_cast<iree_hal_executable_cache_t*>(executable_cache),
      /*.queue_affinity=*/IREE_HAL_QUEUE_AFFINITY_ANY,
      /*.caching_mode=*/IREE_HAL_EXECUTABLE_CACHING_MODE_NONE,
      /*.source_identifier=*/IREE_SV("id4_smoke_configured.loom"),
      /*.source_contents=*/
      iree_make_const_byte_span(kConfiguredSmokeKernelSource,
                                std::strlen(kConfiguredSmokeKernelSource)),
      /*.module_name=*/IREE_SV("id4_smoke_configured"),
      /*.executable_identifier=*/IREE_SV("id4_smoke_configured.hsaco"),
      /*.config_binding_count=*/IREE_ARRAYSIZE(config_bindings),
      /*.config_bindings=*/config_bindings,
      /*.diagnostic_artifact_flags=*/
      ID4_PIPELINE_KERNEL_DIAGNOSTIC_ARTIFACT_FLAG_MODULE_TEXT |
          ID4_PIPELINE_KERNEL_DIAGNOSTIC_ARTIFACT_FLAG_COMPILE_REPORT_JSON |
          ID4_PIPELINE_KERNEL_DIAGNOSTIC_ARTIFACT_FLAG_EMIT_MANIFEST_JSON,
      /*.diagnostics_sink=*/&diagnostics_sink,
  };

  id4_pipeline_kernel_executable_t* executable = nullptr;
  IREE_ASSERT_OK(id4_pipeline_kernel_cache_prepare_executable(
      kernel_cache_, &options, &executable));

  ASSERT_NE(id4_pipeline_kernel_executable_hal_executable(executable), nullptr);
  EXPECT_EQ(ToString(id4_pipeline_kernel_executable_hal_format(executable)),
            "gfx1100");
  const iree_const_byte_span_t primary_data =
      id4_pipeline_kernel_executable_primary_data(executable);
  ASSERT_GE(primary_data.data_length, 4u);
  EXPECT_EQ(primary_data.data[0], 0x7f);
  EXPECT_EQ(primary_data.data[1], 'E');
  EXPECT_EQ(primary_data.data[2], 'L');
  EXPECT_EQ(primary_data.data[3], 'F');

  bool found_hsaco = false;
  bool found_compile_report = false;
  for (iree_host_size_t i = 0;
       i < id4_pipeline_kernel_executable_artifact_count(executable); ++i) {
    const id4_pipeline_kernel_artifact_t* artifact =
        id4_pipeline_kernel_executable_artifact_at(executable, i);
    ASSERT_NE(artifact, nullptr);
    if (ToString(artifact->format) == "amdgpu-hsaco") {
      found_hsaco = true;
    }
    if (artifact->kind == ID4_PIPELINE_KERNEL_ARTIFACT_KIND_REPORT &&
        ToString(artifact->format) == "json") {
      const std::string report = ToString(artifact->contents);
      if (report.find(R"("kind":"loomc.compile")") != std::string::npos) {
        found_compile_report = true;
        EXPECT_NE(report.find(R"("config_binding_count":2)"),
                  std::string::npos);
      }
    }
  }
  EXPECT_TRUE(found_hsaco);
  EXPECT_TRUE(found_compile_report);
  EXPECT_EQ(executable_cache->infer_count, 1u);
  EXPECT_EQ(executable_cache->can_prepare_count, 1u);
  EXPECT_EQ(executable_cache->prepare_count, 1u);
  EXPECT_EQ(executable_cache->last_queue_affinity, IREE_HAL_QUEUE_AFFINITY_ANY);
  EXPECT_EQ(executable_cache->last_caching_mode,
            IREE_HAL_EXECUTABLE_CACHING_MODE_NONE);
  EXPECT_EQ(ToString(iree_make_string_view(
                executable_cache->last_executable_format,
                executable_cache->last_executable_format_length)),
            "gfx1100");
  EXPECT_EQ(executable_cache->last_executable_byte_length,
            primary_data.data_length);
  EXPECT_GE(diagnostics.kernel_event_count, 2u);

  id4_pipeline_kernel_executable_release(executable);
  iree_hal_executable_cache_release(
      reinterpret_cast<iree_hal_executable_cache_t*>(executable_cache));
}

TEST_F(KernelCacheTest, MissingConfigBindingFailsBeforeHalPrepare) {
  id4_pipeline_kernel_cache_test_executable_cache_t* executable_cache = nullptr;
  IREE_ASSERT_OK(TestCacheCreate(iree_allocator_system(), &executable_cache));

  id4_pipeline_kernel_config_binding_t config_bindings[] = {
      {
          /*.key=*/IREE_SV("id4.smoke.workgroups_x"),
          /*.value=*/IREE_SV("2"),
      },
  };
  id4_pipeline_kernel_cache_test_diagnostics_t diagnostics;
  id4_pipeline_diagnostics_sink_t diagnostics_sink =
      MakeDiagnosticsSink(&diagnostics);
  id4_pipeline_kernel_cache_prepare_options_t options = {
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.executable_cache=*/
      reinterpret_cast<iree_hal_executable_cache_t*>(executable_cache),
      /*.queue_affinity=*/IREE_HAL_QUEUE_AFFINITY_ANY,
      /*.caching_mode=*/IREE_HAL_EXECUTABLE_CACHING_MODE_NONE,
      /*.source_identifier=*/IREE_SV("id4_smoke_configured.loom"),
      /*.source_contents=*/
      iree_make_const_byte_span(kConfiguredSmokeKernelSource,
                                std::strlen(kConfiguredSmokeKernelSource)),
      /*.module_name=*/IREE_SV("id4_smoke_configured"),
      /*.executable_identifier=*/IREE_SV("id4_smoke_configured.hsaco"),
      /*.config_binding_count=*/IREE_ARRAYSIZE(config_bindings),
      /*.config_bindings=*/config_bindings,
      /*.diagnostic_artifact_flags=*/
      ID4_PIPELINE_KERNEL_DIAGNOSTIC_ARTIFACT_FLAG_COMPILE_REPORT_JSON,
      /*.diagnostics_sink=*/&diagnostics_sink,
  };

  id4_pipeline_kernel_executable_t* executable = nullptr;
  iree_status_t status = id4_pipeline_kernel_cache_prepare_executable(
      kernel_cache_, &options, &executable);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_FAILED_PRECONDITION);
  iree_status_free(status);
  EXPECT_EQ(executable, nullptr);
  EXPECT_EQ(executable_cache->infer_count, 0u);
  EXPECT_EQ(executable_cache->can_prepare_count, 0u);
  EXPECT_EQ(executable_cache->prepare_count, 0u);
  EXPECT_GT(diagnostics.loom_diagnostic_count, 0u);

  iree_hal_executable_cache_release(
      reinterpret_cast<iree_hal_executable_cache_t*>(executable_cache));
}

TEST_F(KernelCacheTest, MissingDiagnosticsSinkFailsBeforeHalPrepare) {
  id4_pipeline_kernel_cache_test_executable_cache_t* executable_cache = nullptr;
  IREE_ASSERT_OK(TestCacheCreate(iree_allocator_system(), &executable_cache));

  id4_pipeline_kernel_config_binding_t config_bindings[] = {
      {
          /*.key=*/IREE_SV("id4.smoke.workgroups_x"),
          /*.value=*/IREE_SV("2"),
      },
      {
          /*.key=*/IREE_SV("id4.smoke.workgroup_size_x"),
          /*.value=*/IREE_SV("64"),
      },
  };
  id4_pipeline_kernel_cache_prepare_options_t options = {
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.executable_cache=*/
      reinterpret_cast<iree_hal_executable_cache_t*>(executable_cache),
      /*.queue_affinity=*/IREE_HAL_QUEUE_AFFINITY_ANY,
      /*.caching_mode=*/IREE_HAL_EXECUTABLE_CACHING_MODE_NONE,
      /*.source_identifier=*/IREE_SV("id4_smoke_configured.loom"),
      /*.source_contents=*/
      iree_make_const_byte_span(kConfiguredSmokeKernelSource,
                                std::strlen(kConfiguredSmokeKernelSource)),
      /*.module_name=*/IREE_SV("id4_smoke_configured"),
      /*.executable_identifier=*/IREE_SV("id4_smoke_configured.hsaco"),
      /*.config_binding_count=*/IREE_ARRAYSIZE(config_bindings),
      /*.config_bindings=*/config_bindings,
      /*.diagnostic_artifact_flags=*/
      ID4_PIPELINE_KERNEL_DIAGNOSTIC_ARTIFACT_FLAG_COMPILE_REPORT_JSON,
      /*.diagnostics_sink=*/nullptr,
  };

  id4_pipeline_kernel_executable_t* executable = nullptr;
  iree_status_t status = id4_pipeline_kernel_cache_prepare_executable(
      kernel_cache_, &options, &executable);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT);
  iree_status_free(status);
  EXPECT_EQ(executable, nullptr);
  EXPECT_EQ(executable_cache->infer_count, 0u);
  EXPECT_EQ(executable_cache->can_prepare_count, 0u);
  EXPECT_EQ(executable_cache->prepare_count, 0u);

  iree_hal_executable_cache_release(
      reinterpret_cast<iree_hal_executable_cache_t*>(executable_cache));
}

}  // namespace
