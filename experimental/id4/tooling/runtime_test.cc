// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/tooling/runtime.h"

#include <cstring>
#include <memory>
#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

template <typename T, void (*Release)(T*)>
class Ref {
 public:
  Ref() = default;
  Ref(const Ref&) = delete;
  Ref& operator=(const Ref&) = delete;

  ~Ref() { reset(); }

  T* get() const { return value_; }

  T** out() {
    reset();
    return &value_;
  }

  void reset(T* value = nullptr) {
    if (value_) Release(value_);
    value_ = value;
  }

 private:
  T* value_ = nullptr;
};

typedef struct id4_tooling_runtime_test_provider_t {
  iree_io_parameter_provider_t base;
  iree_string_view_t scope;
  int notify_count;
  int gather_count;
  iree_string_view_t last_gather_scope;
} id4_tooling_runtime_test_provider_t;

static id4_tooling_runtime_test_provider_t* TestProviderCast(
    iree_io_parameter_provider_t* provider) {
  return reinterpret_cast<id4_tooling_runtime_test_provider_t*>(provider);
}

static void TestProviderDestroy(iree_io_parameter_provider_t* provider) {}

static iree_status_t TestProviderNotify(
    iree_io_parameter_provider_t* provider,
    iree_io_parameter_provider_signal_t signal) {
  ++TestProviderCast(provider)->notify_count;
  return iree_ok_status();
}

static bool TestProviderQuerySupport(iree_io_parameter_provider_t* provider,
                                     iree_string_view_t scope) {
  return iree_string_view_equal(scope, TestProviderCast(provider)->scope);
}

static iree_status_t TestProviderLoad(
    iree_io_parameter_provider_t* provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_string_view_t source_scope, iree_hal_buffer_params_t target_params,
    iree_host_size_t count, iree_io_parameter_enumerator_t enumerator,
    iree_io_parameter_emitter_t emitter) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

static iree_status_t TestProviderGather(
    iree_io_parameter_provider_t* provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_string_view_t source_scope, iree_hal_buffer_t* target_buffer,
    iree_host_size_t count, iree_io_parameter_enumerator_t enumerator) {
  id4_tooling_runtime_test_provider_t* test_provider =
      TestProviderCast(provider);
  ++test_provider->gather_count;
  test_provider->last_gather_scope = source_scope;
  return iree_ok_status();
}

static iree_status_t TestProviderScatter(
    iree_io_parameter_provider_t* provider, iree_hal_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_string_view_t target_scope,
    iree_host_size_t count, iree_io_parameter_enumerator_t enumerator) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED);
}

static const iree_io_parameter_provider_vtable_t kTestProviderVTable = {
    // Test providers are stack-owned by each test.
    /*.destroy=*/TestProviderDestroy,
    // Records lifecycle notifications.
    /*.notify=*/TestProviderNotify,
    // Supports exactly one configured scope.
    /*.query_support=*/TestProviderQuerySupport,
    // Load is outside this wrapper test's scope.
    /*.load=*/TestProviderLoad,
    // Records gather routing.
    /*.gather=*/TestProviderGather,
    // Scatter is outside this wrapper test's scope.
    /*.scatter=*/TestProviderScatter,
};

static void TestProviderInitialize(
    iree_string_view_t scope,
    id4_tooling_runtime_test_provider_t* out_provider) {
  std::memset(out_provider, 0, sizeof(*out_provider));
  iree_atomic_ref_count_init(&out_provider->base.ref_count);
  out_provider->base.vtable = &kTestProviderVTable;
  out_provider->scope = scope;
}

TEST(Id4ToolingRuntimeTest, CreatesEmbeddedKernelLibrary) {
  Ref<id4_pipeline_kernel_library_t, id4_pipeline_kernel_library_release>
      library;
  IREE_ASSERT_OK(id4_tooling_create_embedded_kernel_library(
      iree_allocator_system(), library.out()));

  const id4_pipeline_kernel_module_t* module = nullptr;
  IREE_ASSERT_OK(id4_pipeline_kernel_library_lookup(
      library.get(), IREE_SV("qwen3_vl/rmsnorm"), &module));
  ASSERT_NE(module, nullptr);
  EXPECT_TRUE(
      iree_string_view_equal(module->module_path, IREE_SV("qwen3_vl/rmsnorm")));
}

TEST(Id4ToolingRuntimeTest, RequiresDeviceFlag) {
  id4_tooling_runtime_context_options_t options;
  std::memset(&options, 0, sizeof(options));
  options.structure_size = sizeof(options);
  options.executable_cache_identifier = IREE_SV("id4.test");

  id4_tooling_runtime_context_t context;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_tooling_runtime_context_initialize_from_flags(
                            &options, iree_allocator_system(), &context));
}

TEST(Id4ToolingRuntimeTest, ParameterProviderSetRoutesByScope) {
  id4_tooling_runtime_test_provider_t bf16_provider;
  TestProviderInitialize(IREE_SV("dit_bf16"), &bf16_provider);
  id4_tooling_runtime_test_provider_t fp8_provider;
  TestProviderInitialize(IREE_SV("dit_fp8"), &fp8_provider);

  id4_tooling_parameter_provider_set_entry_t entries[] = {
      {
          // BF16-expanded parameter source scope.
          /*.scope=*/IREE_SV("dit_bf16"),
          // Provider backing the BF16-expanded source scope.
          /*.provider=*/&bf16_provider.base,
      },
      {
          // FP8 e4m3 parameter source scope.
          /*.scope=*/IREE_SV("dit_fp8"),
          // Provider backing the FP8 e4m3 source scope.
          /*.provider=*/&fp8_provider.base,
      },
  };
  Ref<iree_io_parameter_provider_t, iree_io_parameter_provider_release>
      provider_set;
  IREE_ASSERT_OK(id4_tooling_create_parameter_provider_set(
      IREE_ARRAYSIZE(entries), entries, iree_allocator_system(),
      provider_set.out()));

  EXPECT_TRUE(iree_io_parameter_provider_query_support(provider_set.get(),
                                                       IREE_SV("dit_bf16")));
  EXPECT_TRUE(iree_io_parameter_provider_query_support(provider_set.get(),
                                                       IREE_SV("dit_fp8")));
  EXPECT_FALSE(iree_io_parameter_provider_query_support(provider_set.get(),
                                                        IREE_SV("missing")));

  Ref<iree_hal_allocator_t, iree_hal_allocator_release> allocator;
  IREE_ASSERT_OK(iree_hal_allocator_create_heap(
      IREE_SV("id4-provider-set-test"), iree_allocator_system(),
      iree_allocator_system(), allocator.out()));
  iree_hal_buffer_params_t target_params = {};
  target_params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  target_params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  target_params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING;
  Ref<iree_hal_buffer_t, iree_hal_buffer_release> target_buffer;
  IREE_ASSERT_OK(iree_hal_allocator_allocate_buffer(
      allocator.get(), target_params, /*allocation_size=*/4,
      target_buffer.out()));

  iree_io_parameter_enumerator_t enumerator = {};
  IREE_ASSERT_OK(iree_io_parameter_provider_gather(
      provider_set.get(), /*device=*/nullptr, IREE_HAL_QUEUE_AFFINITY_ANY,
      iree_hal_semaphore_list_empty(), iree_hal_semaphore_list_empty(),
      IREE_SV("dit_fp8"), target_buffer.get(), /*count=*/1, enumerator));
  EXPECT_EQ(bf16_provider.gather_count, 0);
  EXPECT_EQ(fp8_provider.gather_count, 1);
  EXPECT_TRUE(iree_string_view_equal(fp8_provider.last_gather_scope,
                                     IREE_SV("dit_fp8")));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      iree_io_parameter_provider_gather(
          provider_set.get(), /*device=*/nullptr, IREE_HAL_QUEUE_AFFINITY_ANY,
          iree_hal_semaphore_list_empty(), iree_hal_semaphore_list_empty(),
          IREE_SV("missing"), target_buffer.get(), /*count=*/1, enumerator));
}

TEST(Id4ToolingRuntimeTest, ParameterProviderSetRejectsDuplicateScopes) {
  id4_tooling_runtime_test_provider_t first_provider;
  TestProviderInitialize(IREE_SV("dit"), &first_provider);
  id4_tooling_runtime_test_provider_t second_provider;
  TestProviderInitialize(IREE_SV("dit"), &second_provider);

  id4_tooling_parameter_provider_set_entry_t entries[] = {
      {
          // First provider entry for the duplicated source scope.
          /*.scope=*/IREE_SV("dit"),
          // First provider backing the duplicated source scope.
          /*.provider=*/&first_provider.base,
      },
      {
          // Second provider entry for the duplicated source scope.
          /*.scope=*/IREE_SV("dit"),
          // Second provider backing the duplicated source scope.
          /*.provider=*/&second_provider.base,
      },
  };
  Ref<iree_io_parameter_provider_t, iree_io_parameter_provider_release>
      provider_set;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        id4_tooling_create_parameter_provider_set(
                            IREE_ARRAYSIZE(entries), entries,
                            iree_allocator_system(), provider_set.out()));
}

TEST(Id4ToolingRuntimeTest, RequiresParameterScopeFromFlags) {
  Ref<iree_io_parameter_provider_t, iree_io_parameter_provider_release>
      provider;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      id4_tooling_create_parameter_provider_from_flags(
          iree_string_view_empty(), iree_allocator_system(), provider.out()));
}

TEST(Id4ToolingRuntimeTest, BatchProviderCreationRequiresRequestedScopes) {
  Ref<iree_io_parameter_provider_t, iree_io_parameter_provider_release>
      qwen_provider;
  Ref<iree_io_parameter_provider_t, iree_io_parameter_provider_release>
      vae_provider;

  id4_tooling_parameter_provider_request_t requests[] = {
      {
          // Qwen parameter scope required by the caller.
          /*.scope=*/IREE_SV("qwen"),
          // Qwen provider output.
          /*.out_provider=*/qwen_provider.out(),
      },
      {
          // VAE parameter scope required by the caller.
          /*.scope=*/IREE_SV("vae"),
          // VAE provider output.
          /*.out_provider=*/vae_provider.out(),
      },
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      id4_tooling_create_parameter_providers_from_flags(
          IREE_ARRAYSIZE(requests), requests, iree_allocator_system()));
  EXPECT_EQ(qwen_provider.get(), nullptr);
  EXPECT_EQ(vae_provider.get(), nullptr);
}

}  // namespace
