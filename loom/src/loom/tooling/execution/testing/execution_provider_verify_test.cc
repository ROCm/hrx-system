// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/testing/execution_provider_verify.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/low_descriptor_registry_core_test.h"
#include "loom/tooling/execution/execution_provider.h"

namespace loom {
namespace {

iree_status_t FakeExecutionBackendRunOneShot(
    const loom_run_execution_backend_t* backend,
    const loom_run_one_shot_request_t* request) {
  (void)backend;
  (void)request;
  return iree_ok_status();
}

const loom_run_execution_backend_t kFakeExecutionBackend = {
    /*.name=*/IREE_SVL("fake-execution"),
    /*.flags=*/LOOM_RUN_EXECUTION_BACKEND_FLAG_VM_OPTIONS,
    /*.probe=*/{},
    /*.run_one_shot=*/FakeExecutionBackendRunOneShot,
};

const loom_run_execution_backend_t kDuplicateFakeExecutionBackend = {
    /*.name=*/IREE_SVL("fake-execution"),
    /*.flags=*/LOOM_RUN_EXECUTION_BACKEND_FLAG_VM_OPTIONS,
    /*.probe=*/{},
    /*.run_one_shot=*/FakeExecutionBackendRunOneShot,
};

const loom_run_execution_backend_t* const kFakeExecutionBackends[] = {
    &kFakeExecutionBackend,
};

const loom_run_execution_backend_t* const kDuplicateFakeExecutionBackends[] = {
    &kFakeExecutionBackend,
    &kDuplicateFakeExecutionBackend,
};

const loom_target_provider_t kCoreTestTargetProvider = {
    /*.register_context=*/{}, /*.initialize_low_descriptor_registry=*/
    loom_target_core_test_low_descriptor_registry_initialize,
};

const loom_run_execution_provider_t kCoreTestProvider = {
    /*.name=*/IREE_SVL("core-test"),
    /*.target_provider=*/&kCoreTestTargetProvider,
    /*.execution_backends=*/kFakeExecutionBackends,
    /*.execution_backend_count=*/IREE_ARRAYSIZE(kFakeExecutionBackends),
};

const loom_run_execution_provider_t kDuplicateCoreTestProvider = {
    /*.name=*/IREE_SVL("core-test"),
};

const loom_run_execution_provider_t kDuplicateExecutionProvider = {
    /*.name=*/IREE_SVL("duplicate-execution"),
    /*.target_provider=*/{},
    /*.execution_backends=*/kDuplicateFakeExecutionBackends,
    /*.execution_backend_count=*/
    IREE_ARRAYSIZE(kDuplicateFakeExecutionBackends),
};

TEST(ExecutionProviderTest, ComposesDescriptorRegistryAndExecutionBackends) {
  const loom_run_execution_provider_t* providers[] = {
      &kCoreTestProvider,
  };
  const loom_run_execution_provider_set_t provider_set = {
      /*.providers=*/providers,
      /*.provider_count=*/IREE_ARRAYSIZE(providers),
  };
  IREE_ASSERT_OK(loom_run_execution_provider_set_verify(&provider_set));

  loom_run_execution_environment_t environment = {};
  IREE_ASSERT_OK(
      loom_run_execution_environment_initialize(&provider_set, &environment));

  const loom_run_execution_backend_registry_t* execution_backend_registry =
      loom_run_execution_environment_execution_backend_registry(&environment);
  ASSERT_NE(execution_backend_registry, nullptr);
  EXPECT_EQ(execution_backend_registry->backend_count, 1u);
  EXPECT_EQ(loom_run_execution_backend_registry_lookup(
                execution_backend_registry, IREE_SV("fake-execution")),
            &kFakeExecutionBackend);

  loom_target_low_descriptor_registry_t low_registry = {};
  const loom_run_initialize_low_descriptor_registry_callback_t callback =
      loom_run_execution_environment_low_descriptor_registry_callback(
          &environment);
  IREE_ASSERT_OK(callback.fn(callback.user_data, &low_registry));
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_descriptor_registry_lookup(&low_registry.registry,
                                          IREE_SV("test.low.core"));
  EXPECT_NE(descriptor_set, nullptr);

  loom_run_execution_environment_deinitialize(&environment);
}

TEST(ExecutionProviderTest, RejectsDuplicateProviderNames) {
  const loom_run_execution_provider_t* providers[] = {
      &kCoreTestProvider,
      &kDuplicateCoreTestProvider,
  };
  const loom_run_execution_provider_set_t provider_set = {
      /*.providers=*/providers,
      /*.provider_count=*/IREE_ARRAYSIZE(providers),
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_run_execution_provider_set_verify(&provider_set));
}

TEST(ExecutionProviderTest, RejectsDuplicateExecutionBackendNames) {
  const loom_run_execution_provider_t* providers[] = {
      &kDuplicateExecutionProvider,
  };
  const loom_run_execution_provider_set_t provider_set = {
      /*.providers=*/providers,
      /*.provider_count=*/IREE_ARRAYSIZE(providers),
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_run_execution_provider_set_verify(&provider_set));
}

}  // namespace
}  // namespace loom
