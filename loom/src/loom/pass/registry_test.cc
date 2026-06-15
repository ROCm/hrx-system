// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/registry.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

const loom_pass_info_t* AlphaPassInfo(void) {
  static const loom_pass_info_t kInfo = {
      /*.name=*/IREE_SVL("alpha"),
      /*.description=*/IREE_SVL("Alpha pass."),
      /*.kind=*/LOOM_PASS_FUNCTION,
  };
  return &kInfo;
}

const loom_pass_info_t* BetaPassInfo(void) {
  static const loom_pass_info_t kInfo = {
      /*.name=*/IREE_SVL("beta"),
      /*.description=*/IREE_SVL("Beta pass."),
      /*.kind=*/LOOM_PASS_FUNCTION,
  };
  return &kInfo;
}

const loom_pass_info_t* GammaPassInfo(void) {
  static const loom_pass_info_t kInfo = {
      /*.name=*/IREE_SVL("gamma"),
      /*.description=*/IREE_SVL("Gamma pass."),
      /*.kind=*/LOOM_PASS_FUNCTION,
  };
  return &kInfo;
}

iree_status_t NoopFunctionPass(loom_pass_t* pass, loom_module_t* module,
                               loom_func_like_t function) {
  return iree_ok_status();
}

static loom_pass_descriptor_t MakeFunctionPassDescriptor(
    iree_string_view_t key, loom_pass_info_fn_t info) {
  loom_pass_descriptor_t descriptor = {};
  descriptor.key = key;
  descriptor.info = info;
  descriptor.function_run = NoopFunctionPass;
  return descriptor;
}

TEST(PassRegistryStorageTest, MergesSortedRegistries) {
  static const loom_pass_descriptor_t kFirstDescriptors[] = {
      MakeFunctionPassDescriptor(IREE_SV("alpha"), AlphaPassInfo),
      MakeFunctionPassDescriptor(IREE_SV("gamma"), GammaPassInfo),
  };
  static const loom_pass_descriptor_t kSecondDescriptors[] = {
      MakeFunctionPassDescriptor(IREE_SV("beta"), BetaPassInfo),
  };
  static const loom_pass_registry_t kFirstRegistry = {
      /*.descriptors=*/kFirstDescriptors,
      /*.descriptor_count=*/IREE_ARRAYSIZE(kFirstDescriptors),
  };
  static const loom_pass_registry_t kSecondRegistry = {
      /*.descriptors=*/kSecondDescriptors,
      /*.descriptor_count=*/IREE_ARRAYSIZE(kSecondDescriptors),
  };
  const loom_pass_registry_t* registries[] = {
      &kFirstRegistry,
      &kSecondRegistry,
  };

  loom_pass_registry_storage_t storage;
  IREE_ASSERT_OK(loom_pass_registry_storage_initialize_from_registries(
      registries, IREE_ARRAYSIZE(registries), &storage));
  const loom_pass_registry_t* registry =
      loom_pass_registry_storage_registry(&storage);

  ASSERT_EQ(registry->descriptor_count, 3);
  EXPECT_TRUE(
      iree_string_view_equal(registry->descriptors[0].key, IREE_SV("alpha")));
  EXPECT_TRUE(
      iree_string_view_equal(registry->descriptors[1].key, IREE_SV("beta")));
  EXPECT_TRUE(
      iree_string_view_equal(registry->descriptors[2].key, IREE_SV("gamma")));

  const loom_pass_descriptor_t* descriptor = nullptr;
  IREE_ASSERT_OK(
      loom_pass_registry_lookup(registry, IREE_SV("beta"), &descriptor));
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor->info, BetaPassInfo);
}

TEST(PassRegistryStorageTest, RejectsDuplicateKeys) {
  static const loom_pass_descriptor_t kFirstDescriptors[] = {
      MakeFunctionPassDescriptor(IREE_SV("alpha"), AlphaPassInfo),
  };
  static const loom_pass_descriptor_t kSecondDescriptors[] = {
      MakeFunctionPassDescriptor(IREE_SV("alpha"), AlphaPassInfo),
  };
  static const loom_pass_registry_t kFirstRegistry = {
      /*.descriptors=*/kFirstDescriptors,
      /*.descriptor_count=*/IREE_ARRAYSIZE(kFirstDescriptors),
  };
  static const loom_pass_registry_t kSecondRegistry = {
      /*.descriptors=*/kSecondDescriptors,
      /*.descriptor_count=*/IREE_ARRAYSIZE(kSecondDescriptors),
  };
  const loom_pass_registry_t* registries[] = {
      &kFirstRegistry,
      &kSecondRegistry,
  };

  loom_pass_registry_storage_t storage;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ALREADY_EXISTS,
                        loom_pass_registry_storage_initialize_from_registries(
                            registries, IREE_ARRAYSIZE(registries), &storage));
}

}  // namespace
}  // namespace loom
