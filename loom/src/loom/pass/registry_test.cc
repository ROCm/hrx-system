// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/registry.h"

#include <array>
#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

const loom_pass_info_t* AlphaPassInfo(void) {
  static const loom_pass_info_t kInfo = {
      .name = IREE_SVL("alpha"),
      .description = IREE_SVL("Alpha pass."),
      .kind = LOOM_PASS_MODULE,
  };
  return &kInfo;
}

const loom_pass_info_t* BetaPassInfo(void) {
  static const loom_pass_info_t kInfo = {
      .name = IREE_SVL("beta"),
      .description = IREE_SVL("Beta pass."),
      .kind = LOOM_PASS_MODULE,
  };
  return &kInfo;
}

const loom_pass_info_t* GammaPassInfo(void) {
  static const loom_pass_info_t kInfo = {
      .name = IREE_SVL("gamma"),
      .description = IREE_SVL("Gamma pass."),
      .kind = LOOM_PASS_MODULE,
  };
  return &kInfo;
}

iree_status_t NoopModulePass(loom_pass_t* pass, loom_module_t* module) {
  return iree_ok_status();
}

static loom_pass_descriptor_t MakeModulePassDescriptor(
    iree_string_view_t key, loom_pass_info_fn_t info) {
  loom_pass_descriptor_t descriptor = {
      .key = key,
      .info = info,
      .module_run = NoopModulePass,
  };
  return descriptor;
}

TEST(PassRegistryStorageTest, MergesSortedRegistries) {
  static const loom_pass_descriptor_t kFirstDescriptors[] = {
      MakeModulePassDescriptor(IREE_SV("alpha"), AlphaPassInfo),
      MakeModulePassDescriptor(IREE_SV("gamma"), GammaPassInfo),
  };
  static const loom_pass_descriptor_t kSecondDescriptors[] = {
      MakeModulePassDescriptor(IREE_SV("beta"), BetaPassInfo),
  };
  static const loom_pass_registry_t kFirstRegistry = {
      .descriptors = kFirstDescriptors,
      .descriptor_count = IREE_ARRAYSIZE(kFirstDescriptors),
  };
  static const loom_pass_registry_t kSecondRegistry = {
      .descriptors = kSecondDescriptors,
      .descriptor_count = IREE_ARRAYSIZE(kSecondDescriptors),
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
  EXPECT_TRUE(iree_string_view_equal(loom_pass_registry_at(registry, 0)->key,
                                     IREE_SV("alpha")));
  EXPECT_TRUE(iree_string_view_equal(loom_pass_registry_at(registry, 1)->key,
                                     IREE_SV("beta")));
  EXPECT_TRUE(iree_string_view_equal(loom_pass_registry_at(registry, 2)->key,
                                     IREE_SV("gamma")));

  const loom_pass_descriptor_t* descriptor =
      loom_pass_registry_lookup(registry, IREE_SV("beta"));
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(descriptor, &kSecondDescriptors[0]);
  EXPECT_EQ(descriptor->info, &BetaPassInfo);
}

TEST(PassRegistryStorageTest, RejectsDuplicateKeys) {
  static const loom_pass_descriptor_t kFirstDescriptors[] = {
      MakeModulePassDescriptor(IREE_SV("alpha"), AlphaPassInfo),
  };
  static const loom_pass_descriptor_t kSecondDescriptors[] = {
      MakeModulePassDescriptor(IREE_SV("alpha"), AlphaPassInfo),
  };
  static const loom_pass_registry_t kFirstRegistry = {
      .descriptors = kFirstDescriptors,
      .descriptor_count = IREE_ARRAYSIZE(kFirstDescriptors),
  };
  static const loom_pass_registry_t kSecondRegistry = {
      .descriptors = kSecondDescriptors,
      .descriptor_count = IREE_ARRAYSIZE(kSecondDescriptors),
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

TEST(PassRegistryStorageTest, EmptyCompositionResetsStorage) {
  const loom_pass_descriptor_t descriptor = {
      .key = IREE_SVL("alpha"),
  };
  const loom_pass_registry_t input = {
      .descriptors = &descriptor,
      .descriptor_count = 1,
  };
  const loom_pass_registry_t* populated[] = {&input};
  loom_pass_registry_storage_t storage;
  IREE_ASSERT_OK(loom_pass_registry_storage_initialize_from_registries(
      populated, IREE_ARRAYSIZE(populated), &storage));
  ASSERT_EQ(storage.registry.descriptor_count, 1u);

  const loom_pass_registry_t empty = {};
  const loom_pass_registry_t* empty_inputs[] = {nullptr, &empty, nullptr};
  IREE_ASSERT_OK(loom_pass_registry_storage_initialize_from_registries(
      empty_inputs, IREE_ARRAYSIZE(empty_inputs), &storage));
  EXPECT_EQ(storage.registry.descriptor_count, 0u);
  EXPECT_EQ(loom_pass_registry_lookup(&storage.registry, IREE_SV("alpha")),
            nullptr);
  IREE_ASSERT_OK(loom_pass_registry_storage_initialize_from_registries(
      nullptr, 0, &storage));
  EXPECT_EQ(storage.registry.descriptor_count, 0u);
}

TEST(PassRegistryStorageTest, ComposesBorrowedDescriptorReferences) {
  const loom_pass_descriptor_t descriptors[] = {
      {.key = IREE_SVL("alpha")},
      {.key = IREE_SVL("beta")},
      {.key = IREE_SVL("gamma")},
  };
  const loom_pass_descriptor_t* first_refs[] = {&descriptors[0],
                                                &descriptors[2]};
  const loom_pass_registry_t first = {
      .descriptor_count = IREE_ARRAYSIZE(first_refs),
      .descriptor_refs = first_refs,
  };
  const loom_pass_registry_t second = {
      .descriptors = &descriptors[1],
      .descriptor_count = 1,
  };
  const loom_pass_registry_t* inputs[] = {nullptr, &second, &first};
  loom_pass_registry_storage_t storage;
  IREE_ASSERT_OK(loom_pass_registry_storage_initialize_from_registries(
      inputs, IREE_ARRAYSIZE(inputs), &storage));
  ASSERT_EQ(storage.registry.descriptor_count, IREE_ARRAYSIZE(descriptors));
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(descriptors); ++i) {
    EXPECT_EQ(loom_pass_registry_at(&storage.registry, i), &descriptors[i]);
    EXPECT_EQ(loom_pass_registry_lookup(&storage.registry, descriptors[i].key),
              &descriptors[i]);
  }

  const loom_pass_registry_t* duplicate_inputs[] = {&first, &storage.registry};
  loom_pass_registry_storage_t duplicate_storage;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ALREADY_EXISTS,
                        loom_pass_registry_storage_initialize_from_registries(
                            duplicate_inputs, IREE_ARRAYSIZE(duplicate_inputs),
                            &duplicate_storage));
}

TEST(PassRegistryStorageTest, OrdersProvidersThroughCapacity) {
  constexpr size_t kCapacity = LOOM_PASS_REGISTRY_STORAGE_DESCRIPTOR_CAPACITY;
  std::array<std::string, kCapacity> keys;
  std::array<loom_pass_descriptor_t, kCapacity> descriptors = {};
  std::array<const loom_pass_descriptor_t*, kCapacity> references = {};
  std::array<loom_pass_registry_t, kCapacity> providers = {};
  std::array<const loom_pass_registry_t*, kCapacity> inputs = {};
  for (size_t i = 0; i < kCapacity; ++i) {
    keys[i] = "pass-" + std::to_string(1000 + i);
    descriptors[i].key = iree_make_cstring_view(keys[i].c_str());
  }

  for (size_t count : {1u, 2u, 63u, 64u, 65u, 128u, 256u}) {
    SCOPED_TRACE(count);
    const size_t provider_counts[] = {1, 2, 3, count};
    for (size_t provider_count : provider_counts) {
      SCOPED_TRACE(provider_count);
      size_t position = 0;
      for (size_t provider = 0; provider < provider_count; ++provider) {
        const size_t start = position;
        for (size_t i = provider; i < count; i += provider_count) {
          references[position++] = &descriptors[i];
        }
        providers[provider] = {
            .descriptor_count = position - start,
            .descriptor_refs = references.data() + start,
        };
      }
      for (bool reverse : {false, true}) {
        SCOPED_TRACE(reverse);
        for (size_t i = 0; i < provider_count; ++i) {
          inputs[i] = &providers[reverse ? provider_count - i - 1 : i];
        }
        loom_pass_registry_storage_t storage;
        IREE_ASSERT_OK(loom_pass_registry_storage_initialize_from_registries(
            inputs.data(), provider_count, &storage));
        ASSERT_EQ(storage.registry.descriptor_count, count);
        for (size_t i = 0; i < count; ++i) {
          EXPECT_EQ(loom_pass_registry_at(&storage.registry, i),
                    &descriptors[i]);
          EXPECT_EQ(
              loom_pass_registry_lookup(&storage.registry, descriptors[i].key),
              &descriptors[i]);
        }
      }
    }
  }
}

TEST(PassRegistryStorageTest, RejectsCompositionBeyondCapacity) {
  constexpr size_t kCapacity = LOOM_PASS_REGISTRY_STORAGE_DESCRIPTOR_CAPACITY;
  std::array<std::string, kCapacity + 1> keys;
  std::array<loom_pass_descriptor_t, kCapacity + 1> descriptors = {};
  for (size_t i = 0; i < descriptors.size(); ++i) {
    keys[i] = "pass-" + std::to_string(1000 + i);
    descriptors[i].key = iree_make_cstring_view(keys[i].c_str());
  }
  const loom_pass_registry_t oversized = {
      .descriptors = descriptors.data(),
      .descriptor_count = descriptors.size(),
  };
  const loom_pass_registry_t full = {
      .descriptors = descriptors.data(),
      .descriptor_count = kCapacity,
  };
  const loom_pass_registry_t extra = {
      .descriptors = &descriptors[kCapacity],
      .descriptor_count = 1,
  };
  const loom_pass_registry_t* one_input[] = {&oversized};
  const loom_pass_registry_t* two_inputs[] = {&full, &extra};
  loom_pass_registry_storage_t storage;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_pass_registry_storage_initialize_from_registries(
                            one_input, IREE_ARRAYSIZE(one_input), &storage));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_pass_registry_storage_initialize_from_registries(
                            two_inputs, IREE_ARRAYSIZE(two_inputs), &storage));
}

}  // namespace
}  // namespace loom
