// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/configured/compiler_provider_set.h"

#include <unordered_set>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/descriptor_scope.h"
#include "loom/target/selection.h"

#ifndef LOOM_CONFIG_COMPILER_HAVE_X86
#define LOOM_CONFIG_COMPILER_HAVE_X86 0
#endif  // LOOM_CONFIG_COMPILER_HAVE_X86
#ifndef LOOM_CONFIG_COMPILER_HAVE_AMDGPU
#define LOOM_CONFIG_COMPILER_HAVE_AMDGPU 0
#endif  // LOOM_CONFIG_COMPILER_HAVE_AMDGPU
#ifndef LOOM_CONFIG_COMPILER_HAVE_SPIRV
#define LOOM_CONFIG_COMPILER_HAVE_SPIRV 0
#endif  // LOOM_CONFIG_COMPILER_HAVE_SPIRV
#ifndef LOOM_CONFIG_COMPILER_HAVE_VM
#define LOOM_CONFIG_COMPILER_HAVE_VM 0
#endif  // LOOM_CONFIG_COMPILER_HAVE_VM
#ifndef LOOM_CONFIG_COMPILER_HAVE_WASM
#define LOOM_CONFIG_COMPILER_HAVE_WASM 0
#endif  // LOOM_CONFIG_COMPILER_HAVE_WASM
#ifndef LOOM_CONFIG_COMPILER_HAVE_XDNA
#define LOOM_CONFIG_COMPILER_HAVE_XDNA 0
#endif  // LOOM_CONFIG_COMPILER_HAVE_XDNA

namespace loom {
namespace {

template <typename ProviderList>
void ExpectOwnedScopes(ProviderList providers,
                       const loom_target_low_descriptor_registry_t& descriptors,
                       std::unordered_set<const void*>& contributed_providers) {
  for (iree_host_size_t i = 0; i < providers.count; ++i) {
    const auto* provider = providers.values[i];
    SCOPED_TRACE(std::string(provider->name.data, provider->name.size));
    EXPECT_TRUE(contributed_providers.insert(provider).second);
    EXPECT_TRUE(loom_low_descriptor_scope_is_valid(provider->scope));
    EXPECT_NE(provider->scope.target_stable_id, UINT64_MAX)
        << "shared providers belong to common registry composition";
    bool owned = false;
    for (iree_host_size_t j = 0; j < descriptors.descriptor_set_provider_count;
         ++j) {
      owned |= loom_low_descriptor_scope_matches(
          provider->scope, descriptors.descriptor_set_providers[j]());
    }
    EXPECT_TRUE(owned) << "scope must belong to the contributing target";
  }
}

template <typename Registry>
void ExpectOwnedPolicies(
    const Registry& policies,
    const loom_target_low_descriptor_registry_t& descriptors,
    std::unordered_set<uint64_t>& contributed_keys) {
  for (iree_host_size_t i = 0; i < policies.entry_count; ++i) {
    const auto& entry = policies.entries[i];
    SCOPED_TRACE(entry.contract_set_stable_id);
    EXPECT_TRUE(contributed_keys.insert(entry.contract_set_stable_id).second);
    EXPECT_NE(entry.policy, nullptr);
    bool owned = false;
    for (iree_host_size_t j = 0; j < descriptors.descriptor_set_provider_count;
         ++j) {
      owned |= descriptors.descriptor_set_providers[j]()->stable_id ==
               entry.contract_set_stable_id;
    }
    EXPECT_TRUE(owned)
        << "policy must name a descriptor set owned by its target";
  }
}

TEST(ConfiguredCompilerProviderSetTest, ContributionsHaveUniqueTargetOwners) {
  const auto* providers = loom_configured_compiler_provider_set();
  std::unordered_set<uint64_t> descriptor_keys;
  std::unordered_set<uint64_t> lower_policy_keys;
  std::unordered_set<uint64_t> math_policy_keys;
  std::unordered_set<const void*> legalizers;
  std::unordered_set<const void*> legality;
  std::unordered_set<const void*> packet_diagnostics;
  std::unordered_set<const void*> asm_diagnostics;
  std::unordered_set<const void*> verifiers;
  for (iree_host_size_t i = 0; i < providers->provider_count; ++i) {
    SCOPED_TRACE(i);
    const auto* provider = providers->providers[i];
    loom_target_low_descriptor_registry_t descriptors = {};
    if (provider->initialize_low_descriptor_registry) {
      provider->initialize_low_descriptor_registry(&descriptors);
    }
    for (iree_host_size_t j = 0; j < descriptors.descriptor_set_provider_count;
         ++j) {
      EXPECT_TRUE(
          descriptor_keys
              .insert(descriptors.descriptor_set_providers[j]()->stable_id)
              .second);
    }
    ExpectOwnedScopes(provider->legalizer_provider_list, descriptors,
                      legalizers);
    ExpectOwnedScopes(provider->low_legality_provider_list, descriptors,
                      legality);
    ExpectOwnedScopes(provider->low_packet_diagnostic_provider_list,
                      descriptors, packet_diagnostics);
    ExpectOwnedScopes(provider->low_asm_diagnostic_provider_list, descriptors,
                      asm_diagnostics);
    for (iree_host_size_t j = 0;
         j < provider->low_asm_diagnostic_provider_list.count; ++j) {
      EXPECT_NE(provider->low_asm_diagnostic_provider_list.values[j]
                    ->try_unknown_packet,
                nullptr);
    }
    ExpectOwnedScopes(provider->low_verify_provider_list, descriptors,
                      verifiers);
    if (provider->initialize_low_lower_policy_registry) {
      loom_low_lower_policy_registry_t policies = {};
      provider->initialize_low_lower_policy_registry(&policies);
      ExpectOwnedPolicies(policies, descriptors, lower_policy_keys);
    }
    if (provider->initialize_math_policy_registry) {
      loom_target_math_policy_registry_t policies = {};
      provider->initialize_math_policy_registry(&policies);
      ExpectOwnedPolicies(policies, descriptors, math_policy_keys);
    }
  }
}

TEST(ConfiguredCompilerProviderSetTest, ComposesSelectedTargetCompilers) {
  const loom_target_provider_set_t* emitter_provider_set =
      loom_configured_emitter_provider_set();
  const loom_target_provider_set_t* compiler_provider_set =
      loom_configured_compiler_provider_set();
  EXPECT_EQ(compiler_provider_set, loom_configured_compiler_provider_set());
  ASSERT_EQ(compiler_provider_set->provider_count,
            2 * emitter_provider_set->provider_count);
  for (iree_host_size_t i = 0; i < emitter_provider_set->provider_count; ++i) {
    bool found = false;
    for (iree_host_size_t j = 0; j < compiler_provider_set->provider_count;
         ++j) {
      found |= compiler_provider_set->providers[j] ==
               emitter_provider_set->providers[i];
    }
    EXPECT_TRUE(found);
  }

  loom_target_environment_t environment;
  IREE_ASSERT_OK(
      loom_target_environment_initialize(compiler_provider_set, &environment));
  EXPECT_EQ(loom_target_environment_lookup_emitter(&environment,
                                                   IREE_SV("x86-elf")) != NULL,
            static_cast<bool>(LOOM_CONFIG_COMPILER_HAVE_X86));
  EXPECT_EQ(loom_target_environment_lookup_emitter(
                &environment, IREE_SV("amdgpu-hsaco")) != NULL,
            static_cast<bool>(LOOM_CONFIG_COMPILER_HAVE_AMDGPU));
  EXPECT_EQ(loom_target_environment_lookup_emitter(&environment,
                                                   IREE_SV("spirv")) != NULL,
            static_cast<bool>(LOOM_CONFIG_COMPILER_HAVE_SPIRV));
  EXPECT_EQ(loom_target_environment_lookup_emitter(&environment,
                                                   IREE_SV("vm")) != NULL,
            static_cast<bool>(LOOM_CONFIG_COMPILER_HAVE_VM));
  EXPECT_EQ(loom_target_environment_lookup_emitter(
                &environment, IREE_SV("wasm-binary")) != NULL,
            static_cast<bool>(LOOM_CONFIG_COMPILER_HAVE_WASM));
  EXPECT_EQ(loom_target_environment_lookup_emitter(&environment,
                                                   IREE_SV("xdna")) != NULL,
            static_cast<bool>(LOOM_CONFIG_COMPILER_HAVE_XDNA));
  loom_target_environment_deinitialize(&environment);
}

#if LOOM_CONFIG_COMPILER_HAVE_X86
TEST(ConfiguredCompilerProviderSetTest, NativeCpuSelectionUsesDeviceFacts) {
  loom_target_environment_t environment;
  IREE_ASSERT_OK(loom_target_environment_initialize(
      loom_configured_compiler_provider_set(), &environment));
  const loom_target_specification_t scalar_specification = {
      .family = IREE_SVL("x86"),
      .selector = IREE_SVL("scalar"),
  };
  const loom_target_profile_t* scalar_profile = nullptr;
  IREE_ASSERT_OK(loom_target_environment_select_profile(
      &environment, &scalar_specification, &scalar_profile));

  // No optional ISA features are promised by this execution device.
  iree_cpu_data_t cpu_data = {
      .architecture = IREE_CPU_ARCHITECTURE_X86_64,
  };
  loom_target_profile_selection_t selected = {};
  IREE_ASSERT_OK(loom_target_environment_select_cpu_profile(
      &environment, &cpu_data, nullptr, nullptr, iree_allocator_system(),
      &selected));
  ASSERT_NE(selected.profile, nullptr);
  EXPECT_NE(selected.profile, scalar_profile);
  EXPECT_NE(selected.destroy, nullptr);
  const loom_target_bundle_t* selected_bundle =
      loom_target_profile_bundle(selected.profile);
  ASSERT_NE(selected_bundle, nullptr);
  EXPECT_TRUE(
      iree_string_view_equal(selected_bundle->config->contract_set.spelling,
                             IREE_SV("x86.scalar.core")));
  loom_target_profile_selection_release(&selected, iree_allocator_system());
  IREE_ASSERT_OK(loom_target_environment_select_cpu_profile(
      &environment, &cpu_data, nullptr, scalar_profile, iree_allocator_system(),
      &selected));
  ASSERT_NE(selected.profile, nullptr);
  EXPECT_NE(selected.profile, scalar_profile);
  EXPECT_NE(selected.destroy, nullptr);
  selected_bundle = loom_target_profile_bundle(selected.profile);
  ASSERT_NE(selected_bundle, nullptr);
  EXPECT_TRUE(
      iree_string_view_equal(selected_bundle->config->contract_set.spelling,
                             IREE_SV("x86.scalar.core")));
  loom_target_profile_selection_release(&selected, iree_allocator_system());

  // A compiler running on x86 must still reject an x86 profile for ARM.
  cpu_data.architecture = IREE_CPU_ARCHITECTURE_ARM_64;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        loom_target_environment_select_cpu_profile(
                            &environment, &cpu_data, nullptr, scalar_profile,
                            iree_allocator_system(), &selected));
  EXPECT_EQ(selected.profile, nullptr);
  loom_target_environment_deinitialize(&environment);
}
#endif  // LOOM_CONFIG_COMPILER_HAVE_X86

}  // namespace
}  // namespace loom
