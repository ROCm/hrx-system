// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/device_selection.h"

#include <array>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

std::array<iree_hal_device_info_t, 4> PhysicalDevices() {
  return {{
      {19, IREE_SV("GPU-123456789abcdef0"), IREE_SV("first")},
      {3, IREE_SV("GPU-12abcdef34567890"), IREE_SV("second")},
      {41, IREE_SV("GPU-abcdef0123456789"), IREE_SV("third")},
      {7, IREE_SV("id/00000000000000000000000000000007"), IREE_SV("fourth")},
  }};
}

struct SelectionCase {
  // Diagnostic label for this independent compatibility rule.
  const char* name;
  // Borrowed first-stage value, with NULL preserving the input inventory.
  const char* rocr;
  // Borrowed second-stage preferred value.
  const char* hip;
  // Borrowed second-stage fallback value.
  const char* cuda;
  // Expected opaque device IDs, independent of their enumeration ordinals.
  std::vector<iree_hal_device_id_t> expected_ids;
};

TEST(HipDeviceSelectionTest, NativeFilteringRulesAndComposition) {
  const SelectionCase cases[] = {
      {"unset", nullptr, nullptr, nullptr, {19, 3, 41, 7}},
      {"empty HIP and CUDA", nullptr, "", "", {19, 3, 41, 7}},
      {"empty ROCr", "", nullptr, nullptr, {}},
      {"empty ROCr with HIP", "", "0", nullptr, {}},
      {"ROCr disabled", "-1", nullptr, nullptr, {}},
      {"HIP disabled", nullptr, "-1", nullptr, {}},
      {"CUDA disabled", nullptr, nullptr, "-1", {}},
      {"ROCr order", "3,1,0", nullptr, nullptr, {7, 3, 19}},
      {"HIP order", nullptr, "2,0", nullptr, {41, 19}},
      {"CUDA fallback", nullptr, nullptr, "1,3", {3, 7}},
      {"empty HIP fallback", nullptr, "", "1,3", {3, 7}},
      {"HIP precedence", nullptr, "2", "1", {41}},
      {"composition", "3,1", "1,0", nullptr, {3, 7}},
      {"CUDA composition", "3,1", "", "1,0", {3, 7}},
      {"ROCr duplicate terminates", "2,2,1", nullptr, nullptr, {41}},
      {"HIP duplicate skipped", nullptr, "2,2,1,1,3", nullptr, {41, 3, 7}},
      {"ROCr invalid terminates", "1,invalid,2", nullptr, nullptr, {3}},
      {"HIP invalid terminates", nullptr, "1,invalid,2", nullptr, {3}},
      {"ROCr range terminates", "0,4,2", nullptr, nullptr, {19}},
      {"HIP range terminates", nullptr, "0,4,2", nullptr, {19}},
      {"ROCr empty token", "1,,2", nullptr, nullptr, {3}},
      {"HIP empty token", nullptr, "1,,2", nullptr, {3}},
      {"trailing separator", nullptr, "1,", nullptr, {3}},
      {"ROCr bases and whitespace",
       " 0x2 , +01 , -0 ",
       nullptr,
       nullptr,
       {41, 3, 19}},
      {"HIP whitespace rejected", nullptr, " 1", nullptr, {}},
      {"HIP sign rejected", nullptr, "+1", nullptr, {}},
      {"HIP leading zero rejected", nullptr, "01", nullptr, {}},
      {"HIP hexadecimal rejected", nullptr, "0x1", nullptr, {}},
      {"ROCr invalid octal", "08", nullptr, nullptr, {}},
      {"ROCr suffix rejected", "1suffix", nullptr, nullptr, {}},
      {"HIP suffix rejected", nullptr, "1suffix", nullptr, {}},
      {"ROCr overflow rejected", "18446744073709551616", nullptr, nullptr, {}},
      {"HIP overflow rejected", nullptr, "18446744073709551616", nullptr, {}},
      {"ROCr full UUID", "GPU-abcdef0123456789", nullptr, nullptr, {41}},
      {"HIP full UUID", nullptr, "GPU-abcdef0123456789", nullptr, {41}},
      {"ROCr case folding", " gpu-ABCDEF ", nullptr, nullptr, {41}},
      {"HIP case sensitive", nullptr, "gpu-ABCDEF", nullptr, {}},
      {"ROCr unique prefix", "GPU-123", nullptr, nullptr, {19}},
      {"ROCr ambiguous prefix", "GPU-12", nullptr, nullptr, {}},
      {"HIP first prefix match", nullptr, "GPU-12", nullptr, {19}},
      {"ROCr ordinal UUID duplicate", "2,GPU-abcdef", nullptr, nullptr, {41}},
      {"HIP ordinal UUID duplicate",
       nullptr,
       "2,GPU-abcdef,1",
       nullptr,
       {41, 3}},
      {"ROCr missing UUID", "GPU-999999", nullptr, nullptr, {}},
      {"HIP missing UUID", nullptr, "GPU-999999", nullptr, {}},
      {"ROCr unavailable UUID", "GPU-XX", nullptr, nullptr, {}},
      {"HIP unavailable UUID", nullptr, "GPU-XX", nullptr, {}},
      {"ROCr empty UUID", "GPU-", nullptr, nullptr, {}},
      {"HIP empty prefix matches first", nullptr, "GPU-", nullptr, {19}},
      {"UUIDs compose over selected set",
       "2,0",
       "GPU-123,0",
       nullptr,
       {19, 41}},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    auto devices = PhysicalDevices();
    iree_host_size_t count = devices.size();
    iree_hip_device_selection_t selection = {test_case.rocr, test_case.hip,
                                             test_case.cuda};
    IREE_ASSERT_OK(iree_hip_device_selection_apply(
        &selection, IREE_SV("amdgpu"), iree_allocator_system(), &count,
        devices.data()));
    ASSERT_EQ(count, test_case.expected_ids.size());
    for (iree_host_size_t i = 0; i < count; ++i) {
      EXPECT_EQ(devices[i].device_id, test_case.expected_ids[i]);
      for (const auto& original : PhysicalDevices()) {
        if (original.device_id != devices[i].device_id) {
          continue;
        }
        EXPECT_EQ(devices[i].path.data, original.path.data);
        EXPECT_EQ(devices[i].path.size, original.path.size);
        EXPECT_EQ(devices[i].name.data, original.name.data);
        EXPECT_EQ(devices[i].name.size, original.name.size);
      }
    }
  }
}

TEST(HipDeviceSelectionTest, UnavailableUuidDoesNotMatch) {
  for (const auto path : {IREE_SV("GPU-0000000000000000"), IREE_SV("GPU-XX")}) {
    for (bool rocr : {false, true}) {
      iree_hal_device_info_t device = {1, path, IREE_SV("unidentified")};
      iree_host_size_t count = 1;
      iree_hip_device_selection_t selection = {
          rocr ? "GPU-0" : nullptr, rocr ? nullptr : "GPU-0", nullptr};
      IREE_ASSERT_OK(iree_hip_device_selection_apply(
          &selection, IREE_SV("amdgpu"), iree_allocator_system(), &count,
          &device));
      EXPECT_EQ(count, 0);
    }
  }
}

TEST(HipDeviceSelectionTest, NoFilterNeedsNoScratch) {
  auto devices = PhysicalDevices();
  iree_host_size_t count = devices.size();
  iree_hip_device_selection_t selection = {};
  IREE_ASSERT_OK(iree_hip_device_selection_apply(&selection, IREE_SV("amdgpu"),
                                                 iree_allocator_null(), &count,
                                                 devices.data()));
  EXPECT_EQ(count, devices.size());
}

TEST(HipDeviceSelectionTest, EmptyInventoryNeedsNoScratch) {
  iree_host_size_t count = 0;
  iree_hip_device_selection_t selection = {"1", "0", nullptr};
  IREE_ASSERT_OK(iree_hip_device_selection_apply(
      &selection, IREE_SV("amdgpu"), iree_allocator_null(), &count, nullptr));
  EXPECT_EQ(count, 0);
}

TEST(HipDeviceSelectionTest, ScratchFailurePreservesSelectionInput) {
  auto devices = PhysicalDevices();
  iree_host_size_t count = devices.size();
  iree_hip_device_selection_t selection = {nullptr, "2,0", nullptr};
  IREE_EXPECT_NOT_OK(iree_hip_device_selection_apply(
      &selection, IREE_SV("amdgpu"), iree_allocator_null(), &count,
      devices.data()));
  EXPECT_EQ(count, devices.size());
  for (iree_host_size_t i = 0; i < count; ++i) {
    EXPECT_EQ(devices[i].device_id, PhysicalDevices()[i].device_id);
  }
}

TEST(HipDeviceSelectionTest, ExplicitNoDeviceRequestsNeedNoDiscovery) {
  iree_hip_device_selection_t selection = {};
  EXPECT_FALSE(iree_hip_device_selection_selects_none(&selection));
  selection.hip_visible_devices = "-1";
  EXPECT_TRUE(iree_hip_device_selection_selects_none(&selection));
  selection.hip_visible_devices = "";
  selection.cuda_visible_devices = "-1";
  EXPECT_TRUE(iree_hip_device_selection_selects_none(&selection));
  selection.hip_visible_devices = "0";
  EXPECT_FALSE(iree_hip_device_selection_selects_none(&selection));
  selection.rocr_visible_devices = "";
  EXPECT_TRUE(iree_hip_device_selection_selects_none(&selection));
  selection.rocr_visible_devices = " -1 ";
  EXPECT_TRUE(iree_hip_device_selection_selects_none(&selection));
}

}  // namespace
