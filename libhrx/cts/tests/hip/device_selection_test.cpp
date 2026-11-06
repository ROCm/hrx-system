// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "binding/hip/api.h"
#include "binding/hip/hip_dso_test_util.h"
#include "iree/testing/gtest.h"
#include "libhrx/cts/core/amdgpu_executable_test_data.hpp"

namespace {

void CheckSelectionIsFrozen(decltype(&hipGetDeviceCount) device_count,
                            int expected_count) {
  std::optional<std::string> original;
  if (const char* value = std::getenv("HIP_VISIBLE_DEVICES")) {
    original = value;
  }
  EXPECT_EQ(setenv("HIP_VISIBLE_DEVICES", expected_count ? "-1" : "0", 1), 0);
  int retained_count = -1;
  EXPECT_EQ(expected_count ? hipSuccess : hipErrorNoDevice,
            device_count(&retained_count));
  EXPECT_EQ(retained_count, expected_count);
  EXPECT_EQ(original ? setenv("HIP_VISIBLE_DEVICES", original->c_str(), 1)
                     : unsetenv("HIP_VISIBLE_DEVICES"),
            0);
}

class HipDeviceSelectionTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    ASSERT_TRUE(library_.Open()) << library_.error();
    const auto initialize = library_.Resolve<decltype(&hipInit)>("hipInit");
    ASSERT_NE(initialize, nullptr);
    const char* expected_count = std::getenv("HRX_TEST_EXPECT_DEVICE_COUNT");
    ASSERT_NE(expected_count, nullptr);
    ASSERT_EQ(
        std::string(expected_count) == "0" ? hipErrorNoDevice : hipSuccess,
        initialize(0));
  }

  static void TearDownTestSuite() {
    if (!library_.is_open()) {
      return;
    }
    const auto deinitialize =
        library_.Resolve<hipError_t (*)()>("hipHALDeinit");
    ASSERT_NE(deinitialize, nullptr);
    EXPECT_EQ(hipSuccess, deinitialize());
    EXPECT_TRUE(library_.Close()) << library_.error();
  }

  // One initialized HIP DSO spans selection, execution, and terminal teardown.
  static inline hrx::hip::testing::HipDso library_;
};

TEST_F(HipDeviceSelectionTest, SelectedDevicesHaveDenseExecutableOrdinals) {
  const auto device_count =
      library_.Resolve<decltype(&hipGetDeviceCount)>("hipGetDeviceCount");
  const auto set_device =
      library_.Resolve<decltype(&hipSetDevice)>("hipSetDevice");
  const auto get_device =
      library_.Resolve<decltype(&hipGetDevice)>("hipGetDevice");
  const auto device_properties =
      library_.Resolve<decltype(&hipGetDevicePropertiesR0600)>(
          "hipGetDevicePropertiesR0600");
  const auto memory_info =
      library_.Resolve<decltype(&hipMemGetInfo)>("hipMemGetInfo");
  const auto device_name =
      library_.Resolve<decltype(&hipDeviceGetName)>("hipDeviceGetName");
  const auto pci_address =
      library_.Resolve<decltype(&hipDeviceGetPCIBusId)>("hipDeviceGetPCIBusId");
  const auto device_uuid =
      library_.Resolve<decltype(&hipDeviceGetUuid)>("hipDeviceGetUuid");
  const auto module_load =
      library_.Resolve<decltype(&hipModuleLoadData)>("hipModuleLoadData");
  const auto module_unload =
      library_.Resolve<decltype(&hipModuleUnload)>("hipModuleUnload");
  const auto module_function =
      library_.Resolve<decltype(&hipModuleGetFunction)>("hipModuleGetFunction");
  const auto allocate = library_.Resolve<decltype(&hipMalloc)>("hipMalloc");
  const auto release = library_.Resolve<decltype(&hipFree)>("hipFree");
  const auto copy = library_.Resolve<decltype(&hipMemcpy)>("hipMemcpy");
  const auto launch = library_.Resolve<decltype(&hipModuleLaunchKernel)>(
      "hipModuleLaunchKernel");
  ASSERT_TRUE(library_.error().empty()) << library_.error();

  const char* expected_count_text = std::getenv("HRX_TEST_EXPECT_DEVICE_COUNT");
  ASSERT_NE(expected_count_text, nullptr);
  const int expected_count = std::stoi(expected_count_text);
  ASSERT_GE(expected_count, 0);
  RecordProperty("expected_device_count", expected_count);
  int count = 0;
  if (expected_count == 0) {
    ASSERT_EQ(hipErrorNoDevice, device_count(&count));
    EXPECT_EQ(count, 0);
    CheckSelectionIsFrozen(device_count, count);
    return;
  }
  ASSERT_EQ(hipSuccess, device_count(&count));
  ASSERT_EQ(count, expected_count);

  // Optional exact physical identities qualify reordered multi-GPU selection
  // with the same executable on any machine's configured device inventory.
  std::vector<std::string> expected_pci_addresses;
  if (const char* expected = std::getenv("HRX_TEST_EXPECT_DEVICE_PCI")) {
    std::istringstream input(expected);
    for (std::string address; std::getline(input, address, ',');) {
      expected_pci_addresses.push_back(address);
    }
    ASSERT_EQ(expected_pci_addresses.size(), static_cast<size_t>(count));
  }
  std::vector<std::string> expected_uuids;
  if (const char* expected = std::getenv("HRX_TEST_EXPECT_DEVICE_UUID")) {
    std::istringstream input(expected);
    for (std::string uuid; std::getline(input, uuid, ',');) {
      expected_uuids.push_back(uuid);
    }
    ASSERT_EQ(expected_uuids.size(), static_cast<size_t>(count));
  }

  for (int ordinal = 0; ordinal < count; ++ordinal) {
    SCOPED_TRACE(ordinal);
    ASSERT_EQ(hipSuccess, set_device(ordinal));
    int current = -1;
    ASSERT_EQ(hipSuccess, get_device(&current));
    EXPECT_EQ(current, ordinal);
    hipDeviceProp_t properties = {};
    ASSERT_EQ(hipSuccess, device_properties(&properties, ordinal));
    size_t free_memory = 0;
    size_t total_memory = 0;
    ASSERT_EQ(hipSuccess, memory_info(&free_memory, &total_memory));
    EXPECT_GT(total_memory, 0u);
    EXPECT_LE(free_memory, total_memory);
    EXPECT_EQ(total_memory, properties.totalGlobalMem);
    char name[sizeof(properties.name)] = {};
    ASSERT_EQ(hipSuccess, device_name(name, sizeof(name), ordinal));
    EXPECT_STREQ(name, properties.name);
    char actual_pci[32] = {};
    ASSERT_EQ(hipSuccess, pci_address(actual_pci, sizeof(actual_pci), ordinal));
    RecordProperty("device_" + std::to_string(ordinal),
                   std::string(actual_pci) + " " + properties.gcnArchName);
    if (!expected_pci_addresses.empty()) {
      EXPECT_EQ(expected_pci_addresses[ordinal], actual_pci);
    }
    hipUUID uuid = {};
    const hipError_t uuid_result = device_uuid(&uuid, ordinal);
    if (uuid_result == hipSuccess) {
      const std::string text(reinterpret_cast<const char*>(uuid.bytes),
                             sizeof(uuid.bytes));
      EXPECT_EQ(text.find_first_not_of("0123456789abcdef"), std::string::npos);
      EXPECT_EQ(std::memcmp(uuid.bytes, properties.uuid.bytes, sizeof(uuid)),
                0);
      RecordProperty("uuid_" + std::to_string(ordinal), text);
      if (!expected_uuids.empty()) {
        EXPECT_EQ(expected_uuids[ordinal], text);
      }
    } else {
      EXPECT_EQ(uuid_result, hipErrorNotSupported);
      EXPECT_TRUE(expected_uuids.empty()) << "required native UUID missing";
    }
    const auto image =
        hrx_cts::FindAmdgpuExecutableTestImage(properties.gcnArchName);
    ASSERT_NE(image.file, nullptr) << properties.gcnArchName;
    hipModule_t module = nullptr;
    ASSERT_EQ(hipSuccess, module_load(&module, image.file->data));
    hipFunction_t function = nullptr;
    ASSERT_EQ(hipSuccess,
              module_function(&function, module, "hrx_store_output"));
    void* output = nullptr;
    ASSERT_EQ(hipSuccess, allocate(&output, sizeof(uint32_t)));
    uint32_t expected = 0xC0FFEEu + static_cast<uint32_t>(ordinal);
    void* arguments[] = {&output, &expected};
    ASSERT_EQ(hipSuccess, launch(function, 1, 1, 1, 1, 1, 1, 0, nullptr,
                                 arguments, nullptr));
    uint32_t actual = 0;
    ASSERT_EQ(hipSuccess,
              copy(&actual, output, sizeof(actual), hipMemcpyDeviceToHost));
    EXPECT_EQ(actual, expected);
    EXPECT_EQ(hipSuccess, release(output));
    EXPECT_EQ(hipSuccess, module_unload(module));
  }
  EXPECT_EQ(hipErrorInvalidDevice, set_device(count));
  char invalid_name[] = "unchanged";
  EXPECT_EQ(hipErrorInvalidDevice,
            device_name(invalid_name, sizeof(invalid_name), count));
  EXPECT_STREQ(invalid_name, "unchanged");

  // Selection is captured once. A later environment edit cannot mutate the
  // sealed group or turn an otherwise valid API call into hipErrorNoDevice.
  CheckSelectionIsFrozen(device_count, count);
}

}  // namespace
