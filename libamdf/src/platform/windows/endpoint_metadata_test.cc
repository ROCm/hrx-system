// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/platform/windows/endpoint_metadata.h"

#include <cstring>
#include <cwchar>
#include <string>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"

namespace {

constexpr NTSTATUS kQueryFailure = static_cast<NTSTATUS>(0xc0000001u);

struct NativeMetadata {
  // Exact physical index requested through the KMT connection.
  uint32_t physical_index = 5;
  // Returned native hardware key, including the device parameters suffix.
  std::wstring key =
      L"\\REGISTRY\\MACHINE\\SYSTEM\\ControlSet002\\Enum\\PCI\\VEN_1002&DEV_"
      L"1234"
      L"\\physical-device-5\\Device Parameters";
  // KMT query stage at which the native dependency reports a failure.
  uint32_t failed_query = 0;
  // Number of KMT queries performed by the current call.
  uint32_t query_count = 0;
  // Native device lookup result.
  CONFIGRET locate_result = CR_SUCCESS;
  // Native device-property query result.
  CONFIGRET property_result = CR_SUCCESS;
  // Native property type returned with the node number.
  DEVPROPTYPE property_type = DEVPROP_TYPE_UINT32;
  // Native property payload size.
  ULONG property_length = sizeof(uint32_t);
  // OS NUMA node reported for the exact device.
  uint32_t node = 37;
  // Number of processor groups containing active CPUs at that node.
  USHORT active_group_count = 2;
  // Native CPU-topology failure, or zero for the ordinary count protocol.
  DWORD topology_error = 0;
  // Number of CPU eligibility queries made by the current call.
  uint32_t topology_count = 0;
  // Physical PCI location property from the selected device instance.
  std::wstring location = L"PCI segment 7 bus 171, device 29, function 6";
  // Result of reading the independent PCI location property.
  CONFIGRET location_result = CR_SUCCESS;
  // Native type of the PCI location property.
  DEVPROPTYPE location_type = DEVPROP_TYPE_STRING;
  // Optional malformed native length; zero uses the complete string extent.
  ULONG location_length = 0;
  // Whether native output includes the required string terminator.
  bool location_terminated = true;
};

NativeMetadata* current = nullptr;

NTSTATUS APIENTRY QueryAdapter(const D3DKMT_QUERYADAPTERINFO* arguments) {
  EXPECT_EQ(arguments->hAdapter, 123u);
  EXPECT_EQ(arguments->Type, KMTQAITYPE_PHYSICALADAPTERPNPKEY);
  EXPECT_EQ(arguments->PrivateDriverDataSize,
            sizeof(D3DKMT_QUERY_PHYSICAL_ADAPTER_PNP_KEY));
  auto* query = static_cast<D3DKMT_QUERY_PHYSICAL_ADAPTER_PNP_KEY*>(
      arguments->pPrivateDriverData);
  EXPECT_EQ(query->PhysicalAdapterIndex, current->physical_index);
  EXPECT_EQ(query->PnPKeyType, D3DKMT_PNP_KEY_HARDWARE);
  if (++current->query_count == current->failed_query) {
    return kQueryFailure;
  }
  const auto count = static_cast<UINT>(current->key.size() + 1);
  if (query->pDest != nullptr) {
    EXPECT_GE(*query->pCchDest, count);
    std::wmemcpy(query->pDest, current->key.c_str(), count);
  }
  *query->pCchDest = count;
  return 0;
}

// Parameter order is fixed by the CM_Locate_DevNodeW callback ABI.
CONFIGRET WINAPI
LocateDevice(PDEVINST out_device,  // NOLINT(iree-output-parameter-order)
             DEVINSTID_W instance, ULONG flags) {
  EXPECT_STREQ(instance, L"PCI\\VEN_1002&DEV_1234\\physical-device-5");
  EXPECT_EQ(flags, CM_LOCATE_DEVNODE_NORMAL);
  if (current->locate_result == CR_SUCCESS) {
    *out_device = 42;
  }
  return current->locate_result;
}

CONFIGRET WINAPI GetProperty(DEVINST device, const DEVPROPKEY* key,
                             DEVPROPTYPE* type, PBYTE buffer, PULONG length,
                             ULONG flags) {
  EXPECT_EQ(device, 42u);
  if (key->fmtid.Data1 == 0xa45c254eu && key->pid == 15u) {
    EXPECT_EQ(flags, 0u);
    if (current->location_result != CR_SUCCESS) {
      return current->location_result;
    }
    const ULONG required =
        static_cast<ULONG>((current->location.size() + 1) * sizeof(WCHAR));
    if (*length < required) {
      *length = required;
      return CR_BUFFER_SMALL;
    }
    std::memcpy(buffer, current->location.c_str(), required);
    if (!current->location_terminated) {
      reinterpret_cast<WCHAR*>(buffer)[current->location.size()] = L'X';
    }
    *type = current->location_type;
    *length = current->location_length ? current->location_length : required;
    return CR_SUCCESS;
  }
  EXPECT_EQ(key->fmtid.Data1, 0x540b947eu);
  EXPECT_EQ(key->pid, 3u);
  EXPECT_EQ(flags, 0u);
  EXPECT_EQ(*length, sizeof(uint32_t));
  if (current->property_result == CR_SUCCESS) {
    *reinterpret_cast<uint32_t*>(buffer) = current->node;
    *type = current->property_type;
    *length = current->property_length;
  }
  return current->property_result;
}

DWORD WINAPI MapError(CONFIGRET result, DWORD fallback) {
  EXPECT_NE(result, CR_SUCCESS);
  EXPECT_EQ(fallback, ERROR_GEN_FAILURE);
  return ERROR_ACCESS_DENIED;
}

BOOL WINAPI GetNodeMasks(USHORT node, PGROUP_AFFINITY masks, USHORT capacity,
                         PUSHORT count) {
  ++current->topology_count;
  EXPECT_EQ(node, current->node);
  EXPECT_EQ(masks, nullptr);
  EXPECT_EQ(capacity, 0u);
  if (current->topology_error != 0) {
    SetLastError(current->topology_error);
    return FALSE;
  }
  *count = current->active_group_count;
  if (*count != 0) {
    SetLastError(ERROR_INSUFFICIENT_BUFFER);
    return FALSE;
  }
  return TRUE;
}

class WindowsEndpointMetadataTest : public ::testing::Test {
 protected:
  void SetUp() override { current = &native_; }
  void TearDown() override { current = nullptr; }

  amdf_status_t Query(amdf_windows_endpoint_metadata_t* out_metadata) {
    native_.query_count = 0;
    native_.topology_count = 0;
    return amdf_windows_query_endpoint_metadata(
        &api_, &kmt_, 123, native_.physical_index, amdf_allocator_system(),
        out_metadata);
  }

  // Fixture supplying only native OS dependencies of the production query.
  NativeMetadata native_;
  // Exact KMT query dependency used by the production implementation.
  amdf_kmt_api_t kmt_ = {.query_adapter_info = QueryAdapter};
  // Config Manager and CPU topology dependencies without a loaded module.
  amdf_windows_endpoint_metadata_api_t api_ = {
      .locate_device = LocateDevice,
      .get_property = GetProperty,
      .map_error = MapError,
      .get_node_masks = GetNodeMasks,
  };
};

TEST_F(WindowsEndpointMetadataTest,
       PreservesCpuNodeIdentityAcrossProcessorGroups) {
  for (const uint32_t node : {0u, 37u}) {
    for (const USHORT groups : {USHORT{1}, USHORT{2}}) {
      SCOPED_TRACE(node);
      SCOPED_TRACE(groups);
      native_.node = node;
      native_.active_group_count = groups;
      amdf_windows_endpoint_metadata_t metadata = {.host_numa_node =
                                                       UINT32_MAX};
      ASSERT_EQ(Query(&metadata), AMDF_STATUS_OK);
      EXPECT_EQ(metadata.host_numa_node, node);
      EXPECT_EQ(native_.query_count, 2u);
      EXPECT_EQ(native_.topology_count, 1u);
    }
  }
}

TEST_F(WindowsEndpointMetadataTest, ExcludesAbsentAndMemoryOnlyNodes) {
  amdf_windows_endpoint_metadata_t metadata = {.host_numa_node = 123};
  native_.property_result = CR_NO_SUCH_VALUE;
  ASSERT_EQ(Query(&metadata), AMDF_STATUS_OK);
  EXPECT_EQ(metadata.host_numa_node, UINT32_MAX);
  EXPECT_EQ(native_.topology_count, 0u);
  native_.property_result = CR_SUCCESS;
  native_.node = UINT32_MAX;
  ASSERT_EQ(Query(&metadata), AMDF_STATUS_OK);
  EXPECT_EQ(metadata.host_numa_node, UINT32_MAX);
  EXPECT_EQ(native_.topology_count, 0u);
  native_.node = 37;
  native_.active_group_count = 0;
  ASSERT_EQ(Query(&metadata), AMDF_STATUS_OK);
  EXPECT_EQ(metadata.host_numa_node, UINT32_MAX);
  EXPECT_EQ(native_.topology_count, 1u);
}

TEST_F(WindowsEndpointMetadataTest,
       UnavailableNativeTopologyDoesNotInventNodeZero) {
  api_.get_node_masks = nullptr;
  amdf_windows_endpoint_metadata_t metadata = {.host_numa_node = 123};
  ASSERT_EQ(Query(&metadata), AMDF_STATUS_OK);
  EXPECT_EQ(metadata.host_numa_node, UINT32_MAX);
  EXPECT_EQ(metadata.pci_address.available, 1u);
  EXPECT_EQ(native_.query_count, 2u);
}

TEST_F(WindowsEndpointMetadataTest, PreservesPhysicalPciAddressAndSegment) {
  struct Case {
    // PCI location returned by the exact physical adapter's PnP device.
    const WCHAR* location;
    // Expected domain, including zero for the documented single-segment form.
    uint32_t domain;
    // Expected PCI bus.
    uint32_t bus;
    // Expected PCI device.
    uint32_t device;
    // Expected PCI function.
    uint32_t function;
  };
  for (const auto& value : {
           Case{L"PCI bus 0, device 0, function 0", 0, 0, 0, 0},
           Case{L"PCI bus 171, device 29, function 6", 0, 171, 29, 6},
           Case{L"PCI segment 7 bus 171, device 29, function 6", 7, 171, 29, 6},
           Case{L"PCI segment 65535 bus 255, device 31, function 7", 65535, 255,
                31, 7},
       }) {
    native_.location = value.location;
    amdf_windows_endpoint_metadata_t metadata = {};
    ASSERT_EQ(Query(&metadata), AMDF_STATUS_OK);
    EXPECT_EQ(metadata.pci_address.available, 1u);
    EXPECT_EQ(metadata.pci_address.domain, value.domain);
    EXPECT_EQ(metadata.pci_address.bus, value.bus);
    EXPECT_EQ(metadata.pci_address.device, value.device);
    EXPECT_EQ(metadata.pci_address.function, value.function);
    EXPECT_EQ(metadata.host_numa_node, native_.node);
    EXPECT_EQ(native_.query_count, 2u);
  }
}

TEST_F(WindowsEndpointMetadataTest,
       UnavailablePciLocationPreservesCpuLocality) {
  native_.location_result = CR_NO_SUCH_VALUE;
  amdf_windows_endpoint_metadata_t metadata = {};
  ASSERT_EQ(Query(&metadata), AMDF_STATUS_OK);
  EXPECT_EQ(metadata.pci_address.available, 0u);
  EXPECT_EQ(metadata.pci_address.domain | metadata.pci_address.bus |
                metadata.pci_address.device | metadata.pci_address.function,
            0u);
  EXPECT_EQ(metadata.host_numa_node, native_.node);
}

TEST_F(WindowsEndpointMetadataTest, UnknownPciLocationsDoNotInventAddresses) {
  for (const auto* location : {
           L"",
           L"ACPI device",
           L"PCI bus 256, device 0, function 0",
           L"PCI bus 0, device 32, function 0",
           L"PCI bus 0, device 0, function 8",
           L"PCI segment 65536 bus 0, device 0, function 0",
           L"PCI segment 4294967296 bus 0, device 0, function 0",
           L"PCI bus -1, device 0, function 0",
           L"PCI bus 0, device 0, function 0 trailing",
       }) {
    native_.location = location;
    amdf_windows_endpoint_metadata_t metadata = {};
    ASSERT_EQ(Query(&metadata), AMDF_STATUS_OK);
    EXPECT_EQ(metadata.pci_address.available, 0u);
    EXPECT_EQ(metadata.pci_address.domain | metadata.pci_address.bus |
                  metadata.pci_address.device | metadata.pci_address.function,
              0u);
    EXPECT_EQ(metadata.host_numa_node, native_.node);
  }
}

TEST_F(WindowsEndpointMetadataTest, PciQueryFailurePreservesAllCallerOutput) {
  amdf_windows_endpoint_metadata_t metadata;
  std::memset(&metadata, 0xa5, sizeof(metadata));
  const auto original = metadata;
  native_.location_result = CR_ACCESS_DENIED;
  EXPECT_EQ(Query(&metadata),
            amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, ERROR_ACCESS_DENIED));
  EXPECT_EQ(std::memcmp(&metadata, &original, sizeof(metadata)), 0);
  native_.location_result = CR_SUCCESS;
  native_.location_type = DEVPROP_TYPE_UINT32;
  EXPECT_EQ(Query(&metadata),
            amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, ERROR_INVALID_DATA));
  EXPECT_EQ(std::memcmp(&metadata, &original, sizeof(metadata)), 0);
  native_.location_type = DEVPROP_TYPE_STRING;
  native_.location_length = 1;
  EXPECT_EQ(Query(&metadata),
            amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, ERROR_INVALID_DATA));
  EXPECT_EQ(std::memcmp(&metadata, &original, sizeof(metadata)), 0);
  native_.location_length = 0;
  native_.location_terminated = false;
  EXPECT_EQ(Query(&metadata),
            amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, ERROR_INVALID_DATA));
  EXPECT_EQ(std::memcmp(&metadata, &original, sizeof(metadata)), 0);
}

TEST_F(WindowsEndpointMetadataTest, NativeQueryFailuresPreserveCallerOutput) {
  amdf_windows_endpoint_metadata_t metadata = {.host_numa_node = 123};
  for (const uint32_t stage : {1u, 2u}) {
    native_.failed_query = stage;
    EXPECT_EQ(Query(&metadata), amdf_kmt_make_status(kQueryFailure));
    EXPECT_EQ(metadata.host_numa_node, 123u);
  }
  native_.failed_query = 0;
  native_.locate_result = CR_NO_SUCH_DEVNODE;
  EXPECT_EQ(Query(&metadata),
            amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, ERROR_ACCESS_DENIED));
  EXPECT_EQ(metadata.host_numa_node, 123u);
  native_.locate_result = CR_SUCCESS;
  native_.property_result = CR_ACCESS_DENIED;
  EXPECT_EQ(Query(&metadata),
            amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, ERROR_ACCESS_DENIED));
  EXPECT_EQ(metadata.host_numa_node, 123u);
  native_.property_result = CR_SUCCESS;
  native_.topology_error = ERROR_INVALID_PARAMETER;
  EXPECT_EQ(Query(&metadata), amdf_make_status(AMDF_STATUS_DOMAIN_WIN32,
                                               ERROR_INVALID_PARAMETER));
  EXPECT_EQ(metadata.host_numa_node, 123u);
}

TEST_F(WindowsEndpointMetadataTest,
       MalformedNativePropertiesPreserveCallerOutput) {
  amdf_windows_endpoint_metadata_t metadata = {.host_numa_node = 123};
  native_.property_type = DEVPROP_TYPE_STRING;
  EXPECT_EQ(Query(&metadata),
            amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, ERROR_INVALID_DATA));
  native_.property_type = DEVPROP_TYPE_UINT32;
  native_.property_length = 2;
  EXPECT_EQ(Query(&metadata),
            amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, ERROR_INVALID_DATA));
  native_.property_length = sizeof(uint32_t);
  native_.node = UINT32_C(0x10000);
  EXPECT_EQ(Query(&metadata),
            amdf_make_status(AMDF_STATUS_DOMAIN_WIN32, ERROR_INVALID_DATA));
  EXPECT_EQ(metadata.host_numa_node, 123u);
}

TEST_F(WindowsEndpointMetadataTest, RejectsUnrecognizedHardwareKeyShape) {
  for (const auto* key :
       {L"", L"\\REGISTRY\\MACHINE\\OTHER\\Enum\\device",
        L"\\REGISTRY\\MACHINE\\SYSTEM\\ControlSet001\\device",
        L"\\REGISTRY\\MACHINE\\SYSTEM\\ControlSet001\\Enum\\device"}) {
    native_.key = key;
    amdf_windows_endpoint_metadata_t metadata = {.host_numa_node = 123};
    EXPECT_EQ(amdf_status_code(Query(&metadata)), AMDF_STATUS_CODE_UNSUPPORTED);
    EXPECT_EQ(metadata.host_numa_node, 123u);
    EXPECT_EQ(native_.topology_count, 0u);
  }
}

}  // namespace
