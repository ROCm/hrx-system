// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEMORY_TEST_FIXTURE_H_
#define AMDF_SRC_MEMORY_TEST_FIXTURE_H_

#include <array>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/device.h"
#include "libamdf/src/instance.h"
#include "libamdf/src/memory.h"
#include "libamdf/src/memory_scope.h"

namespace amdf::testing {

struct ReleaseState {
  // Number of external ownership obligations consumed.
  uint32_t count;
  // Representation passed to the last release.
  amdf_external_memory_type_t type;
  // Exact payload passed to the last release.
  amdf_external_memory_payload_t payload;
};

enum class ImportFailureStage {
  kNone,
  kBeforeAttachment,
  kAfterAttachment,
};

struct FakeDevice {
  // Real common device dependency used by the API under test.
  amdf_device_t base;
  // Complete capabilities returned by the device dependency.
  amdf_memory_native_profile_t profile;
  // Caller-owned access request consumed synchronously by construction.
  amdf_memory_device_access_t request;
  // Consumer addresses established by the fake native construction boundary.
  std::array<uint64_t, AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE + 1> addresses;
  // Injected query result before profile publication.
  amdf_status_t profile_status;
  // Injected allocation result after native metadata is acquired.
  amdf_status_t create_status;
  // Injected import result at import_failure_stage.
  amdf_status_t import_status;
  // Import prefix completed before reporting the injected error.
  ImportFailureStage import_failure_stage;
  // Number of calls into native allocation preparation.
  uint32_t create_call_count;
  // Number of consumers delivered together to the last preparation.
  uint32_t prepared_access_count;
  // Caller pages received by the last native registration preparation.
  void* registered_host_pointer;
  // Number of calls into native import preparation.
  uint32_t import_call_count;
  // Native destruction status copied into each prepared allocation.
  amdf_status_t destroy_status;
  // Number of native destruction attempts.
  uint32_t destroy_call_count;
  // Number of metadata-only abandonments after terminal native failure.
  uint32_t abandon_call_count;
  // Observer of temporary transports exported during aggregate construction.
  ReleaseState export_release;
  // Optional caller-owned log of native release attempts in device order.
  std::vector<uint64_t>* release_order;
  // Native backing identity reported by allocation.
  amdf_physical_memory_id_t backing_id;
};

struct FakeMemory {
  // Device dependency outliving this native allocation.
  FakeDevice* device;
  // Injected export result before publishing a payload.
  amdf_status_t export_status;
  // Injected host-mapping result.
  amdf_status_t map_status;
  // Injected site-query result before publication.
  amdf_status_t site_status;
  // Native release result injected before relinquishing backing ownership.
  amdf_status_t destroy_status;
  // Number of export calls on this native object.
  uint32_t export_call_count;
  // Number of host-mapping calls on this native object.
  uint32_t map_call_count;
  // Number of site queries on this native object.
  uint32_t site_description_count;
  // Queue family received by the last site query.
  uint32_t last_queue_family_ordinal;
  // Complete native facts returned by site queries.
  amdf_memory_site_description_t site_description;
  // Borrowed observer of ownership exported by this native object.
  ReleaseState* export_release_state;
};

// Records the exact caller-owned external release obligation.
void AMDF_CALL RecordRelease(void* user_data, amdf_external_memory_type_t type,
                             amdf_external_memory_payload_t payload);

// Initializes a native dependency with complete ordinary memory capabilities.
void InitializeFakeDevice(uint64_t identity, amdf_instance_t* instance,
                          FakeDevice* out_device);

// Constructs a single-consumer allocation request for the native defaults.
amdf_memory_create_info_t MakeMemoryCreateInfo(FakeDevice& device);
// Constructs a single-consumer import request for the native defaults.
amdf_memory_import_info_t MakeMemoryImportInfo(FakeDevice& device);
// Constructs an export request matching the native dependency's transport.
amdf_memory_export_info_t MakeMemoryExportInfo();

// Names one queue consumer of the memory's first access.
amdf_memory_site_t MakeMemorySite(amdf_memory_t* memory,
                                  uint32_t queue_family_ordinal);

// Only native device operations are substituted. Construction, selection and
// teardown use the production common owner and a real instance representation.
class MemoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    instance_.host_allocator = amdf_allocator_system();
    instance_.system_memory_scope.kind = AMDF_MEMORY_SCOPE_KIND_SYSTEM;
    instance_.system_memory_scope.owner.instance = &instance_;
  }

  // Common lifetime owner; these tests do not require a platform connection.
  amdf_instance_t instance_ = {};
};

}  // namespace amdf::testing

#endif  // AMDF_SRC_MEMORY_TEST_FIXTURE_H_
