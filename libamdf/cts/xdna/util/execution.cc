// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/xdna/util/execution.h"

#include <algorithm>
#include <utility>

::testing::AssertionResult FindXdnaKernelQueueFamily(
    const amdf_api_t* api, amdf_endpoint_t* endpoint,
    uint32_t* out_queue_family_ordinal) {
  amdf_endpoint_info_t info = {.type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO,
                               .structure_size = sizeof(info)};
  auto status = api->endpoint_query_info(endpoint, &info);
  if (!amdf_status_is_ok(status)) {
    return ::testing::AssertionFailure() << "endpoint query: " << status;
  }
  for (uint32_t ordinal = 0; ordinal < info.queue_family_count; ++ordinal) {
    amdf_queue_family_info_t family = {
        .type = AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO,
        .structure_size = sizeof(family)};
    status = api->endpoint_query_queue_family_info(endpoint, ordinal, &family);
    if (!amdf_status_is_ok(status)) {
      return ::testing::AssertionFailure() << "family query: " << status;
    }
    if (family.command_type == AMDF_QUEUE_COMMAND_TYPE_XDNA &&
        family.format_version == AMDF_XDNA_QUEUE_FORMAT_VERSION_1 &&
        (family.publication_modes & AMDF_QUEUE_PUBLICATION_MODE_KERNEL) != 0) {
      *out_queue_family_ordinal = ordinal;
      return ::testing::AssertionSuccess();
    }
  }
  return ::testing::AssertionFailure() << "native XDNA queue family is absent";
}

void XdnaExecution::Prepare(const amdf_api_t* api,
                            const amdf_xdna_api_t* xdna_api,
                            amdf_device_t* device,
                            uint32_t queue_family_ordinal,
                            uint32_t logical_column_count,
                            std::span<const uint8_t> commands,
                            uint64_t command_alignment) {
  amdf_xdna_context_create_info_t context_create = {
      .type = AMDF_STRUCTURE_TYPE_XDNA_CONTEXT_CREATE_INFO,
      .structure_size = sizeof(context_create),
      .logical_column_count = logical_column_count,
      .physical_column_origin = AMDF_XDNA_PHYSICAL_COLUMN_ORIGIN_ANY,
      .acceptable_scheduling_modes = AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED};
  ASSERT_EQ(xdna_api->context_create(device, &context_create, &context),
            AMDF_STATUS_OK);

  amdf_memory_scope_t* scope = nullptr;
  uint32_t scope_count = 0;
  ASSERT_EQ(xdna_api->context_enumerate_memory_scopes(context, 1, &scope,
                                                      &scope_count),
            AMDF_STATUS_OK);
  ASSERT_EQ(scope_count, 1u);
  amdf_memory_device_access_t access = {.device = device};
  access.requirements.access = AMDF_MEMORY_ACCESS_READ |
                               AMDF_MEMORY_ACCESS_WRITE |
                               AMDF_MEMORY_ACCESS_EXECUTE;
  access.requirements.flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
  access.requirements.address_kinds = UINT64_C(1)
                                      << AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE;
  amdf_memory_profile_t profile = {.type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
                                   .structure_size = sizeof(profile)};
  amdf_memory_access_capabilities_t capabilities = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
      .structure_size = sizeof(capabilities)};
  ASSERT_EQ(api->memory_scope_query_device_profile(scope, 0, 1, &access,
                                                   &profile, &capabilities),
            AMDF_STATUS_OK);
  const uint64_t granularity = profile.allocation.byte_length_granularity;
  ASSERT_GT(granularity, 0u);
  amdf_memory_create_info_t create = {};
  create.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
  create.structure_size = sizeof(create);
  create.memory_profile_ordinal = profile.ordinal;
  create.access_count = 1;
  create.accesses = &access;
  create.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
  const uint64_t required_byte_length = commands.size() + command_alignment - 1;
  create.byte_length =
      (required_byte_length + granularity - 1) / granularity * granularity;
  create.minimum_alignment = profile.allocation.minimum_alignment;
  ASSERT_NO_FATAL_FAILURE(instructions.Create(api, scope, create));
  uint64_t firmware_address = 0;
  ASSERT_EQ(api->memory_query_address(instructions.memory, 0,
                                      AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE,
                                      &firmware_address),
            AMDF_STATUS_OK);
  // The profile's allocation alignment and the program's firmware-address
  // alignment are distinct. Place an aligned command slice after querying the
  // actual native address instead of overclaiming the allocation guarantee.
  const uint64_t byte_offset =
      (command_alignment - firmware_address % command_alignment) %
      command_alignment;
  std::fill(instructions.bytes().begin(), instructions.bytes().end(), 0xA5);
  std::copy(commands.begin(), commands.end(),
            instructions.bytes().begin() + byte_offset);
  command = {.memory = instructions.memory,
             .access_ordinal = 0,
             .reserved = 0,
             .byte_offset = byte_offset,
             .byte_length = commands.size()};

  amdf_xdna_kernel_queue_create_info_t queue_create = {
      .type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_CREATE_INFO,
      .structure_size = sizeof(queue_create),
      .queue_family_ordinal = queue_family_ordinal,
      .maximum_pending_submission_count = 1};
  ASSERT_EQ(xdna_api->kernel_queue_create(context, &queue_create, &queue),
            AMDF_STATUS_OK);
}

bool XdnaExecution::Release(const amdf_api_t* api,
                            const amdf_xdna_api_t* xdna_api) {
  if (queue) {
    const auto status = api->kernel_queue_destroy(queue);
    if (status != amdf_make_api_status(AMDF_STATUS_CODE_BUSY)) {
      queue = nullptr;
    }
    EXPECT_EQ(status, AMDF_STATUS_OK);
    if (!amdf_status_is_ok(status)) {
      return false;
    }
  }
  if (!instructions.Release(api)) {
    return false;
  }
  if (context) {
    const auto status = xdna_api->context_destroy(context);
    EXPECT_EQ(status, AMDF_STATUS_OK);
    if (!amdf_status_is_ok(status)) {
      return false;
    }
    context = nullptr;
  }
  return true;
}
