// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/transfer.h"

#include <array>
#include <vector>

#include "iree/hal/device_spec.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(TransferTest, ProgramPromotionRequiresAnExecutingFamily) {
  const uint16_t types[] = {IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS};
  const iree_hal_buffer_binding_layout_t layout = {
      sizeof(iree_hal_buffer_native_binding_t), 1, 0, types};
  iree_hal_memory_contract_t* contract = nullptr;
  IREE_ASSERT_OK(iree_hal_memory_contract_create(
      &layout, 3, &layout, iree_allocator_system(), &contract));
  for (uint32_t scope : {1u, 2u}) {
    contract->scopes[scope].interfaces =
        1u << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS;
    contract->scopes[scope].usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
    contract->scopes[scope].bindings[IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS] =
        0;
  }
  const iree_hal_buffer_native_binding_t native = {.device_address = 0x1000};
  iree_hal_buffer_t buffer;
  const iree_hal_buffer_vtable_t vtable = {
      .destroy = [](iree_hal_buffer_t*) {},
  };
  iree_hal_buffer_initialize({}, &buffer, 64, 0, 64,
                             IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
                             IREE_HAL_MEMORY_ACCESS_ALL,
                             IREE_HAL_BUFFER_USAGE_TRANSFER, &vtable, &buffer);
  buffer.memory = {.contract = contract, .bindings = &native};
  std::array<iree_hal_transfer_operation_t, 2> operations = {};
  for (uint32_t i = 0; i < operations.size(); ++i) {
    operations[i] = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
        .copy =
            {
                .source_buffer = &buffer,
                .source_offset = i * 16,
                .target_buffer = &buffer,
                .target_offset = 32 + i * 16,
                .length = 16,
            },
    };
  }
  iree_hal_amd_gpu_transfer_layout_t capture_layout;
  IREE_ASSERT_OK(iree_hal_amd_gpu_transfer_measure(
      operations.size(), operations.data(), &capture_layout));
  std::vector<uint64_t> storage((capture_layout.byte_length + 7) / 8);
  for (const bool dispatch : {false, true}) {
    iree_hal_queue_family_spec_t spec = {};
    spec.role_flags = IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER |
                      (dispatch ? IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_DISPATCH : 0);
    const iree_hal_queue_family_t family = {
        .spec = &spec,
        .memory = {.domain = &layout, .queue_scope_id = 1},
    };
    iree_hal_amd_gpu_transfer_t transfer;
    iree_status_t status = iree_hal_amd_gpu_transfer_initialize(
        &family, IREE_HAL_MEMORY_SITE_QUEUE, {}, operations.size(),
        operations.data(), capture_layout, storage.data(), &transfer);
    IREE_EXPECT_OK(status);
    EXPECT_EQ(transfer.program_accessible, dispatch);
    iree_hal_amd_gpu_transfer_deinitialize(&transfer);
  }
  EXPECT_EQ(iree_atomic_ref_count_load(&buffer.resource.ref_count), 1);
  iree_hal_buffer_release(&buffer);
  iree_hal_memory_contract_release(contract);
}

TEST(DownloadTest, PartialEndpointsPreserveEveryDestinationAndGuard) {
  std::array<uint8_t, 24> staging;
  for (size_t i = 0; i < staging.size(); ++i) {
    staging[i] = static_cast<uint8_t>(i * 7 + 1);
  }
  std::array<std::array<uint8_t, 10>, 3> targets;
  std::array<iree_hal_amd_gpu_transfer_operation_t, 3> operations = {};
  for (size_t i = 0; i < operations.size(); ++i) {
    operations[i] = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .target = {.host = targets[i].data() + 1},
        .length = 8,
    };
  }
  // Every start/end combination includes empty, partial, and complete rows.
  for (size_t offset = 0; offset < 8; ++offset) {
    for (size_t length = 0; length <= staging.size() - offset; ++length) {
      SCOPED_TRACE(::testing::Message() << offset << '+' << length);
      for (auto& target : targets) {
        target.fill(0xCC);
      }
      const iree_hal_amd_gpu_download_t download = {
          .source = staging.data(),
          .operation = operations.data(),
          .offset = offset,
          .length = length,
      };
      iree_hal_amd_gpu_download_complete(&download);
      for (size_t row = 0; row < targets.size(); ++row) {
        EXPECT_EQ(targets[row].front(), 0xCC);
        EXPECT_EQ(targets[row].back(), 0xCC);
        for (size_t i = 0; i < 8; ++i) {
          const size_t position = row * 8 + i;
          EXPECT_EQ(targets[row][i + 1],
                    position >= offset && position < offset + length
                        ? staging[position - offset]
                        : 0xCC);
        }
      }
    }
  }
}

}  // namespace
