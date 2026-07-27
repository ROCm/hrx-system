// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/epoch_signal_table.h"

#include <vector>

#include "iree/testing/gtest.h"

namespace {

class EpochSignalTable {
 public:
  EpochSignalTable(uint8_t session_epoch, uint8_t machine_index,
                   uint8_t device_index, uint16_t queue_count) {
    storage_.resize(iree_hal_amdgpu_epoch_signal_table_size(queue_count));
    table_ = reinterpret_cast<iree_hal_amdgpu_epoch_signal_table_t*>(
        storage_.data());
    iree_hal_amdgpu_epoch_signal_table_initialize(
        table_, session_epoch, machine_index, device_index, queue_count);
  }

  iree_hal_amdgpu_epoch_signal_table_t* get() { return table_; }

 private:
  std::vector<uint8_t> storage_;
  iree_hal_amdgpu_epoch_signal_table_t* table_;
};

static hsa_signal_t MakeSignal(uint64_t handle) {
  hsa_signal_t signal;
  signal.handle = handle;
  return signal;
}

TEST(EpochSignalTable, ResolvesFlattenedLogicalQueues) {
  EpochSignalTable table(/*session_epoch=*/5, /*machine_index=*/2,
                         /*device_index=*/9, /*queue_count=*/4);
  for (uint8_t queue = 0; queue < 4; ++queue) {
    iree_hal_amdgpu_epoch_signal_table_register(table.get(), queue,
                                                MakeSignal(100 + queue));
  }

  for (uint8_t queue = 0; queue < 4; ++queue) {
    const iree_async_axis_t axis = iree_async_axis_make_queue(5, 2, 9, queue);
    hsa_signal_t signal;
    ASSERT_TRUE(
        iree_hal_amdgpu_epoch_signal_table_lookup(table.get(), axis, &signal));
    EXPECT_EQ(signal.handle, 100u + queue);
  }
}

TEST(EpochSignalTable, RejectsForeignAxes) {
  EpochSignalTable table(/*session_epoch=*/3, /*machine_index=*/7,
                         /*device_index=*/5, /*queue_count=*/2);
  iree_hal_amdgpu_epoch_signal_table_register(table.get(), 0, MakeSignal(100));

  const iree_async_axis_t foreign_axes[] = {
      iree_async_axis_make_queue(4, 7, 5, 0),
      iree_async_axis_make_queue(3, 8, 5, 0),
      iree_async_axis_make_queue(3, 7, 6, 0),
      iree_async_axis_make(3, 7, IREE_ASYNC_CAUSAL_DOMAIN_COLLECTIVE, 0),
      iree_async_axis_make(3, 7, IREE_ASYNC_CAUSAL_DOMAIN_HOST, 0),
  };
  for (iree_async_axis_t axis : foreign_axes) {
    hsa_signal_t signal;
    EXPECT_FALSE(
        iree_hal_amdgpu_epoch_signal_table_lookup(table.get(), axis, &signal));
  }
}

TEST(EpochSignalTable, RejectsUnavailableQueues) {
  EpochSignalTable table(/*session_epoch=*/1, /*machine_index=*/0,
                         /*device_index=*/4, /*queue_count=*/2);
  iree_hal_amdgpu_epoch_signal_table_register(table.get(), 0, MakeSignal(99));

  hsa_signal_t signal;
  EXPECT_FALSE(iree_hal_amdgpu_epoch_signal_table_lookup(
      table.get(), iree_async_axis_make_queue(1, 0, 4, 1), &signal));
  EXPECT_FALSE(iree_hal_amdgpu_epoch_signal_table_lookup(
      table.get(), iree_async_axis_make_queue(1, 0, 4, 2), &signal));
}

TEST(EpochSignalTable, DeregisterRemovesQueue) {
  EpochSignalTable table(/*session_epoch=*/1, /*machine_index=*/0,
                         /*device_index=*/4, /*queue_count=*/2);
  iree_hal_amdgpu_epoch_signal_table_register(table.get(), 1, MakeSignal(77));
  const iree_async_axis_t axis = iree_async_axis_make_queue(1, 0, 4, 1);
  hsa_signal_t signal;
  ASSERT_TRUE(
      iree_hal_amdgpu_epoch_signal_table_lookup(table.get(), axis, &signal));

  iree_hal_amdgpu_epoch_signal_table_deregister(table.get(), 1);
  EXPECT_FALSE(
      iree_hal_amdgpu_epoch_signal_table_lookup(table.get(), axis, &signal));
}

}  // namespace
