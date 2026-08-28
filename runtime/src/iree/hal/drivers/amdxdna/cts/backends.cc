// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS backend registration for the amdxdna HAL driver.

#include "iree/hal/api.h"
#include "iree/hal/cts/util/registry.h"
#include "iree/hal/drivers/amdxdna/registration/driver_module.h"

namespace iree::hal::cts {

static iree_status_t CreateAmdxdnaDevice(
    const iree_hal_device_create_params_t* create_params,
    iree_hal_driver_t** out_driver, iree_hal_device_t** out_device) {
  iree_status_t status = iree_hal_amdxdna_driver_module_register(
      iree_hal_driver_registry_default());
  if (iree_status_is_already_exists(status)) {
    iree_status_ignore(status);
    status = iree_ok_status();
  }

  iree_hal_driver_t* driver = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_registry_try_create(
        iree_hal_driver_registry_default(), iree_make_cstring_view("amdxdna"),
        iree_allocator_system(), &driver);
  }

  iree_hal_device_t* device = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_create_default_device(
        driver, create_params, iree_allocator_system(), &device);
  }

  if (iree_status_is_ok(status)) {
    *out_driver = driver;
    *out_device = device;
  } else {
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
  }
  return status;
}

static bool amdxdna_registered_ =
    (CtsRegistry::RegisterBackend({
         "amdxdna",
         {"amdxdna",
          CreateAmdxdnaDevice,
          /*executable_target_family=*/nullptr,
          /*executable_target_key=*/nullptr,
          /*executable_data=*/nullptr,
          RecordingMode::kDirect,
          /*unsupported_tests=*/
          {
              // amdxdna has a single XDNA hwctx and a single hardware
              // queue. The four tests below assume cross-queue release/
              // alloca interleaving or pool-notification retry semantics
              // that are only meaningful for drivers with multiple physical
              // queues. local_sync skips them for the same reason.
              //
              // Revisit this when a real workload measurably benefits from
              // parallel hwctx submission on disjoint AIE column groups. The
              // XDNA kernel and IREE HAL both support multi-queue today;
              // amdxdna would need ~500-800 LOC ported from the amdgpu HAL
              // pattern to enable it. The linked issue tracks the benchmark
              // plan and decision criteria.
              {"QueueAllocaTest.ExplicitFixedBlockPoolCrossQueueWaitFrontier",
               "single-queue driver: pool's OK_NEEDS_WAIT path is only "
               "meaningful when peer queues release while the freed work is "
               "still in flight on another queue."},
              {"QueueAllocaTest.ExplicitFixedBlockPoolRequiresWaitFrontierFlag",
               "single-queue driver: cannot distinguish async queue-owned "
               "hidden frontier waits from pool-notification retries when "
               "all alloca/dealloca ordering is on one physical queue."},
              {"QueueAllocaTest.ExplicitTLSFPoolCrossQueueStaleBlockGrows",
               "single-queue driver: stale cross-queue block growth requires "
               "a peer queue still holding the released reservation, which "
               "this backend cannot model."},
              {"QueueAllocaTest.ExplicitFixedBlockPoolNotificationRetry",
               "single-queue driver: cannot submit the dealloca that "
               "releases the first block while the second alloca is waiting "
               "on pool notification on the same physical queue."},
              // Native amdxdna queues do not support blit/fill/update/copy
              // commands yet. Keep these CTS cases disabled instead of
              // reintroducing blocking host-emulated transfers behind queue
              // entry points. TODO(#amdxdna): remove these once native blit
              // support exists and QUEUE_TRANSFER compatibility can be
              // advertised.
              {"AllocatorTest.BaselineBufferCompatibility",
               "amdxdna does not advertise QUEUE_TRANSFER compatibility until "
               "native blit support exists."},
              {"AllocatorTest.AllocateBuffer",
               "amdxdna HOST_ONLY memory rejects DEVICE_LOCAL without OPTIMAL; "
               "CTS still requests DEVICE_LOCAL as a HIP-shaped default."},
              {"AllocatorTest.AllocateEmptyBuffer",
               "amdxdna HOST_ONLY memory rejects DEVICE_LOCAL without OPTIMAL; "
               "CTS still requests DEVICE_LOCAL as a HIP-shaped default."},
              {"QueueAllocaTest.BasicAlloca",
               "CTS validates allocated contents with queue_fill; amdxdna does "
               "not expose queue transfer operations until native blit support "
               "exists."},
              {"QueueAllocaTest.ExplicitPassthroughPoolAllocaDealloca",
               "CTS validates allocated contents with queue_fill; amdxdna does "
               "not expose queue transfer operations until native blit support "
               "exists."},
              {"QueueAllocaTest.ExplicitTLSFPoolTransferAllocaDealloca",
               "CTS validates allocated contents with queue_fill; amdxdna does "
               "not expose queue transfer operations until native blit support "
               "exists."},
              {"QueueAllocaTest."
               "ExplicitFixedBlockPoolPendingDeallocaWaitFrontier",
               "CTS validates allocated contents with queue_fill; amdxdna does "
               "not expose queue transfer operations until native blit support "
               "exists."},
              {"QueueAllocaTest.AllocaWithWaitSemaphores",
               "CTS validates allocated contents with queue_fill; amdxdna does "
               "not expose queue transfer operations until native blit support "
               "exists."},
              {"QueueAllocaTest.AllocaDeallocaCycle",
               "CTS validates allocated contents with queue_fill; amdxdna does "
               "not expose queue transfer operations until native blit support "
               "exists."},
              {"QueueAllocaTest.DeallocaReleasesMemory",
               "CTS validates allocated contents with queue_fill; amdxdna does "
               "not expose queue transfer operations until native blit support "
               "exists."},
              {"QueueAllocaTest.FailedDeallocaWaitDoesNotDealloca",
               "CTS validates allocated contents with queue_fill; amdxdna does "
               "not expose queue transfer operations until native blit support "
               "exists."},
              {"QueueAllocaTest.ZeroAccessFlagsCanonicalized",
               "CTS validates allocated contents with queue_fill; amdxdna does "
               "not expose queue transfer operations until native blit support "
               "exists."},
              {"QueueTransferTest.*",
               "amdxdna does not expose queue transfer operations until it has "
               "native blit support; host-emulated map/sync/memcpy transfers "
               "are intentionally unsupported on device queues."},
              {"CommandBufferFillBufferTest.*",
               "amdxdna command buffers require native blit support for fill "
               "commands; host-emulated transfer commands are unsupported."},
              {"CommandBufferUpdateBufferTest.*",
               "amdxdna command buffers require native blit support for update "
               "commands; host-emulated transfer commands are unsupported."},
              {"CommandBufferCopyBufferTest.*",
               "amdxdna command buffers require native blit support for copy "
               "commands; host-emulated transfer commands are unsupported."},
              {"CommandBufferStressTest.*",
               "amdxdna command-buffer transfer stress cases require native "
               "blit support."},
              {"TransientBufferTest.*",
               "amdxdna transient-buffer CTS cases exercise command-buffer "
               "transfer commands, which require native blit support."},
              {"AsyncTransientBufferTest.*",
               "amdxdna transient-buffer CTS cases exercise command-buffer "
               "transfer commands, which require native blit support."},
              {"SemaphoreSubmissionTest."
               "IndirectCommandBufferBindingTableRetainedUntilSignal",
               "this CTS case is specifically a command-buffer copy test; "
               "amdxdna does not expose copy commands until native blits "
               "exist."},
          },
          /*expected_failures=*/{}},
         /*tags=*/{"allocator", "buffer_mapping", "driver"},
     }),
     true);

}  // namespace iree::hal::cts
