// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <atomic>
#include <cstdint>
#include <cstring>

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/device/atomic.h"
#include "iree/hal/drivers/amdgpu/util/aql_emitter.h"
#include "iree/hal/drivers/amdgpu/util/aql_ring.h"
#include "iree/hal/drivers/amdgpu/util/device_library.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/hal/drivers/amdgpu/util/vmem.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

struct QueueError {
  // Number of asynchronous HSA queue errors observed.
  std::atomic<uint32_t> callback_count{0};
  // Last asynchronous HSA queue error status.
  std::atomic<uint32_t> status{HSA_STATUS_SUCCESS};
};

struct alignas(64) LiveMemory {
  // Independently live consumer and producer kernarg blocks.
  alignas(
      16) uint8_t kernargs[2][IREE_HAL_AMDGPU_DEVICE_ATOMIC_WAIT_KERNARG_SIZE];
  // Fine-grained target and host staging word for device-local targets.
  alignas(8) uint64_t target;
};

enum class AtomicTargetKind {
  kFineHost,
  kCoarseDevice,
};

struct AtomicTarget {
  // Name reported with failures from this target.
  const char* name;
  // Memory class controlling host staging and valid scopes.
  AtomicTargetKind kind;
  // Device-visible atomic word.
  uint64_t* ptr;
  // HSA agent owning the memory allocation.
  hsa_agent_t owner_agent;
};

static void HsaQueueErrorCallback(hsa_status_t status, hsa_queue_t* queue,
                                  void* user_data) {
  (void)queue;
  QueueError* error = reinterpret_cast<QueueError*>(user_data);
  error->status.store(static_cast<uint32_t>(status), std::memory_order_relaxed);
  error->callback_count.fetch_add(1, std::memory_order_relaxed);
}

template <typename EmplaceFn>
static void SubmitAtomicDispatch(const iree_hal_amdgpu_libhsa_t* libhsa,
                                 iree_hal_amdgpu_aql_ring_t* ring,
                                 iree_hsa_signal_t completion_signal,
                                 EmplaceFn emplace) {
  iree_hsa_signal_store_screlease(IREE_LIBHSA(libhsa), completion_signal, 1);
  const uint64_t packet_id = iree_hal_amdgpu_aql_ring_reserve(ring, 1);
  iree_hal_amdgpu_aql_packet_t* packet =
      iree_hal_amdgpu_aql_ring_packet(ring, packet_id);
  std::memset(packet, 0, sizeof(*packet));
  emplace(&packet->dispatch);
  packet->dispatch.completion_signal = completion_signal;
  const uint16_t header = iree_hal_amdgpu_aql_make_header(
      IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH,
      iree_hal_amdgpu_aql_packet_control_barrier_system());
  iree_hal_amdgpu_aql_ring_commit(packet, header, packet->dispatch.setup);
  iree_hal_amdgpu_aql_ring_doorbell(ring, packet_id);
}

static void WaitForCompletion(const iree_hal_amdgpu_libhsa_t* libhsa,
                              iree_hsa_signal_t completion_signal) {
  EXPECT_EQ(iree_hsa_signal_wait_scacquire(
                IREE_LIBHSA(libhsa), completion_signal, HSA_SIGNAL_CONDITION_EQ,
                /*compare_value=*/0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED),
            0);
}

static void CopyMemoryAndWait(const iree_hal_amdgpu_libhsa_t* libhsa,
                              void* target, hsa_agent_t target_agent,
                              const void* source, hsa_agent_t source_agent,
                              iree_device_size_t byte_length,
                              iree_hsa_signal_t completion_signal) {
  iree_hsa_signal_store_screlease(IREE_LIBHSA(libhsa), completion_signal, 1);
  IREE_ASSERT_OK(iree_hsa_amd_memory_async_copy(
      IREE_LIBHSA(libhsa), target, target_agent, source, source_agent,
      byte_length, /*num_dep_signals=*/0, /*dep_signals=*/nullptr,
      completion_signal));
  WaitForCompletion(libhsa, completion_signal);
}

template <typename EmplaceFn>
static void SubmitAtomicDispatchAndWait(const iree_hal_amdgpu_libhsa_t* libhsa,
                                        iree_hal_amdgpu_aql_ring_t* ring,
                                        iree_hsa_signal_t completion_signal,
                                        EmplaceFn emplace) {
  SubmitAtomicDispatch(libhsa, ring, completion_signal, emplace);
  WaitForCompletion(libhsa, completion_signal);
}

static uint64_t ApplyRmw(iree_hal_atomic_width_t width,
                         iree_hal_atomic_rmw_operation_t operation,
                         uint64_t value, uint64_t operand) {
  if (width == IREE_HAL_ATOMIC_WIDTH_32) {
    const uint32_t value32 = static_cast<uint32_t>(value);
    const uint32_t operand32 = static_cast<uint32_t>(operand);
    switch (operation) {
      case IREE_HAL_ATOMIC_RMW_OPERATION_ADD:
        return value32 + operand32;
      case IREE_HAL_ATOMIC_RMW_OPERATION_SUBTRACT:
        return value32 - operand32;
      case IREE_HAL_ATOMIC_RMW_OPERATION_AND:
        return value32 & operand32;
      case IREE_HAL_ATOMIC_RMW_OPERATION_OR:
        return value32 | operand32;
      case IREE_HAL_ATOMIC_RMW_OPERATION_XOR:
        return value32 ^ operand32;
      default:
        IREE_ASSERT_UNREACHABLE("atomic RMW operation must be validated");
        return value32;
    }
  }
  switch (operation) {
    case IREE_HAL_ATOMIC_RMW_OPERATION_ADD:
      return value + operand;
    case IREE_HAL_ATOMIC_RMW_OPERATION_SUBTRACT:
      return value - operand;
    case IREE_HAL_ATOMIC_RMW_OPERATION_AND:
      return value & operand;
    case IREE_HAL_ATOMIC_RMW_OPERATION_OR:
      return value | operand;
    case IREE_HAL_ATOMIC_RMW_OPERATION_XOR:
      return value ^ operand;
    default:
      IREE_ASSERT_UNREACHABLE("atomic RMW operation must be validated");
      return value;
  }
}

class AtomicLiveTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    host_allocator = iree_allocator_system();
    iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
        IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
        host_allocator, &libhsa);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      GTEST_SKIP() << "HSA not available, skipping tests";
    }
    IREE_ASSERT_OK(
        iree_hal_amdgpu_topology_initialize_with_defaults(&libhsa, &topology));
    if (topology.gpu_agent_count == 0 || topology.cpu_agent_count == 0) {
      GTEST_SKIP() << "CPU and GPU agents are required, skipping tests";
    }
    for (iree_host_size_t i = 0; i < topology.gpu_agent_count; ++i) {
      IREE_ASSERT_OK(iree_hal_amdgpu_agent_target_query(
          &libhsa, topology.gpu_agents[i], host_allocator,
          &gpu_agent_targets[i]));
    }
  }

  static void TearDownTestSuite() {
    for (iree_host_size_t i = 0; i < topology.gpu_agent_count; ++i) {
      iree_hal_amdgpu_agent_target_deinitialize(&gpu_agent_targets[i]);
    }
    iree_hal_amdgpu_topology_deinitialize(&topology);
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa);
  }

  static iree_allocator_t host_allocator;
  static iree_hal_amdgpu_libhsa_t libhsa;
  static iree_hal_amdgpu_topology_t topology;
  static iree_hal_amdgpu_agent_target_t
      gpu_agent_targets[IREE_HAL_AMDGPU_MAX_GPU_AGENT];
};

iree_allocator_t AtomicLiveTest::host_allocator;
iree_hal_amdgpu_libhsa_t AtomicLiveTest::libhsa;
iree_hal_amdgpu_topology_t AtomicLiveTest::topology;
iree_hal_amdgpu_agent_target_t
    AtomicLiveTest::gpu_agent_targets[IREE_HAL_AMDGPU_MAX_GPU_AGENT];

TEST_F(AtomicLiveTest, CompleteKernelMatrix) {
  const hsa_agent_t cpu_agent = topology.cpu_agents[0];
  const hsa_agent_t gpu_agent = topology.gpu_agents[0];

  iree_hal_amdgpu_device_library_t library = {};
  IREE_ASSERT_OK(iree_hal_amdgpu_device_library_initialize(
      &libhsa, &topology, gpu_agent_targets, host_allocator, &library));
  iree_hal_amdgpu_device_kernels_t kernels = {};
  IREE_ASSERT_OK(iree_hal_amdgpu_device_library_populate_agent_kernels(
      &library, gpu_agent, &kernels));

  hsa_amd_memory_pool_t host_memory_pool = {};
  IREE_ASSERT_OK(iree_hal_amdgpu_find_fine_global_memory_pool(
      &libhsa, cpu_agent, &host_memory_pool));
  LiveMemory* memory = nullptr;
  IREE_ASSERT_OK(iree_hsa_amd_memory_pool_allocate(
      IREE_LIBHSA(&libhsa), host_memory_pool, sizeof(*memory),
      HSA_AMD_MEMORY_POOL_STANDARD_FLAG, reinterpret_cast<void**>(&memory)));
  IREE_ASSERT_OK(iree_hsa_amd_agents_allow_access(IREE_LIBHSA(&libhsa),
                                                  /*num_agents=*/1, &gpu_agent,
                                                  /*flags=*/nullptr, memory));
  std::memset(memory, 0, sizeof(*memory));

  hsa_amd_memory_pool_t device_memory_pool = {};
  IREE_ASSERT_OK(iree_hal_amdgpu_find_coarse_global_memory_pool(
      &libhsa, gpu_agent, &device_memory_pool));
  uint64_t* device_target = nullptr;
  IREE_ASSERT_OK(iree_hsa_amd_memory_pool_allocate(
      IREE_LIBHSA(&libhsa), device_memory_pool, sizeof(*device_target),
      HSA_AMD_MEMORY_POOL_STANDARD_FLAG,
      reinterpret_cast<void**>(&device_target)));

  QueueError queue_errors[2];
  hsa_queue_t* queues[2] = {};
  iree_hal_amdgpu_aql_ring_t rings[2] = {};
  iree_hsa_signal_t completion_signals[2] = {};
  for (uint32_t i = 0; i < 2; ++i) {
    IREE_ASSERT_OK(iree_hsa_queue_create(
        IREE_LIBHSA(&libhsa), gpu_agent, /*size=*/64, HSA_QUEUE_TYPE_MULTI,
        HsaQueueErrorCallback, &queue_errors[i], UINT32_MAX, UINT32_MAX,
        &queues[i]));
    iree_hal_amdgpu_aql_ring_initialize(
        &libhsa, reinterpret_cast<iree_amd_queue_t*>(queues[i]), &rings[i]);
    IREE_ASSERT_OK(iree_hsa_amd_signal_create(
        IREE_LIBHSA(&libhsa), /*initial_value=*/1, /*num_consumers=*/0,
        /*consumers=*/nullptr, /*attributes=*/0, &completion_signals[i]));
  }

  const iree_hal_atomic_width_t widths[] = {
      IREE_HAL_ATOMIC_WIDTH_32,
      IREE_HAL_ATOMIC_WIDTH_64,
  };
  const AtomicTarget targets[] = {
      {
          /*.name=*/"fine-host",
          /*.kind=*/AtomicTargetKind::kFineHost,
          /*.ptr=*/&memory->target,
          /*.owner_agent=*/cpu_agent,
      },
      {
          /*.name=*/"coarse-device",
          /*.kind=*/AtomicTargetKind::kCoarseDevice,
          /*.ptr=*/device_target,
          /*.owner_agent=*/gpu_agent,
      },
  };
  auto write_target = [&](const AtomicTarget& target, uint64_t value) {
    memory->target = value;
    if (target.kind == AtomicTargetKind::kCoarseDevice) {
      CopyMemoryAndWait(&libhsa, target.ptr, target.owner_agent,
                        &memory->target, cpu_agent, sizeof(memory->target),
                        completion_signals[0]);
    }
  };
  auto read_target = [&](const AtomicTarget& target) {
    if (target.kind == AtomicTargetKind::kCoarseDevice) {
      CopyMemoryAndWait(&libhsa, &memory->target, cpu_agent, target.ptr,
                        target.owner_agent, sizeof(memory->target),
                        completion_signals[0]);
    }
    return memory->target;
  };

  for (const AtomicTarget& target : targets) {
    SCOPED_TRACE(target.name);
    const uint32_t scope_count =
        target.kind == AtomicTargetKind::kFineHost ? 2u : 1u;
    for (const iree_hal_atomic_width_t width : widths) {
      const uint64_t width_mask =
          width == IREE_HAL_ATOMIC_WIDTH_32 ? UINT32_MAX : UINT64_MAX;

      for (uint32_t scope = 0; scope < scope_count; ++scope) {
        for (uint32_t release = 0; release < 2; ++release) {
          const iree_hal_atomic_flags_t flags =
              IREE_HAL_ATOMIC_FLAG_ACQUIRE |
              (release ? IREE_HAL_ATOMIC_FLAG_RELEASE : 0) |
              (scope ? IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE : 0);
          const uint64_t value = 0x89ABCDEF01234567ull & width_mask;
          write_target(target, 0);
          const iree_hal_atomic_store_params_t params = {
              /*.value=*/value,
              /*.flags=*/flags,
              /*.width=*/width,
          };
          SubmitAtomicDispatchAndWait(
              &libhsa, &rings[0], completion_signals[0],
              [&](iree_hsa_kernel_dispatch_packet_t* packet) {
                iree_hal_amdgpu_device_atomic_store_emplace(
                    &kernels, packet, target.ptr, params, memory->kernargs[0]);
              });
          EXPECT_EQ(read_target(target) & width_mask, value);
        }
      }

      const iree_hal_atomic_rmw_operation_t operations[] = {
          IREE_HAL_ATOMIC_RMW_OPERATION_ADD,
          IREE_HAL_ATOMIC_RMW_OPERATION_SUBTRACT,
          IREE_HAL_ATOMIC_RMW_OPERATION_AND,
          IREE_HAL_ATOMIC_RMW_OPERATION_OR,
          IREE_HAL_ATOMIC_RMW_OPERATION_XOR,
      };
      for (uint32_t scope = 0; scope < scope_count; ++scope) {
        for (uint32_t order = 0; order < 4; ++order) {
          const iree_hal_atomic_flags_t flags =
              (order & 1 ? IREE_HAL_ATOMIC_FLAG_ACQUIRE : 0) |
              (order & 2 ? IREE_HAL_ATOMIC_FLAG_RELEASE : 0) |
              (scope ? IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE : 0);
          for (const iree_hal_atomic_rmw_operation_t operation : operations) {
            const uint64_t initial = 0x76543210FEDCBA98ull & width_mask;
            const uint64_t operand = 0x111111110F0F0F0Full & width_mask;
            write_target(target, initial);
            const iree_hal_atomic_rmw_params_t params = {
                /*.operand=*/operand,
                /*.flags=*/flags,
                /*.width=*/width,
                /*.operation=*/operation,
            };
            SubmitAtomicDispatchAndWait(
                &libhsa, &rings[0], completion_signals[0],
                [&](iree_hsa_kernel_dispatch_packet_t* packet) {
                  iree_hal_amdgpu_device_atomic_rmw_emplace(
                      &kernels, packet, target.ptr, params,
                      memory->kernargs[0]);
                });
            EXPECT_EQ(read_target(target) & width_mask,
                      ApplyRmw(width, operation, initial, operand));
          }
        }
      }

      const iree_hal_atomic_wait_condition_t conditions[] = {
          IREE_HAL_ATOMIC_WAIT_CONDITION_EQUAL,
          IREE_HAL_ATOMIC_WAIT_CONDITION_NOT_EQUAL,
          IREE_HAL_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL,
      };
      for (uint32_t scope = 0; scope < scope_count; ++scope) {
        for (uint32_t acquire = 0; acquire < 2; ++acquire) {
          const iree_hal_atomic_flags_t flags =
              IREE_HAL_ATOMIC_FLAG_RELEASE |
              (acquire ? IREE_HAL_ATOMIC_FLAG_ACQUIRE : 0) |
              (scope ? IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE : 0);
          for (const iree_hal_atomic_wait_condition_t condition : conditions) {
            write_target(target, 0x1234u);
            const uint64_t value =
                condition == IREE_HAL_ATOMIC_WAIT_CONDITION_NOT_EQUAL ? 0x35u
                : condition ==
                        IREE_HAL_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL
                    ? 0x30u
                    : 0x34u;
            const iree_hal_atomic_wait_params_t params = {
                /*.value=*/value,
                /*.mask=*/0xFFu,
                /*.flags=*/flags,
                /*.width=*/width,
                /*.condition=*/condition,
            };
            SubmitAtomicDispatchAndWait(
                &libhsa, &rings[0], completion_signals[0],
                [&](iree_hsa_kernel_dispatch_packet_t* packet) {
                  iree_hal_amdgpu_device_atomic_wait_emplace(
                      &kernels, packet, target.ptr, params,
                      memory->kernargs[0]);
                });
          }
        }
      }

      write_target(target, 0);
      const iree_hal_atomic_flags_t cross_queue_flags =
          target.kind == AtomicTargetKind::kFineHost
              ? IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE
              : IREE_HAL_ATOMIC_FLAG_NONE;
      const iree_hal_atomic_wait_params_t wait_params = {
          /*.value=*/1,
          /*.mask=*/width_mask,
          /*.flags=*/IREE_HAL_ATOMIC_FLAG_ACQUIRE | cross_queue_flags,
          /*.width=*/width,
          /*.condition=*/IREE_HAL_ATOMIC_WAIT_CONDITION_EQUAL,
      };
      SubmitAtomicDispatch(&libhsa, &rings[0], completion_signals[0],
                           [&](iree_hsa_kernel_dispatch_packet_t* packet) {
                             iree_hal_amdgpu_device_atomic_wait_emplace(
                                 &kernels, packet, target.ptr, wait_params,
                                 memory->kernargs[0]);
                           });
      const iree_hal_atomic_store_params_t store_params = {
          /*.value=*/1,
          /*.flags=*/IREE_HAL_ATOMIC_FLAG_RELEASE | cross_queue_flags,
          /*.width=*/width,
      };
      SubmitAtomicDispatch(&libhsa, &rings[1], completion_signals[1],
                           [&](iree_hsa_kernel_dispatch_packet_t* packet) {
                             iree_hal_amdgpu_device_atomic_store_emplace(
                                 &kernels, packet, target.ptr, store_params,
                                 memory->kernargs[1]);
                           });
      WaitForCompletion(&libhsa, completion_signals[1]);
      WaitForCompletion(&libhsa, completion_signals[0]);
      EXPECT_EQ(read_target(target) & width_mask, 1u);
    }
  }

  for (uint32_t i = 0; i < 2; ++i) {
    EXPECT_EQ(queue_errors[i].callback_count.load(std::memory_order_relaxed),
              0u);
    EXPECT_EQ(queue_errors[i].status.load(std::memory_order_relaxed),
              static_cast<uint32_t>(HSA_STATUS_SUCCESS));
    IREE_ASSERT_OK(
        iree_hsa_signal_destroy(IREE_LIBHSA(&libhsa), completion_signals[i]));
    IREE_ASSERT_OK(iree_hsa_queue_destroy(IREE_LIBHSA(&libhsa), queues[i]));
  }
  IREE_ASSERT_OK(
      iree_hsa_amd_memory_pool_free(IREE_LIBHSA(&libhsa), device_target));
  IREE_ASSERT_OK(iree_hsa_amd_memory_pool_free(IREE_LIBHSA(&libhsa), memory));
  iree_hal_amdgpu_device_library_deinitialize(&library);
}

}  // namespace
}  // namespace iree::hal::amdgpu
