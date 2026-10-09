// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/internal/debugging.h"
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
  // Unretained HSA dispatch table used to publish terminal errors.
  const iree_hal_amdgpu_libhsa_t* libhsa = nullptr;
  // Host-only signal set to zero when the queue enters a terminal error state.
  iree_hsa_signal_t signal = {};
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
  IREE_TSAN_ACQUIRE(error);
  error->status.store(static_cast<uint32_t>(status), std::memory_order_relaxed);
  error->callback_count.fetch_add(1, std::memory_order_relaxed);
  IREE_TSAN_RELEASE(error);
  iree_hsa_signal_store_screlease(IREE_LIBHSA(error->libhsa), error->signal, 0);
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
  uint16_t setup = 0;
  emplace(&packet->dispatch, &setup);
  packet->dispatch.completion_signal = completion_signal;
  const uint16_t header = iree_hal_amdgpu_aql_make_header(
      IREE_HSA_PACKET_TYPE_KERNEL_DISPATCH,
      iree_hal_amdgpu_aql_packet_control_barrier_system());
  iree_hal_amdgpu_aql_ring_commit(packet, header, setup);
  iree_hal_amdgpu_aql_ring_doorbell(ring, packet_id);
}

static bool WaitForSignal(const iree_hal_amdgpu_libhsa_t* libhsa,
                          iree_hsa_signal_t signal) {
  const hsa_signal_value_t value = iree_hsa_signal_wait_scacquire(
      IREE_LIBHSA(libhsa), signal, HSA_SIGNAL_CONDITION_EQ,
      /*compare_value=*/0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
  if (value == 0) {
    return true;
  }
  ADD_FAILURE() << "HSA signal wait returned unexpected value " << value;
  return false;
}

static bool WaitForQueueCompletion(const iree_hal_amdgpu_libhsa_t* libhsa,
                                   iree_hsa_signal_t completion_signal,
                                   QueueError* queue_error) {
  enum {
    kCompletionSignalIndex = 0,
    kErrorSignalIndex = 1,
    kSignalCount = 2,
  };
  hsa_signal_t signals[kSignalCount] = {
      completion_signal,
      queue_error->signal,
  };
  hsa_signal_condition_t conditions[kSignalCount] = {
      HSA_SIGNAL_CONDITION_EQ,
      HSA_SIGNAL_CONDITION_EQ,
  };
  hsa_signal_value_t values[kSignalCount] = {0, 0};
  const uint32_t signal_index = iree_hsa_amd_signal_wait_any(
      IREE_LIBHSA(libhsa), kSignalCount, signals, conditions, values,
      UINT64_MAX, HSA_WAIT_STATE_BLOCKED, /*satisfying_value=*/nullptr);
  if (signal_index >= kSignalCount) {
    ADD_FAILURE() << "hsa_amd_signal_wait_any returned invalid signal index "
                  << signal_index;
    return false;
  }
  // The multi-signal wait is relaxed. Both signals remain satisfied until this
  // caller returns; acquire the observed signal before consuming device writes
  // or callback state, or allowing the next dispatch to reuse kernarg storage.
  (void)iree_hsa_signal_load_scacquire(IREE_LIBHSA(libhsa),
                                       signals[signal_index]);
  if (signal_index == kErrorSignalIndex) {
    IREE_TSAN_ACQUIRE(queue_error);
    ADD_FAILURE() << "HSA queue entered terminal error state "
                  << queue_error->status.load(std::memory_order_relaxed);
    return false;
  }
  return true;
}

static bool CopyMemoryAndWait(const iree_hal_amdgpu_libhsa_t* libhsa,
                              void* target, hsa_agent_t target_agent,
                              const void* source, hsa_agent_t source_agent,
                              iree_device_size_t byte_length,
                              iree_hsa_signal_t completion_signal) {
  iree_hsa_signal_store_screlease(IREE_LIBHSA(libhsa), completion_signal, 1);
  iree_status_t status = iree_hsa_amd_memory_async_copy(
      IREE_LIBHSA(libhsa), target, target_agent, source, source_agent,
      byte_length, /*num_dep_signals=*/0, /*dep_signals=*/nullptr,
      completion_signal);
  if (!iree_status_is_ok(status)) {
    IREE_EXPECT_OK(status);
    return false;
  }
  return WaitForSignal(libhsa, completion_signal);
}

template <typename EmplaceFn>
static bool SubmitAtomicDispatchAndWait(const iree_hal_amdgpu_libhsa_t* libhsa,
                                        iree_hal_amdgpu_aql_ring_t* ring,
                                        iree_hsa_signal_t completion_signal,
                                        QueueError* queue_error,
                                        EmplaceFn emplace) {
  SubmitAtomicDispatch(libhsa, ring, completion_signal, emplace);
  return WaitForQueueCompletion(libhsa, completion_signal, queue_error);
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

enum class AtomicOperation {
  kStore,
  kAdd,
  kSubtract,
  kAnd,
  kOr,
  kXor,
  kWaitEqual,
  kWaitNotEqual,
  kWaitGreaterEqual,
  kWaitBeforeStore,
};

// Each parameter is one independently seeded and completed matrix case.
struct AtomicCase {
  // Allocation class used by the atomic target.
  AtomicTargetKind target_kind;
  // Width of the atomic word.
  iree_hal_atomic_width_t width;
  // Builtin operation or consumer-first cross-queue pair.
  AtomicOperation operation;
  // Ordering and scope flags passed to the builtin.
  iree_hal_atomic_flags_t flags;
};

static std::vector<AtomicCase> GenerateAtomicCases() {
  std::vector<AtomicCase> cases;
  for (AtomicTargetKind target_kind :
       {AtomicTargetKind::kFineHost, AtomicTargetKind::kCoarseDevice}) {
    const uint32_t scope_count =
        target_kind == AtomicTargetKind::kFineHost ? 2u : 1u;
    for (iree_hal_atomic_width_t width :
         {IREE_HAL_ATOMIC_WIDTH_32, IREE_HAL_ATOMIC_WIDTH_64}) {
      for (uint32_t scope = 0; scope < scope_count; ++scope) {
        for (uint32_t release = 0; release < 2; ++release) {
          // The store builtin discards acquire; exercise that input as well.
          const iree_hal_atomic_flags_t flags =
              IREE_HAL_ATOMIC_FLAG_ACQUIRE |
              (release ? IREE_HAL_ATOMIC_FLAG_RELEASE : 0) |
              (scope ? IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE : 0);
          cases.push_back({target_kind, width, AtomicOperation::kStore, flags});
        }
      }
      for (uint32_t scope = 0; scope < scope_count; ++scope) {
        for (uint32_t order = 0; order < 4; ++order) {
          const iree_hal_atomic_flags_t flags =
              (order & 1 ? IREE_HAL_ATOMIC_FLAG_ACQUIRE : 0) |
              (order & 2 ? IREE_HAL_ATOMIC_FLAG_RELEASE : 0) |
              (scope ? IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE : 0);
          for (AtomicOperation operation :
               {AtomicOperation::kAdd, AtomicOperation::kSubtract,
                AtomicOperation::kAnd, AtomicOperation::kOr,
                AtomicOperation::kXor}) {
            cases.push_back({target_kind, width, operation, flags});
          }
        }
      }
      for (uint32_t scope = 0; scope < scope_count; ++scope) {
        for (uint32_t acquire = 0; acquire < 2; ++acquire) {
          // The wait builtin discards release; exercise that input as well.
          const iree_hal_atomic_flags_t flags =
              IREE_HAL_ATOMIC_FLAG_RELEASE |
              (acquire ? IREE_HAL_ATOMIC_FLAG_ACQUIRE : 0) |
              (scope ? IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE : 0);
          for (AtomicOperation operation :
               {AtomicOperation::kWaitEqual, AtomicOperation::kWaitNotEqual,
                AtomicOperation::kWaitGreaterEqual}) {
            cases.push_back({target_kind, width, operation, flags});
          }
        }
      }
      const iree_hal_atomic_flags_t cross_queue_flags =
          target_kind == AtomicTargetKind::kFineHost
              ? IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE
              : IREE_HAL_ATOMIC_FLAG_NONE;
      cases.push_back({target_kind, width, AtomicOperation::kWaitBeforeStore,
                       IREE_HAL_ATOMIC_FLAG_ACQUIRE |
                           IREE_HAL_ATOMIC_FLAG_RELEASE | cross_queue_flags});
    }
  }
  return cases;
}

static const char* AtomicOperationName(AtomicOperation operation) {
  switch (operation) {
    case AtomicOperation::kStore:
      return "Store";
    case AtomicOperation::kAdd:
      return "Add";
    case AtomicOperation::kSubtract:
      return "Subtract";
    case AtomicOperation::kAnd:
      return "And";
    case AtomicOperation::kOr:
      return "Or";
    case AtomicOperation::kXor:
      return "Xor";
    case AtomicOperation::kWaitEqual:
      return "WaitEqual";
    case AtomicOperation::kWaitNotEqual:
      return "WaitNotEqual";
    case AtomicOperation::kWaitGreaterEqual:
      return "WaitGreaterEqual";
    case AtomicOperation::kWaitBeforeStore:
      return "WaitBeforeStore";
  }
  IREE_ASSERT_UNREACHABLE("unknown atomic test operation");
  return "";
}

static std::string AtomicCaseName(const AtomicCase& test_case) {
  std::string name = test_case.target_kind == AtomicTargetKind::kFineHost
                         ? "FineHost"
                         : "CoarseDevice";
  name += test_case.width == IREE_HAL_ATOMIC_WIDTH_32 ? "32" : "64";
  name += AtomicOperationName(test_case.operation);
  name +=
      test_case.flags & IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE ? "System" : "Agent";
  if (test_case.flags & IREE_HAL_ATOMIC_FLAG_ACQUIRE) {
    name += "Acquire";
  }
  if (test_case.flags & IREE_HAL_ATOMIC_FLAG_RELEASE) {
    name += "Release";
  }
  if (!(test_case.flags &
        (IREE_HAL_ATOMIC_FLAG_ACQUIRE | IREE_HAL_ATOMIC_FLAG_RELEASE))) {
    name += "Relaxed";
  }
  return name;
}

// Native owners are shared by the suite. Individual cases only reseed targets
// and reuse kernargs and queue slots after observing dispatch completion.
struct LiveQueue {
  // Callback state retained until queue destruction has joined the callback.
  QueueError error;
  // Owned HSA compute queue.
  hsa_queue_t* handle = nullptr;
  // Borrowed publication view of the compute queue.
  iree_hal_amdgpu_aql_ring_t ring = {};
  // Owned signal reused after each completed dispatch on this queue.
  iree_hsa_signal_t completion_signal = {};
};

struct AtomicLiveState {
  // Allocator for the loaded library and agent metadata.
  iree_allocator_t host_allocator = iree_allocator_system();
  // ROCr owner, retained until all native resources are released.
  iree_hal_amdgpu_libhsa_t libhsa = {};
  // CPU and GPU topology owned by this suite.
  iree_hal_amdgpu_topology_t topology = {};
  // Owned target metadata for the library's GPU agents.
  iree_hal_amdgpu_agent_target_t
      gpu_agent_targets[IREE_HAL_AMDGPU_MAX_GPU_AGENT] = {};
  // Loaded builtin executable, retained while queues can execute its kernels.
  iree_hal_amdgpu_device_library_t library = {};
  // Borrowed builtin metadata for the selected GPU agent.
  iree_hal_amdgpu_device_kernels_t kernels = {};
  // Owned fine-host target, coarse-target staging word and kernarg storage.
  LiveMemory* memory = nullptr;
  // Owned coarse-device target.
  uint64_t* device_target = nullptr;
  // Consumer and producer queues reused by all matrix cases.
  LiveQueue queues[2];
  // Owned completion signal for coarse-target staging copies.
  iree_hsa_signal_t copy_signal = {};
  // Cleared after a native operation fails, preventing subsequent submission.
  bool can_submit = false;
};

class AtomicLiveTest : public ::testing::TestWithParam<AtomicCase> {
 protected:
  static void SetUpTestSuite() {
    state_ = new AtomicLiveState();
    auto& state = *state_;
    iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
        IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
        state.host_allocator, &state.libhsa);
    if (iree_status_is_unavailable(status)) {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      GTEST_SKIP() << "HSA not available, skipping tests";
    }
    IREE_ASSERT_OK(status);
    IREE_ASSERT_OK(iree_hal_amdgpu_topology_initialize_with_defaults(
        &state.libhsa, &state.topology));
    if (state.topology.gpu_agent_count == 0 ||
        state.topology.cpu_agent_count == 0) {
      GTEST_SKIP() << "CPU and GPU agents are required, skipping tests";
    }
    for (iree_host_size_t i = 0; i < state.topology.gpu_agent_count; ++i) {
      IREE_ASSERT_OK(iree_hal_amdgpu_agent_target_query(
          &state.libhsa, state.topology.gpu_agents[i], state.host_allocator,
          &state.gpu_agent_targets[i]));
    }

    const hsa_agent_t cpu_agent = state.topology.cpu_agents[0];
    const hsa_agent_t gpu_agent = state.topology.gpu_agents[0];

    IREE_ASSERT_OK(iree_hal_amdgpu_device_library_initialize(
        &state.libhsa, &state.topology, state.gpu_agent_targets,
        state.host_allocator, &state.library));
    IREE_ASSERT_OK(iree_hal_amdgpu_device_library_populate_agent_kernels(
        &state.library, gpu_agent, &state.kernels));

    hsa_amd_memory_pool_t host_memory_pool = {};
    IREE_ASSERT_OK(iree_hal_amdgpu_find_fine_global_memory_pool(
        &state.libhsa, cpu_agent, &host_memory_pool));
    IREE_ASSERT_OK(iree_hsa_amd_memory_pool_allocate(
        IREE_LIBHSA(&state.libhsa), host_memory_pool, sizeof(*state.memory),
        HSA_AMD_MEMORY_POOL_STANDARD_FLAG,
        reinterpret_cast<void**>(&state.memory)));
    IREE_ASSERT_OK(
        iree_hsa_amd_agents_allow_access(IREE_LIBHSA(&state.libhsa),
                                         /*num_agents=*/1, &gpu_agent,
                                         /*flags=*/nullptr, state.memory));
    std::memset(state.memory, 0, sizeof(*state.memory));

    hsa_amd_memory_pool_t device_memory_pool = {};
    IREE_ASSERT_OK(iree_hal_amdgpu_find_coarse_global_memory_pool(
        &state.libhsa, gpu_agent, &device_memory_pool));
    IREE_ASSERT_OK(iree_hsa_amd_memory_pool_allocate(
        IREE_LIBHSA(&state.libhsa), device_memory_pool,
        sizeof(*state.device_target), HSA_AMD_MEMORY_POOL_STANDARD_FLAG,
        reinterpret_cast<void**>(&state.device_target)));

    iree_hal_amdgpu_aql_queue_execution_mode_t aql_queue_execution_mode;
    IREE_ASSERT_OK(iree_hal_amdgpu_query_aql_queue_execution_mode(
        &state.libhsa, gpu_agent, &aql_queue_execution_mode));
    IREE_ASSERT_OK(iree_hsa_amd_signal_create(
        IREE_LIBHSA(&state.libhsa), /*initial_value=*/1, /*num_consumers=*/0,
        /*consumers=*/nullptr, /*attributes=*/0, &state.copy_signal));
    for (uint32_t i = 0; i < 2; ++i) {
      state.queues[i].error.libhsa = &state.libhsa;
      IREE_ASSERT_OK(iree_hsa_amd_signal_create(
          IREE_LIBHSA(&state.libhsa), /*initial_value=*/1, /*num_consumers=*/0,
          /*consumers=*/nullptr, /*attributes=*/0,
          &state.queues[i].error.signal));
      IREE_ASSERT_OK(iree_hsa_amd_signal_create(
          IREE_LIBHSA(&state.libhsa), /*initial_value=*/1, /*num_consumers=*/0,
          /*consumers=*/nullptr, /*attributes=*/0,
          &state.queues[i].completion_signal));
      // The callback executes on an uninstrumented ROCr thread. Publish the
      // callback state explicitly so TSAN can model the external handoff.
      IREE_TSAN_RELEASE(&state.queues[i].error);
      IREE_ASSERT_OK(iree_hsa_queue_create(
          IREE_LIBHSA(&state.libhsa), gpu_agent, /*size=*/64,
          HSA_QUEUE_TYPE_MULTI, HsaQueueErrorCallback, &state.queues[i].error,
          UINT32_MAX, UINT32_MAX, &state.queues[i].handle));
      iree_hal_amdgpu_aql_ring_initialize(
          &state.libhsa,
          reinterpret_cast<iree_amd_queue_t*>(state.queues[i].handle),
          aql_queue_execution_mode, &state.queues[i].ring);
    }

    state.can_submit = true;
  }

  static void TearDownTestSuite() {
    if (!state_) {
      return;
    }
    auto& state = *state_;
    // Join callbacks before releasing any signal or memory they can reach.
    // A failed destroy cannot authorize freeing the remaining native owners.
    for (LiveQueue& queue : state.queues) {
      if (queue.handle) {
        IREE_ASSERT_OK(
            iree_hsa_queue_destroy(IREE_LIBHSA(&state.libhsa), queue.handle));
        queue.handle = nullptr;
        IREE_TSAN_ACQUIRE(&queue.error);
        EXPECT_EQ(queue.error.callback_count.load(std::memory_order_relaxed),
                  0u);
        EXPECT_EQ(queue.error.status.load(std::memory_order_relaxed),
                  static_cast<uint32_t>(HSA_STATUS_SUCCESS));
      }
    }
    for (LiveQueue& queue : state.queues) {
      if (queue.completion_signal.handle) {
        IREE_EXPECT_OK(iree_hsa_signal_destroy(IREE_LIBHSA(&state.libhsa),
                                               queue.completion_signal));
      }
      if (queue.error.signal.handle) {
        IREE_EXPECT_OK(iree_hsa_signal_destroy(IREE_LIBHSA(&state.libhsa),
                                               queue.error.signal));
      }
    }
    if (state.copy_signal.handle) {
      IREE_EXPECT_OK(iree_hsa_signal_destroy(IREE_LIBHSA(&state.libhsa),
                                             state.copy_signal));
    }
    if (state.device_target) {
      IREE_EXPECT_OK(iree_hsa_amd_memory_pool_free(IREE_LIBHSA(&state.libhsa),
                                                   state.device_target));
    }
    if (state.memory) {
      IREE_EXPECT_OK(iree_hsa_amd_memory_pool_free(IREE_LIBHSA(&state.libhsa),
                                                   state.memory));
    }
    iree_hal_amdgpu_device_library_deinitialize(&state.library);
    for (iree_host_size_t i = 0; i < state.topology.gpu_agent_count; ++i) {
      iree_hal_amdgpu_agent_target_deinitialize(&state.gpu_agent_targets[i]);
    }
    iree_hal_amdgpu_topology_deinitialize(&state.topology);
    iree_hal_amdgpu_libhsa_deinitialize(&state.libhsa);
    delete state_;
    state_ = nullptr;
  }

  void SetUp() override {
    if (!state_->can_submit) {
      GTEST_SKIP()
          << "shared native state is unusable after an earlier failure";
    }
  }

  static bool WriteTarget(const AtomicTarget& target, uint64_t value) {
    auto& state = *state_;
    state.memory->target = value;
    if (target.kind == AtomicTargetKind::kCoarseDevice) {
      return CopyMemoryAndWait(&state.libhsa, target.ptr, target.owner_agent,
                               &state.memory->target,
                               state.topology.cpu_agents[0],
                               sizeof(state.memory->target), state.copy_signal);
    }
    return true;
  }

  static bool ReadTarget(const AtomicTarget& target, uint64_t* out_value) {
    auto& state = *state_;
    if (target.kind == AtomicTargetKind::kCoarseDevice) {
      if (!CopyMemoryAndWait(&state.libhsa, &state.memory->target,
                             state.topology.cpu_agents[0], target.ptr,
                             target.owner_agent, sizeof(state.memory->target),
                             state.copy_signal)) {
        return false;
      }
    }
    *out_value = state.memory->target;
    return true;
  }

  static bool ReleaseWaitTarget(const AtomicTarget& target,
                                iree_hal_atomic_width_t width) {
    if (target.kind == AtomicTargetKind::kCoarseDevice) {
      return WriteTarget(target, 1);
    }
    if (width == IREE_HAL_ATOMIC_WIDTH_32) {
      iree_atomic_store(reinterpret_cast<iree_atomic_int32_t*>(target.ptr), 1,
                        iree_memory_order_release);
    } else {
      iree_atomic_store(reinterpret_cast<iree_atomic_int64_t*>(target.ptr), 1,
                        iree_memory_order_release);
    }
    return true;
  }

  static bool RunStore(const AtomicTarget& target,
                       iree_hal_atomic_width_t width,
                       iree_hal_atomic_flags_t flags) {
    auto& state = *state_;
    const uint64_t width_mask =
        width == IREE_HAL_ATOMIC_WIDTH_32 ? UINT32_MAX : UINT64_MAX;
    const uint64_t value = 0x89ABCDEF01234567ull & width_mask;
    if (!WriteTarget(target, 0)) {
      return false;
    }
    const iree_hal_atomic_store_params_t params = {
        .value = value,
        .flags = flags,
        .width = width,
    };
    if (!SubmitAtomicDispatchAndWait(
            &state.libhsa, &state.queues[0].ring,
            state.queues[0].completion_signal, &state.queues[0].error,
            [&](iree_hsa_kernel_dispatch_packet_t* packet,
                uint16_t* out_setup) {
              iree_hal_amdgpu_device_atomic_store_emplace(
                  &state.kernels, packet, target.ptr, params,
                  state.memory->kernargs[0], out_setup);
            })) {
      return false;
    }
    uint64_t actual_value = 0;
    if (!ReadTarget(target, &actual_value)) {
      return false;
    }
    EXPECT_EQ(actual_value & width_mask, value);
    return true;
  }

  static bool RunRmw(const AtomicTarget& target, iree_hal_atomic_width_t width,
                     iree_hal_atomic_flags_t flags,
                     iree_hal_atomic_rmw_operation_t operation) {
    auto& state = *state_;
    const uint64_t width_mask =
        width == IREE_HAL_ATOMIC_WIDTH_32 ? UINT32_MAX : UINT64_MAX;
    const uint64_t initial = 0x76543210FEDCBA98ull & width_mask;
    const uint64_t operand = 0x111111110F0F0F0Full & width_mask;
    if (!WriteTarget(target, initial)) {
      return false;
    }
    const iree_hal_atomic_rmw_params_t params = {
        .operand = operand,
        .flags = flags,
        .width = width,
        .operation = operation,
    };
    if (!SubmitAtomicDispatchAndWait(
            &state.libhsa, &state.queues[0].ring,
            state.queues[0].completion_signal, &state.queues[0].error,
            [&](iree_hsa_kernel_dispatch_packet_t* packet,
                uint16_t* out_setup) {
              iree_hal_amdgpu_device_atomic_rmw_emplace(
                  &state.kernels, packet, target.ptr, params,
                  state.memory->kernargs[0], out_setup);
            })) {
      return false;
    }
    uint64_t actual_value = 0;
    if (!ReadTarget(target, &actual_value)) {
      return false;
    }
    EXPECT_EQ(actual_value & width_mask,
              ApplyRmw(width, operation, initial, operand));
    return true;
  }

  static bool RunWait(const AtomicTarget& target, iree_hal_atomic_width_t width,
                      iree_hal_atomic_flags_t flags,
                      iree_hal_atomic_wait_condition_t condition) {
    auto& state = *state_;
    if (!WriteTarget(target, 0x1234u)) {
      return false;
    }
    const uint64_t value =
        condition == IREE_HAL_ATOMIC_WAIT_CONDITION_NOT_EQUAL ? 0x35u
        : condition == IREE_HAL_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL
            ? 0x30u
            : 0x34u;
    const iree_hal_atomic_wait_params_t params = {
        .value = value,
        .mask = 0xFFu,
        .flags = flags,
        .width = width,
        .condition = condition,
    };
    if (!SubmitAtomicDispatchAndWait(
            &state.libhsa, &state.queues[0].ring,
            state.queues[0].completion_signal, &state.queues[0].error,
            [&](iree_hsa_kernel_dispatch_packet_t* packet,
                uint16_t* out_setup) {
              iree_hal_amdgpu_device_atomic_wait_emplace(
                  &state.kernels, packet, target.ptr, params,
                  state.memory->kernargs[0], out_setup);
            })) {
      return false;
    }
    return true;
  }

  static bool RunWaitBeforeStore(const AtomicTarget& target,
                                 iree_hal_atomic_width_t width) {
    auto& state = *state_;
    const uint64_t width_mask =
        width == IREE_HAL_ATOMIC_WIDTH_32 ? UINT32_MAX : UINT64_MAX;
    if (!WriteTarget(target, 0)) {
      return false;
    }
    const iree_hal_atomic_flags_t cross_queue_flags =
        target.kind == AtomicTargetKind::kFineHost
            ? IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE
            : IREE_HAL_ATOMIC_FLAG_NONE;
    const iree_hal_atomic_wait_params_t wait_params = {
        .value = 1,
        .mask = width_mask,
        .flags = IREE_HAL_ATOMIC_FLAG_ACQUIRE | cross_queue_flags,
        .width = width,
        .condition = IREE_HAL_ATOMIC_WAIT_CONDITION_EQUAL,
    };
    SubmitAtomicDispatch(
        &state.libhsa, &state.queues[0].ring, state.queues[0].completion_signal,
        [&](iree_hsa_kernel_dispatch_packet_t* packet, uint16_t* out_setup) {
          iree_hal_amdgpu_device_atomic_wait_emplace(
              &state.kernels, packet, target.ptr, wait_params,
              state.memory->kernargs[0], out_setup);
        });
    const iree_hal_atomic_store_params_t store_params = {
        .value = 1,
        .flags = IREE_HAL_ATOMIC_FLAG_RELEASE | cross_queue_flags,
        .width = width,
    };
    SubmitAtomicDispatch(
        &state.libhsa, &state.queues[1].ring, state.queues[1].completion_signal,
        [&](iree_hsa_kernel_dispatch_packet_t* packet, uint16_t* out_setup) {
          iree_hal_amdgpu_device_atomic_store_emplace(
              &state.kernels, packet, target.ptr, store_params,
              state.memory->kernargs[1], out_setup);
        });
    const bool producer_completed =
        WaitForQueueCompletion(&state.libhsa, state.queues[1].completion_signal,
                               &state.queues[1].error);
    if (!producer_completed && !ReleaseWaitTarget(target, width)) {
      return false;
    }
    const bool consumer_completed =
        WaitForQueueCompletion(&state.libhsa, state.queues[0].completion_signal,
                               &state.queues[0].error);
    if (!producer_completed || !consumer_completed) {
      return false;
    }
    uint64_t actual_value = 0;
    if (!ReadTarget(target, &actual_value)) {
      return false;
    }
    EXPECT_EQ(actual_value & width_mask, 1u);
    return true;
  }

  static bool RunCase(const AtomicCase& test_case) {
    auto& state = *state_;
    const AtomicTarget target =
        test_case.target_kind == AtomicTargetKind::kFineHost
            ? AtomicTarget{AtomicTargetKind::kFineHost, &state.memory->target,
                           state.topology.cpu_agents[0]}
            : AtomicTarget{AtomicTargetKind::kCoarseDevice, state.device_target,
                           state.topology.gpu_agents[0]};
    const auto width = test_case.width;
    const auto flags = test_case.flags;
    switch (test_case.operation) {
      case AtomicOperation::kStore:
        return RunStore(target, width, flags);
      case AtomicOperation::kAdd:
        return RunRmw(target, width, flags, IREE_HAL_ATOMIC_RMW_OPERATION_ADD);
      case AtomicOperation::kSubtract:
        return RunRmw(target, width, flags,
                      IREE_HAL_ATOMIC_RMW_OPERATION_SUBTRACT);
      case AtomicOperation::kAnd:
        return RunRmw(target, width, flags, IREE_HAL_ATOMIC_RMW_OPERATION_AND);
      case AtomicOperation::kOr:
        return RunRmw(target, width, flags, IREE_HAL_ATOMIC_RMW_OPERATION_OR);
      case AtomicOperation::kXor:
        return RunRmw(target, width, flags, IREE_HAL_ATOMIC_RMW_OPERATION_XOR);
      case AtomicOperation::kWaitEqual:
        return RunWait(target, width, flags,
                       IREE_HAL_ATOMIC_WAIT_CONDITION_EQUAL);
      case AtomicOperation::kWaitNotEqual:
        return RunWait(target, width, flags,
                       IREE_HAL_ATOMIC_WAIT_CONDITION_NOT_EQUAL);
      case AtomicOperation::kWaitGreaterEqual:
        return RunWait(target, width, flags,
                       IREE_HAL_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL);
      case AtomicOperation::kWaitBeforeStore:
        return RunWaitBeforeStore(target, width);
    }
    IREE_ASSERT_UNREACHABLE("unknown atomic test operation");
    return false;
  }

  // The suite owns exactly one library, allocation set and pair of queues.
  // Explicit teardown releases the state only after native queues are joined.
  // Failed destruction retains callback storage through process termination;
  // a global C++ destructor must not release it independently of that join.
  static AtomicLiveState* state_;
};

AtomicLiveState* AtomicLiveTest::state_ = nullptr;

TEST_P(AtomicLiveTest, ExecutesBuiltin) {
  state_->can_submit = RunCase(GetParam());
}

INSTANTIATE_TEST_SUITE_P(KernelMatrix, AtomicLiveTest,
                         ::testing::ValuesIn(GenerateAtomicCases()),
                         [](const ::testing::TestParamInfo<AtomicCase>& info) {
                           return AtomicCaseName(info.param);
                         });

}  // namespace
}  // namespace iree::hal::amdgpu
