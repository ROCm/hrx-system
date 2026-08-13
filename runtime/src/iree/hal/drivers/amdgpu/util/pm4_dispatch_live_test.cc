// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/threading/processor.h"
#include "iree/hal/drivers/amdgpu/abi/kernel_descriptor.h"
#include "iree/hal/drivers/amdgpu/util/aql_ring.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/hal/drivers/amdgpu/util/pm4_atomic.h"
#include "iree/hal/drivers/amdgpu/util/pm4_barrier.h"
#include "iree/hal/drivers/amdgpu/util/pm4_dispatch.h"
#include "iree/hal/drivers/amdgpu/util/pm4_dispatch_test_kernels.h"
#include "iree/hal/drivers/amdgpu/util/pm4_program.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/hal/drivers/amdgpu/util/vmem.h"
#include "iree/hal/executable/amdgpu/target_id.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

constexpr char kTestCodeObjectBaseName[] = "pm4_dispatch_test_kernels";
constexpr uint32_t kAqlValueA = 0xA1100001u;
constexpr uint32_t kAqlValueB = 0xB2200002u;
constexpr uint32_t kPm4ValueA = 0xA4400004u;
constexpr uint32_t kPm4ValueB = 0xB5500005u;
constexpr uint32_t kPm4BarrierValue = 0xC6600006u;
constexpr uint32_t kPm4BarrierAdd = 0x330u;
constexpr uint32_t kPm4PatchValue = 0xD7700007u;
constexpr uint32_t kPm4PatchWrongValue = 0xE8800008u;
constexpr uint32_t kPm4LdsValue = 0x40u;
constexpr uint16_t kWorkgroupSize[3] = {64, 1, 1};
constexpr uint32_t kDispatchThreadCount[3] = {64, 1, 1};

struct StoreKernargs {
  uint32_t* target;
  uint32_t value;
};

struct ReadAddKernargs {
  uint32_t* source;
  uint32_t* target;
  uint32_t value;
};

struct PatchUserDataKernargs {
  uint32_t* target_dwords;
  uint32_t dword_offset;
  uint64_t kernarg_address;
};

struct alignas(64) LiveMemory {
  uint32_t outputs[5];
  uint32_t scratch[2];
  uint32_t completion;
  uint32_t reserved;
  StoreKernargs store_kernargs[5];
  ReadAddKernargs read_add_kernargs;
  PatchUserDataKernargs patch_user_data_kernargs;
};

struct KernelInfo {
  hsa_executable_symbol_t symbol = {0};
  uint64_t kernel_object = 0;
  uint32_t kernarg_size = 0;
  uint32_t kernarg_alignment = 0;
  uint32_t private_segment_size = 0;
  uint32_t group_segment_size = 0;
};

struct QueueError {
  std::atomic<uint32_t> callback_count{0};
  std::atomic<uint32_t> status{HSA_STATUS_SUCCESS};
};

static std::string StringViewToString(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

static std::string TargetLabelFragment(std::string target) {
  std::replace(target.begin(), target.end(), '-', '_');
  std::replace(target.begin(), target.end(), '.', '_');
  return target;
}

static std::string TestCodeObjectFileName(
    const std::string& code_object_target) {
  return std::string(kTestCodeObjectBaseName) + "_" +
         TargetLabelFragment(code_object_target) + ".so";
}

static iree_const_byte_span_t FindTestCodeObjectData(
    const std::string& code_object_target) {
  const std::string file_name = TestCodeObjectFileName(code_object_target);
  const iree_file_toc_t* toc =
      iree_hal_amdgpu_pm4_dispatch_test_kernels_create();
  for (iree_host_size_t i = 0;
       i < iree_hal_amdgpu_pm4_dispatch_test_kernels_size(); ++i) {
    if (iree_string_view_equal(iree_make_cstring_view(toc[i].name),
                               iree_make_cstring_view(file_name.c_str()))) {
      return iree_make_const_byte_span(toc[i].data, toc[i].size);
    }
  }
  return iree_const_byte_span_empty();
}

static void HsaQueueErrorCallback(hsa_status_t status, hsa_queue_t* queue,
                                  void* user_data) {
  (void)queue;
  QueueError* error = reinterpret_cast<QueueError*>(user_data);
  error->status.store(static_cast<uint32_t>(status), std::memory_order_relaxed);
  error->callback_count.fetch_add(1, std::memory_order_relaxed);
}

struct IsaQuery {
  const iree_hal_amdgpu_libhsa_t* libhsa = nullptr;
  bool found = false;
  iree_hal_amdgpu_gfxip_version_t gfxip_version = {};
  std::string exact_target;
  std::string code_object_target;
};

static hsa_status_t FindAgentCodeObjectTarget(hsa_isa_t isa, void* user_data) {
  IsaQuery* query = reinterpret_cast<IsaQuery*>(user_data);
  uint32_t name_length = 0;
  if (!iree_status_is_ok(
          iree_hsa_isa_get_info_alt(IREE_LIBHSA(query->libhsa), isa,
                                    HSA_ISA_INFO_NAME_LENGTH, &name_length))) {
    return HSA_STATUS_ERROR;
  }
  std::vector<char> name(name_length + 1);
  if (!iree_status_is_ok(iree_hsa_isa_get_info_alt(
          IREE_LIBHSA(query->libhsa), isa, HSA_ISA_INFO_NAME, name.data()))) {
    return HSA_STATUS_ERROR;
  }

  iree_hal_amdgpu_target_identity_t exact_target_id;
  iree_status_t status = iree_hal_amdgpu_target_identity_parse_hsa_isa_name(
      iree_make_cstring_view(name.data()), &exact_target_id);
  if (!iree_status_is_ok(status)) {
    iree_status_free(status);
    return HSA_STATUS_SUCCESS;
  }

  iree_hal_amdgpu_target_identity_t code_object_target_id;
  status = iree_hal_amdgpu_target_identity_project_code_object(
      &exact_target_id, &code_object_target_id);
  if (!iree_status_is_ok(status)) {
    iree_status_free(status);
    return HSA_STATUS_ERROR;
  }
  query->exact_target = StringViewToString(exact_target_id.processor);
  query->code_object_target =
      StringViewToString(code_object_target_id.processor);
  query->gfxip_version = exact_target_id.version;
  query->found = true;
  return HSA_STATUS_INFO_BREAK;
}

static bool QueryAgentCodeObjectTarget(
    const iree_hal_amdgpu_libhsa_t* libhsa, hsa_agent_t agent,
    iree_hal_amdgpu_gfxip_version_t* out_gfxip_version,
    std::string* out_exact_target, std::string* out_code_object_target) {
  IsaQuery query = {/*.libhsa=*/libhsa};
  iree_status_t status = iree_hsa_agent_iterate_isas(
      IREE_LIBHSA(libhsa), agent, FindAgentCodeObjectTarget, &query);
  if (!iree_status_is_ok(status)) {
    iree_status_free(status);
  }
  if (!query.found) return false;
  *out_gfxip_version = query.gfxip_version;
  *out_exact_target = query.exact_target;
  *out_code_object_target = query.code_object_target;
  return true;
}

static iree_hal_amdgpu_vendor_packet_capability_flags_t
BarrierCapabilitiesForGfxIp(iree_hal_amdgpu_gfxip_version_t gfxip_version) {
  iree_hal_amdgpu_vendor_packet_capability_flags_t capabilities =
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_EVENT_WRITE |
      IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_ACQUIRE_MEM;
  capabilities |=
      gfxip_version.major == 9
          ? IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_ACQUIRE_MEM_GFX9
          : IREE_HAL_AMDGPU_VENDOR_PACKET_CAPABILITY_PM4_ACQUIRE_MEM_GFX10;
  return capabilities;
}

static iree_status_t LookupKernel(const iree_hal_amdgpu_libhsa_t* libhsa,
                                  hsa_executable_t executable,
                                  hsa_agent_t agent, const char* kernel_name,
                                  KernelInfo* out_info) {
  *out_info = KernelInfo{};

  char descriptor_symbol_name[128] = {0};
  std::snprintf(descriptor_symbol_name, sizeof(descriptor_symbol_name), "%s.kd",
                kernel_name);
  hsa_status_t raw_status = iree_hsa_executable_get_symbol_by_name_raw(
      libhsa, executable, descriptor_symbol_name, &agent, &out_info->symbol);
  if (raw_status != HSA_STATUS_SUCCESS) {
    IREE_RETURN_IF_ERROR(iree_hsa_executable_get_symbol_by_name(
        IREE_LIBHSA(libhsa), executable, kernel_name, &agent,
        &out_info->symbol));
  }
  IREE_RETURN_IF_ERROR(iree_hsa_executable_symbol_get_info(
      IREE_LIBHSA(libhsa), out_info->symbol,
      HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &out_info->kernel_object));
  IREE_RETURN_IF_ERROR(iree_hsa_executable_symbol_get_info(
      IREE_LIBHSA(libhsa), out_info->symbol,
      HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
      &out_info->kernarg_size));
  IREE_RETURN_IF_ERROR(iree_hsa_executable_symbol_get_info(
      IREE_LIBHSA(libhsa), out_info->symbol,
      HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT,
      &out_info->kernarg_alignment));
  IREE_RETURN_IF_ERROR(iree_hsa_executable_symbol_get_info(
      IREE_LIBHSA(libhsa), out_info->symbol,
      HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
      &out_info->private_segment_size));
  return iree_hsa_executable_symbol_get_info(
      IREE_LIBHSA(libhsa), out_info->symbol,
      HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
      &out_info->group_segment_size);
}

static void EmitDispatchDirect(uint32_t* target_dwords, uint32_t thread_count_x,
                               uint32_t thread_count_y, uint32_t thread_count_z,
                               uint32_t dispatch_initiator) {
  target_dwords[0] = iree_hal_amdgpu_pm4_make_compute_header(
      IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_DISPATCH_DIRECT,
      IREE_HAL_AMDGPU_PM4_DISPATCH_DIRECT_DWORD_COUNT);
  target_dwords[1] = thread_count_x;
  target_dwords[2] = thread_count_y;
  target_dwords[3] = thread_count_z;
  target_dwords[4] = dispatch_initiator;
}

static iree_status_t AppendPm4Barrier(
    iree_hal_amdgpu_vendor_packet_capability_flags_t capabilities,
    iree_hal_amdgpu_pm4_barrier_flags_t barrier_flags,
    iree_hsa_fence_scope_t acquire_scope, iree_hsa_fence_scope_t release_scope,
    uint32_t* dwords, uint32_t capacity, uint32_t* inout_dword_count) {
  if (*inout_dword_count > capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "PM4 barrier append cursor %u exceeds capacity %u",
                            *inout_dword_count, capacity);
  }
  const uint32_t barrier_dword_count = iree_hal_amdgpu_pm4_barrier_dword_count(
      capabilities, barrier_flags, acquire_scope, release_scope);
  if (barrier_dword_count == 0) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "PM4 barrier cannot be emitted for capabilities "
                            "0x%08" PRIx32,
                            capabilities);
  }
  if (capacity - *inout_dword_count < barrier_dword_count) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "PM4 barrier requires %u dwords but only %u are available",
        barrier_dword_count, capacity - *inout_dword_count);
  }
  uint32_t emitted_dword_count = 0;
  if (!iree_hal_amdgpu_pm4_barrier_emit(
          capabilities, barrier_flags, acquire_scope, release_scope,
          capacity - *inout_dword_count, &dwords[*inout_dword_count],
          &emitted_dword_count) ||
      emitted_dword_count != barrier_dword_count) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "PM4 barrier emission changed size");
  }
  *inout_dword_count += emitted_dword_count;
  return iree_ok_status();
}

static iree_status_t AppendPm4HostAcquire(
    iree_hal_amdgpu_vendor_packet_capability_flags_t capabilities,
    uint32_t* dwords, uint32_t capacity, uint32_t* inout_dword_count) {
  return AppendPm4Barrier(
      capabilities, IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_EXECUTION,
      IREE_HSA_FENCE_SCOPE_SYSTEM, IREE_HSA_FENCE_SCOPE_NONE, dwords, capacity,
      inout_dword_count);
}

static iree_status_t AppendPm4WriteData32(uint32_t* dwords, uint32_t capacity,
                                          uint32_t* inout_dword_count,
                                          void* target, uint32_t value) {
  if (*inout_dword_count > capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "PM4 write-data append cursor %u exceeds "
                            "capacity %u",
                            *inout_dword_count, capacity);
  }
  if (!iree_host_ptr_has_alignment(target, 4)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "PM4 write-data target is not 4-byte aligned");
  }
  constexpr uint32_t kWriteData32DwordCount = 5;
  if (capacity - *inout_dword_count < kWriteData32DwordCount) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "PM4 write-data requires %u dwords but only %u are available",
        kWriteData32DwordCount, capacity - *inout_dword_count);
  }
  const uintptr_t address = reinterpret_cast<uintptr_t>(target);
  uint32_t* target_dwords = &dwords[*inout_dword_count];
  target_dwords[0] = iree_hal_amdgpu_pm4_make_header(
      IREE_HAL_AMDGPU_PM4_HDR_IT_OPCODE_WRITE_DATA, kWriteData32DwordCount);
  target_dwords[1] =
      IREE_HAL_AMDGPU_PM4_WRITE_DATA_DST_SEL_TC_L2 |
      IREE_HAL_AMDGPU_PM4_WRITE_DATA_WR_CONFIRM_WAIT_CONFIRMATION;
  target_dwords[2] = iree_hal_amdgpu_pm4_addr_lo(address);
  target_dwords[3] = iree_hal_amdgpu_pm4_addr_hi(address);
  target_dwords[4] = value;
  *inout_dword_count += kWriteData32DwordCount;
  return iree_ok_status();
}

static iree_status_t AppendPm4AtomicWait(
    iree_hal_atomic_width_t width, iree_hal_atomic_wait_condition_t condition,
    void* target, uint64_t value, uint64_t mask, uint32_t* dwords,
    uint32_t capacity, uint32_t* inout_dword_count) {
  if (*inout_dword_count > capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "PM4 atomic wait cursor exceeds capacity");
  }
  uint32_t emitted_dword_count = 0;
  if (!iree_hal_amdgpu_pm4_atomic_wait_emit(
          width, condition, reinterpret_cast<uintptr_t>(target), value, mask,
          capacity - *inout_dword_count, &dwords[*inout_dword_count],
          &emitted_dword_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "PM4 atomic wait cannot be encoded");
  }
  *inout_dword_count += emitted_dword_count;
  return iree_ok_status();
}

static iree_status_t AppendPm4AtomicStore(iree_hal_atomic_width_t width,
                                          void* target, uint64_t value,
                                          uint32_t* dwords, uint32_t capacity,
                                          uint32_t* inout_dword_count) {
  if (*inout_dword_count > capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "PM4 atomic store cursor exceeds capacity");
  }
  uint32_t emitted_dword_count = 0;
  if (!iree_hal_amdgpu_pm4_atomic_store_emit(
          width, reinterpret_cast<uintptr_t>(target), value,
          capacity - *inout_dword_count, &dwords[*inout_dword_count],
          &emitted_dword_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "PM4 atomic store cannot be encoded");
  }
  *inout_dword_count += emitted_dword_count;
  return iree_ok_status();
}

static iree_status_t AppendPm4AtomicRmw(
    iree_hal_atomic_width_t width, iree_hal_atomic_rmw_operation_t operation,
    void* target, uint64_t operand, uint32_t* dwords, uint32_t capacity,
    uint32_t* inout_dword_count) {
  if (*inout_dword_count > capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "PM4 atomic RMW cursor exceeds capacity");
  }
  uint32_t emitted_dword_count = 0;
  if (!iree_hal_amdgpu_pm4_atomic_rmw_emit(
          width, operation, reinterpret_cast<uintptr_t>(target), operand,
          capacity - *inout_dword_count, &dwords[*inout_dword_count],
          &emitted_dword_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "PM4 atomic RMW cannot be encoded");
  }
  *inout_dword_count += emitted_dword_count;
  return iree_ok_status();
}

static iree_status_t AppendPm4CopyData64(uint32_t* dwords, uint32_t capacity,
                                         uint32_t* inout_dword_count,
                                         const void* source, void* target) {
  if (*inout_dword_count > capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "PM4 copy-data cursor exceeds capacity");
  }
  iree_hal_amdgpu_pm4_ib_slot_t slot;
  const uint32_t emitted_dword_count =
      iree_hal_amdgpu_pm4_emit_copy_data64(&slot, source, target);
  if (capacity - *inout_dword_count < emitted_dword_count) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "PM4 copy-data exceeds capacity");
  }
  memcpy(&dwords[*inout_dword_count], slot.dwords,
         emitted_dword_count * sizeof(*dwords));
  *inout_dword_count += emitted_dword_count;
  return iree_ok_status();
}

static iree_status_t AppendPm4CompletionWrite(
    iree_hal_amdgpu_vendor_packet_capability_flags_t capabilities,
    uint32_t* dwords, uint32_t capacity, uint32_t* inout_dword_count,
    void* target, uint32_t value) {
  IREE_RETURN_IF_ERROR(
      AppendPm4Barrier(capabilities, IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_EXECUTION,
                       IREE_HSA_FENCE_SCOPE_NONE, IREE_HSA_FENCE_SCOPE_SYSTEM,
                       dwords, capacity, inout_dword_count));
  return AppendPm4WriteData32(dwords, capacity, inout_dword_count, target,
                              value);
}

static iree_status_t AppendPm4DispatchDirect(
    const iree_hal_amdgpu_pm4_dispatch_launch_state_t& launch_state,
    uint64_t kernarg_address, uint32_t* dwords, uint32_t capacity,
    uint32_t* inout_dword_count) {
  if (*inout_dword_count > capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "PM4 dispatch append cursor %u exceeds capacity %u",
                            *inout_dword_count, capacity);
  }
  uint32_t emitted_dword_count = 0;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dispatch_emit_setup(
      &launch_state, capacity - *inout_dword_count, &dwords[*inout_dword_count],
      &emitted_dword_count));
  *inout_dword_count += emitted_dword_count;
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_pm4_dispatch_emit_user_data(
      &launch_state, kernarg_address, /*kernarg_preload_data=*/nullptr,
      capacity - *inout_dword_count, &dwords[*inout_dword_count],
      &emitted_dword_count));
  *inout_dword_count += emitted_dword_count;
  if (capacity - *inout_dword_count <
      IREE_HAL_AMDGPU_PM4_DISPATCH_DIRECT_DWORD_COUNT) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "PM4 dispatch packet requires %u dwords but only %u are available",
        IREE_HAL_AMDGPU_PM4_DISPATCH_DIRECT_DWORD_COUNT,
        capacity - *inout_dword_count);
  }
  EmitDispatchDirect(&dwords[*inout_dword_count], kDispatchThreadCount[0],
                     kDispatchThreadCount[1], kDispatchThreadCount[2],
                     launch_state.dispatch_initiator);
  *inout_dword_count += IREE_HAL_AMDGPU_PM4_DISPATCH_DIRECT_DWORD_COUNT;
  return iree_ok_status();
}

static void WaitForCompletionWord(volatile uint32_t* completion,
                                  uint32_t value) {
  while (*completion != value) {
    iree_processor_yield();
  }
}

static void WaitForMarker(iree_atomic_int32_t* marker, int32_t value) {
  while (iree_atomic_load(marker, iree_memory_order_acquire) != value) {
    iree_processor_yield();
  }
}

class PM4DispatchLiveTest : public ::testing::Test {
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

    if (!QueryAgentCodeObjectTarget(&libhsa, topology.gpu_agents[0],
                                    &agent_gfxip_version, &agent_exact_target,
                                    &agent_code_object_target)) {
      GTEST_SKIP() << "could not query AMDGPU agent ISA";
    }
    if (agent_gfxip_version.major < 9 || agent_gfxip_version.major > 12) {
      GTEST_SKIP() << "PM4 dispatch test does not support agent "
                   << agent_exact_target;
    }
    agent_pm4_barrier_capabilities =
        BarrierCapabilitiesForGfxIp(agent_gfxip_version);
    const std::string file_name =
        TestCodeObjectFileName(agent_code_object_target);
    test_code_object_data = FindTestCodeObjectData(agent_code_object_target);
    if (test_code_object_data.data_length == 0) {
      GTEST_SKIP() << "PM4 dispatch code object " << file_name << " for agent "
                   << agent_exact_target << " via " << agent_code_object_target
                   << " was not generated; configure IREE_HAL_AMDGPU_TARGETS "
                      "or //runtime/src/iree/hal/drivers/amdgpu:targets";
    }
  }

  static void TearDownTestSuite() {
    iree_hal_amdgpu_topology_deinitialize(&topology);
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa);
  }

  static iree_allocator_t host_allocator;
  static iree_hal_amdgpu_libhsa_t libhsa;
  static iree_hal_amdgpu_topology_t topology;
  static iree_hal_amdgpu_gfxip_version_t agent_gfxip_version;
  static iree_hal_amdgpu_vendor_packet_capability_flags_t
      agent_pm4_barrier_capabilities;
  static std::string agent_exact_target;
  static std::string agent_code_object_target;
  static iree_const_byte_span_t test_code_object_data;
};

iree_allocator_t PM4DispatchLiveTest::host_allocator;
iree_hal_amdgpu_libhsa_t PM4DispatchLiveTest::libhsa;
iree_hal_amdgpu_topology_t PM4DispatchLiveTest::topology;
iree_hal_amdgpu_gfxip_version_t PM4DispatchLiveTest::agent_gfxip_version;
iree_hal_amdgpu_vendor_packet_capability_flags_t
    PM4DispatchLiveTest::agent_pm4_barrier_capabilities;
std::string PM4DispatchLiveTest::agent_exact_target;
std::string PM4DispatchLiveTest::agent_code_object_target;
iree_const_byte_span_t PM4DispatchLiveTest::test_code_object_data;

TEST_F(PM4DispatchLiveTest, AqlAndAqlPm4IbLaunchMixedKernels) {
  hsa_agent_t cpu_agent = topology.cpu_agents[0];
  hsa_agent_t gpu_agent = topology.gpu_agents[0];

  hsa_code_object_reader_t code_object_reader = {0};
  IREE_ASSERT_OK(iree_hsa_code_object_reader_create_from_memory(
      IREE_LIBHSA(&libhsa), test_code_object_data.data,
      test_code_object_data.data_length, &code_object_reader));

  hsa_executable_t executable = {0};
  IREE_ASSERT_OK(
      iree_hsa_executable_create_alt(IREE_LIBHSA(&libhsa), HSA_PROFILE_FULL,
                                     HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                     /*options=*/nullptr, &executable));
  hsa_loaded_code_object_t loaded_code_object = {0};
  IREE_ASSERT_OK(iree_hsa_executable_load_agent_code_object(
      IREE_LIBHSA(&libhsa), executable, gpu_agent, code_object_reader,
      /*options=*/nullptr, &loaded_code_object));
  IREE_ASSERT_OK(iree_hsa_executable_freeze(IREE_LIBHSA(&libhsa), executable,
                                            /*options=*/nullptr));
  IREE_ASSERT_OK(iree_hsa_code_object_reader_destroy(IREE_LIBHSA(&libhsa),
                                                     code_object_reader));
  code_object_reader = {0};

  KernelInfo kernels[5];
  IREE_ASSERT_OK(LookupKernel(&libhsa, executable, gpu_agent,
                              "iree_hal_amdgpu_pm4_dispatch_test_store_a",
                              &kernels[0]));
  IREE_ASSERT_OK(LookupKernel(&libhsa, executable, gpu_agent,
                              "iree_hal_amdgpu_pm4_dispatch_test_store_b",
                              &kernels[1]));
  IREE_ASSERT_OK(LookupKernel(&libhsa, executable, gpu_agent,
                              "iree_hal_amdgpu_pm4_dispatch_test_read_add",
                              &kernels[2]));
  IREE_ASSERT_OK(LookupKernel(
      &libhsa, executable, gpu_agent,
      "iree_hal_amdgpu_pm4_dispatch_test_patch_user_data", &kernels[3]));
  IREE_ASSERT_OK(LookupKernel(&libhsa, executable, gpu_agent,
                              "iree_hal_amdgpu_pm4_dispatch_test_lds_sum",
                              &kernels[4]));
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(kernels); ++i) {
    const KernelInfo& kernel = kernels[i];
    EXPECT_LE(kernel.kernarg_size, sizeof(PatchUserDataKernargs));
    EXPECT_LE(kernel.kernarg_alignment, alignof(LiveMemory));
    EXPECT_EQ(kernel.private_segment_size, 0u);
    if (i == 4) {
      EXPECT_EQ(kernel.group_segment_size, 64u * sizeof(uint32_t));
    } else {
      EXPECT_EQ(kernel.group_segment_size, 0u);
    }
  }

  hsa_amd_memory_pool_t host_memory_pool = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_find_fine_global_memory_pool(
      &libhsa, cpu_agent, &host_memory_pool));
  LiveMemory* memory = nullptr;
  IREE_ASSERT_OK(iree_hsa_amd_memory_pool_allocate(
      IREE_LIBHSA(&libhsa), host_memory_pool, sizeof(LiveMemory),
      HSA_AMD_MEMORY_POOL_STANDARD_FLAG, reinterpret_cast<void**>(&memory)));
  IREE_ASSERT_OK(iree_hsa_amd_agents_allow_access(IREE_LIBHSA(&libhsa),
                                                  /*num_agents=*/1, &gpu_agent,
                                                  /*flags=*/nullptr, memory));

  hsa_amd_memory_pool_t pm4_memory_pool = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_find_coarse_global_memory_pool(
      &libhsa, cpu_agent, &pm4_memory_pool));

  QueueError queue_error;
  hsa_queue_t* queue = nullptr;
  IREE_ASSERT_OK(iree_hsa_queue_create(
      IREE_LIBHSA(&libhsa), gpu_agent, /*size=*/64, HSA_QUEUE_TYPE_MULTI,
      HsaQueueErrorCallback, &queue_error, UINT32_MAX, UINT32_MAX, &queue));
  iree_hal_amdgpu_aql_ring_t aql_ring;
  iree_hal_amdgpu_aql_ring_initialize(
      &libhsa, reinterpret_cast<iree_amd_queue_t*>(queue), &aql_ring);

  hsa_signal_t completion_signal = iree_hsa_signal_null();
  IREE_ASSERT_OK(iree_hsa_amd_signal_create(
      IREE_LIBHSA(&libhsa), /*initial_value=*/1, /*num_consumers=*/0,
      /*consumers=*/nullptr, /*attributes=*/0, &completion_signal));

  memset(memory, 0, sizeof(*memory));
  memory->store_kernargs[0] = {/*.target=*/&memory->outputs[0],
                               /*.value=*/kAqlValueA};
  memory->store_kernargs[1] = {/*.target=*/&memory->outputs[1],
                               /*.value=*/kAqlValueB};

  const uint64_t aql_first_packet_id =
      iree_hal_amdgpu_aql_ring_reserve(&aql_ring, /*count=*/2);
  for (uint32_t i = 0; i < 2; ++i) {
    iree_hal_amdgpu_aql_packet_t* packet =
        iree_hal_amdgpu_aql_ring_packet(&aql_ring, aql_first_packet_id + i);
    memset(packet, 0, sizeof(*packet));
    uint16_t setup = 0;
    const uint16_t header = iree_hal_amdgpu_aql_emit_dispatch(
        &packet->dispatch, kernels[i].kernel_object, &memory->store_kernargs[i],
        kWorkgroupSize, kDispatchThreadCount, kernels[i].private_segment_size,
        kernels[i].group_segment_size,
        iree_hal_amdgpu_aql_packet_control_barrier_system(),
        i == 1 ? completion_signal : iree_hsa_signal_null(), &setup);
    iree_hal_amdgpu_aql_ring_commit(packet, header, setup);
  }
  iree_hal_amdgpu_aql_ring_doorbell(&aql_ring, aql_first_packet_id + 1);
  EXPECT_EQ(
      iree_hsa_signal_wait_scacquire(
          IREE_LIBHSA(&libhsa), completion_signal, HSA_SIGNAL_CONDITION_EQ,
          /*compare_value=*/0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED),
      0);
  EXPECT_EQ(memory->outputs[0], kAqlValueA);
  EXPECT_EQ(memory->outputs[1], kAqlValueB + 0x100u);

  iree_hal_amdgpu_pm4_dispatch_launch_state_t launch_states[5];
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(launch_states); ++i) {
    const iree_hal_amdgpu_kernel_descriptor_t* descriptor =
        reinterpret_cast<const iree_hal_amdgpu_kernel_descriptor_t*>(
            static_cast<uintptr_t>(kernels[i].kernel_object));
    IREE_ASSERT_OK(iree_hal_amdgpu_pm4_dispatch_launch_state_initialize(
        agent_gfxip_version, descriptor, kernels[i].kernel_object,
        kWorkgroupSize, IREE_HAL_AMDGPU_PM4_DISPATCH_LAUNCH_FLAG_ORDER_MODE,
        &launch_states[i]));
  }

  uint32_t pm4_dwords[256] = {0};
  uint32_t pm4_dword_count = 0;
  memset(memory, 0, sizeof(*memory));
  memory->store_kernargs[0] = {/*.target=*/&memory->outputs[0],
                               /*.value=*/kPm4ValueA};
  memory->store_kernargs[1] = {/*.target=*/&memory->outputs[1],
                               /*.value=*/kPm4ValueB};

  IREE_ASSERT_OK(AppendPm4HostAcquire(agent_pm4_barrier_capabilities,
                                      pm4_dwords, IREE_ARRAYSIZE(pm4_dwords),
                                      &pm4_dword_count));
  for (uint32_t i = 0; i < 2; ++i) {
    IREE_ASSERT_OK(AppendPm4DispatchDirect(
        launch_states[i],
        reinterpret_cast<uintptr_t>(&memory->store_kernargs[i]), pm4_dwords,
        IREE_ARRAYSIZE(pm4_dwords), &pm4_dword_count));
  }
  IREE_ASSERT_OK(AppendPm4CompletionWrite(
      agent_pm4_barrier_capabilities, pm4_dwords, IREE_ARRAYSIZE(pm4_dwords),
      &pm4_dword_count, &memory->completion, /*value=*/1));

  iree_hal_amdgpu_pm4_program_t pm4_program = {};
  IREE_ASSERT_OK(iree_hal_amdgpu_pm4_program_initialize(
      &libhsa, gpu_agent, pm4_memory_pool, pm4_dwords, pm4_dword_count,
      &pm4_program));

  iree_hsa_signal_store_screlease(IREE_LIBHSA(&libhsa), completion_signal, 1);
  const uint64_t pm4_packet_id =
      iree_hal_amdgpu_aql_ring_reserve(&aql_ring, /*count=*/1);
  iree_hal_amdgpu_aql_packet_t* pm4_packet =
      iree_hal_amdgpu_aql_ring_packet(&aql_ring, pm4_packet_id);
  memset(pm4_packet, 0, sizeof(*pm4_packet));
  uint16_t pm4_setup = 0;
  const uint16_t pm4_header = iree_hal_amdgpu_aql_emit_pm4_ib_dwords(
      &pm4_packet->pm4_ib, pm4_program.dwords, pm4_program.dword_count,
      iree_hal_amdgpu_aql_packet_control_barrier_system(), completion_signal,
      &pm4_setup);
  iree_hal_amdgpu_aql_ring_commit(pm4_packet, pm4_header, pm4_setup);
  iree_hal_amdgpu_aql_ring_doorbell(&aql_ring, pm4_packet_id);
  EXPECT_EQ(
      iree_hsa_signal_wait_scacquire(
          IREE_LIBHSA(&libhsa), completion_signal, HSA_SIGNAL_CONDITION_EQ,
          /*compare_value=*/0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED),
      0);
  WaitForCompletionWord(&memory->completion, /*value=*/1);
  EXPECT_EQ(memory->outputs[0], kPm4ValueA);
  EXPECT_EQ(memory->outputs[1], kPm4ValueB + 0x100u);
  iree_hal_amdgpu_pm4_program_deinitialize(&pm4_program);

  memset(memory, 0, sizeof(*memory));
  pm4_dword_count = 0;
  memory->store_kernargs[0] = {/*.target=*/&memory->scratch[0],
                               /*.value=*/kPm4BarrierValue};
  memory->read_add_kernargs = {/*.source=*/&memory->scratch[0],
                               /*.target=*/&memory->outputs[2],
                               /*.value=*/kPm4BarrierAdd};
  IREE_ASSERT_OK(AppendPm4HostAcquire(agent_pm4_barrier_capabilities,
                                      pm4_dwords, IREE_ARRAYSIZE(pm4_dwords),
                                      &pm4_dword_count));
  IREE_ASSERT_OK(AppendPm4DispatchDirect(
      launch_states[0], reinterpret_cast<uintptr_t>(&memory->store_kernargs[0]),
      pm4_dwords, IREE_ARRAYSIZE(pm4_dwords), &pm4_dword_count));
  IREE_ASSERT_OK(AppendPm4Barrier(
      agent_pm4_barrier_capabilities,
      IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_EXECUTION, IREE_HSA_FENCE_SCOPE_SYSTEM,
      IREE_HSA_FENCE_SCOPE_SYSTEM, pm4_dwords, IREE_ARRAYSIZE(pm4_dwords),
      &pm4_dword_count));
  IREE_ASSERT_OK(AppendPm4DispatchDirect(
      launch_states[2], reinterpret_cast<uintptr_t>(&memory->read_add_kernargs),
      pm4_dwords, IREE_ARRAYSIZE(pm4_dwords), &pm4_dword_count));
  IREE_ASSERT_OK(AppendPm4CompletionWrite(
      agent_pm4_barrier_capabilities, pm4_dwords, IREE_ARRAYSIZE(pm4_dwords),
      &pm4_dword_count, &memory->completion, /*value=*/2));
  IREE_ASSERT_OK(iree_hal_amdgpu_pm4_program_initialize(
      &libhsa, gpu_agent, pm4_memory_pool, pm4_dwords, pm4_dword_count,
      &pm4_program));

  iree_hsa_signal_store_screlease(IREE_LIBHSA(&libhsa), completion_signal, 1);
  memory->completion = 0;
  const uint64_t barrier_pm4_packet_id =
      iree_hal_amdgpu_aql_ring_reserve(&aql_ring, /*count=*/1);
  pm4_packet =
      iree_hal_amdgpu_aql_ring_packet(&aql_ring, barrier_pm4_packet_id);
  memset(pm4_packet, 0, sizeof(*pm4_packet));
  pm4_setup = 0;
  const uint16_t barrier_pm4_header = iree_hal_amdgpu_aql_emit_pm4_ib_dwords(
      &pm4_packet->pm4_ib, pm4_program.dwords, pm4_program.dword_count,
      iree_hal_amdgpu_aql_packet_control_barrier_system(), completion_signal,
      &pm4_setup);
  iree_hal_amdgpu_aql_ring_commit(pm4_packet, barrier_pm4_header, pm4_setup);
  iree_hal_amdgpu_aql_ring_doorbell(&aql_ring, barrier_pm4_packet_id);
  EXPECT_EQ(
      iree_hsa_signal_wait_scacquire(
          IREE_LIBHSA(&libhsa), completion_signal, HSA_SIGNAL_CONDITION_EQ,
          /*compare_value=*/0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED),
      0);
  WaitForCompletionWord(&memory->completion, /*value=*/2);
  EXPECT_EQ(memory->scratch[0], kPm4BarrierValue);
  EXPECT_EQ(memory->outputs[2], kPm4BarrierValue + kPm4BarrierAdd);
  iree_hal_amdgpu_pm4_program_deinitialize(&pm4_program);

  memset(memory, 0, sizeof(*memory));
  memory->store_kernargs[2] = {/*.target=*/&memory->outputs[3],
                               /*.value=*/kPm4PatchWrongValue};
  memory->store_kernargs[3] = {/*.target=*/&memory->outputs[2],
                               /*.value=*/kPm4PatchValue};
  pm4_dword_count = 0;
  IREE_ASSERT_OK(AppendPm4HostAcquire(agent_pm4_barrier_capabilities,
                                      pm4_dwords, IREE_ARRAYSIZE(pm4_dwords),
                                      &pm4_dword_count));
  uint32_t barrier_dword_count = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_pm4_dispatch_emit_setup(
      &launch_states[1], IREE_ARRAYSIZE(pm4_dwords) - pm4_dword_count,
      &pm4_dwords[pm4_dword_count], &barrier_dword_count));
  pm4_dword_count += barrier_dword_count;
  const uint32_t patched_user_data_offset = pm4_dword_count + 2;
  IREE_ASSERT_OK(iree_hal_amdgpu_pm4_dispatch_emit_user_data(
      &launch_states[1],
      reinterpret_cast<uintptr_t>(&memory->store_kernargs[2]),
      /*kernarg_preload_data=*/nullptr,
      IREE_ARRAYSIZE(pm4_dwords) - pm4_dword_count,
      &pm4_dwords[pm4_dword_count], &barrier_dword_count));
  pm4_dword_count += barrier_dword_count;
  ASSERT_LE(pm4_dword_count + IREE_HAL_AMDGPU_PM4_DISPATCH_DIRECT_DWORD_COUNT,
            IREE_ARRAYSIZE(pm4_dwords));
  EmitDispatchDirect(&pm4_dwords[pm4_dword_count], kDispatchThreadCount[0],
                     kDispatchThreadCount[1], kDispatchThreadCount[2],
                     launch_states[1].dispatch_initiator);
  pm4_dword_count += IREE_HAL_AMDGPU_PM4_DISPATCH_DIRECT_DWORD_COUNT;
  IREE_ASSERT_OK(AppendPm4CompletionWrite(
      agent_pm4_barrier_capabilities, pm4_dwords, IREE_ARRAYSIZE(pm4_dwords),
      &pm4_dword_count, &memory->completion, /*value=*/3));
  iree_hal_amdgpu_pm4_program_t target_pm4_program = {};
  IREE_ASSERT_OK(iree_hal_amdgpu_pm4_program_initialize(
      &libhsa, gpu_agent, pm4_memory_pool, pm4_dwords, pm4_dword_count,
      &target_pm4_program));

  memory->patch_user_data_kernargs = {
      /*.target_dwords=*/target_pm4_program.dwords,
      /*.dword_offset=*/patched_user_data_offset,
      /*.kernarg_address=*/
      reinterpret_cast<uintptr_t>(&memory->store_kernargs[3]),
  };

  // The target userdata lives in a following IB. Later dwords in the current IB
  // may already be fetched by the command processor before a shader fixup runs.
  pm4_dword_count = 0;
  IREE_ASSERT_OK(AppendPm4HostAcquire(agent_pm4_barrier_capabilities,
                                      pm4_dwords, IREE_ARRAYSIZE(pm4_dwords),
                                      &pm4_dword_count));
  IREE_ASSERT_OK(AppendPm4DispatchDirect(
      launch_states[3],
      reinterpret_cast<uintptr_t>(&memory->patch_user_data_kernargs),
      pm4_dwords, IREE_ARRAYSIZE(pm4_dwords), &pm4_dword_count));
  barrier_dword_count = 0;
  ASSERT_TRUE(iree_hal_amdgpu_pm4_barrier_emit(
      agent_pm4_barrier_capabilities,
      IREE_HAL_AMDGPU_PM4_BARRIER_FLAG_FIXUP_TO_IB, IREE_HSA_FENCE_SCOPE_NONE,
      IREE_HSA_FENCE_SCOPE_NONE, IREE_ARRAYSIZE(pm4_dwords) - pm4_dword_count,
      &pm4_dwords[pm4_dword_count], &barrier_dword_count));
  pm4_dword_count += barrier_dword_count;
  IREE_ASSERT_OK(iree_hal_amdgpu_pm4_program_initialize(
      &libhsa, gpu_agent, pm4_memory_pool, pm4_dwords, pm4_dword_count,
      &pm4_program));

  iree_hsa_signal_store_screlease(IREE_LIBHSA(&libhsa), completion_signal, 1);
  memory->completion = 0;
  const uint64_t fixup_pm4_packet_id =
      iree_hal_amdgpu_aql_ring_reserve(&aql_ring, /*count=*/2);
  pm4_packet = iree_hal_amdgpu_aql_ring_packet(&aql_ring, fixup_pm4_packet_id);
  memset(pm4_packet, 0, sizeof(*pm4_packet));
  pm4_setup = 0;
  const uint16_t fixup_pm4_header = iree_hal_amdgpu_aql_emit_pm4_ib_dwords(
      &pm4_packet->pm4_ib, pm4_program.dwords, pm4_program.dword_count,
      iree_hal_amdgpu_aql_packet_control_barrier(IREE_HSA_FENCE_SCOPE_SYSTEM,
                                                 IREE_HSA_FENCE_SCOPE_NONE),
      iree_hsa_signal_null(), &pm4_setup);
  iree_hal_amdgpu_aql_ring_commit(pm4_packet, fixup_pm4_header, pm4_setup);
  pm4_packet =
      iree_hal_amdgpu_aql_ring_packet(&aql_ring, fixup_pm4_packet_id + 1);
  memset(pm4_packet, 0, sizeof(*pm4_packet));
  pm4_setup = 0;
  const uint16_t target_pm4_header = iree_hal_amdgpu_aql_emit_pm4_ib_dwords(
      &pm4_packet->pm4_ib, target_pm4_program.dwords,
      target_pm4_program.dword_count,
      iree_hal_amdgpu_aql_packet_control_barrier(IREE_HSA_FENCE_SCOPE_NONE,
                                                 IREE_HSA_FENCE_SCOPE_NONE),
      completion_signal, &pm4_setup);
  iree_hal_amdgpu_aql_ring_commit(pm4_packet, target_pm4_header, pm4_setup);
  iree_hal_amdgpu_aql_ring_doorbell(&aql_ring, fixup_pm4_packet_id + 1);
  EXPECT_EQ(
      iree_hsa_signal_wait_scacquire(
          IREE_LIBHSA(&libhsa), completion_signal, HSA_SIGNAL_CONDITION_EQ,
          /*compare_value=*/0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED),
      0);
  WaitForCompletionWord(&memory->completion, /*value=*/3);
  EXPECT_EQ(memory->outputs[2], kPm4PatchValue + 0x100u);
  EXPECT_EQ(memory->outputs[3], 0u);
  iree_hal_amdgpu_pm4_program_deinitialize(&pm4_program);
  iree_hal_amdgpu_pm4_program_deinitialize(&target_pm4_program);

  memset(memory, 0, sizeof(*memory));
  pm4_dword_count = 0;
  memory->store_kernargs[4] = {/*.target=*/&memory->outputs[4],
                               /*.value=*/kPm4LdsValue};
  IREE_ASSERT_OK(AppendPm4HostAcquire(agent_pm4_barrier_capabilities,
                                      pm4_dwords, IREE_ARRAYSIZE(pm4_dwords),
                                      &pm4_dword_count));
  IREE_ASSERT_OK(AppendPm4DispatchDirect(
      launch_states[4], reinterpret_cast<uintptr_t>(&memory->store_kernargs[4]),
      pm4_dwords, IREE_ARRAYSIZE(pm4_dwords), &pm4_dword_count));
  IREE_ASSERT_OK(AppendPm4CompletionWrite(
      agent_pm4_barrier_capabilities, pm4_dwords, IREE_ARRAYSIZE(pm4_dwords),
      &pm4_dword_count, &memory->completion, /*value=*/4));
  IREE_ASSERT_OK(iree_hal_amdgpu_pm4_program_initialize(
      &libhsa, gpu_agent, pm4_memory_pool, pm4_dwords, pm4_dword_count,
      &pm4_program));

  iree_hsa_signal_store_screlease(IREE_LIBHSA(&libhsa), completion_signal, 1);
  const uint64_t lds_pm4_packet_id =
      iree_hal_amdgpu_aql_ring_reserve(&aql_ring, /*count=*/1);
  pm4_packet = iree_hal_amdgpu_aql_ring_packet(&aql_ring, lds_pm4_packet_id);
  memset(pm4_packet, 0, sizeof(*pm4_packet));
  pm4_setup = 0;
  const uint16_t lds_pm4_header = iree_hal_amdgpu_aql_emit_pm4_ib_dwords(
      &pm4_packet->pm4_ib, pm4_program.dwords, pm4_program.dword_count,
      iree_hal_amdgpu_aql_packet_control_barrier_system(), completion_signal,
      &pm4_setup);
  iree_hal_amdgpu_aql_ring_commit(pm4_packet, lds_pm4_header, pm4_setup);
  iree_hal_amdgpu_aql_ring_doorbell(&aql_ring, lds_pm4_packet_id);
  EXPECT_EQ(
      iree_hsa_signal_wait_scacquire(
          IREE_LIBHSA(&libhsa), completion_signal, HSA_SIGNAL_CONDITION_EQ,
          /*compare_value=*/0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED),
      0);
  WaitForCompletionWord(&memory->completion, /*value=*/4);
  EXPECT_EQ(memory->outputs[4], 64u * kPm4LdsValue + 2016u);
  iree_hal_amdgpu_pm4_program_deinitialize(&pm4_program);

  EXPECT_EQ(queue_error.callback_count.load(std::memory_order_relaxed), 0u);
  EXPECT_EQ(queue_error.status.load(std::memory_order_relaxed),
            static_cast<uint32_t>(HSA_STATUS_SUCCESS));

  IREE_ASSERT_OK(
      iree_hsa_signal_destroy(IREE_LIBHSA(&libhsa), completion_signal));
  IREE_ASSERT_OK(iree_hsa_queue_destroy(IREE_LIBHSA(&libhsa), queue));
  IREE_ASSERT_OK(iree_hsa_amd_memory_pool_free(IREE_LIBHSA(&libhsa), memory));
  IREE_ASSERT_OK(iree_hsa_executable_destroy(IREE_LIBHSA(&libhsa), executable));
}

struct alignas(64) AtomicPacketMemory {
  uint32_t wait32[3];
  uint64_t wait64[3];
  iree_atomic_int32_t staged_wait32[3];
  iree_atomic_int64_t staged_wait64[3];
  uint64_t payload[6];
  uint64_t observed_payload[6];
  uint32_t store32;
  uint64_t store64;
  iree_atomic_int32_t marker;
  uint32_t completion;
};

struct alignas(64) AtomicPacketDeviceMemory {
  uint32_t rmw32[5];
  uint64_t rmw64[5];
  uint32_t store32;
  uint64_t store64;
};

template <typename T>
static T ApplyAtomicRmw(T value, iree_hal_atomic_rmw_operation_t operation,
                        T operand) {
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

TEST_F(PM4DispatchLiveTest, NativeMemoryPacketMatrix) {
  hsa_agent_t cpu_agent = topology.cpu_agents[0];
  hsa_agent_t gpu_agent = topology.gpu_agents[0];
  uint32_t xcc_count = 0;
  IREE_ASSERT_OK(iree_hsa_agent_get_info(
      IREE_LIBHSA(&libhsa), gpu_agent,
      (hsa_agent_info_t)HSA_AMD_AGENT_INFO_NUM_XCC, &xcc_count));
  ASSERT_GT(xcc_count, 0u);
  hsa_amd_memory_pool_t host_memory_pool = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_find_fine_global_memory_pool(
      &libhsa, cpu_agent, &host_memory_pool));
  AtomicPacketMemory* memory = nullptr;
  IREE_ASSERT_OK(iree_hsa_amd_memory_pool_allocate(
      IREE_LIBHSA(&libhsa), host_memory_pool, sizeof(*memory),
      HSA_AMD_MEMORY_POOL_STANDARD_FLAG, reinterpret_cast<void**>(&memory)));
  IREE_ASSERT_OK(iree_hsa_amd_agents_allow_access(IREE_LIBHSA(&libhsa),
                                                  /*num_agents=*/1, &gpu_agent,
                                                  /*flags=*/nullptr, memory));

  hsa_amd_memory_pool_t device_memory_pool = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_find_coarse_global_memory_pool(
      &libhsa, gpu_agent, &device_memory_pool));
  AtomicPacketDeviceMemory* device_memory = nullptr;
  IREE_ASSERT_OK(iree_hsa_amd_memory_pool_allocate(
      IREE_LIBHSA(&libhsa), device_memory_pool, sizeof(*device_memory),
      HSA_AMD_MEMORY_POOL_STANDARD_FLAG,
      reinterpret_cast<void**>(&device_memory)));
  IREE_ASSERT_OK(iree_hsa_amd_agents_allow_access(
      IREE_LIBHSA(&libhsa), /*num_agents=*/1, &cpu_agent, /*flags=*/nullptr,
      device_memory));

  hsa_amd_memory_pool_t pm4_memory_pool = {0};
  IREE_ASSERT_OK(iree_hal_amdgpu_find_coarse_global_memory_pool(
      &libhsa, cpu_agent, &pm4_memory_pool));
  QueueError queue_error;
  hsa_queue_t* queue = nullptr;
  IREE_ASSERT_OK(iree_hsa_queue_create(
      IREE_LIBHSA(&libhsa), gpu_agent, /*size=*/64, HSA_QUEUE_TYPE_MULTI,
      HsaQueueErrorCallback, &queue_error, UINT32_MAX, UINT32_MAX, &queue));
  iree_hal_amdgpu_aql_ring_t aql_ring;
  iree_hal_amdgpu_aql_ring_initialize(
      &libhsa, reinterpret_cast<iree_amd_queue_t*>(queue), &aql_ring);
  hsa_signal_t completion_signal = iree_hsa_signal_null();
  IREE_ASSERT_OK(iree_hsa_amd_signal_create(
      IREE_LIBHSA(&libhsa), /*initial_value=*/1, /*num_consumers=*/0,
      /*consumers=*/nullptr, /*attributes=*/0, &completion_signal));

  memset(memory, 0, sizeof(*memory));
  memset(device_memory, 0, sizeof(*device_memory));
  memory->wait32[0] = 0xAA55;
  memory->wait32[1] = 0xAA55;
  memory->wait32[2] = 0x80000000u;
  memory->wait64[0] = 0x123456789ABCDEF0ull;
  memory->wait64[1] = 0x123456789ABCDEF0ull;
  memory->wait64[2] = 0x8000000000000000ull;
  const uint32_t initial_rmw32[] = {10, 10, 0xFF00, 0xF0, 0xAA};
  const uint64_t initial_rmw64[] = {
      0x100000000ull,        0x100000010ull,        0xFFFF0000FFFF0000ull,
      0xF000000000000000ull, 0xAAAAAAAAAAAAAAAAull,
  };
  memcpy(device_memory->rmw32, initial_rmw32, sizeof(initial_rmw32));
  memcpy(device_memory->rmw64, initial_rmw64, sizeof(initial_rmw64));

  uint32_t pm4_dwords[256] = {0};
  uint32_t pm4_dword_count = 0;
  IREE_ASSERT_OK(AppendPm4HostAcquire(agent_pm4_barrier_capabilities,
                                      pm4_dwords, IREE_ARRAYSIZE(pm4_dwords),
                                      &pm4_dword_count));
  IREE_ASSERT_OK(AppendPm4AtomicWait(
      IREE_HAL_ATOMIC_WIDTH_32, IREE_HAL_ATOMIC_WAIT_CONDITION_EQUAL,
      &memory->wait32[0], 0x55, 0xFF, pm4_dwords, IREE_ARRAYSIZE(pm4_dwords),
      &pm4_dword_count));
  IREE_ASSERT_OK(AppendPm4AtomicWait(
      IREE_HAL_ATOMIC_WIDTH_32, IREE_HAL_ATOMIC_WAIT_CONDITION_NOT_EQUAL,
      &memory->wait32[1], 0x54, 0xFF, pm4_dwords, IREE_ARRAYSIZE(pm4_dwords),
      &pm4_dword_count));
  IREE_ASSERT_OK(AppendPm4AtomicWait(
      IREE_HAL_ATOMIC_WIDTH_32,
      IREE_HAL_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL, &memory->wait32[2],
      0x7FFFFFFFu, UINT32_MAX, pm4_dwords, IREE_ARRAYSIZE(pm4_dwords),
      &pm4_dword_count));
  IREE_ASSERT_OK(AppendPm4AtomicWait(
      IREE_HAL_ATOMIC_WIDTH_64, IREE_HAL_ATOMIC_WAIT_CONDITION_EQUAL,
      &memory->wait64[0], 0x1234000000000000ull, 0xFFFF000000000000ull,
      pm4_dwords, IREE_ARRAYSIZE(pm4_dwords), &pm4_dword_count));
  IREE_ASSERT_OK(AppendPm4AtomicWait(
      IREE_HAL_ATOMIC_WIDTH_64, IREE_HAL_ATOMIC_WAIT_CONDITION_NOT_EQUAL,
      &memory->wait64[1], 0x1235000000000000ull, 0xFFFF000000000000ull,
      pm4_dwords, IREE_ARRAYSIZE(pm4_dwords), &pm4_dword_count));
  IREE_ASSERT_OK(AppendPm4AtomicWait(
      IREE_HAL_ATOMIC_WIDTH_64,
      IREE_HAL_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL, &memory->wait64[2],
      0x7FFFFFFFFFFFFFFFull, UINT64_MAX, pm4_dwords, IREE_ARRAYSIZE(pm4_dwords),
      &pm4_dword_count));

  const iree_hal_atomic_rmw_operation_t operations[] = {
      IREE_HAL_ATOMIC_RMW_OPERATION_ADD, IREE_HAL_ATOMIC_RMW_OPERATION_SUBTRACT,
      IREE_HAL_ATOMIC_RMW_OPERATION_AND, IREE_HAL_ATOMIC_RMW_OPERATION_OR,
      IREE_HAL_ATOMIC_RMW_OPERATION_XOR,
  };
  const uint64_t operands32[] = {5, 3, 0x0FF0, 0x0F, 0xFF};
  const uint64_t operands64[] = {
      0x200000000ull,        0x10,       0x0FFF0FFF0FFF0FFFull,
      0x0F0000000000000Full, UINT64_MAX,
  };
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(operations); ++i) {
    IREE_ASSERT_OK(AppendPm4AtomicRmw(IREE_HAL_ATOMIC_WIDTH_32, operations[i],
                                      &device_memory->rmw32[i], operands32[i],
                                      pm4_dwords, IREE_ARRAYSIZE(pm4_dwords),
                                      &pm4_dword_count));
    IREE_ASSERT_OK(AppendPm4AtomicRmw(IREE_HAL_ATOMIC_WIDTH_64, operations[i],
                                      &device_memory->rmw64[i], operands64[i],
                                      pm4_dwords, IREE_ARRAYSIZE(pm4_dwords),
                                      &pm4_dword_count));
  }
  IREE_ASSERT_OK(AppendPm4AtomicStore(
      IREE_HAL_ATOMIC_WIDTH_32, &device_memory->store32, 0x76543210u,
      pm4_dwords, IREE_ARRAYSIZE(pm4_dwords), &pm4_dword_count));
  IREE_ASSERT_OK(AppendPm4AtomicStore(
      IREE_HAL_ATOMIC_WIDTH_64, &device_memory->store64, 0xFEDCBA9876543210ull,
      pm4_dwords, IREE_ARRAYSIZE(pm4_dwords), &pm4_dword_count));
  IREE_ASSERT_OK(AppendPm4AtomicStore(
      IREE_HAL_ATOMIC_WIDTH_32, &memory->store32, 0x89ABCDEFu, pm4_dwords,
      IREE_ARRAYSIZE(pm4_dwords), &pm4_dword_count));
  IREE_ASSERT_OK(AppendPm4AtomicStore(
      IREE_HAL_ATOMIC_WIDTH_64, &memory->store64, 0x0123456789ABCDEFull,
      pm4_dwords, IREE_ARRAYSIZE(pm4_dwords), &pm4_dword_count));
  IREE_ASSERT_OK(AppendPm4CompletionWrite(
      agent_pm4_barrier_capabilities, pm4_dwords, IREE_ARRAYSIZE(pm4_dwords),
      &pm4_dword_count, &memory->completion, /*value=*/1));

  iree_hal_amdgpu_pm4_program_t pm4_program = {};
  IREE_ASSERT_OK(iree_hal_amdgpu_pm4_program_initialize(
      &libhsa, gpu_agent, pm4_memory_pool, pm4_dwords, pm4_dword_count,
      &pm4_program));
  const uint64_t packet_id =
      iree_hal_amdgpu_aql_ring_reserve(&aql_ring, /*count=*/1);
  iree_hal_amdgpu_aql_packet_t* packet =
      iree_hal_amdgpu_aql_ring_packet(&aql_ring, packet_id);
  memset(packet, 0, sizeof(*packet));
  uint16_t setup = 0;
  const uint16_t header = iree_hal_amdgpu_aql_emit_pm4_ib_dwords(
      &packet->pm4_ib, pm4_program.dwords, pm4_program.dword_count,
      iree_hal_amdgpu_aql_packet_control_barrier_system(), completion_signal,
      &setup);
  iree_hal_amdgpu_aql_ring_commit(packet, header, setup);
  iree_hal_amdgpu_aql_ring_doorbell(&aql_ring, packet_id);
  EXPECT_EQ(
      iree_hsa_signal_wait_scacquire(
          IREE_LIBHSA(&libhsa), completion_signal, HSA_SIGNAL_CONDITION_EQ,
          /*compare_value=*/0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED),
      0);
  WaitForCompletionWord(&memory->completion, /*value=*/1);

  for (uint32_t i = 0; i < IREE_ARRAYSIZE(operations); ++i) {
    uint32_t expected32 = initial_rmw32[i];
    uint64_t expected64 = initial_rmw64[i];
    for (uint32_t xcc = 0; xcc < xcc_count; ++xcc) {
      expected32 = ApplyAtomicRmw(expected32, operations[i],
                                  static_cast<uint32_t>(operands32[i]));
      expected64 = ApplyAtomicRmw(expected64, operations[i], operands64[i]);
    }
    EXPECT_EQ(device_memory->rmw32[i], expected32);
    EXPECT_EQ(device_memory->rmw64[i], expected64);
  }
  EXPECT_EQ(device_memory->store32, 0x76543210u);
  EXPECT_EQ(device_memory->store64, 0xFEDCBA9876543210ull);
  EXPECT_EQ(memory->store32, 0x89ABCDEFu);
  EXPECT_EQ(memory->store64, 0x0123456789ABCDEFull);
  EXPECT_EQ(queue_error.callback_count.load(std::memory_order_relaxed), 0u);

  iree_hal_amdgpu_pm4_program_deinitialize(&pm4_program);

  iree_atomic_store(&memory->marker, 0, iree_memory_order_relaxed);
  iree_atomic_store(&memory->staged_wait32[0], 0, iree_memory_order_relaxed);
  iree_atomic_store(&memory->staged_wait32[1], 0x54, iree_memory_order_relaxed);
  iree_atomic_store(&memory->staged_wait32[2], 0x7FFFFFFE,
                    iree_memory_order_relaxed);
  iree_atomic_store(&memory->staged_wait64[0], 0, iree_memory_order_relaxed);
  iree_atomic_store(&memory->staged_wait64[1],
                    static_cast<int64_t>(0x1234000000000000ull),
                    iree_memory_order_relaxed);
  iree_atomic_store(&memory->staged_wait64[2],
                    static_cast<int64_t>(0x7FFFFFFFFFFFFFFEull),
                    iree_memory_order_relaxed);
  memset(memory->payload, 0, sizeof(memory->payload));
  memset(memory->observed_payload, 0, sizeof(memory->observed_payload));
  memory->completion = 0;

  pm4_dword_count = 0;
  const uint64_t staged_wait_values[6] = {
      0x55,
      0x54,
      0x7FFFFFFFu,
      0x1234000000000000ull,
      0x1234000000000000ull,
      0x7FFFFFFFFFFFFFFFull,
  };
  const uint64_t staged_wait_masks[6] = {
      0xFF,
      0xFF,
      UINT32_MAX,
      0xFFFF000000000000ull,
      0xFFFF000000000000ull,
      UINT64_MAX,
  };
  const iree_hal_atomic_wait_condition_t staged_wait_conditions[6] = {
      IREE_HAL_ATOMIC_WAIT_CONDITION_EQUAL,
      IREE_HAL_ATOMIC_WAIT_CONDITION_NOT_EQUAL,
      IREE_HAL_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL,
      IREE_HAL_ATOMIC_WAIT_CONDITION_EQUAL,
      IREE_HAL_ATOMIC_WAIT_CONDITION_NOT_EQUAL,
      IREE_HAL_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL,
  };
  for (uint32_t i = 0; i < 6; ++i) {
    IREE_ASSERT_OK(AppendPm4CompletionWrite(
        agent_pm4_barrier_capabilities, pm4_dwords, IREE_ARRAYSIZE(pm4_dwords),
        &pm4_dword_count, &memory->marker,
        /*value=*/i + 1));
    void* target = i < 3 ? static_cast<void*>(&memory->staged_wait32[i])
                         : static_cast<void*>(&memory->staged_wait64[i - 3]);
    const iree_hal_atomic_width_t width =
        i < 3 ? IREE_HAL_ATOMIC_WIDTH_32 : IREE_HAL_ATOMIC_WIDTH_64;
    IREE_ASSERT_OK(AppendPm4AtomicWait(
        width, staged_wait_conditions[i], target, staged_wait_values[i],
        staged_wait_masks[i], pm4_dwords, IREE_ARRAYSIZE(pm4_dwords),
        &pm4_dword_count));
    IREE_ASSERT_OK(AppendPm4HostAcquire(agent_pm4_barrier_capabilities,
                                        pm4_dwords, IREE_ARRAYSIZE(pm4_dwords),
                                        &pm4_dword_count));
    IREE_ASSERT_OK(AppendPm4CopyData64(pm4_dwords, IREE_ARRAYSIZE(pm4_dwords),
                                       &pm4_dword_count, &memory->payload[i],
                                       &memory->observed_payload[i]));
  }
  IREE_ASSERT_OK(AppendPm4CompletionWrite(
      agent_pm4_barrier_capabilities, pm4_dwords, IREE_ARRAYSIZE(pm4_dwords),
      &pm4_dword_count, &memory->completion, /*value=*/2));
  IREE_ASSERT_OK(iree_hal_amdgpu_pm4_program_initialize(
      &libhsa, gpu_agent, pm4_memory_pool, pm4_dwords, pm4_dword_count,
      &pm4_program));

  iree_hsa_signal_store_screlease(IREE_LIBHSA(&libhsa), completion_signal, 1);
  const uint64_t staged_packet_id =
      iree_hal_amdgpu_aql_ring_reserve(&aql_ring, /*count=*/1);
  packet = iree_hal_amdgpu_aql_ring_packet(&aql_ring, staged_packet_id);
  memset(packet, 0, sizeof(*packet));
  setup = 0;
  const uint16_t staged_header = iree_hal_amdgpu_aql_emit_pm4_ib_dwords(
      &packet->pm4_ib, pm4_program.dwords, pm4_program.dword_count,
      iree_hal_amdgpu_aql_packet_control_barrier_system(), completion_signal,
      &setup);
  iree_hal_amdgpu_aql_ring_commit(packet, staged_header, setup);
  iree_hal_amdgpu_aql_ring_doorbell(&aql_ring, staged_packet_id);

  const uint64_t payload_values[6] = {
      0xA000000000000001ull, 0xA000000000000002ull, 0xA000000000000003ull,
      0xA000000000000004ull, 0xA000000000000005ull, 0xA000000000000006ull,
  };
  const int32_t published_wait32[3] = {
      0x55,
      0x55,
      static_cast<int32_t>(0x80000000u),
  };
  const int64_t published_wait64[3] = {
      static_cast<int64_t>(0x123456789ABCDEF0ull),
      static_cast<int64_t>(0x1235000000000000ull),
      static_cast<int64_t>(0x8000000000000000ull),
  };
  for (uint32_t i = 0; i < 6; ++i) {
    WaitForMarker(&memory->marker, static_cast<int32_t>(i + 1));
    memory->payload[i] = payload_values[i];
    if (i < 3) {
      iree_atomic_store(&memory->staged_wait32[i], published_wait32[i],
                        iree_memory_order_release);
    } else {
      iree_atomic_store(&memory->staged_wait64[i - 3], published_wait64[i - 3],
                        iree_memory_order_release);
    }
  }

  EXPECT_EQ(
      iree_hsa_signal_wait_scacquire(
          IREE_LIBHSA(&libhsa), completion_signal, HSA_SIGNAL_CONDITION_EQ,
          /*compare_value=*/0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED),
      0);
  WaitForCompletionWord(&memory->completion, /*value=*/2);
  for (uint32_t i = 0; i < IREE_ARRAYSIZE(payload_values); ++i) {
    EXPECT_EQ(memory->observed_payload[i], payload_values[i]);
  }
  EXPECT_EQ(queue_error.callback_count.load(std::memory_order_relaxed), 0u);
  iree_hal_amdgpu_pm4_program_deinitialize(&pm4_program);

  IREE_ASSERT_OK(
      iree_hsa_signal_destroy(IREE_LIBHSA(&libhsa), completion_signal));
  IREE_ASSERT_OK(iree_hsa_queue_destroy(IREE_LIBHSA(&libhsa), queue));
  IREE_ASSERT_OK(
      iree_hsa_amd_memory_pool_free(IREE_LIBHSA(&libhsa), device_memory));
  IREE_ASSERT_OK(iree_hsa_amd_memory_pool_free(IREE_LIBHSA(&libhsa), memory));
}

}  // namespace
}  // namespace iree::hal::amdgpu
