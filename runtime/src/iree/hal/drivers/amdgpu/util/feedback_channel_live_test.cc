// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/abi/kernel_descriptor.h"
#include "iree/hal/drivers/amdgpu/util/aql_emitter.h"
#include "iree/hal/drivers/amdgpu/util/aql_ring.h"
#include "iree/hal/drivers/amdgpu/util/feedback_channel.h"
#include "iree/hal/drivers/amdgpu/util/feedback_channel_test_kernels.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"
#include "iree/hal/drivers/amdgpu/util/target_id.h"
#include "iree/hal/drivers/amdgpu/util/topology.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

constexpr char kTestCodeObjectBaseName[] = "feedback_channel_test_kernels";
constexpr uint64_t kExecutableId = 0x454845434944ull;
constexpr uint64_t kMagic = 0x4644424752494E47ull;
constexpr uint16_t kWorkgroupSize[3] = {64, 1, 1};
constexpr uint32_t kDispatchGridSize[3] = {64, 1, 1};

struct FeedbackKernargs {
  // Device-readable pointer to the feedback configuration.
  const iree_hal_amdgpu_feedback_config_t* config;
  // Magic value copied into each emitted payload.
  uint64_t magic;
};

struct FeedbackPayload {
  // Magic value supplied by the host.
  uint64_t magic;
  // X dimension workgroup id captured by the kernel.
  uint32_t workgroup_id_x;
  // X dimension workitem id captured by the kernel.
  uint32_t workitem_id_x;
  // Dispatch packet pointer captured by the kernel.
  uint64_t dispatch_ptr;
};

struct alignas(64) LiveMemory {
  // Device-readable copy of the channel configuration.
  iree_hal_amdgpu_feedback_config_t config;
  // Kernel argument storage.
  FeedbackKernargs kernargs;
};

struct KernelInfo {
  // HSA executable symbol for the kernel.
  hsa_executable_symbol_t symbol = {};
  // Device kernel object pointer placed in AQL dispatch packets.
  uint64_t kernel_object = 0;
  // Kernel argument byte length reported by HSA metadata.
  uint32_t kernarg_size = 0;
  // Kernel argument alignment reported by HSA metadata.
  uint32_t kernarg_alignment = 0;
  // Private segment byte length reported by HSA metadata.
  uint32_t private_segment_size = 0;
  // Group segment byte length reported by HSA metadata.
  uint32_t group_segment_size = 0;
};

struct QueueError {
  // Number of queue error callbacks observed.
  std::atomic<uint32_t> callback_count{0};
  // Last HSA status observed by the queue error callback.
  std::atomic<uint32_t> status{HSA_STATUS_SUCCESS};
};

struct IsaQuery {
  // Borrowed HSA API table.
  const iree_hal_amdgpu_libhsa_t* libhsa = nullptr;
  // True when a compatible ISA was found.
  bool found = false;
  // Exact target processor reported by the agent.
  std::string exact_target;
  // Built code object target compatible with the exact target.
  std::string code_object_target;
};

struct LiveDrainState {
  // Packet headers captured in drain order.
  std::vector<iree_hal_amdgpu_feedback_packet_t> packets;
  // Packet payloads captured in drain order.
  std::vector<FeedbackPayload> payloads;
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
      iree_hal_amdgpu_feedback_channel_test_kernels_create();
  for (iree_host_size_t i = 0;
       i < iree_hal_amdgpu_feedback_channel_test_kernels_size(); ++i) {
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

  iree_hal_amdgpu_target_id_t exact_target_id;
  iree_status_t status = iree_hal_amdgpu_target_id_parse_hsa_isa_name(
      iree_make_cstring_view(name.data()), &exact_target_id);
  if (!iree_status_is_ok(status)) {
    iree_status_free(status);
    return HSA_STATUS_SUCCESS;
  }

  iree_hal_amdgpu_target_id_t code_object_target_id;
  status = iree_hal_amdgpu_target_id_lookup_code_object_target(
      &exact_target_id, &code_object_target_id);
  if (!iree_status_is_ok(status)) {
    iree_status_free(status);
    return HSA_STATUS_ERROR;
  }
  query->exact_target = StringViewToString(exact_target_id.processor);
  query->code_object_target =
      StringViewToString(code_object_target_id.processor);
  query->found = true;
  return HSA_STATUS_INFO_BREAK;
}

static bool QueryAgentCodeObjectTarget(const iree_hal_amdgpu_libhsa_t* libhsa,
                                       hsa_agent_t agent,
                                       std::string* out_exact_target,
                                       std::string* out_code_object_target) {
  IsaQuery query;
  query.libhsa = libhsa;
  iree_status_t status = iree_hsa_agent_iterate_isas(
      IREE_LIBHSA(libhsa), agent, FindAgentCodeObjectTarget, &query);
  if (!iree_status_is_ok(status)) {
    iree_status_free(status);
  }
  if (!query.found) return false;
  *out_exact_target = query.exact_target;
  *out_code_object_target = query.code_object_target;
  return true;
}

static iree_status_t LookupKernel(const iree_hal_amdgpu_libhsa_t* libhsa,
                                  hsa_executable_t executable,
                                  hsa_agent_t agent, const char* kernel_name,
                                  KernelInfo* out_info) {
  *out_info = KernelInfo{};

  char descriptor_symbol_name[128] = {};
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

static iree_status_t CaptureLivePacket(
    const iree_hal_amdgpu_feedback_packet_t* packet, void* user_data) {
  LiveDrainState* state = reinterpret_cast<LiveDrainState*>(user_data);
  const FeedbackPayload* payload = reinterpret_cast<const FeedbackPayload*>(
      (const uint8_t*)packet + packet->header_length);
  state->packets.push_back(*packet);
  state->payloads.push_back(*payload);
  return iree_ok_status();
}

class FeedbackChannelLiveTest : public ::testing::Test {
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
                                    &agent_exact_target,
                                    &agent_code_object_target)) {
      GTEST_SKIP() << "could not query AMDGPU agent ISA";
    }
    const std::string file_name =
        TestCodeObjectFileName(agent_code_object_target);
    test_code_object_data = FindTestCodeObjectData(agent_code_object_target);
    if (test_code_object_data.data_length == 0) {
      GTEST_SKIP() << "feedback channel code object " << file_name
                   << " for agent " << agent_exact_target << " via "
                   << agent_code_object_target
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
  static std::string agent_exact_target;
  static std::string agent_code_object_target;
  static iree_const_byte_span_t test_code_object_data;
};

iree_allocator_t FeedbackChannelLiveTest::host_allocator;
iree_hal_amdgpu_libhsa_t FeedbackChannelLiveTest::libhsa;
iree_hal_amdgpu_topology_t FeedbackChannelLiveTest::topology;
std::string FeedbackChannelLiveTest::agent_exact_target;
std::string FeedbackChannelLiveTest::agent_code_object_target;
iree_const_byte_span_t FeedbackChannelLiveTest::test_code_object_data;

TEST_F(FeedbackChannelLiveTest, DeviceProducerSignalsHost) {
  hsa_agent_t cpu_agent = topology.cpu_agents[0];
  hsa_agent_t gpu_agent = topology.gpu_agents[0];

  hsa_amd_memory_pool_t control_memory_pool = {};
  IREE_ASSERT_OK(iree_hal_amdgpu_find_fine_global_memory_pool(
      &libhsa, cpu_agent, &control_memory_pool));
  hsa_amd_memory_pool_t ring_memory_pool = {};
  IREE_ASSERT_OK(iree_hal_amdgpu_find_coarse_global_memory_pool(
      &libhsa, gpu_agent, &ring_memory_pool));

  iree_hal_amdgpu_feedback_channel_t channel = {};
  iree_hal_amdgpu_feedback_channel_params_t channel_params = {};
  channel_params.libhsa = &libhsa;
  channel_params.device_agent = gpu_agent;
  channel_params.control_memory_pool = control_memory_pool;
  channel_params.ring_memory_pool = ring_memory_pool;
  channel_params.topology = &topology;
  channel_params.minimum_capacity = 4 * 1024;
  IREE_ASSERT_OK(
      iree_hal_amdgpu_feedback_channel_initialize(&channel_params, &channel));

  hsa_code_object_reader_t code_object_reader = {};
  IREE_ASSERT_OK(iree_hsa_code_object_reader_create_from_memory(
      IREE_LIBHSA(&libhsa), test_code_object_data.data,
      test_code_object_data.data_length, &code_object_reader));

  hsa_executable_t executable = {};
  IREE_ASSERT_OK(
      iree_hsa_executable_create_alt(IREE_LIBHSA(&libhsa), HSA_PROFILE_FULL,
                                     HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                     /*options=*/nullptr, &executable));
  hsa_loaded_code_object_t loaded_code_object = {};
  IREE_ASSERT_OK(iree_hsa_executable_load_agent_code_object(
      IREE_LIBHSA(&libhsa), executable, gpu_agent, code_object_reader,
      /*options=*/nullptr, &loaded_code_object));
  IREE_ASSERT_OK(iree_hsa_executable_freeze(IREE_LIBHSA(&libhsa), executable,
                                            /*options=*/nullptr));
  IREE_ASSERT_OK(iree_hsa_code_object_reader_destroy(IREE_LIBHSA(&libhsa),
                                                     code_object_reader));

  KernelInfo kernel;
  IREE_ASSERT_OK(LookupKernel(&libhsa, executable, gpu_agent,
                              "iree_hal_amdgpu_feedback_channel_test_emit",
                              &kernel));
  ASSERT_LE(kernel.kernarg_size, sizeof(FeedbackKernargs));
  ASSERT_LE(kernel.kernarg_alignment, alignof(LiveMemory));
  EXPECT_EQ(kernel.private_segment_size, 0u);
  EXPECT_EQ(kernel.group_segment_size, 0u);

  LiveMemory* memory = nullptr;
  IREE_ASSERT_OK(iree_hsa_amd_memory_pool_allocate(
      IREE_LIBHSA(&libhsa), control_memory_pool, sizeof(LiveMemory),
      HSA_AMD_MEMORY_POOL_STANDARD_FLAG, reinterpret_cast<void**>(&memory)));
  IREE_ASSERT_OK(iree_hsa_amd_agents_allow_access(IREE_LIBHSA(&libhsa),
                                                  /*num_agents=*/1, &gpu_agent,
                                                  /*flags=*/nullptr, memory));
  memory->config = channel.config;
  memory->config.source_context = kExecutableId;
  memory->kernargs.config = &memory->config;
  memory->kernargs.magic = kMagic;

  QueueError queue_error;
  hsa_queue_t* queue = nullptr;
  IREE_ASSERT_OK(iree_hsa_queue_create(
      IREE_LIBHSA(&libhsa), gpu_agent, /*size=*/64, HSA_QUEUE_TYPE_MULTI,
      HsaQueueErrorCallback, &queue_error, UINT32_MAX, UINT32_MAX, &queue));
  iree_hal_amdgpu_aql_ring_t aql_ring;
  iree_hal_amdgpu_aql_ring_initialize(
      reinterpret_cast<iree_amd_queue_t*>(queue), &aql_ring);

  hsa_signal_t completion_signal = iree_hsa_signal_null();
  IREE_ASSERT_OK(iree_hsa_amd_signal_create(
      IREE_LIBHSA(&libhsa), /*initial_value=*/1, /*num_consumers=*/0,
      /*consumers=*/nullptr, /*attributes=*/0, &completion_signal));

  const uint64_t packet_id = iree_hal_amdgpu_aql_ring_reserve(&aql_ring, 1);
  iree_hal_amdgpu_aql_packet_t* packet =
      iree_hal_amdgpu_aql_ring_packet(&aql_ring, packet_id);
  memset(packet, 0, sizeof(*packet));
  uint16_t setup = 0;
  const uint16_t header = iree_hal_amdgpu_aql_emit_dispatch(
      &packet->dispatch, kernel.kernel_object, &memory->kernargs,
      kWorkgroupSize, kDispatchGridSize, kernel.private_segment_size,
      kernel.group_segment_size,
      iree_hal_amdgpu_aql_packet_control_barrier_system(), completion_signal,
      &setup);
  iree_hal_amdgpu_aql_ring_commit(packet, header, setup);
  iree_hal_amdgpu_aql_ring_doorbell(&aql_ring, packet_id);

  const hsa_signal_value_t feedback_signal_value =
      iree_hsa_signal_wait_scacquire(
          IREE_LIBHSA(&libhsa), channel.notify_signal, HSA_SIGNAL_CONDITION_NE,
          /*compare_value=*/0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
  EXPECT_GT(feedback_signal_value, 0);
  EXPECT_EQ(
      iree_hsa_signal_wait_scacquire(
          IREE_LIBHSA(&libhsa), completion_signal, HSA_SIGNAL_CONDITION_EQ,
          /*compare_value=*/0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED),
      0);

  LiveDrainState drain_state;
  iree_host_size_t packet_count = 0;
  IREE_ASSERT_OK(iree_hal_amdgpu_feedback_channel_drain(
      &channel, kWorkgroupSize[0], CaptureLivePacket, &drain_state,
      &packet_count));
  ASSERT_EQ(packet_count, kWorkgroupSize[0]);
  ASSERT_EQ(drain_state.packets.size(), kWorkgroupSize[0]);
  ASSERT_EQ(drain_state.payloads.size(), kWorkgroupSize[0]);

  std::vector<bool> seen_workitems(kWorkgroupSize[0], false);
  for (iree_host_size_t i = 0; i < packet_count; ++i) {
    const iree_hal_amdgpu_feedback_packet_t& packet = drain_state.packets[i];
    const FeedbackPayload& payload = drain_state.payloads[i];
    EXPECT_EQ(packet.kind, IREE_HAL_AMDGPU_FEEDBACK_PACKET_KIND_USER);
    EXPECT_EQ(packet.flags, IREE_HAL_AMDGPU_FEEDBACK_PACKET_FLAG_ASYNC);
    EXPECT_EQ(packet.header_length, sizeof(iree_hal_amdgpu_feedback_packet_t));
    EXPECT_EQ(packet.source_workgroup_id_x, 0u);
    EXPECT_LT(packet.source_workitem_id_x, kWorkgroupSize[0]);
    EXPECT_NE(packet.source_dispatch_ptr, 0u);
    EXPECT_EQ(packet.source_context, kExecutableId);
    EXPECT_EQ(payload.magic, kMagic);
    EXPECT_EQ(payload.workgroup_id_x, packet.source_workgroup_id_x);
    EXPECT_EQ(payload.workitem_id_x, packet.source_workitem_id_x);
    EXPECT_EQ(payload.dispatch_ptr, packet.source_dispatch_ptr);
    seen_workitems[packet.source_workitem_id_x] = true;
  }
  for (uint32_t i = 0; i < kWorkgroupSize[0]; ++i) {
    EXPECT_TRUE(seen_workitems[i]) << "missing workitem " << i;
  }
  EXPECT_EQ(channel.control->dropped_packet_count, 0u);
  EXPECT_EQ(channel.control->read_tail, channel.control->reservation_head);
  EXPECT_EQ(queue_error.callback_count.load(std::memory_order_relaxed), 0u);
  EXPECT_EQ(queue_error.status.load(std::memory_order_relaxed),
            static_cast<uint32_t>(HSA_STATUS_SUCCESS));

  IREE_ASSERT_OK(
      iree_hsa_signal_destroy(IREE_LIBHSA(&libhsa), completion_signal));
  IREE_ASSERT_OK(iree_hsa_queue_destroy(IREE_LIBHSA(&libhsa), queue));
  IREE_ASSERT_OK(iree_hsa_amd_memory_pool_free(IREE_LIBHSA(&libhsa), memory));
  IREE_ASSERT_OK(iree_hsa_executable_destroy(IREE_LIBHSA(&libhsa), executable));
  iree_hal_amdgpu_feedback_channel_deinitialize(&channel);
}

}  // namespace
}  // namespace iree::hal::amdgpu
