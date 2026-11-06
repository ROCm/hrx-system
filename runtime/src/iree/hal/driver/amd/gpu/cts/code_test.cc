// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

#include "iree/base/byte_sequence.h"
#include "iree/base/internal/atomics.h"
#include "iree/hal/driver/amd/gpu/code_queue.h"
#include "iree/hal/driver/amd/gpu/cts/test_device.h"
#include "iree/hal/driver/amd/gpu/cts/testdata.h"
#include "iree/hal/driver/amd/gpu/image/hsaco.h"
#include "iree/hal/driver/amd/gpu/queue/aql/ring.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/barrier.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/commands.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/kmq/storage.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/launch.h"
#include "iree/hal/driver/amd/gpu/queue/user_queue.h"
#include "iree/hal/driver/amd/status.h"
#include "iree/hal/memory_scope.h"
#include "iree/testing/status_matchers.h"

namespace {

struct CodePublication {
  // Native compute carrier used by the code loader.
  amdf_queue_command_type_t command_type;
  // Native producer contract, also used by the independent consumer queue.
  amdf_queue_publication_modes_t publication_mode;
  // Use native writes into unmapped device-local code.
  bool upload;
};

class CodeTest : public iree::hal::amd::gpu::cts::GpuNativeTest,
                 public ::testing::WithParamInterface<CodePublication> {
 protected:
  amdf_queue_command_type_t command_type() const override {
    return GetParam().command_type;
  }

  void SetUp() override {
    GpuNativeTest::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    uint32_t publisher_family = UINT32_MAX;
    for (uint32_t i = 0; i < context_->endpoint_info.queue_family_count; ++i) {
      const auto& family = context_->families[i];
      if (family.command_type == GetParam().command_type &&
          (family.publication_modes & GetParam().publication_mode)) {
        publisher_family = i;
        break;
      }
    }
    if (publisher_family == UINT32_MAX) {
      GTEST_SKIP() << "Native compute publication mode is not advertised";
    }
    options_.queue_family_ordinal = publisher_family;
    IREE_ASSERT_OK(
        iree_hal_amd_gpu_context_prepare_memory(context_, UINT32_MAX));
    IREE_ASSERT_OK(iree_hal_amd_gpu_context_prepare_completion(
        context_, GetParam().publication_mode));
    ASSERT_NO_FATAL_FAILURE(SelectControlMemory(&access_, &control_policy_));
    IREE_ASSERT_OK(iree_hal_amd_gpu_code_queue_allocate(
        context_, publisher_family, GetParam().publication_mode,
        &control_policy_, iree_allocator_system(), &publisher_));
    IREE_ASSERT_OK(iree_hal_amd_fabric_memory_allocate(
        context_->api, &control_policy_, 4096, 64, context_->event_sink,
        &control_, nullptr));
    IREE_ASSERT_OK(IREE_HAL_AMD_STATUS_FROM_AMDF(
        context_->api->memory_query_address(
            control_.handle, 0, AMDF_MEMORY_ADDRESS_GPU, &control_address_),
        "query code test control address"));
    std::memset(control_.contents.data, 0, control_.contents.data_length);
    completion_ =
        reinterpret_cast<iree_atomic_int64_t*>(control_.contents.data);
    if (command_type() == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL) {
      auto* signal = reinterpret_cast<iree_hal_amd_gpu_aql_signal_t*>(
          control_.contents.data);
      *signal = {
          .kind = 1,
          .event_mailbox = context_->event_info.mailbox_address,
          .event_id = context_->event_info.interrupt_context,
      };
      completion_ = reinterpret_cast<iree_atomic_int64_t*>(&signal->value);
    } else {
      const iree_hal_amd_gpu_gfxip_version_t gfxip = {
          context_->target_info.gfx_ip.major,
          context_->target_info.gfx_ip.minor,
          context_->target_info.gfx_ip.stepping,
      };
      IREE_ASSERT_OK(
          iree_hal_amd_gpu_pm4_barrier_profile_initialize(gfxip, &barriers_));
    }
    if (GetParam().publication_mode == AMDF_QUEUE_PUBLICATION_MODE_KERNEL) {
      IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_kmq_storage_allocate(
          context_, publisher_family, 4096, 1, &control_policy_,
          iree_allocator_system(), &kernel_, &pm4_ring_));
      return;
    }
    const amdf_gpu_user_queue_create_info_t queue_info = {
        .type = AMDF_STRUCTURE_TYPE_GPU_USER_QUEUE_CREATE_INFO,
        .structure_size = sizeof(queue_info),
        .queue_family_ordinal = options_.queue_family_ordinal,
        .priority = AMDF_QUEUE_PRIORITY_NORMAL,
        .producer_mode = AMDF_QUEUE_PRODUCER_MODE_SINGLE,
        .required_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER,
        .ring_byte_length = 4096,
    };
    IREE_ASSERT_OK(iree_hal_amd_gpu_user_queue_initialize(
        context_, &queue_info, &user_, &mapping_, nullptr));
    if (command_type() == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL) {
      iree_hal_amd_gpu_aql_ring_initialize(
          reinterpret_cast<iree_hal_amd_gpu_aql_packet_t*>(
              mapping_.ring_address),
          0, static_cast<uint32_t>(mapping_.ring_byte_length / 64),
          reinterpret_cast<const iree_atomic_int64_t*>(
              mapping_.read_index_address),
          reinterpret_cast<iree_atomic_int64_t*>(mapping_.write_index_address),
          reinterpret_cast<volatile uint64_t*>(mapping_.doorbell_address),
          user_.host_publish_executor, &aql_ring_);
    } else {
      iree_hal_amd_gpu_pm4_ring_initialize(
          reinterpret_cast<uint32_t*>(mapping_.ring_address),
          static_cast<uint32_t>(mapping_.ring_byte_length / 4),
          user_.host_publish_executor, &pm4_ring_);
    }
  }

  void TearDown() override {
    if (pending_ && HasFatalFailure()) {
      // A failed execution wait supplies no completion proof. Stop this
      // consumer before releasing its borrowed code and control storage.
      IREE_ASSERT_OK(IREE_HAL_AMD_STATUS_FROM_AMDF(
          kernel_ ? iree_hal_amd_gpu_pm4_kmq_storage_inactivate(kernel_)
                  : iree_hal_amd_gpu_user_queue_inactivate(&user_),
          "stop failed code consumer"));
      pending_ = false;
    } else if (pending_) {
      ASSERT_NO_FATAL_FAILURE(Wait());
    }
    if (kernel_) {
      IREE_ASSERT_OK(IREE_HAL_AMD_STATUS_FROM_AMDF(
          iree_hal_amd_gpu_pm4_kmq_storage_free(kernel_),
          "join KERNEL code test queue"));
    } else if (user_.context) {
      IREE_ASSERT_OK(IREE_HAL_AMD_STATUS_FROM_AMDF(
          iree_hal_amd_gpu_user_queue_deinitialize(&user_, Published()),
          "join code test queue"));
    }
    IREE_ASSERT_OK(IREE_HAL_AMD_STATUS_FROM_AMDF(
        iree_hal_amd_gpu_code_queue_free(publisher_), "join code publication"));
    if (context_) {
      iree_hal_amd_fabric_memory_deinitialize(context_->api,
                                              context_->event_sink, &memory_);
      iree_hal_amd_fabric_memory_deinitialize(context_->api,
                                              context_->event_sink, &control_);
    }
    for (auto& image : images_) {
      iree_hal_amd_gpu_image_deinitialize(&image);
    }
    GpuNativeTest::TearDown();
  }

  void ReadImages() {
    constexpr const char* kPrograms[] = {"transform.bin",
                                         "transform_alternate.bin"};
    for (size_t ordinal = 0; ordinal < IREE_ARRAYSIZE(kPrograms); ++ordinal) {
      iree_const_byte_span_t artifact;
      IREE_ASSERT_OK(iree::hal::amd::gpu::cts::LookupExecutableData(
          "amd_gpu_code", context_->target_info,
          iree_make_cstring_view(kPrograms[ordinal]), &artifact));
      iree_byte_span_t bytes = {nullptr, artifact.data_length};
      IREE_ASSERT_OK(
          iree_allocator_malloc(iree_allocator_system(), bytes.data_length,
                                reinterpret_cast<void**>(&bytes.data)));
      std::memcpy(bytes.data, artifact.data, bytes.data_length);
      iree_byte_sequence_t* sequence = nullptr;
      iree_status_t status = iree_byte_sequence_create_from_span_move(
          &bytes, iree_allocator_system(), &sequence);
      iree_allocator_free(iree_allocator_system(), bytes.data);
      IREE_ASSERT_OK(status);
      status = iree_hal_amd_gpu_hsaco_image_initialize(
          sequence, iree_allocator_system(), &images_[ordinal]);
      iree_byte_sequence_release(sequence);
      IREE_ASSERT_OK(status);
    }
    for (auto& image : images_) {
      ASSERT_EQ(image.metadata.kernel_count, 1u);
    }
  }

  void Dispatch(const iree_hal_amd_gpu_image_t& image, uint64_t address) {
    iree_atomic_store(completion_, 1, iree_memory_order_release);
    if (command_type() == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL) {
      const iree_hal_amd_gpu_aql_dispatch_t dispatch = {
          .header = IREE_HAL_AMD_GPU_AQL_HEADER_DISPATCH | (1u << 16) |
                    IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM |
                    IREE_HAL_AMD_GPU_AQL_HEADER_RELEASE_SYSTEM,
          .workgroup_size = {64, 1, 1},
          .grid_size = {320, 1, 1},
          .kernel_object = address + image.kernels[0].descriptor_offset,
          .kernarg_address = control_address_ + 64,
          .completion_signal = control_address_,
      };
      const auto packet = iree_hal_amd_gpu_aql_packet_dispatch(&dispatch);
      uint64_t position = 0;
      ASSERT_TRUE(
          iree_hal_amd_gpu_aql_ring_try_reserve(&aql_ring_, 1, &position));
      iree_hal_amd_gpu_aql_ring_write_body(&aql_ring_, position, &packet);
      ASSERT_EQ(iree_hal_amd_gpu_aql_ring_publish_header(
                    &aql_ring_, &user_, position, packet.words[0]),
                AMDF_STATUS_OK);
      iree_hal_amd_gpu_aql_ring_notify(&aql_ring_);
    } else {
      ASSERT_NO_FATAL_FAILURE(DispatchPm4(image, address));
    }
    pending_ = true;
  }

  void DispatchPm4(const iree_hal_amd_gpu_image_t& image, uint64_t address) {
    const iree_hal_amd_gpu_gfxip_version_t gfxip = {
        context_->target_info.gfx_ip.major,
        context_->target_info.gfx_ip.minor,
        context_->target_info.gfx_ip.stepping,
    };
    iree_hal_amd_gpu_pm4_launch_t launch;
    IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_launch_initialize(
        gfxip,
        context_->families[options_.queue_family_ordinal].format_features,
        &image.kernels[0], address, &launch));
    ASSERT_EQ(launch.requirements, 0u);
    const uint32_t workgroup_size[] = {64, 1, 1};
    const uint32_t workgroup_count[] = {5, 1, 1};
    const uint32_t partial_size[] = {0, 0, 0};
    const iree_hal_amd_gpu_pm4_launch_inputs_t inputs = {
        .kernarg_pointer = control_address_ + 64,
    };
    // Wait joins complete native consumption before another dispatch reuses
    // the ring. Publication must invalidate instructions; the consumer only
    // acquires data and must not hide a missing loader cache action.
    constexpr uint32_t kWords = 128;
    iree_hal_amd_gpu_pm4_ring_reservation_t reservation;
    ASSERT_TRUE(iree_hal_amd_gpu_pm4_ring_try_reserve(
        &pm4_ring_, pm4_ring_.reserved_words, kWords, &reservation));
    uint32_t* words =
        iree_hal_amd_gpu_pm4_emit_acquire(reservation.words, barriers_.acquire);
    words = iree_hal_amd_gpu_pm4_launch_emit(words, &launch, workgroup_size,
                                             partial_size, 0, &inputs);
    words = iree_hal_amd_gpu_pm4_emit_dispatch_direct(
        words, workgroup_count, launch.dispatch_initiator);
    const iree_hal_amd_gpu_pm4_barrier_t release = {
        .cache_control = barriers_.release,
    };
    words = iree_hal_amd_gpu_pm4_barrier_emit_tail(words, release);
    words = iree_hal_amd_gpu_pm4_emit_release64(words, 0, control_address_, 0);
    if (!kernel_) {
      words = iree_hal_amd_gpu_pm4_emit_notify32(
          words, 0, context_->event_info.mailbox_address,
          context_->event_info.trigger_data,
          context_->event_info.interrupt_context);
    }
    ASSERT_LT(words, reservation.words + kWords - 1);
    iree_hal_amd_gpu_pm4_emit_nop(
        words, static_cast<uint32_t>(reservation.words + kWords - words));
    if (kernel_) {
      kernel_->pending = {
          .memory = kernel_->memory.handle,
          .byte_offset = (reservation.position & (pm4_ring_.capacity - 1)) * 4,
          .byte_length = kWords * 4,
      };
      IREE_ASSERT_OK(IREE_HAL_AMD_STATUS_FROM_AMDF(
          iree_hal_amd_gpu_pm4_kmq_storage_submit(kernel_, 0),
          "submit replaced code dispatch"));
    } else {
      IREE_ASSERT_OK(IREE_HAL_AMD_STATUS_FROM_AMDF(
          iree_hal_amd_gpu_user_queue_flush(&user_),
          "publish code consumer commands"));
      iree_atomic_store(
          reinterpret_cast<iree_atomic_int64_t*>(mapping_.write_index_address),
          pm4_ring_.reserved_words, iree_memory_order_release);
      const iree_hal_memory_transition_recipe_info_t fence = {
          .host = {.fence_after = IREE_HAL_HOST_CACHE_FENCE_X86_SFENCE},
      };
      iree_hal_memory_transition_execute_host_direct(&fence, nullptr, 1);
      *reinterpret_cast<volatile uint64_t*>(mapping_.doorbell_address) =
          pm4_ring_.reserved_words;
    }
  }

  uint64_t Published() const {
    return command_type() == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL
               ? aql_ring_.reserved
               : pm4_ring_.reserved_words;
  }

  void Wait() {
    if (kernel_) {
      IREE_ASSERT_OK(IREE_HAL_AMD_STATUS_FROM_AMDF(
          context_->api->kernel_queue_wait(kernel_->queue,
                                           kernel_->last_submission,
                                           AMDF_TIMEOUT_INFINITE, 0),
          "join KERNEL replaced code dispatch"));
    }
    while (iree_atomic_load(completion_, iree_memory_order_acquire) != 0) {
      ASSERT_EQ(kernel_, nullptr)
          << "Native retirement must cover shader writes and completion";
      IREE_ASSERT_OK(IREE_HAL_AMD_STATUS_FROM_AMDF(
          context_->gpu->event_wait(context_->event, AMDF_TIMEOUT_INFINITE),
          "wait replaced code"));
    }
    if (!kernel_) {
      IREE_ASSERT_OK(IREE_HAL_AMD_STATUS_FROM_AMDF(
          context_->api->user_queue_wait_consumed(user_.queue, Published(),
                                                  AMDF_TIMEOUT_INFINITE, 0),
          "join USER replaced code dispatch"));
    }
    pending_ = false;
  }

  // Immutable source images, both safe under the same argument/access contract.
  std::array<iree_hal_amd_gpu_image_t, 2> images_ = {};
  // A single native code allocation retained across all completed generations.
  iree_hal_amd_fabric_memory_t memory_ = {};
  // Device-owned cold loader machinery under test.
  iree_hal_amd_gpu_code_queue_t* publisher_ = nullptr;
  // Independent consumer queue: publication cannot rely on a same-queue fence.
  iree_hal_amd_gpu_user_queue_t user_ = {};
  // Native publication operands for a USER consumer.
  amdf_user_queue_mapping_info_t mapping_ = {};
  // Finite command storage and checked KERNEL consumer retirement.
  iree_hal_amd_gpu_pm4_kmq_storage_t* kernel_ = nullptr;
  // AQL consumer ring operands retained through native consumption.
  iree_hal_amd_gpu_aql_ring_t aql_ring_ = {};
  // PM4 consumer command storage, independently of its native carrier.
  iree_hal_amd_gpu_pm4_ring_t pm4_ring_ = {};
  // PM4 data-cache actions; instruction publication belongs to the loader.
  iree_hal_amd_gpu_pm4_barrier_profile_t barriers_ = {};
  // Borrowed scope requirements for coherent control storage.
  amdf_memory_device_access_t access_ = {};
  // Native control allocation policy.
  iree_hal_amd_fabric_memory_options_t control_policy_ = {};
  // Signal, native arguments, input, and output in one coherent page.
  iree_hal_amd_fabric_memory_t control_ = {};
  // GPU address of control byte zero.
  uint64_t control_address_ = 0;
  // Completion value borrowed from the native signal/marker in control_.
  iree_atomic_int64_t* completion_ = nullptr;
  // Accepted work must complete before teardown releases code or arguments.
  bool pending_ = false;
};

TEST_P(CodeTest, ReplacesCompletedInstructionsAtSameAddress) {
  ASSERT_NO_FATAL_FAILURE(ReadImages());
  const amdf_memory_device_access_t access = {
      .device = context_->device,
      .requirements =
          {
              .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE |
                        AMDF_MEMORY_ACCESS_EXECUTE,
              .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
              .address_kinds = UINT64_C(1) << AMDF_MEMORY_ADDRESS_GPU,
          },
  };
  iree_hal_amd_fabric_memory_options_t staging, resident;
  IREE_ASSERT_OK(iree_hal_amd_gpu_context_query_program_memory(
      context_, options_.queue_family_ordinal, &access, &staging, &resident));
  const bool upload = GetParam().upload;
  if (upload) {
    resident.host_access = 0;
    resident.required_flags &= ~AMDF_MEMORY_FLAG_HOST_VISIBLE;
  } else if (!resident.host_access) {
    GTEST_SKIP()
        << "Retained-address host publication requires a local CPU view";
  }
  // The initialized image crosses the bounded publisher's command capacity.
  // Its trailing bytes remain part of the same executable allocation.
  const uint64_t length = iree_align_uint64(
      std::max(images_[0].byte_length, images_[1].byte_length) + 65536, 256);
  IREE_ASSERT_OK(iree_hal_amd_fabric_memory_allocate(
      context_->api, &resident, length,
      std::max(images_[0].alignment, images_[1].alignment),
      context_->event_sink, &memory_, nullptr));
  EXPECT_NE(memory_.info.flags & AMDF_MEMORY_FLAG_DEVICE_LOCAL, 0u);
  if (upload) {
    EXPECT_EQ(memory_.contents.data, nullptr);
  }
  std::vector<uint8_t> upload_bytes(upload ? length : 0);
  uint8_t* contents = upload ? upload_bytes.data() : memory_.contents.data;
  uint64_t address = 0;
  IREE_ASSERT_OK(IREE_HAL_AMD_STATUS_FROM_AMDF(
      context_->api->memory_query_address(memory_.handle, 0,
                                          AMDF_MEMORY_ADDRESS_GPU, &address),
      "query retained code address"));
  ASSERT_EQ(images_[0].kernels[0].descriptor_offset,
            images_[1].kernels[0].descriptor_offset);
  ASSERT_EQ(images_[0].kernels[0].entry_offset,
            images_[1].kernels[0].entry_offset);
  constexpr uint32_t kCount = 259;
  const uint64_t arguments[] = {control_address_ + 512, control_address_ + 2048,
                                uint64_t{7} << 32 | kCount};
  std::memcpy(control_.contents.data + 64, arguments, sizeof(arguments));
  auto* input = reinterpret_cast<uint32_t*>(control_.contents.data + 512);
  auto* output = reinterpret_cast<uint32_t*>(control_.contents.data + 2048);
  for (uint32_t i = 0; i < kCount; ++i) {
    input[i] = 2 * i + 1;
  }
  // Exceed the publication ring's capacity to exercise consumption/reuse too.
  for (uint32_t generation = 0; generation < 65; ++generation) {
    SCOPED_TRACE(generation);
    const auto& image = images_[generation % 2];
    std::memset(contents, 0, length);
    std::vector<iree_byte_span_t> mappings;
    for (size_t i = 0; i < image.segment_count; ++i) {
      mappings.push_back(iree_make_byte_span(
          contents + image.segments[i].offset, image.segments[i].length));
    }
    IREE_ASSERT_OK(iree_hal_amd_gpu_image_load(
        &image, address, 0, nullptr, mappings.size(), mappings.data()));
    if (!upload) {
      IREE_ASSERT_OK(IREE_HAL_AMD_STATUS_FROM_AMDF(
          context_->api->host_mapping_cache_control(
              memory_.mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 0, length),
          "publish mapped code writes"));
    }
    IREE_ASSERT_OK(iree_hal_amd_gpu_code_queue_publish(
        publisher_, &memory_,
        upload ? iree_make_const_byte_span(contents, length)
               : iree_const_byte_span_empty()));
    std::fill_n(output, kCount + 1, 0xDEADBEEFu);
    ASSERT_NO_FATAL_FAILURE(Dispatch(image, address));
    ASSERT_NO_FATAL_FAILURE(Wait());
    const uint32_t multiplier = generation % 2 ? 5 : 3;
    for (uint32_t i = 0; i < kCount; ++i) {
      ASSERT_EQ(output[i], input[i] * multiplier + 7) << "element=" << i;
      ASSERT_EQ(input[i], 2 * i + 1);
    }
    ASSERT_EQ(output[kCount], 0xDEADBEEFu);
  }
}

INSTANTIATE_TEST_SUITE_P(
    LocalCode, CodeTest,
    ::testing::Values(CodePublication{AMDF_QUEUE_COMMAND_TYPE_GPU_AQL,
                                      AMDF_QUEUE_PUBLICATION_MODE_USER, false},
                      CodePublication{AMDF_QUEUE_COMMAND_TYPE_GPU_AQL,
                                      AMDF_QUEUE_PUBLICATION_MODE_USER, true},
                      CodePublication{AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
                                      AMDF_QUEUE_PUBLICATION_MODE_USER, false},
                      CodePublication{AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
                                      AMDF_QUEUE_PUBLICATION_MODE_USER, true},
                      CodePublication{AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
                                      AMDF_QUEUE_PUBLICATION_MODE_KERNEL,
                                      false},
                      CodePublication{AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
                                      AMDF_QUEUE_PUBLICATION_MODE_KERNEL,
                                      true}));

}  // namespace
