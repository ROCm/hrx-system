// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Vulkan SPIR-V BDA coverage. The driver publishes the exact HAL binding
// table the dispatch provides and the shader consumes it through the hidden BDA
// root.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/hal/cts/util/profile_test_util.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/vulkan/buffer.h"
#include "iree/hal/drivers/vulkan/command_buffer.h"
#include "iree/hal/drivers/vulkan/cts/bda_spirv_test_spv.h"
#include "iree/hal/drivers/vulkan/queue_stats.h"
#include "iree/hal/drivers/vulkan/queue_test_util.h"
#include "iree/hal/memory/tlsf_pool.h"

namespace iree::hal::cts {

using ::testing::ContainerEq;

constexpr char kBdaSpirvAdd7[] = "bda_i32_add7.spv";
constexpr char kBdaSpirvAdd7Binding1Length4_17[] =
    "bda_i32_add7_binding_1_length_4_17.spv";
constexpr char kBdaSpirvAdd7Bindings2[] = "bda_i32_add7_bindings_2.spv";
constexpr char kBdaSpirvAdd7ConstantLength4[] =
    "bda_i32_add7_constant_length_4.spv";
constexpr char kBdaSpirvAdd7DescriptorDecorated[] =
    "bda_i32_add7_descriptor_decorated.spv";
constexpr char kBdaSpirvAdd7NoMetadata[] = "bda_i32_add7_no_metadata.spv";
constexpr char kBdaSpirvDescriptorStorageVariable[] =
    "bda_descriptor_storage_variable.spv";
constexpr char kBdaSpirvMissingPhysicalStorageBufferAddresses[] =
    "bda_missing_physical_storage_buffer_addresses.spv";
constexpr char kBdaSpirvMissingPushConstantRoot[] =
    "bda_missing_push_constant_root.spv";
constexpr char kBdaSpirvNoopWithoutPushConstantRoot[] =
    "bda_noop_without_push_constant_root.spv";

static iree_const_byte_span_t BdaSpirvFixture(const char* file_name) {
  const iree_file_toc_t* toc = iree_hal_vulkan_cts_bda_spirv_test_spv_create();
  for (iree_host_size_t i = 0;
       i < iree_hal_vulkan_cts_bda_spirv_test_spv_size(); ++i) {
    if (std::strcmp(toc[i].name, file_name) == 0) {
      return iree_make_const_byte_span(toc[i].data, toc[i].size);
    }
  }
  ADD_FAILURE() << "BDA SPIR-V fixture not found: " << file_name;
  return iree_const_byte_span_empty();
}

// Small hand-authored malformed word sequences for parser states that cannot
// be represented as ordinary SPIR-V assembly fixtures.
static const uint32_t kBdaSpirvWithoutComputeEntryPoints[] = {
    0x07230203u,
    0x00010600u,
    0u,
    8u,
    0u,
    // Declares OpCapability Shader.
    0x00020011u,
    1u,
    // Declares OpCapability PhysicalStorageBufferAddresses.
    0x00020011u,
    5347u,
    // Declares OpMemoryModel PhysicalStorageBuffer64 GLSL450.
    0x0003000eu,
    5348u,
    1u,
};

static const uint32_t kBdaSpirvWithDuplicateComputeEntryNames[] = {
    0x07230203u,
    0x00010600u,
    0u,
    8u,
    0u,
    // Declares OpCapability Shader.
    0x00020011u,
    1u,
    // Declares OpCapability PhysicalStorageBufferAddresses.
    0x00020011u,
    5347u,
    // Declares OpMemoryModel PhysicalStorageBuffer64 GLSL450.
    0x0003000eu,
    5348u,
    1u,
    // Declares OpEntryPoint GLCompute %1 "main".
    0x0005000fu,
    5u,
    1u,
    0x6e69616du,
    0u,
    // Declares OpEntryPoint GLCompute %2 "main".
    0x0005000fu,
    5u,
    2u,
    0x6e69616du,
    0u,
};

static const uint32_t kBdaSpirvWithTruncatedLocalSize[] = {
    0x07230203u,
    0x00010600u,
    0u,
    8u,
    0u,
    // Declares OpCapability Shader.
    0x00020011u,
    1u,
    // Declares OpCapability PhysicalStorageBufferAddresses.
    0x00020011u,
    5347u,
    // Declares OpMemoryModel PhysicalStorageBuffer64 GLSL450.
    0x0003000eu,
    5348u,
    1u,
    // Declares OpEntryPoint GLCompute %1 "main".
    0x0005000fu,
    5u,
    1u,
    0x6e69616du,
    0u,
    // Declares truncated OpExecutionMode %1 LocalSize 1.
    0x00040010u,
    1u,
    17u,
    1u,
};

class BdaSpirvTest : public CtsTestBase<> {
 protected:
  static constexpr iree_host_size_t kElementCount = 4;
  static constexpr iree_device_size_t kDispatchByteLength =
      kElementCount * sizeof(int32_t);

  void SetUp() override {
    CtsTestBase::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }

    dispatch_queue_ =
        QueueForCommandCategories(IREE_HAL_COMMAND_CATEGORY_DISPATCH);
    if (!dispatch_queue_) {
      GTEST_SKIP() << "device has no dispatch-capable queue";
    }
    const iree_hal_executable_target_selection_result_t target_result =
        SelectExecutableTarget(IREE_SV("spirv"), IREE_SV("vulkan1.3+bda"));
    ASSERT_EQ(IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_SELECTED,
              target_result.outcome);
    executable_target_ = target_result.target;

    IREE_ASSERT_OK(
        PrepareBdaExecutable(BdaSpirvFixture(kBdaSpirvAdd7), &executable_));
  }

  void TearDown() override {
    iree_hal_executable_release(executable_);
    executable_ = nullptr;
    executable_target_ = nullptr;
    CtsTestBase::TearDown();
  }

  iree_status_t PrepareBdaExecutable(iree_const_byte_span_t executable_data,
                                     iree_hal_executable_load_flags_t flags,
                                     iree_hal_executable_t** out_executable) {
    return LoadExecutable(iree_hal_queue_family(dispatch_queue_),
                          executable_target_, flags, executable_data,
                          out_executable);
  }

  iree_status_t PrepareBdaExecutable(iree_const_byte_span_t executable_data,
                                     iree_hal_executable_t** out_executable) {
    return PrepareBdaExecutable(
        executable_data, IREE_HAL_EXECUTABLE_LOAD_FLAG_NONE, out_executable);
  }

  void CreateInputOutputBuffers(Ref<iree_hal_buffer_t>* input_buffer,
                                Ref<iree_hal_buffer_t>* output_buffer) {
    const int32_t input_data[kElementCount] = {1, -2, 30, 400};
    IREE_ASSERT_OK(CreateDeviceBufferWithData(input_data, sizeof(input_data),
                                              input_buffer->out()));
    IREE_ASSERT_OK(
        CreateZeroedDeviceBuffer(sizeof(input_data), output_buffer->out()));
  }

  static iree_hal_buffer_ref_list_t MakeBindings(
      iree_hal_buffer_t* input_buffer, iree_hal_buffer_t* output_buffer,
      iree_hal_buffer_ref_t binding_refs[2]) {
    binding_refs[0] = iree_hal_make_buffer_ref(
        input_buffer, /*offset=*/0, iree_hal_buffer_byte_length(input_buffer));
    binding_refs[1] =
        iree_hal_make_buffer_ref(output_buffer, /*offset=*/0,
                                 iree_hal_buffer_byte_length(output_buffer));
    return {
        .count = 2,
        .values = binding_refs,
    };
  }

  static std::vector<iree_hal_buffer_ref_t> MakeOversizedPublicationBindings(
      iree_hal_buffer_t* input_buffer, iree_hal_buffer_t* output_buffer) {
    static constexpr iree_host_size_t kDefaultPublicationBlockLength =
        64 * 1024;
    static constexpr iree_host_size_t kBindingCount =
        kDefaultPublicationBlockLength / sizeof(uint64_t) + 1;

    std::vector<iree_hal_buffer_ref_t> binding_refs(kBindingCount);
    binding_refs[0] = iree_hal_make_buffer_ref(
        input_buffer, /*offset=*/0, iree_hal_buffer_byte_length(input_buffer));
    binding_refs[1] =
        iree_hal_make_buffer_ref(output_buffer, /*offset=*/0,
                                 iree_hal_buffer_byte_length(output_buffer));
    for (iree_host_size_t i = 2; i < binding_refs.size(); ++i) {
      binding_refs[i] =
          iree_hal_make_buffer_ref(input_buffer, /*offset=*/0,
                                   iree_hal_buffer_byte_length(input_buffer));
    }
    return binding_refs;
  }

  void CreateIndirectDispatchCommandBuffer(
      uint32_t workgroup_count,
      Ref<iree_hal_command_buffer_t>* command_buffer) {
    iree_hal_buffer_ref_t binding_refs[2] = {
        iree_hal_make_indirect_buffer_ref(
            /*buffer_slot=*/0, /*offset=*/0,
            /*length=*/(iree_device_size_t)workgroup_count * sizeof(int32_t)),
        iree_hal_make_indirect_buffer_ref(
            /*buffer_slot=*/1, /*offset=*/0,
            /*length=*/(iree_device_size_t)workgroup_count * sizeof(int32_t)),
    };
    iree_hal_buffer_ref_list_t bindings = {
        .count = IREE_ARRAYSIZE(binding_refs),
        .values = binding_refs,
    };

    IREE_ASSERT_OK(CreateCommandBuffer(IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
                                       IREE_HAL_COMMAND_CATEGORY_DISPATCH,
                                       /*binding_capacity=*/2,
                                       command_buffer->out()));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer->get()));
    IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
        command_buffer->get(), executable_,
        iree_hal_executable_function_from_index(0),
        iree_hal_make_static_dispatch_config(workgroup_count, 1, 1),
        iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer->get()));
  }

  static iree_hal_buffer_binding_table_t MakeBindingTable(
      iree_hal_buffer_t* input_buffer, iree_hal_buffer_t* output_buffer,
      iree_hal_buffer_binding_t binding_table_entries[2]) {
    binding_table_entries[0] = {
        /*buffer=*/input_buffer,
        /*offset=*/0,
        /*length=*/iree_hal_buffer_byte_length(input_buffer),
    };
    binding_table_entries[1] = {
        /*buffer=*/output_buffer,
        /*offset=*/0,
        /*length=*/iree_hal_buffer_byte_length(output_buffer),
    };
    return {
        .count = 2,
        .bindings = binding_table_entries,
    };
  }

  void ExpectOutput(iree_hal_buffer_t* output_buffer) {
    std::vector<int32_t> output_data = ReadBufferData<int32_t>(output_buffer);
    EXPECT_THAT(output_data, ContainerEq(std::vector<int32_t>{8, 5, 37, 407}));
  }

  void ExpectFilledOutputPrefix(iree_hal_buffer_t* output_buffer,
                                int32_t expected_value) {
    std::vector<uint8_t> bytes =
        ReadBufferBytes(output_buffer, /*offset=*/0, kDispatchByteLength);
    ASSERT_EQ(kDispatchByteLength, bytes.size());
    std::vector<int32_t> output_data(kElementCount);
    std::memcpy(output_data.data(), bytes.data(), kDispatchByteLength);
    EXPECT_THAT(output_data, ContainerEq(std::vector<int32_t>(kElementCount,
                                                              expected_value)));
  }

  iree_hal_vulkan_native_replay_cache_stats_t NativeReplayCacheStats() {
    iree_hal_vulkan_native_replay_cache_stats_t stats;
    iree_hal_vulkan_logical_device_sample_native_replay_cache_stats(device_,
                                                                    &stats);
    return stats;
  }

  iree_hal_vulkan_bda_publication_cache_stats_t BdaPublicationCacheStats() {
    iree_hal_vulkan_bda_publication_cache_stats_t stats;
    iree_hal_vulkan_logical_device_sample_bda_publication_cache_stats(device_,
                                                                      &stats);
    return stats;
  }

  void ExecuteCommandBufferAndWait(
      iree_hal_command_buffer_t* command_buffer,
      iree_hal_buffer_binding_table_t binding_table) {
    SemaphoreList execute_signal(device_, {0}, {1});
    iree_hal_queue_t* queue = QueueForCommandBuffer(command_buffer);
    ASSERT_NE(nullptr, queue);
    IREE_ASSERT_OK(iree_hal_queue_execute(
        queue, iree_hal_semaphore_list_empty(), execute_signal, command_buffer,
        binding_table, IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        execute_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  }

  void TestEmptyGridDispatches(bool capture_profile) {
    Ref<iree_hal_buffer_t> input_buffer;
    Ref<iree_hal_buffer_t> output_buffer;
    CreateInputOutputBuffers(&input_buffer, &output_buffer);
    enum class Mode { kQueue, kRecorded, kBindingTable };
    for (auto mode : {Mode::kQueue, Mode::kRecorded, Mode::kBindingTable}) {
      SCOPED_TRACE(static_cast<int>(mode));
      for (uint32_t zero_axes = 1; zero_axes < 8; ++zero_axes) {
        SCOPED_TRACE(zero_axes);
        const auto config = iree_hal_make_static_dispatch_config(
            zero_axes & 1 ? 0 : 1, zero_axes & 2 ? 0 : 1,
            zero_axes & 4 ? 0 : 1);
        iree_hal_buffer_ref_t refs[2] = {
            iree_hal_make_buffer_ref(input_buffer, 0, 0),
            iree_hal_make_buffer_ref(output_buffer, 0, 0),
        };
        const iree_hal_buffer_binding_t entries[2] = {{input_buffer, 0, 0},
                                                      {output_buffer, 0, 0}};
        auto table = iree_hal_buffer_binding_table_empty();
        if (mode == Mode::kBindingTable) {
          refs[0] = iree_hal_make_indirect_buffer_ref(0, 0, 0);
          refs[1] = iree_hal_make_indirect_buffer_ref(1, 0, 0);
          table = {2, entries};
        }
        const iree_hal_buffer_ref_list_t bindings = {2, refs};
        Ref<iree_hal_command_buffer_t> commands;
        if (mode != Mode::kQueue) {
          IREE_ASSERT_OK(CreateCommandBuffer(
              IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
              IREE_HAL_COMMAND_CATEGORY_DISPATCH,
              mode == Mode::kBindingTable ? 2 : 0, commands.out()));
          IREE_ASSERT_OK(iree_hal_command_buffer_begin(commands));
          IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
              commands, executable_, iree_hal_executable_function_from_index(0),
              config, iree_const_byte_span_empty(), bindings,
              IREE_HAL_DISPATCH_FLAG_NONE));
          IREE_ASSERT_OK(iree_hal_command_buffer_end(commands));
        }

        TestProfileSink sink = {};
        TestProfileSinkInitialize(&sink);
        for (int axis = 0; axis < 3; ++axis) {
          sink.expected_workgroup_count[axis] = config.workgroup_count[axis];
        }
        if (commands) {
          sink.expected_dispatch_flags =
              IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_COMMAND_BUFFER;
          sink.expected_dispatch_command_indices = {0, 0};
        }
        DeviceProfilingScope profiling(device_);
        if (capture_profile) {
          iree_status_t status =
              profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS,
                              TestProfileSinkAsBase(&sink));
          if (IsProfilingUnsupported(iree_status_code(status))) {
            iree_status_ignore(status);
            GTEST_SKIP() << "Vulkan dispatch profiling is unavailable";
          }
          IREE_ASSERT_OK(status);
        }

        // Replay the same command buffer after completion to exercise native
        // replay caching as well as initial BDA publication.
        for (int iteration = 0; iteration < 2; ++iteration) {
          SemaphoreList gate(device_, {0}, {1});
          SemaphoreList completed(device_, {0}, {1});
          if (commands) {
            IREE_ASSERT_OK(iree_hal_queue_execute(
                QueueForCommandBuffer(commands), gate, completed, commands,
                table, IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
          } else {
            IREE_ASSERT_OK(iree_hal_queue_dispatch(
                dispatch_queue_, gate, completed, executable_,
                iree_hal_executable_function_from_index(0), config,
                iree_const_byte_span_empty(), bindings, /*barriers=*/nullptr,
                IREE_HAL_DISPATCH_FLAG_NONE));
          }
          uint64_t value = 0;
          IREE_ASSERT_OK(
              iree_hal_semaphore_query(completed.semaphores[0], &value));
          EXPECT_EQ(0u, value);
          IREE_ASSERT_OK(iree_hal_semaphore_list_signal(gate, nullptr));
          IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
              completed, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
        }
        if (capture_profile) {
          IREE_ASSERT_OK(iree_hal_device_profiling_flush(device_));
          IREE_ASSERT_OK(profiling.End());
          // The common HAL recorder elides an all-zero grid before reaching
          // the driver. Other empty shapes remain recorded dispatch events.
          EXPECT_EQ(commands && zero_axes == 7 ? 0u : 2u,
                    sink.dispatch_events.size());
        }
        EXPECT_THAT(ReadBufferData<int32_t>(output_buffer),
                    ContainerEq(std::vector<int32_t>(kElementCount, 0)));
      }
    }
  }

  iree_hal_queue_t* dispatch_queue_ = nullptr;
  const iree_hal_executable_target_t* executable_target_ = nullptr;
  iree_hal_executable_t* executable_ = nullptr;
};

TEST_P(BdaSpirvTest, EmptyGridsAcceptEmptyBindings) {
  TestEmptyGridDispatches(/*capture_profile=*/false);
}

TEST_P(BdaSpirvTest, EmptyGridsPreserveDispatchProfiling) {
  TestEmptyGridDispatches(/*capture_profile=*/true);
}

static void ExpectNativeBindingRange(iree_hal_buffer_t* root,
                                     iree_hal_buffer_t* view,
                                     iree_device_size_t offset) {
  const iree_hal_buffer_native_binding_slot_t resource_slot = {
      IREE_HAL_VULKAN_BUFFER_BINDING_RESOURCE,
      IREE_HAL_BUFFER_INTERFACE_VULKAN_BUFFER};
  const iree_hal_buffer_native_binding_slot_t address_slot = {
      IREE_HAL_VULKAN_BUFFER_BINDING_DEVICE_ADDRESS,
      IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS};
  const auto root_resource =
      iree_hal_buffer_native_binding(root, resource_slot).vulkan;
  const auto view_resource =
      iree_hal_buffer_native_binding(view, resource_slot).vulkan;
  const uint64_t root_address =
      iree_hal_buffer_native_binding(root, address_slot).device_address;
  const uint64_t view_address =
      iree_hal_buffer_native_binding(view, address_slot).device_address;
  ASSERT_NE(root_resource.buffer, 0u);
  ASSERT_NE(root_address, 0u);
  EXPECT_EQ(view->memory.bindings, root->memory.bindings);
  EXPECT_EQ(view_resource.buffer, root_resource.buffer);
  EXPECT_EQ(view_resource.offset, root_resource.offset + offset);
  EXPECT_EQ(view_address, root_address + offset);

  VkDeviceAddress packet_address = 0;
  IREE_ASSERT_OK(iree_hal_vulkan_buffer_device_address(view, &packet_address));
  EXPECT_EQ(packet_address, view_address);
  iree_hal_buffer_t* backing = nullptr;
  IREE_ASSERT_OK(iree_hal_vulkan_buffer_resolve_backing(view, &backing));
  iree_device_size_t packet_offset = 0;
  IREE_ASSERT_OK(iree_hal_vulkan_buffer_resolve_backing_offset(view, backing, 0,
                                                               &packet_offset));
  EXPECT_EQ(packet_offset, view_resource.offset);
}

TEST_P(BdaSpirvTest, OwnedPoolHonorsOveralignedDeviceAddresses) {
  constexpr uint64_t kDeviceAddressInterface =
      UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS;
  std::array<iree_hal_pool_family_access_t, 2> families = {};
  families[0].family = iree_hal_queue_family(transfer_queue_);
  families[0].usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  families[0].interfaces = kDeviceAddressInterface;
  families[1].family = iree_hal_queue_family(dispatch_queue_);
  families[1].usage = IREE_HAL_BUFFER_USAGE_STORAGE;
  families[1].interfaces = kDeviceAddressInterface;
  iree_hal_pool_scope_t scope = {};
  scope.family_count = families[0].family == families[1].family ? 1 : 2;
  if (scope.family_count == 1) {
    families[0].usage |= families[1].usage;
  }
  scope.families = families.data();

  iree_hal_slab_pool_options_t options;
  iree_hal_slab_pool_options_initialize(&options);
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(
      device_group_, scope, &options, iree_allocator_system(), source.out()));
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(source, &capabilities);
  ASSERT_GE(capabilities.max_allocation_alignment, 4096u);

  const iree_hal_memory_site_t site = {
      IREE_HAL_MEMORY_SITE_PROGRAM,
      iree_hal_queue_family(dispatch_queue_),
  };
  iree_hal_memory_scope_t memory_scope;
  IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(device_group_, site,
                                                            &memory_scope));
  iree_hal_buffer_native_binding_slot_t address_slot;
  IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
      source, memory_scope, IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
      &address_slot));

  iree_device_size_t native_alignment = 0;
  {
    Ref<iree_hal_buffer_t> buffer;
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        source, iree_hal_make_pool_reservation_request(4, 1),
        iree_infinite_timeout(), buffer.out()));
    const auto memory = iree_hal_buffer_memory_view(buffer);
    ASSERT_NE(memory.backing, nullptr);
    native_alignment = memory.backing->allocation_alignment;
    ASSERT_TRUE(iree_device_size_is_power_of_two(native_alignment));
  }

  const std::array<uint32_t, 5> input = {11, 22, 33, 44, 55};
  for (iree_device_size_t alignment : {256u, 4096u}) {
    SCOPED_TRACE(alignment);
    Ref<iree_hal_buffer_t> buffer;
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        source,
        iree_hal_make_pool_reservation_request(sizeof(input), alignment),
        iree_infinite_timeout(), buffer.out()));
    const auto memory = iree_hal_buffer_memory_view(buffer);
    ASSERT_NE(memory.backing, nullptr);
    EXPECT_GE(memory.backing->allocation_alignment, alignment);
    const uint64_t address =
        iree_hal_buffer_native_binding(buffer, address_slot).device_address;
    ASSERT_NE(address, 0u);
    EXPECT_EQ(address % alignment, 0u);
    const uint64_t raw_address =
        iree_hal_vulkan_buffer_native(buffer)
            ->bindings[IREE_HAL_VULKAN_BUFFER_BINDING_DEVICE_ADDRESS]
            .device_address;
    EXPECT_LT(memory.binding_offset, alignment);
    EXPECT_EQ(address, raw_address + memory.binding_offset);

    Ref<iree_hal_buffer_t> view;
    IREE_ASSERT_OK(
        iree_hal_buffer_subspan(buffer, sizeof(uint32_t), 3 * sizeof(uint32_t),
                                iree_allocator_system(), view.out()));
    const auto view_memory = iree_hal_buffer_memory_view(view);
    EXPECT_EQ(view_memory.binding_offset,
              memory.binding_offset + sizeof(uint32_t));
    EXPECT_EQ(iree_hal_buffer_native_binding(view, address_slot).device_address,
              address + sizeof(uint32_t));

    SemaphoreList uploaded(device_, {0}, {1});
    IREE_ASSERT_OK(iree_hal_queue_upload(
        transfer_queue_, iree_hal_semaphore_list_empty(), uploaded,
        input.data(), buffer, 0, sizeof(input), /*barriers=*/NULL));
    std::array<uint32_t, 3> output = {};
    SemaphoreList downloaded(device_, {0}, {1});
    IREE_ASSERT_OK(iree_hal_queue_download(transfer_queue_, uploaded,
                                           downloaded, view, 0, output.data(),
                                           sizeof(output), /*barriers=*/NULL));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        downloaded, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    EXPECT_THAT(output, ContainerEq(std::array<uint32_t, 3>{22, 33, 44}));
  }

  Ref<iree_hal_buffer_t> invalid_buffer;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_pool_allocate_buffer(
          source, iree_hal_make_pool_reservation_request(4, 3),
          iree_infinite_timeout(), invalid_buffer.out()));
  ASSERT_LE(native_alignment, IREE_DEVICE_SIZE_MAX / 2);
  const iree_device_size_t overflow_alignment =
      std::max<iree_device_size_t>(256, native_alignment * 2);
  ASSERT_LE(overflow_alignment, capabilities.max_allocation_alignment);
  const iree_device_size_t allocation_granule =
      std::max<iree_device_size_t>(4, native_alignment);
  const iree_device_size_t overflow_size =
      IREE_DEVICE_SIZE_MAX & ~(allocation_granule - 1);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_pool_allocate_buffer(
                            source,
                            iree_hal_make_pool_reservation_request(
                                overflow_size, overflow_alignment),
                            iree_infinite_timeout(), invalid_buffer.out()));
}

TEST_P(BdaSpirvTest, PrepareRejectsDescriptorDecoratedBdaSpirv) {
  iree_hal_executable_t* decorated_executable = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      PrepareBdaExecutable(BdaSpirvFixture(kBdaSpirvAdd7DescriptorDecorated),
                           &decorated_executable));
  EXPECT_EQ(nullptr, decorated_executable);
  iree_hal_executable_release(decorated_executable);
}

TEST_P(BdaSpirvTest,
       PrepareRejectsBdaSpirvWithoutPhysicalStorageBufferAddresses) {
  iree_hal_executable_t* executable = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      PrepareBdaExecutable(
          BdaSpirvFixture(kBdaSpirvMissingPhysicalStorageBufferAddresses),
          &executable));
  EXPECT_EQ(nullptr, executable);
  iree_hal_executable_release(executable);
}

TEST_P(BdaSpirvTest, PrepareRejectsBdaSpirvWithoutMetadata) {
  iree_hal_executable_t* executable = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      PrepareBdaExecutable(BdaSpirvFixture(kBdaSpirvAdd7NoMetadata),
                           &executable));
  EXPECT_EQ(nullptr, executable);
  iree_hal_executable_release(executable);
}

TEST_P(BdaSpirvTest, PrepareRejectsMalformedBdaSpirvHeader) {
  const uint32_t not_spirv[5] = {0};
  iree_hal_executable_t* executable = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        PrepareBdaExecutable(iree_make_const_byte_span(
                                                 not_spirv, sizeof(not_spirv)),
                                             &executable));
  EXPECT_EQ(nullptr, executable);
  iree_hal_executable_release(executable);
}

TEST_P(BdaSpirvTest, PrepareRejectsBdaSpirvWithoutComputeEntryPoints) {
  iree_hal_executable_t* executable = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      PrepareBdaExecutable(
          iree_make_const_byte_span(kBdaSpirvWithoutComputeEntryPoints,
                                    sizeof(kBdaSpirvWithoutComputeEntryPoints)),
          &executable));
  EXPECT_EQ(nullptr, executable);
  iree_hal_executable_release(executable);
}

TEST_P(BdaSpirvTest, PrepareRejectsBdaSpirvWithDuplicateEntryNames) {
  iree_hal_executable_t* executable = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      PrepareBdaExecutable(iree_make_const_byte_span(
                               kBdaSpirvWithDuplicateComputeEntryNames,
                               sizeof(kBdaSpirvWithDuplicateComputeEntryNames)),
                           &executable));
  EXPECT_EQ(nullptr, executable);
  iree_hal_executable_release(executable);
}

TEST_P(BdaSpirvTest, PrepareRejectsBdaSpirvWithTruncatedLocalSize) {
  iree_hal_executable_t* executable = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      PrepareBdaExecutable(
          iree_make_const_byte_span(kBdaSpirvWithTruncatedLocalSize,
                                    sizeof(kBdaSpirvWithTruncatedLocalSize)),
          &executable));
  EXPECT_EQ(nullptr, executable);
  iree_hal_executable_release(executable);
}

TEST_P(BdaSpirvTest, PrepareRejectsBdaSpirvWithoutPushConstantRoot) {
  iree_hal_executable_t* executable = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      PrepareBdaExecutable(BdaSpirvFixture(kBdaSpirvMissingPushConstantRoot),
                           &executable));
  EXPECT_EQ(nullptr, executable);
  iree_hal_executable_release(executable);
}

TEST_P(BdaSpirvTest, PrepareRejectsBdaNoopWithoutVerificationDisabled) {
  iree_hal_executable_t* executable = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      PrepareBdaExecutable(
          BdaSpirvFixture(kBdaSpirvNoopWithoutPushConstantRoot), &executable));
  EXPECT_EQ(nullptr, executable);
  iree_hal_executable_release(executable);
}

TEST_P(BdaSpirvTest, QueueDispatchExecutesUnverifiedBdaNoop) {
  Ref<iree_hal_executable_t> executable;
  IREE_ASSERT_OK(PrepareBdaExecutable(
      BdaSpirvFixture(kBdaSpirvNoopWithoutPushConstantRoot),
      IREE_HAL_EXECUTABLE_LOAD_FLAG_DISABLE_VERIFICATION, executable.out()));

  iree_hal_buffer_ref_list_t bindings = iree_hal_buffer_ref_list_empty();
  SemaphoreList dispatch_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      dispatch_queue_, iree_hal_semaphore_list_empty(), dispatch_signal,
      executable.get(), iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1),
      iree_const_byte_span_empty(), bindings, /*barriers=*/NULL,
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
}

TEST_P(BdaSpirvTest, PrepareRejectsBdaSpirvWithDescriptorVariable) {
  iree_hal_executable_t* executable = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      PrepareBdaExecutable(BdaSpirvFixture(kBdaSpirvDescriptorStorageVariable),
                           &executable));
  EXPECT_EQ(nullptr, executable);
  iree_hal_executable_release(executable);
}

class BdaMemoryReservation {
 public:
  BdaMemoryReservation() = default;
  BdaMemoryReservation(const BdaMemoryReservation&) = delete;
  BdaMemoryReservation& operator=(const BdaMemoryReservation&) = delete;

  ~BdaMemoryReservation() { iree_hal_buffer_release(reservation_); }

  iree_hal_buffer_t* owner() const { return reservation_; }
  iree_hal_buffer_t** out() { return &reservation_; }
  iree_hal_buffer_t* get() const { return (reservation_); }

 private:
  // Explicit address owner, closed after all submitted accesses have completed.
  iree_hal_buffer_t* reservation_ = nullptr;
};

TEST_P(BdaSpirvTest, QueueDispatchExecutesBdaShader) {
  Ref<iree_hal_buffer_t> input_buffer;
  Ref<iree_hal_buffer_t> output_buffer;
  CreateInputOutputBuffers(&input_buffer, &output_buffer);

  iree_hal_buffer_ref_t binding_refs[2];
  iree_hal_buffer_ref_list_t bindings =
      MakeBindings(input_buffer, output_buffer, binding_refs);

  SemaphoreList dispatch_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      dispatch_queue_, iree_hal_semaphore_list_empty(), dispatch_signal,
      executable_, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(4, 1, 1),
      iree_const_byte_span_empty(), bindings, /*barriers=*/NULL,
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  ExpectOutput(output_buffer);
}

TEST_P(BdaSpirvTest, RejectsUnsupportedSubgroupSize) {
  Ref<iree_hal_executable_t> executable;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      PrepareBdaExecutable(BdaSpirvFixture("bda_subgroup_unsupported.spv"),
                           executable.out()));
  EXPECT_EQ(executable.get(), nullptr);
}

TEST_P(BdaSpirvTest, QueueDispatchHonorsPerEntrySubgroupSizes) {
  const auto* dispatch =
      iree_hal_device_spec_dispatch(iree_hal_device_spec(device_));
  ASSERT_NE(dispatch, nullptr);
  if (dispatch->subgroup.minimum_size > 32 ||
      dispatch->subgroup.maximum_size < 64) {
    GTEST_SKIP()
        << "device does not support both 32- and 64-lane compute subgroups";
  }
  Ref<iree_hal_executable_t> executable;
  IREE_ASSERT_OK(PrepareBdaExecutable(BdaSpirvFixture("bda_subgroup_sizes.spv"),
                                      executable.out()));
  for (uint32_t subgroup_size : {32u, 64u}) {
    SCOPED_TRACE(subgroup_size);
    Ref<iree_hal_buffer_t> output;
    IREE_ASSERT_OK(
        CreateZeroedDeviceBuffer(128 * sizeof(uint32_t), output.out()));
    const iree_hal_buffer_ref_t binding =
        iree_hal_make_buffer_ref(output, 0, IREE_HAL_WHOLE_BUFFER);
    const iree_hal_buffer_ref_list_t bindings = {1, &binding};
    iree_hal_executable_function_t function =
        iree_hal_executable_function_invalid();
    IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
        executable, subgroup_size == 32 ? IREE_SV("wave32") : IREE_SV("wave64"),
        &function));
    SemaphoreList completed(device_, {0}, {1});
    IREE_ASSERT_OK(iree_hal_queue_dispatch(
        dispatch_queue_, iree_hal_semaphore_list_empty(), completed, executable,
        function, iree_hal_make_static_dispatch_config(1, 1, 1),
        iree_const_byte_span_empty(), bindings, /*barriers=*/NULL,
        IREE_HAL_DISPATCH_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        completed, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    EXPECT_THAT(ReadBufferData<uint32_t>(output),
                ContainerEq(std::vector<uint32_t>(128, subgroup_size)));
  }
}

TEST_P(BdaSpirvTest, QueueDispatchExecutesBdaShaderWithMetadata) {
  Ref<iree_hal_executable_t> executable;
  IREE_ASSERT_OK(PrepareBdaExecutable(BdaSpirvFixture(kBdaSpirvAdd7Bindings2),
                                      executable.out()));

  Ref<iree_hal_buffer_t> input_buffer;
  Ref<iree_hal_buffer_t> output_buffer;
  CreateInputOutputBuffers(&input_buffer, &output_buffer);

  iree_hal_buffer_ref_t binding_refs[2];
  iree_hal_buffer_ref_list_t bindings =
      MakeBindings(input_buffer, output_buffer, binding_refs);

  SemaphoreList dispatch_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      dispatch_queue_, iree_hal_semaphore_list_empty(), dispatch_signal,
      executable.get(), iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(4, 1, 1),
      iree_const_byte_span_empty(), bindings, /*barriers=*/NULL,
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  ExpectOutput(output_buffer);
}

TEST_P(BdaSpirvTest, FunctionReflectionOwnsLogicalParameterMetadata) {
  const iree_const_byte_span_t fixture =
      BdaSpirvFixture(kBdaSpirvAdd7ConstantLength4);
  std::vector<uint8_t> mutable_artifact(fixture.data,
                                        fixture.data + fixture.data_length);
  Ref<iree_hal_executable_t> executable;
  IREE_ASSERT_OK(
      PrepareBdaExecutable(iree_make_const_byte_span(mutable_artifact.data(),
                                                     mutable_artifact.size()),
                           executable.out()));
  std::fill(mutable_artifact.begin(), mutable_artifact.end(), 0);

  iree_hal_executable_function_t function =
      iree_hal_executable_function_invalid();
  IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
      executable.get(), IREE_SV("main"), &function));
  iree_hal_executable_function_info_t info = {};
  IREE_ASSERT_OK(
      iree_hal_executable_function_info(executable.get(), function, &info));
  EXPECT_TRUE(iree_string_view_equal(info.name, IREE_SV("main")));
  EXPECT_EQ(4u, info.constant_byte_length);
  EXPECT_EQ(2u, info.binding_count);
  ASSERT_EQ(3u, info.parameter_count);

  std::array<iree_hal_executable_function_parameter_t, 3> parameters = {};
  IREE_ASSERT_OK(iree_hal_executable_function_parameters(
      executable.get(), function, parameters.size(), parameters.data()));
  EXPECT_EQ(IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_CONSTANT,
            parameters[0].type);
  EXPECT_EQ(0u, parameters[0].offset);
  EXPECT_EQ(4u, parameters[0].size);
  EXPECT_EQ(IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_FLAG_NATIVE_ABI_OFFSET,
            parameters[0].flags);
  EXPECT_EQ(32u, parameters[0].native_abi_offset);
  EXPECT_TRUE(iree_string_view_is_empty(parameters[0].name));
  for (iree_host_size_t i = 1; i < parameters.size(); ++i) {
    EXPECT_EQ(IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BINDING,
              parameters[i].type);
    EXPECT_EQ(i - 1, parameters[i].offset);
    EXPECT_EQ(IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_FLAG_NONE,
              parameters[i].flags);
    EXPECT_TRUE(iree_string_view_is_empty(parameters[i].name));
  }
}

TEST_P(BdaSpirvTest, QueueDispatchRejectsBdaMetadataBindingMismatch) {
  Ref<iree_hal_executable_t> executable;
  IREE_ASSERT_OK(PrepareBdaExecutable(BdaSpirvFixture(kBdaSpirvAdd7Bindings2),
                                      executable.out()));

  Ref<iree_hal_buffer_t> input_buffer;
  Ref<iree_hal_buffer_t> output_buffer;
  CreateInputOutputBuffers(&input_buffer, &output_buffer);

  iree_hal_buffer_ref_t binding_refs[1] = {
      iree_hal_make_buffer_ref(input_buffer.get(), /*offset=*/0,
                               iree_hal_buffer_byte_length(input_buffer.get())),
  };
  iree_hal_buffer_ref_list_t bindings = {
      .count = IREE_ARRAYSIZE(binding_refs),
      .values = binding_refs,
  };

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_queue_dispatch(dispatch_queue_, iree_hal_semaphore_list_empty(),
                              iree_hal_semaphore_list_empty(), executable.get(),
                              iree_hal_executable_function_from_index(0),
                              iree_hal_make_static_dispatch_config(4, 1, 1),
                              iree_const_byte_span_empty(), bindings,
                              /*barriers=*/NULL, IREE_HAL_DISPATCH_FLAG_NONE));
}

TEST_P(BdaSpirvTest, QueueDispatchRejectsBdaMetadataBindingLength) {
  Ref<iree_hal_executable_t> executable;
  IREE_ASSERT_OK(PrepareBdaExecutable(
      BdaSpirvFixture(kBdaSpirvAdd7Binding1Length4_17), executable.out()));

  Ref<iree_hal_buffer_t> input_buffer;
  Ref<iree_hal_buffer_t> output_buffer;
  CreateInputOutputBuffers(&input_buffer, &output_buffer);

  iree_hal_buffer_ref_t binding_refs[2];
  iree_hal_buffer_ref_list_t bindings =
      MakeBindings(input_buffer, output_buffer, binding_refs);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_queue_dispatch(dispatch_queue_, iree_hal_semaphore_list_empty(),
                              iree_hal_semaphore_list_empty(), executable.get(),
                              iree_hal_executable_function_from_index(0),
                              iree_hal_make_static_dispatch_config(4, 1, 1),
                              iree_const_byte_span_empty(), bindings,
                              /*barriers=*/NULL, IREE_HAL_DISPATCH_FLAG_NONE));
}

TEST_P(BdaSpirvTest, QueueDispatchUsesBdaMetadataConstantLength) {
  Ref<iree_hal_executable_t> executable;
  IREE_ASSERT_OK(PrepareBdaExecutable(
      BdaSpirvFixture(kBdaSpirvAdd7ConstantLength4), executable.out()));

  Ref<iree_hal_buffer_t> input_buffer;
  Ref<iree_hal_buffer_t> output_buffer;
  CreateInputOutputBuffers(&input_buffer, &output_buffer);

  iree_hal_buffer_ref_t binding_refs[2];
  iree_hal_buffer_ref_list_t bindings =
      MakeBindings(input_buffer, output_buffer, binding_refs);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_queue_dispatch(dispatch_queue_, iree_hal_semaphore_list_empty(),
                              iree_hal_semaphore_list_empty(), executable.get(),
                              iree_hal_executable_function_from_index(0),
                              iree_hal_make_static_dispatch_config(4, 1, 1),
                              iree_const_byte_span_empty(), bindings,
                              /*barriers=*/NULL, IREE_HAL_DISPATCH_FLAG_NONE));

  const uint32_t ignored_constant = 123u;
  SemaphoreList dispatch_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      dispatch_queue_, iree_hal_semaphore_list_empty(), dispatch_signal,
      executable.get(), iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(4, 1, 1),
      iree_make_const_byte_span(&ignored_constant, sizeof(ignored_constant)),
      bindings, /*barriers=*/NULL, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  ExpectOutput(output_buffer);
}

TEST_P(BdaSpirvTest, QueueDispatchHandlesOversizedBdaPublication) {
  Ref<iree_hal_buffer_t> input_buffer;
  Ref<iree_hal_buffer_t> output_buffer;
  CreateInputOutputBuffers(&input_buffer, &output_buffer);

  std::vector<iree_hal_buffer_ref_t> binding_refs =
      MakeOversizedPublicationBindings(input_buffer, output_buffer);
  iree_hal_buffer_ref_list_t bindings = {
      .count = binding_refs.size(),
      .values = binding_refs.data(),
  };

  SemaphoreList dispatch_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      dispatch_queue_, iree_hal_semaphore_list_empty(), dispatch_signal,
      executable_, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(4, 1, 1),
      iree_const_byte_span_empty(), bindings, /*barriers=*/NULL,
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  ExpectOutput(output_buffer);
}

TEST_P(BdaSpirvTest, CommandBufferExecutesBdaShader) {
  Ref<iree_hal_buffer_t> input_buffer;
  Ref<iree_hal_buffer_t> output_buffer;
  CreateInputOutputBuffers(&input_buffer, &output_buffer);

  iree_hal_buffer_ref_t binding_refs[2];
  iree_hal_buffer_ref_list_t bindings =
      MakeBindings(input_buffer, output_buffer, binding_refs);

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(CreateCommandBuffer(
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT, IREE_HAL_COMMAND_CATEGORY_DISPATCH,
      /*binding_capacity=*/0, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable_, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(4, 1, 1),
      iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  const uint64_t initial_one_shot_bypass_count =
      NativeReplayCacheStats().one_shot_bypass_count;
  IREE_ASSERT_OK(SubmitCommandBufferAndWait(command_buffer));
  ExpectOutput(output_buffer);
  EXPECT_GE(NativeReplayCacheStats().one_shot_bypass_count -
                initial_one_shot_bypass_count,
            1);
}

TEST_P(BdaSpirvTest, CommandBufferCachesBdaPublicationRequirements) {
  Ref<iree_hal_buffer_t> input_buffer;
  Ref<iree_hal_buffer_t> output_buffer;
  CreateInputOutputBuffers(&input_buffer, &output_buffer);

  iree_hal_buffer_ref_t binding_refs[2];
  iree_hal_buffer_ref_list_t bindings =
      MakeBindings(input_buffer.get(), output_buffer.get(), binding_refs);
  std::vector<iree_hal_buffer_ref_t> oversized_bindings =
      MakeOversizedPublicationBindings(input_buffer.get(), output_buffer.get());

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(CreateCommandBuffer(
      IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT, IREE_HAL_COMMAND_CATEGORY_DISPATCH,
      /*binding_capacity=*/0, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable_, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1),
      iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable_, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1),
      iree_const_byte_span_empty(),
      iree_hal_buffer_ref_list_t{
          .count = oversized_bindings.size(),
          .values = oversized_bindings.data(),
      },
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  EXPECT_TRUE(
      iree_hal_vulkan_command_buffer_has_native_commands(command_buffer));
  EXPECT_EQ(iree_hal_vulkan_command_buffer_dispatch_count(command_buffer), 2u);

  iree_hal_vulkan_command_buffer_descriptor_requirements_t requirements = {0};
  IREE_ASSERT_OK(
      iree_hal_vulkan_command_buffer_native_descriptor_pool_requirements(
          command_buffer, &requirements));
  EXPECT_EQ(requirements.set_count, 0u);
  EXPECT_EQ(requirements.sampler_count, 0u);
  EXPECT_EQ(requirements.uniform_buffer_count, 0u);
  EXPECT_EQ(requirements.storage_buffer_count, 0u);

  iree_device_size_t bda_publication_length = 0;
  IREE_ASSERT_OK(iree_hal_vulkan_command_buffer_native_bda_publication_length(
      command_buffer, &bda_publication_length));
  EXPECT_EQ(bda_publication_length,
            (bindings.count + oversized_bindings.size()) * sizeof(uint64_t));
}

TEST_P(BdaSpirvTest, TrimDropsIdleOversizedBdaPublicationBlock) {
  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  const uint64_t initial_block_count = BdaPublicationCacheStats().block_count;

  Ref<iree_hal_buffer_t> input_buffer;
  Ref<iree_hal_buffer_t> output_buffer;
  CreateInputOutputBuffers(&input_buffer, &output_buffer);
  std::vector<iree_hal_buffer_ref_t> oversized_bindings =
      MakeOversizedPublicationBindings(input_buffer.get(), output_buffer.get());

  SemaphoreList dispatch_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      dispatch_queue_, iree_hal_semaphore_list_empty(), dispatch_signal,
      executable_, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1),
      iree_const_byte_span_empty(),
      iree_hal_buffer_ref_list_t{
          .count = oversized_bindings.size(),
          .values = oversized_bindings.data(),
      },
      /*barriers=*/NULL, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  EXPECT_GT(BdaPublicationCacheStats().block_count, initial_block_count);
  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  EXPECT_LE(BdaPublicationCacheStats().block_count, initial_block_count);
}

TEST_P(BdaSpirvTest, CommandBufferProfilingBypassesCachedNativeReplay) {
  if (NativeReplayCacheStats().max_instance_count == 0) {
    GTEST_SKIP() << "Vulkan BDA native replay cache is disabled";
  }

  Ref<iree_hal_buffer_t> input_buffer;
  Ref<iree_hal_buffer_t> output_buffer;
  CreateInputOutputBuffers(&input_buffer, &output_buffer);

  Ref<iree_hal_command_buffer_t> command_buffer;
  CreateIndirectDispatchCommandBuffer(/*workgroup_count=*/kElementCount,
                                      &command_buffer);

  iree_hal_buffer_binding_t binding_table_entries[2];
  iree_hal_buffer_binding_table_t binding_table = MakeBindingTable(
      input_buffer.get(), output_buffer.get(), binding_table_entries);

  TestProfileSink sink = {};
  TestProfileSinkInitialize(&sink);
  sink.expected_dispatch_flags =
      IREE_HAL_PROFILE_DISPATCH_EVENT_FLAG_COMMAND_BUFFER;
  sink.expected_dispatch_command_indices = {0};
  sink.expected_workgroup_count[0] = kElementCount;
  sink.expected_workgroup_count[1] = 1;
  sink.expected_workgroup_count[2] = 1;
  DeviceProfilingScope profiling(device_);
  iree_status_t status =
      profiling.Begin(IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS,
                      TestProfileSinkAsBase(&sink));
  if (IsProfilingUnsupported(iree_status_code(status))) {
    iree_status_ignore(status);
    GTEST_SKIP() << "Vulkan dispatch profiling is unavailable";
  }
  IREE_ASSERT_OK(status);

  const uint64_t initial_profile_bypass_count =
      NativeReplayCacheStats().profile_bypass_count;
  const uint64_t initial_hit_count = NativeReplayCacheStats().hit_count;
  const uint64_t initial_create_count = NativeReplayCacheStats().create_count;

  ExecuteCommandBufferAndWait(command_buffer.get(), binding_table);
  ExpectOutput(output_buffer.get());

  IREE_ASSERT_OK(iree_hal_device_profiling_flush(device_));
  IREE_ASSERT_OK(profiling.End());

  EXPECT_GE(NativeReplayCacheStats().profile_bypass_count -
                initial_profile_bypass_count,
            1);
  EXPECT_EQ(initial_hit_count, NativeReplayCacheStats().hit_count);
  EXPECT_EQ(initial_create_count, NativeReplayCacheStats().create_count);
  EXPECT_GE(sink.dispatch_event_count, 1);
}

TEST_P(BdaSpirvTest, CommandBufferHandlesOversizedBdaPublication) {
  Ref<iree_hal_buffer_t> input_buffer;
  Ref<iree_hal_buffer_t> output_buffer;
  CreateInputOutputBuffers(&input_buffer, &output_buffer);

  std::vector<iree_hal_buffer_ref_t> binding_refs =
      MakeOversizedPublicationBindings(input_buffer, output_buffer);
  iree_hal_buffer_ref_list_t bindings = {
      .count = binding_refs.size(),
      .values = binding_refs.data(),
  };

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(CreateCommandBuffer(
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT, IREE_HAL_COMMAND_CATEGORY_DISPATCH,
      /*binding_capacity=*/0, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable_, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(4, 1, 1),
      iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  IREE_ASSERT_OK(SubmitCommandBufferAndWait(command_buffer));
  ExpectOutput(output_buffer);
}

TEST_P(BdaSpirvTest, NativeBindingsExecuteFromInteriorPoolRanges) {
  std::array<iree_hal_pool_family_access_t, 2> families = {};
  families[0].family = iree_hal_queue_family(transfer_queue_);
  families[0].usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  families[1].family = iree_hal_queue_family(dispatch_queue_);
  families[1].usage = IREE_HAL_BUFFER_USAGE_STORAGE |
                      IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS;
  iree_hal_pool_scope_t scope = {};
  scope.family_count = families[0].family == families[1].family ? 1 : 2;
  if (scope.family_count == 1) {
    families[0].usage |= families[1].usage;
  }
  scope.families = families.data();
  iree_hal_slab_pool_options_t source_options;
  iree_hal_slab_pool_options_initialize(&source_options);
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(
      iree_hal_slab_pool_create(device_group_, scope, &source_options,
                                iree_allocator_system(), source.out()));
  Ref<iree_hal_buffer_t> backing;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      source, iree_hal_make_pool_reservation_request(8192, /*min_alignment=*/1),
      iree_infinite_timeout(), backing.out()));
  iree_hal_tlsf_pool_options_t options = {};
  options.tlsf_options.frontier_capacity = 4;
  Ref<iree_hal_pool_t> arena_pool;
  IREE_ASSERT_OK(iree_hal_tlsf_pool_create_from_buffer(
      backing, 256, 7168, &options, iree_allocator_system(), arena_pool.out()));
  for (bool queued_arena : {false, true}) {
    SCOPED_TRACE(::testing::Message() << "queued arena: " << queued_arena);
    Ref<iree_hal_buffer_t> arena;
    if (queued_arena) {
      const iree_hal_pool_reservation_request_t request =
          iree_hal_make_pool_reservation_request(/*allocation_size=*/4096,
                                                 /*min_alignment=*/1);
      SemaphoreList allocated(device_, {0}, {1});
      IREE_ASSERT_OK(iree_hal_queue_alloca(
          transfer_queue_, iree_hal_semaphore_list_empty(), allocated,
          arena_pool, 1, &request, arena.out()));
      IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
          allocated, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    } else {
      IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
          arena_pool,
          iree_hal_make_pool_reservation_request(4096, /*min_alignment=*/1),
          iree_infinite_timeout(), arena.out()));
    }
    Ref<iree_hal_pool_t> pool;
    IREE_ASSERT_OK(iree_hal_tlsf_pool_create_from_buffer(
        arena, 256, 3584, &options, iree_allocator_system(), pool.out()));

    enum class DispatchMode {
      kQueue,
      kRecorded,
      kBindingTable,
      kIndirectParameters,
    };
    for (bool queued : {false, true}) {
      SCOPED_TRACE(queued);
      for (auto mode :
           {DispatchMode::kQueue, DispatchMode::kRecorded,
            DispatchMode::kBindingTable, DispatchMode::kIndirectParameters}) {
        SCOPED_TRACE(static_cast<int>(mode));
        const std::array<int32_t, 8> input = {11, 22, 1, -2, 30, 400, 77, 88};
        const std::array<int32_t, 8> initial_output = {11, 22, -1, -1,
                                                       -1, -1, 77, 88};
        const std::array<uint32_t, 3> workgroups = {4, 1, 1};
        std::array<Ref<iree_hal_buffer_t>, 3> roots;
        std::array<Ref<iree_hal_buffer_t>, 2> views;
        SemaphoreList gate(device_, {0}, {1});
        SemaphoreList allocated(device_, {0}, {1});
        if (queued) {
          std::array<iree_hal_pool_reservation_request_t, 3> requests = {};
          for (auto& request : requests) {
            request.allocation_size = sizeof(input);
            request.min_alignment = 1;
          }
          iree_hal_buffer_t* buffers[3] = {};
          IREE_ASSERT_OK(iree_hal_queue_alloca(transfer_queue_, gate, allocated,
                                               pool, requests.size(),
                                               requests.data(), buffers));
          for (size_t i = 0; i < roots.size(); ++i) {
            roots[i].reset(buffers[i]);
          }
        } else {
          for (auto& root : roots) {
            IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
                pool,
                iree_hal_make_pool_reservation_request(sizeof(input),
                                                       /*min_alignment=*/1),
                iree_infinite_timeout(), root.out()));
          }
        }
        // The unsignaled gate keeps queued storage uncommitted while both
        // levels of ordinary views capture their immutable binding-table
        // pointer.
        for (size_t i = 0; i < views.size(); ++i) {
          Ref<iree_hal_buffer_t> outer;
          IREE_ASSERT_OK(iree_hal_buffer_subspan(
              roots[i], sizeof(int32_t), 6 * sizeof(int32_t),
              iree_allocator_system(), outer.out()));
          IREE_ASSERT_OK(iree_hal_buffer_subspan(
              outer, sizeof(int32_t), 4 * sizeof(int32_t),
              iree_allocator_system(), views[i].out()));
          EXPECT_EQ(views[i].get()->memory.bindings,
                    roots[i].get()->memory.bindings);
          iree_hal_buffer_mapping_t mapping = {};
          IREE_EXPECT_STATUS_IS(
              IREE_STATUS_PERMISSION_DENIED,
              iree_hal_buffer_map_range(views[i], IREE_HAL_MAPPING_MODE_SCOPED,
                                        IREE_HAL_MEMORY_ACCESS_READ,
                                        IREE_HAL_BUFFER_MAP_FLAG_NONE, 0,
                                        IREE_HAL_WHOLE_BUFFER, &mapping));
        }
        IREE_ASSERT_OK(iree_hal_semaphore_list_signal(gate, nullptr));
        const iree_hal_semaphore_list_t ready =
            queued ? static_cast<iree_hal_semaphore_list_t>(allocated)
                   : iree_hal_semaphore_list_empty();
        SemaphoreList uploaded(device_, {0, 0, 0}, {1, 1, 1});
        const void* upload_data[] = {input.data(), initial_output.data(),
                                     workgroups.data()};
        const iree_device_size_t upload_lengths[] = {
            sizeof(input), sizeof(initial_output), sizeof(workgroups)};
        for (size_t i = 0; i < roots.size(); ++i) {
          const iree_hal_semaphore_list_t done = {1, &uploaded.semaphores[i],
                                                  &uploaded.payload_values[i]};
          IREE_ASSERT_OK(iree_hal_queue_upload(
              transfer_queue_, ready, done, upload_data[i], roots[i], 0,
              upload_lengths[i], /*barriers=*/NULL));
        }

        iree_hal_buffer_ref_t refs[2];
        const auto bindings = MakeBindings(views[0], views[1], refs);
        SemaphoreList executed(device_, {0}, {1});
        if (mode == DispatchMode::kRecorded ||
            mode == DispatchMode::kBindingTable) {
          Ref<iree_hal_command_buffer_t> commands;
          iree_hal_buffer_binding_t entries[2];
          auto table = iree_hal_buffer_binding_table_empty();
          if (mode == DispatchMode::kBindingTable) {
            CreateIndirectDispatchCommandBuffer(kElementCount, &commands);
            table = MakeBindingTable(views[0], views[1], entries);
          } else {
            IREE_ASSERT_OK(CreateCommandBuffer(
                IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
                IREE_HAL_COMMAND_CATEGORY_DISPATCH, 0, commands.out()));
            IREE_ASSERT_OK(iree_hal_command_buffer_begin(commands));
            IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
                commands, executable_,
                iree_hal_executable_function_from_index(0),
                iree_hal_make_static_dispatch_config(kElementCount, 1, 1),
                iree_const_byte_span_empty(), bindings,
                IREE_HAL_DISPATCH_FLAG_NONE));
            IREE_ASSERT_OK(iree_hal_command_buffer_end(commands));
          }
          IREE_ASSERT_OK(iree_hal_queue_execute(
              dispatch_queue_, uploaded, executed, commands, table,
              IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
        } else {
          auto config =
              iree_hal_make_static_dispatch_config(kElementCount, 1, 1);
          iree_hal_dispatch_flags_t flags = IREE_HAL_DISPATCH_FLAG_NONE;
          if (mode == DispatchMode::kIndirectParameters) {
            config.workgroup_count_ref =
                iree_hal_make_buffer_ref(roots[2], 0, sizeof(workgroups));
            flags |= IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS;
          }
          IREE_ASSERT_OK(iree_hal_queue_dispatch(
              dispatch_queue_, uploaded, executed, executable_,
              iree_hal_executable_function_from_index(0), config,
              iree_const_byte_span_empty(), bindings, /*barriers=*/NULL,
              flags));
        }
        std::array<int32_t, 8> result = {};
        SemaphoreList downloaded(device_, {0}, {1});
        IREE_ASSERT_OK(iree_hal_queue_download(
            transfer_queue_, executed, downloaded, roots[1], 0, result.data(),
            sizeof(result), /*barriers=*/NULL));
        IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
            downloaded, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
        EXPECT_THAT(result,
                    ::testing::ElementsAre(11, 22, 8, 5, 37, 407, 77, 88));
        for (size_t i = 0; i < views.size(); ++i) {
          ExpectNativeBindingRange(roots[i], views[i], 2 * sizeof(int32_t));
        }
        if (queued) {
          iree_hal_buffer_t* buffers[] = {roots[0], roots[1], roots[2]};
          SemaphoreList deallocated(device_, {0}, {1});
          IREE_ASSERT_OK(
              iree_hal_queue_dealloca(transfer_queue_, downloaded, deallocated,
                                      IREE_ARRAYSIZE(buffers), buffers));
          IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
              deallocated, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
        }
      }
    }
    pool.reset();
    if (queued_arena) {
      iree_hal_buffer_t* buffers[] = {arena};
      SemaphoreList deallocated(device_, {0}, {1});
      IREE_ASSERT_OK(iree_hal_queue_dealloca(
          transfer_queue_, iree_hal_semaphore_list_empty(), deallocated,
          IREE_ARRAYSIZE(buffers), buffers));
      IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
          deallocated, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    }
  }
}

TEST_P(BdaSpirvTest, CommandBufferExecutesBdaShaderWithSparseBindings) {
  std::array<iree_hal_pool_family_access_t, 2> families = {{
      {
          .family = iree_hal_queue_family(transfer_queue_),
          .usage =
              IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_STORAGE,
      },
      {
          .family = iree_hal_queue_family(dispatch_queue_),
          .usage =
              IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_STORAGE,
      },
  }};
  const iree_hal_pool_scope_t scope = {
      .family_count = families[0].family == families[1].family ? 1u : 2u,
      .families = families.data(),
  };
  iree_hal_virtual_buffer_capabilities_t capabilities;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
      device_group_, scope, iree_allocator_system(), &capabilities));
  if (!capabilities.max_byte_length) {
    GTEST_SKIP() << "Vulkan sparse virtual memory is not available";
  }
  iree_hal_queue_t* mapping_queue = FindVulkanSparseBindingQueue(device_);
  ASSERT_NE(nullptr, mapping_queue);
  for (auto& family : families) {
    family.interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE;
  }
  iree_hal_slab_pool_options_t source_options;
  iree_hal_slab_pool_options_initialize(&source_options);
  Ref<iree_hal_pool_t> source_pool;
  IREE_ASSERT_OK(
      iree_hal_slab_pool_create(device_group_, scope, &source_options,
                                iree_allocator_system(), source_pool.out()));
  const auto requirements =
      iree_hal_pool_query_mapping_requirements(source_pool);
  const iree_device_size_t page_size = std::max(
      {capabilities.reservation_granularity, capabilities.mapping_granularity,
       requirements.minimum_granularity});
  ASSERT_GE(page_size, 8 * sizeof(int32_t));
  for (auto& family : families) {
    family.interfaces = 0;
  }
  Ref<iree_hal_buffer_t> input_memory;
  Ref<iree_hal_buffer_t> output_memory;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      source_pool, iree_hal_make_pool_reservation_request(page_size, page_size),
      iree_infinite_timeout(), input_memory.out()));
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      source_pool, iree_hal_make_pool_reservation_request(page_size, page_size),
      iree_infinite_timeout(), output_memory.out()));
  const iree_hal_virtual_buffer_options_t reservation_options = {
      .byte_length = page_size,
      .mapping_granularity = page_size,
  };
  BdaMemoryReservation input_buffer;
  BdaMemoryReservation output_buffer;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_create(
      device_group_, scope, &reservation_options, iree_allocator_system(),
      input_buffer.out()));
  IREE_ASSERT_OK(iree_hal_virtual_buffer_create(
      device_group_, scope, &reservation_options, iree_allocator_system(),
      output_buffer.out()));
  iree_hal_memory_mapping_t mappings[2] = {
      {
          .target = input_buffer.owner(),
          .scope =
              {
                  .domain = iree_hal_device_group_memory_domain(device_group_),
                  .id = IREE_HAL_MEMORY_SCOPE_ANY,
              },
          .target_range = iree_hal_make_buffer_range(0, page_size),
          .source = input_memory.get(),
          .access = IREE_HAL_MEMORY_ACCESS_ALL,
          .operation = IREE_HAL_MEMORY_MAPPING_OPERATION_BIND,
      },
      {
          .target = output_buffer.owner(),
          .scope =
              {
                  .domain = iree_hal_device_group_memory_domain(device_group_),
                  .id = IREE_HAL_MEMORY_SCOPE_ANY,
              },
          .target_range = iree_hal_make_buffer_range(0, page_size),
          .source = output_memory.get(),
          .access = IREE_HAL_MEMORY_ACCESS_ALL,
          .operation = IREE_HAL_MEMORY_MAPPING_OPERATION_BIND,
      },
  };
  SemaphoreList mapped(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_map(
      mapping_queue, iree_hal_semaphore_list_empty(), mapped,
      IREE_ARRAYSIZE(mappings), mappings, IREE_HAL_QUEUE_MAP_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(mapped, iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));

  std::array<Ref<iree_hal_buffer_t>, 2> views;
  iree_hal_buffer_t* roots[] = {input_buffer.get(), output_buffer.get()};
  for (size_t i = 0; i < views.size(); ++i) {
    Ref<iree_hal_buffer_t> outer;
    IREE_ASSERT_OK(
        iree_hal_buffer_subspan(roots[i], sizeof(int32_t), 6 * sizeof(int32_t),
                                iree_allocator_system(), outer.out()));
    IREE_ASSERT_OK(
        iree_hal_buffer_subspan(outer, sizeof(int32_t), 4 * sizeof(int32_t),
                                iree_allocator_system(), views[i].out()));
    ExpectNativeBindingRange(roots[i], views[i], 2 * sizeof(int32_t));
  }

  const int32_t input_pattern = 5;
  const int32_t output_pattern = -1;
  SemaphoreList fill_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_fill(
      transfer_queue_, iree_hal_semaphore_list_empty(), fill_signal,
      input_buffer.get(), /*target_offset=*/0, 8 * sizeof(input_pattern),
      &input_pattern, sizeof(input_pattern), /*barriers=*/NULL,
      IREE_HAL_FILL_FLAG_NONE));
  SemaphoreList output_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_fill(
      transfer_queue_, fill_signal, output_signal, output_buffer.get(),
      /*target_offset=*/0, 8 * sizeof(output_pattern), &output_pattern,
      sizeof(output_pattern), /*barriers=*/NULL, IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      output_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_buffer_ref_t binding_refs[2];
  iree_hal_buffer_ref_list_t bindings =
      MakeBindings(views[0], views[1], binding_refs);

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(CreateCommandBuffer(
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT, IREE_HAL_COMMAND_CATEGORY_DISPATCH,
      /*binding_capacity=*/0, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable_, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(4, 1, 1),
      iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));
  IREE_ASSERT_OK(SubmitCommandBufferAndWait(command_buffer));

  Ref<iree_hal_buffer_t> readback_buffer;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(8 * sizeof(int32_t), readback_buffer.out()));
  SemaphoreList copy_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_copy(
      transfer_queue_, iree_hal_semaphore_list_empty(), copy_signal,
      output_buffer.get(), /*source_offset=*/0, readback_buffer.get(),
      /*target_offset=*/0, 8 * sizeof(int32_t), /*barriers=*/NULL,
      IREE_HAL_COPY_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      copy_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  std::vector<int32_t> output_data = ReadBufferData<int32_t>(readback_buffer);
  EXPECT_THAT(output_data, ContainerEq(std::vector<int32_t>{-1, -1, 12, 12, 12,
                                                            12, -1, -1}));

  for (auto& mapping : mappings) {
    mapping.source = nullptr;
    mapping.access = IREE_HAL_MEMORY_ACCESS_NONE;
    mapping.operation = IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND;
  }
  SemaphoreList unmapped(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_map(
      mapping_queue, iree_hal_semaphore_list_empty(), unmapped,
      IREE_ARRAYSIZE(mappings), mappings, IREE_HAL_QUEUE_MAP_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(unmapped, iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));
}

class BdaSpirvReplayCacheTest : public BdaSpirvTest {};

TEST_P(BdaSpirvReplayCacheTest, EmptyGridsAcceptEmptyBindings) {
  TestEmptyGridDispatches(/*capture_profile=*/false);
}

TEST_P(BdaSpirvReplayCacheTest, CommandBufferRebindsIndirectCounts) {
  ASSERT_GE(NativeReplayCacheStats().max_instance_count, 1);
  for (auto flags : {IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS,
                     IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS}) {
    SCOPED_TRACE(flags);
    for (bool indirect_reference : {false, true}) {
      SCOPED_TRACE(indirect_reference);
      Ref<iree_hal_buffer_t> input_buffer;
      const int32_t input_data[] = {1, -2, 30, 400};
      IREE_ASSERT_OK(CreateDeviceBufferWithData(input_data, sizeof(input_data),
                                                input_buffer.out()));
      const uint32_t counts_a[] = {0, 1, 1, kElementCount, 1, 1};
      const uint32_t counts_b[] = {2, 1, 1};
      Ref<iree_hal_buffer_t> count_buffer_a;
      Ref<iree_hal_buffer_t> count_buffer_b;
      IREE_ASSERT_OK(CreateDeviceBufferWithData(counts_a, sizeof(counts_a),
                                                count_buffer_a.out()));
      IREE_ASSERT_OK(CreateDeviceBufferWithData(counts_b, sizeof(counts_b),
                                                count_buffer_b.out()));
      const iree_device_size_t count_length = 3 * sizeof(uint32_t);
      const iree_hal_dispatch_config_t config = {
          .workgroup_count_ref =
              indirect_reference
                  ? iree_hal_make_indirect_buffer_ref(2, 0, count_length)
                  : iree_hal_make_buffer_ref(count_buffer_a, count_length,
                                             count_length),
      };
      iree_hal_buffer_ref_t binding_refs[] = {
          iree_hal_make_indirect_buffer_ref(0, 0, kDispatchByteLength),
          iree_hal_make_indirect_buffer_ref(1, 0, kDispatchByteLength),
      };
      Ref<iree_hal_command_buffer_t> command_buffer;
      IREE_ASSERT_OK(CreateCommandBuffer(IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
                                         IREE_HAL_COMMAND_CATEGORY_DISPATCH,
                                         /*binding_capacity=*/3,
                                         command_buffer.out()));
      IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
      IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
          command_buffer, executable_,
          iree_hal_executable_function_from_index(0), config,
          iree_const_byte_span_empty(),
          {IREE_ARRAYSIZE(binding_refs), binding_refs}, flags));
      IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

      // Change the offset, then the allocation, then return to the empty grid.
      // Shader bindings and native indirect-parameter references both rebind.
      const uint32_t counts[] = {0, kElementCount, 2, 0};
      const int32_t expected_values[] = {8, 5, 37, 407};
      for (size_t iteration = 0; iteration < IREE_ARRAYSIZE(counts);
           ++iteration) {
        SCOPED_TRACE(iteration);
        Ref<iree_hal_buffer_t> output_buffer;
        IREE_ASSERT_OK(
            CreateZeroedDeviceBuffer(kDispatchByteLength, output_buffer.out()));
        iree_hal_buffer_binding_t table_entries[] = {
            {input_buffer, 0, kDispatchByteLength},
            {output_buffer, 0, kDispatchByteLength},
            {iteration == 2 ? count_buffer_b.get() : count_buffer_a.get(),
             iteration == 1 ? count_length : 0, count_length},
        };
        ExecuteCommandBufferAndWait(
            command_buffer, {IREE_ARRAYSIZE(table_entries), table_entries});
        std::vector<int32_t> expected(kElementCount, 0);
        const uint32_t count =
            indirect_reference ? counts[iteration] : kElementCount;
        std::copy_n(expected_values, count, expected.begin());
        EXPECT_THAT(ReadBufferData<int32_t>(output_buffer),
                    ContainerEq(expected));
      }
    }
  }
}

TEST_P(BdaSpirvReplayCacheTest, CommandBufferReusesCachedNativeReplay) {
  ASSERT_GE(NativeReplayCacheStats().max_instance_count, 1);
  ASSERT_EQ(0, NativeReplayCacheStats().retained_instance_count);
  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  ASSERT_EQ(0, NativeReplayCacheStats().instance_count);

  Ref<iree_hal_buffer_t> input_buffer;
  Ref<iree_hal_buffer_t> output_buffer;
  CreateInputOutputBuffers(&input_buffer, &output_buffer);

  Ref<iree_hal_command_buffer_t> command_buffer;
  CreateIndirectDispatchCommandBuffer(/*workgroup_count=*/kElementCount,
                                      &command_buffer);

  iree_hal_buffer_binding_t binding_table_entries[2];
  iree_hal_buffer_binding_table_t binding_table = MakeBindingTable(
      input_buffer.get(), output_buffer.get(), binding_table_entries);

  const uint64_t initial_hit_count = NativeReplayCacheStats().hit_count;
  const uint64_t initial_miss_count = NativeReplayCacheStats().miss_count;
  const uint64_t initial_create_count = NativeReplayCacheStats().create_count;
  const uint64_t initial_publication_bytes =
      NativeReplayCacheStats().publication_bytes;

  ExecuteCommandBufferAndWait(command_buffer.get(), binding_table);
  ExpectOutput(output_buffer.get());
  ExecuteCommandBufferAndWait(command_buffer.get(), binding_table);
  ExpectOutput(output_buffer.get());

  EXPECT_GE(NativeReplayCacheStats().miss_count - initial_miss_count, 1);
  EXPECT_GE(NativeReplayCacheStats().create_count - initial_create_count, 1);
  EXPECT_GE(NativeReplayCacheStats().hit_count - initial_hit_count, 1);
  EXPECT_GE(
      NativeReplayCacheStats().publication_bytes - initial_publication_bytes,
      (uint64_t)(2 * sizeof(uint64_t)));

  command_buffer.reset();
  IREE_ASSERT_OK(iree_hal_device_trim(device_));
}

TEST_P(BdaSpirvReplayCacheTest, TrimDropsIdleCachedNativeReplay) {
  ASSERT_EQ(0, NativeReplayCacheStats().retained_instance_count);

  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  ASSERT_EQ(0, NativeReplayCacheStats().instance_count);

  Ref<iree_hal_buffer_t> input_buffer;
  Ref<iree_hal_buffer_t> output_buffer;
  CreateInputOutputBuffers(&input_buffer, &output_buffer);

  Ref<iree_hal_command_buffer_t> command_buffer;
  CreateIndirectDispatchCommandBuffer(/*workgroup_count=*/kElementCount,
                                      &command_buffer);

  iree_hal_buffer_binding_t binding_table_entries[2];
  iree_hal_buffer_binding_table_t binding_table = MakeBindingTable(
      input_buffer.get(), output_buffer.get(), binding_table_entries);

  const uint64_t initial_trim_count = NativeReplayCacheStats().trim_count;
  ExecuteCommandBufferAndWait(command_buffer.get(), binding_table);
  ExpectOutput(output_buffer.get());

  ASSERT_GE(NativeReplayCacheStats().instance_count, 1);
  ASSERT_GE(NativeReplayCacheStats().publication_bytes,
            (uint64_t)(2 * sizeof(uint64_t)));

  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  EXPECT_EQ(0, NativeReplayCacheStats().instance_count);
  EXPECT_EQ(0, NativeReplayCacheStats().publication_bytes);
  EXPECT_GE(NativeReplayCacheStats().trim_count - initial_trim_count, 1);
}

TEST_P(BdaSpirvReplayCacheTest, CommandBufferSkipsUnchangedBdaPublication) {
  ASSERT_GE(NativeReplayCacheStats().max_instance_count, 1);
  ASSERT_EQ(0, NativeReplayCacheStats().retained_instance_count);

  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  ASSERT_EQ(0, NativeReplayCacheStats().instance_count);

  Ref<iree_hal_buffer_t> input_buffer;
  Ref<iree_hal_buffer_t> output_buffer;
  CreateInputOutputBuffers(&input_buffer, &output_buffer);

  Ref<iree_hal_command_buffer_t> command_buffer;
  CreateIndirectDispatchCommandBuffer(/*workgroup_count=*/kElementCount,
                                      &command_buffer);

  iree_hal_buffer_binding_t binding_table_entries[2];
  iree_hal_buffer_binding_table_t binding_table = MakeBindingTable(
      input_buffer.get(), output_buffer.get(), binding_table_entries);

  const uint64_t initial_hit_count = NativeReplayCacheStats().hit_count;
  const uint64_t initial_miss_count = NativeReplayCacheStats().miss_count;
  const uint64_t initial_publication_skip_count =
      NativeReplayCacheStats().publication_skip_count;
  const uint64_t initial_publication_update_count =
      NativeReplayCacheStats().publication_update_count;

  ExecuteCommandBufferAndWait(command_buffer.get(), binding_table);
  ExpectOutput(output_buffer.get());

  const int32_t reset_pattern = 0;
  SemaphoreList fill_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_fill(
      transfer_queue_, iree_hal_semaphore_list_empty(), fill_signal,
      output_buffer.get(), /*target_offset=*/0, kDispatchByteLength,
      &reset_pattern, sizeof(reset_pattern), /*barriers=*/NULL,
      IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      fill_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  ExpectFilledOutputPrefix(output_buffer.get(), /*expected_value=*/0);

  ExecuteCommandBufferAndWait(command_buffer.get(), binding_table);
  ExpectOutput(output_buffer.get());

  EXPECT_GE(NativeReplayCacheStats().miss_count - initial_miss_count, 1);
  EXPECT_GE(NativeReplayCacheStats().hit_count - initial_hit_count, 1);
  EXPECT_GE(NativeReplayCacheStats().publication_skip_count -
                initial_publication_skip_count,
            1);
  EXPECT_EQ(initial_publication_update_count,
            NativeReplayCacheStats().publication_update_count);
}

TEST_P(BdaSpirvReplayCacheTest, CommandBufferRepublishesChangedBdaPublication) {
  ASSERT_GE(NativeReplayCacheStats().max_instance_count, 1);
  ASSERT_EQ(0, NativeReplayCacheStats().retained_instance_count);

  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  ASSERT_EQ(0, NativeReplayCacheStats().instance_count);

  Ref<iree_hal_buffer_t> input_buffer;
  Ref<iree_hal_buffer_t> output_a_buffer;
  CreateInputOutputBuffers(&input_buffer, &output_a_buffer);
  Ref<iree_hal_buffer_t> output_b_buffer;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(kDispatchByteLength, output_b_buffer.out()));

  Ref<iree_hal_command_buffer_t> command_buffer;
  CreateIndirectDispatchCommandBuffer(/*workgroup_count=*/kElementCount,
                                      &command_buffer);

  iree_hal_buffer_binding_t binding_table_a_entries[2];
  iree_hal_buffer_binding_table_t binding_table_a = MakeBindingTable(
      input_buffer.get(), output_a_buffer.get(), binding_table_a_entries);
  iree_hal_buffer_binding_t binding_table_b_entries[2];
  iree_hal_buffer_binding_table_t binding_table_b = MakeBindingTable(
      input_buffer.get(), output_b_buffer.get(), binding_table_b_entries);

  const uint64_t initial_hit_count = NativeReplayCacheStats().hit_count;
  const uint64_t initial_miss_count = NativeReplayCacheStats().miss_count;
  const uint64_t initial_publication_skip_count =
      NativeReplayCacheStats().publication_skip_count;
  const uint64_t initial_publication_update_count =
      NativeReplayCacheStats().publication_update_count;

  ExecuteCommandBufferAndWait(command_buffer.get(), binding_table_a);
  ExpectOutput(output_a_buffer.get());
  ExecuteCommandBufferAndWait(command_buffer.get(), binding_table_b);
  ExpectOutput(output_b_buffer.get());

  EXPECT_GE(NativeReplayCacheStats().miss_count - initial_miss_count, 1);
  EXPECT_GE(NativeReplayCacheStats().hit_count - initial_hit_count, 1);
  EXPECT_EQ(initial_publication_skip_count,
            NativeReplayCacheStats().publication_skip_count);
  EXPECT_GE(NativeReplayCacheStats().publication_update_count -
                initial_publication_update_count,
            1);
}

TEST_P(BdaSpirvReplayCacheTest, ConcurrentExecutionsForkCachedNativeReplay) {
  ASSERT_GE(NativeReplayCacheStats().max_instance_count, 2);
  ASSERT_EQ(0, NativeReplayCacheStats().retained_instance_count);

  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  ASSERT_EQ(0, NativeReplayCacheStats().instance_count);

  static constexpr uint32_t kLargeDispatchElementCount = 8 * 1024 * 1024;
  const iree_device_size_t dispatch_byte_length =
      kLargeDispatchElementCount * sizeof(int32_t);

  Ref<iree_hal_buffer_t> input_buffer;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(dispatch_byte_length, input_buffer.out()));
  Ref<iree_hal_buffer_t> output_a_buffer;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(dispatch_byte_length, output_a_buffer.out()));
  Ref<iree_hal_buffer_t> output_b_buffer;
  IREE_ASSERT_OK(
      CreateZeroedDeviceBuffer(dispatch_byte_length, output_b_buffer.out()));

  const int32_t input_pattern = 5;
  SemaphoreList fill_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_fill(
      transfer_queue_, iree_hal_semaphore_list_empty(), fill_signal,
      input_buffer.get(), /*target_offset=*/0, dispatch_byte_length,
      &input_pattern, sizeof(input_pattern), /*barriers=*/NULL,
      IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      fill_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  Ref<iree_hal_command_buffer_t> command_buffer;
  CreateIndirectDispatchCommandBuffer(kLargeDispatchElementCount,
                                      &command_buffer);

  iree_hal_buffer_binding_t binding_table_a_entries[2];
  iree_hal_buffer_binding_table_t binding_table_a = MakeBindingTable(
      input_buffer.get(), output_a_buffer.get(), binding_table_a_entries);
  iree_hal_buffer_binding_t binding_table_b_entries[2];
  iree_hal_buffer_binding_table_t binding_table_b = MakeBindingTable(
      input_buffer.get(), output_b_buffer.get(), binding_table_b_entries);

  const uint64_t initial_create_count = NativeReplayCacheStats().create_count;
  const uint64_t initial_fork_count = NativeReplayCacheStats().fork_count;
  iree_hal_queue_t* queue = QueueForCommandBuffer(command_buffer.get());
  ASSERT_NE(nullptr, queue);
  SemaphoreList execute_a_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_execute(
      queue, iree_hal_semaphore_list_empty(), execute_a_signal,
      command_buffer.get(), binding_table_a, IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
  SemaphoreList execute_b_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_execute(
      queue, iree_hal_semaphore_list_empty(), execute_b_signal,
      command_buffer.get(), binding_table_b, IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      execute_a_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      execute_b_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  ExpectFilledOutputPrefix(output_a_buffer.get(), /*expected_value=*/12);
  ExpectFilledOutputPrefix(output_b_buffer.get(), /*expected_value=*/12);

  EXPECT_GE(NativeReplayCacheStats().create_count - initial_create_count, 2);
  EXPECT_GE(NativeReplayCacheStats().fork_count - initial_fork_count, 1);
  EXPECT_GE(NativeReplayCacheStats().peak_instance_count, 2);

  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  EXPECT_EQ(0, NativeReplayCacheStats().instance_count);
}

CTS_REGISTER_TEST_SUITE_WITH_TAGS(BdaSpirvTest, {"vulkan"},
                                  {"vulkan_replay_cache"});
CTS_REGISTER_TEST_SUITE_WITH_TAGS(BdaSpirvReplayCacheTest,
                                  {"vulkan_replay_cache"}, {});

}  // namespace iree::hal::cts
