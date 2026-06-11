// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Vulkan SPIR-V BDA coverage. The driver publishes the exact HAL binding
// table the dispatch provides and the shader consumes it through the hidden BDA
// root.

#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/hal/cts/util/profile_test_util.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/vulkan/command_buffer.h"
#include "iree/hal/drivers/vulkan/cts/bda_spirv_test_spv.h"

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
    if (HasFatalFailure() || IsSkipped()) return;

    IREE_ASSERT_OK(iree_hal_executable_cache_create(
        device_, iree_make_cstring_view("default"), &executable_cache_));

    IREE_ASSERT_OK(
        PrepareBdaExecutable(BdaSpirvFixture(kBdaSpirvAdd7), &executable_));
  }

  void TearDown() override {
    iree_hal_executable_release(executable_);
    executable_ = nullptr;
    iree_hal_executable_cache_release(executable_cache_);
    executable_cache_ = nullptr;
    CtsTestBase::TearDown();
  }

  iree_status_t PrepareBdaExecutable(
      iree_const_byte_span_t executable_data,
      iree_hal_executable_caching_mode_t caching_mode,
      iree_hal_executable_t** out_executable) {
    *out_executable = nullptr;
    iree_hal_executable_params_t executable_params;
    iree_hal_executable_params_initialize(&executable_params);
    executable_params.caching_mode = caching_mode;
    executable_params.executable_format =
        iree_make_cstring_view("vulkan-spirv-bda");
    executable_params.executable_data = executable_data;
    return iree_hal_executable_cache_prepare_executable(
        executable_cache_, &executable_params, out_executable);
  }

  iree_status_t PrepareBdaExecutable(iree_const_byte_span_t executable_data,
                                     iree_hal_executable_t** out_executable) {
    return PrepareBdaExecutable(
        executable_data, IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA,
        out_executable);
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
        /*.count=*/2,
        /*.values=*/binding_refs,
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
        /*.count=*/IREE_ARRAYSIZE(binding_refs),
        /*.values=*/binding_refs,
    };

    IREE_ASSERT_OK(iree_hal_command_buffer_create(
        device_, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
        IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
        /*binding_capacity=*/2, command_buffer->out()));
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
        /*.count=*/2,
        /*.bindings=*/binding_table_entries,
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

  int64_t QueryNativeReplayCache(iree_string_view_t key) {
    int64_t value = 0;
    IREE_EXPECT_OK(iree_hal_device_query_i64(
        device_, IREE_SV("vulkan.queue.native_replay_cache"), key, &value));
    return value;
  }

  int64_t QueryBdaPublicationCache(iree_string_view_t key) {
    int64_t value = 0;
    IREE_EXPECT_OK(iree_hal_device_query_i64(
        device_, IREE_SV("vulkan.queue.bda_publication_cache"), key, &value));
    return value;
  }

  void ExecuteCommandBufferAndWait(
      iree_hal_command_buffer_t* command_buffer,
      iree_hal_buffer_binding_table_t binding_table) {
    SemaphoreList execute_signal(device_, {0}, {1});
    IREE_ASSERT_OK(iree_hal_device_queue_execute(
        device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
        execute_signal, command_buffer, binding_table,
        IREE_HAL_EXECUTE_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        execute_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  }

  iree_hal_executable_cache_t* executable_cache_ = nullptr;
  iree_hal_executable_t* executable_ = nullptr;
};

static iree_hal_buffer_params_t SparseDispatchBufferParams() {
  iree_hal_buffer_params_t params = {0};
  params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE |
                IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
  return params;
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
      IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA |
          IREE_HAL_EXECUTABLE_CACHING_MODE_DISABLE_VERIFICATION,
      executable.out()));

  iree_hal_buffer_ref_list_t bindings = {};
  SemaphoreList dispatch_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dispatch(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      dispatch_signal, executable.get(),
      iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1),
      iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));
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

class BdaSparseVirtualBufferRef {
 public:
  explicit BdaSparseVirtualBufferRef(iree_hal_allocator_t* allocator)
      : allocator_(allocator) {}
  ~BdaSparseVirtualBufferRef() { reset(); }

  BdaSparseVirtualBufferRef(const BdaSparseVirtualBufferRef&) = delete;
  BdaSparseVirtualBufferRef& operator=(const BdaSparseVirtualBufferRef&) =
      delete;

  iree_hal_buffer_t* get() const { return buffer_; }
  iree_hal_buffer_t** out() { return &buffer_; }

  void reset() {
    if (buffer_) {
      IREE_EXPECT_OK(
          iree_hal_allocator_virtual_memory_release(allocator_, buffer_));
      buffer_ = nullptr;
    }
  }

 private:
  // Allocator that owns the virtual reservation.
  iree_hal_allocator_t* allocator_ = nullptr;

  // Reserved virtual sparse buffer released through |allocator_|.
  iree_hal_buffer_t* buffer_ = nullptr;
};

class BdaSparsePhysicalMemoryRef {
 public:
  explicit BdaSparsePhysicalMemoryRef(iree_hal_allocator_t* allocator)
      : allocator_(allocator) {}
  ~BdaSparsePhysicalMemoryRef() { reset(); }

  BdaSparsePhysicalMemoryRef(const BdaSparsePhysicalMemoryRef&) = delete;
  BdaSparsePhysicalMemoryRef& operator=(const BdaSparsePhysicalMemoryRef&) =
      delete;

  iree_hal_physical_memory_t* get() const { return memory_; }
  iree_hal_physical_memory_t** out() { return &memory_; }

  void reset() {
    if (memory_) {
      IREE_EXPECT_OK(
          iree_hal_allocator_physical_memory_free(allocator_, memory_));
      memory_ = nullptr;
    }
  }

 private:
  // Allocator that owns the physical sparse memory.
  iree_hal_allocator_t* allocator_ = nullptr;

  // Physical sparse memory handle released through |allocator_|.
  iree_hal_physical_memory_t* memory_ = nullptr;
};

TEST_P(BdaSpirvTest, QueueDispatchExecutesBdaShader) {
  Ref<iree_hal_buffer_t> input_buffer;
  Ref<iree_hal_buffer_t> output_buffer;
  CreateInputOutputBuffers(&input_buffer, &output_buffer);

  iree_hal_buffer_ref_t binding_refs[2];
  iree_hal_buffer_ref_list_t bindings =
      MakeBindings(input_buffer, output_buffer, binding_refs);

  SemaphoreList dispatch_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dispatch(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      dispatch_signal, executable_, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(4, 1, 1),
      iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  ExpectOutput(output_buffer);
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
  IREE_ASSERT_OK(iree_hal_device_queue_dispatch(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      dispatch_signal, executable.get(),
      iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(4, 1, 1),
      iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  ExpectOutput(output_buffer);
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
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_device_queue_dispatch(
          device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
          iree_hal_semaphore_list_empty(), executable.get(),
          iree_hal_executable_function_from_index(0),
          iree_hal_make_static_dispatch_config(4, 1, 1),
          iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));
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
      iree_hal_device_queue_dispatch(
          device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
          iree_hal_semaphore_list_empty(), executable.get(),
          iree_hal_executable_function_from_index(0),
          iree_hal_make_static_dispatch_config(4, 1, 1),
          iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));
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
      iree_hal_device_queue_dispatch(
          device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
          iree_hal_semaphore_list_empty(), executable.get(),
          iree_hal_executable_function_from_index(0),
          iree_hal_make_static_dispatch_config(4, 1, 1),
          iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));

  const uint32_t ignored_constant = 123u;
  SemaphoreList dispatch_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dispatch(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      dispatch_signal, executable.get(),
      iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(4, 1, 1),
      iree_make_const_byte_span(&ignored_constant, sizeof(ignored_constant)),
      bindings, IREE_HAL_DISPATCH_FLAG_NONE));
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
      /*.count=*/binding_refs.size(),
      /*.values=*/binding_refs.data(),
  };

  SemaphoreList dispatch_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dispatch(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      dispatch_signal, executable_, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(4, 1, 1),
      iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));
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
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, command_buffer.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable_, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(4, 1, 1),
      iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  const int64_t initial_one_shot_bypass_count =
      QueryNativeReplayCache(IREE_SV("one_shot_bypass_count"));
  IREE_ASSERT_OK(SubmitCommandBufferAndWait(command_buffer));
  ExpectOutput(output_buffer);
  EXPECT_GE(QueryNativeReplayCache(IREE_SV("one_shot_bypass_count")) -
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
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
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
          /*.count=*/oversized_bindings.size(),
          /*.values=*/oversized_bindings.data(),
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
  const int64_t initial_block_count =
      QueryBdaPublicationCache(IREE_SV("block_count"));

  Ref<iree_hal_buffer_t> input_buffer;
  Ref<iree_hal_buffer_t> output_buffer;
  CreateInputOutputBuffers(&input_buffer, &output_buffer);
  std::vector<iree_hal_buffer_ref_t> oversized_bindings =
      MakeOversizedPublicationBindings(input_buffer.get(), output_buffer.get());

  SemaphoreList dispatch_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_dispatch(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      dispatch_signal, executable_, iree_hal_executable_function_from_index(0),
      iree_hal_make_static_dispatch_config(1, 1, 1),
      iree_const_byte_span_empty(),
      iree_hal_buffer_ref_list_t{
          /*.count=*/oversized_bindings.size(),
          /*.values=*/oversized_bindings.data(),
      },
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      dispatch_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  EXPECT_GT(QueryBdaPublicationCache(IREE_SV("block_count")),
            initial_block_count);
  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  EXPECT_LE(QueryBdaPublicationCache(IREE_SV("block_count")),
            initial_block_count);
}

TEST_P(BdaSpirvTest, CommandBufferReusesCachedNativeReplay) {
  if (QueryNativeReplayCache(IREE_SV("max_instance_count")) == 0) {
    GTEST_SKIP() << "Vulkan BDA native replay cache is disabled";
  }
  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  if (QueryNativeReplayCache(IREE_SV("instance_count")) >=
      QueryNativeReplayCache(IREE_SV("max_instance_count"))) {
    GTEST_SKIP() << "Vulkan BDA native replay cache is already full";
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

  const int64_t initial_hit_count =
      QueryNativeReplayCache(IREE_SV("hit_count"));
  const int64_t initial_miss_count =
      QueryNativeReplayCache(IREE_SV("miss_count"));
  const int64_t initial_create_count =
      QueryNativeReplayCache(IREE_SV("create_count"));
  const int64_t initial_publication_bytes =
      QueryNativeReplayCache(IREE_SV("publication_bytes"));

  ExecuteCommandBufferAndWait(command_buffer.get(), binding_table);
  ExpectOutput(output_buffer.get());
  ExecuteCommandBufferAndWait(command_buffer.get(), binding_table);
  ExpectOutput(output_buffer.get());

  EXPECT_GE(QueryNativeReplayCache(IREE_SV("miss_count")) - initial_miss_count,
            1);
  EXPECT_GE(
      QueryNativeReplayCache(IREE_SV("create_count")) - initial_create_count,
      1);
  EXPECT_GE(QueryNativeReplayCache(IREE_SV("hit_count")) - initial_hit_count,
            1);
  EXPECT_GE(QueryNativeReplayCache(IREE_SV("publication_bytes")) -
                initial_publication_bytes,
            (int64_t)(2 * sizeof(uint64_t)));

  command_buffer.reset();
  IREE_ASSERT_OK(iree_hal_device_trim(device_));
}

TEST_P(BdaSpirvTest, CommandBufferProfilingBypassesCachedNativeReplay) {
  if (QueryNativeReplayCache(IREE_SV("max_instance_count")) == 0) {
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

  const int64_t initial_profile_bypass_count =
      QueryNativeReplayCache(IREE_SV("profile_bypass_count"));
  const int64_t initial_hit_count =
      QueryNativeReplayCache(IREE_SV("hit_count"));
  const int64_t initial_create_count =
      QueryNativeReplayCache(IREE_SV("create_count"));

  ExecuteCommandBufferAndWait(command_buffer.get(), binding_table);
  ExpectOutput(output_buffer.get());

  IREE_ASSERT_OK(iree_hal_device_profiling_flush(device_));
  IREE_ASSERT_OK(profiling.End());

  EXPECT_GE(QueryNativeReplayCache(IREE_SV("profile_bypass_count")) -
                initial_profile_bypass_count,
            1);
  EXPECT_EQ(initial_hit_count, QueryNativeReplayCache(IREE_SV("hit_count")));
  EXPECT_EQ(initial_create_count,
            QueryNativeReplayCache(IREE_SV("create_count")));
  EXPECT_GE(sink.dispatch_event_count, 1);
}

TEST_P(BdaSpirvTest, CommandBufferHandlesOversizedBdaPublication) {
  Ref<iree_hal_buffer_t> input_buffer;
  Ref<iree_hal_buffer_t> output_buffer;
  CreateInputOutputBuffers(&input_buffer, &output_buffer);

  std::vector<iree_hal_buffer_ref_t> binding_refs =
      MakeOversizedPublicationBindings(input_buffer, output_buffer);
  iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/binding_refs.size(),
      /*.values=*/binding_refs.data(),
  };

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
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

TEST_P(BdaSpirvTest, CommandBufferExecutesBdaShaderWithSparseBindings) {
  if (!iree_hal_allocator_supports_virtual_memory(device_allocator_)) {
    GTEST_SKIP() << "Vulkan sparse virtual memory is not available";
  }

  iree_hal_buffer_params_t params = SparseDispatchBufferParams();
  iree_device_size_t minimum_page_size = 0;
  iree_device_size_t recommended_page_size = 0;
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_query_granularity(
      device_allocator_, params, &minimum_page_size, &recommended_page_size));
  ASSERT_NE(0u, minimum_page_size);
  ASSERT_GE(recommended_page_size, minimum_page_size);
  ASSERT_GE(recommended_page_size, 4 * sizeof(int32_t));

  BdaSparseVirtualBufferRef input_buffer(device_allocator_);
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_reserve(
      device_allocator_, IREE_HAL_QUEUE_AFFINITY_ANY, recommended_page_size,
      input_buffer.out()));
  BdaSparseVirtualBufferRef output_buffer(device_allocator_);
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_reserve(
      device_allocator_, IREE_HAL_QUEUE_AFFINITY_ANY, recommended_page_size,
      output_buffer.out()));

  BdaSparsePhysicalMemoryRef input_memory(device_allocator_);
  IREE_ASSERT_OK(iree_hal_allocator_physical_memory_allocate(
      device_allocator_, params, recommended_page_size, iree_allocator_system(),
      input_memory.out()));
  BdaSparsePhysicalMemoryRef output_memory(device_allocator_);
  IREE_ASSERT_OK(iree_hal_allocator_physical_memory_allocate(
      device_allocator_, params, recommended_page_size, iree_allocator_system(),
      output_memory.out()));

  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_map(
      device_allocator_, input_buffer.get(), /*virtual_offset=*/0,
      input_memory.get(), /*physical_offset=*/0, recommended_page_size));
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_map(
      device_allocator_, output_buffer.get(), /*virtual_offset=*/0,
      output_memory.get(), /*physical_offset=*/0, recommended_page_size));
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_protect(
      device_allocator_, input_buffer.get(), /*virtual_offset=*/0,
      recommended_page_size, IREE_HAL_QUEUE_AFFINITY_ANY,
      IREE_HAL_MEMORY_PROTECTION_READ_WRITE));
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_protect(
      device_allocator_, output_buffer.get(), /*virtual_offset=*/0,
      recommended_page_size, IREE_HAL_QUEUE_AFFINITY_ANY,
      IREE_HAL_MEMORY_PROTECTION_READ_WRITE));

  const int32_t input_pattern = 5;
  SemaphoreList fill_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_fill(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      fill_signal, input_buffer.get(), /*target_offset=*/0,
      4 * sizeof(input_pattern), &input_pattern, sizeof(input_pattern),
      IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      fill_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  iree_hal_buffer_ref_t binding_refs[2];
  iree_hal_buffer_ref_list_t bindings =
      MakeBindings(input_buffer.get(), output_buffer.get(), binding_refs);

  Ref<iree_hal_command_buffer_t> command_buffer;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
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
      CreateZeroedDeviceBuffer(4 * sizeof(int32_t), readback_buffer.out()));
  SemaphoreList copy_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_copy(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      copy_signal, output_buffer.get(), /*source_offset=*/0,
      readback_buffer.get(), /*target_offset=*/0, 4 * sizeof(int32_t),
      IREE_HAL_COPY_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      copy_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  std::vector<int32_t> output_data = ReadBufferData<int32_t>(readback_buffer);
  EXPECT_THAT(output_data, ContainerEq(std::vector<int32_t>{12, 12, 12, 12}));

  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_unmap(
      device_allocator_, output_buffer.get(), /*virtual_offset=*/0,
      recommended_page_size));
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_unmap(
      device_allocator_, input_buffer.get(), /*virtual_offset=*/0,
      recommended_page_size));
}

class BdaSpirvReplayCacheTest : public BdaSpirvTest {};

TEST_P(BdaSpirvReplayCacheTest, TrimDropsIdleCachedNativeReplay) {
  ASSERT_EQ(0, QueryNativeReplayCache(IREE_SV("retained_instance_count")));

  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  ASSERT_EQ(0, QueryNativeReplayCache(IREE_SV("instance_count")));

  Ref<iree_hal_buffer_t> input_buffer;
  Ref<iree_hal_buffer_t> output_buffer;
  CreateInputOutputBuffers(&input_buffer, &output_buffer);

  Ref<iree_hal_command_buffer_t> command_buffer;
  CreateIndirectDispatchCommandBuffer(/*workgroup_count=*/kElementCount,
                                      &command_buffer);

  iree_hal_buffer_binding_t binding_table_entries[2];
  iree_hal_buffer_binding_table_t binding_table = MakeBindingTable(
      input_buffer.get(), output_buffer.get(), binding_table_entries);

  const int64_t initial_trim_count =
      QueryNativeReplayCache(IREE_SV("trim_count"));
  ExecuteCommandBufferAndWait(command_buffer.get(), binding_table);
  ExpectOutput(output_buffer.get());

  ASSERT_GE(QueryNativeReplayCache(IREE_SV("instance_count")), 1);
  ASSERT_GE(QueryNativeReplayCache(IREE_SV("publication_bytes")),
            (int64_t)(2 * sizeof(uint64_t)));

  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  EXPECT_EQ(0, QueryNativeReplayCache(IREE_SV("instance_count")));
  EXPECT_EQ(0, QueryNativeReplayCache(IREE_SV("publication_bytes")));
  EXPECT_GE(QueryNativeReplayCache(IREE_SV("trim_count")) - initial_trim_count,
            1);
}

TEST_P(BdaSpirvReplayCacheTest, CommandBufferSkipsUnchangedBdaPublication) {
  ASSERT_GE(QueryNativeReplayCache(IREE_SV("max_instance_count")), 1);
  ASSERT_EQ(0, QueryNativeReplayCache(IREE_SV("retained_instance_count")));

  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  ASSERT_EQ(0, QueryNativeReplayCache(IREE_SV("instance_count")));

  Ref<iree_hal_buffer_t> input_buffer;
  Ref<iree_hal_buffer_t> output_buffer;
  CreateInputOutputBuffers(&input_buffer, &output_buffer);

  Ref<iree_hal_command_buffer_t> command_buffer;
  CreateIndirectDispatchCommandBuffer(/*workgroup_count=*/kElementCount,
                                      &command_buffer);

  iree_hal_buffer_binding_t binding_table_entries[2];
  iree_hal_buffer_binding_table_t binding_table = MakeBindingTable(
      input_buffer.get(), output_buffer.get(), binding_table_entries);

  const int64_t initial_hit_count =
      QueryNativeReplayCache(IREE_SV("hit_count"));
  const int64_t initial_miss_count =
      QueryNativeReplayCache(IREE_SV("miss_count"));
  const int64_t initial_publication_skip_count =
      QueryNativeReplayCache(IREE_SV("publication_skip_count"));
  const int64_t initial_publication_update_count =
      QueryNativeReplayCache(IREE_SV("publication_update_count"));

  ExecuteCommandBufferAndWait(command_buffer.get(), binding_table);
  ExpectOutput(output_buffer.get());

  const int32_t reset_pattern = 0;
  SemaphoreList fill_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_fill(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      fill_signal, output_buffer.get(), /*target_offset=*/0,
      kDispatchByteLength, &reset_pattern, sizeof(reset_pattern),
      IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      fill_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  ExpectFilledOutputPrefix(output_buffer.get(), /*expected_value=*/0);

  ExecuteCommandBufferAndWait(command_buffer.get(), binding_table);
  ExpectOutput(output_buffer.get());

  EXPECT_GE(QueryNativeReplayCache(IREE_SV("miss_count")) - initial_miss_count,
            1);
  EXPECT_GE(QueryNativeReplayCache(IREE_SV("hit_count")) - initial_hit_count,
            1);
  EXPECT_GE(QueryNativeReplayCache(IREE_SV("publication_skip_count")) -
                initial_publication_skip_count,
            1);
  EXPECT_EQ(initial_publication_update_count,
            QueryNativeReplayCache(IREE_SV("publication_update_count")));
}

TEST_P(BdaSpirvReplayCacheTest, CommandBufferRepublishesChangedBdaPublication) {
  ASSERT_GE(QueryNativeReplayCache(IREE_SV("max_instance_count")), 1);
  ASSERT_EQ(0, QueryNativeReplayCache(IREE_SV("retained_instance_count")));

  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  ASSERT_EQ(0, QueryNativeReplayCache(IREE_SV("instance_count")));

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

  const int64_t initial_hit_count =
      QueryNativeReplayCache(IREE_SV("hit_count"));
  const int64_t initial_miss_count =
      QueryNativeReplayCache(IREE_SV("miss_count"));
  const int64_t initial_publication_skip_count =
      QueryNativeReplayCache(IREE_SV("publication_skip_count"));
  const int64_t initial_publication_update_count =
      QueryNativeReplayCache(IREE_SV("publication_update_count"));

  ExecuteCommandBufferAndWait(command_buffer.get(), binding_table_a);
  ExpectOutput(output_a_buffer.get());
  ExecuteCommandBufferAndWait(command_buffer.get(), binding_table_b);
  ExpectOutput(output_b_buffer.get());

  EXPECT_GE(QueryNativeReplayCache(IREE_SV("miss_count")) - initial_miss_count,
            1);
  EXPECT_GE(QueryNativeReplayCache(IREE_SV("hit_count")) - initial_hit_count,
            1);
  EXPECT_EQ(initial_publication_skip_count,
            QueryNativeReplayCache(IREE_SV("publication_skip_count")));
  EXPECT_GE(QueryNativeReplayCache(IREE_SV("publication_update_count")) -
                initial_publication_update_count,
            1);
}

TEST_P(BdaSpirvReplayCacheTest, ConcurrentExecutionsForkCachedNativeReplay) {
  ASSERT_GE(QueryNativeReplayCache(IREE_SV("max_instance_count")), 2);
  ASSERT_EQ(0, QueryNativeReplayCache(IREE_SV("retained_instance_count")));

  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  ASSERT_EQ(0, QueryNativeReplayCache(IREE_SV("instance_count")));

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
  IREE_ASSERT_OK(iree_hal_device_queue_fill(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      fill_signal, input_buffer.get(), /*target_offset=*/0,
      dispatch_byte_length, &input_pattern, sizeof(input_pattern),
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

  const int64_t initial_create_count =
      QueryNativeReplayCache(IREE_SV("create_count"));
  const int64_t initial_fork_count =
      QueryNativeReplayCache(IREE_SV("fork_count"));
  SemaphoreList execute_a_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      execute_a_signal, command_buffer.get(), binding_table_a,
      IREE_HAL_EXECUTE_FLAG_NONE));
  SemaphoreList execute_b_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_execute(
      device_, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
      execute_b_signal, command_buffer.get(), binding_table_b,
      IREE_HAL_EXECUTE_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      execute_a_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      execute_b_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  ExpectFilledOutputPrefix(output_a_buffer.get(), /*expected_value=*/12);
  ExpectFilledOutputPrefix(output_b_buffer.get(), /*expected_value=*/12);

  EXPECT_GE(
      QueryNativeReplayCache(IREE_SV("create_count")) - initial_create_count,
      2);
  EXPECT_GE(QueryNativeReplayCache(IREE_SV("fork_count")) - initial_fork_count,
            1);
  EXPECT_GE(QueryNativeReplayCache(IREE_SV("peak_instance_count")), 2);

  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  EXPECT_EQ(0, QueryNativeReplayCache(IREE_SV("instance_count")));
}

CTS_REGISTER_TEST_SUITE_WITH_TAGS(BdaSpirvTest, {"vulkan"},
                                  {"vulkan_replay_cache"});
CTS_REGISTER_TEST_SUITE_WITH_TAGS(BdaSpirvReplayCacheTest,
                                  {"vulkan_replay_cache"}, {});

}  // namespace iree::hal::cts
