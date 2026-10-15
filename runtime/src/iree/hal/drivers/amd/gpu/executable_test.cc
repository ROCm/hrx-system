// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/executable.h"

#include <algorithm>
#include <cstring>
#include <semaphore>
#include <set>
#include <thread>
#include <vector>

#include "iree/base/alignment.h"
#include "iree/hal/drivers/amd/gpu/image/hsaco.h"
#include "iree/hal/drivers/amd/gpu/image/hsaco_test_util.h"
#include "iree/hal/memory/maintenance_thread.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amd::gpu {
namespace {

// Native memory is the dependency under control. Parsing, placement,
// relocation, reflection, global views, and HAL ownership use the production
// implementation.
struct NativeMemory {
  // API table borrowed by the executable's native context.
  amdf_api_t api = {};
  // Live native backing, zeroed only by the production image uploader.
  std::vector<uint8_t> bytes;
  // Construction geometry captured from the actual allocation request.
  amdf_memory_create_info_t request = {};
  // Ordinal of the fallible native call to fail, or zero for normal operation.
  uint32_t fail_at = 0;
  // Number of fallible calls performed in this construction.
  uint32_t calls = 0;
  // Number of live native allocations.
  uint32_t allocations = 0;
  // Number of live native host mappings.
  uint32_t mappings = 0;
  // Number of successful image-publication flushes.
  uint32_t flushes = 0;
  // Achieved coherence; write-back cacheability alone does not imply this.
  amdf_memory_flags_t access_flags = AMDF_MEMORY_FLAG_HOST_COHERENT;
  // Thread running the last native allocation, captured for placement checks.
  std::thread::id allocation_thread;
  // Thread running native destruction, captured after the service joins.
  std::thread::id destruction_thread;
  // Optional externally controlled stall inside the native cleanup dependency.
  struct {
    // Signals that the simulated driver has entered its blocking operation.
    std::binary_semaphore* entered = nullptr;
    // Releases that operation without a timer or polling loop.
    std::binary_semaphore* resume = nullptr;
  } cleanup;

  NativeMemory() {
    api.memory_create = Create;
    api.memory_query_info = QueryInfo;
    api.memory_query_access_info = QueryAccess;
    api.memory_query_address = QueryAddress;
    api.memory_query_pair_info = QueryPair;
    api.memory_map = Map;
    api.host_mapping_query_info = MappingInfo;
    api.host_mapping_cache_control = CacheControl;
    api.host_mapping_destroy = Unmap;
    api.memory_destroy = Destroy;
  }
  amdf_memory_scope_t* scope() {
    return reinterpret_cast<amdf_memory_scope_t*>(this);
  }
  amdf_status_t Step() {
    return ++calls == fail_at
               ? amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED)
               : AMDF_STATUS_OK;
  }
  static amdf_status_t AMDF_CALL Create(amdf_memory_scope_t* scope,
                                        const amdf_memory_create_info_t* info,
                                        amdf_memory_t** out_memory) {
    auto& self = *reinterpret_cast<NativeMemory*>(scope);
    const auto status = self.Step();
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    EXPECT_EQ(self.allocations, 0u);
    EXPECT_EQ(info->access_count, 1u);
    EXPECT_EQ(info->accesses[0].requirements.access,
              AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE |
                  AMDF_MEMORY_ACCESS_EXECUTE);
    self.request = *info;
    self.allocation_thread = std::this_thread::get_id();
    self.bytes.assign(info->byte_length, 0xa5);
    ++self.allocations;
    *out_memory = reinterpret_cast<amdf_memory_t*>(&self);
    return AMDF_STATUS_OK;
  }
  static amdf_status_t AMDF_CALL QueryInfo(amdf_memory_t* memory,
                                           amdf_memory_info_t* info) {
    auto& self = *reinterpret_cast<NativeMemory*>(memory);
    const auto status = self.Step();
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    info->byte_length = self.bytes.size();
    info->alignment = self.request.minimum_alignment;
    info->native_allocation_byte_length = self.bytes.size();
    info->native_allocation_granularity = 4096;
    info->flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
    return AMDF_STATUS_OK;
  }
  static amdf_status_t AMDF_CALL QueryAccess(amdf_memory_t* memory,
                                             uint32_t ordinal,
                                             amdf_memory_access_info_t* info) {
    auto& self = *reinterpret_cast<NativeMemory*>(memory);
    const auto status = self.Step();
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    EXPECT_EQ(ordinal, 0u);
    info->flags = self.access_flags | AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
    info->access = self.request.accesses[0].requirements.access;
    return AMDF_STATUS_OK;
  }
  static amdf_status_t AMDF_CALL QueryAddress(amdf_memory_t* memory,
                                              uint32_t ordinal,
                                              amdf_memory_address_kind_t kind,
                                              uint64_t* out_address) {
    auto& self = *reinterpret_cast<NativeMemory*>(memory);
    const auto status = self.Step();
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    EXPECT_EQ(ordinal, 0u);
    EXPECT_EQ(kind, AMDF_MEMORY_ADDRESS_GPU);
    *out_address = UINT64_C(0x1200000000);
    return AMDF_STATUS_OK;
  }
  static amdf_status_t AMDF_CALL QueryPair(const amdf_memory_site_t* producer,
                                           const amdf_memory_site_t* consumer,
                                           amdf_memory_pair_info_t* info) {
    auto* self =
        producer->kind == AMDF_MEMORY_SITE_KIND_HOST
            ? reinterpret_cast<NativeMemory*>(producer->value.host.mapping)
            : reinterpret_cast<NativeMemory*>(producer->value.device.memory);
    const auto status = self->Step();
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    for (const auto* site : {producer, consumer}) {
      if (site->kind == AMDF_MEMORY_SITE_KIND_DEVICE) {
        EXPECT_EQ(site->value.device.access_ordinal, 0u);
        EXPECT_EQ(site->value.device.queue_family_ordinal, 1u);
      }
    }
    info->flags = AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE;
    info->release.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
    info->acquire.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
    return AMDF_STATUS_OK;
  }
  static amdf_status_t AMDF_CALL Map(amdf_memory_t* memory,
                                     const amdf_memory_map_info_t* info,
                                     amdf_host_mapping_t** out_mapping) {
    auto& self = *reinterpret_cast<NativeMemory*>(memory);
    const auto status = self.Step();
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    EXPECT_EQ(info->byte_length, self.bytes.size());
    ++self.mappings;
    *out_mapping = reinterpret_cast<amdf_host_mapping_t*>(&self);
    return AMDF_STATUS_OK;
  }
  static amdf_status_t AMDF_CALL MappingInfo(amdf_host_mapping_t* mapping,
                                             amdf_host_mapping_info_t* info) {
    auto& self = *reinterpret_cast<NativeMemory*>(mapping);
    const auto status = self.Step();
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    info->pointer = self.bytes.data();
    info->byte_length = self.bytes.size();
    info->cacheability = AMDF_HOST_CACHEABILITY_WRITE_BACK;
    return AMDF_STATUS_OK;
  }
  static amdf_status_t AMDF_CALL CacheControl(
      amdf_host_mapping_t* mapping, amdf_host_cache_operation_t operation,
      uint64_t offset, uint64_t length) {
    auto& self = *reinterpret_cast<NativeMemory*>(mapping);
    const auto status = self.Step();
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    EXPECT_LE(offset + length, self.bytes.size());
    if (operation == AMDF_HOST_CACHE_OPERATION_FLUSH) {
      ++self.flushes;
    }
    return AMDF_STATUS_OK;
  }
  static amdf_status_t AMDF_CALL Unmap(amdf_host_mapping_t* mapping) {
    auto& self = *reinterpret_cast<NativeMemory*>(mapping);
    EXPECT_EQ(self.mappings, 1u);
    --self.mappings;
    return AMDF_STATUS_OK;
  }
  static amdf_status_t AMDF_CALL Destroy(amdf_memory_t* memory) {
    auto& self = *reinterpret_cast<NativeMemory*>(memory);
    self.destruction_thread = std::this_thread::get_id();
    if (self.cleanup.entered) {
      self.cleanup.entered->release();
      self.cleanup.resume->acquire();
    }
    EXPECT_EQ(self.mappings, 0u);
    EXPECT_EQ(self.allocations, 1u);
    --self.allocations;
    self.bytes.clear();
    return AMDF_STATUS_OK;
  }
};

struct FailingAllocator {
  // One-based allocation ordinal to reject.
  size_t fail_at;
  // Number of attempted allocations.
  size_t calls = 0;
  // Live host ownership, independent of the native-memory dependency.
  std::set<void*> live;

  static iree_status_t Control(void* data, iree_allocator_command_t command,
                               const void* parameters, void** pointer) {
    auto& self = *static_cast<FailingAllocator*>(data);
    if (command == IREE_ALLOCATOR_COMMAND_FREE) {
      EXPECT_EQ(self.live.erase(*pointer), 1u);
    } else if (++self.calls == self.fail_at) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED);
    }
    const auto system = iree_allocator_system();
    iree_status_t status =
        system.ctl(system.self, command, parameters, pointer);
    if (iree_status_is_ok(status) && command != IREE_ALLOCATOR_COMMAND_FREE) {
      self.live.insert(*pointer);
    }
    return status;
  }
};

class ExecutableTest : public ::testing::TestWithParam<size_t> {
 protected:
  void SetUp() override {
    const auto& file = iree_hal_amd_gpu_hip_images_create()[GetParam()];
    bytes_.assign(file.data, file.data + file.size);
    RecordProperty("target", file.name);
    auto* source = testing::ByteSequence::Create(
        iree_make_const_byte_span(bytes_.data(), bytes_.size()), 0);
    IREE_ASSERT_OK(iree_hal_amd_gpu_hsaco_image_initialize(
        source, iree_allocator_system(), &image_));
    iree_byte_sequence_release(source);
    target_ = image_.target;
    if (target_.version.major < 11) {
      GTEST_SKIP() << "The PM4 launch encoder does not yet admit this ISA";
    }
    if (target_.kind == IREE_HAL_AMD_GPU_TARGET_KIND_GENERIC) {
      const auto physical = target_.version.major == 11  ? IREE_SV("gfx1100")
                            : target_.version.minor == 5 ? IREE_SV("gfx1250")
                                                         : IREE_SV("gfx1201");
      IREE_ASSERT_OK(iree_hal_amd_gpu_target_identity_parse_artifact_key(
          physical, &target_));
    }
    native_families_[1].command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_PM4;
    context_.api = &native_.api;
    context_.families = native_families_;
    // Only identity is borrowed from the parent. These tests never invoke a
    // device method or pretend to qualify public queue execution.
    family_.device = reinterpret_cast<iree_hal_device_t*>(this);
    family_.ordinal = 3;
    family_.memory.domain = this;
    family_.memory.queue_scope_id = 4;
    access_.device = reinterpret_cast<amdf_device_t*>(this);
    access_.requirements.access = AMDF_MEMORY_ACCESS_READ |
                                  AMDF_MEMORY_ACCESS_WRITE |
                                  AMDF_MEMORY_ACCESS_EXECUTE;
    memory_.scope = native_.scope();
    memory_.profile.allocation.byte_length_granularity = 4096;
    memory_.profile.allocation.minimum_alignment = 4096;
    memory_.profile.allocation.maximum_byte_length = UINT64_MAX;
    memory_.access_count = 1;
    memory_.accesses = &access_;
    memory_.host_access =
        AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
    options_ = {
        .target = &target_,
        .native_family_ordinal = 1,
        .memory = &memory_,
        .placement = {IREE_HAL_POOL_PLACEMENT_REQUIRED, 17},
    };
    IREE_ASSERT_OK(iree_hal_memory_maintenance_thread_create(
        {}, iree_allocator_system(), &options_.maintenance));
  }
  void TearDown() override {
    iree_hal_executable_release(executable_);
    iree_hal_memory_maintenance_release(options_.maintenance);
    EXPECT_EQ(native_.mappings, 0u);
    EXPECT_EQ(native_.allocations, 0u);
    iree_hal_amd_gpu_image_deinitialize(&image_);
  }
  void Release() {
    iree_hal_executable_release(executable_);
    executable_ = nullptr;
    iree_hal_memory_maintenance_call(
        options_.maintenance, [](void*) {}, nullptr);
  }
  iree_status_t Load(iree_allocator_t allocator = iree_allocator_system()) {
    iree_hal_executable_load_params_t params;
    iree_hal_executable_load_params_initialize(&params);
    params.executable_data =
        iree_make_const_byte_span(bytes_.data(), bytes_.size());
    return iree_hal_amd_gpu_executable_create(&family_, &context_, &options_,
                                              &params, allocator, &executable_);
  }
  iree_hal_executable_global_t Global(const char* name) {
    bool found = false;
    iree_hal_executable_global_t global;
    IREE_CHECK_OK(iree_hal_executable_try_lookup_global_by_name(
        executable_, iree_make_cstring_view(name), &found, &global));
    EXPECT_TRUE(found);
    return global;
  }
  // Real compiler-produced artifact borrowed only for each load call.
  std::vector<uint8_t> bytes_;
  // Independent parse supplies fixture expectations and physical target.
  iree_hal_amd_gpu_image_t image_ = {};
  // Qualified physical identity for this test's selected artifact.
  iree_hal_amd_gpu_target_identity_t target_ = {};
  // Native allocation/mapping dependency with explicit failure points.
  NativeMemory native_;
  // Parent native owner, borrowed without executable ownership cycles.
  iree_hal_amd_gpu_context_t context_ = {};
  // Native PM4 family differs from the HAL family ordinal.
  amdf_queue_family_info_t native_families_[2] = {};
  // Sealed-group identity and exact family selected by the parent.
  iree_hal_queue_family_t family_ = {};
  // Qualified native access for the executable's physical device.
  amdf_memory_device_access_t access_ = {};
  // Explicit backing policy; no implicit allocator or pool.
  iree_hal_amd_fabric_memory_options_t memory_ = {};
  // Call-only construction arguments.
  iree_hal_amd_gpu_executable_options_t options_ = {};
  // Owning HAL resource reference.
  iree_hal_executable_t* executable_ = nullptr;
};

TEST_P(ExecutableTest, OwnsArtifactAndReflectsPreparedFunctions) {
  IREE_ASSERT_OK(Load());
  EXPECT_EQ(native_.flushes, 1u);
  EXPECT_NE(native_.allocation_thread, std::this_thread::get_id());
  std::fill(bytes_.begin(), bytes_.end(), 0xa5);
  iree_hal_amd_gpu_image_deinitialize(&image_);
  EXPECT_EQ(iree_hal_executable_queue_family(executable_), &family_);
  EXPECT_EQ(iree_hal_executable_function_count(executable_), 5u);
  for (auto name : {"image_transform", "image_increment", "image_accumulate",
                    "image_counts", "image_geometry"}) {
    iree_hal_executable_function_t token;
    IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
        executable_, iree_make_cstring_view(name), &token));
    iree_hal_executable_function_info_t info;
    IREE_ASSERT_OK(
        iree_hal_executable_function_info(executable_, token, &info));
    EXPECT_TRUE(
        iree_string_view_equal(info.name, iree_make_cstring_view(name)));
    const bool count_producer = std::strcmp(name, "image_counts") == 0;
    const bool geometry_consumer = std::strcmp(name, "image_geometry") == 0;
    EXPECT_EQ(info.binding_count, count_producer ? 2u : 1u);
    EXPECT_EQ(info.constant_byte_length, count_producer ? 0u : 4u);
    EXPECT_EQ(info.parameter_count, 2u);
    iree_hal_executable_function_parameter_t parameters[2];
    IREE_ASSERT_OK(iree_hal_executable_function_parameters(executable_, token,
                                                           2, parameters));
    EXPECT_EQ(parameters[0].native_abi_offset, 0u);
    EXPECT_EQ(parameters[1].native_abi_offset, 8u);
    const iree_hal_amd_gpu_pm4_function_t* function = nullptr;
    IREE_ASSERT_OK(
        iree_hal_amd_gpu_executable_lookup_pm4(executable_, token, &function));
    EXPECT_EQ(function->kernarg->native_byte_length,
              count_producer ? 16u : 12u);
    if (geometry_consumer) {
      EXPECT_GT(function->kernarg->byte_length,
                function->kernarg->native_byte_length);
      EXPECT_EQ(function->kernarg->indirect_copy_count, 1u);
    } else {
      EXPECT_EQ(function->kernarg->byte_length, count_producer ? 16u : 12u);
    }
    EXPECT_EQ(function->launch.requirements, 0u);
  }
}

TEST_P(ExecutableTest, GlobalsPreserveRelocationsPermissionsAndBorrowIdentity) {
  IREE_ASSERT_OK(Load());
  const auto global = Global("image_value");
  iree_hal_buffer_t* value = nullptr;
  IREE_ASSERT_OK(
      iree_hal_executable_global_buffer(executable_, global, &value));
  iree_hal_buffer_t* same = nullptr;
  IREE_ASSERT_OK(iree_hal_executable_global_buffer(executable_, global, &same));
  EXPECT_EQ(same, value);
  EXPECT_EQ(iree_hal_buffer_byte_length(value), 4u);
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(value, IREE_HAL_MAPPING_MODE_SCOPED,
                                           IREE_HAL_MEMORY_ACCESS_READ, 0, 0, 4,
                                           &mapping));
  EXPECT_EQ(iree_unaligned_load_le_u32(mapping.contents.data), 19u);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
  iree_hal_buffer_t* pointer = nullptr;
  IREE_ASSERT_OK(iree_hal_executable_global_buffer(
      executable_, Global("image_pointer"), &pointer));
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      pointer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ, 0, 0,
      8, &mapping));
  EXPECT_EQ(iree_unaligned_load_le_u64(mapping.contents.data),
            UINT64_C(0x1200000000) + value->memory.binding_offset);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
  const auto* descriptor = iree_hal_amd_gpu_image_find_symbol(
      &image_, image_.metadata.kernels[0].symbol_name);
  ASSERT_NE(descriptor, nullptr);
  ASSERT_EQ(descriptor->access, IREE_HAL_AMD_GPU_IMAGE_ACCESS_READ);
  bool found = false;
  iree_hal_executable_global_t descriptor_token;
  IREE_ASSERT_OK(iree_hal_executable_try_lookup_global_by_name(
      executable_, descriptor->name, &found, &descriptor_token));
  ASSERT_TRUE(found);
  iree_hal_buffer_t* read_only = nullptr;
  IREE_ASSERT_OK(iree_hal_executable_global_buffer(
      executable_, descriptor_token, &read_only));
  EXPECT_EQ(read_only->memory.contract->host.access,
            IREE_HAL_MEMORY_ACCESS_READ);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED,
                        iree_hal_buffer_map_range(
                            read_only, IREE_HAL_MAPPING_MODE_SCOPED,
                            IREE_HAL_MEMORY_ACCESS_WRITE, 0, 0, 4, &mapping));
  EXPECT_NE(value->memory.contract, read_only->memory.contract);
  EXPECT_EQ(value->memory.contract, pointer->memory.contract);
  EXPECT_EQ(value->memory.contract->placement.mode,
            IREE_HAL_POOL_PLACEMENT_REQUIRED);
  EXPECT_EQ(value->memory.contract->placement.node, 17u);
  EXPECT_EQ(value->memory.contract->domain, family_.memory.domain);
  IREE_EXPECT_OK(iree_hal_buffer_validate_family_usage(
      value, &family_, IREE_HAL_BUFFER_USAGE_STORAGE_READ));
  auto excluded_family = family_;
  excluded_family.memory.queue_scope_id = 2;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_buffer_validate_family_usage(
          value, &excluded_family, IREE_HAL_BUFFER_USAGE_STORAGE_READ));
}

TEST_P(ExecutableTest, HostMappingPermissionsStayWithinTheNativeView) {
  memory_.host_access = AMDF_MEMORY_MAP_FLAG_WRITE;
  IREE_ASSERT_OK(Load());
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_executable_global_buffer(
      executable_, Global("image_value"), &buffer));
  EXPECT_EQ(buffer->memory.contract->host.access, IREE_HAL_MEMORY_ACCESS_WRITE);
  iree_hal_buffer_mapping_t mapping;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED,
                        iree_hal_buffer_map_range(
                            buffer, IREE_HAL_MAPPING_MODE_SCOPED,
                            IREE_HAL_MEMORY_ACCESS_READ, 0, 0, 4, &mapping));
}

TEST_P(ExecutableTest, CapturesExportClassesAndPreservesGlobalRange) {
  iree_hal_external_storage_class_t storage_class = {
      .type = IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER,
  };
  const auto requested_class = storage_class;
  memory_.exports = {1, &storage_class};
  IREE_ASSERT_OK(Load());
  storage_class = {};
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_executable_global_buffer(
      executable_, Global("image_value"), &buffer));
  iree_hal_external_storage_t storage;
  iree_hal_buffer_range_t range;
  IREE_ASSERT_OK(iree_hal_buffer_export(
      buffer, iree_hal_make_buffer_range(0, IREE_HAL_WHOLE_BUFFER),
      requested_class, &storage, &range));
  EXPECT_EQ(storage.payload.host_pointer, native_.bytes.data());
  EXPECT_EQ(storage.byte_length, native_.bytes.size());
  EXPECT_EQ(range.byte_offset, buffer->memory.binding_offset);
  EXPECT_EQ(range.byte_length, 4u);
  EXPECT_EQ(iree_unaligned_load_le_u32(
                static_cast<uint8_t*>(storage.payload.host_pointer) +
                range.byte_offset),
            19u);
  iree_hal_external_storage_release(&storage);
}

TEST_P(ExecutableTest, InvalidTokensCannotSelectImageStorage) {
  IREE_ASSERT_OK(Load());
  const iree_hal_amd_gpu_pm4_function_t* function = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_amd_gpu_executable_lookup_pm4(
          executable_, iree_hal_executable_function_from_value(UINT64_MAX),
          &function));
  EXPECT_EQ(function, nullptr);
  iree_hal_executable_global_info_t info;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_executable_global_info(
          executable_, iree_hal_executable_global_from_value(UINT64_MAX),
          &info));
  bool found = true;
  iree_hal_executable_global_t global;
  IREE_ASSERT_OK(iree_hal_executable_try_lookup_global_by_name(
      executable_, IREE_SV("absent"), &found, &global));
  EXPECT_FALSE(found);
  EXPECT_FALSE(iree_hal_executable_global_is_valid(global));
}

TEST_P(ExecutableTest, ConcurrentGlobalQueriesPublishOneOwnedView) {
  IREE_ASSERT_OK(Load());
  const auto token = Global("image_zero");
  std::vector<iree_hal_buffer_t*> buffers(16);
  std::vector<std::thread> threads;
  for (size_t i = 0; i < buffers.size(); ++i) {
    threads.emplace_back([&, i] {
      IREE_EXPECT_OK(
          iree_hal_executable_global_buffer(executable_, token, &buffers[i]));
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  ASSERT_NE(buffers[0], nullptr);
  for (auto* buffer : buffers) {
    EXPECT_EQ(buffer, buffers[0]);
  }
}

TEST_P(ExecutableTest, WriteBackHostMappingDoesNotInventCoherence) {
  native_.access_flags = 0;
  IREE_ASSERT_OK(Load());
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_executable_global_buffer(
      executable_, Global("image_value"), &buffer));
  EXPECT_FALSE(iree_any_bit_set(iree_hal_buffer_memory_type(buffer),
                                IREE_HAL_MEMORY_TYPE_HOST_COHERENT));
}

TEST_P(ExecutableTest, RejectsWrongTargetBeforeAllocatingNativeStorage) {
  IREE_ASSERT_OK(iree_hal_amd_gpu_target_identity_parse_artifact_key(
      IREE_SV("gfx942"), &target_));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INCOMPATIBLE, Load());
  EXPECT_EQ(executable_, nullptr);
  EXPECT_EQ(native_.calls, 0u);
}

TEST_P(ExecutableTest, RuntimeConfigurationRequiresProvisionedServices) {
  const auto original = bytes_;
  constexpr char source[] = "image_runtime_config";
  for (const char* name :
       {"iree_asan_config", "iree_tsan_config", "iree_feedback_config"}) {
    SCOPED_TRACE(name);
    bytes_ = original;
    size_t replacements = 0;
    auto cursor = bytes_.begin();
    while ((cursor = std::search(cursor, bytes_.end(), std::begin(source),
                                 std::end(source))) != bytes_.end()) {
      std::fill_n(cursor, sizeof(source), uint8_t{0});
      std::memcpy(&*cursor, name, std::strlen(name));
      cursor += sizeof(source);
      ++replacements;
    }
    ASSERT_GT(replacements, 0u);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED, Load());
    EXPECT_EQ(executable_, nullptr);
    EXPECT_EQ(native_.calls, 0u);
  }
}

TEST_P(ExecutableTest, NativeConstructionFailuresReleasePartialOwnership) {
  IREE_ASSERT_OK(Load());
  const uint32_t call_count = native_.calls;
  Release();
  for (uint32_t i = 1; i <= call_count; ++i) {
    SCOPED_TRACE(i);
    native_.fail_at = i;
    native_.calls = 0;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Load());
    EXPECT_EQ(executable_, nullptr);
    EXPECT_EQ(native_.allocations, 0u);
    EXPECT_EQ(native_.mappings, 0u);
  }
}

TEST_P(ExecutableTest, HostConstructionFailuresReleasePartialOwnership) {
  FailingAllocator baseline = {.fail_at = 0};
  IREE_ASSERT_OK(Load({&baseline, FailingAllocator::Control}));
  Release();
  ASSERT_TRUE(baseline.live.empty());
  for (size_t i = 1; i <= baseline.calls; ++i) {
    SCOPED_TRACE(i);
    FailingAllocator state = {.fail_at = i};
    const iree_allocator_t allocator = {&state, FailingAllocator::Control};
    IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Load(allocator));
    EXPECT_EQ(executable_, nullptr);
    EXPECT_TRUE(state.live.empty());
    EXPECT_EQ(native_.allocations, 0u);
    EXPECT_EQ(native_.mappings, 0u);
  }
}

TEST_P(ExecutableTest, FinalReleaseDoesNotWaitForNativeCleanup) {
  IREE_ASSERT_OK(Load());
  std::binary_semaphore entered{0};
  std::binary_semaphore resume{0};
  native_.cleanup = {&entered, &resume};
  iree_hal_executable_release(executable_);
  executable_ = nullptr;
  entered.acquire();
  EXPECT_EQ(native_.allocations, 1u);
  EXPECT_EQ(native_.mappings, 0u);
  EXPECT_EQ(native_.destruction_thread, native_.allocation_thread);
  EXPECT_NE(native_.destruction_thread, std::this_thread::get_id());
  resume.release();
  iree_hal_memory_maintenance_call(options_.maintenance, [](void*) {}, nullptr);
  EXPECT_EQ(native_.allocations, 0u);
}

INSTANTIATE_TEST_SUITE_P(HipImages, ExecutableTest,
                         ::testing::Range(size_t{0},
                                          iree_hal_amd_gpu_hip_images_size()));
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ExecutableTest);

}  // namespace
}  // namespace iree::hal::amd::gpu
