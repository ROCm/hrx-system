// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/parameter_index_provider.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/task/registration/driver_module.h"
#include "iree/io/file_handle.h"
#include "iree/io/parameter_index.h"
#include "iree/io/parameter_provider.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

template <typename T, void (*Release)(T*)>
class Ref {
 public:
  Ref() = default;
  Ref(const Ref&) = delete;
  Ref& operator=(const Ref&) = delete;

  ~Ref() { reset(); }

  T* get() const { return value_; }

  T** out() {
    reset();
    return &value_;
  }

  void reset(T* value = nullptr) {
    if (value_) {
      Release(value_);
    }
    value_ = value;
  }

 private:
  T* value_ = nullptr;
};

static iree_hal_device_group_t* CreateTaskDeviceGroup() {
  Ref<iree_async_proactor_pool_t, iree_async_proactor_pool_release>
      proactor_pool;
  IREE_CHECK_OK(iree_async_proactor_pool_create(
      /*node_count=*/1, /*node_ids=*/nullptr,
      iree_async_proactor_pool_options_default(), iree_allocator_system(),
      proactor_pool.out()));

  iree_hal_driver_registry_t* registry = nullptr;
  IREE_CHECK_OK(
      iree_hal_driver_registry_allocate(iree_allocator_system(), &registry));
  IREE_CHECK_OK(iree_hal_task_driver_module_register(registry));

  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = proactor_pool.get();

  iree_hal_device_t* device = nullptr;
  iree_status_t status =
      iree_hal_create_device(registry, IREE_SV("task"), &create_params,
                             iree_allocator_system(), &device);
  iree_hal_driver_registry_free(registry);
  IREE_CHECK_OK(status);

  Ref<iree_async_frontier_tracker_t, iree_async_frontier_tracker_release>
      frontier_tracker;
  IREE_CHECK_OK(iree_async_frontier_tracker_create(
      iree_async_frontier_tracker_options_default(), iree_allocator_system(),
      frontier_tracker.out()));

  iree_hal_device_group_t* device_group = nullptr;
  IREE_CHECK_OK(iree_hal_device_group_create_from_device(
      device, frontier_tracker.get(), iree_allocator_system(), &device_group));
  iree_hal_device_release(device);
  return device_group;
}

static iree_status_t AddSplatEntry(iree_io_parameter_index_t* index,
                                   iree_string_view_t key, uint64_t length,
                                   uint8_t pattern) {
  iree_io_parameter_index_entry_t entry;
  std::memset(&entry, 0, sizeof(entry));
  entry.key = key;
  entry.length = length;
  entry.type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT;
  entry.storage.splat.pattern_length = 1;
  entry.storage.splat.pattern[0] = pattern;
  return iree_io_parameter_index_add(index, &entry);
}

static iree_status_t AddFileEntry(iree_io_parameter_index_t* index,
                                  iree_string_view_t key, uint64_t offset,
                                  uint64_t length,
                                  iree_io_file_handle_t* handle) {
  iree_io_parameter_index_entry_t entry;
  std::memset(&entry, 0, sizeof(entry));
  entry.key = key;
  entry.length = length;
  entry.type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_FILE;
  entry.storage.file.handle = handle;
  entry.storage.file.offset = offset;
  return iree_io_parameter_index_add(index, &entry);
}

typedef struct ParameterRequest {
  // Parameter key.
  iree_string_view_t key;
  // Parameter and buffer span.
  iree_io_parameter_span_t span;
} ParameterRequest;

static iree_status_t EnumerateParameterRequest(
    void* user_data, iree_host_size_t i, iree_string_view_t* out_key,
    iree_io_parameter_span_t* out_span) {
  const ParameterRequest* requests =
      static_cast<const ParameterRequest*>(user_data);
  *out_key = requests[i].key;
  *out_span = requests[i].span;
  return iree_ok_status();
}

static iree_hal_buffer_t* AllocateTransferBuffer(
    iree_hal_pool_t* pool, iree_device_size_t byte_length) {
  iree_hal_buffer_t* buffer = nullptr;
  IREE_CHECK_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(byte_length, 1),
      iree_infinite_timeout(), &buffer));
  return buffer;
}

static void WriteBufferBytes(iree_hal_buffer_t* buffer,
                             iree_const_byte_span_t source) {
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_WRITE,
      IREE_HAL_BUFFER_MAP_FLAG_DISCARD, 0, source.data_length, &mapping));
  iree_byte_span_t span;
  IREE_ASSERT_OK(iree_hal_buffer_mapping_subspan(
      &mapping, IREE_HAL_MEMORY_ACCESS_WRITE, 0, source.data_length, &span));
  std::memcpy(span.data, source.data, source.data_length);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
}

static void ExpectBufferBytes(iree_hal_buffer_t* buffer,
                              iree_device_size_t byte_length,
                              uint8_t expected_value) {
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
      IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, byte_length, &mapping));
  iree_byte_span_t span;
  IREE_ASSERT_OK(iree_hal_buffer_mapping_subspan(
      &mapping, IREE_HAL_MEMORY_ACCESS_READ, 0, byte_length, &span));
  for (iree_host_size_t i = 0; i < byte_length; ++i) {
    EXPECT_EQ(span.data[i], expected_value);
  }
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
}

static void ExpectBufferBytesEqual(iree_hal_buffer_t* buffer,
                                   iree_const_byte_span_t expected) {
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
      IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, expected.data_length, &mapping));
  iree_byte_span_t span;
  IREE_ASSERT_OK(iree_hal_buffer_mapping_subspan(
      &mapping, IREE_HAL_MEMORY_ACCESS_READ, 0, expected.data_length, &span));
  EXPECT_EQ(span.data_length, expected.data_length);
  EXPECT_EQ(0, std::memcmp(span.data, expected.data, expected.data_length));
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
}

static iree_status_t CreateTransferPool(iree_hal_device_group_t* device_group,
                                        bool admit_host_imports,
                                        iree_hal_pool_t** out_pool) {
  iree_hal_device_t* device = iree_hal_device_group_device_at(device_group, 0);
  iree_hal_pool_family_access_t family = {
      .family = iree_hal_device_queue_family(device, 0),
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
  };
  iree_hal_pool_scope_t scope = {
      .family_count = 1,
      .families = &family,
      .host =
          {
              .access = IREE_HAL_MEMORY_ACCESS_ALL,
              .modes = IREE_HAL_MAPPING_MODE_SCOPED,
              .properties = (IREE_HAL_HOST_MEMORY_PROPERTY_READ_CACHED |
                             IREE_HAL_HOST_MEMORY_PROPERTY_WRITE_BACK),
          },
  };
  const iree_hal_external_storage_class_t host_pointer_class =
      iree_hal_make_external_storage_class(
          IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER);
  const iree_hal_pool_import_class_t import_class = {
      .storage_class = host_pointer_class,
  };
  iree_hal_slab_pool_options_t options;
  iree_hal_slab_pool_options_initialize(&options);
  if (admit_host_imports) {
    options.imports.class_count = 1;
    options.imports.classes = &import_class;
  }
  options.preferences.host = scope.host;
  return iree_hal_slab_pool_create(device_group, scope, &options,
                                   iree_allocator_system(), out_pool);
}

// Observes the provider's real submissions while forwarding all execution to
// the task HAL. Tests hold the producer gate while inspecting the captured
// dependency graph, then verify the resulting byte movement on the real queue.
class SubmissionTrace {
 public:
  enum class Kind { kAllocate, kTransfer, kRead, kWrite };
  using Timepoint = std::pair<iree_hal_semaphore_t*, uint64_t>;
  struct Submission {
    // Operation observed at the public queue boundary.
    Kind kind;
    // Borrowed timepoints kept alive by the gated queue operations.
    std::vector<Timepoint> waits;
    // Borrowed completion timepoints captured by the real queue.
    std::vector<Timepoint> signals;
  };

  explicit SubmissionTrace(iree_hal_queue_t* target) : target_(target) {
    iree_hal_queue_retain(target_);
    const iree_hal_queue_params_t params = {
        .priority = target->priority,
        .features = target->features,
        .execution_resources = target->execution_resources,
    };
    static const iree_hal_queue_vtable_t vtable = {
        .destroy = Destroy,
        .alloca = Allocate,
        .transfer = Transfer,
        .read = Read,
        .write = Write,
    };
    iree_hal_queue_initialize(iree_hal_queue_family(target), &params, &vtable,
                              &base_);
  }
  ~SubmissionTrace() { iree_hal_queue_release(&base_); }
  iree_hal_queue_t* queue() { return &base_; }
  const std::vector<Submission>& submissions() const { return submissions_; }

 private:
  static SubmissionTrace* Cast(iree_hal_queue_t* queue) {
    return reinterpret_cast<SubmissionTrace*>(queue);
  }
  static void Destroy(iree_hal_queue_t* queue) {
    iree_hal_queue_release(Cast(queue)->target_);
  }
  void Record(Kind kind, iree_hal_semaphore_list_t waits,
              iree_hal_semaphore_list_t signals) {
    Submission submission = {kind, {}, {}};
    for (iree_host_size_t i = 0; i < waits.count; ++i) {
      submission.waits.emplace_back(waits.semaphores[i],
                                    waits.payload_values[i]);
    }
    for (iree_host_size_t i = 0; i < signals.count; ++i) {
      submission.signals.emplace_back(signals.semaphores[i],
                                      signals.payload_values[i]);
    }
    submissions_.push_back(std::move(submission));
  }
  static iree_status_t Allocate(
      iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
      iree_hal_semaphore_list_t signals, iree_hal_pool_t* pool,
      iree_host_size_t count,
      const iree_hal_pool_reservation_request_t* requests,
      iree_hal_buffer_t** buffers) {
    auto* self = Cast(queue);
    self->Record(Kind::kAllocate, waits, signals);
    return iree_hal_queue_alloca(self->target_, waits, signals, pool, count,
                                 requests, buffers);
  }
  static iree_status_t Transfer(iree_hal_queue_t* queue,
                                iree_hal_semaphore_list_t waits,
                                iree_hal_semaphore_list_t signals,
                                iree_host_size_t count,
                                const iree_hal_transfer_operation_t* operations,
                                const iree_hal_queue_barriers_t* barriers) {
    auto* self = Cast(queue);
    self->Record(Kind::kTransfer, waits, signals);
    return iree_hal_queue_transfer(self->target_, waits, signals, count,
                                   operations, barriers);
  }
  static iree_status_t Read(
      iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
      iree_hal_semaphore_list_t signals, iree_hal_file_t* file, uint64_t offset,
      iree_hal_buffer_t* buffer, iree_device_size_t buffer_offset,
      iree_device_size_t length, const iree_hal_queue_barriers_t* barriers,
      iree_hal_read_flags_t flags) {
    auto* self = Cast(queue);
    self->Record(Kind::kRead, waits, signals);
    return iree_hal_queue_read(self->target_, waits, signals, file, offset,
                               buffer, buffer_offset, length, barriers, flags);
  }
  static iree_status_t Write(
      iree_hal_queue_t* queue, iree_hal_semaphore_list_t waits,
      iree_hal_semaphore_list_t signals, iree_hal_buffer_t* buffer,
      iree_device_size_t buffer_offset, iree_hal_file_t* file, uint64_t offset,
      iree_device_size_t length, const iree_hal_queue_barriers_t* barriers,
      iree_hal_write_flags_t flags) {
    auto* self = Cast(queue);
    self->Record(Kind::kWrite, waits, signals);
    return iree_hal_queue_write(self->target_, waits, signals, buffer,
                                buffer_offset, file, offset, length, barriers,
                                flags);
  }

  // Queue prefix; the vtable casts this first member back to its owner.
  iree_hal_queue_t base_;
  // Retained real queue handling resources, execution and synchronization.
  iree_hal_queue_t* target_;
  // Submission-time dependency snapshots, inspected before opening the gate.
  std::vector<Submission> submissions_;
};

class ParameterIndexProviderTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    device_group_ = CreateTaskDeviceGroup();
    IREE_ASSERT_OK(CreateTransferPool(device_group_,
                                      /*admit_host_imports=*/false,
                                      &transfer_pool_));
  }

  static void TearDownTestSuite() {
    iree_hal_pool_release(transfer_pool_);
    transfer_pool_ = nullptr;
    iree_hal_device_group_release(device_group_);
    device_group_ = nullptr;
  }

  static iree_hal_device_t* device() {
    return iree_hal_device_group_device_at(device_group_, 0);
  }

  static iree_hal_queue_t* queue() {
    return iree_hal_device_queue(device(), /*family_ordinal=*/0,
                                 /*queue_ordinal=*/0);
  }

  static iree_hal_device_group_t* device_group() { return device_group_; }

  static iree_hal_pool_t* transfer_pool() { return transfer_pool_; }

 private:
  static iree_hal_device_group_t* device_group_;
  static iree_hal_pool_t* transfer_pool_;
};

iree_hal_device_group_t* ParameterIndexProviderTest::device_group_ = nullptr;
iree_hal_pool_t* ParameterIndexProviderTest::transfer_pool_ = nullptr;

static iree_status_t CaptureLoadedBuffer(void* user_data, iree_host_size_t i,
                                         iree_hal_buffer_t* buffer) {
  if (i != 0) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "test emitter expected one buffer");
  }
  iree_hal_buffer_retain(buffer);
  *static_cast<iree_hal_buffer_t**>(user_data) = buffer;
  return iree_ok_status();
}

TEST_F(ParameterIndexProviderTest, LoadImportsExactParameterRange) {
  uint8_t source_data[12] = {
      0x10, 0x11, 0x12, 0x13, 0x20, 0x21, 0x22, 0x23, 0x30, 0x31, 0x32, 0x33,
  };
  Ref<iree_io_file_handle_t, iree_io_file_handle_release> source_handle;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ,
      iree_make_byte_span(source_data, sizeof(source_data)),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      source_handle.out()));

  Ref<iree_io_parameter_index_t, iree_io_parameter_index_release> index;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), index.out()));
  IREE_ASSERT_OK(AddFileEntry(index.get(), IREE_SV("weights"),
                              /*offset=*/2, /*length=*/8, source_handle.get()));
  Ref<iree_io_parameter_provider_t, iree_io_parameter_provider_release>
      provider;
  IREE_ASSERT_OK(iree_io_parameter_index_provider_create(
      IREE_SV("model"), index.get(),
      IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS,
      iree_allocator_system(), provider.out()));

  Ref<iree_hal_pool_t, iree_hal_pool_release> pool;
  IREE_ASSERT_OK(CreateTransferPool(device_group(), /*admit_host_imports=*/true,
                                    pool.out()));
  Ref<iree_hal_semaphore_t, iree_hal_semaphore_release> signal;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, signal.out()));

  ParameterRequest request = {
      .key = IREE_SV("weights"),
      .span = {.parameter_offset = 3, .buffer_offset = 0, .length = 4},
  };
  uint64_t signal_value = 1;
  iree_hal_semaphore_t* signal_ptr = signal.get();
  iree_hal_semaphore_list_t signal_list = {
      .count = 1,
      .semaphores = &signal_ptr,
      .payload_values = &signal_value,
  };
  Ref<iree_hal_buffer_t, iree_hal_buffer_release> loaded_buffer;
  iree_io_parameter_emitter_t emitter = {
      .fn = CaptureLoadedBuffer,
      .user_data = loaded_buffer.out(),
  };
  IREE_ASSERT_OK(iree_io_parameter_provider_load(
      provider.get(), device(), queue(), pool.get(),
      iree_hal_semaphore_list_empty(), signal_list, IREE_SV("model"),
      /*target_min_alignment=*/1, /*count=*/1,
      {.fn = EnumerateParameterRequest, .user_data = &request}, emitter));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(signal.get(), signal_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  ASSERT_NE(loaded_buffer.get(), nullptr);
  EXPECT_EQ(iree_hal_buffer_byte_length(loaded_buffer.get()), 4u);
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      loaded_buffer.get(), IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ, IREE_HAL_BUFFER_MAP_FLAG_NONE, 0,
      IREE_HAL_WHOLE_BUFFER, &mapping));
  EXPECT_EQ(mapping.contents.data, source_data + 5);
  EXPECT_EQ(0, std::memcmp(mapping.contents.data, source_data + 5, 4));
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
}

TEST_F(ParameterIndexProviderTest, LoadAllocatesFullTargetSpan) {
  Ref<iree_io_parameter_index_t, iree_io_parameter_index_release> index;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), index.out()));
  IREE_ASSERT_OK(AddSplatEntry(index.get(), IREE_SV("padding"), 4, 0xAB));
  Ref<iree_io_parameter_provider_t, iree_io_parameter_provider_release>
      provider;
  IREE_ASSERT_OK(iree_io_parameter_index_provider_create(
      IREE_SV("model"), index.get(),
      IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS,
      iree_allocator_system(), provider.out()));

  Ref<iree_hal_pool_t, iree_hal_pool_release> pool;
  IREE_ASSERT_OK(CreateTransferPool(device_group(),
                                    /*admit_host_imports=*/false, pool.out()));
  Ref<iree_hal_semaphore_t, iree_hal_semaphore_release> signal;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, signal.out()));

  ParameterRequest request = {
      .key = IREE_SV("padding"),
      .span = {.parameter_offset = 0, .buffer_offset = 3, .length = 4},
  };
  uint64_t signal_value = 1;
  iree_hal_semaphore_t* signal_ptr = signal.get();
  iree_hal_semaphore_list_t signal_list = {
      .count = 1,
      .semaphores = &signal_ptr,
      .payload_values = &signal_value,
  };
  Ref<iree_hal_buffer_t, iree_hal_buffer_release> loaded_buffer;
  IREE_ASSERT_OK(iree_io_parameter_provider_load(
      provider.get(), device(), queue(), pool.get(),
      iree_hal_semaphore_list_empty(), signal_list, IREE_SV("model"),
      /*target_min_alignment=*/16, /*count=*/1,
      {.fn = EnumerateParameterRequest, .user_data = &request},
      {.fn = CaptureLoadedBuffer, .user_data = loaded_buffer.out()}));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(signal.get(), signal_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  ASSERT_NE(loaded_buffer.get(), nullptr);
  EXPECT_EQ(iree_hal_buffer_byte_length(loaded_buffer.get()), 7u);
  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      loaded_buffer.get(), IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ, IREE_HAL_BUFFER_MAP_FLAG_NONE, 3, 4,
      &mapping));
  for (iree_host_size_t i = 0; i < mapping.contents.data_length; ++i) {
    EXPECT_EQ(mapping.contents.data[i], 0xAB);
  }
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
}

TEST_F(ParameterIndexProviderTest, GatherBatchPreservesGroupSignals) {
  iree_hal_device_t* device = this->device();

  Ref<iree_io_parameter_index_t, iree_io_parameter_index_release> index;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), index.out()));
  IREE_ASSERT_OK(AddSplatEntry(index.get(), IREE_SV("first"), 4, 0xAB));
  IREE_ASSERT_OK(AddSplatEntry(index.get(), IREE_SV("second"), 4, 0xCD));

  Ref<iree_io_parameter_provider_t, iree_io_parameter_provider_release>
      provider;
  IREE_ASSERT_OK(iree_io_parameter_index_provider_create(
      IREE_SV("model"), index.get(),
      IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS,
      iree_allocator_system(), provider.out()));

  Ref<iree_hal_buffer_t, iree_hal_buffer_release> first_buffer;
  first_buffer.reset(AllocateTransferBuffer(transfer_pool(), 4));
  Ref<iree_hal_buffer_t, iree_hal_buffer_release> second_buffer;
  second_buffer.reset(AllocateTransferBuffer(transfer_pool(), 4));

  Ref<iree_hal_semaphore_t, iree_hal_semaphore_release> first_signal;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, first_signal.out()));
  Ref<iree_hal_semaphore_t, iree_hal_semaphore_release> second_signal;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, second_signal.out()));

  ParameterRequest first_request = {
      .key = IREE_SV("first"),
      .span = {.parameter_offset = 0, .buffer_offset = 0, .length = 4},
  };
  ParameterRequest second_request = {
      .key = IREE_SV("second"),
      .span = {.parameter_offset = 0, .buffer_offset = 0, .length = 4},
  };

  uint64_t first_signal_value = 1;
  uint64_t second_signal_value = 1;
  iree_hal_semaphore_t* first_signal_ptr = first_signal.get();
  iree_hal_semaphore_t* second_signal_ptr = second_signal.get();
  iree_io_parameter_gather_t gathers[2] = {
      {
          .source_scope = IREE_SV("model"),
          .target_buffer = first_buffer.get(),
          .count = 1,
          .enumerator =
              {
                  .fn = EnumerateParameterRequest,
                  .user_data = &first_request,
              },
          .wait_semaphore_list = iree_hal_semaphore_list_empty(),
          .signal_semaphore_list =
              {
                  .count = 1,
                  .semaphores = &first_signal_ptr,
                  .payload_values = &first_signal_value,
              },
      },
      {
          .source_scope = IREE_SV("model"),
          .target_buffer = second_buffer.get(),
          .count = 1,
          .enumerator =
              {
                  .fn = EnumerateParameterRequest,
                  .user_data = &second_request,
              },
          .wait_semaphore_list = iree_hal_semaphore_list_empty(),
          .signal_semaphore_list =
              {
                  .count = 1,
                  .semaphores = &second_signal_ptr,
                  .payload_values = &second_signal_value,
              },
      },
  };

  IREE_ASSERT_OK(iree_io_parameter_provider_gather_batch(
      provider.get(), device, queue(), IREE_ARRAYSIZE(gathers), gathers));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(first_signal.get(), first_signal_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      second_signal.get(), second_signal_value, iree_infinite_timeout(),
      IREE_ASYNC_WAIT_FLAG_NONE));

  ExpectBufferBytes(first_buffer.get(), 4, 0xAB);
  ExpectBufferBytes(second_buffer.get(), 4, 0xCD);
}

TEST_F(ParameterIndexProviderTest, GatherBatchReadsAdjacentFileSpans) {
  iree_hal_device_t* device = this->device();

  uint8_t source_data[12] = {
      0x10, 0x11, 0x12, 0x13, 0x20, 0x21, 0x22, 0x23, 0x30, 0x31, 0x32, 0x33,
  };
  Ref<iree_io_file_handle_t, iree_io_file_handle_release> source_handle;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ,
      iree_make_byte_span(source_data, sizeof(source_data)),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      source_handle.out()));

  Ref<iree_io_parameter_index_t, iree_io_parameter_index_release> index;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), index.out()));
  IREE_ASSERT_OK(
      AddFileEntry(index.get(), IREE_SV("first"), 0, 4, source_handle.get()));
  IREE_ASSERT_OK(
      AddFileEntry(index.get(), IREE_SV("second"), 4, 4, source_handle.get()));
  IREE_ASSERT_OK(
      AddFileEntry(index.get(), IREE_SV("third"), 8, 4, source_handle.get()));

  Ref<iree_io_parameter_provider_t, iree_io_parameter_provider_release>
      provider;
  IREE_ASSERT_OK(iree_io_parameter_index_provider_create(
      IREE_SV("model"), index.get(),
      IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS,
      iree_allocator_system(), provider.out()));

  Ref<iree_hal_buffer_t, iree_hal_buffer_release> target_buffer;
  target_buffer.reset(
      AllocateTransferBuffer(transfer_pool(), sizeof(source_data)));

  Ref<iree_hal_semaphore_t, iree_hal_semaphore_release> signal;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, signal.out()));

  ParameterRequest requests[3] = {
      {
          .key = IREE_SV("first"),
          .span = {.parameter_offset = 0, .buffer_offset = 0, .length = 4},
      },
      {
          .key = IREE_SV("second"),
          .span = {.parameter_offset = 0, .buffer_offset = 4, .length = 4},
      },
      {
          .key = IREE_SV("third"),
          .span = {.parameter_offset = 0, .buffer_offset = 8, .length = 4},
      },
  };

  uint64_t signal_value = 1;
  iree_hal_semaphore_t* signal_ptr = signal.get();
  iree_io_parameter_gather_t gather = {
      .source_scope = IREE_SV("model"),
      .target_buffer = target_buffer.get(),
      .count = IREE_ARRAYSIZE(requests),
      .enumerator =
          {
              .fn = EnumerateParameterRequest,
              .user_data = requests,
          },
      .wait_semaphore_list = iree_hal_semaphore_list_empty(),
      .signal_semaphore_list =
          {
              .count = 1,
              .semaphores = &signal_ptr,
              .payload_values = &signal_value,
          },
  };

  IREE_ASSERT_OK(iree_io_parameter_provider_gather_batch(
      provider.get(), device, queue(), /*gather_count=*/1, &gather));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(signal.get(), signal_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  ExpectBufferBytesEqual(
      target_buffer.get(),
      iree_make_const_byte_span(source_data, sizeof(source_data)));
}

TEST_F(ParameterIndexProviderTest, ScatterBatchPreservesGroupSemaphores) {
  iree_hal_device_t* device = this->device();

  uint8_t target_data[8] = {0};
  Ref<iree_io_file_handle_t, iree_io_file_handle_release> target_handle;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(target_data, sizeof(target_data)),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      target_handle.out()));

  Ref<iree_io_parameter_index_t, iree_io_parameter_index_release> index;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), index.out()));
  IREE_ASSERT_OK(
      AddFileEntry(index.get(), IREE_SV("first"), 0, 4, target_handle.get()));
  IREE_ASSERT_OK(
      AddFileEntry(index.get(), IREE_SV("second"), 4, 4, target_handle.get()));

  Ref<iree_io_parameter_provider_t, iree_io_parameter_provider_release>
      provider;
  IREE_ASSERT_OK(iree_io_parameter_index_provider_create(
      IREE_SV("model"), index.get(),
      IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS,
      iree_allocator_system(), provider.out()));

  Ref<iree_hal_buffer_t, iree_hal_buffer_release> first_buffer;
  first_buffer.reset(AllocateTransferBuffer(transfer_pool(), 4));
  Ref<iree_hal_buffer_t, iree_hal_buffer_release> second_buffer;
  second_buffer.reset(AllocateTransferBuffer(transfer_pool(), 4));

  Ref<iree_hal_semaphore_t, iree_hal_semaphore_release> first_wait;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, first_wait.out()));
  Ref<iree_hal_semaphore_t, iree_hal_semaphore_release> second_wait;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, second_wait.out()));
  Ref<iree_hal_semaphore_t, iree_hal_semaphore_release> first_signal;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, first_signal.out()));
  Ref<iree_hal_semaphore_t, iree_hal_semaphore_release> second_signal;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, second_signal.out()));

  ParameterRequest first_request = {
      .key = IREE_SV("first"),
      .span = {.parameter_offset = 0, .buffer_offset = 0, .length = 4},
  };
  ParameterRequest second_request = {
      .key = IREE_SV("second"),
      .span = {.parameter_offset = 0, .buffer_offset = 0, .length = 4},
  };

  uint64_t first_wait_value = 1;
  uint64_t second_wait_value = 1;
  uint64_t first_signal_value = 1;
  uint64_t second_signal_value = 1;
  iree_hal_semaphore_t* first_wait_ptr = first_wait.get();
  iree_hal_semaphore_t* second_wait_ptr = second_wait.get();
  iree_hal_semaphore_t* first_signal_ptr = first_signal.get();
  iree_hal_semaphore_t* second_signal_ptr = second_signal.get();
  iree_io_parameter_scatter_t scatters[2] = {
      {
          .target_scope = IREE_SV("model"),
          .source_buffer = first_buffer.get(),
          .count = 1,
          .enumerator =
              {
                  .fn = EnumerateParameterRequest,
                  .user_data = &first_request,
              },
          .wait_semaphore_list =
              {
                  .count = 1,
                  .semaphores = &first_wait_ptr,
                  .payload_values = &first_wait_value,
              },
          .signal_semaphore_list =
              {
                  .count = 1,
                  .semaphores = &first_signal_ptr,
                  .payload_values = &first_signal_value,
              },
      },
      {
          .target_scope = IREE_SV("model"),
          .source_buffer = second_buffer.get(),
          .count = 1,
          .enumerator =
              {
                  .fn = EnumerateParameterRequest,
                  .user_data = &second_request,
              },
          .wait_semaphore_list =
              {
                  .count = 1,
                  .semaphores = &second_wait_ptr,
                  .payload_values = &second_wait_value,
              },
          .signal_semaphore_list =
              {
                  .count = 1,
                  .semaphores = &second_signal_ptr,
                  .payload_values = &second_signal_value,
              },
      },
  };

  const uint8_t first_source[4] = {0x11, 0x11, 0x11, 0x11};
  const uint8_t second_source[4] = {0x22, 0x22, 0x22, 0x22};
  WriteBufferBytes(first_buffer.get(), iree_make_const_byte_span(
                                           first_source, sizeof(first_source)));
  WriteBufferBytes(
      second_buffer.get(),
      iree_make_const_byte_span(second_source, sizeof(second_source)));

  std::atomic<bool> scatter_started = false;
  std::thread scatter_thread([&]() {
    scatter_started.store(true, std::memory_order_release);
    IREE_EXPECT_OK(iree_io_parameter_provider_scatter_batch(
        provider.get(), device, queue(), IREE_ARRAYSIZE(scatters), scatters));
  });
  while (!scatter_started.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  IREE_EXPECT_OK(iree_hal_semaphore_signal(first_wait.get(), first_wait_value,
                                           /*frontier=*/nullptr));
  IREE_EXPECT_OK(iree_hal_semaphore_signal(second_wait.get(), second_wait_value,
                                           /*frontier=*/nullptr));
  scatter_thread.join();

  IREE_ASSERT_OK(iree_hal_semaphore_wait(first_signal.get(), first_signal_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(
      second_signal.get(), second_signal_value, iree_infinite_timeout(),
      IREE_ASYNC_WAIT_FLAG_NONE));

  const uint8_t expected[8] = {0x11, 0x11, 0x11, 0x11, 0x22, 0x22, 0x22, 0x22};
  EXPECT_EQ(0, std::memcmp(target_data, expected, sizeof(expected)));
}

TEST_F(ParameterIndexProviderTest, ScatterBatchWritesAdjacentFileSpans) {
  iree_hal_device_t* device = this->device();

  uint8_t target_data[12] = {0};
  Ref<iree_io_file_handle_t, iree_io_file_handle_release> target_handle;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(target_data, sizeof(target_data)),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      target_handle.out()));

  Ref<iree_io_parameter_index_t, iree_io_parameter_index_release> index;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), index.out()));
  IREE_ASSERT_OK(
      AddFileEntry(index.get(), IREE_SV("first"), 0, 4, target_handle.get()));
  IREE_ASSERT_OK(
      AddFileEntry(index.get(), IREE_SV("second"), 4, 4, target_handle.get()));
  IREE_ASSERT_OK(
      AddFileEntry(index.get(), IREE_SV("third"), 8, 4, target_handle.get()));

  Ref<iree_io_parameter_provider_t, iree_io_parameter_provider_release>
      provider;
  IREE_ASSERT_OK(iree_io_parameter_index_provider_create(
      IREE_SV("model"), index.get(),
      IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS,
      iree_allocator_system(), provider.out()));

  const uint8_t source_data[12] = {
      0x10, 0x11, 0x12, 0x13, 0x20, 0x21, 0x22, 0x23, 0x30, 0x31, 0x32, 0x33,
  };
  Ref<iree_hal_buffer_t, iree_hal_buffer_release> source_buffer;
  source_buffer.reset(
      AllocateTransferBuffer(transfer_pool(), sizeof(source_data)));
  WriteBufferBytes(source_buffer.get(),
                   iree_make_const_byte_span(source_data, sizeof(source_data)));

  Ref<iree_hal_semaphore_t, iree_hal_semaphore_release> signal;
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, signal.out()));

  ParameterRequest requests[3] = {
      {
          .key = IREE_SV("first"),
          .span = {.parameter_offset = 0, .buffer_offset = 0, .length = 4},
      },
      {
          .key = IREE_SV("second"),
          .span = {.parameter_offset = 0, .buffer_offset = 4, .length = 4},
      },
      {
          .key = IREE_SV("third"),
          .span = {.parameter_offset = 0, .buffer_offset = 8, .length = 4},
      },
  };

  uint64_t signal_value = 1;
  iree_hal_semaphore_t* signal_ptr = signal.get();
  iree_io_parameter_scatter_t scatter = {
      .target_scope = IREE_SV("model"),
      .source_buffer = source_buffer.get(),
      .count = IREE_ARRAYSIZE(requests),
      .enumerator =
          {
              .fn = EnumerateParameterRequest,
              .user_data = requests,
          },
      .wait_semaphore_list = iree_hal_semaphore_list_empty(),
      .signal_semaphore_list =
          {
              .count = 1,
              .semaphores = &signal_ptr,
              .payload_values = &signal_value,
          },
  };

  IREE_ASSERT_OK(iree_io_parameter_provider_scatter_batch(
      provider.get(), device, queue(), /*scatter_count=*/1, &scatter));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(signal.get(), signal_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  EXPECT_EQ(0, std::memcmp(target_data, source_data, sizeof(source_data)));
}

TEST_F(ParameterIndexProviderTest, SingleTransferUsesCallerDependencies) {
  enum class Operation { kSplat, kRead, kWrite };
  for (auto operation :
       {Operation::kSplat, Operation::kRead, Operation::kWrite}) {
    for (bool surrounding_empty_spans : {false, true}) {
      SCOPED_TRACE(static_cast<int>(operation));
      SCOPED_TRACE(surrounding_empty_spans);
      std::array<uint8_t, 8> expected;
      expected.fill(0xA7);
      auto file_bytes = expected;
      if (operation == Operation::kWrite) {
        file_bytes.fill(0);
      }
      Ref<iree_io_file_handle_t, iree_io_file_handle_release> file;
      IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
          IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
          iree_make_byte_span(file_bytes.data(), file_bytes.size()),
          iree_io_file_handle_release_callback_null(), iree_allocator_system(),
          file.out()));
      Ref<iree_io_parameter_index_t, iree_io_parameter_index_release> index;
      IREE_ASSERT_OK(
          iree_io_parameter_index_create(iree_allocator_system(), index.out()));
      if (operation == Operation::kSplat) {
        IREE_ASSERT_OK(AddSplatEntry(index.get(), IREE_SV("value"), 8, 0xA7));
      } else {
        IREE_ASSERT_OK(
            AddFileEntry(index.get(), IREE_SV("value"), 0, 8, file.get()));
      }
      Ref<iree_io_parameter_provider_t, iree_io_parameter_provider_release>
          provider;
      IREE_ASSERT_OK(iree_io_parameter_index_provider_create(
          IREE_SV("model"), index.get(), IREE_HOST_SIZE_MAX,
          iree_allocator_system(), provider.out()));
      Ref<iree_hal_buffer_t, iree_hal_buffer_release> buffer;
      buffer.reset(AllocateTransferBuffer(transfer_pool(), expected.size()));
      std::array<Ref<iree_hal_semaphore_t, iree_hal_semaphore_release>, 2>
          gates;
      std::array<Ref<iree_hal_semaphore_t, iree_hal_semaphore_release>, 2> done;
      for (size_t i = 0; i < gates.size(); ++i) {
        IREE_ASSERT_OK(iree_hal_semaphore_create(
            device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
            IREE_HAL_SEMAPHORE_FLAG_DEFAULT, gates[i].out()));
        IREE_ASSERT_OK(iree_hal_semaphore_create(
            device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
            IREE_HAL_SEMAPHORE_FLAG_DEFAULT, done[i].out()));
      }
      iree_hal_semaphore_t* gate_handles[] = {gates[0].get(), gates[1].get()};
      iree_hal_semaphore_t* done_handles[] = {done[0].get(), done[1].get()};
      ParameterRequest requests[] = {{IREE_SV("value"), {8, 8, 0}},
                                     {IREE_SV("value"), {0, 0, 8}},
                                     {IREE_SV("value"), {8, 8, 0}}};
      const iree_io_parameter_enumerator_t enumerator = {
          EnumerateParameterRequest,
          surrounding_empty_spans ? requests : requests + 1};
      const iree_host_size_t count = surrounding_empty_spans ? 3 : 1;
      // Successive groups reuse the caller's same identities at new values.
      for (uint64_t round = 0; round < 3; ++round) {
        std::array<uint8_t, 8> initial = {};
        if (operation == Operation::kWrite) {
          initial = expected;
        }
        WriteBufferBytes(buffer.get(), iree_make_const_byte_span(
                                           initial.data(), initial.size()));
        uint64_t wait_values[] = {round * 10 + 3, round * 10 + 5};
        uint64_t signal_values[] = {round * 10 + 7, round * 10 + 9};
        const iree_hal_semaphore_list_t waits = {2, gate_handles, wait_values};
        const iree_hal_semaphore_list_t signals = {2, done_handles,
                                                   signal_values};
        SubmissionTrace trace(queue());
        if (operation == Operation::kWrite) {
          IREE_EXPECT_OK(iree_io_parameter_provider_scatter(
              provider.get(), device(), trace.queue(), waits, signals,
              buffer.get(), IREE_SV("model"), count, enumerator));
        } else {
          IREE_EXPECT_OK(iree_io_parameter_provider_gather(
              provider.get(), device(), trace.queue(), waits, signals,
              IREE_SV("model"), buffer.get(), count, enumerator));
        }
        const auto& submissions = trace.submissions();
        EXPECT_EQ(submissions.size(), 1u);
        if (submissions.size() == 1) {
          const auto expected_kind =
              operation == Operation::kSplat  ? SubmissionTrace::Kind::kTransfer
              : operation == Operation::kRead ? SubmissionTrace::Kind::kRead
                                              : SubmissionTrace::Kind::kWrite;
          EXPECT_EQ(submissions[0].kind, expected_kind);
          const std::vector<SubmissionTrace::Timepoint> producer = {
              {gate_handles[0], wait_values[0]},
              {gate_handles[1], wait_values[1]}};
          const std::vector<SubmissionTrace::Timepoint> completion = {
              {done_handles[0], signal_values[0]},
              {done_handles[1], signal_values[1]}};
          EXPECT_EQ(submissions[0].waits, producer);
          EXPECT_EQ(submissions[0].signals, completion);
        }
        if (round == 2) {
          // Queue captures own the payload after the provider call returns.
          provider.reset();
          index.reset();
          file.reset();
        }
        IREE_EXPECT_OK(
            iree_hal_semaphore_signal(gates[0].get(), wait_values[0], nullptr));
        uint64_t observed = 0;
        IREE_EXPECT_OK(iree_hal_semaphore_query(done[0].get(), &observed));
        EXPECT_EQ(observed, round ? signal_values[0] - 10 : 0);
        IREE_EXPECT_OK(
            iree_hal_semaphore_signal(gates[1].get(), wait_values[1], nullptr));
        for (size_t i = 0; i < done.size(); ++i) {
          IREE_EXPECT_OK(iree_hal_semaphore_wait(
              done[i].get(), signal_values[i], iree_infinite_timeout(),
              IREE_ASYNC_WAIT_FLAG_NONE));
        }
        if (operation == Operation::kWrite) {
          EXPECT_EQ(file_bytes, expected);
        } else {
          ExpectBufferBytesEqual(
              buffer.get(),
              iree_make_const_byte_span(expected.data(), expected.size()));
        }
      }
    }
  }
}

TEST_F(ParameterIndexProviderTest, HonorsConcurrencyWithoutAnInternalCeiling) {
  enum class Operation { kLoad, kGather, kScatter };
  constexpr size_t kRows = 48;
  constexpr size_t kRowBytes = 256;
  const iree_host_size_t policies[] = {0, 1, 8, 16, 64, IREE_HOST_SIZE_MAX};
  for (auto operation :
       {Operation::kLoad, Operation::kGather, Operation::kScatter}) {
    for (auto policy : policies) {
      SCOPED_TRACE(static_cast<int>(operation));
      SCOPED_TRACE(policy);
      const size_t expected_concurrency =
          std::min(std::max<iree_host_size_t>(policy, 1), kRows);
      std::array<uint8_t, kRows * kRowBytes * 2> file_bytes;
      for (size_t i = 0; i < file_bytes.size(); ++i) {
        file_bytes[i] = 1 + (i * 13 + i / 251) % 251;
      }
      const auto original = file_bytes;
      std::array<uint8_t, kRows * kRowBytes> packed;
      std::array<std::string, kRows> keys;
      std::array<ParameterRequest, kRows> requests;
      for (size_t i = 0; i < kRows; ++i) {
        const size_t source_offset = ((i * 17) % kRows) * kRowBytes * 2;
        keys[i] = "row" + std::to_string(i);
        requests[i] = {
            iree_make_string_view(keys[i].data(), keys[i].size()),
            {0, operation == Operation::kLoad ? 0 : i * kRowBytes, kRowBytes}};
        std::memcpy(packed.data() + i * kRowBytes,
                    file_bytes.data() + source_offset, kRowBytes);
      }
      if (operation == Operation::kScatter) {
        file_bytes.fill(0);
      }
      Ref<iree_io_file_handle_t, iree_io_file_handle_release> file;
      IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
          IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
          iree_make_byte_span(file_bytes.data(), file_bytes.size()),
          iree_io_file_handle_release_callback_null(), iree_allocator_system(),
          file.out()));
      Ref<iree_io_parameter_index_t, iree_io_parameter_index_release> index;
      IREE_ASSERT_OK(
          iree_io_parameter_index_create(iree_allocator_system(), index.out()));
      for (size_t i = 0; i < kRows; ++i) {
        const size_t source_offset = ((i * 17) % kRows) * kRowBytes * 2;
        IREE_ASSERT_OK(AddFileEntry(index.get(), requests[i].key, source_offset,
                                    kRowBytes, file.get()));
      }
      Ref<iree_io_parameter_provider_t, iree_io_parameter_provider_release>
          provider;
      IREE_ASSERT_OK(iree_io_parameter_index_provider_create(
          IREE_SV("model"), index.get(), policy, iree_allocator_system(),
          provider.out()));
      Ref<iree_hal_buffer_t, iree_hal_buffer_release> buffer;
      buffer.reset(AllocateTransferBuffer(transfer_pool(), packed.size()));
      if (operation == Operation::kScatter) {
        WriteBufferBytes(buffer.get(), iree_make_const_byte_span(
                                           packed.data(), packed.size()));
      }
      Ref<iree_hal_semaphore_t, iree_hal_semaphore_release> gate;
      Ref<iree_hal_semaphore_t, iree_hal_semaphore_release> done;
      IREE_ASSERT_OK(iree_hal_semaphore_create(
          device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
          IREE_HAL_SEMAPHORE_FLAG_DEFAULT, gate.out()));
      IREE_ASSERT_OK(iree_hal_semaphore_create(
          device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
          IREE_HAL_SEMAPHORE_FLAG_DEFAULT, done.out()));
      auto* gate_ptr = gate.get();
      auto* done_ptr = done.get();
      uint64_t value = 1;
      const iree_hal_semaphore_list_t waits = {1, &gate_ptr, &value};
      const iree_hal_semaphore_list_t signals = {1, &done_ptr, &value};
      const iree_io_parameter_enumerator_t enumerator = {
          EnumerateParameterRequest, requests.data()};
      std::array<Ref<iree_hal_buffer_t, iree_hal_buffer_release>, kRows> loaded;
      SubmissionTrace trace(queue());
      switch (operation) {
        case Operation::kLoad: {
          const iree_io_parameter_emitter_t emitter = {
              [](void* data, iree_host_size_t i, iree_hal_buffer_t* buffer) {
                auto* outputs = static_cast<decltype(loaded)*>(data);
                iree_hal_buffer_retain(buffer);
                (*outputs)[i].reset(buffer);
                return iree_ok_status();
              },
              &loaded};
          IREE_EXPECT_OK(iree_io_parameter_provider_load(
              provider.get(), device(), trace.queue(), transfer_pool(), waits,
              signals, IREE_SV("model"), 1, kRows, enumerator, emitter));
          break;
        }
        case Operation::kGather:
          IREE_EXPECT_OK(iree_io_parameter_provider_gather(
              provider.get(), device(), trace.queue(), waits, signals,
              IREE_SV("model"), buffer.get(), kRows, enumerator));
          break;
        case Operation::kScatter:
          IREE_EXPECT_OK(iree_io_parameter_provider_scatter(
              provider.get(), device(), trace.queue(), waits, signals,
              buffer.get(), IREE_SV("model"), kRows, enumerator));
          break;
      }
      size_t independent_count = 0;
      size_t transfer_count = 0;
      std::set<iree_hal_semaphore_t*> timelines;
      std::vector<SubmissionTrace::Timepoint> allocation;
      const std::vector<SubmissionTrace::Timepoint> producer = {{gate_ptr, 1}};
      for (const auto& submission : trace.submissions()) {
        if (submission.kind == SubmissionTrace::Kind::kTransfer) {
          EXPECT_EQ(submission.waits.size(), expected_concurrency);
          continue;
        }
        EXPECT_EQ(submission.signals.size(), 1);
        if (!submission.signals.empty()) {
          timelines.insert(submission.signals[0].first);
        }
        independent_count += submission.waits == producer;
        if (submission.kind == SubmissionTrace::Kind::kAllocate) {
          allocation = submission.signals;
        } else {
          ++transfer_count;
          if (operation == Operation::kLoad) {
            EXPECT_EQ(submission.waits, allocation);
          }
        }
      }
      EXPECT_EQ(transfer_count, kRows);
      EXPECT_EQ(independent_count, expected_concurrency);
      EXPECT_EQ(timelines.size(), expected_concurrency);
      IREE_EXPECT_OK(iree_hal_semaphore_signal(gate.get(), 1, nullptr));
      IREE_EXPECT_OK(iree_hal_semaphore_wait(
          done.get(), 1, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
      if (operation == Operation::kGather) {
        ExpectBufferBytesEqual(buffer.get(), iree_make_const_byte_span(
                                                 packed.data(), packed.size()));
      } else {
        for (size_t i = 0; i < kRows; ++i) {
          const size_t source_offset = ((i * 17) % kRows) * kRowBytes * 2;
          if (operation == Operation::kLoad) {
            EXPECT_NE(loaded[i].get(), nullptr);
            if (loaded[i].get()) {
              ExpectBufferBytesEqual(
                  loaded[i].get(),
                  iree_make_const_byte_span(packed.data() + i * kRowBytes,
                                            kRowBytes));
            }
          } else {
            EXPECT_EQ(0,
                      std::memcmp(file_bytes.data() + source_offset,
                                  original.data() + source_offset, kRowBytes));
            for (size_t j = 0; j < kRowBytes; ++j) {
              EXPECT_EQ(file_bytes[source_offset + kRowBytes + j], 0);
            }
          }
        }
      }
    }
  }
}

TEST_F(ParameterIndexProviderTest, MixedTransfersKeepIndependentCompletion) {
  for (bool splat_first : {false, true}) {
    SCOPED_TRACE(splat_first);
    std::array<uint8_t, 4> bytes = {1, 2, 3, 4};
    Ref<iree_io_file_handle_t, iree_io_file_handle_release> file;
    IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
        IREE_IO_FILE_ACCESS_READ,
        iree_make_byte_span(bytes.data(), bytes.size()),
        iree_io_file_handle_release_callback_null(), iree_allocator_system(),
        file.out()));
    Ref<iree_io_parameter_index_t, iree_io_parameter_index_release> index;
    IREE_ASSERT_OK(
        iree_io_parameter_index_create(iree_allocator_system(), index.out()));
    IREE_ASSERT_OK(
        AddFileEntry(index.get(), IREE_SV("file"), 0, 4, file.get()));
    IREE_ASSERT_OK(AddSplatEntry(index.get(), IREE_SV("splat"), 4, 0xA7));
    Ref<iree_io_parameter_provider_t, iree_io_parameter_provider_release>
        provider;
    IREE_ASSERT_OK(iree_io_parameter_index_provider_create(
        IREE_SV("model"), index.get(), IREE_HOST_SIZE_MAX,
        iree_allocator_system(), provider.out()));
    Ref<iree_hal_buffer_t, iree_hal_buffer_release> buffer;
    buffer.reset(AllocateTransferBuffer(transfer_pool(), 8));
    Ref<iree_hal_semaphore_t, iree_hal_semaphore_release> gate;
    Ref<iree_hal_semaphore_t, iree_hal_semaphore_release> done;
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, gate.out()));
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, done.out()));
    auto* gate_ptr = gate.get();
    auto* done_ptr = done.get();
    uint64_t value = 1;
    ParameterRequest requests[] = {{IREE_SV("file"), {0, 0, 4}},
                                   {IREE_SV("splat"), {0, 4, 4}}};
    if (splat_first) {
      std::swap(requests[0], requests[1]);
    }
    SubmissionTrace trace(queue());
    IREE_EXPECT_OK(iree_io_parameter_provider_gather(
        provider.get(), device(), trace.queue(), {1, &gate_ptr, &value},
        {1, &done_ptr, &value}, IREE_SV("model"), buffer.get(), 2,
        {EnumerateParameterRequest, requests}));
    const auto& submissions = trace.submissions();
    EXPECT_EQ(submissions.size(), 3u);
    if (submissions.size() == 3) {
      const std::vector<SubmissionTrace::Timepoint> producer = {{gate_ptr, 1}};
      const std::vector<SubmissionTrace::Timepoint> completion = {
          {done_ptr, 1}};
      for (size_t i = 0; i < 2; ++i) {
        EXPECT_EQ(submissions[i].waits, producer);
        EXPECT_EQ(submissions[i].signals.size(), 1u);
        EXPECT_NE(submissions[i].signals, completion);
      }
      EXPECT_NE(submissions[0].signals, submissions[1].signals);
      auto joins = submissions[0].signals;
      joins.insert(joins.end(), submissions[1].signals.begin(),
                   submissions[1].signals.end());
      EXPECT_EQ(submissions[2].kind, SubmissionTrace::Kind::kTransfer);
      EXPECT_EQ(submissions[2].waits, joins);
      EXPECT_EQ(submissions[2].signals, completion);
    }
    provider.reset();
    index.reset();
    file.reset();
    IREE_EXPECT_OK(iree_hal_semaphore_signal(gate.get(), 1, nullptr));
    IREE_EXPECT_OK(iree_hal_semaphore_wait(
        done.get(), 1, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    const uint8_t expected[] = {1, 2, 3, 4, 0xA7, 0xA7, 0xA7, 0xA7};
    ExpectBufferBytesEqual(
        buffer.get(), iree_make_const_byte_span(expected, sizeof(expected)));
  }
}

TEST_F(ParameterIndexProviderTest, EmptySpansPreserveBatchSynchronization) {
  // A device-produced window can contain mostly duplicate or inactive spans.
  // Empty entries carry no independent synchronization contract, including
  // empty splats between adjacent file-backed ranges.
  constexpr size_t kSpanCount = 24 * 1024;
  for (bool scatter : {false, true}) {
    for (bool all_empty : {false, true}) {
      SCOPED_TRACE(scatter);
      SCOPED_TRACE(all_empty);
      std::array<uint8_t, 64> expected;
      for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = static_cast<uint8_t>(i * 3 + 1);
      }
      auto file_bytes = expected;
      if (scatter) {
        file_bytes.fill(0);
      }
      Ref<iree_io_file_handle_t, iree_io_file_handle_release> file;
      IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
          IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
          iree_make_byte_span(file_bytes.data(), file_bytes.size()),
          iree_io_file_handle_release_callback_null(), iree_allocator_system(),
          file.out()));
      Ref<iree_io_parameter_index_t, iree_io_parameter_index_release> index;
      IREE_ASSERT_OK(
          iree_io_parameter_index_create(iree_allocator_system(), index.out()));
      IREE_ASSERT_OK(AddFileEntry(index.get(), IREE_SV("rows"), 0,
                                  file_bytes.size(), file.get()));
      IREE_ASSERT_OK(AddSplatEntry(index.get(), IREE_SV("empty"),
                                   file_bytes.size(), 0xA5));
      Ref<iree_io_parameter_provider_t, iree_io_parameter_provider_release>
          provider;
      IREE_ASSERT_OK(iree_io_parameter_index_provider_create(
          IREE_SV("model"), index.get(), IREE_HOST_SIZE_MAX,
          iree_allocator_system(), provider.out()));
      Ref<iree_hal_buffer_t, iree_hal_buffer_release> buffer;
      buffer.reset(AllocateTransferBuffer(transfer_pool(), expected.size()));
      std::array<uint8_t, 64> initial = {};
      if (scatter) {
        initial = expected;
      }
      WriteBufferBytes(buffer.get(), iree_make_const_byte_span(initial.data(),
                                                               initial.size()));
      std::vector<ParameterRequest> requests(kSpanCount,
                                             {IREE_SV("rows"), {64, 64, 0}});
      for (size_t i = 0; i < requests.size(); ++i) {
        requests[i].span.parameter_offset = i % 65;
        requests[i].span.buffer_offset = (i * 3) % 65;
      }
      if (!scatter) {
        for (size_t i = 1; i < requests.size(); i += 2) {
          requests[i].key = IREE_SV("empty");
        }
      }
      if (!all_empty) {
        requests.front() = {IREE_SV("rows"), {0, 0, 32}};
        requests[kSpanCount / 2] = {IREE_SV("rows"), {32, 32, 32}};
      }
      Ref<iree_hal_semaphore_t, iree_hal_semaphore_release> gate;
      Ref<iree_hal_semaphore_t, iree_hal_semaphore_release> done;
      IREE_ASSERT_OK(iree_hal_semaphore_create(
          device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
          IREE_HAL_SEMAPHORE_FLAG_DEFAULT, gate.out()));
      IREE_ASSERT_OK(iree_hal_semaphore_create(
          device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
          IREE_HAL_SEMAPHORE_FLAG_DEFAULT, done.out()));
      auto* gate_ptr = gate.get();
      auto* done_ptr = done.get();
      uint64_t value = 1;
      const iree_hal_semaphore_list_t waits = {1, &gate_ptr, &value};
      const iree_hal_semaphore_list_t signals = {1, &done_ptr, &value};
      const iree_io_parameter_enumerator_t enumerator = {
          EnumerateParameterRequest, requests.data()};
      SubmissionTrace trace(queue());
      if (scatter) {
        IREE_EXPECT_OK(iree_io_parameter_provider_scatter(
            provider.get(), device(), trace.queue(), waits, signals,
            buffer.get(), IREE_SV("model"), requests.size(), enumerator));
      } else {
        IREE_EXPECT_OK(iree_io_parameter_provider_gather(
            provider.get(), device(), trace.queue(), waits, signals,
            IREE_SV("model"), buffer.get(), requests.size(), enumerator));
      }
      const auto& submissions = trace.submissions();
      EXPECT_EQ(submissions.size(), 1u);
      const std::vector<SubmissionTrace::Timepoint> producer = {{gate_ptr, 1}};
      const std::vector<SubmissionTrace::Timepoint> completion = {
          {done_ptr, 1}};
      if (submissions.size() == 1) {
        EXPECT_EQ(submissions.front().waits, producer);
        EXPECT_EQ(submissions.front().kind,
                  all_empty ? SubmissionTrace::Kind::kTransfer
                  : scatter ? SubmissionTrace::Kind::kWrite
                            : SubmissionTrace::Kind::kRead);
        EXPECT_EQ(submissions.front().signals, completion);
      }
      uint64_t observed = 99;
      IREE_EXPECT_OK(iree_hal_semaphore_query(done.get(), &observed));
      EXPECT_EQ(observed, 0);
      IREE_EXPECT_OK(iree_hal_semaphore_signal(gate.get(), 1, nullptr));
      IREE_EXPECT_OK(iree_hal_semaphore_wait(
          done.get(), 1, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
      if (scatter) {
        if (all_empty) {
          expected.fill(0);
        }
        EXPECT_EQ(file_bytes, expected);
      } else {
        const auto& output = all_empty ? initial : expected;
        ExpectBufferBytesEqual(buffer.get(), iree_make_const_byte_span(
                                                 output.data(), output.size()));
      }
    }
  }
}

TEST_F(ParameterIndexProviderTest, EmptySpansStillValidateKeyAccessAndRange) {
  std::array<uint8_t, 8> bytes = {1, 2, 3, 4, 5, 6, 7, 8};
  Ref<iree_io_file_handle_t, iree_io_file_handle_release> file;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(bytes.data(), bytes.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      file.out()));
  Ref<iree_io_parameter_index_t, iree_io_parameter_index_release> index;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), index.out()));
  IREE_ASSERT_OK(
      AddFileEntry(index.get(), IREE_SV("rows"), 0, bytes.size(), file.get()));
  IREE_ASSERT_OK(AddSplatEntry(index.get(), IREE_SV("readonly"), 8, 1));
  Ref<iree_io_parameter_provider_t, iree_io_parameter_provider_release>
      provider;
  IREE_ASSERT_OK(iree_io_parameter_index_provider_create(
      IREE_SV("model"), index.get(), IREE_HOST_SIZE_MAX,
      iree_allocator_system(), provider.out()));
  Ref<iree_hal_buffer_t, iree_hal_buffer_release> buffer;
  buffer.reset(AllocateTransferBuffer(transfer_pool(), bytes.size()));
  const ParameterRequest requests[] = {
      {IREE_SV("missing"), {0, 0, 0}},
      {IREE_SV("rows"), {9, 0, 0}},
      {IREE_SV("readonly"), {0, 0, 0}},
  };
  const iree_status_code_t codes[] = {IREE_STATUS_NOT_FOUND,
                                      IREE_STATUS_OUT_OF_RANGE,
                                      IREE_STATUS_PERMISSION_DENIED};
  for (bool scatter : {false, true}) {
    for (size_t i = 0; i < (scatter ? 3u : 2u); ++i) {
      SCOPED_TRACE(scatter);
      SCOPED_TRACE(i);
      Ref<iree_hal_semaphore_t, iree_hal_semaphore_release> done;
      IREE_ASSERT_OK(iree_hal_semaphore_create(
          device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
          IREE_HAL_SEMAPHORE_FLAG_DEFAULT, done.out()));
      auto* done_ptr = done.get();
      uint64_t value = 1;
      auto request = requests[i];
      const iree_io_parameter_enumerator_t enumerator = {
          EnumerateParameterRequest, &request};
      SubmissionTrace trace(queue());
      if (scatter) {
        IREE_EXPECT_STATUS_IS(
            codes[i],
            iree_io_parameter_provider_scatter(
                provider.get(), device(), trace.queue(),
                iree_hal_semaphore_list_empty(), {1, &done_ptr, &value},
                buffer.get(), IREE_SV("model"), 1, enumerator));
      } else {
        IREE_EXPECT_STATUS_IS(
            codes[i],
            iree_io_parameter_provider_gather(
                provider.get(), device(), trace.queue(),
                iree_hal_semaphore_list_empty(), {1, &done_ptr, &value},
                IREE_SV("model"), buffer.get(), 1, enumerator));
      }
      EXPECT_TRUE(trace.submissions().empty());
      IREE_EXPECT_STATUS_IS(codes[i],
                            iree_hal_semaphore_query(done.get(), &value));
    }
  }
}

TEST_F(ParameterIndexProviderTest,
       LateValidationFailureKeepsAcceptedOwnership) {
  enum class Operation { kSplat, kRead, kWrite };
  for (auto operation :
       {Operation::kSplat, Operation::kRead, Operation::kWrite}) {
    for (iree_host_size_t payload_count : {1, 2}) {
      SCOPED_TRACE(static_cast<int>(operation));
      SCOPED_TRACE(payload_count);
      std::array<uint8_t, 12> file_bytes;
      file_bytes.fill(operation == Operation::kWrite ? 0 : 0xA7);
      Ref<iree_io_file_handle_t, iree_io_file_handle_release> file;
      IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
          IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
          iree_make_byte_span(file_bytes.data(), file_bytes.size()),
          iree_io_file_handle_release_callback_null(), iree_allocator_system(),
          file.out()));
      Ref<iree_io_parameter_index_t, iree_io_parameter_index_release> index;
      IREE_ASSERT_OK(
          iree_io_parameter_index_create(iree_allocator_system(), index.out()));
      if (operation == Operation::kSplat) {
        IREE_ASSERT_OK(AddSplatEntry(index.get(), IREE_SV("value"), 12, 0xA7));
      } else {
        IREE_ASSERT_OK(
            AddFileEntry(index.get(), IREE_SV("value"), 0, 12, file.get()));
      }
      Ref<iree_io_parameter_provider_t, iree_io_parameter_provider_release>
          provider;
      IREE_ASSERT_OK(iree_io_parameter_index_provider_create(
          IREE_SV("model"), index.get(), IREE_HOST_SIZE_MAX,
          iree_allocator_system(), provider.out()));
      Ref<iree_hal_buffer_t, iree_hal_buffer_release> buffer;
      buffer.reset(AllocateTransferBuffer(transfer_pool(), 8));
      std::array<uint8_t, 8> expected;
      expected.fill(operation == Operation::kWrite ? 0xA7 : 0);
      WriteBufferBytes(buffer.get(), iree_make_const_byte_span(
                                         expected.data(), expected.size()));
      Ref<iree_hal_semaphore_t, iree_hal_semaphore_release> gate;
      Ref<iree_hal_semaphore_t, iree_hal_semaphore_release> done;
      IREE_ASSERT_OK(iree_hal_semaphore_create(
          device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
          IREE_HAL_SEMAPHORE_FLAG_DEFAULT, gate.out()));
      IREE_ASSERT_OK(iree_hal_semaphore_create(
          device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
          IREE_HAL_SEMAPHORE_FLAG_DEFAULT, done.out()));
      auto* gate_ptr = gate.get();
      auto* done_ptr = done.get();
      uint64_t value = 1;
      ParameterRequest requests[] = {{IREE_SV("value"), {0, 0, 4}},
                                     {IREE_SV("value"), {8, 4, 4}},
                                     {IREE_SV("missing"), {0, 0, 0}}};
      requests[payload_count] = requests[2];
      SubmissionTrace trace(queue());
      if (operation == Operation::kWrite) {
        IREE_EXPECT_STATUS_IS(
            IREE_STATUS_NOT_FOUND,
            iree_io_parameter_provider_scatter(
                provider.get(), device(), trace.queue(), {1, &gate_ptr, &value},
                {1, &done_ptr, &value}, buffer.get(), IREE_SV("model"),
                payload_count + 1, {EnumerateParameterRequest, requests}));
      } else {
        IREE_EXPECT_STATUS_IS(
            IREE_STATUS_NOT_FOUND,
            iree_io_parameter_provider_gather(
                provider.get(), device(), trace.queue(), {1, &gate_ptr, &value},
                {1, &done_ptr, &value}, IREE_SV("model"), buffer.get(),
                payload_count + 1, {EnumerateParameterRequest, requests}));
      }
      IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                            iree_hal_semaphore_query(done.get(), &value));
      const auto& submissions = trace.submissions();
      EXPECT_EQ(submissions.size(), payload_count - 1);
      Ref<iree_hal_semaphore_t, iree_hal_semaphore_release> accepted;
      uint64_t accepted_value = 0;
      if (submissions.size() == 1 && submissions[0].signals.size() == 1) {
        auto signal = submissions[0].signals[0];
        iree_hal_semaphore_retain(signal.first);
        accepted.reset(signal.first);
        accepted_value = signal.second;
      }
      provider.reset();
      index.reset();
      file.reset();
      IREE_EXPECT_OK(iree_hal_semaphore_signal(gate.get(), 1, nullptr));
      if (accepted.get()) {
        IREE_EXPECT_OK(iree_hal_semaphore_wait(accepted.get(), accepted_value,
                                               iree_infinite_timeout(),
                                               IREE_ASYNC_WAIT_FLAG_NONE));
      }
      if (operation == Operation::kWrite) {
        for (size_t i = 0; i < file_bytes.size(); ++i) {
          EXPECT_EQ(file_bytes[i], payload_count == 2 && i < 4 ? 0xA7 : 0);
        }
      } else {
        if (payload_count == 2) {
          std::fill_n(expected.begin(), 4, 0xA7);
        }
        ExpectBufferBytesEqual(
            buffer.get(),
            iree_make_const_byte_span(expected.data(), expected.size()));
      }
    }
  }
}

TEST_F(ParameterIndexProviderTest, IndependentGroupsDoNotShareDependencyLanes) {
  for (bool scatter : {false, true}) {
    SCOPED_TRACE(scatter);
    std::array<uint8_t, 8> bytes = {1, 2, 3, 4, 5, 6, 7, 8};
    Ref<iree_io_file_handle_t, iree_io_file_handle_release> file;
    IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
        IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
        iree_make_byte_span(bytes.data(), bytes.size()),
        iree_io_file_handle_release_callback_null(), iree_allocator_system(),
        file.out()));
    Ref<iree_io_parameter_index_t, iree_io_parameter_index_release> index;
    IREE_ASSERT_OK(
        iree_io_parameter_index_create(iree_allocator_system(), index.out()));
    IREE_ASSERT_OK(
        AddFileEntry(index.get(), IREE_SV("first"), 0, 4, file.get()));
    IREE_ASSERT_OK(
        AddFileEntry(index.get(), IREE_SV("second"), 4, 4, file.get()));
    Ref<iree_io_parameter_provider_t, iree_io_parameter_provider_release>
        provider;
    IREE_ASSERT_OK(iree_io_parameter_index_provider_create(
        IREE_SV("model"), index.get(), /*max_concurrent_operations=*/1,
        iree_allocator_system(), provider.out()));
    std::array<Ref<iree_hal_buffer_t, iree_hal_buffer_release>, 2> buffers;
    for (size_t i = 0; i < buffers.size(); ++i) {
      buffers[i].reset(AllocateTransferBuffer(transfer_pool(), 4));
      const uint8_t pattern[] = {9, 10, 11, 12};
      WriteBufferBytes(buffers[i].get(),
                       iree_make_const_byte_span(pattern, sizeof(pattern)));
    }
    Ref<iree_hal_semaphore_t, iree_hal_semaphore_release> gate;
    std::array<Ref<iree_hal_semaphore_t, iree_hal_semaphore_release>, 2> done;
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, gate.out()));
    for (auto& signal : done) {
      IREE_ASSERT_OK(iree_hal_semaphore_create(
          device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
          IREE_HAL_SEMAPHORE_FLAG_DEFAULT, signal.out()));
    }
    uint64_t value = 1;
    auto* gate_ptr = gate.get();
    iree_hal_semaphore_t* done_ptrs[] = {done[0].get(), done[1].get()};
    const ParameterRequest requests[] = {{IREE_SV("first"), {0, 0, 4}},
                                         {IREE_SV("second"), {0, 0, 4}}};
    iree_io_parameter_gather_t gathers[2] = {};
    iree_io_parameter_scatter_t scatters[2] = {};
    for (size_t i = 0; i < 2; ++i) {
      const iree_hal_semaphore_list_t waits =
          i ? iree_hal_semaphore_list_empty()
            : iree_hal_semaphore_list_t{1, &gate_ptr, &value};
      const iree_hal_semaphore_list_t signals = {1, &done_ptrs[i], &value};
      const iree_io_parameter_enumerator_t enumerator = {
          EnumerateParameterRequest,
          const_cast<ParameterRequest*>(&requests[i])};
      gathers[i] = {IREE_SV("model"), buffers[i].get(), 1, enumerator, waits,
                    signals};
      scatters[i] = {IREE_SV("model"), buffers[i].get(), 1, enumerator, waits,
                     signals};
    }
    if (scatter) {
      IREE_EXPECT_OK(iree_io_parameter_provider_scatter_batch(
          provider.get(), device(), queue(), 2, scatters));
    } else {
      IREE_EXPECT_OK(iree_io_parameter_provider_gather_batch(
          provider.get(), device(), queue(), 2, gathers));
    }
    // The later group completes while the earlier group's prerequisite is
    // still closed. No submission-order or shared-lane edge may prevent it.
    IREE_EXPECT_OK(iree_hal_semaphore_wait(
        done[1].get(), 1, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    uint64_t held_value = 0;
    IREE_EXPECT_OK(iree_hal_semaphore_query(done[0].get(), &held_value));
    EXPECT_EQ(held_value, 0);
    if (scatter) {
      const uint8_t expected[] = {1, 2, 3, 4, 9, 10, 11, 12};
      EXPECT_EQ(0, std::memcmp(bytes.data(), expected, sizeof(expected)));
    } else {
      const uint8_t held[] = {9, 10, 11, 12};
      ExpectBufferBytesEqual(buffers[0].get(),
                             iree_make_const_byte_span(held, sizeof(held)));
      ExpectBufferBytesEqual(buffers[1].get(),
                             iree_make_const_byte_span(bytes.data() + 4, 4));
    }
    IREE_EXPECT_OK(iree_hal_semaphore_signal(gate.get(), 1, nullptr));
    IREE_EXPECT_OK(iree_hal_semaphore_wait(
        done[0].get(), 1, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    if (scatter) {
      const uint8_t expected[] = {9, 10, 11, 12, 9, 10, 11, 12};
      EXPECT_EQ(0, std::memcmp(bytes.data(), expected, sizeof(expected)));
    } else {
      ExpectBufferBytesEqual(buffers[0].get(),
                             iree_make_const_byte_span(bytes.data(), 4));
    }
  }
}

TEST_F(ParameterIndexProviderTest,
       RejectsRangeAndBatchSizeOverflowBeforeSubmission) {
  Ref<iree_io_parameter_index_t, iree_io_parameter_index_release> index;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), index.out()));
  IREE_ASSERT_OK(AddSplatEntry(index.get(), IREE_SV("rows"), 8, 1));
  Ref<iree_io_parameter_provider_t, iree_io_parameter_provider_release>
      provider;
  IREE_ASSERT_OK(iree_io_parameter_index_provider_create(
      IREE_SV("model"), index.get(), IREE_HOST_SIZE_MAX,
      iree_allocator_system(), provider.out()));
  Ref<iree_hal_buffer_t, iree_hal_buffer_release> buffer;
  buffer.reset(AllocateTransferBuffer(transfer_pool(), 8));
  for (bool batch_overflow : {false, true}) {
    Ref<iree_hal_semaphore_t, iree_hal_semaphore_release> done;
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, done.out()));
    auto* done_ptr = done.get();
    uint64_t value = 1;
    ParameterRequest request = {IREE_SV("rows"), {UINT64_MAX - 1, 0, 4}};
    SubmissionTrace trace(queue());
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_OUT_OF_RANGE,
        iree_io_parameter_provider_gather(
            provider.get(), device(), trace.queue(),
            iree_hal_semaphore_list_empty(), {1, &done_ptr, &value},
            IREE_SV("model"), buffer.get(),
            batch_overflow ? IREE_HOST_SIZE_MAX : 1,
            {EnumerateParameterRequest, &request}));
    EXPECT_TRUE(trace.submissions().empty());
    IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                          iree_hal_semaphore_query(done.get(), &value));
  }
}

TEST_F(ParameterIndexProviderTest,
       BatchFailurePreservesAcceptedGroupOwnership) {
  for (bool scatter : {false, true}) {
    SCOPED_TRACE(scatter);
    std::array<uint8_t, 8> file_bytes;
    file_bytes.fill(0xA7);
    Ref<iree_io_file_handle_t, iree_io_file_handle_release> file;
    IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
        IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
        iree_make_byte_span(file_bytes.data(), file_bytes.size()),
        iree_io_file_handle_release_callback_null(), iree_allocator_system(),
        file.out()));
    Ref<iree_io_parameter_index_t, iree_io_parameter_index_release> index;
    IREE_ASSERT_OK(
        iree_io_parameter_index_create(iree_allocator_system(), index.out()));
    IREE_ASSERT_OK(
        AddFileEntry(index.get(), IREE_SV("first"), 0, 4, file.get()));
    IREE_ASSERT_OK(
        AddFileEntry(index.get(), IREE_SV("second"), 4, 4, file.get()));
    Ref<iree_io_parameter_provider_t, iree_io_parameter_provider_release>
        provider;
    IREE_ASSERT_OK(iree_io_parameter_index_provider_create(
        IREE_SV("model"), index.get(), IREE_HOST_SIZE_MAX,
        iree_allocator_system(), provider.out()));
    Ref<iree_hal_buffer_t, iree_hal_buffer_release> buffer;
    buffer.reset(AllocateTransferBuffer(transfer_pool(), 12));
    const std::array<uint8_t, 12> source = {0xBC, 0xBC, 0xBC, 0xBC, 0xBC, 0xBC,
                                            0xBC, 0xBC, 0xBC, 0xBC, 0xBC, 0xBC};
    WriteBufferBytes(buffer.get(),
                     iree_make_const_byte_span(source.data(), source.size()));
    Ref<iree_hal_semaphore_t, iree_hal_semaphore_release> gate;
    std::array<Ref<iree_hal_semaphore_t, iree_hal_semaphore_release>, 3> done;
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, gate.out()));
    for (auto& signal : done) {
      IREE_ASSERT_OK(iree_hal_semaphore_create(
          device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
          IREE_HAL_SEMAPHORE_FLAG_DEFAULT, signal.out()));
    }
    ParameterRequest requests[] = {{IREE_SV("first"), {0, 0, 4}},
                                   {IREE_SV("missing"), {0, 4, 4}},
                                   {IREE_SV("second"), {0, 8, 4}}};
    iree_io_parameter_gather_t gathers[3] = {};
    iree_io_parameter_scatter_t scatters[3] = {};
    auto* gate_ptr = gate.get();
    iree_hal_semaphore_t* done_ptrs[] = {done[0].get(), done[1].get(),
                                         done[2].get()};
    uint64_t value = 1;
    for (size_t i = 0; i < 3; ++i) {
      gathers[i] = {IREE_SV("model"),
                    buffer.get(),
                    1,
                    {EnumerateParameterRequest, &requests[i]},
                    {1, &gate_ptr, &value},
                    {1, &done_ptrs[i], &value}};
      scatters[i] = {IREE_SV("model"),
                     buffer.get(),
                     1,
                     {EnumerateParameterRequest, &requests[i]},
                     {1, &gate_ptr, &value},
                     {1, &done_ptrs[i], &value}};
    }
    if (scatter) {
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_NOT_FOUND,
          iree_io_parameter_provider_scatter_batch(provider.get(), device(),
                                                   queue(), 3, scatters));
    } else {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                            iree_io_parameter_provider_gather_batch(
                                provider.get(), device(), queue(), 3, gathers));
    }
    uint64_t observed = 0;
    IREE_EXPECT_OK(iree_hal_semaphore_query(done[0].get(), &observed));
    EXPECT_EQ(observed, 0);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                          iree_hal_semaphore_query(done[1].get(), &observed));
    IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                          iree_hal_semaphore_query(done[2].get(), &observed));
    provider.reset();
    index.reset();
    file.reset();
    IREE_EXPECT_OK(iree_hal_semaphore_signal(gate.get(), 1, nullptr));
    IREE_EXPECT_OK(iree_hal_semaphore_wait(
        done[0].get(), 1, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    if (scatter) {
      for (size_t i = 0; i < file_bytes.size(); ++i) {
        EXPECT_EQ(file_bytes[i], i < 4 ? 0xBC : 0xA7);
      }
    } else {
      ExpectBufferBytes(buffer.get(), 4, 0xA7);
    }
  }
}

TEST_F(ParameterIndexProviderTest, EmptyGroupsPreserveOnlyTheirOwnWaits) {
  Ref<iree_io_parameter_index_t, iree_io_parameter_index_release> index;
  IREE_ASSERT_OK(
      iree_io_parameter_index_create(iree_allocator_system(), index.out()));
  Ref<iree_io_parameter_provider_t, iree_io_parameter_provider_release>
      provider;
  IREE_ASSERT_OK(iree_io_parameter_index_provider_create(
      IREE_SV("model"), index.get(), IREE_HOST_SIZE_MAX,
      iree_allocator_system(), provider.out()));
  for (bool scatter : {false, true}) {
    SCOPED_TRACE(scatter);
    Ref<iree_hal_semaphore_t, iree_hal_semaphore_release> gate;
    std::array<Ref<iree_hal_semaphore_t, iree_hal_semaphore_release>, 2> done;
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, gate.out()));
    for (auto& signal : done) {
      IREE_ASSERT_OK(iree_hal_semaphore_create(
          device(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
          IREE_HAL_SEMAPHORE_FLAG_DEFAULT, signal.out()));
    }
    auto* gate_ptr = gate.get();
    iree_hal_semaphore_t* done_ptrs[] = {done[0].get(), done[1].get()};
    uint64_t value = 1;
    iree_io_parameter_gather_t gathers[2] = {};
    iree_io_parameter_scatter_t scatters[2] = {};
    for (size_t i = 0; i < 2; ++i) {
      const auto waits = i ? iree_hal_semaphore_list_empty()
                           : iree_hal_semaphore_list_t{1, &gate_ptr, &value};
      const iree_hal_semaphore_list_t signals = {1, &done_ptrs[i], &value};
      gathers[i] = {IREE_SV("model"), nullptr, 0, {}, waits, signals};
      scatters[i] = {IREE_SV("model"), nullptr, 0, {}, waits, signals};
    }
    if (scatter) {
      IREE_EXPECT_OK(iree_io_parameter_provider_scatter_batch(
          provider.get(), device(), queue(), 2, scatters));
    } else {
      IREE_EXPECT_OK(iree_io_parameter_provider_gather_batch(
          provider.get(), device(), queue(), 2, gathers));
    }
    IREE_EXPECT_OK(iree_hal_semaphore_wait(
        done[1].get(), 1, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    uint64_t observed = 99;
    IREE_EXPECT_OK(iree_hal_semaphore_query(done[0].get(), &observed));
    EXPECT_EQ(observed, 0);
    IREE_EXPECT_OK(iree_hal_semaphore_signal(gate.get(), 1, nullptr));
    IREE_EXPECT_OK(iree_hal_semaphore_wait(
        done[0].get(), 1, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  }
}

}  // namespace
