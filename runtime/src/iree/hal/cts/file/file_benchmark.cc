// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <semaphore>
#include <utility>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/async/frontier_tracker.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/base/status_cc.h"
#include "iree/base/threading/numa.h"
#include "iree/base/tooling/flags.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/init.h"
#include "iree/io/file_handle.h"
#include "iree/testing/benchmark.h"
#include "iree/tooling/device_util.h"

IREE_FLAG(string, file, "", "Existing immutable file to read.");
IREE_FLAG(int64_t, file_length, 0,
          "Source prefix length in bytes; zero uses the entire file.");
IREE_FLAG(int64_t, read_bytes, 65536, "Byte count of each queued read.");
IREE_FLAG(int64_t, read_depth, 8,
          "Maximum number of independent reads in flight.");
IREE_FLAG(int64_t, buffer_bytes, 0,
          "Bytes retained per lane; zero uses read_bytes.");
IREE_FLAG(bool, direct, false, "Open the source for native uncached I/O.");
IREE_FLAG(bool, registered_io, false,
          "Require retained REGISTERED_IO bindings from the buffer pool.");

namespace {

// The device, file, and physical buffers survive benchmark repetitions. Each
// lane has its own timeline because queue completion is not implicitly FIFO.
class FileReads {
 public:
  FileReads() { status_ = Initialize(); }

  ~FileReads() {
    // The registry remains a live tracked allocation while retirement is
    // armed. Its final release, or a later native callback, joins all owners
    // before their allocator and borrowed download storage leave scope.
    const bool join = registry_ != nullptr;
    retiring_.store(true);
    for (auto& lane : lanes_) {
      iree_hal_buffer_release(lane.buffer);
      iree_hal_semaphore_release(lane.timeline);
    }
    iree_hal_pool_release(pool_);
    iree_hal_file_release(file_);
    iree_io_file_mapping_release(oracle_);
    iree_io_file_handle_release(handle_);
    iree_hal_device_group_release(group_);
    iree_hal_device_release(device_);
    iree_async_frontier_tracker_release(tracker_);
    iree_async_proactor_pool_release(proactors_);
    iree_hal_driver_registry_free(registry_);
    if (join) {
      reclaimed_.acquire();
    }
    iree_status_free(status_);
  }

  iree_status_t status() const { return iree_status_clone(status_); }
  uint64_t length() const { return length_; }
  uint64_t extent() const { return extent_; }
  size_t depth() const { return lanes_.size(); }
  uint64_t retained_bytes() const { return retained_bytes_; }
  uint64_t allocation_calls() const { return allocation_calls_.load(); }

  iree_status_t Burst() {
    iree_status_t status = iree_ok_status();
    const uint64_t source_count = length_ / extent_;
    for (size_t i = 0; i < lanes_.size() && iree_status_is_ok(status); ++i) {
      status = Issue(lanes_[i], (i % source_count) * extent_, extent_);
    }
    return iree_status_join(status, Join());
  }

  iree_status_t Stream() {
    iree_status_t status = iree_ok_status();
    size_t index = 0;
    for (uint64_t offset = 0; offset < length_ && iree_status_is_ok(status);
         offset += std::min(extent_, length_ - offset)) {
      auto& lane = lanes_[index];
      status = Wait(lane);
      if (iree_status_is_ok(status)) {
        status = Issue(lane, offset, std::min(extent_, length_ - offset));
      }
      index = (index + 1) % lanes_.size();
    }
    return iree_status_join(status, Join());
  }

  // Consume the results through a real queue download instead of requiring
  // host-visible placement solely for the benchmark's byte oracle.
  iree_status_t Verify() {
    const auto expected = iree_io_file_mapping_contents_ro(oracle_);
    iree_status_t status = iree_ok_status();
    for (size_t i = 0; i < lanes_.size() && iree_status_is_ok(status); ++i) {
      auto& lane = lanes_[i];
      if (!lane.length) {
        continue;
      }
      const iree_hal_transfer_operation_t operation = {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
          .download =
              {
                  .source_buffer = lane.buffer,
                  .target = verification_.data(),
                  .length = lane.length,
              },
      };
      uint64_t next = lane.value + 1;
      status = iree_hal_queue_transfer(queue_, {1, &lane.timeline, &lane.value},
                                       {1, &lane.timeline, &next}, 1,
                                       &operation, nullptr);
      if (iree_status_is_ok(status)) {
        lane.value = next;
        status = Wait(lane);
      }
      if (iree_status_is_ok(status) &&
          std::memcmp(verification_.data(), expected.data + lane.offset,
                      lane.length) != 0) {
        status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                  "lane %zu differs from source at offset "
                                  "%" PRIu64 " for %" PRIu64 " bytes",
                                  i, lane.offset, lane.length);
      }
    }
    return status;
  }

 private:
  struct Lane {
    // Owned physical destination, reused only after its timeline completes.
    iree_hal_buffer_t* buffer = nullptr;
    // Owned completion timeline for this lane alone.
    iree_hal_semaphore_t* timeline = nullptr;
    // Last accepted operation's timepoint.
    uint64_t value = 0;
    // Source offset of the last read, used by byte verification.
    uint64_t offset = 0;
    // Exact length of the last read, including a shorter stream tail.
    uint64_t length = 0;
  };

  iree_status_t Initialize() {
    if (!FLAG_file[0] || FLAG_file_length < 0 || FLAG_read_bytes <= 0 ||
        FLAG_read_depth <= 0 || FLAG_buffer_bytes < 0 ||
        (FLAG_buffer_bytes && FLAG_buffer_bytes < FLAG_read_bytes)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "file is required; read_bytes/read_depth must be positive, "
          "file_length/buffer_bytes nonnegative, and buffer_bytes >= "
          "read_bytes");
    }
    const iree_allocator_t allocator = {this, Allocate};
    IREE_RETURN_IF_ERROR(
        iree_hal_driver_registry_allocate(allocator, &registry_));
    IREE_RETURN_IF_ERROR(iree_hal_register_all_available_drivers(registry_));
    IREE_RETURN_IF_ERROR(iree_async_proactor_pool_create(
        iree_numa_node_count(), nullptr,
        iree_async_proactor_pool_options_default(), allocator, &proactors_));
    IREE_RETURN_IF_ERROR(iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), allocator, &tracker_));
    auto params = iree_hal_device_create_params_default();
    params.proactor_pool = proactors_;
    IREE_RETURN_IF_ERROR(iree_hal_create_device_from_flags(
        registry_, IREE_SV("task"), &params, allocator, &device_));
    IREE_RETURN_IF_ERROR(iree_hal_device_group_create_from_device(
        device_, tracker_, allocator, &group_));
    const auto* queues =
        iree_hal_device_spec_queues(iree_hal_device_spec(device_));
    uint32_t family_ordinal = 0;
    for (; family_ordinal < queues->family_count; ++family_ordinal) {
      const auto& family = queues->families[family_ordinal];
      if (family.provisioned_queue_count &&
          (family.role_flags & IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER)) {
        queue_ = iree_hal_device_queue(device_, family_ordinal, 0);
        break;
      }
    }
    if (!queue_) {
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "device has no provisioned transfer queue");
    }
    const uint64_t affinity = UINT64_C(1) << family_ordinal;
    const auto mode = IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_SHARE_READ |
                      IREE_IO_FILE_MODE_ASYNC |
                      (FLAG_direct ? IREE_IO_FILE_MODE_DIRECT : 0);
    IREE_RETURN_IF_ERROR(iree_io_file_handle_open(
        mode, iree_make_cstring_view(FLAG_file), allocator, &handle_));
    IREE_RETURN_IF_ERROR(iree_hal_file_import(device_, nullptr, affinity,
                                              IREE_HAL_MEMORY_ACCESS_READ,
                                              handle_, 0, &file_));
    extent_ = FLAG_read_bytes;
    length_ = FLAG_file_length ? FLAG_file_length : iree_hal_file_length(file_);
    if (length_ > iree_hal_file_length(file_) || length_ < extent_ ||
        length_ > IREE_HOST_SIZE_MAX || extent_ > IREE_HOST_SIZE_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "read extent must fit the requested source range");
    }
    const auto& requirements = iree_hal_file_requirements(file_)->read;
    if (extent_ % requirements.offset_alignment ||
        extent_ % requirements.length_alignment ||
        length_ % requirements.length_alignment) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "read_bytes/file_length violate native alignment");
    }
    IREE_RETURN_IF_ERROR(iree_io_file_map_view(
        handle_, IREE_IO_FILE_ACCESS_READ, 0, length_,
        IREE_IO_FILE_MAPPING_FLAG_NONE, allocator, &oracle_));
    const iree_hal_pool_family_access_t access = {
        .family = iree_hal_queue_family(queue_),
        .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
        .interfaces = FLAG_registered_io
                          ? UINT64_C(1)
                                << IREE_HAL_BUFFER_INTERFACE_REGISTERED_IO
                          : 0,
    };
    const iree_hal_pool_scope_t scope = {1, &access, {}};
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    IREE_RETURN_IF_ERROR(
        iree_hal_slab_pool_create(group_, scope, &options, allocator, &pool_));
    const auto buffer_length = FLAG_buffer_bytes ? FLAG_buffer_bytes : extent_;
    const auto request = iree_hal_make_pool_reservation_request(
        buffer_length,
        std::max(iree_host_size_t{64}, requirements.buffer_alignment));
    lanes_.resize(FLAG_read_depth);
    verification_.resize(extent_);
    iree_status_t status = iree_ok_status();
    for (size_t i = 0; i < lanes_.size() && iree_status_is_ok(status); ++i) {
      status = iree_hal_pool_allocate_buffer(
          pool_, request, iree_infinite_timeout(), &lanes_[i].buffer);
      if (iree_status_is_ok(status)) {
        retained_bytes_ += iree_hal_buffer_allocation_size(lanes_[i].buffer);
        status = iree_hal_semaphore_create(device_, affinity, 0,
                                           IREE_HAL_SEMAPHORE_FLAG_NONE,
                                           &lanes_[i].timeline);
      }
    }
    return status;
  }

  iree_status_t Issue(Lane& lane, uint64_t offset, uint64_t length) {
    uint64_t next = lane.value + 1;
    IREE_RETURN_IF_ERROR(iree_hal_queue_read(
        queue_, {lane.value ? 1u : 0u, &lane.timeline, &lane.value},
        {1, &lane.timeline, &next}, file_, offset, lane.buffer, 0, length,
        nullptr, IREE_HAL_READ_FLAG_NONE));
    lane.value = next;
    lane.offset = offset;
    lane.length = length;
    return iree_ok_status();
  }

  static iree_status_t Wait(const Lane& lane) {
    return lane.value ? iree_hal_semaphore_wait(lane.timeline, lane.value,
                                                iree_infinite_timeout(),
                                                IREE_ASYNC_WAIT_FLAG_NONE)
                      : iree_ok_status();
  }

  iree_status_t Join() {
    iree_status_t status = iree_ok_status();
    for (const auto& lane : lanes_) {
      status = iree_status_join(status, Wait(lane));
    }
    return status;
  }

  static iree_status_t Allocate(void* user_data,
                                iree_allocator_command_t command,
                                const void* params, void** inout_pointer) {
    auto* self = static_cast<FileReads*>(user_data);
    const bool had_pointer = (command == IREE_ALLOCATOR_COMMAND_FREE ||
                              command == IREE_ALLOCATOR_COMMAND_REALLOC) &&
                             *inout_pointer != nullptr;
    auto system = iree_allocator_system();
    IREE_RETURN_IF_ERROR(
        system.ctl(system.self, command, params, inout_pointer));
    if (command == IREE_ALLOCATOR_COMMAND_FREE) {
      if (had_pointer && self->live_allocations_.fetch_sub(1) == 1 &&
          self->retiring_.load()) {
        self->reclaimed_.release();
      }
    } else {
      self->allocation_calls_.fetch_add(1);
      if (!had_pointer) {
        self->live_allocations_.fetch_add(1);
      }
    }
    return iree_ok_status();
  }

  // Retained initialization failure; subsequent benchmark rows report it too.
  iree_status_t status_ = iree_ok_status();
  // Owned driver cache, retained until all benchmark submissions are finished.
  iree_hal_driver_registry_t* registry_ = nullptr;
  // Caller-supplied I/O progress owners shared with the selected device.
  iree_async_proactor_pool_t* proactors_ = nullptr;
  // Owned group completion tracker.
  iree_async_frontier_tracker_t* tracker_ = nullptr;
  // Owned selected device, also retained by its group and pools.
  iree_hal_device_t* device_ = nullptr;
  // Owned sealed group providing the pool's memory contract.
  iree_hal_device_group_t* group_ = nullptr;
  // Borrowed provisioned transfer queue, valid through device retirement.
  iree_hal_queue_t* queue_ = nullptr;
  // Owned source handle, shared with the HAL import and byte oracle.
  iree_io_file_handle_t* handle_ = nullptr;
  // Owned asynchronous HAL file using the selected queue family's proactor.
  iree_hal_file_t* file_ = nullptr;
  // Owned read-only source mapping, accessed only outside timing.
  iree_io_file_mapping_t* oracle_ = nullptr;
  // Owned physical pool, without a public host-mapping requirement.
  iree_hal_pool_t* pool_ = nullptr;
  // Independent, retained destinations and completion identities.
  std::vector<Lane> lanes_;
  // Owned download destination used only by untimed byte verification.
  std::vector<uint8_t> verification_;
  // Exact source prefix length, never implicitly rounded for direct I/O.
  uint64_t length_ = 0;
  // Maximum bytes per queue_read operation.
  uint64_t extent_ = 0;
  // Sum of the actual buffer allocation sizes reported by the HAL.
  uint64_t retained_bytes_ = 0;
  // Successful allocate/reallocate calls through the runtime's host allocator.
  std::atomic<uint64_t> allocation_calls_{0};
  // Live runtime allocations, including the registry until final teardown.
  std::atomic<uint64_t> live_allocations_{0};
  // Allows the last resource to publish terminal allocator ownership.
  std::atomic<bool> retiring_{false};
  // Joins asynchronous final resource destruction without a polling timer.
  std::binary_semaphore reclaimed_{0};
};

std::unique_ptr<FileReads> reads;

bool Check(benchmark::State& state, iree_status_t raw_status) {
  iree::Status status(std::move(raw_status));
  if (status.ok()) {
    return true;
  }
  state.SkipWithError(status.ToString().c_str());
  return false;
}

void Read(benchmark::State& state, bool stream) {
  if (!reads) {
    reads = std::make_unique<FileReads>();
  }
  if (!Check(state, reads->status())) {
    return;
  }
  const auto operation = stream ? &FileReads::Stream : &FileReads::Burst;
  if (!Check(state, ((*reads).*operation)()) ||
      !Check(state, reads->Verify())) {
    return;
  }
  const uint64_t allocations_before = reads->allocation_calls();
  for (auto _ : state) {
    if (!Check(state, ((*reads).*operation)())) {
      break;
    }
  }
  const uint64_t allocations = reads->allocation_calls() - allocations_before;
  if (state.skipped() || !Check(state, reads->Verify())) {
    return;
  }
  const uint64_t operations =
      stream ? (reads->length() - 1) / reads->extent() + 1 : reads->depth();
  const uint64_t bytes =
      stream ? reads->length() : reads->extent() * reads->depth();
  state.SetItemsProcessed(state.iterations() * operations);
  state.SetBytesProcessed(state.iterations() * bytes);
  state.counters["host_allocations_per_read"] =
      static_cast<double>(allocations) / (state.iterations() * operations);
  state.counters["read_bytes"] = reads->extent();
  state.counters["read_depth"] = reads->depth();
  state.counters["retained_bytes"] = reads->retained_bytes();
  state.counters["direct"] = FLAG_direct;
  state.counters["registered_io"] = FLAG_registered_io;
}

void BM_ReadBurst(benchmark::State& state) { Read(state, false); }
BENCHMARK(BM_ReadBurst)->MeasureProcessCPUTime()->UseRealTime();

void BM_ReadStream(benchmark::State& state) { Read(state, true); }
BENCHMARK(BM_ReadStream)->MeasureProcessCPUTime()->UseRealTime();

}  // namespace

int main(int argc, char** argv) {
  iree_benchmark_initialize(&argc, argv);
  const bool succeeded = iree_benchmark_run_specified();
  reads.reset();
  return succeeded ? 0 : 1;
}
