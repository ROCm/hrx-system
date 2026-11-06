// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cerrno>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/async/platform/io_uring/defs.h"
#include "iree/async/platform/io_uring/sparse_table.h"
#include "iree/async/platform/io_uring/uring.h"
#include "iree/async/platform/io_uring/uring_registration.h"
#include "iree/base/api.h"

namespace {

// Native table replacement only: pin creation and retirement are outside
// timing. One shared physical page makes occupancy a metadata cost, not a
// memory load.
class RegisteredBuffers {
 public:
  ~RegisteredBuffers() {
    if (source_fd_ >= 0) {
      iree_io_uring_registration_release_buffers(source_fd_);
    }
    if (initialized_) {
      iree_io_uring_ring_deinitialize(&ring_);
    }
  }

  iree_status_t Initialize(uint32_t live_count, bool compose,
                           const char** out_skip_reason) {
    *out_skip_reason = nullptr;
    auto options = iree_io_uring_ring_options_default();
    options.sq_entries = 2;
    IREE_RETURN_IF_ERROR(iree_io_uring_ring_initialize(options, &ring_));
    initialized_ = true;
    iree_io_uring_rsrc_register_t table = {
        .nr = kCapacity,
        .flags = IREE_IORING_RSRC_REGISTER_SPARSE,
    };
    const int table_result = iree_io_uring_ring_register(
        &ring_, IREE_IORING_REGISTER_BUFFERS2, &table, sizeof(table));
    if (IsUnavailable(table_result)) {
      *out_skip_reason = "kernel cannot create sparse buffer tables";
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(CheckNative(table_result));
    std::vector<iovec> buffers(live_count, {backing_, sizeof(backing_)});
    IREE_RETURN_IF_ERROR(CheckNative(iree_io_uring_registration_prepare_buffers(
        buffers.data(), live_count, &source_fd_)));
    replacement_.src_fd = source_fd_;
    replacement_.flags = IREE_IORING_REGISTER_DST_REPLACE;
    replacement_.nr = live_count;
    const int clone_result = Replace();
    if (IsUnavailable(clone_result)) {
      *out_skip_reason = "kernel cannot clone registered buffer ranges";
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(CheckNative(clone_result));
    uint64_t retirement_tag = 1;
    iree_io_uring_rsrc_update2_t tagged = {
        .offset = live_count - 1,
        .data = reinterpret_cast<uintptr_t>(buffers.data()),
        .tags = reinterpret_cast<uintptr_t>(&retirement_tag),
        .nr = 1,
    };
    IREE_RETURN_IF_ERROR(CheckNative(iree_io_uring_ring_register(
        &ring_, IREE_IORING_REGISTER_BUFFERS_UPDATE, &tagged, sizeof(tagged))));
    replacement_.nr = 1;
    ring_.registration.buffer_clone_capacity = compose ? kCapacity : 0;
    // Allocate the reusable scratch descriptor before measuring replacements.
    IREE_RETURN_IF_ERROR(CheckNative(Replace()));
    if (!compose && iree_io_uring_ring_cq_ready(&ring_)) {
      *out_skip_reason = "native clone drops live suffix entries";
    }
    iree_io_uring_ring_cq_advance(&ring_, iree_io_uring_ring_cq_count(&ring_));
    return iree_ok_status();
  }

  int Replace() {
    return iree_io_uring_ring_register(
        &ring_, IREE_IORING_REGISTER_CLONE_BUFFERS, &replacement_, 1);
  }

 private:
  // Only initial, known-valid capability probes use this classification.
  static bool IsUnavailable(int result) {
    return result == -EINVAL || result == -ENOSYS || result == -EOPNOTSUPP ||
           result == -EPERM || result == -EACCES;
  }

  static iree_status_t CheckNative(int result) {
    return result < 0
               ? iree_make_status(iree_status_code_from_errno(-result),
                                  "native registration failed (%d)", -result)
               : iree_ok_status();
  }

  // Matches the proactor's default sparse table capacity.
  static constexpr uint32_t kCapacity =
      IREE_IO_URING_SPARSE_TABLE_DEFAULT_CAPACITY;
  // Destination table and its optional owner-private scratch ring.
  iree_io_uring_ring_t ring_ = {};
  // Distinguishes an initialized ring from zero-initialized storage.
  bool initialized_ = false;
  // One registered physical page shared by all live table entries.
  alignas(4096) uint8_t backing_[4096] = {};
  // Immutable pin owner retained through all replacements.
  int source_fd_ = -1;
  // Single-range replacement reused without user-space allocation.
  iree_io_uring_clone_buffers_t replacement_ = {};
};

template <bool Compose>
void BM_Attach(benchmark::State& state) {
  RegisteredBuffers buffers;
  const char* skip_reason = nullptr;
  iree::Status status = buffers.Initialize(
      static_cast<uint32_t>(state.range(0)), Compose, &skip_reason);
  if (!status.ok()) {
    if (iree_status_is_unavailable(status.get())) {
      state.SkipWithMessage("io_uring is unavailable on this host");
    } else {
      state.SkipWithError(status.ToString().c_str());
    }
    return;
  }
  if (skip_reason) {
    state.SkipWithMessage(skip_reason);
    return;
  }
  for (auto _ : state) {
    if (buffers.Replace() < 0) {
      state.SkipWithError("native buffer replacement failed");
      break;
    }
  }
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK_TEMPLATE(BM_Attach, false)->Arg(2)->Arg(64)->Arg(512);
BENCHMARK_TEMPLATE(BM_Attach, true)->Arg(2)->Arg(64)->Arg(512);

}  // namespace
