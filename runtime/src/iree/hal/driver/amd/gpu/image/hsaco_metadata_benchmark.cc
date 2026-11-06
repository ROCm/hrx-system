// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/hal/driver/amd/gpu/image/hsaco_metadata.h"
#include "iree/hal/driver/amd/gpu/image/hsaco_test_util.h"

namespace iree::hal::amd::gpu {
namespace {

// Measures only MessagePack admission, including record allocation and
// teardown. Compiler output and ELF note extraction happen outside the timed
// loop. Batch rows repeat the same note to isolate scaling of the multi-payload
// API.
static void BM_Decode(benchmark::State& state) {
  if (!iree_hal_amd_gpu_hip_images_size()) {
    state.SkipWithMessage("HIP compiler corpus is unavailable");
    return;
  }
  const auto& file = iree_hal_amd_gpu_hip_images_create()[state.range(0)];
  const auto notes = testing::HipMetadataPayloads(file);
  testing::Check(!notes.empty());
  std::vector<iree_const_byte_span_t> payloads;
  for (int64_t i = 0; i < state.range(1); ++i) {
    payloads.insert(payloads.end(), notes.begin(), notes.end());
  }
  size_t byte_length = 0;
  for (const auto payload : payloads) {
    byte_length += payload.data_length;
  }
  state.SetLabel(file.name);
  state.counters["metadata_bytes"] = byte_length;
  state.counters["notes"] = payloads.size();
  size_t kernel_count = 0;
  size_t argument_count = 0;
  for (auto _ : state) {
    iree_hal_amd_gpu_metadata_t metadata = {};
    IREE_CHECK_OK(iree_hal_amd_gpu_hsaco_metadata_initialize(
        payloads.size(), payloads.data(), iree_allocator_system(), &metadata));
    kernel_count = metadata.kernel_count;
    argument_count = metadata.argument_count;
    benchmark::DoNotOptimize(metadata);
    iree_hal_amd_gpu_metadata_deinitialize(&metadata);
  }
  state.counters["kernels"] = kernel_count;
  state.counters["arguments"] = argument_count;
  state.SetItemsProcessed(state.iterations() * payloads.size());
  state.SetBytesProcessed(state.iterations() * byte_length);
}

static void CorpusArguments(benchmark::Benchmark* benchmark) {
  const size_t count = std::max(size_t{1}, iree_hal_amd_gpu_hip_images_size());
  for (size_t i = 0; i < count; ++i) {
    for (int64_t note_count : {1, 8, 64}) {
      benchmark->Args({static_cast<int64_t>(i), note_count});
    }
  }
}
BENCHMARK(BM_Decode)
    ->ArgNames({"image", "copies"})
    ->Apply(CorpusArguments)
    ->Unit(benchmark::kMicrosecond);

}  // namespace
}  // namespace iree::hal::amd::gpu
