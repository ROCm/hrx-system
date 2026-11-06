// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/hal/driver/amd/gpu/image/hsaco.h"
#include "iree/hal/driver/amd/gpu/image/hsaco_test_util.h"

namespace iree::hal::amd::gpu {
namespace {

static void CorpusArguments(benchmark::Benchmark* benchmark) {
  // Preserve an explicit missing-corpus row on hosts without a HIP compiler.
  const size_t count = std::max(size_t{1}, iree_hal_amd_gpu_hip_images_size());
  for (size_t i = 0; i < count; ++i) {
    for (int64_t segment_length : {0, 64, 4096}) {
      benchmark->Args({static_cast<int64_t>(i), segment_length});
    }
  }
}

static const iree_file_toc_t* SelectImage(benchmark::State& state) {
  if (!iree_hal_amd_gpu_hip_images_size()) {
    state.SkipWithMessage("HIP compiler corpus is unavailable");
    return nullptr;
  }
  const auto* file = &iree_hal_amd_gpu_hip_images_create()[state.range(0)];
  state.SetLabel(file->name);
  state.counters["source_bytes"] = file->size;
  return file;
}

// Includes structural validation, metadata decoding, allocations and teardown.
// Source construction and compiler fixture generation are outside the timing.
static void BM_Admit(benchmark::State& state) {
  const auto* file = SelectImage(state);
  if (!file) {
    return;
  }
  auto* source = testing::ByteSequence::Create(
      iree_make_const_byte_span(file->data, file->size), state.range(1));
  for (auto _ : state) {
    iree_hal_amd_gpu_image_t image = {};
    IREE_CHECK_OK(iree_hal_amd_gpu_hsaco_image_initialize(
        source, iree_allocator_system(), &image));
    benchmark::DoNotOptimize(image);
    iree_hal_amd_gpu_image_deinitialize(&image);
  }
  iree_byte_sequence_release(source);
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * file->size);
}
BENCHMARK(BM_Admit)
    ->ArgNames({"image", "segment_bytes"})
    ->Apply(CorpusArguments)
    ->Unit(benchmark::kMicrosecond);

// Measures only materialization: preflight, source traversal, copies, BSS and
// relocations. Admission and writable storage allocation happen before timing.
static void BM_Load(benchmark::State& state) {
  const auto* file = SelectImage(state);
  if (!file) {
    return;
  }
  auto* source = testing::ByteSequence::Create(
      iree_make_const_byte_span(file->data, file->size), state.range(1));
  iree_hal_amd_gpu_image_t image = {};
  IREE_CHECK_OK(iree_hal_amd_gpu_hsaco_image_initialize(
      source, iree_allocator_system(), &image));
  iree_byte_sequence_release(source);
  std::vector<std::vector<uint8_t>> storage(image.segment_count);
  std::vector<iree_byte_span_t> mappings(image.segment_count);
  size_t mapped_length = 0;
  for (size_t i = 0; i < image.segment_count; ++i) {
    storage[i].resize(image.segments[i].length);
    mappings[i] = iree_make_byte_span(storage[i].data(), storage[i].size());
    mapped_length += storage[i].size();
  }
  state.counters["mapped_bytes"] = mapped_length;
  state.counters["segments"] = image.segment_count;
  state.counters["relocations"] = image.relocation_count;
  state.counters["kernels"] = image.metadata.kernel_count;
  for (auto _ : state) {
    IREE_CHECK_OK(iree_hal_amd_gpu_image_load(&image, UINT64_C(0x100000000), 0,
                                              nullptr, mappings.size(),
                                              mappings.data()));
    benchmark::ClobberMemory();
  }
  iree_hal_amd_gpu_image_deinitialize(&image);
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * mapped_length);
}
BENCHMARK(BM_Load)
    ->ArgNames({"image", "segment_bytes"})
    ->Apply(CorpusArguments)
    ->Unit(benchmark::kMicrosecond);

}  // namespace
}  // namespace iree::hal::amd::gpu
