// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>

#include "benchmark/benchmark.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/commands.h"

namespace iree::hal::amd::gpu {
namespace {

// CPU emission only: no resource capture, native publication, or device wait.
// Addresses and control values stand in for already-prepared launch operands.
static void BM_DirectPacketBaseline(benchmark::State& state) {
  alignas(64) std::array<uint32_t, 16> words;
  uint32_t dimensions[] = {256, 1, 1};
  benchmark::DoNotOptimize(dimensions);
  for (auto _ : state) {
    words[0] = 0xc0031502;
    words[1] = dimensions[0];
    words[2] = dimensions[1];
    words[3] = dimensions[2];
    words[4] = 0x8025;
    benchmark::DoNotOptimize(words.data() + 5);
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * 5 * sizeof(uint32_t));
}
BENCHMARK(BM_DirectPacketBaseline);

static void BM_DirectDispatch(benchmark::State& state) {
  alignas(64) std::array<uint32_t, 16> words;
  uint32_t dimensions[] = {256, 1, 1};
  benchmark::DoNotOptimize(dimensions);
  for (auto _ : state) {
    benchmark::DoNotOptimize(iree_hal_amd_gpu_pm4_emit_dispatch_direct(
        words.data(), dimensions, 0x8025));
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * 5 * sizeof(uint32_t));
}
BENCHMARK(BM_DirectDispatch);

static void BM_IndirectDispatch(benchmark::State& state) {
  alignas(64) std::array<uint32_t, 16> words;
  uint64_t address = UINT64_C(0x1234000080);
  benchmark::DoNotOptimize(address);
  for (auto _ : state) {
    benchmark::DoNotOptimize(iree_hal_amd_gpu_pm4_emit_dispatch_indirect(
        words.data(), address, 0x8005));
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * 4 * sizeof(uint32_t));
}
BENCHMARK(BM_IndirectDispatch);

static void BM_RegisterInterval(benchmark::State& state) {
  alignas(64) std::array<uint32_t, 32> words;
  std::array<uint32_t, 16> values = {};
  const uint32_t count = state.range(0);
  benchmark::DoNotOptimize(values);
  for (auto _ : state) {
    benchmark::DoNotOptimize(iree_hal_amd_gpu_pm4_emit_registers(
        words.data(), 0x2e40, count, values.data()));
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * (count + 2) * sizeof(uint32_t));
}
BENCHMARK(BM_RegisterInterval)->Arg(1)->Arg(2)->Arg(8)->Arg(16);

static void BM_BindAndDispatch(benchmark::State& state) {
  alignas(64) std::array<uint32_t, 64> words;
  std::array<uint32_t, 2> entry = {0x12345678, 0};
  std::array<uint32_t, 2> resources = {0xe0af0000, 0x9e};
  uint32_t resource3 = 0x20;
  uint32_t limits = 0;
  uint32_t local_size[] = {32, 1, 1};
  uint32_t arguments[] = {0x56780000, 0x1234};
  uint32_t dimensions[] = {256, 1, 1};
  benchmark::DoNotOptimize(entry);
  benchmark::DoNotOptimize(resources);
  benchmark::DoNotOptimize(resource3);
  benchmark::DoNotOptimize(limits);
  benchmark::DoNotOptimize(local_size);
  benchmark::DoNotOptimize(arguments);
  benchmark::DoNotOptimize(dimensions);
  for (auto _ : state) {
    auto* end = iree_hal_amd_gpu_pm4_emit_registers(words.data(), 0x2e0c, 2,
                                                    entry.data());
    end = iree_hal_amd_gpu_pm4_emit_registers(end, 0x2e12, 2, resources.data());
    end = iree_hal_amd_gpu_pm4_emit_registers(end, 0x2e28, 1, &resource3);
    end = iree_hal_amd_gpu_pm4_emit_registers(end, 0x2e15, 1, &limits);
    end = iree_hal_amd_gpu_pm4_emit_registers(end, 0x2e07, 3, local_size);
    end = iree_hal_amd_gpu_pm4_emit_registers(end, 0x2e40, 2, arguments);
    end = iree_hal_amd_gpu_pm4_emit_dispatch_direct(end, dimensions, 0x8025);
    benchmark::DoNotOptimize(end);
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * 28 * sizeof(uint32_t));
}
BENCHMARK(BM_BindAndDispatch);

static void BM_CompletionAndNotification(benchmark::State& state) {
  alignas(64) std::array<uint32_t, 16> words;
  uint64_t completion_address = UINT64_C(0x1234000080);
  uint64_t mailbox_address = UINT64_C(0x5678000100);
  benchmark::DoNotOptimize(completion_address);
  benchmark::DoNotOptimize(mailbox_address);
  uint32_t generation = 0;
  for (auto _ : state) {
    auto* end = iree_hal_amd_gpu_pm4_emit_release32(
        words.data(), 0x30e, completion_address, ++generation);
    end =
        iree_hal_amd_gpu_pm4_emit_notify32(end, 0x30e, mailbox_address, 17, 17);
    benchmark::DoNotOptimize(end);
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * words.size() * sizeof(uint32_t));
}
BENCHMARK(BM_CompletionAndNotification);

static void BM_ExecutionAndAcquire(benchmark::State& state) {
  alignas(64) std::array<uint32_t, 16> words;
  uint32_t cache_control = 0xc3a1;
  benchmark::DoNotOptimize(cache_control);
  for (auto _ : state) {
    auto* end = iree_hal_amd_gpu_pm4_emit_compute_flush(words.data());
    end = iree_hal_amd_gpu_pm4_emit_acquire(end, cache_control);
    benchmark::DoNotOptimize(end);
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * 10 * sizeof(uint32_t));
}
BENCHMARK(BM_ExecutionAndAcquire);

static void BM_CopyArgumentAndRelease(benchmark::State& state) {
  alignas(64) std::array<uint32_t, 32> words;
  uint64_t source = UINT64_C(0x1234000080);
  uint64_t target = UINT64_C(0x5678000100);
  benchmark::DoNotOptimize(source);
  benchmark::DoNotOptimize(target);
  uint64_t generation = 0;
  for (auto _ : state) {
    auto* end = iree_hal_amd_gpu_pm4_emit_copy64(words.data(), source, target);
    end = iree_hal_amd_gpu_pm4_emit_copy32(end, source + 8, target + 8);
    end = iree_hal_amd_gpu_pm4_emit_release64(end, 0x30e, target + 16,
                                              ++generation);
    benchmark::DoNotOptimize(end);
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * 20 * sizeof(uint32_t));
}
BENCHMARK(BM_CopyArgumentAndRelease);

static void BM_CapturedWrite(benchmark::State& state) {
  alignas(64) std::array<uint32_t, 260> words;
  std::array<uint32_t, 256> values = {};
  const uint32_t count = state.range(0);
  uint64_t address = UINT64_C(0x1234000080);
  benchmark::DoNotOptimize(values);
  benchmark::DoNotOptimize(address);
  for (auto _ : state) {
    benchmark::DoNotOptimize(iree_hal_amd_gpu_pm4_emit_write(
        words.data(), address, count, values.data()));
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * (count + 4) * sizeof(uint32_t));
}
BENCHMARK(BM_CapturedWrite)->Arg(1)->Arg(2)->Arg(16)->Arg(256);

static void BM_Padding(benchmark::State& state) {
  alignas(64) std::array<uint32_t, 32> words;
  const uint32_t count = state.range(0);
  for (auto _ : state) {
    benchmark::DoNotOptimize(
        iree_hal_amd_gpu_pm4_emit_nop(words.data(), count));
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * count * sizeof(uint32_t));
}
BENCHMARK(BM_Padding)->Arg(2)->Arg(8)->Arg(9);

}  // namespace
}  // namespace iree::hal::amd::gpu
