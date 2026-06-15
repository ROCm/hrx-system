// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmarks for Loom bytecode reader throughput.
//
// Fixtures are generated and serialized before timing starts. The measured
// loops cover metadata-only validation, full IR materialization, and
// materialization plus verifier handoff for representative generated modules.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/internal/arena.h"
#include "iree/io/vec_stream.h"
#include "loom/format/bytecode/reader.h"
#include "loom/format/bytecode/writer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/test/ops.h"
#include "loom/testing/gen.h"

namespace {

static void IgnoreStatusOrAbort(iree_status_t status) {
  if (iree_status_is_ok(status)) {
    return;
  }
  iree_status_abort(status);
}

static iree_status_t IgnoreDiagnostic(void* user_data,
                                      const loom_diagnostic_t* diagnostic) {
  (void)diagnostic;
  auto* diagnostic_count = static_cast<uint32_t*>(user_data);
  ++*diagnostic_count;
  return iree_ok_status();
}

class BytecodeFixture {
 public:
  BytecodeFixture(uint8_t preset, uint32_t scale) {
    iree_arena_block_pool_initialize(65536, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = loom_test_dialect_vtables(&count);
    IgnoreStatusOrAbort(loom_context_register_dialect(
        &context_, LOOM_DIALECT_TEST, vtables, (uint16_t)count));
    IgnoreStatusOrAbort(loom_context_finalize(&context_));

    loom_test_gen_module_config_t config =
        loom_test_gen_module_config_fuzz_preset(preset, scale);
    loom_test_gen_t generator;
    loom_test_gen_initialize_seeded(
        UINT64_C(0xBADC0DE000000000) | ((uint64_t)preset << 32) | scale,
        &generator);

    loom_module_t* module = nullptr;
    IgnoreStatusOrAbort(loom_test_gen_module(&generator, &config, &context_,
                                             &block_pool_, &module));
    if (!module) {
      abort();
    }
    bytes_ = WriteModule(module);
    loom_module_free(module);
  }

  ~BytecodeFixture() {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  const std::vector<uint8_t>& bytes() const { return bytes_; }
  loom_context_t* context() { return &context_; }

 private:
  std::vector<uint8_t> WriteModule(const loom_module_t* module) {
    iree_io_stream_t* stream = nullptr;
    IgnoreStatusOrAbort(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
            IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE,
        4096, iree_allocator_system(), &stream));
    IgnoreStatusOrAbort(
        loom_bytecode_write_module(module, stream, nullptr, &block_pool_));

    iree_io_stream_pos_t length = iree_io_stream_length(stream);
    std::vector<uint8_t> bytes((size_t)length);
    IgnoreStatusOrAbort(
        iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
    IgnoreStatusOrAbort(
        iree_io_stream_read(stream, bytes.size(), bytes.data(), nullptr));
    iree_io_stream_release(stream);
    return bytes;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  std::vector<uint8_t> bytes_;
};

static loom_bytecode_read_options_t ReadOptions(bool verify_module,
                                                uint32_t* diagnostic_count) {
  return loom_bytecode_read_options_t{
      /*.diagnostic_sink=*/
      {
          /*.fn=*/IgnoreDiagnostic,
          /*.user_data=*/diagnostic_count,
      },
      /*.verify_module=*/verify_module,
      /*.verify_max_errors=*/16,
  };
}

static void BenchmarkReadMetadata(benchmark::State& state, uint8_t preset) {
  BytecodeFixture fixture(preset, (uint32_t)state.range(0));
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(65536, iree_allocator_system(), &block_pool);
  uint32_t diagnostic_count = 0;
  loom_bytecode_read_options_t options =
      ReadOptions(/*verify_module=*/false, &diagnostic_count);

  for (auto _ : state) {
    loom_bytecode_read_result_t result = {0};
    IgnoreStatusOrAbort(loom_bytecode_read_metadata(
        iree_make_const_byte_span(fixture.bytes().data(),
                                  fixture.bytes().size()),
        IREE_SV("benchmark.loombc"), fixture.context(), &block_pool, &options,
        &result));
    benchmark::DoNotOptimize(result.first_module.op_count);
  }

  state.SetBytesProcessed(state.iterations() * fixture.bytes().size());
  iree_arena_block_pool_deinitialize(&block_pool);
}

static void BenchmarkReadModule(benchmark::State& state, uint8_t preset,
                                bool verify_module) {
  BytecodeFixture fixture(preset, (uint32_t)state.range(0));
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(65536, iree_allocator_system(), &block_pool);
  uint32_t diagnostic_count = 0;
  loom_bytecode_read_options_t options =
      ReadOptions(verify_module, &diagnostic_count);

  for (auto _ : state) {
    loom_bytecode_read_result_t result = {0};
    loom_module_t* module = nullptr;
    IgnoreStatusOrAbort(loom_bytecode_read_module(
        iree_make_const_byte_span(fixture.bytes().data(),
                                  fixture.bytes().size()),
        IREE_SV("benchmark.loombc"), fixture.context(), &block_pool, &options,
        &result, &module, iree_allocator_system()));
    benchmark::DoNotOptimize(module);
    if (module) {
      loom_module_free(module);
    }
  }

  state.SetBytesProcessed(state.iterations() * fixture.bytes().size());
  iree_arena_block_pool_deinitialize(&block_pool);
}

static void BM_ReadMetadata_Representative(benchmark::State& state) {
  BenchmarkReadMetadata(state, /*preset=*/0);
}
BENCHMARK(BM_ReadMetadata_Representative)->Arg(1)->Arg(5)->Arg(10);

static void BM_ReadMetadata_NestingStress(benchmark::State& state) {
  BenchmarkReadMetadata(state, /*preset=*/3);
}
BENCHMARK(BM_ReadMetadata_NestingStress)->Arg(1)->Arg(5)->Arg(10);

static void BM_ReadMetadata_FormatStress(benchmark::State& state) {
  BenchmarkReadMetadata(state, /*preset=*/4);
}
BENCHMARK(BM_ReadMetadata_FormatStress)->Arg(1)->Arg(5)->Arg(10);

static void BM_ReadModule_Representative(benchmark::State& state) {
  BenchmarkReadModule(state, /*preset=*/0, /*verify_module=*/false);
}
BENCHMARK(BM_ReadModule_Representative)->Arg(1)->Arg(5)->Arg(10);

static void BM_ReadModule_NestingStress(benchmark::State& state) {
  BenchmarkReadModule(state, /*preset=*/3, /*verify_module=*/false);
}
BENCHMARK(BM_ReadModule_NestingStress)->Arg(1)->Arg(5)->Arg(10);

static void BM_ReadModule_FormatStress(benchmark::State& state) {
  BenchmarkReadModule(state, /*preset=*/4, /*verify_module=*/false);
}
BENCHMARK(BM_ReadModule_FormatStress)->Arg(1)->Arg(5)->Arg(10);

static void BM_ReadModuleVerify_Representative(benchmark::State& state) {
  BenchmarkReadModule(state, /*preset=*/0, /*verify_module=*/true);
}
BENCHMARK(BM_ReadModuleVerify_Representative)->Arg(1)->Arg(5)->Arg(10);

static void BM_ReadModuleVerify_FormatStress(benchmark::State& state) {
  BenchmarkReadModule(state, /*preset=*/4, /*verify_module=*/true);
}
BENCHMARK(BM_ReadModuleVerify_FormatStress)->Arg(1)->Arg(5)->Arg(10);

}  // namespace

BENCHMARK_MAIN();
