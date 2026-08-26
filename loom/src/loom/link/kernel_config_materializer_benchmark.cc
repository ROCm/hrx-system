// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmarks body-blind launch-configuration projection.

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/internal/arena.h"
#include "iree/io/vec_stream.h"
#include "loom/format/bytecode/writer.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/link/module_index.h"
#include "loom/link/plan_materializer.h"
#include "loom/link/planner.h"
#include "loom/ops/op_registry.h"
#include "loom/verify/verify.h"

namespace {

static void CheckStatus(iree_status_t status) {
  if (!iree_status_is_ok(status)) iree_status_abort(status);
}

class KernelConfigFixture {
 public:
  explicit KernelConfigFixture(iree_host_size_t implementation_source_bytes)
      : implementation_source_bytes_(implementation_source_bytes) {
    iree_arena_block_pool_initialize(32 * 1024, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    CheckStatus(loom_op_registry_register_all_dialects(&context_));
    CheckStatus(loom_context_finalize(&context_));

    loom_module_t* source_module = Parse(BuildSource());
    bytecode_ = Write(source_module);
    loom_module_free(source_module);
    CheckStatus(loom_link_module_index_allocate(
        &context_, &block_pool_, iree_allocator_system(), &index_));
    CheckStatus(loom_link_module_index_add_bytecode(
        index_, iree_make_const_byte_span(bytecode_.data(), bytecode_.size()),
        IREE_SV("provider.loombc"), /*index_options=*/nullptr,
        /*options=*/nullptr, /*out_provider_ordinal=*/nullptr));
    kernel_ = loom_link_module_index_lookup_name(index_, IREE_SV("benchmark"));
    if (kernel_ == nullptr) std::abort();
    plan_ = BuildPlan();

    const loom_link_module_index_provider_t* provider =
        loom_link_module_index_symbol_provider(index_, kernel_);
    const loom_link_module_index_module_t* indexed_module =
        loom_link_module_index_symbol_module(index_, kernel_);
    const loom_bytecode_module_metadata_t* metadata =
        &provider->bytecode.metadata
             .modules[indexed_module->provider_module_ordinal];
    const loom_bytecode_symbol_metadata_t* symbol =
        &metadata->symbols[kernel_->module_symbol_ordinal];
    const loom_bytecode_region_payload_metadata_t* config_payload =
        &metadata->region_payloads
             [symbol->first_region_payload_index +
              symbol->kernel_workload_region_payload_ordinal_plus_one - 1];
    const loom_bytecode_region_payload_metadata_t* body_payload =
        &metadata
             ->region_payloads[symbol->first_region_payload_index +
                               symbol->body_region_payload_ordinal_plus_one -
                               1];
    config_payload_bytes_ = config_payload->length;
    implementation_payload_bytes_ = body_payload->length;
    if (implementation_payload_bytes_ < implementation_source_bytes_) {
      std::abort();
    }
    std::fill_n(bytecode_.data() + body_payload->absolute_offset,
                body_payload->length, UINT8_C(0xFF));
  }

  KernelConfigFixture(const KernelConfigFixture&) = delete;
  KernelConfigFixture& operator=(const KernelConfigFixture&) = delete;

  ~KernelConfigFixture() {
    loom_link_plan_free(plan_);
    loom_link_module_index_free(index_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_link_plan_materialization_t MaterializePlan(
      iree_arena_allocator_t* arena) {
    loom_link_plan_materialization_environment_t environment = {};
    environment.context = &context_;
    environment.block_pool = &block_pool_;
    environment.allocator = iree_allocator_system();
    loom_link_plan_materialization_t materialization = {};
    CheckStatus(loom_link_plan_materialize(
        plan_, &environment, IREE_SV("projected"), arena, &materialization));
    return materialization;
  }

  iree_arena_block_pool_t* block_pool() { return &block_pool_; }

  loom_link_plan_t* BuildPlan() const {
    const loom_link_plan_root_facet_t root = {
        /*.symbol_ordinal=*/kernel_->ordinal,
        /*.kind=*/LOOM_LINK_SYMBOL_FACET_KERNEL_CONFIGURATION,
    };
    loom_link_plan_options_t options = {};
    options.mode = LOOM_LINK_PLAN_SELECTIVE;
    options.root_facets = {1, &root};
    loom_link_plan_t* plan = nullptr;
    CheckStatus(
        loom_link_plan_build(index_, &options, iree_allocator_system(), &plan));
    return plan;
  }

  iree_host_size_t config_payload_bytes() const {
    return config_payload_bytes_;
  }

  iree_host_size_t implementation_payload_bytes() const {
    return implementation_payload_bytes_;
  }

 private:
  std::string BuildSource() const {
    std::string source = R"(
target.generic<reference> @benchmark_target

kernel.def target(@benchmark_target) @benchmark(%element_count: index) {
  %one = index.constant 1 : index
  %sixty_three = index.constant 63 : index
  %subgroup_size = target.subgroup.size : index
  %rounded_count = index.add %element_count, %sixty_three : index
  %workgroup_count = index.div %rounded_count, %subgroup_size : index
  kernel.launch.config workgroups(%workgroup_count, %one, %one) workgroup_size(%subgroup_size, %one, %one) : index
} launch(%output: buffer) {
  //)";
    source.append(implementation_source_bytes_, 'x');
    source.append(R"(
  %zero = index.constant 0 : index
  kernel.return
}
)");
    return source;
  }

  loom_module_t* Parse(const std::string& source) {
    loom_text_parse_options_t options = {};
    options.diagnostic_sink = {loom_diagnostic_stderr_sink, nullptr};
    options.max_errors = 20;
    loom_module_t* module = nullptr;
    CheckStatus(loom_text_parse(
        iree_make_string_view(source.data(), source.size()),
        IREE_SV("provider.loom"), &context_, &block_pool_, &options, &module));
    if (module == nullptr) std::abort();
    loom_verify_options_t verify_options = {};
    verify_options.sink.fn = loom_diagnostic_stderr_sink;
    verify_options.max_errors = 20;
    loom_verify_result_t verify_result = {};
    CheckStatus(loom_verify_module(module, &verify_options, &verify_result));
    if (verify_result.error_count != 0) std::abort();
    return module;
  }

  std::vector<uint8_t> Write(const loom_module_t* module) {
    iree_io_stream_t* stream = nullptr;
    CheckStatus(iree_io_vec_stream_create(
        IREE_IO_STREAM_MODE_WRITABLE | IREE_IO_STREAM_MODE_SEEKABLE |
            IREE_IO_STREAM_MODE_READABLE | IREE_IO_STREAM_MODE_RESIZABLE,
        4096, iree_allocator_system(), &stream));
    CheckStatus(loom_bytecode_write_module(module, stream, /*options=*/nullptr,
                                           &block_pool_));
    std::vector<uint8_t> bytes(iree_io_stream_length(stream));
    CheckStatus(iree_io_stream_seek(stream, IREE_IO_STREAM_SEEK_SET, 0));
    CheckStatus(
        iree_io_stream_read(stream, bytes.size(), bytes.data(), nullptr));
    iree_io_stream_release(stream);
    return bytes;
  }

  iree_host_size_t implementation_source_bytes_;
  iree_arena_block_pool_t block_pool_ = {};
  loom_context_t context_ = {};
  std::vector<uint8_t> bytecode_;
  loom_link_module_index_t* index_ = nullptr;
  const loom_link_module_index_symbol_t* kernel_ = nullptr;
  loom_link_plan_t* plan_ = nullptr;
  iree_host_size_t config_payload_bytes_ = 0;
  iree_host_size_t implementation_payload_bytes_ = 0;
};

static void BM_PlanKernelConfig(benchmark::State& state) {
  KernelConfigFixture fixture(static_cast<iree_host_size_t>(state.range(0)));
  iree_host_size_t selected_facet_count = 0;
  iree_host_size_t selected_symbol_count = 0;
  for (auto _ : state) {
    loom_link_plan_t* plan = fixture.BuildPlan();
    selected_facet_count = loom_link_plan_facet_count(plan);
    selected_symbol_count = loom_link_plan_symbol_count(plan);
    benchmark::DoNotOptimize(plan);
    state.PauseTiming();
    loom_link_plan_free(plan);
    state.ResumeTiming();
  }
  state.counters["configuration_payload_bytes"] =
      static_cast<double>(fixture.config_payload_bytes());
  state.counters["implementation_payload_bytes"] =
      static_cast<double>(fixture.implementation_payload_bytes());
  state.counters["selected_facets"] = static_cast<double>(selected_facet_count);
  state.counters["selected_symbols"] =
      static_cast<double>(selected_symbol_count);
  state.SetComplexityN(fixture.implementation_payload_bytes());
}

static void BM_MaterializeKernelConfigPlan(benchmark::State& state) {
  KernelConfigFixture fixture(static_cast<iree_host_size_t>(state.range(0)));
  for (auto _ : state) {
    state.PauseTiming();
    iree_arena_allocator_t arena;
    iree_arena_initialize(fixture.block_pool(), &arena);
    state.ResumeTiming();
    loom_link_plan_materialization_t materialization =
        fixture.MaterializePlan(&arena);
    benchmark::DoNotOptimize(materialization.module);
    state.PauseTiming();
    loom_module_free(materialization.module);
    iree_arena_deinitialize(&arena);
    state.ResumeTiming();
  }
  state.counters["configuration_payload_bytes"] =
      static_cast<double>(fixture.config_payload_bytes());
  state.counters["implementation_payload_bytes"] =
      static_cast<double>(fixture.implementation_payload_bytes());
  state.SetBytesProcessed(state.iterations() * fixture.config_payload_bytes());
  state.SetComplexityN(fixture.implementation_payload_bytes());
}

static void ImplementationPayloadScales(benchmark::Benchmark* benchmark) {
  benchmark->Arg(0)->Arg(4 * 1024)->Arg(64 * 1024)->Arg(1024 * 1024);
}

BENCHMARK(BM_PlanKernelConfig)
    ->Apply(ImplementationPayloadScales)
    ->Complexity(benchmark::o1);
BENCHMARK(BM_MaterializeKernelConfigPlan)
    ->Apply(ImplementationPayloadScales)
    ->Complexity(benchmark::o1);

}  // namespace

BENCHMARK_MAIN();
