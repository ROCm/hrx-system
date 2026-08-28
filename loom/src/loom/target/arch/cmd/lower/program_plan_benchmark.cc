// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmarks command-plan preparation at materialized and indexed boundaries.

#include <chrono>
#include <cstdlib>
#include <string>
#include <vector>

#include "benchmark/benchmark.h"
#include "iree/base/internal/arena.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/link/module_index.h"
#include "loom/link/plan_materializer.h"
#include "loom/link/planner.h"
#include "loom/ops/op_registry.h"
#include "loom/pass/builtin_registry.h"
#include "loom/target/arch/cmd/lower/program_plan.h"
#include "loom/target/arch/cmd/lower/program_plan_index.h"
#include "loom/verify/verify.h"

namespace {

static void CheckStatus(iree_status_t status) {
  if (!iree_status_is_ok(status)) iree_status_abort(status);
}

enum class LaunchShape {
  // Configured dispatches with no launch-configuration computation.
  kDirect,
  // Logical launches sharing one exact workload SSA value.
  kRepeatedWorkload,
  // Logical launches whose workloads differ after loop unrolling.
  kDistinctWorkloads,
  // Equal launch arguments selecting one implementation class.
  kRepeatedKernelClass,
  // Distinct launch arguments selecting one or two implementation classes.
  kDistinctKernelClasses,
};

enum class KernelRequestMode {
  // Prepare command artifacts without opening kernel implementation bodies.
  kBodyBlind,
  // Publish independently owned products for reachable kernel classes.
  kPublish,
};

class ProgramPlanFixture {
 public:
  ProgramPlanFixture(LaunchShape launch_shape, iree_host_size_t launch_count)
      : launch_count_(launch_count) {
    iree_arena_block_pool_initialize(32 * 1024, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    CheckStatus(loom_op_registry_register_all_dialects(&context_));
    CheckStatus(loom_context_finalize(&context_));

    source_module_ = ParseAndVerify(BuildSource(launch_shape, launch_count));
    CheckStatus(loom_link_module_index_allocate(
        &context_, &block_pool_, iree_allocator_system(), &index_));
    loom_link_module_index_add_options_t add_options = {};
    add_options.provider_name = IREE_SV("command_plan_benchmark");
    CheckStatus(loom_link_module_index_add_materialized(
        index_, source_module_, &add_options,
        /*out_provider_ordinal=*/nullptr));

    root_symbol_ = loom_link_module_index_lookup_name(index_, IREE_SV("root"));
    if (root_symbol_ == nullptr) std::abort();
    const loom_link_plan_root_facet_t root_facet = {
        /*.symbol_ordinal=*/root_symbol_->ordinal,
        /*.kind=*/LOOM_LINK_SYMBOL_FACET_COMMAND_IMPLEMENTATION,
    };
    loom_link_plan_options_t options = {};
    options.mode = LOOM_LINK_PLAN_LINK;
    options.root_facets = {1, &root_facet};
    options.dependency_policy = LOOM_LINK_PLAN_DEPENDENCY_REQUESTED_FACETS;
    CheckStatus(loom_link_plan_build(index_, &options, iree_allocator_system(),
                                     &plan_));
  }

  ProgramPlanFixture(const ProgramPlanFixture&) = delete;
  ProgramPlanFixture& operator=(const ProgramPlanFixture&) = delete;

  ~ProgramPlanFixture() {
    loom_link_plan_free(plan_);
    loom_link_module_index_free(index_);
    loom_module_free(source_module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_link_plan_materialization_t Materialize(
      iree_arena_allocator_t* arena, loom_symbol_ref_t* out_root_ref) {
    loom_link_plan_materialization_environment_t environment = {};
    environment.context = &context_;
    environment.block_pool = &block_pool_;
    environment.allocator = iree_allocator_system();
    loom_link_plan_materialization_t materialization = {};
    CheckStatus(loom_link_plan_materialize(plan_, &environment,
                                           IREE_SV("command_plan_benchmark"),
                                           arena, &materialization));
    if (root_symbol_->ordinal >= materialization.target_symbols.count) {
      std::abort();
    }
    *out_root_ref =
        materialization.target_symbols.values[root_symbol_->ordinal];
    if (!loom_symbol_ref_is_valid(*out_root_ref)) std::abort();
    return materialization;
  }

  iree_arena_block_pool_t* block_pool() { return &block_pool_; }

  loom_context_t* context() { return &context_; }

  const loom_link_module_index_t* index() const { return index_; }

  iree_host_size_t root_symbol_ordinal() const { return root_symbol_->ordinal; }

  iree_host_size_t launch_count() const { return launch_count_; }

 private:
  static std::string BuildSource(LaunchShape launch_shape,
                                 iree_host_size_t launch_count) {
    std::string source;
    if (launch_shape == LaunchShape::kDirect) {
      source.append("kernel.entry.decl @entry()\n\n");
    } else {
      if (launch_shape == LaunchShape::kRepeatedKernelClass ||
          launch_shape == LaunchShape::kDistinctKernelClasses) {
        source.append(R"(
template.decl @request.schedule(%extent: index)

template.def<@request.schedule> priority(10) @large(%extent: index) where [ge(%extent, 128)] {
  template.return
}

template.def<@request.schedule> priority(1) @small(%extent: index) {
  template.return
}

)");
      }
      source.append(R"(
kernel.def @tiles(%extent: index) {
  %c1 = index.constant 1 : index
  %rounding = index.constant 15 : index
  %tile_size = index.constant 16 : index
  %rounded = index.add %extent, %rounding : index
  %groups = index.div %rounded, %tile_size : index
  kernel.launch.config workgroups(%groups, %c1, %c1) workgroup_size(%tile_size, %c1, %c1) : index
)");
      if (launch_shape == LaunchShape::kRepeatedKernelClass ||
          launch_shape == LaunchShape::kDistinctKernelClasses) {
        source.append(R"(} launch(%class_extent: index) {
  template.apply<@request.schedule>(%class_extent) : (index)
)");
      } else {
        source.append("} launch() {\n");
      }
      source.append(R"(
  kernel.return
}

)");
    }

    source.append(R"(command.program.def public @root() launch() {
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %cN = index.constant )");
    source.append(std::to_string(launch_count));
    source.append(" : index\n");
    if (launch_shape == LaunchShape::kRepeatedWorkload ||
        launch_shape == LaunchShape::kRepeatedKernelClass) {
      source.append("  %extent = index.constant 1024 : index\n");
    }
    source.append("  scf.for %iteration = [%c0 to %cN step %c1] unroll {\n");
    switch (launch_shape) {
      case LaunchShape::kDirect:
        source.append("    %count = index.add %iteration, %c1 : index\n");
        source.append("    kernel.dispatch @entry[%count]() : [index]()\n");
        break;
      case LaunchShape::kRepeatedWorkload:
        source.append("    kernel.launch @tiles[%extent]() : [index]()\n");
        break;
      case LaunchShape::kDistinctWorkloads:
        source.append("    kernel.launch @tiles[%iteration]() : [index]()\n");
        break;
      case LaunchShape::kRepeatedKernelClass:
        source.append(
            "    kernel.launch @tiles[%extent](%extent) : [index](index)\n");
        break;
      case LaunchShape::kDistinctKernelClasses:
        source.append(
            "    kernel.launch @tiles[%iteration](%iteration) : "
            "[index](index)\n");
        break;
    }
    source.append(R"(  }
  command.return
}
)");
    return source;
  }

  loom_module_t* ParseAndVerify(const std::string& source) {
    loom_text_parse_options_t parse_options = {};
    parse_options.diagnostic_sink = {loom_diagnostic_stderr_sink, nullptr};
    parse_options.max_errors = 20;
    loom_module_t* module = nullptr;
    CheckStatus(
        loom_text_parse(iree_make_string_view(source.data(), source.size()),
                        IREE_SV("command_plan_benchmark.loom"), &context_,
                        &block_pool_, &parse_options, &module));
    if (module == nullptr) std::abort();

    loom_verify_options_t verify_options = {};
    verify_options.sink.fn = loom_diagnostic_stderr_sink;
    verify_options.max_errors = 20;
    loom_verify_result_t verify_result = {};
    CheckStatus(loom_verify_module(module, &verify_options, &verify_result));
    if (verify_result.error_count != 0) std::abort();
    return module;
  }

  // Number of launch or dispatch sites produced by source unrolling.
  iree_host_size_t launch_count_;
  // Shared storage pool for source, materialized, and prepared modules.
  iree_arena_block_pool_t block_pool_ = {};
  // Frozen dialect context shared by all fixture modules.
  loom_context_t context_ = {};
  // Parsed provider module retained by the materialized module index.
  loom_module_t* source_module_ = nullptr;
  // Reusable index over the source provider.
  loom_link_module_index_t* index_ = nullptr;
  // Indexed public command root selected by the reusable plan.
  const loom_link_module_index_symbol_t* root_symbol_ = nullptr;
  // Selective command implementation plan materialized outside timed regions.
  loom_link_plan_t* plan_ = nullptr;
};

static void RunProgramPlanBenchmark(benchmark::State& state,
                                    LaunchShape launch_shape) {
  ProgramPlanFixture fixture(launch_shape,
                             static_cast<iree_host_size_t>(state.range(0)));
  for (auto _ : state) {
    // Selective materialization supplies the real consumed input boundary but
    // is independently benchmarked by the linker. Time only command planning.
    state.PauseTiming();
    iree_arena_allocator_t materialization_arena;
    iree_arena_initialize(fixture.block_pool(), &materialization_arena);
    loom_symbol_ref_t root_ref = loom_symbol_ref_null();
    loom_link_plan_materialization_t materialization =
        fixture.Materialize(&materialization_arena, &root_ref);
    state.ResumeTiming();

    bool valid = false;
    loom_cmd_program_plan_t program_plan = {};
    CheckStatus(loom_cmd_program_plan_prepare_materialization(
        &materialization, &root_ref, /*program_count=*/1,
        /*kernel_source=*/nullptr, loom_pass_builtin_registry(),
        /*diagnostic_emitter=*/{}, fixture.block_pool(), &valid, &program_plan,
        iree_allocator_system()));
    if (!valid) std::abort();
    benchmark::DoNotOptimize(program_plan.root_count);

    state.PauseTiming();
    loom_cmd_program_plan_deinitialize(&program_plan);
    iree_arena_deinitialize(&materialization_arena);
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() * fixture.launch_count());
  state.SetComplexityN(fixture.launch_count());
}

static void BM_PrepareDirectDispatches(benchmark::State& state) {
  RunProgramPlanBenchmark(state, LaunchShape::kDirect);
}

static void BM_PrepareRepeatedWorkload(benchmark::State& state) {
  RunProgramPlanBenchmark(state, LaunchShape::kRepeatedWorkload);
}

static void BM_PrepareDistinctWorkloads(benchmark::State& state) {
  RunProgramPlanBenchmark(state, LaunchShape::kDistinctWorkloads);
}

struct KernelRequestCapture {
  // Products transferred by the current command-plan preparation.
  std::vector<loom_cmd_program_kernel_request_t> requests;

  // Timestamp immediately before the current preparation began.
  std::chrono::steady_clock::time_point start_time;

  // Time from preparation entry to the first transferred request.
  std::chrono::nanoseconds first_request_time = {};
};

static iree_status_t CaptureKernelRequest(
    void* user_data, loom_cmd_program_kernel_request_t request) {
  KernelRequestCapture* capture = static_cast<KernelRequestCapture*>(user_data);
  if (capture->requests.empty()) {
    capture->first_request_time =
        std::chrono::steady_clock::now() - capture->start_time;
  }
  capture->requests.push_back(request);
  return iree_ok_status();
}

static void RunIndexedProgramPlanBenchmark(benchmark::State& state,
                                           LaunchShape launch_shape,
                                           KernelRequestMode request_mode) {
  ProgramPlanFixture fixture(launch_shape,
                             static_cast<iree_host_size_t>(state.range(0)));
  const iree_host_size_t root_symbol_ordinal = fixture.root_symbol_ordinal();
  const loom_link_plan_materialization_environment_t environment = {
      /*.context=*/fixture.context(),
      /*.block_pool=*/fixture.block_pool(),
      /*.low_repr_environment=*/{},
      /*.diagnostic_sink=*/nullptr,
      /*.prepare_module=*/nullptr,
      /*.user_data=*/nullptr,
      /*.allocator=*/iree_allocator_system(),
  };
  loom_cmd_program_plan_index_options_t options;
  loom_cmd_program_plan_index_options_initialize(&options);
  KernelRequestCapture capture;
  capture.requests.reserve(options.kernel_class_collection.class_limit);
  if (request_mode == KernelRequestMode::kPublish) {
    options.kernel_request_sink = {
        /*.publish=*/CaptureKernelRequest,
        /*.user_data=*/&capture,
    };
  }

  int64_t first_request_nanoseconds = 0;
  iree_host_size_t request_count = 0;
  for (auto _ : state) {
    state.PauseTiming();
    iree_arena_allocator_t scratch_arena;
    iree_arena_initialize(fixture.block_pool(), &scratch_arena);
    capture.requests.clear();
    capture.first_request_time = {};
    state.ResumeTiming();

    capture.start_time = std::chrono::steady_clock::now();
    bool valid = false;
    loom_cmd_program_plan_t program_plan = {};
    CheckStatus(loom_cmd_program_plan_prepare_index(
        fixture.index(), &root_symbol_ordinal, /*program_count=*/1,
        request_mode == KernelRequestMode::kPublish ? &options : nullptr,
        loom_pass_builtin_registry(), /*diagnostic_emitter=*/{}, &environment,
        &scratch_arena, &valid, &program_plan));
    if (!valid) std::abort();
    benchmark::DoNotOptimize(program_plan.root_count);

    request_count = capture.requests.size();
    if (request_count != 0) {
      first_request_nanoseconds += capture.first_request_time.count();
    }
    state.PauseTiming();
    for (loom_cmd_program_kernel_request_t& request : capture.requests) {
      loom_kernel_class_product_deinitialize(&request.source.product);
    }
    loom_cmd_program_plan_deinitialize(&program_plan);
    iree_arena_deinitialize(&scratch_arena);
    state.ResumeTiming();
  }
  state.SetItemsProcessed(state.iterations() * fixture.launch_count());
  state.SetComplexityN(fixture.launch_count());
  state.counters["requests"] = static_cast<double>(request_count);
  if (first_request_nanoseconds != 0) {
    state.counters["first_request_ns"] =
        static_cast<double>(first_request_nanoseconds) / state.iterations();
  }
}

static void BM_PrepareIndexedExternalBodyBlind(benchmark::State& state) {
  RunIndexedProgramPlanBenchmark(state, LaunchShape::kDirect,
                                 KernelRequestMode::kBodyBlind);
}

static void BM_PrepareIndexedExternalRequestSink(benchmark::State& state) {
  RunIndexedProgramPlanBenchmark(state, LaunchShape::kDirect,
                                 KernelRequestMode::kPublish);
}

static void BM_PrepareIndexedRepeatedClassBodyBlind(benchmark::State& state) {
  RunIndexedProgramPlanBenchmark(state, LaunchShape::kRepeatedKernelClass,
                                 KernelRequestMode::kBodyBlind);
}

static void BM_PrepareIndexedRepeatedClassRequests(benchmark::State& state) {
  RunIndexedProgramPlanBenchmark(state, LaunchShape::kRepeatedKernelClass,
                                 KernelRequestMode::kPublish);
}

static void BM_PrepareIndexedDistinctClassBodyBlind(benchmark::State& state) {
  RunIndexedProgramPlanBenchmark(state, LaunchShape::kDistinctKernelClasses,
                                 KernelRequestMode::kBodyBlind);
}

static void BM_PrepareIndexedDistinctClassRequests(benchmark::State& state) {
  RunIndexedProgramPlanBenchmark(state, LaunchShape::kDistinctKernelClasses,
                                 KernelRequestMode::kPublish);
}

static void LaunchScales(benchmark::Benchmark* benchmark) {
  benchmark->Arg(1)->Arg(4)->Arg(16)->Arg(64)->Arg(256)->Arg(1024);
}

static void RequestScales(benchmark::Benchmark* benchmark) {
  benchmark->Arg(1)->Arg(16)->Arg(256)->Arg(1024);
}

BENCHMARK(BM_PrepareDirectDispatches)
    ->Apply(LaunchScales)
    ->Complexity(benchmark::oN);
BENCHMARK(BM_PrepareRepeatedWorkload)
    ->Apply(LaunchScales)
    ->Complexity(benchmark::oN);
BENCHMARK(BM_PrepareDistinctWorkloads)
    ->Apply(LaunchScales)
    ->Complexity(benchmark::oN);
BENCHMARK(BM_PrepareIndexedExternalBodyBlind)
    ->Apply(RequestScales)
    ->Complexity(benchmark::oN);
BENCHMARK(BM_PrepareIndexedExternalRequestSink)
    ->Apply(RequestScales)
    ->Complexity(benchmark::oN);
BENCHMARK(BM_PrepareIndexedRepeatedClassBodyBlind)
    ->Apply(RequestScales)
    ->Complexity(benchmark::oN);
BENCHMARK(BM_PrepareIndexedRepeatedClassRequests)
    ->Apply(RequestScales)
    ->Complexity(benchmark::oN);
BENCHMARK(BM_PrepareIndexedDistinctClassBodyBlind)
    ->Apply(RequestScales)
    ->Complexity(benchmark::oN);
BENCHMARK(BM_PrepareIndexedDistinctClassRequests)
    ->Apply(RequestScales)
    ->Complexity(benchmark::oN);

}  // namespace

BENCHMARK_MAIN();
