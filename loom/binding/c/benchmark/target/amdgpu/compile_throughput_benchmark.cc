// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/binding/c/benchmark/compile_throughput_benchmark.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "benchmark/benchmark.h"
#include "loom/binding/c/benchmark/workload_compile_benchmark.h"
#include "loomc/target/amdgpu.h"

namespace {

using loomc::bench::CloneModule;
using loomc::bench::CompileScenario;
using loomc::bench::CreateBenchmarkKernelSource;
using loomc::bench::CreateTextModule;
using loomc::bench::CreateTextSource;
using loomc::bench::CreateWorkspace;
using loomc::bench::DeserializeSource;
using loomc::bench::loom_allocator;
using loomc::bench::ModulePtr;
using loomc::bench::PassProgramPtr;
using loomc::bench::PreparePassProgram;
using loomc::bench::ReadArtifactPrefix;
using loomc::bench::RegisterInputScalingCompileBenchmarks;
using loomc::bench::RegisterQwenAttentionCompileBenchmarks;
using loomc::bench::RequireSucceededResult;
using loomc::bench::ResultPtr;
using loomc::bench::RunCompileBenchmarkDirect;
using loomc::bench::RunCompileBenchmarkDirectCold;
using loomc::bench::SourcePtr;
using loomc::bench::TargetCompileScenario;
using loomc::bench::TargetEnvironmentPtr;
using loomc::bench::TargetProfilePtr;
using loomc::bench::to_iree_status;
using loomc::bench::ValidateArtifact;
using loomc::bench::WorkloadCompileTarget;
using loomc::bench::WorkspacePtr;

struct AmdgpuBenchmarkTarget {
  // Stable target/profile name included in benchmark result names.
  const char* benchmark_name;

  // Processor key resolved by the production AMDGPU profile table.
  const char* processor;
};

constexpr AmdgpuBenchmarkTarget kGfx1100Target = {"amdgpu-gfx1100", "gfx1100"};
constexpr AmdgpuBenchmarkTarget kGfx942Target = {"amdgpu-gfx942", "gfx942"};
constexpr AmdgpuBenchmarkTarget kGfx1200Target = {"amdgpu-gfx1200", "gfx1200"};
constexpr AmdgpuBenchmarkTarget kGfx1250Target = {"amdgpu-gfx1250", "gfx1250"};

static iree_status_t CreateAmdgpuBenchmarkTarget(
    AmdgpuBenchmarkTarget target, TargetEnvironmentPtr* out_target_environment,
    TargetProfilePtr* out_target_profile) {
  loomc_target_environment_t* raw_target_environment = nullptr;
  IREE_RETURN_IF_ERROR(to_iree_status(loomc_target_environment_create_amdgpu(
      loom_allocator(), &raw_target_environment)));
  TargetEnvironmentPtr target_environment(raw_target_environment);

  const loomc_string_view_t processor =
      loomc_make_cstring_view(target.processor);
  const loomc_amdgpu_profile_options_t profile_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS,
      /*.structure_size=*/sizeof(profile_options),
      /*.next=*/nullptr,
      /*.identifier=*/processor,
      /*.identity=*/
      {
          /*.processor=*/processor,
      },
  };
  loomc_target_profile_t* raw_profile = nullptr;
  IREE_RETURN_IF_ERROR(to_iree_status(loomc_target_profile_create_amdgpu(
      target_environment.get(), &profile_options, loom_allocator(),
      &raw_profile)));
  out_target_environment->reset(target_environment.release());
  out_target_profile->reset(raw_profile);
  return iree_ok_status();
}

static iree_status_t EmitAmdgpuBenchmarkArtifact(
    loomc_target_environment_t* target_environment,
    loomc_workspace_t* workspace, loomc_module_t* module,
    loomc_string_view_t identifier, int64_t* out_artifact_byte_count) {
  const loomc_amdgpu_emit_options_t amdgpu_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_AMDGPU_EMIT_OPTIONS,
      /*.structure_size=*/sizeof(amdgpu_options),
      /*.next=*/nullptr,
      /*.runtime_globals=*/LOOMC_AMDGPU_RUNTIME_GLOBAL_NONE,
  };
  const loomc_emit_options_t emit_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      /*.structure_size=*/sizeof(emit_options),
      /*.next=*/&amdgpu_options,
      /*.artifact_format=*/
      loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO),
      /*.identifier=*/identifier,
      /*.artifact_flags=*/LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
  };

  loomc_result_t* raw_result = nullptr;
  iree_status_t status = to_iree_status(
      loomc_emit_module(target_environment, workspace, module, &emit_options,
                        loom_allocator(), &raw_result));
  ResultPtr result(raw_result);
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(RequireSucceededResult(result.get(), "AMDGPU emission"));

  constexpr uint8_t kElfMagic[] = {0x7F, 'E', 'L', 'F'};
  IREE_RETURN_IF_ERROR(ValidateArtifact(
      result.get(), loomc_make_cstring_view(LOOMC_ARTIFACT_ROLE_KERNEL),
      loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO),
      sizeof(kElfMagic), "AMDGPU HSACO executable", out_artifact_byte_count));
  const loomc_artifact_t* artifact = loomc::bench::FindArtifact(
      result.get(), loomc_make_cstring_view(LOOMC_ARTIFACT_ROLE_KERNEL),
      loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO));
  uint8_t magic[sizeof(kElfMagic)] = {0};
  IREE_RETURN_IF_ERROR(
      ReadArtifactPrefix(artifact, iree_make_byte_span(magic, sizeof(magic))));
  if (std::memcmp(magic, kElfMagic, sizeof(kElfMagic)) != 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU executable is not an ELF image");
  }
  return iree_ok_status();
}

class AmdgpuTargetCompileScenario : public TargetCompileScenario {
 public:
  explicit AmdgpuTargetCompileScenario(
      AmdgpuBenchmarkTarget target, iree_host_size_t workspace_block_size = 0)
      : TargetCompileScenario(workspace_block_size), target_(target) {}

 protected:
  iree_status_t SetUpAmdgpuTarget(iree_host_size_t worker_count) {
    TargetEnvironmentPtr target_environment;
    TargetProfilePtr target_profile;
    IREE_RETURN_IF_ERROR(CreateAmdgpuBenchmarkTarget(
        target_, &target_environment, &target_profile));

    return SetUpTarget(
        worker_count, std::move(target_environment), std::move(target_profile),
        loomc_make_cstring_view("benchmark-amdgpu-prepared-low"));
  }

  iree_status_t EmitAmdgpuArtifact(WorkspacePtr& workspace, ModulePtr& module,
                                   loomc_string_view_t identifier) {
    int64_t artifact_bytes = 0;
    IREE_RETURN_IF_ERROR(
        EmitAmdgpuBenchmarkArtifact(target_environment(), workspace.get(),
                                    module.get(), identifier, &artifact_bytes));
    RecordArtifactBytes(artifact_bytes);
    return iree_ok_status();
  }

 private:
  // Immutable target row selected by the benchmark registration.
  AmdgpuBenchmarkTarget target_;
};

class AmdgpuWorkloadCompileTarget final : public WorkloadCompileTarget {
 public:
  explicit AmdgpuWorkloadCompileTarget(AmdgpuBenchmarkTarget target)
      : target_(target) {}

  const char* benchmark_name() const override { return target_.benchmark_name; }

  loomc_string_view_t pipeline_identifier() const override {
    return loomc_make_cstring_view("benchmark-amdgpu-prepared-low");
  }

  iree_status_t CreateTarget(
      TargetEnvironmentPtr* out_target_environment,
      TargetProfilePtr* out_target_profile) const override {
    return CreateAmdgpuBenchmarkTarget(target_, out_target_environment,
                                       out_target_profile);
  }

  iree_status_t EmitArtifact(loomc_target_environment_t* target_environment,
                             loomc_workspace_t* workspace,
                             loomc_module_t* module,
                             loomc_string_view_t identifier,
                             int64_t* out_artifact_byte_count) const override {
    return EmitAmdgpuBenchmarkArtifact(target_environment, workspace, module,
                                       identifier, out_artifact_byte_count);
  }

 private:
  // Exact AMDGPU target selected for every workload benchmark.
  AmdgpuBenchmarkTarget target_;
};

const AmdgpuWorkloadCompileTarget kAmdgpuQwenTarget(kGfx1100Target);

[[maybe_unused]] const bool kAmdgpuQwenBenchmarksRegistered = [] {
  RegisterQwenAttentionCompileBenchmarks(
      kAmdgpuQwenTarget,
      {
          /*.source_identifier=*/"qwen38_attention_prefill_wmma.loom",
          /*.function_symbol=*/"qwen38_attention_prefill_f16_wmma",
          /*.artifact_identifier=*/"qwen_attention_benchmark.hsaco",
      });
  RegisterInputScalingCompileBenchmarks(
      kAmdgpuQwenTarget, "QwenMoeQ4K",
      {
          /*.source_identifier=*/"qwen3_moe_routed_gate_up_q4k.loom",
          /*.function_symbol=*/
          "qwen3_moe_routed_gate_up_swiglu_q4k_q8",
          /*.artifact_identifier=*/"qwen_moe_benchmark.hsaco",
          /*.input_size_config_symbol=*/
          "qwen3_moe.routed_gate_up.input_size",
      });
  RegisterInputScalingCompileBenchmarks(
      kAmdgpuQwenTarget, "QwenFfn",
      {
          /*.source_identifier=*/"qwen3_ffn_gate_up_f32_amdgpu.loom",
          /*.function_symbol=*/"qwen3_ffn_gate_up_quadratic_f32",
          /*.artifact_identifier=*/"qwen_ffn_benchmark.hsaco",
          /*.input_size_config_symbol=*/"qwen3_ffn.input_size",
      });
  return true;
}();

class AmdgpuI32ChainScenario final : public AmdgpuTargetCompileScenario {
 public:
  AmdgpuI32ChainScenario(
      iree_host_size_t job_count,
      std::initializer_list<iree_host_size_t> operation_counts,
      AmdgpuBenchmarkTarget target, iree_host_size_t workspace_block_size = 0)
      : AmdgpuTargetCompileScenario(target, workspace_block_size),
        job_count_(std::max<iree_host_size_t>(job_count, 1)) {
    operation_counts_.reserve(operation_counts.size());
    for (iree_host_size_t operation_count : operation_counts) {
      operation_count = std::max<iree_host_size_t>(operation_count, 1);
      operation_counts_.push_back({operation_count, {}});
    }
  }

  iree_status_t SetUp(iree_host_size_t worker_count) override {
    IREE_RETURN_IF_ERROR(SetUpAmdgpuTarget(worker_count));
    IREE_RETURN_IF_ERROR(CreateBenchmarkKernelSource(
        loomc_make_cstring_view("i32_memory_chain.loom"), &source_));
    IREE_RETURN_IF_ERROR(
        CreateWorkspace(/*block_size=*/0, &template_workspace_));
    IREE_RETURN_IF_ERROR(DeserializeSource(context_.get(),
                                           template_workspace_.get(),
                                           source_.get(), &template_module_));
    for (OperationCount& operation_count : operation_counts_) {
      std::ostringstream config_text;
      config_text << "config.def @benchmark.workgroup_size = 64 : index\n"
                  << "config.def @benchmark.operation_count = "
                  << operation_count.value << " : index\n";
      IREE_RETURN_IF_ERROR(CreateTextModule(
          context_.get(), template_workspace_.get(), "i32_chain_config.loom",
          config_text.str(), &operation_count.config_module));
    }
    return iree_ok_status();
  }

  iree_host_size_t job_count() const override { return job_count_; }

  iree_status_t WarmUp(iree_host_size_t worker_count) override {
    for (iree_host_size_t worker_ordinal = 0; worker_ordinal < worker_count;
         ++worker_ordinal) {
      for (iree_host_size_t pattern_ordinal = 0;
           pattern_ordinal < operation_counts_.size(); ++pattern_ordinal) {
        IREE_RETURN_IF_ERROR(RunJob(worker_ordinal, pattern_ordinal));
      }
    }
    return iree_ok_status();
  }

  iree_status_t RunJob(iree_host_size_t worker_ordinal,
                       iree_host_size_t job_ordinal) override {
    WorkspacePtr& workspace = workspace_at(worker_ordinal);
    const OperationCount& operation_count =
        operation_counts_[job_ordinal % operation_counts_.size()];

    ModulePtr module;
    IREE_RETURN_IF_ERROR(
        CloneModule(template_module_.get(), workspace.get(), &module));
    IREE_RETURN_IF_ERROR(CompileModuleToPreparedLow(
        workspace, module, loomc_make_cstring_view("i32_memory_chain"),
        loomc_make_cstring_view("amdgpu_i32_chain"),
        operation_count.config_module.get(),
        LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED));
    return EmitAmdgpuArtifact(
        workspace, module, loomc_make_cstring_view("i32_memory_chain.hsaco"));
  }

  void SetExtraCounters(::benchmark::State& state) const override {
    iree_host_size_t minimum = operation_counts_.front().value;
    iree_host_size_t maximum = minimum;
    for (const OperationCount& operation_count : operation_counts_) {
      minimum = std::min(minimum, operation_count.value);
      maximum = std::max(maximum, operation_count.value);
    }
    state.counters["operation_count_min"] = (double)minimum;
    state.counters["operation_count_max"] = (double)maximum;
  }

 private:
  struct OperationCount {
    // Numeric operation count reported by the benchmark.
    iree_host_size_t value;

    // Immutable typed config module shared by compile invocations.
    ModulePtr config_module;
  };

  // Number of kernels compiled by each benchmark iteration.
  iree_host_size_t job_count_ = 0;

  // Cyclic sequence of arithmetic-chain sizes compiled by each iteration.
  std::vector<OperationCount> operation_counts_;

  // Compact targetless source shared by all invocations.
  SourcePtr source_;

  // Setup-only workspace retaining the parsed template module.
  WorkspacePtr template_workspace_;

  // Immutable parsed template cloned into worker workspaces.
  ModulePtr template_module_;
};

static std::string BuildAmdgpuClusterAsyncDisjointSource(
    iree_host_size_t transfer_count) {
  constexpr uint64_t kPacketByteCount = 16;
  constexpr uint64_t kWorkgroupSize = 64;
  const uint64_t lane_stride = transfer_count * kPacketByteCount;
  const uint64_t storage_byte_count = kWorkgroupSize * lane_stride;

  std::ostringstream source;
  source << R"(
kernel.def export("cluster_async_disjoint") @cluster_async_disjoint() {
  %one = index.constant 1 : index
  %two = index.constant 2 : index
  %workgroup_size = index.constant 64 : index
  kernel.launch.config workgroups(%one, %two, %one) workgroup_size(%workgroup_size, %one, %one) cluster_size(%one, %two, %one) : index
} launch(%input: buffer) {
  %zero = index.constant 0 : offset
  %lane_stride = index.constant )"
         << lane_stride << R"( : index
  %storage_bytes = index.constant )"
         << storage_byte_count << R"( : offset
  %participants = scalar.constant 3 : i32
  %lane = kernel.workitem.id<x> : index
  %lane_base = index.mul %lane, %lane_stride : index
  %global = buffer.assume.memory_space<global> %input : buffer
  %source = buffer.view %global[%zero] : buffer -> view<16xi8>
  %scratch = buffer.alloca<workgroup> align(16) %storage_bytes : buffer
)";

  for (iree_host_size_t i = 0; i < transfer_count; ++i) {
    source << "  %slot" << i << " = index.constant " << i * kPacketByteCount
           << " : index\n"
           << "  %dest_index" << i << " = index.add %lane_base, %slot" << i
           << " : index\n"
           << "  %dest_offset" << i << " = index.cast %dest_index" << i
           << " : index to offset\n"
           << "  %dest" << i << " = buffer.view %scratch[%dest_offset" << i
           << "] : buffer -> view<16xi8>\n";
  }
  for (iree_host_size_t i = 0; i < transfer_count; ++i) {
    source << "  %copy" << i
           << " = kernel.async.cluster.gather %source to %dest" << i
           << " using %participants {cache_scope = device, cache_temporal = "
              "regular} : view<16xi8> to view<16xi8>, i32 -> "
              "kernel.async.token\n";
  }

  source << "  %group = kernel.async.group ";
  for (iree_host_size_t i = 0; i < transfer_count; ++i) {
    if (i != 0) source << ", ";
    source << "%copy" << i;
  }
  source << " : ";
  for (iree_host_size_t i = 0; i < transfer_count; ++i) {
    if (i != 0) source << ", ";
    source << "kernel.async.token";
  }
  source << R"( -> kernel.async.group
  kernel.async.wait %group {newer_groups = 0} : kernel.async.group
  kernel.return
}
)";
  return source.str();
}

class AmdgpuClusterAsyncDisjointScenario final
    : public AmdgpuTargetCompileScenario {
 public:
  AmdgpuClusterAsyncDisjointScenario(iree_host_size_t job_count,
                                     iree_host_size_t transfer_count,
                                     AmdgpuBenchmarkTarget target)
      : AmdgpuTargetCompileScenario(target),
        job_count_(std::max<iree_host_size_t>(job_count, 1)),
        transfer_count_(std::max<iree_host_size_t>(transfer_count, 1)) {}

  iree_status_t SetUp(iree_host_size_t worker_count) override {
    IREE_RETURN_IF_ERROR(SetUpAmdgpuTarget(worker_count));
    IREE_RETURN_IF_ERROR(CreateTextSource(
        "cluster_async_disjoint.loom",
        BuildAmdgpuClusterAsyncDisjointSource(transfer_count_), &source_));
    IREE_RETURN_IF_ERROR(
        CreateWorkspace(/*block_size=*/0, &template_workspace_));
    return DeserializeSource(context_.get(), template_workspace_.get(),
                             source_.get(), &template_module_);
  }

  iree_host_size_t job_count() const override { return job_count_; }

  iree_status_t RunJob(iree_host_size_t worker_ordinal,
                       iree_host_size_t job_ordinal) override {
    (void)job_ordinal;
    WorkspacePtr& workspace = workspace_at(worker_ordinal);
    ModulePtr module;
    IREE_RETURN_IF_ERROR(
        CloneModule(template_module_.get(), workspace.get(), &module));
    IREE_RETURN_IF_ERROR(CompileModuleToPreparedLow(
        workspace, module, loomc_make_cstring_view("cluster_async_disjoint"),
        loomc_make_cstring_view("amdgpu_cluster_async_disjoint"),
        /*config_module=*/nullptr, /*config_flags=*/0));
    return EmitAmdgpuArtifact(
        workspace, module,
        loomc_make_cstring_view("cluster_async_disjoint.hsaco"));
  }

  void SetExtraCounters(::benchmark::State& state) const override {
    state.counters["transfer_count"] = (double)transfer_count_;
  }

 private:
  // Number of kernels compiled by each benchmark iteration.
  iree_host_size_t job_count_ = 0;

  // Number of pairwise-disjoint cluster transfers in each kernel.
  iree_host_size_t transfer_count_ = 0;

  // Generated targetless source shared by all invocations.
  SourcePtr source_;

  // Setup-only workspace retaining the parsed template module.
  WorkspacePtr template_workspace_;

  // Immutable parsed template cloned into worker workspaces.
  ModulePtr template_module_;
};

static std::unique_ptr<CompileScenario> CreateAmdgpuI32ChainScenario(
    const ::benchmark::State& state, const void* user_data) {
  const auto* target = static_cast<const AmdgpuBenchmarkTarget*>(user_data);
  return std::make_unique<AmdgpuI32ChainScenario>(
      (iree_host_size_t)state.range(1),
      std::initializer_list<iree_host_size_t>{(iree_host_size_t)state.range(2)},
      *target);
}

static std::unique_ptr<CompileScenario> CreateAmdgpuI32AlternatingChainScenario(
    const ::benchmark::State& state, const void* user_data) {
  const auto* target = static_cast<const AmdgpuBenchmarkTarget*>(user_data);
  return std::make_unique<AmdgpuI32ChainScenario>(
      (iree_host_size_t)state.range(1),
      std::initializer_list<iree_host_size_t>{(iree_host_size_t)state.range(2),
                                              (iree_host_size_t)state.range(3)},
      *target);
}

static std::unique_ptr<CompileScenario> CreateAmdgpuI32ChainWorkspaceScenario(
    const ::benchmark::State& state, const void* user_data) {
  const auto* target = static_cast<const AmdgpuBenchmarkTarget*>(user_data);
  return std::make_unique<AmdgpuI32ChainScenario>(
      (iree_host_size_t)state.range(1),
      std::initializer_list<iree_host_size_t>{(iree_host_size_t)state.range(2)},
      *target, (iree_host_size_t)state.range(3));
}

static std::unique_ptr<CompileScenario>
CreateAmdgpuClusterAsyncDisjointScenario(const ::benchmark::State& state,
                                         const void* user_data) {
  const auto* target = static_cast<const AmdgpuBenchmarkTarget*>(user_data);
  return std::make_unique<AmdgpuClusterAsyncDisjointScenario>(
      (iree_host_size_t)state.range(1), (iree_host_size_t)state.range(2),
      *target);
}

static void BM_AmdgpuI32ChainSmoke(::benchmark::State& state,
                                   const AmdgpuBenchmarkTarget* target) {
  RunCompileBenchmarkDirect(state, CreateAmdgpuI32ChainScenario, target);
}
BENCHMARK_CAPTURE(BM_AmdgpuI32ChainSmoke, Gfx1100, &kGfx1100Target)
    ->Args({1, 1, 16})
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_AmdgpuI32ChainSmoke, Gfx942, &kGfx942Target)
    ->Args({1, 1, 16})
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_AmdgpuI32ChainSmoke, Gfx1200, &kGfx1200Target)
    ->Args({1, 1, 16})
    ->UseRealTime();

static void BM_AmdgpuClusterAsyncDisjointSmoke(
    ::benchmark::State& state, const AmdgpuBenchmarkTarget* target) {
  RunCompileBenchmarkDirect(state, CreateAmdgpuClusterAsyncDisjointScenario,
                            target);
}
BENCHMARK_CAPTURE(BM_AmdgpuClusterAsyncDisjointSmoke, Gfx1250, &kGfx1250Target)
    ->Args({1, 1, 1})
    ->Args({1, 1, 4})
    ->UseRealTime();

static void BM_AmdgpuI32ChainCold(::benchmark::State& state,
                                  const AmdgpuBenchmarkTarget* target) {
  RunCompileBenchmarkDirectCold(state, CreateAmdgpuI32ChainScenario, target);
}
BENCHMARK_CAPTURE(BM_AmdgpuI32ChainCold, Gfx1100, &kGfx1100Target)
    ->Args({1, 1, 16})
    ->Args({1, 1, 1024})
    ->Iterations(1)
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_AmdgpuI32ChainCold, Gfx942, &kGfx942Target)
    ->Args({1, 1, 16})
    ->Args({1, 1, 1024})
    ->Iterations(1)
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_AmdgpuI32ChainCold, Gfx1200, &kGfx1200Target)
    ->Args({1, 1, 16})
    ->Args({1, 1, 1024})
    ->Iterations(1)
    ->UseRealTime();

static void BM_AmdgpuI32ChainWorkspaceCold(
    ::benchmark::State& state, const AmdgpuBenchmarkTarget* target) {
  RunCompileBenchmarkDirectCold(state, CreateAmdgpuI32ChainWorkspaceScenario,
                                target);
}

static void BM_AmdgpuI32ChainWorkspace(::benchmark::State& state,
                                       const AmdgpuBenchmarkTarget* target) {
  RunCompileBenchmarkDirect(state, CreateAmdgpuI32ChainWorkspaceScenario,
                            target);
}

#define LOOM_BENCHMARK_AMDGPU_I32_CHAIN_WORKSPACE(target_name, target)      \
  BENCHMARK_CAPTURE(BM_AmdgpuI32ChainWorkspaceCold, target_name, target)    \
      ->ArgsProduct({{1}, {1}, {1024}, {32 * 1024, 64 * 1024, 128 * 1024}}) \
      ->Iterations(1)                                                       \
      ->UseRealTime();                                                      \
  BENCHMARK_CAPTURE(BM_AmdgpuI32ChainWorkspace, target_name, target)        \
      ->ArgsProduct({{1}, {1}, {1024}, {32 * 1024, 64 * 1024, 128 * 1024}}) \
      ->UseRealTime()

LOOM_BENCHMARK_AMDGPU_I32_CHAIN_WORKSPACE(Gfx1100, &kGfx1100Target);
LOOM_BENCHMARK_AMDGPU_I32_CHAIN_WORKSPACE(Gfx942, &kGfx942Target);
LOOM_BENCHMARK_AMDGPU_I32_CHAIN_WORKSPACE(Gfx1200, &kGfx1200Target);

#undef LOOM_BENCHMARK_AMDGPU_I32_CHAIN_WORKSPACE

static void BM_AmdgpuI32ChainAlternating(::benchmark::State& state,
                                         const AmdgpuBenchmarkTarget* target) {
  RunCompileBenchmarkDirect(state, CreateAmdgpuI32AlternatingChainScenario,
                            target);
}
BENCHMARK_CAPTURE(BM_AmdgpuI32ChainAlternating, Gfx1100, &kGfx1100Target)
    ->Args({1, 2, 16, 1024})
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_AmdgpuI32ChainAlternating, Gfx942, &kGfx942Target)
    ->Args({1, 2, 16, 1024})
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_AmdgpuI32ChainAlternating, Gfx1200, &kGfx1200Target)
    ->Args({1, 2, 16, 1024})
    ->UseRealTime();

static void BM_AmdgpuI32Chain(::benchmark::State& state,
                              const AmdgpuBenchmarkTarget* target) {
  RunCompileBenchmarkDirect(state, CreateAmdgpuI32ChainScenario, target);
}
#define LOOM_BENCHMARK_AMDGPU_I32_CHAIN(target_name, target) \
  BENCHMARK_CAPTURE(BM_AmdgpuI32Chain, target_name, target)  \
      ->Args({1, 16, 1})                                     \
      ->Args({1, 16, 64})                                    \
      ->Args({1, 4, 1024})                                   \
      ->UseRealTime()

LOOM_BENCHMARK_AMDGPU_I32_CHAIN(Gfx1100, &kGfx1100Target);
LOOM_BENCHMARK_AMDGPU_I32_CHAIN(Gfx942, &kGfx942Target);
LOOM_BENCHMARK_AMDGPU_I32_CHAIN(Gfx1200, &kGfx1200Target);

#undef LOOM_BENCHMARK_AMDGPU_I32_CHAIN

static void BM_AmdgpuClusterAsyncDisjointCold(
    ::benchmark::State& state, const AmdgpuBenchmarkTarget* target) {
  RunCompileBenchmarkDirectCold(state, CreateAmdgpuClusterAsyncDisjointScenario,
                                target);
}
BENCHMARK_CAPTURE(BM_AmdgpuClusterAsyncDisjointCold, Gfx1250, &kGfx1250Target)
    ->Args({1, 1, 1})
    ->Args({1, 1, 4})
    ->Args({1, 1, 64})
    ->Iterations(1)
    ->UseRealTime();

static void BM_AmdgpuClusterAsyncDisjoint(::benchmark::State& state,
                                          const AmdgpuBenchmarkTarget* target) {
  RunCompileBenchmarkDirect(state, CreateAmdgpuClusterAsyncDisjointScenario,
                            target);
}
BENCHMARK_CAPTURE(BM_AmdgpuClusterAsyncDisjoint, Gfx1250, &kGfx1250Target)
    ->Args({1, 1, 1})
    ->Args({1, 1, 4})
    ->Args({1, 1, 16})
    ->Args({1, 1, 64})
    ->UseRealTime();

}  // namespace
