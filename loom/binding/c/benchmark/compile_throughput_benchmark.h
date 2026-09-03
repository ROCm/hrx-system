// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_BENCHMARK_COMPILE_THROUGHPUT_BENCHMARK_H_
#define LOOMC_BENCHMARK_COMPILE_THROUGHPUT_BENCHMARK_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "iree/base/api.h"
#include "loom/binding/c/benchmark/util/benchmark_support.h"
#include "loomc/iree.h"

namespace benchmark {
class State;
}  // namespace benchmark

namespace loomc::bench {

struct WorkerSlot {
  // Invocation-local scratch workspace used by one compile-pool worker.
  WorkspacePtr workspace;
  // Workspace allocation counters captured after benchmark warmup.
  loomc_workspace_statistics_t allocation_baseline = {};
};

class CompileScenario {
 public:
  explicit CompileScenario(iree_host_size_t workspace_block_size = 0);

  virtual ~CompileScenario();

  virtual iree_status_t SetUp(iree_host_size_t worker_count);

  virtual iree_host_size_t job_count() const = 0;

  // Prepares every worker-owned state and workload shape needed by timed jobs.
  virtual iree_status_t WarmUp(iree_host_size_t worker_count);

  virtual iree_status_t RunJob(iree_host_size_t worker_ordinal,
                               iree_host_size_t job_ordinal) = 0;

  virtual void SetExtraCounters(::benchmark::State& state) const;

  void ResetCounters();

  int64_t artifact_bytes() const;

  void SetWorkspaceAllocationCounters(::benchmark::State& state,
                                      int64_t total_jobs) const;

 protected:
  iree_status_t SetUpWorkerSlots(iree_host_size_t worker_count);

  WorkspacePtr& workspace_at(iree_host_size_t worker_ordinal);

  void RecordArtifactBytes(int64_t byte_count);

  // Shared immutable Loom context for this benchmark scenario.
  ContextPtr context_;

  // Shared immutable compiler handle for this benchmark scenario.
  CompilerPtr compiler_;

  // Shared immutable pass program for this benchmark scenario.
  PassProgramPtr pass_program_;

  // Per-worker invocation state.
  std::vector<WorkerSlot> workers_;

 private:
  // Total block size requested for each worker workspace. Zero selects the
  // production default.
  iree_host_size_t workspace_block_size_ = 0;

  // Total result artifact bytes observed by timed benchmark iterations.
  std::atomic<int64_t> artifact_bytes_{0};
};

// Shared production target setup for compile-and-emit benchmark scenarios.
class TargetCompileScenario : public CompileScenario {
 public:
  explicit TargetCompileScenario(iree_host_size_t workspace_block_size = 0);

 protected:
  iree_status_t SetUpTarget(iree_host_size_t worker_count,
                            TargetEnvironmentPtr target_environment,
                            TargetProfilePtr target_profile,
                            loomc_string_view_t pipeline_identifier);

  iree_status_t CompileModuleToPreparedLow(
      WorkspacePtr& workspace, ModulePtr& module,
      loomc_string_view_t function_symbol, loomc_string_view_t module_name,
      const loomc_module_t* config_module,
      loomc_config_policy_flags_t config_flags);

  loomc_target_environment_t* target_environment() const {
    return target_environment_.get();
  }

 private:
  // Target provider set shared by all jobs in the scenario.
  TargetEnvironmentPtr target_environment_;

  // Concrete immutable target facts shared by all jobs in the scenario.
  TargetProfilePtr target_profile_;
};

using CompileScenarioFactory = std::unique_ptr<CompileScenario> (*)(
    const ::benchmark::State& state, const void* user_data);

void RunCompileBenchmark(::benchmark::State& state,
                         CompileScenarioFactory factory, const void* user_data);

void RunCompileBenchmarkDirect(::benchmark::State& state,
                               CompileScenarioFactory factory,
                               const void* user_data);

// Runs without compile warmup. Registrations must use Iterations(1) so the
// counters and timing describe first growth rather than a cold/warm mixture.
void RunCompileBenchmarkDirectCold(::benchmark::State& state,
                                   CompileScenarioFactory factory,
                                   const void* user_data);

const loomc_artifact_t* FindArtifact(const loomc_result_t* result,
                                     loomc_string_view_t role,
                                     loomc_string_view_t format);

iree_status_t ValidateArtifact(const loomc_result_t* result,
                               loomc_string_view_t role,
                               loomc_string_view_t format,
                               iree_host_size_t minimum_data_length,
                               const char* description,
                               int64_t* out_artifact_bytes);

// Copies an artifact prefix into caller-owned storage without flattening the
// complete sequence.
iree_status_t ReadArtifactPrefix(const loomc_artifact_t* artifact,
                                 iree_byte_span_t prefix);

iree_status_t ValidateModuleBytecodeArtifact(const loomc_result_t* result,
                                             int64_t* out_artifact_bytes);

iree_status_t CreateTextSource(const std::string& identifier,
                               const std::string& text, SourcePtr* out_source);

iree_status_t CreateBenchmarkKernelSource(loomc_string_view_t identifier,
                                          SourcePtr* out_source);

iree_status_t CreateWorkspace(iree_host_size_t block_size,
                              WorkspacePtr* out_workspace);

iree_status_t DeserializeSource(loomc_context_t* context,
                                loomc_workspace_t* workspace,
                                const loomc_source_t* source,
                                ModulePtr* out_module);

iree_status_t CreateTextModule(loomc_context_t* context,
                               loomc_workspace_t* workspace,
                               const std::string& identifier,
                               const std::string& text, ModulePtr* out_module);

iree_status_t CloneModule(const loomc_module_t* source_module,
                          loomc_workspace_t* workspace, ModulePtr* out_module);

iree_status_t AddSourceToIndex(loomc_link_index_builder_t* builder,
                               loomc_source_t* source,
                               const std::string& provider_name,
                               loomc_link_provider_role_t role);

iree_status_t PreparePassProgram(loomc_context_t* context,
                                 PassProgramPtr* out_pass_program);

}  // namespace loomc::bench

#endif  // LOOMC_BENCHMARK_COMPILE_THROUGHPUT_BENCHMARK_H_
