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
#include "loomc/iree.h"

namespace benchmark {
class State;
}  // namespace benchmark

namespace loomc::bench {

template <typename T, void (*Release)(T*)>
struct HandleDeleter {
  void operator()(T* handle) const { Release(handle); }
};

template <typename T, void (*Release)(T*)>
using HandlePtr = std::unique_ptr<T, HandleDeleter<T, Release>>;

using CompilerPtr = HandlePtr<loomc_compiler_t, loomc_compiler_release>;
using ContextPtr = HandlePtr<loomc_context_t, loomc_context_release>;
using LinkIndexBuilderPtr =
    HandlePtr<loomc_link_index_builder_t, loomc_link_index_builder_release>;
using LinkIndexPtr = HandlePtr<loomc_link_index_t, loomc_link_index_release>;
using LinkerPtr = HandlePtr<loomc_linker_t, loomc_linker_release>;
using ModulePtr = HandlePtr<loomc_module_t, loomc_module_release>;
using PassProgramPtr =
    HandlePtr<loomc_pass_program_t, loomc_pass_program_release>;
using ResultPtr = HandlePtr<loomc_result_t, loomc_result_release>;
using SourcePtr = HandlePtr<loomc_source_t, loomc_source_release>;
using TargetEnvironmentPtr =
    HandlePtr<loomc_target_environment_t, loomc_target_environment_release>;
using TargetProfilePtr =
    HandlePtr<loomc_target_profile_t, loomc_target_profile_release>;
using TargetSelectionPtr =
    HandlePtr<loomc_target_selection_t, loomc_target_selection_release>;
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;

struct WorkerSlot {
  // Invocation-local scratch workspace used by one compile-pool worker.
  WorkspacePtr workspace;
};

class CompileScenario {
 public:
  virtual ~CompileScenario();

  virtual iree_status_t SetUp(iree_host_size_t worker_count);

  virtual iree_host_size_t job_count() const = 0;

  virtual iree_status_t RunJob(iree_host_size_t worker_ordinal,
                               iree_host_size_t job_ordinal) = 0;

  virtual void SetExtraCounters(::benchmark::State& state) const;

  void ResetCounters();

  int64_t artifact_bytes() const;

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
  // Total result artifact bytes observed by timed benchmark iterations.
  std::atomic<int64_t> artifact_bytes_{0};
};

using CompileScenarioFactory = std::unique_ptr<CompileScenario> (*)(
    const ::benchmark::State& state, void* user_data);

iree_allocator_t host_allocator();

loomc_allocator_t loom_allocator();

iree_status_t to_iree_status(loomc_status_t status);

void RunCompileBenchmark(::benchmark::State& state,
                         CompileScenarioFactory factory, void* user_data);

void RunCompileBenchmarkDirect(::benchmark::State& state,
                               CompileScenarioFactory factory, void* user_data);

iree_status_t RequireSucceededResult(const loomc_result_t* result,
                                     const char* operation);

const loomc_artifact_t* FindArtifact(const loomc_result_t* result,
                                     loomc_artifact_kind_t kind,
                                     loomc_string_view_t format);

iree_status_t ValidateArtifact(const loomc_result_t* result,
                               loomc_artifact_kind_t kind,
                               loomc_string_view_t format,
                               iree_host_size_t minimum_data_length,
                               const char* description,
                               int64_t* out_artifact_bytes);

iree_status_t ValidateModuleBytecodeArtifact(const loomc_result_t* result,
                                             int64_t* out_artifact_bytes);

iree_status_t CreateTextSource(const std::string& identifier,
                               const std::string& text, SourcePtr* out_source);

iree_status_t CreateBenchmarkKernelSource(loomc_string_view_t identifier,
                                          SourcePtr* out_source);

iree_status_t CreateWorkspace(WorkspacePtr* out_workspace);

iree_status_t DeserializeSource(loomc_context_t* context,
                                loomc_workspace_t* workspace,
                                const loomc_source_t* source,
                                ModulePtr* out_module);

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
