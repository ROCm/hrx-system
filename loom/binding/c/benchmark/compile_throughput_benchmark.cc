// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/binding/c/benchmark/compile_throughput_benchmark.h"

#include <cstring>
#include <string>
#include <thread>
#include <utility>

#include "benchmark/benchmark.h"
#include "loom/binding/c/benchmark/benchmark_kernels.h"
#include "loom/binding/c/benchmark/util/compile_pool_prototype.h"

namespace loomc::bench {
namespace {

static bool ShouldSkipWorkerCount(::benchmark::State& state,
                                  int64_t worker_count) {
  unsigned int available = std::thread::hardware_concurrency();
  if (available > 0 && worker_count > static_cast<int64_t>(available)) {
    state.SkipWithMessage("worker_count exceeds available cores");
    return true;
  }
  return false;
}

static iree_status_t RunScenarioJob(void* user_data,
                                    iree_host_size_t worker_ordinal,
                                    iree_host_size_t job_ordinal) {
  CompileScenario* scenario = static_cast<CompileScenario*>(user_data);
  return scenario->RunJob(worker_ordinal, job_ordinal);
}

static void SetThroughputCounters(::benchmark::State& state,
                                  const CompileScenario& scenario,
                                  int64_t total_jobs, int64_t worker_count) {
  int64_t total_artifact_bytes = scenario.artifact_bytes();
  state.SetItemsProcessed(total_jobs);
  state.SetBytesProcessed(total_artifact_bytes);
  state.counters["kernels/s"] =
      ::benchmark::Counter(total_jobs, ::benchmark::Counter::kIsRate);
  state.counters["worker_count"] = (double)worker_count;
  state.counters["artifact_bytes"] = (double)total_artifact_bytes;
  state.counters["artifact_bytes/kernel"] =
      total_jobs == 0 ? 0.0 : (double)total_artifact_bytes / (double)total_jobs;
  scenario.SetWorkspaceAllocationCounters(state, total_jobs);
  scenario.SetExtraCounters(state);
}

static iree_status_t CreateTextSourceFromViews(loomc_string_view_t identifier,
                                               loomc_byte_span_t contents,
                                               SourcePtr* out_source) {
  out_source->reset();
  loomc_source_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.format=*/LOOMC_SOURCE_FORMAT_TEXT,
      /*.identifier=*/identifier,
      /*.contents=*/contents,
      /*.storage=*/LOOMC_SOURCE_STORAGE_COPY,
      /*.release=*/nullptr,
      /*.release_user_data=*/nullptr,
  };
  loomc_source_t* source = nullptr;
  IREE_RETURN_IF_ERROR(
      to_iree_status(loomc_source_create(&options, loom_allocator(), &source)));
  out_source->reset(source);
  return iree_ok_status();
}

}  // namespace

const loomc_artifact_t* FindArtifact(const loomc_result_t* result,
                                     loomc_artifact_kind_t kind,
                                     loomc_string_view_t format) {
  for (loomc_host_size_t i = 0; i < loomc_result_artifact_count(result); ++i) {
    const loomc_artifact_t* artifact = loomc_result_artifact_at(result, i);
    if (artifact != nullptr && artifact->kind == kind &&
        loomc_string_view_equal(artifact->format, format)) {
      return artifact;
    }
  }
  return nullptr;
}

iree_status_t ValidateArtifact(const loomc_result_t* result,
                               loomc_artifact_kind_t kind,
                               loomc_string_view_t format,
                               iree_host_size_t minimum_data_length,
                               const char* description,
                               int64_t* out_artifact_bytes) {
  const loomc_artifact_t* artifact = FindArtifact(result, kind, format);
  if (artifact == nullptr) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "%s artifact was not produced", description);
  }
  if (artifact->contents.data == nullptr ||
      artifact->contents.data_length < minimum_data_length) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "%s artifact is empty or truncated", description);
  }
  const uint8_t* artifact_data = artifact->contents.data;
  loomc_host_size_t artifact_data_length = artifact->contents.data_length;
  ::benchmark::DoNotOptimize(artifact_data);
  ::benchmark::DoNotOptimize(artifact_data_length);
  *out_artifact_bytes = (int64_t)artifact->contents.data_length;
  return iree_ok_status();
}

iree_status_t ValidateModuleBytecodeArtifact(const loomc_result_t* result,
                                             int64_t* out_artifact_bytes) {
  return ValidateArtifact(
      result, LOOMC_ARTIFACT_KIND_MODULE,
      loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_LOOM_BYTECODE),
      /*minimum_data_length=*/1, "module bytecode", out_artifact_bytes);
}

iree_status_t CreateTextSource(const std::string& identifier,
                               const std::string& text, SourcePtr* out_source) {
  return CreateTextSourceFromViews(
      loomc_make_string_view(identifier.data(), identifier.size()),
      loomc_make_byte_span(text.data(), text.size()), out_source);
}

iree_status_t CreateBenchmarkKernelSource(loomc_string_view_t identifier,
                                          SourcePtr* out_source) {
  const iree_file_toc_t* kernels = loomc_benchmark_kernels_create();
  for (size_t i = 0; i < loomc_benchmark_kernels_size(); ++i) {
    const iree_file_toc_t& file = kernels[i];
    loomc_string_view_t file_name = loomc_make_cstring_view(file.name);
    if (!loomc_string_view_equal(identifier, file_name)) {
      continue;
    }
    return CreateTextSourceFromViews(
        file_name, loomc_make_byte_span(file.data, file.size), out_source);
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "benchmark kernel %.*s not found",
                          (int)identifier.size, identifier.data);
}

iree_status_t CreateWorkspace(iree_host_size_t block_size,
                              WorkspacePtr* out_workspace) {
  out_workspace->reset();
  const loomc_workspace_options_t options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_WORKSPACE_OPTIONS,
      /*.structure_size=*/sizeof(options),
      /*.next=*/nullptr,
      /*.block_size=*/block_size,
  };
  loomc_workspace_t* workspace = nullptr;
  IREE_RETURN_IF_ERROR(to_iree_status(
      loomc_workspace_create(&options, loom_allocator(), &workspace)));
  out_workspace->reset(workspace);
  return iree_ok_status();
}

iree_status_t DeserializeSource(loomc_context_t* context,
                                loomc_workspace_t* workspace,
                                const loomc_source_t* source,
                                ModulePtr* out_module) {
  out_module->reset();
  loomc_module_t* module = nullptr;
  loomc_result_t* raw_result = nullptr;
  iree_status_t status = to_iree_status(loomc_module_deserialize_from_source(
      context, workspace, source, /*options=*/nullptr, loom_allocator(),
      &module, &raw_result));
  ModulePtr module_ptr(module);
  ResultPtr result(raw_result);
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(
      RequireSucceededResult(result.get(), "source deserialization"));
  out_module->reset(module_ptr.release());
  return iree_ok_status();
}

iree_status_t CloneModule(const loomc_module_t* source_module,
                          loomc_workspace_t* workspace, ModulePtr* out_module) {
  out_module->reset();
  loomc_module_t* module = nullptr;
  IREE_RETURN_IF_ERROR(to_iree_status(
      loomc_module_clone(source_module, workspace, loom_allocator(), &module)));
  out_module->reset(module);
  return iree_ok_status();
}

iree_status_t AddSourceToIndex(loomc_link_index_builder_t* builder,
                               loomc_source_t* source,
                               const std::string& provider_name,
                               loomc_link_provider_role_t role) {
  loomc_link_index_provider_options_t options = {
      /*.provider_name=*/
      loomc_make_string_view(provider_name.data(), provider_name.size()),
      /*.role=*/role,
  };
  return to_iree_status(loomc_link_index_builder_add_source(
      builder, source, &options, /*out_slot=*/nullptr));
}

iree_status_t PreparePassProgram(loomc_context_t* context,
                                 PassProgramPtr* out_pass_program) {
  out_pass_program->reset();
  loomc_pass_program_t* pass_program = nullptr;
  loomc_result_t* raw_result = nullptr;
  iree_status_t status =
      to_iree_status(loomc_pass_program_create_from_pipeline_text(
          context, loomc_make_cstring_view("canonicalize,dce"),
          /*options=*/nullptr, loom_allocator(), &pass_program, &raw_result));
  PassProgramPtr pass_program_ptr(pass_program);
  ResultPtr result(raw_result);
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(
      RequireSucceededResult(result.get(), "pass program preparation"));
  out_pass_program->reset(pass_program_ptr.release());
  return iree_ok_status();
}

CompileScenario::CompileScenario(iree_host_size_t workspace_block_size)
    : workspace_block_size_(workspace_block_size) {}

CompileScenario::~CompileScenario() = default;

iree_status_t CompileScenario::SetUp(iree_host_size_t worker_count) {
  loomc_context_t* context = nullptr;
  IREE_RETURN_IF_ERROR(to_iree_status(loomc_context_create(
      /*options=*/nullptr, loom_allocator(), &context)));
  context_.reset(context);

  loomc_compiler_t* compiler = nullptr;
  IREE_RETURN_IF_ERROR(to_iree_status(loomc_compiler_create(
      context_.get(), /*options=*/nullptr, loom_allocator(), &compiler)));
  compiler_.reset(compiler);

  IREE_RETURN_IF_ERROR(PreparePassProgram(context_.get(), &pass_program_));
  return SetUpWorkerSlots(worker_count);
}

void CompileScenario::SetExtraCounters(::benchmark::State& state) const {
  (void)state;
}

iree_status_t CompileScenario::WarmUp(iree_host_size_t worker_count) {
  for (iree_host_size_t worker_ordinal = 0; worker_ordinal < worker_count;
       ++worker_ordinal) {
    IREE_RETURN_IF_ERROR(RunJob(worker_ordinal, worker_ordinal % job_count()));
  }
  return iree_ok_status();
}

void CompileScenario::ResetCounters() {
  artifact_bytes_.store(0, std::memory_order_relaxed);
  for (WorkerSlot& worker : workers_) {
    loomc_workspace_query_statistics(worker.workspace.get(),
                                     &worker.allocation_baseline);
  }
}

int64_t CompileScenario::artifact_bytes() const {
  return artifact_bytes_.load(std::memory_order_relaxed);
}

void CompileScenario::SetWorkspaceAllocationCounters(::benchmark::State& state,
                                                     int64_t total_jobs) const {
  loomc_workspace_statistics_t total = {0};
  uint64_t lifetime_block_system_allocation_count = 0;
  uint64_t lifetime_block_system_allocation_bytes = 0;
  for (const WorkerSlot& worker : workers_) {
    loomc_workspace_statistics_t current;
    loomc_workspace_query_statistics(worker.workspace.get(), &current);
    lifetime_block_system_allocation_count +=
        current.block_system_allocation_count;
    lifetime_block_system_allocation_bytes +=
        current.block_system_allocation_bytes;
    total.block_system_allocation_count +=
        current.block_system_allocation_count -
        worker.allocation_baseline.block_system_allocation_count;
    total.block_system_allocation_bytes +=
        current.block_system_allocation_bytes -
        worker.allocation_baseline.block_system_allocation_bytes;
    total.oversized_allocation_count +=
        current.oversized_allocation_count -
        worker.allocation_baseline.oversized_allocation_count;
    total.oversized_allocation_bytes +=
        current.oversized_allocation_bytes -
        worker.allocation_baseline.oversized_allocation_bytes;
  }

  const double job_count = (double)total_jobs;
  if (!workers_.empty()) {
    loomc_workspace_statistics_t statistics;
    loomc_workspace_query_statistics(workers_.front().workspace.get(),
                                     &statistics);
    state.counters["workspace_block_size"] =
        (double)statistics.total_block_size;
    state.counters["workspace_usable_block_size"] =
        (double)statistics.usable_block_size;
  }
  state.counters["workspace_block_allocations/kernel"] =
      total_jobs == 0 ? 0.0
                      : (double)total.block_system_allocation_count / job_count;
  state.counters["workspace_block_bytes/kernel"] =
      total_jobs == 0 ? 0.0
                      : (double)total.block_system_allocation_bytes / job_count;
  state.counters["workspace_block_system_allocations"] =
      (double)lifetime_block_system_allocation_count;
  state.counters["workspace_block_system_bytes"] =
      (double)lifetime_block_system_allocation_bytes;
  state.counters["workspace_oversized_allocations/kernel"] =
      total_jobs == 0 ? 0.0
                      : (double)total.oversized_allocation_count / job_count;
  state.counters["workspace_oversized_bytes/kernel"] =
      total_jobs == 0 ? 0.0
                      : (double)total.oversized_allocation_bytes / job_count;
}

iree_status_t CompileScenario::SetUpWorkerSlots(iree_host_size_t worker_count) {
  workers_.resize(worker_count);
  for (WorkerSlot& worker : workers_) {
    IREE_RETURN_IF_ERROR(
        CreateWorkspace(workspace_block_size_, &worker.workspace));
  }
  return iree_ok_status();
}

WorkspacePtr& CompileScenario::workspace_at(iree_host_size_t worker_ordinal) {
  return workers_[worker_ordinal].workspace;
}

void CompileScenario::RecordArtifactBytes(int64_t byte_count) {
  artifact_bytes_.fetch_add(byte_count, std::memory_order_relaxed);
}

TargetCompileScenario::TargetCompileScenario(
    iree_host_size_t workspace_block_size)
    : CompileScenario(workspace_block_size) {}

iree_status_t TargetCompileScenario::SetUpTarget(
    iree_host_size_t worker_count, TargetEnvironmentPtr target_environment,
    TargetProfilePtr target_profile, loomc_string_view_t pipeline_identifier) {
  target_environment_ = std::move(target_environment);
  target_profile_ = std::move(target_profile);

  loomc_context_target_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.target_environment=*/target_environment_.get(),
  };
  loomc_context_options_t context_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
      /*.structure_size=*/sizeof(context_options),
      /*.next=*/&target_options,
  };
  loomc_context_t* raw_context = nullptr;
  IREE_RETURN_IF_ERROR(to_iree_status(
      loomc_context_create(&context_options, loom_allocator(), &raw_context)));
  context_.reset(raw_context);

  loomc_compiler_t* raw_compiler = nullptr;
  IREE_RETURN_IF_ERROR(to_iree_status(loomc_compiler_create(
      context_.get(), /*options=*/nullptr, loom_allocator(), &raw_compiler)));
  compiler_.reset(raw_compiler);

  loomc_target_pipeline_options_t pipeline_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS,
      /*.structure_size=*/sizeof(pipeline_options),
      /*.next=*/nullptr,
      /*.identifier=*/pipeline_identifier,
      /*.kind=*/LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW,
      /*.control_flow_lowering=*/LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG,
      /*.source_to_low_max_errors=*/20,
  };
  loomc_pass_program_t* raw_pass_program = nullptr;
  loomc_result_t* raw_result = nullptr;
  iree_status_t status =
      to_iree_status(loomc_pass_program_create_from_target_pipeline(
          context_.get(), &pipeline_options, loom_allocator(),
          &raw_pass_program, &raw_result));
  PassProgramPtr pass_program(raw_pass_program);
  ResultPtr result(raw_result);
  IREE_RETURN_IF_ERROR(status);
  IREE_RETURN_IF_ERROR(
      RequireSucceededResult(result.get(), "target pipeline preparation"));
  pass_program_.reset(pass_program.release());

  return SetUpWorkerSlots(worker_count);
}

iree_status_t TargetCompileScenario::CompileModuleToPreparedLow(
    WorkspacePtr& workspace, ModulePtr& module,
    loomc_string_view_t function_symbol, loomc_string_view_t module_name,
    loomc_config_options_t config) {
  const loomc_target_specialization_t specialization = {
      /*.function_symbol=*/function_symbol,
      /*.target_profile=*/target_profile_.get(),
  };
  loomc_target_specialization_options_t target_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS,
      /*.structure_size=*/sizeof(target_options),
      /*.next=*/nullptr,
      /*.specializations=*/&specialization,
      /*.specialization_count=*/1,
  };
  loomc_compile_options_t compile_options = {
      /*.type=*/LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
      /*.structure_size=*/sizeof(compile_options),
      /*.next=*/&target_options,
      /*.module_name=*/module_name,
      /*.artifact_flags=*/0,
      /*.config=*/config,
  };

  loomc_result_t* raw_result = nullptr;
  iree_status_t status = to_iree_status(loomc_compile_module(
      compiler_.get(), workspace.get(), pass_program_.get(), module.get(),
      &compile_options, loom_allocator(), &raw_result));
  ResultPtr result(raw_result);
  IREE_RETURN_IF_ERROR(status);
  return RequireSucceededResult(result.get(), "compilation");
}

void RunCompileBenchmark(::benchmark::State& state,
                         CompileScenarioFactory factory,
                         const void* user_data) {
  const int64_t worker_count = state.range(0);
  if (ShouldSkipWorkerCount(state, worker_count)) {
    return;
  }

  loomc_benchmark_compile_pool_t pool;
  loomc_benchmark_compile_pool_initialize_empty(&pool);
  iree_status_t status = loomc_benchmark_compile_pool_initialize_owning(
      (iree_host_size_t)worker_count, host_allocator(), &pool);
  if (!iree_status_is_ok(status)) {
    std::string message = FormatStatus(status);
    iree_status_free(status);
    state.SkipWithError(message.c_str());
    return;
  }

  std::unique_ptr<CompileScenario> scenario = factory(state, user_data);
  if (!scenario) {
    state.SkipWithError("scenario factory returned null");
    loomc_benchmark_compile_pool_deinitialize(&pool);
    return;
  }

  status = scenario->SetUp((iree_host_size_t)worker_count);
  if (iree_status_is_ok(status)) {
    status = scenario->WarmUp((iree_host_size_t)worker_count);
  }
  if (!iree_status_is_ok(status)) {
    std::string message = FormatStatus(status);
    iree_status_free(status);
    state.SkipWithError(message.c_str());
    loomc_benchmark_compile_pool_deinitialize(&pool);
    return;
  }

  scenario->ResetCounters();
  int64_t total_jobs = 0;
  for (auto _ : state) {
    status = loomc_benchmark_compile_pool_run_batch(
        &pool, scenario->job_count(), RunScenarioJob, scenario.get());
    if (!iree_status_is_ok(status)) {
      std::string message = FormatStatus(status);
      iree_status_free(status);
      state.SkipWithError(message.c_str());
      break;
    }
    total_jobs += (int64_t)scenario->job_count();
  }

  SetThroughputCounters(state, *scenario, total_jobs, worker_count);

  loomc_benchmark_compile_pool_deinitialize(&pool);
}

enum class DirectBenchmarkWarmup {
  kScenario,
  kNone,
};

static void RunCompileBenchmarkDirectImpl(::benchmark::State& state,
                                          CompileScenarioFactory factory,
                                          const void* user_data,
                                          DirectBenchmarkWarmup warmup) {
  constexpr int64_t kWorkerCount = 1;
  std::unique_ptr<CompileScenario> scenario = factory(state, user_data);
  if (!scenario) {
    state.SkipWithError("scenario factory returned null");
    return;
  }

  iree_status_t status = scenario->SetUp((iree_host_size_t)kWorkerCount);
  if (iree_status_is_ok(status) && warmup == DirectBenchmarkWarmup::kScenario) {
    status = scenario->WarmUp(kWorkerCount);
  }
  if (!iree_status_is_ok(status)) {
    std::string message = FormatStatus(status);
    iree_status_free(status);
    state.SkipWithError(message.c_str());
    return;
  }

  scenario->ResetCounters();
  int64_t total_jobs = 0;
  for (auto _ : state) {
    for (iree_host_size_t job_ordinal = 0; job_ordinal < scenario->job_count();
         ++job_ordinal) {
      status = scenario->RunJob(/*worker_ordinal=*/0, job_ordinal);
      if (!iree_status_is_ok(status)) {
        break;
      }
    }
    if (!iree_status_is_ok(status)) {
      std::string message = FormatStatus(status);
      iree_status_free(status);
      state.SkipWithError(message.c_str());
      break;
    }
    total_jobs += (int64_t)scenario->job_count();
  }

  SetThroughputCounters(state, *scenario, total_jobs, kWorkerCount);
}

void RunCompileBenchmarkDirect(::benchmark::State& state,
                               CompileScenarioFactory factory,
                               const void* user_data) {
  RunCompileBenchmarkDirectImpl(state, factory, user_data,
                                DirectBenchmarkWarmup::kScenario);
}

void RunCompileBenchmarkDirectCold(::benchmark::State& state,
                                   CompileScenarioFactory factory,
                                   const void* user_data) {
  RunCompileBenchmarkDirectImpl(state, factory, user_data,
                                DirectBenchmarkWarmup::kNone);
}

}  // namespace loomc::bench
