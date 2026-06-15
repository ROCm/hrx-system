// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/binding/c/benchmark/compile_throughput_benchmark.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <thread>

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

static std::string FormatStatus(iree_status_t status) {
  char buffer[4096] = {0};
  iree_host_size_t length = 0;
  iree_status_format(status, sizeof(buffer), buffer, &length);
  return std::string(buffer, std::min(length, sizeof(buffer) - 1));
}

static iree_status_t MakeResultFailureStatus(const loomc_result_t* result,
                                             const char* operation) {
  if (result == nullptr) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "%s failed without producing a result", operation);
  }

  loomc_host_size_t diagnostic_count = loomc_result_diagnostic_count(result);
  if (diagnostic_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "%s failed without diagnostics", operation);
  }

  const loomc_diagnostic_t* diagnostic = loomc_result_diagnostic_at(result, 0);
  if (diagnostic == nullptr) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "%s failed with an unreadable diagnostic",
                            operation);
  }
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION, "%s failed: %.*s: %.*s", operation,
      (int)diagnostic->code.size, diagnostic->code.data,
      (int)diagnostic->message.size, diagnostic->message.data);
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

iree_allocator_t host_allocator() { return iree_allocator_system(); }

loomc_allocator_t loom_allocator() {
  return loomc_allocator_from_iree(host_allocator());
}

iree_status_t to_iree_status(loomc_status_t status) {
  return iree_status_from_loomc(status);
}

iree_status_t RequireSucceededResult(const loomc_result_t* result,
                                     const char* operation) {
  if (result != nullptr && loomc_result_succeeded(result)) {
    return iree_ok_status();
  }
  return MakeResultFailureStatus(result, operation);
}

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

iree_status_t CreateWorkspace(WorkspacePtr* out_workspace) {
  out_workspace->reset();
  loomc_workspace_t* workspace = nullptr;
  IREE_RETURN_IF_ERROR(to_iree_status(loomc_workspace_create(
      /*options=*/nullptr, loom_allocator(), &workspace)));
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
  loomc_link_index_source_options_t options = {
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

void CompileScenario::ResetCounters() {
  artifact_bytes_.store(0, std::memory_order_relaxed);
}

int64_t CompileScenario::artifact_bytes() const {
  return artifact_bytes_.load(std::memory_order_relaxed);
}

iree_status_t CompileScenario::SetUpWorkerSlots(iree_host_size_t worker_count) {
  workers_.resize(worker_count);
  for (WorkerSlot& worker : workers_) {
    IREE_RETURN_IF_ERROR(CreateWorkspace(&worker.workspace));
  }
  return iree_ok_status();
}

WorkspacePtr& CompileScenario::workspace_at(iree_host_size_t worker_ordinal) {
  return workers_[worker_ordinal].workspace;
}

void CompileScenario::RecordArtifactBytes(int64_t byte_count) {
  artifact_bytes_.fetch_add(byte_count, std::memory_order_relaxed);
}

void RunCompileBenchmark(::benchmark::State& state,
                         CompileScenarioFactory factory, void* user_data) {
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
    status = loomc_benchmark_compile_pool_run_batch(
        &pool,
        std::min<iree_host_size_t>(scenario->job_count(),
                                   (iree_host_size_t)worker_count),
        RunScenarioJob, scenario.get());
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

void RunCompileBenchmarkDirect(::benchmark::State& state,
                               CompileScenarioFactory factory,
                               void* user_data) {
  constexpr int64_t kWorkerCount = 1;
  std::unique_ptr<CompileScenario> scenario = factory(state, user_data);
  if (!scenario) {
    state.SkipWithError("scenario factory returned null");
    return;
  }

  iree_status_t status = scenario->SetUp((iree_host_size_t)kWorkerCount);
  if (iree_status_is_ok(status)) {
    iree_host_size_t warmup_count =
        std::min<iree_host_size_t>(scenario->job_count(), kWorkerCount);
    for (iree_host_size_t i = 0; i < warmup_count && iree_status_is_ok(status);
         ++i) {
      status = scenario->RunJob(/*worker_ordinal=*/0, i);
    }
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

}  // namespace loomc::bench
