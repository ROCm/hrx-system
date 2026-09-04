// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_BENCHMARK_WORKLOAD_COMPILE_BENCHMARK_H_
#define LOOMC_BENCHMARK_WORKLOAD_COMPILE_BENCHMARK_H_

#include <cstdint>

#include "iree/base/api.h"
#include "loom/binding/c/benchmark/compile_throughput_benchmark.h"

namespace loomc::bench {

// Target-specific operations required by the shared workload benchmarks.
// Implementations live beside their target binding so the workload runner does
// not learn target profile, artifact, or executable-format details.
class WorkloadCompileTarget {
 public:
  virtual ~WorkloadCompileTarget() = default;

  // Stable target/profile name included in registered benchmark names.
  virtual const char* benchmark_name() const = 0;

  // Stable identifier assigned to the shared prepared-low pass pipeline.
  virtual loomc_string_view_t pipeline_identifier() const = 0;

  // Creates the target environment and optional exact specialization profile.
  // A null profile means that the fixture already names its exact target.
  virtual iree_status_t CreateTarget(
      TargetEnvironmentPtr* out_target_environment,
      TargetProfilePtr* out_target_profile) const = 0;

  // Emits and validates the primary executable artifact for one compiled
  // workload module.
  virtual iree_status_t EmitArtifact(
      loomc_target_environment_t* target_environment,
      loomc_workspace_t* workspace, loomc_module_t* module,
      loomc_string_view_t identifier,
      int64_t* out_artifact_byte_count) const = 0;
};

// One target implementation of a target-independent functional workload.
struct CompileWorkload {
  // Embedded source file read during benchmark setup.
  EmbeddedSource source;

  // Root function specialized to the target profile.
  const char* function_symbol;

  // Logical artifact identifier passed to the target emitter.
  const char* artifact_identifier;
};

// One target implementation of a workload scaled by an exact configuration
// value before target lowering.
struct InputScalingCompileWorkload {
  // Embedded source file read during benchmark setup.
  EmbeddedSource source;

  // Root function specialized to the target profile.
  const char* function_symbol;

  // Logical artifact identifier passed to the target emitter.
  const char* artifact_identifier;

  // Configuration symbol assigned the benchmark's input-size argument.
  const char* input_size_config_symbol;
};

// Registers the full-attention phase and module-scaling benchmarks for one
// target implementation.
void RegisterAttentionCompileBenchmarks(const WorkloadCompileTarget& target,
                                        CompileWorkload workload);

// Registers one input-size compiler-scaling workload for a target
// implementation. The workload name identifies the shared functional contract
// implemented by every registered target.
void RegisterInputScalingCompileBenchmarks(
    const WorkloadCompileTarget& target, const char* workload_name,
    InputScalingCompileWorkload workload);

}  // namespace loomc::bench

#endif  // LOOMC_BENCHMARK_WORKLOAD_COMPILE_BENCHMARK_H_
