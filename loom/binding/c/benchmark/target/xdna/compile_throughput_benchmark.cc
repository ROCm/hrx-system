// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/binding/c/benchmark/kernels/ffn_gate_up_smoke.h"
#include "loom/binding/c/benchmark/workload_compile_benchmark.h"
#include "loomc/target/amd/xdna.h"

namespace {

using loomc::bench::FindEmbeddedSource;
using loomc::bench::loom_allocator;
using loomc::bench::RegisterInputScalingCompileBenchmarks;
using loomc::bench::RequireSucceededResult;
using loomc::bench::ResultPtr;
using loomc::bench::TargetEnvironmentPtr;
using loomc::bench::TargetProfilePtr;
using loomc::bench::to_iree_status;
using loomc::bench::ValidateArtifact;
using loomc::bench::WorkloadCompileTarget;

class XdnaWorkloadCompileTarget final : public WorkloadCompileTarget {
 public:
  const char* benchmark_name() const override { return "xdna-strix-halo"; }

  loomc_string_view_t pipeline_identifier() const override {
    return loomc_make_cstring_view("benchmark-xdna-prepared-low");
  }

  loomc_target_control_flow_lowering_t control_flow_lowering() const override {
    return LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG;
  }

  iree_status_t CreateTarget(
      TargetEnvironmentPtr* out_target_environment,
      TargetProfilePtr* out_target_profile) const override {
    loomc_target_environment_t* raw_environment = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_target_environment_create_xdna(
        loom_allocator(), &raw_environment)));
    TargetEnvironmentPtr environment(raw_environment);
    loomc_target_profile_t* raw_profile = nullptr;
    IREE_RETURN_IF_ERROR(to_iree_status(loomc_target_profile_create_xdna(
        environment.get(),
        loomc_make_cstring_view("amd.xdna.strix_halo.17f0_11"),
        loom_allocator(), &raw_profile)));
    out_target_environment->reset(environment.release());
    out_target_profile->reset(raw_profile);
    return iree_ok_status();
  }

  iree_status_t EmitArtifact(loomc_target_environment_t* target_environment,
                             loomc_workspace_t* workspace,
                             loomc_module_t* module,
                             loomc_string_view_t identifier,
                             loomc_compile_report_mode_t report_mode,
                             int64_t* out_artifact_byte_count) const override {
    const loomc_compile_report_options_t report_options = {
        .type = LOOMC_STRUCTURE_TYPE_COMPILE_REPORT_OPTIONS,
        .structure_size = sizeof(report_options),
        .next = nullptr,
        .mode = report_mode,
        .format = LOOMC_COMPILE_REPORT_FORMAT_JSON,
    };
    const loomc_emit_options_t emit_options = {
        .type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
        .structure_size = sizeof(emit_options),
        .next = report_mode != LOOMC_COMPILE_REPORT_MODE_NONE ? &report_options
                                                              : nullptr,
        .artifact_format = loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_XDNA),
        .identifier = identifier,
        .artifact_flags = LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY,
    };
    loomc_result_t* raw_result = nullptr;
    iree_status_t status = to_iree_status(
        loomc_emit_module(target_environment, workspace, module, &emit_options,
                          loom_allocator(), &raw_result));
    ResultPtr result(raw_result);
    IREE_RETURN_IF_ERROR(status);
    IREE_RETURN_IF_ERROR(RequireSucceededResult(result.get(), "XDNA emission"));
    return ValidateArtifact(result.get(), LOOMC_ARTIFACT_KIND_EXECUTABLE,
                            loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_XDNA),
                            1, "XDNA executable", out_artifact_byte_count);
  }
};

const XdnaWorkloadCompileTarget kXdnaWorkloadTarget;

[[maybe_unused]] const bool kXdnaWorkloadBenchmarksRegistered = [] {
  RegisterInputScalingCompileBenchmarks(
      kXdnaWorkloadTarget, "FfnGateUpQuadraticBF16",
      {
          .source =
              FindEmbeddedSource(loomc_benchmark_ffn_gate_up_smoke_create(),
                                 loomc_benchmark_ffn_gate_up_smoke_size(),
                                 "gate_up_quadratic_bf16_xdna.loom"),
          .function_symbol = "ffn_gate_up_quadratic_bf16",
          .artifact_identifier = "ffn_gate_up_quadratic_bf16.xdna",
          .input_size_config_symbol = "ffn_gate_up.input_size",
      },
      {512, 1024, 4096}, {512, 1024, 4096});
  return true;
}();

}  // namespace
