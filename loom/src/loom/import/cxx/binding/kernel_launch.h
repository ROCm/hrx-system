// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_BINDING_KERNEL_LAUNCH_H_
#define LOOM_IMPORT_CXX_BINDING_KERNEL_LAUNCH_H_

#include <cxx/attributes.h>
#include <cxx/types_fwd.h>

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace loom::cxx_import {

class Diagnostics;
class LaunchContracts;
class PipelineContracts;
class Types;
struct StagedSignatureProjection;
struct KernelRecordProjection;

// Admits a record-valued source launch argument and returns its retained
// projection, or null for a non-record parameter. Records preserve nominal
// source type and supply every resource as a typed buffer. Native pointer
// fields cannot provide that binding contract at a source launch site.
const KernelRecordProjection* admit_kernel_record_argument(
    cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
    const cxx::Type* expected, const cxx::Type* actual, cxx::AST* owner);

// Source contract shared by check-runner and command-program kernel launches.
// Admission resolves the selected kernel, configuration, workload bundle, and
// ABI-shaped argument family once. Each body translator retains ownership of
// its source values: check cases materialize tensor storage while command
// programs forward native buffer and scalar values.
struct KernelLaunchIntrinsic {
  static bool supports(std::string_view name);
  static std::optional<KernelLaunchIntrinsic> resolve(
      cxx::TranslationUnit& unit, Diagnostics& diagnostics, Types& types,
      LaunchContracts& launches, PipelineContracts& pipelines,
      cxx::FunctionSymbol* function, const cxx::Attribute& attribute,
      cxx::AST* owner);

  bool equivalent(const KernelLaunchIntrinsic& other) const {
    return kernel == other.kernel && configuration == other.configuration &&
           pipeline == other.pipeline &&
           pipeline_buffer_materializations ==
               other.pipeline_buffer_materializations;
  }

  // Statically selected concrete kernel declaration.
  cxx::FunctionSymbol* kernel;
  // Kernel-owned launch configuration, absent for fixed geometry.
  cxx::FunctionSymbol* configuration;
  // Staged source/native signature for a kernel-scoped pipeline contract.
  const StagedSignatureProjection* pipeline;
  // Source arguments requiring buffer-boundary materialization. Exact source
  // arguments remain zero; nonzero entries select one transport operand from
  // a source buffer, shaped handle, or proven zero-origin pointer root.
  std::vector<uint8_t> pipeline_buffer_materializations;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_BINDING_KERNEL_LAUNCH_H_
