// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_BINDING_C_BENCHMARK_UTIL_BENCHMARK_SUPPORT_H_
#define LOOM_BINDING_C_BENCHMARK_UTIL_BENCHMARK_SUPPORT_H_

#include <memory>
#include <string>

#include "iree/base/api.h"
#include "loomc/iree.h"

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
using WorkspacePtr = HandlePtr<loomc_workspace_t, loomc_workspace_release>;

iree_allocator_t host_allocator();

loomc_allocator_t loom_allocator();

iree_status_t to_iree_status(loomc_status_t status);

std::string FormatStatus(iree_status_t status);

iree_status_t RequireSucceededResult(const loomc_result_t* result,
                                     const char* operation);

}  // namespace loomc::bench

#endif  // LOOM_BINDING_C_BENCHMARK_UTIL_BENCHMARK_SUPPORT_H_
