// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/testing/jit_module.h"

#include <filesystem>

#include "iree/vm/bytecode/module.h"
#include "iree/vm/sync.h"

namespace loom_serve::testing {

iree_status_t JitModuleTest::PrepareVm() {
  IREE_RETURN_IF_ERROR(
      iree_vm_environment_allocate(host_allocator_, &environment_));
  const iree_vm_ref_type_table_t* table = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_module_register_types(environment_, &table));
  IREE_RETURN_IF_ERROR(iree_hal_module_types_resolve(table, &types_));
  const loom_serve_stage_t stages[] = {
      {commands_[0], 1, transients_[0], dispatches_[0]},
      {commands_[1], 1, transients_[1], dispatches_[1]},
  };
  loom_serve_module_options_t options = {
      .binding_capacity = 1,
      .stages = {commands_[1] || dispatches_[1] ? 2u : 1u, stages},
  };
  IREE_RETURN_IF_ERROR(loom_serve_module_create(&types_, execution_, options,
                                                host_allocator_, &native_));
  const std::string directory = std::filesystem::absolute(directory_).string();
  const std::string path = directory + "/control.loom";
  iree_const_byte_span_t image;
  const iree_string_view_t roots[] = {IREE_SVL("step"),
                                      IREE_SVL("selected_step")};
  const iree_string_view_t providers[] = {IREE_SV("control.loom"),
                                          IREE_SV("vm_helpers.loom"),
                                          IREE_SV("stage.loom")};
  const loom_serve_sources_t sources = {
      .base_directory = iree_make_cstring_view(directory.c_str()),
      .count = IREE_ARRAYSIZE(providers),
      .values = providers,
  };
  IREE_RETURN_IF_ERROR(loom_serve_jit_compile_vm(
      loom_serve_host_target_environment(loom_serve_device_host(device_owner_)),
      &sources, iree_make_cstring_view(path.c_str()), IREE_ARRAYSIZE(roots),
      roots, &compilation_, host_allocator_, &image));
  auto status = iree_vm_bytecode_module_create_trusted(
      environment_, IREE_SV("model"), {image, host_allocator_}, host_allocator_,
      &bytecode_);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator_, (void*)image.data);
    return status;
  }
  iree_vm_module_t* libraries[] = {native_};
  IREE_RETURN_IF_ERROR(iree_vm_program_create(
      {bytecode_, iree_vm_module_span_from_array(libraries)}, host_allocator_,
      &program_));
  IREE_RETURN_IF_ERROR(iree_vm_invocation_initialize(
      iree_make_byte_span(invocation_storage_.data(),
                          invocation_storage_.size()),
      &invocation_));
  IREE_RETURN_IF_ERROR(iree_vm_process_create(program_, invocation_,
                                              iree_vm_variant_span_empty(),
                                              host_allocator_, &process_));
  return iree_vm_process_lookup_function(process_, IREE_SV("model"),
                                         IREE_SV("step"), &step_);
}

void JitModuleTest::ReleaseProgram() {
  iree_vm_process_release(process_);
  if (invocation_) {
    iree_vm_invocation_deinitialize(invocation_);
  }
  iree_vm_program_release(program_);
  iree_vm_module_release(native_);
  iree_vm_module_release(bytecode_);
  iree_vm_environment_free(environment_);
}

}  // namespace loom_serve::testing
