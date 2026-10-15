// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/program/program.h"

#include "iree/vm/bytecode/module.h"
#include "iree/vm/sync.h"
#include "loom/serve/runtime/program/jit.h"

struct loom_serve_program_t {
  // Host allocation policy for the owner and all VM storage.
  iree_allocator_t host_allocator;
  // Owned linked program retaining bytecode and native libraries.
  iree_vm_program_t* program;
  // Owned process borrowing no session state.
  iree_vm_process_t* process;
  // Fixed invocation backing, independent of request and token counts.
  iree_alignas(iree_max_align_t) uint8_t storage[16384];
  // Invocation borrowing storage; calls are serialized by the model owner.
  iree_vm_invocation_t* invocation;
};

void loom_serve_program_free(loom_serve_program_t* program) {
  if (!program) {
    return;
  }
  iree_vm_process_release(program->process);
  if (program->invocation) {
    iree_vm_invocation_deinitialize(program->invocation);
  }
  iree_vm_program_release(program->program);
  iree_allocator_free(program->host_allocator, program);
}

iree_status_t loom_serve_program_compile(
    loomc_target_environment_t* target_environment,
    iree_vm_environment_t* environment, const loom_serve_sources_t* sources,
    iree_string_view_t source_path, iree_host_size_t root_count,
    const iree_string_view_t* roots, loom_serve_compilation_report_t* report,
    iree_allocator_t host_allocator, iree_vm_module_t** out_module) {
  *out_module = NULL;
  iree_const_byte_span_t image = iree_const_byte_span_empty();
  iree_status_t status = loom_serve_jit_compile_vm(
      target_environment, sources, source_path, root_count, roots, report,
      host_allocator, &image);
  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_module_create_trusted(
        environment, IREE_SV("model"),
        (iree_vm_bytecode_module_storage_t){image, host_allocator},
        host_allocator, out_module);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(host_allocator, (void*)image.data);
    }
  }
  return status;
}

iree_status_t loom_serve_program_link(iree_vm_module_t* module,
                                      iree_vm_module_span_t libraries,
                                      iree_allocator_t host_allocator,
                                      loom_serve_program_t** out_program) {
  *out_program = NULL;
  loom_serve_program_t* program = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*program),
                                             (void**)&program));
  program->host_allocator = host_allocator;
  iree_status_t status =
      iree_vm_program_create((iree_vm_program_modules_t){module, libraries},
                             host_allocator, &program->program);
  if (iree_status_is_ok(status)) {
    status = iree_vm_invocation_initialize(
        iree_make_byte_span(program->storage, sizeof(program->storage)),
        &program->invocation);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_process_create(program->program, program->invocation,
                                    iree_vm_variant_span_empty(),
                                    host_allocator, &program->process);
  }
  if (iree_status_is_ok(status)) {
    *out_program = program;
  } else {
    loom_serve_program_free(program);
  }
  return status;
}

iree_status_t loom_serve_program_allocate(
    loomc_target_environment_t* target_environment,
    iree_vm_environment_t* environment, const loom_serve_sources_t* sources,
    iree_string_view_t source_path, iree_host_size_t root_count,
    const iree_string_view_t* roots, iree_vm_module_span_t libraries,
    loom_serve_compilation_report_t* report, iree_allocator_t host_allocator,
    loom_serve_program_t** out_program) {
  *out_program = NULL;
  iree_vm_module_t* module = NULL;
  iree_status_t status = loom_serve_program_compile(
      target_environment, environment, sources, source_path, root_count, roots,
      report, host_allocator, &module);
  if (iree_status_is_ok(status)) {
    status =
        loom_serve_program_link(module, libraries, host_allocator, out_program);
  }
  iree_vm_module_release(module);
  return status;
}

iree_vm_process_t* loom_serve_program_process(loom_serve_program_t* program) {
  return program->process;
}

iree_vm_invocation_t* loom_serve_program_invocation(
    loom_serve_program_t* program) {
  return program->invocation;
}
