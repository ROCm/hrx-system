// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/abi_v1.h"

#include "iree/vm/sync.h"
#include "loom/runtime/program/package.h"
#include "loom/serve/service/text/abi_v1_data.h"

static iree_status_t loom_serve_text_v1_compile(
    loom_runtime_jit_t* jit, iree_vm_environment_t* environment,
    iree_allocator_t host_allocator, loom_runtime_program_t** out_program) {
  const iree_file_toc_t* entry = loom_serve_text_v1_data_create();
  const loom_runtime_package_file_t file = {
      .path = iree_make_cstring_view(entry->name),
      .contents = iree_make_const_byte_span(entry->data, entry->size),
  };
  loom_runtime_package_t* package = NULL;
  IREE_RETURN_IF_ERROR(loom_runtime_package_create(file.path, 1, &file,
                                                   host_allocator, &package));
  const loom_runtime_sources_t sources = {.package = package};
  const iree_string_view_t roots[] = {IREE_SVL("catalog"), IREE_SVL("packets")};
  iree_vm_module_t* module = NULL;
  iree_status_t status = loom_runtime_program_compile_service_cached(
      jit, environment, &sources, file.path, IREE_ARRAYSIZE(roots), roots,
      (iree_string_view_list_t){0}, host_allocator, &module);
  if (iree_status_is_ok(status)) {
    status = loom_runtime_program_link(module, (iree_vm_module_span_t){0},
                                       host_allocator, out_program);
  }
  iree_vm_module_release(module);
  loom_runtime_package_release(package);
  return status;
}

iree_status_t loom_serve_text_v1_allocate(loom_runtime_jit_t* jit,
                                          iree_vm_environment_t* environment,
                                          const iree_vm_ref_types_t* types,
                                          int32_t token_capacity,
                                          iree_allocator_t host_allocator,
                                          loom_serve_text_v1_t** out_v1) {
  *out_v1 = NULL;
  loom_serve_text_v1_t* v1 = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*v1), (void**)&v1));
  v1->token_capacity = token_capacity;
  iree_status_t status = loom_serve_text_v1_compile(
      jit, environment, host_allocator, &v1->program);
  iree_vm_function_t catalog = {0};
  if (iree_status_is_ok(status)) {
    status =
        loom_runtime_program_lookup(v1->program, IREE_SV("catalog"), &catalog);
  }
  if (iree_status_is_ok(status)) {
    status = loom_runtime_program_lookup(v1->program, IREE_SV("packets"),
                                         &v1->packets);
  }
  iree_vm_variant_t results[2] = {0};
  if (iree_status_is_ok(status)) {
    iree_vm_variant_t limit = iree_vm_variant_from_i32(token_capacity);
    status = iree_vm_invoke(loom_runtime_program_invocation(v1->program),
                            catalog, iree_vm_variant_span_from_ptr(&limit, 1),
                            iree_vm_variant_span_from_array(results));
  }
  if (iree_status_is_ok(status)) {
    status =
        iree_vm_buffer_ptr_from_variant_move(types, &results[0], &v1->catalog);
  }
  if (iree_status_is_ok(status)) {
    status = iree_vm_i64_from_variant(results[1], &v1->catalog_count);
  }
  iree_vm_variant_span_reset(iree_vm_variant_span_from_array(results));
  if (iree_status_is_ok(status)) {
    *out_v1 = v1;
  } else {
    loom_serve_text_v1_free(v1, host_allocator);
  }
  return status;
}

void loom_serve_text_v1_free(loom_serve_text_v1_t* v1,
                             iree_allocator_t host_allocator) {
  if (!v1) {
    return;
  }
  iree_vm_buffer_release(v1->catalog);
  loom_runtime_program_free(v1->program);
  iree_allocator_free(host_allocator, v1);
}

iree_status_t loom_serve_text_v1_select_packets(
    loom_serve_text_v1_t* v1, int32_t tokens, int32_t spans,
    int32_t next_tokens, int32_t next_spans,
    loom_serve_text_packet_operands_t* first,
    loom_serve_text_packet_operands_t* next) {
  iree_vm_variant_t arguments[] = {iree_vm_variant_from_i32(v1->token_capacity),
                                   iree_vm_variant_from_i32(tokens),
                                   iree_vm_variant_from_i32(spans),
                                   iree_vm_variant_from_i32(next_tokens),
                                   iree_vm_variant_from_i32(next_spans)};
  iree_vm_variant_t results[4] = {0};
  IREE_RETURN_IF_ERROR(
      iree_vm_invoke(loom_runtime_program_invocation(v1->program), v1->packets,
                     iree_vm_variant_span_from_array(arguments),
                     iree_vm_variant_span_from_array(results)));
  IREE_RETURN_IF_ERROR(iree_vm_i32_from_variant(results[0], &first->workload));
  IREE_RETURN_IF_ERROR(iree_vm_i32_from_variant(results[1], &first->capacity));
  IREE_RETURN_IF_ERROR(iree_vm_i32_from_variant(results[2], &next->workload));
  return iree_vm_i32_from_variant(results[3], &next->capacity);
}
