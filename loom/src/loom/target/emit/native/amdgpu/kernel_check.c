// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/kernel_check.h"

#include "loom/tools/loom-check/compile.h"
#include "loomc/iree.h"

static bool loom_amdgpu_kernel_check_matches(
    const loom_check_emit_provider_t* provider,
    iree_string_view_t target_name) {
  (void)provider;
  return iree_string_view_equal(target_name,
                                IREE_SV("amdgpu-kernel-assembly")) ||
         iree_string_view_equal(target_name, IREE_SV("amdgpu-kernel-asm"));
}

static loomc_status_t loom_amdgpu_kernel_check_append_artifact_segment(
    void* user_data, loomc_byte_span_t segment) {
  return loomc_status_from_iree(iree_string_builder_append_string(
      (iree_string_builder_t*)user_data,
      iree_make_string_view((const char*)segment.data, segment.data_length)));
}

static iree_status_t loom_amdgpu_kernel_check_execute(
    const loom_check_emit_provider_t* provider,
    const loom_check_emit_provider_request_t* request) {
  (void)provider;
  const loomc_emit_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
      .structure_size = sizeof(options),
      .artifact_format = loomc_make_cstring_view("amdgpu-hsaco"),
      .artifact_flags = LOOMC_EMIT_ARTIFACT_FLAG_TARGET_LISTING,
  };
  loomc_result_t* result = NULL;
  iree_status_t status = loom_check_emit_artifact(request, &options, &result);
  const loomc_artifact_t* listing = NULL;
  if (iree_status_is_ok(status) && result != NULL) {
    status = loom_check_compile_select_artifact(
        result, LOOMC_ARTIFACT_KIND_TEXT, IREE_SV("amdgpu-assembly"), &listing);
  }
  if (iree_status_is_ok(status) && listing != NULL) {
    status = iree_status_from_loomc(loomc_byte_sequence_enumerate(
        listing->contents,
        (loomc_byte_sequence_callback_t){
            .fn = loom_amdgpu_kernel_check_append_artifact_segment,
            .user_data = &request->result->actual_output,
        }));
  }
  loomc_result_release(result);
  return status;
}

static iree_status_t loom_amdgpu_kernel_check_append_names(
    const loom_check_emit_provider_t* provider,
    iree_string_builder_t* builder) {
  (void)provider;
  return iree_string_builder_append_cstring(
      builder, "amdgpu-kernel-assembly, amdgpu-kernel-asm");
}

const loom_check_emit_provider_t loom_amdgpu_kernel_loom_check_emit_provider = {
    .name = IREE_SVL("amdgpu-kernel-assembly"),
    .match = loom_amdgpu_kernel_check_matches,
    .execute = loom_amdgpu_kernel_check_execute,
    .append_names = loom_amdgpu_kernel_check_append_names,
};
