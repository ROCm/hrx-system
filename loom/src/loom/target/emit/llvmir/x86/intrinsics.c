// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/llvmir/x86/intrinsics.h"

iree_status_t loom_llvmir_declare_x86_rdtsc(
    loom_llvmir_module_t* module, loom_llvmir_function_t** out_function) {
  loom_llvmir_type_id_t i64_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_llvmir_module_get_integer_type(module, 64, &i64_type));
  return loom_llvmir_module_add_function(
      module,
      &(loom_llvmir_function_desc_t){
          .kind = LOOM_LLVMIR_FUNCTION_DECLARATION,
          .name = IREE_SV("llvm.x86.rdtsc"),
          .return_type = i64_type,
          .linkage = LOOM_LLVMIR_LINKAGE_DEFAULT,
          .calling_convention = LOOM_LLVMIR_CALLING_CONVENTION_DEFAULT,
          .attr_group_id = LOOM_LLVMIR_ATTR_GROUP_ID_INVALID,
      },
      out_function);
}

iree_status_t loom_llvmir_declare_x86_sse2_pause(
    loom_llvmir_module_t* module, loom_llvmir_function_t** out_function) {
  loom_llvmir_type_id_t void_type = LOOM_LLVMIR_TYPE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_llvmir_module_get_void_type(module, &void_type));
  return loom_llvmir_module_add_function(
      module,
      &(loom_llvmir_function_desc_t){
          .kind = LOOM_LLVMIR_FUNCTION_DECLARATION,
          .name = IREE_SV("llvm.x86.sse2.pause"),
          .return_type = void_type,
          .linkage = LOOM_LLVMIR_LINKAGE_DEFAULT,
          .calling_convention = LOOM_LLVMIR_CALLING_CONVENTION_DEFAULT,
          .attr_group_id = LOOM_LLVMIR_ATTR_GROUP_ID_INVALID,
      },
      out_function);
}
