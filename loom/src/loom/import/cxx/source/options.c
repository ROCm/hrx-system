// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/source/options.h"

bool loom_cxx_fp_model_parse(iree_string_view_t value,
                             loom_cxx_fp_model_t* out_model) {
  if (iree_string_view_equal(value, IREE_SV("precise"))) {
    *out_model = LOOM_CXX_FP_MODEL_PRECISE;
    return true;
  }
  if (iree_string_view_equal(value, IREE_SV("fast"))) {
    *out_model = LOOM_CXX_FP_MODEL_FAST;
    return true;
  }
  return false;
}

void loom_cxx_import_options_initialize(loom_cxx_import_options_t* options) {
  memset(options, 0, sizeof(*options));
  options->standard = IREE_SV("c++26");
  options->data_model = LOOM_CXX_DATA_MODEL_LP64;
  options->fp_model = LOOM_CXX_FP_MODEL_PRECISE;
}

iree_string_view_t loom_cxx_import_options_standard(
    const loom_cxx_import_options_t* options) {
  if (!iree_string_view_is_empty(options->standard)) {
    return options->standard;
  }
  // Hosted import reads the platform's C and C++ library headers, which cxx
  // parses at C++20 but not yet every C++26 libc++ header.
  return iree_all_bits_set(options->flags, LOOM_CXX_IMPORT_FLAG_HOSTED)
             ? IREE_SV("c++20")
             : IREE_SV("c++26");
}

iree_status_t loom_cxx_import_options_validate(
    const loom_cxx_import_options_t* options) {
  if (options->data_model < LOOM_CXX_DATA_MODEL_LP64 ||
      options->data_model > LOOM_CXX_DATA_MODEL_ILP32 ||
      options->fp_model < LOOM_CXX_FP_MODEL_PRECISE ||
      options->fp_model > LOOM_CXX_FP_MODEL_FAST ||
      (options->flags & ~LOOM_CXX_IMPORT_FLAG_ALL)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid C/C++ source configuration");
  }
  if ((options->flags & LOOM_CXX_IMPORT_FLAG_EXCEPTIONS) &&
      !(options->flags & LOOM_CXX_IMPORT_FLAG_HOSTED)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "the EXCEPTIONS import flag requires HOSTED");
  }
  if ((!options->include_paths && options->include_path_count) ||
      (!options->system_include_paths && options->system_include_path_count) ||
      (!options->defines && options->define_count) ||
      (!options->undefines && options->undefine_count) ||
      (!options->roots && options->root_count) ||
      (!options->kernel_refs && options->kernel_ref_count)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "nonempty import option arrays require storage");
  }
  return iree_ok_status();
}
