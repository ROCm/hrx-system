// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_OPTION_CHAIN_H_
#define LOOMC_OPTION_CHAIN_H_

#include "loom/sanitizer/options.h"
#include "loomc/sanitizer.h"
#include "loomc/target.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

enum loomc_option_chain_allowed_bit_e {
  LOOMC_OPTION_CHAIN_ALLOW_TARGET_SELECTION = 1u << 0,
  LOOMC_OPTION_CHAIN_ALLOW_SANITIZER = 1u << 1,
};
typedef uint32_t loomc_option_chain_allowed_t;

typedef struct loomc_option_chain_t {
  // Target selection descriptor found in the option chain, or NULL.
  loomc_target_selection_t* target_selection;
  // True when a sanitizer descriptor was present in the option chain.
  bool has_sanitizer;
  // Sanitizer options found in the option chain.
  loom_sanitizer_options_t sanitizer;
} loomc_option_chain_t;

// Resolves a public option-extension chain into the descriptors allowed by
// |allowed_options|. Known descriptors outside that set and unknown descriptors
// fail loudly.
LOOMC_API_PRIVATE loomc_status_t loomc_option_chain_resolve(
    const void* next, loomc_option_chain_allowed_t allowed_options,
    loomc_option_chain_t* out_options);

// Converts public sanitizer options into compiler-internal sanitizer options.
LOOMC_API_PRIVATE loomc_status_t
loomc_sanitizer_options_resolve(const loomc_sanitizer_options_t* options,
                                loom_sanitizer_options_t* out_options);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_OPTION_CHAIN_H_
