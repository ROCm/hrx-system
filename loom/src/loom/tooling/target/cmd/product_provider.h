// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Portable command-program product provider.

#ifndef LOOM_TOOLING_TARGET_CMD_PRODUCT_PROVIDER_H_
#define LOOM_TOOLING_TARGET_CMD_PRODUCT_PROVIDER_H_

#include "loom/product/registry.h"
#include "loom/target/arch/cmd/artifact_builder.h"

#ifdef __cplusplus
extern "C" {
#endif

// Optional command-product build behavior supplied by an embedding.
typedef struct loom_cmd_product_build_options_t {
  // Sink receiving independent kernel compilation requests discovered while
  // preparing command programs.
  loom_cmd_program_kernel_request_sink_t kernel_request_sink;
} loom_cmd_product_build_options_t;

// Initializes command-product build options with no kernel request sink.
void loom_cmd_product_build_options_initialize(
    loom_cmd_product_build_options_t* out_options);

// Target-neutral portable command-program product implementation.
extern const loom_product_format_provider_t loom_cmd_product_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_TARGET_CMD_PRODUCT_PROVIDER_H_
