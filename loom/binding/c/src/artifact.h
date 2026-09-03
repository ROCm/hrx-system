// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_ARTIFACT_ADAPTER_H_
#define LOOMC_ARTIFACT_ADAPTER_H_

#include "loom/product/artifact.h"
#include "loomc/artifact.h"
#include "loomc/iree.h"

#ifdef __cplusplus
extern "C" {
#endif

// Projects a public artifact view into the core representation.
static inline loom_product_artifact_t loomc_artifact_to_product(
    const loomc_artifact_t* artifact) {
  return (loom_product_artifact_t){
      .role = iree_string_view_from_loomc(artifact->role),
      .format = iree_string_view_from_loomc(artifact->format),
      .identifier = iree_string_view_from_loomc(artifact->identifier),
      .contents = iree_byte_sequence_from_loomc(artifact->contents),
  };
}

// Projects a core artifact view into the public representation.
static inline loomc_artifact_t loomc_artifact_from_product(
    const loom_product_artifact_t* artifact) {
  return (loomc_artifact_t){
      .role = loomc_string_view_from_iree(artifact->role),
      .format = loomc_string_view_from_iree(artifact->format),
      .identifier = loomc_string_view_from_iree(artifact->identifier),
      .contents = loomc_byte_sequence_from_iree(artifact->contents),
  };
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_ARTIFACT_ADAPTER_H_
