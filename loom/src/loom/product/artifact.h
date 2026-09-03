// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Immutable compiler-product artifacts.

#ifndef LOOM_PRODUCT_ARTIFACT_H_
#define LOOM_PRODUCT_ARTIFACT_H_

#include "iree/base/api.h"
#include "iree/base/byte_sequence.h"

#ifdef __cplusplus
extern "C" {
#endif

// Loadable or inspectable kernel payload.
#define LOOM_PRODUCT_ARTIFACT_ROLE_KERNEL "kernel"

// Portable command-program payload.
#define LOOM_PRODUCT_ARTIFACT_ROLE_COMMAND_PROGRAM "command-program"

// Loom source or bytecode module.
#define LOOM_PRODUCT_ARTIFACT_ROLE_MODULE "module"

// Compiled kernel launch-configuration program.
#define LOOM_PRODUCT_ARTIFACT_ROLE_LAUNCH_CONFIG "launch-config"

// Structured artifact manifest.
#define LOOM_PRODUCT_ARTIFACT_ROLE_ARTIFACT_MANIFEST "artifact-manifest"

// Structured compile report.
#define LOOM_PRODUCT_ARTIFACT_ROLE_COMPILE_REPORT "compile-report"

// Structured link-dependency report.
#define LOOM_PRODUCT_ARTIFACT_ROLE_LINK_DEPENDENCY_REPORT \
  "link-dependency-report"

// Target-owned human-readable listing.
#define LOOM_PRODUCT_ARTIFACT_ROLE_LISTING "listing"

// UTF-8 JSON document.
#define LOOM_PRODUCT_ARTIFACT_FORMAT_JSON "json"

// Borrowed immutable artifact owned by a compiler product or result.
//
// Roles and formats are open names. A role identifies how the artifact
// participates in its product while a format identifies the artifact's byte
// representation and consumer contract. Adding a product or format never
// extends a central artifact enum.
typedef struct loom_product_artifact_t {
  // Semantic role within the owning product.
  iree_string_view_t role;

  // Stable representation and consumer-contract name.
  iree_string_view_t format;

  // Human-readable product-local artifact identifier.
  iree_string_view_t identifier;

  // Immutable artifact bytes retained by the owner.
  iree_byte_sequence_t* contents;
} loom_product_artifact_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_PRODUCT_ARTIFACT_H_
