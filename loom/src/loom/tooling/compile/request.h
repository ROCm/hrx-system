// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Allocation-free product and format request resolution.

#ifndef LOOM_TOOLING_COMPILE_REQUEST_H_
#define LOOM_TOOLING_COMPILE_REQUEST_H_

#include "iree/base/api.h"
#include "loom/ir/module.h"
#include "loom/tooling/compile/artifact.h"

#ifdef __cplusplus
extern "C" {
#endif

// Semantic product inferred from the selected module roots.
typedef enum loom_compile_product_e {
  LOOM_COMPILE_PRODUCT_INVALID = 0,
  LOOM_COMPILE_PRODUCT_KERNEL = 1,
  LOOM_COMPILE_PRODUCT_COMMAND = 2,
  LOOM_COMPILE_PRODUCT_MODULE = 3,
} loom_compile_product_t;

// Returns the stable public name of |product|.
iree_string_view_t loom_compile_product_name(loom_compile_product_t product);

// Concrete producer selected for one compile request.
typedef enum loom_compile_producer_kind_e {
  LOOM_COMPILE_PRODUCER_INVALID = 0,
  LOOM_COMPILE_PRODUCER_COMMAND = 1,
  LOOM_COMPILE_PRODUCER_ARTIFACT = 2,
  LOOM_COMPILE_PRODUCER_TARGET_EMITTER = 3,
} loom_compile_producer_kind_t;

typedef struct loom_compile_producer_t {
  // Active member of |value|.
  loom_compile_producer_kind_t kind;

  // Producer selected by |kind|.
  union {
    // Offline kernel artifact provider.
    const loom_artifact_provider_t* artifact_provider;
    // Target-owned diagnostic or intermediate emitter.
    const loom_target_emitter_t* target_emitter;
  } value;
} loom_compile_producer_t;

// User constraints applied while resolving one compilation request.
typedef struct loom_compile_request_options_t {
  // Explicit root names, or an empty list to use product root policy.
  iree_string_view_list_t roots;
  // Optional product selection or explicit-root assertion.
  iree_string_view_t product;
  // Optional exact artifact format.
  iree_string_view_t format;
  // Optional family-qualified target profile.
  iree_string_view_t target;
} loom_compile_request_options_t;

// Borrowed target selected for source specialization and artifact emission.
typedef struct loom_compile_target_t {
  // Immutable structured profile used to specialize source IR.
  const loom_target_profile_t* target_profile;
  // Family-owned selector used for diagnostics and artifact metadata.
  iree_string_view_t target_key;
} loom_compile_target_t;

// Returns the prepared-artifact target projected by |target|.
static inline loom_artifact_target_t loom_compile_target_artifact_target(
    const loom_compile_target_t* target) {
  loom_artifact_target_t artifact_target = {0};
  artifact_target.target_bundle =
      target ? loom_target_profile_bundle(target->target_profile) : NULL;
  artifact_target.target_key =
      target ? target->target_key : iree_string_view_empty();
  return artifact_target;
}

// Fully resolved compile request borrowing immutable configured state.
typedef struct loom_compile_request_t {
  // Product inferred from selected roots.
  loom_compile_product_t product;
  // Explicit root names, or an empty list when product root policy applies.
  iree_string_view_list_t roots;
  // Exact public artifact format.
  iree_string_view_t format;
  // Producer implementing |format| for |product|.
  loom_compile_producer_t producer;
  // Explicit target selected by --target, or empty for authored targets.
  loom_compile_target_t explicit_target;
  // Effective target fact type, or NULL for target-independent products.
  const loom_target_fact_type_t* target_fact_type;
} loom_compile_request_t;

// Returns the selected artifact provider, or NULL for other producers.
static inline const loom_artifact_provider_t*
loom_compile_request_artifact_provider(const loom_compile_request_t* request) {
  return request != NULL &&
                 request->producer.kind == LOOM_COMPILE_PRODUCER_ARTIFACT
             ? request->producer.value.artifact_provider
             : NULL;
}

// Returns true when portable command emission was selected.
static inline bool loom_compile_request_is_command(
    const loom_compile_request_t* request) {
  return request != NULL &&
         request->producer.kind == LOOM_COMPILE_PRODUCER_COMMAND;
}

// Resolves roots, product, target, format, and producer exactly once.
//
// All inputs and outputs are borrowed. Resolution performs no allocation and
// never probes a producer by compiling. With explicit roots, an explicit
// product only validates the inferred product and cannot reinterpret them. With
// no roots, an explicit product selects that product's canonical root policy.
// An omitted format selects the unique configured kernel artifact provider for
// the selected target or the target-independent command format. An optional
// |inferred_target_fact_type| supplies the target family selected by a
// surrounding execution environment when kernel roots and --target do not.
// It must agree with any authored or explicit target family.
iree_status_t loom_compile_request_resolve(
    const loom_module_t* module, const loom_compile_request_options_t* options,
    const loom_artifact_provider_registry_t* artifact_provider_registry,
    const loom_target_environment_t* target_environment,
    const loom_target_fact_type_t* inferred_target_fact_type,
    loom_compile_request_t* out_request);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_COMPILE_REQUEST_H_
