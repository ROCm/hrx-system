// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Open product-operation, format, and implementation registries.

#ifndef LOOM_PRODUCT_REGISTRY_H_
#define LOOM_PRODUCT_REGISTRY_H_

#include "iree/base/api.h"
#include "loom/ir/attribute_schema.h"
#include "loom/product/product.h"
#include "loom/target/pipeline_options.h"
#include "loom/target/product_contract.h"
#include "loom/target/profile.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_product_build_request_t loom_product_build_request_t;
typedef struct loom_product_format_provider_t loom_product_format_provider_t;

// Exact indexed-symbol classification accepted by one product operation.
//
// The carrier is an operation-local enum value. UNCLASSIFIED matches only
// symbol definitions without a carrier contract; it is not a wildcard.
typedef struct loom_product_root_match_t {
  // Public name of the operation defining the durable root.
  iree_string_view_t defining_op_name;

  // Exact product carrier or UNCLASSIFIED for non-polymorphic root ops.
  loom_symbol_product_carrier_t carrier;
} loom_product_root_match_t;

// Semantic compiler product and the indexed symbols that may define its roots.
//
// Product names are open configuration keys. Each complete root match must
// have exactly one owner in a registry; distinct carrier values allow a
// polymorphic operation such as pipeline.def to route to different products.
typedef struct loom_product_operation_t {
  // Public product name used by tools and build systems.
  iree_string_view_t name;

  // Process-local descriptor carried by products built by this operation.
  const loom_product_descriptor_t* product_descriptor;

  // Exact indexed-symbol classifications accepted as durable product roots.
  const loom_product_root_match_t* root_matches;

  // Number of entries in |root_matches|.
  iree_host_size_t root_match_count;
} loom_product_operation_t;

// On-disk persistence shape selected by a product format.
typedef enum loom_product_persistence_e {
  // Invalid or unspecified persistence shape.
  LOOM_PRODUCT_PERSISTENCE_UNKNOWN = 0,

  // One required payload artifact is written directly to a file or stdout.
  LOOM_PRODUCT_PERSISTENCE_SINGLE_FILE = 1,

  // The complete artifact table is written to a directory plus set manifest.
  LOOM_PRODUCT_PERSISTENCE_ARTIFACT_SET = 2,
} loom_product_persistence_t;

// Allowed artifact role, representation, and cardinality in a product format.
typedef struct loom_product_artifact_schema_t {
  // Semantic artifact role within the product.
  iree_string_view_t role;

  // Public representation and consumer-contract name for the artifact bytes.
  iree_string_view_t format;

  // Minimum number of artifacts with this role.
  iree_host_size_t minimum_count;

  // Maximum number of artifacts with this role, or IREE_HOST_SIZE_MAX.
  iree_host_size_t maximum_count;
} loom_product_artifact_schema_t;

// Public format contract for one semantic product.
//
// A format descriptor is shared by every target-specific implementation of
// the same product/format tuple. Build-system declarations mirror this
// persistence and artifact schema so native output can be checked rather than
// repackaged heuristically.
typedef struct loom_product_format_t {
  // Product operation supported by this format.
  const loom_product_operation_t* operation;

  // Public format name used by tools and build systems.
  iree_string_view_t name;

  // Persistence shape used to publish the artifact table.
  loom_product_persistence_t persistence;

  // Single-file persistence contract. Empty for artifact-set formats.
  struct {
    // Required payload role written to the selected output path.
    iree_string_view_t role;

    // Conventional filename extension, including its leading period.
    iree_string_view_t extension;
  } single_file;

  // Allowed artifact roles and cardinalities.
  const loom_product_artifact_schema_t* artifact_schemas;

  // Number of entries in |artifact_schemas|.
  iree_host_size_t artifact_schema_count;
} loom_product_format_t;

// Flags controlling product-format provider selection.
typedef uint32_t loom_product_format_provider_flags_t;
enum loom_product_format_provider_flag_bits_e {
  // This provider supplies the canonical format for its applicable targets.
  LOOM_PRODUCT_FORMAT_PROVIDER_CANONICAL = 1u << 0,
};

// Returns whether |provider| accepts a profile after its profile type matches.
//
// The callback is optional. It is useful when two implementations of the same
// format divide a target family by structured profile facts. Registration
// order never resolves overlapping predicates.
typedef bool (*loom_product_format_provider_accepts_target_fn_t)(
    const loom_product_format_provider_t* provider,
    const loom_target_profile_t* profile);

// Builds one immutable product from a fully planned compile request.
//
// Implementations return OK with a NULL product only when product diagnostics
// were emitted through the request's diagnostic sink. Infrastructure failures
// return a non-OK status. The request representation is owned by the compile
// front door and is shared by every registered implementation.
typedef iree_status_t (*loom_product_format_provider_build_fn_t)(
    const loom_product_format_provider_t* provider,
    const loom_product_build_request_t* request, loom_product_t** out_product);

// One native implementation of a product/format/target-profile tuple.
typedef struct loom_product_format_provider_t {
  // Stable implementation name used only in diagnostics and reports.
  iree_string_view_t name;

  // Semantic product built by this provider.
  const loom_product_operation_t* operation;

  // Public format produced by this provider.
  const loom_product_format_t* format;

  // Target-profile representation accepted by this provider, or NULL for a
  // target-neutral product.
  const loom_target_profile_type_t* target_profile_type;

  // Optional predicate refining target applicability within the profile type.
  loom_product_format_provider_accepts_target_fn_t accepts_target;

  // Selection behavior for this provider.
  loom_product_format_provider_flags_t flags;

  // Product-owned target lowering contract, or NULL for target-neutral
  // products that do not lower target-specialized functions.
  const loom_target_product_contract_t* target_product_contract;

  // Optional provider-owned default target pipeline options.
  const loom_target_pipeline_options_t* default_pipeline_options;

  // Product build callback.
  loom_product_format_provider_build_fn_t build;
} loom_product_format_provider_t;

// Immutable configured registry of product operations, formats, and providers.
typedef struct loom_product_registry_t {
  // Registered semantic product operations.
  struct {
    // Operation descriptor table.
    const loom_product_operation_t* const* values;

    // Number of entries in |values|.
    iree_host_size_t count;
  } operations;

  // Registered public product formats.
  struct {
    // Format descriptor table.
    const loom_product_format_t* const* values;

    // Number of entries in |values|.
    iree_host_size_t count;
  } formats;

  // Registered native format implementations.
  struct {
    // Provider descriptor table.
    const loom_product_format_provider_t* const* values;

    // Number of entries in |values|.
    iree_host_size_t count;
  } providers;
} loom_product_registry_t;

// Validates the complete configured registry.
//
// This is a cold startup check. It rejects duplicate public identities,
// ambiguous durable-root ownership, malformed persistence schemas, dangling
// descriptor references, and providers without implementations.
iree_status_t loom_product_registry_validate(
    const loom_product_registry_t* registry);

// Looks up a semantic product by exact public name, or returns NULL.
const loom_product_operation_t* loom_product_registry_lookup_operation(
    const loom_product_registry_t* registry, iree_string_view_t name);

// Returns whether |operation| accepts the exact indexed root classification.
bool loom_product_operation_matches_root(
    const loom_product_operation_t* operation,
    iree_string_view_t defining_op_name,
    loom_symbol_product_carrier_t product_carrier);

// Returns the unique product owning the exact root classification, or NULL.
const loom_product_operation_t* loom_product_registry_lookup_root_operation(
    const loom_product_registry_t* registry,
    iree_string_view_t defining_op_name,
    loom_symbol_product_carrier_t product_carrier);

// Looks up |name| within |operation|, or returns NULL.
const loom_product_format_t* loom_product_registry_lookup_format(
    const loom_product_registry_t* registry,
    const loom_product_operation_t* operation, iree_string_view_t name);

// Selects exactly one provider for a product, format, and target.
//
// An empty |format_name| selects the unique applicable canonical provider.
// A non-empty name selects that exact public format, including noncanonical
// formats. A NULL |target_profile| matches only target-neutral providers.
iree_status_t loom_product_registry_select_provider(
    const loom_product_registry_t* registry,
    const loom_product_operation_t* operation, iree_string_view_t format_name,
    const loom_target_profile_t* target_profile,
    const loom_product_format_provider_t** out_provider);

// Validates an emitted product against |format|'s descriptor and artifact
// role/cardinality schema.
iree_status_t loom_product_format_validate_product(
    const loom_product_format_t* format, const loom_product_t* product);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_PRODUCT_REGISTRY_H_
