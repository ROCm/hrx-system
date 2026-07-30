// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target provider composition shared by tools and compile drivers.
//
// A target provider owns the target dialects, descriptor registries, lowering
// policies, and diagnostic providers linked into a binary. Tool-specific layers
// may add execution, checking, or artifact-emission providers around this core
// target contribution, but those layers should not duplicate target registry
// aggregation.

#ifndef LOOM_TARGET_PROVIDER_H_
#define LOOM_TARGET_PROVIDER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/lower/lower.h"
#include "loom/codegen/low/verify.h"
#include "loom/ir/context.h"
#include "loom/ir/ir.h"
#include "loom/ops/target/facts.h"
#include "loom/pass/environment.h"
#include "loom/pass/registry.h"
#include "loom/target/legalization.h"
#include "loom/target/low_asm_diagnostics.h"
#include "loom/target/low_descriptor_registry.h"
#include "loom/target/low_legality.h"
#include "loom/target/low_packet_diagnostics.h"
#include "loom/target/math_policy.h"
#include "loom/target/profile.h"
#include "loom/target/reporting/artifact_manifest.h"
#include "loom/target/reporting/report.h"

#ifdef __cplusplus
extern "C" {
#endif

// Registers target-owned dialects and encoding families.
typedef iree_status_t (*loom_target_provider_context_registration_fn_t)(
    loom_context_t* context);

// Initializes a target-low descriptor registry package.
typedef void (*loom_target_low_descriptor_registry_initializer_t)(
    loom_target_low_descriptor_registry_t* out_registry);

// Initializes a source-to-target-low lowering policy registry package.
typedef void (*loom_target_low_lower_policy_registry_initializer_t)(
    loom_low_lower_policy_registry_t* out_registry);

// Initializes a target math legalization policy registry package.
typedef void (*loom_target_math_policy_registry_initializer_t)(
    loom_target_math_policy_registry_t* out_registry);

typedef struct loom_builder_t loom_builder_t;
typedef struct loom_target_environment_t loom_target_environment_t;
typedef struct loom_target_provider_t loom_target_provider_t;

// Borrowed view of one indexed target record and its owning module. The module
// and symbol-fact arena must outlive the view and every query using it.
typedef struct loom_target_record_view_t {
  // Module that owns |facts|.
  const loom_module_t* module;

  // Indexed typed facts for the target record.
  const loom_target_symbol_facts_t* facts;
} loom_target_record_view_t;

// Creates a borrowed target record view.
static inline loom_target_record_view_t loom_target_record_view_make(
    const loom_module_t* module, const loom_target_symbol_facts_t* facts) {
  return (loom_target_record_view_t){
      /*.module=*/module,
      /*.facts=*/facts,
  };
}

// Returns true when |view| contains a target record.
static inline bool loom_target_record_view_is_valid(
    loom_target_record_view_t view) {
  return view.module != NULL && view.facts != NULL &&
         loom_target_like_isa(view.facts->target);
}

// Returns whether an effective target satisfies an authored target
// requirement.
//
// Identical record views satisfy by identity. Distinct records must have the
// same target op kind and a provider in |environment| that owns the relation.
// Views may belong to different modules. The query never mutates either
// record. |environment| may be NULL when only identity satisfaction is
// required.
bool loom_target_satisfies_requirement(
    const loom_target_environment_t* environment,
    loom_target_record_view_t effective_target,
    loom_target_record_view_t target_requirement);

// Returns whether |effective_snapshot| satisfies the structural requirements
// in |target_requirement|. Representation widths and address spaces must
// match, fixed subgroup sizes must agree, and effective capacity limits must
// meet or exceed nonzero required limits. Names, ABI/export plans, target
// configuration, and family identity are outside this structural comparison.
bool loom_target_snapshot_satisfies_requirement(
    const loom_target_snapshot_t* effective_snapshot,
    const loom_target_snapshot_t* target_requirement);

// Target-family satisfaction callback for distinct target records.
typedef bool (*loom_target_provider_satisfies_requirement_fn_t)(
    loom_target_record_view_t effective_target,
    loom_target_record_view_t target_requirement);

// Provider-owned semantics for one target-record op kind.
typedef struct loom_target_provider_record_semantics_t {
  // Target op kind whose records are owned by this provider.
  loom_op_kind_t op_kind;

  // Infallible satisfaction relation for distinct verified records of
  // |op_kind|. The indexed records are borrowed and may belong to different
  // modules.
  loom_target_provider_satisfies_requirement_fn_t satisfies_requirement;
} loom_target_provider_record_semantics_t;

// Returns the diagnostic symbol-name stem for a materialized profile record.
//
// The returned name is never used as semantic identity. Common materialization
// reuses an equal record under any name and uniquifies this stem when another
// symbol already occupies it.
typedef iree_string_view_t (
    *loom_target_provider_materialization_symbol_stem_fn_t)(
    const loom_target_profile_t* profile);

// Returns whether |target_op| has the same provider-owned durable projection
// as |profile| refined by |authored_target_op|.
//
// Common materialization calls this only for records whose op kind matches the
// provider's record semantics. The profile type is the provider's registered
// |profile_type|. |authored_target_op| is NULL for a targetless function.
typedef bool (*loom_target_provider_record_matches_effective_target_fn_t)(
    const loom_module_t* module, const loom_op_t* target_op,
    const loom_target_profile_t* profile, const loom_op_t* authored_target_op);

// Builds one provider-owned durable target record for |profile| refined by
// |authored_target_op|.
typedef iree_status_t (
    *loom_target_provider_build_effective_target_record_fn_t)(
    loom_builder_t* builder, const loom_target_profile_t* profile,
    const loom_op_t* authored_target_op, loom_symbol_ref_t symbol,
    loom_location_id_t location, loom_op_t** out_target_op);

// Provider-owned projection hooks used by common target materialization.
typedef struct loom_target_provider_materialization_t {
  // Produces an incidental symbol-name stem for a new target record.
  loom_target_provider_materialization_symbol_stem_fn_t symbol_stem;

  // Compares one existing record with a refined structured target profile.
  loom_target_provider_record_matches_effective_target_fn_t
      record_matches_effective_target;

  // Builds a record carrying the complete durable refined projection.
  loom_target_provider_build_effective_target_record_fn_t
      build_effective_target_record;
} loom_target_provider_materialization_t;

// Target emission artifact storage release callback.
typedef void (*loom_target_emit_artifact_release_fn_t)(
    void* storage, iree_allocator_t allocator);

typedef enum loom_target_emit_sidecar_artifact_kind_e {
  // Machine-readable artifact manifest for the primary artifact.
  LOOM_TARGET_EMIT_SIDECAR_ARTIFACT_KIND_ARTIFACT_MANIFEST = 0,
} loom_target_emit_sidecar_artifact_kind_t;

// One sidecar artifact produced beside a primary target artifact.
typedef struct loom_target_emit_sidecar_artifact_t {
  // Sidecar artifact kind.
  loom_target_emit_sidecar_artifact_kind_t kind;

  // Sidecar artifact identifier.
  iree_string_view_t identifier;

  // Borrowed view over sidecar artifact bytes.
  iree_const_byte_span_t contents;
} loom_target_emit_sidecar_artifact_t;

// One target artifact produced by an emitter.
typedef struct loom_target_emit_artifact_t {
  // Target-neutral artifact format produced by the emitter.
  loom_target_artifact_format_t target_artifact_format;

  // Borrowed view over emitted artifact bytes.
  iree_const_byte_span_t contents;

  // Optional emitter-owned sidecar artifacts.
  const loom_target_emit_sidecar_artifact_t* sidecars;

  // Number of entries in |sidecars|.
  iree_host_size_t sidecar_count;

  // Emitter-owned storage that keeps artifact bytes, sidecar descriptors, and
  // sidecar bytes alive until released.
  void* storage;

  // Optional callback that releases |storage|.
  loom_target_emit_artifact_release_fn_t release;
} loom_target_emit_artifact_t;

// Artifact manifest request passed to target-owned emitters.
typedef struct loom_target_emit_artifact_manifest_request_t {
  // Selected manifest detail mode.
  loom_target_artifact_manifest_mode_t mode;

  // Manifest sidecar artifact identifier.
  iree_string_view_t identifier;
} loom_target_emit_artifact_manifest_request_t;

// Emission request passed to a target-owned emitter.
typedef struct loom_target_emit_request_t {
  // Composed target environment receiving the emission request.
  const loom_target_environment_t* target_environment;

  // Target-low descriptor registry prepared from target providers.
  const loom_low_descriptor_registry_t* low_descriptor_registry;

  // Mutable module containing already-prepared target-low IR.
  loom_module_t* module;

  // Embedding-owned option chain borrowed for the duration of the call.
  const void* option_chain;

  // Stable artifact identifier requested by the caller.
  iree_string_view_t identifier;

  // Optional artifact manifest request.
  loom_target_emit_artifact_manifest_request_t artifact_manifest;

  // Optional caller-owned structured compile report to populate.
  loom_target_compile_report_t* compile_report;

  // Diagnostic emitter that reports target diagnostics into the operation
  // result.
  iree_diagnostic_emitter_t diagnostic_emitter;

  // Invocation-local scratch arena.
  iree_arena_allocator_t* scratch_arena;

  // Host allocator used for artifact storage that escapes the call.
  iree_allocator_t allocator;
} loom_target_emit_request_t;

// Emits one target artifact from prepared target-low IR.
typedef iree_status_t (*loom_target_emit_fn_t)(
    const loom_target_emit_request_t* request,
    loom_target_emit_artifact_t* out_artifact);

// Target-owned emission backend linked into a binary or embedding.
typedef struct loom_target_emitter_t {
  // Stable emitter name used in diagnostics.
  iree_string_view_t name;

  // Public artifact format string returned by binding layers.
  iree_string_view_t public_artifact_format;

  // Default artifact identifier used when the caller leaves it empty.
  iree_string_view_t default_identifier;

  // Target-neutral artifact format produced by this emitter.
  loom_target_artifact_format_t target_artifact_format;

  // Emission callback.
  loom_target_emit_fn_t emit;
} loom_target_emitter_t;

// Borrowed list of target-owned emitters.
typedef struct loom_target_emitter_list_t {
  // Emitter table.
  const loom_target_emitter_t* const* values;

  // Number of entries in |values|.
  iree_host_size_t count;
} loom_target_emitter_list_t;

// Creates a borrowed emitter list.
static inline loom_target_emitter_list_t loom_target_emitter_list_make(
    const loom_target_emitter_t* const* values, iree_host_size_t count) {
  return (loom_target_emitter_list_t){
      /*.values=*/values,
      /*.count=*/count,
  };
}

// Coarse phase hooks owned by a top-level compile pipeline builder. Target
// providers append pass IR into these hooks; they do not execute passes.
typedef enum loom_target_pipeline_phase_e {
  // Source/kernel normalization before source-to-low lowering.
  LOOM_TARGET_PIPELINE_PHASE_SOURCE_NORMALIZATION = 0,
  // Source-to-target-low lowering.
  LOOM_TARGET_PIPELINE_PHASE_SOURCE_TO_LOW = 1,
  // Target-owned cleanup for human-facing source-low asm artifacts.
  LOOM_TARGET_PIPELINE_PHASE_SOURCE_LOW_ARTIFACT_PREPARATION = 2,
  // Target ABI/resource materialization after source-to-low.
  LOOM_TARGET_PIPELINE_PHASE_TARGET_LOW_MATERIALIZATION = 3,
  // Target-low cleanup and operand-form preparation before emission.
  LOOM_TARGET_PIPELINE_PHASE_TARGET_LOW_PREPARATION = 4,
  LOOM_TARGET_PIPELINE_PHASE_COUNT_,
} loom_target_pipeline_phase_t;

typedef struct loom_target_pipeline_contribution_t {
  // Composed target environment receiving the contribution request.
  const loom_target_environment_t* target_environment;
  // Compile phase being built.
  loom_target_pipeline_phase_t phase;
  // Builder positioned in the caller-owned pass pipeline region for |phase|.
  loom_builder_t* builder;
  // Pass capabilities that the produced pass IR will verify and run with.
  loom_pass_environment_t pass_environment;
} loom_target_pipeline_contribution_t;

// Appends target-owned pass IR for one compile phase.
typedef iree_status_t (*loom_target_provider_pipeline_contribution_fn_t)(
    const loom_target_pipeline_contribution_t* contribution);

// Target-owned compiler capability contribution linked into a tool or driver.
struct loom_target_provider_t {
  // Target-family profile representation owned by this provider, or NULL when
  // the provider contributes no profile-driven semantics.
  const loom_target_profile_type_t* profile_type;
  // Optional function that registers target-owned dialects.
  loom_target_provider_context_registration_fn_t register_context;
  // Optional function that initializes target-low descriptor-set providers.
  loom_target_low_descriptor_registry_initializer_t
      initialize_low_descriptor_registry;
  // Optional function that initializes source-to-low lowering policies.
  loom_target_low_lower_policy_registry_initializer_t
      initialize_low_lower_policy_registry;
  // Optional function that initializes target math legalization policies.
  loom_target_math_policy_registry_initializer_t
      initialize_math_policy_registry;
  // Optional source legality providers contributed by this target.
  loom_target_low_legality_provider_list_t low_legality_provider_list;
  // Optional source legalization providers contributed by this target.
  loom_target_legalizer_provider_list_t legalizer_provider_list;
  // Optional low-packet diagnostic providers contributed by this target.
  loom_target_low_packet_diagnostic_provider_list_t
      low_packet_diagnostic_provider_list;
  // Optional text low-asm diagnostic providers contributed by this target.
  loom_target_low_asm_diagnostic_provider_list_t
      low_asm_diagnostic_provider_list;
  // Optional target-owned low verifier providers contributed by this target.
  loom_low_verify_provider_list_t low_verify_provider_list;
  // Optional target-owned emitters contributed by this target.
  loom_target_emitter_list_t emitter_list;
  // Optional target-owned pass descriptors contributed by this target.
  const loom_pass_registry_t* pass_registry;
  // Optional pass-pipeline contribution callback.
  loom_target_provider_pipeline_contribution_fn_t contribute_pipeline;
  // Optional structured target-record materialization hooks.
  loom_target_provider_materialization_t materialization;
  // Optional target-record semantics owned by this provider.
  loom_target_provider_record_semantics_t record_semantics;
};

// Static target provider table linked into a binary or embedding.
typedef struct loom_target_provider_set_t {
  // Target provider contribution table.
  const loom_target_provider_t* const* providers;
  // Number of entries in |providers|.
  iree_host_size_t provider_count;
} loom_target_provider_set_t;

enum {
  LOOM_TARGET_PROVIDER_DESCRIPTOR_SET_PROVIDER_CAPACITY = 256,
  LOOM_TARGET_PROVIDER_LOW_LOWER_POLICY_CAPACITY = 128,
  LOOM_TARGET_PROVIDER_MATH_POLICY_CAPACITY = 128,
  LOOM_TARGET_PROVIDER_LOW_LEGALITY_PROVIDER_CAPACITY = 64,
  LOOM_TARGET_PROVIDER_LEGALIZER_PROVIDER_CAPACITY = 64,
  LOOM_TARGET_PROVIDER_LOW_PACKET_DIAGNOSTIC_PROVIDER_CAPACITY = 64,
  LOOM_TARGET_PROVIDER_LOW_ASM_DIAGNOSTIC_PROVIDER_CAPACITY = 64,
  LOOM_TARGET_PROVIDER_LOW_VERIFY_PROVIDER_CAPACITY = 64,
  LOOM_TARGET_PROVIDER_EMITTER_CAPACITY = 64,
  LOOM_TARGET_PROVIDER_PASS_REGISTRY_CAPACITY = 64,
};

// Composed target environment derived from a target provider set.
struct loom_target_environment_t {
  // Borrowed provider table selected by the linked binary or embedding.
  const loom_target_provider_set_t* provider_set;
  // Descriptor-set provider table assembled once.
  loom_low_descriptor_set_provider_t descriptor_set_providers
      [LOOM_TARGET_PROVIDER_DESCRIPTOR_SET_PROVIDER_CAPACITY];
  // Number of entries in |descriptor_set_providers|.
  iree_host_size_t descriptor_set_provider_count;
  // Source-to-low policy table assembled once.
  loom_low_lower_policy_registry_entry_t
      low_lower_policy_entries[LOOM_TARGET_PROVIDER_LOW_LOWER_POLICY_CAPACITY];
  // Number of entries in |low_lower_policy_entries|.
  iree_host_size_t low_lower_policy_entry_count;
  // Target math policy table assembled once.
  loom_target_math_policy_registry_entry_t
      math_policy_entries[LOOM_TARGET_PROVIDER_MATH_POLICY_CAPACITY];
  // Number of entries in |math_policy_entries|.
  iree_host_size_t math_policy_entry_count;
  // Target-low source legality provider table assembled once.
  const loom_target_low_legality_provider_t* low_legality_providers
      [LOOM_TARGET_PROVIDER_LOW_LEGALITY_PROVIDER_CAPACITY];
  // Number of entries in |low_legality_providers|.
  iree_host_size_t low_legality_provider_count;
  // Target legalizer provider table assembled once.
  const loom_target_legalizer_provider_t*
      legalizer_providers[LOOM_TARGET_PROVIDER_LEGALIZER_PROVIDER_CAPACITY];
  // Number of entries in |legalizer_providers|.
  iree_host_size_t legalizer_provider_count;
  // Target-low packet diagnostic provider table assembled once.
  const loom_target_low_packet_diagnostic_provider_t*
      low_packet_diagnostic_providers
          [LOOM_TARGET_PROVIDER_LOW_PACKET_DIAGNOSTIC_PROVIDER_CAPACITY];
  // Number of entries in |low_packet_diagnostic_providers|.
  iree_host_size_t low_packet_diagnostic_provider_count;
  // Text low-asm diagnostic provider table assembled once.
  const loom_target_low_asm_diagnostic_provider_t* low_asm_diagnostic_providers
      [LOOM_TARGET_PROVIDER_LOW_ASM_DIAGNOSTIC_PROVIDER_CAPACITY];
  // Number of entries in |low_asm_diagnostic_providers|.
  iree_host_size_t low_asm_diagnostic_provider_count;
  // Target-owned low verifier provider table assembled once.
  const loom_low_verify_provider_t*
      low_verify_providers[LOOM_TARGET_PROVIDER_LOW_VERIFY_PROVIDER_CAPACITY];
  // Number of entries in |low_verify_providers|.
  iree_host_size_t low_verify_provider_count;
  // Target-owned emitter table assembled once.
  const loom_target_emitter_t* emitters[LOOM_TARGET_PROVIDER_EMITTER_CAPACITY];
  // Number of entries in |emitters|.
  iree_host_size_t emitter_count;
  // Composed target-owned pass registry storage.
  loom_pass_registry_storage_t pass_registry_storage;
};

// Creates a borrowed provider set view.
static inline loom_target_provider_set_t loom_target_provider_set_make(
    const loom_target_provider_t* const* providers,
    iree_host_size_t provider_count) {
  return (loom_target_provider_set_t){
      /*.providers=*/providers,
      /*.provider_count=*/provider_count,
  };
}

// Initializes |out_environment| from |provider_set|. The environment borrows
// |provider_set| until deinitialized.
iree_status_t loom_target_environment_initialize(
    const loom_target_provider_set_t* provider_set,
    loom_target_environment_t* out_environment);

// Resets |environment| to an empty state. No provider-owned storage is freed.
void loom_target_environment_deinitialize(
    loom_target_environment_t* environment);

// Registers every target dialect contributed by |environment|.
iree_status_t loom_target_environment_register_context(
    const loom_target_environment_t* environment, loom_context_t* context);

// Initializes a composed target-low descriptor registry package.
iree_status_t loom_target_environment_initialize_low_descriptor_registry(
    const loom_target_environment_t* environment,
    loom_target_low_descriptor_registry_t* out_registry);

// Initializes a composed source-to-target-low lowering policy registry package.
iree_status_t loom_target_environment_initialize_low_lower_policy_registry(
    const loom_target_environment_t* environment,
    loom_low_lower_policy_registry_t* out_registry);

// Initializes a composed target math legalization policy registry package.
iree_status_t loom_target_environment_initialize_math_policy_registry(
    const loom_target_environment_t* environment,
    loom_target_math_policy_registry_t* out_registry);

// Returns target-low source legality providers linked into |environment|.
loom_target_low_legality_provider_list_t
loom_target_environment_low_legality_provider_list(
    const loom_target_environment_t* environment);

// Returns target legalizer providers linked into |environment|.
loom_target_legalizer_provider_list_t
loom_target_environment_legalizer_provider_list(
    const loom_target_environment_t* environment);

// Returns target-low packet diagnostic providers linked into |environment|.
loom_target_low_packet_diagnostic_provider_list_t
loom_target_environment_low_packet_diagnostic_provider_list(
    const loom_target_environment_t* environment);

// Returns target text low-asm diagnostic providers linked into |environment|.
loom_target_low_asm_diagnostic_provider_list_t
loom_target_environment_low_asm_diagnostic_provider_list(
    const loom_target_environment_t* environment);

// Returns target-owned low verifier providers linked into |environment|.
loom_low_verify_provider_list_t
loom_target_environment_low_verify_provider_list(
    const loom_target_environment_t* environment);

// Returns target-owned emitters linked into |environment|.
loom_target_emitter_list_t loom_target_environment_emitter_list(
    const loom_target_environment_t* environment);

// Returns target-owned pass descriptors linked into |environment|.
const loom_pass_registry_t* loom_target_environment_pass_registry(
    const loom_target_environment_t* environment);

// Invokes target-provider pass-pipeline contributions for |phase|. The caller
// owns phase ordering, surrounding pass.for/pass.where scopes, and global
// cleanup insertion.
iree_status_t loom_target_environment_contribute_pipeline(
    const loom_target_environment_t* environment,
    loom_target_pipeline_phase_t phase,
    loom_pass_environment_t pass_environment, loom_builder_t* builder);

// Materializes the effective target formed by refining |target_profile| with
// |authored_target_op| into |module| using the provider owning its profile
// type.
//
// An existing provider-owned record with the same complete durable projection
// is reused regardless of its symbol name. Otherwise a new record receives a
// collision-free incidental symbol name. |authored_target_op| may be NULL for
// a targetless function.
iree_status_t loom_target_environment_materialize_effective_target(
    const loom_target_environment_t* environment, loom_module_t* module,
    const loom_target_profile_t* target_profile,
    const loom_op_t* authored_target_op, loom_symbol_ref_t* out_target_ref);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_PROVIDER_H_
