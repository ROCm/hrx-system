// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "target.h"

#include <string.h>

#include "iree/base/internal/atomics.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/pass/builtin_registry.h"
#include "loomc/iree.h"
#include "source.h"

struct loomc_target_environment_t {
  // Atomic reference count for shared immutable ownership.
  iree_atomic_ref_count_t ref_count;

  // Allocator used to release target-environment storage.
  loomc_allocator_t allocator;

  // Prepared target provider composition.
  loom_target_environment_t environment;

  // Prepared immutable pass capability tables over environment.
  loomc_target_pass_environment_t pass_environment;
};

struct loomc_target_profile_t {
  // Atomic reference count for shared immutable ownership.
  iree_atomic_ref_count_t ref_count;

  // Allocator used to release target-profile storage.
  loomc_allocator_t allocator;

  // Target environment whose provider package understands this profile.
  loomc_target_environment_t* target_environment;

  // Copied stable identifier used in diagnostics and reports.
  loomc_string_view_t identifier;

  // Compiler-facing target selection produced by this profile.
  loom_target_selection_t selection;

  // Stable process-local payload type token.
  const void* payload_type;

  // Target-owned payload storage referenced by selection.data.
  void* payload;

  // Releases payload when the final profile reference is released.
  loomc_target_profile_payload_deinitialize_fn_t payload_deinitialize;
};

struct loomc_target_selection_t {
  // Atomic reference count for shared immutable ownership.
  iree_atomic_ref_count_t ref_count;

  // Allocator used to release target-selection storage.
  loomc_allocator_t allocator;

  // Profile retained by this selection, or NULL for an empty selection.
  loomc_target_profile_t* profile;

  // Compiler-facing target selection snapshot.
  loom_target_selection_t selection;
};

typedef struct loomc_descriptor_prefix_t {
  // Structure type identifying the descriptor.
  loomc_structure_type_t type;

  // Size of the descriptor in bytes.
  loomc_host_size_t structure_size;

  // Next descriptor in the option extension chain.
  const void* next;
} loomc_descriptor_prefix_t;

static loomc_status_t loomc_target_validate_string_view(
    loomc_string_view_t value) {
  if (value.data == NULL && value.size != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "string view has length but no data");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_context_target_options_validate(
    const loomc_context_target_options_t* options) {
  if (options->type != LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "context target options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "context target options structure_size is too small");
  }
  if (options->target_environment == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "context target options require a target environment");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_target_profile_options_validate(
    const loomc_target_profile_options_t* options) {
  if (options == NULL) {
    return loomc_ok_status();
  }
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_TARGET_PROFILE_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "target profile options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "target profile options structure_size is too small");
  }
  if (options->next != NULL) {
    return loomc_make_status(
        LOOMC_STATUS_UNIMPLEMENTED,
        "target profile option extensions are not supported");
  }
  return loomc_target_validate_string_view(options->identifier);
}

loomc_status_t loomc_target_selection_options_validate(
    const loomc_target_selection_options_t* options) {
  if (options->type != LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "target selection options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "target selection options structure_size is too small");
  }
  if (options->target_selection == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "target selection options require a target selection");
  }
  return loomc_ok_status();
}

static const loom_target_provider_set_t* loomc_target_environment_provider_set(
    const loomc_target_environment_t* target_environment) {
  return target_environment ? target_environment->environment.provider_set
                            : NULL;
}

static bool loomc_target_environment_is_compatible(
    const loomc_target_environment_t* target_environment,
    const loomc_target_environment_t* profile_environment) {
  if (profile_environment == NULL) {
    return true;
  }
  if (target_environment == NULL) {
    return false;
  }
  if (target_environment == profile_environment) {
    return true;
  }
  return loomc_target_environment_provider_set(target_environment) ==
         loomc_target_environment_provider_set(profile_environment);
}

static loomc_status_t loomc_target_pass_environment_initialize(
    const loomc_target_environment_t* target_environment,
    loomc_target_pass_environment_t* out_environment) {
  if (target_environment == NULL || out_environment == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "target_environment and out_environment must not be NULL");
  }
  *out_environment = (loomc_target_pass_environment_t){0};
  const loom_target_environment_t* internal_environment =
      &target_environment->environment;
  LOOMC_RETURN_IF_ERROR(loomc_status_from_iree(
      loom_target_environment_initialize_low_descriptor_registry(
          internal_environment, &out_environment->low_descriptor_registry)));
  LOOMC_RETURN_IF_ERROR(loomc_status_from_iree(
      loom_target_environment_initialize_low_lower_policy_registry(
          internal_environment, &out_environment->low_lower_policy_registry)));
  LOOMC_RETURN_IF_ERROR(loomc_status_from_iree(
      loom_target_environment_initialize_math_policy_registry(
          internal_environment, &out_environment->math_policy_registry)));
  out_environment->low_legality_provider_list =
      loom_target_environment_low_legality_provider_list(internal_environment);
  out_environment->legalizer_provider_list =
      loom_target_environment_legalizer_provider_list(internal_environment);
  return loomc_ok_status();
}

static loomc_status_t loomc_target_selection_allocate(
    loomc_allocator_t allocator,
    loomc_target_selection_t** out_target_selection) {
  *out_target_selection = NULL;
  loomc_target_selection_t* target_selection = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_allocator_malloc(
      allocator, sizeof(*target_selection), (void**)&target_selection));
  memset(target_selection, 0, sizeof(*target_selection));
  iree_atomic_ref_count_init(&target_selection->ref_count);
  target_selection->allocator = allocator;
  *out_target_selection = target_selection;
  return loomc_ok_status();
}

static void loomc_target_pass_environment_deinitialize(
    loomc_target_pass_environment_t* environment) {
  if (environment == NULL) {
    return;
  }
  *environment = (loomc_target_pass_environment_t){0};
}

static void loomc_target_profile_deinitialize_payload(
    void* payload,
    loomc_target_profile_payload_deinitialize_fn_t payload_deinitialize,
    loomc_allocator_t allocator) {
  if (payload == NULL || payload_deinitialize == NULL) {
    return;
  }
  payload_deinitialize(payload, allocator);
}

loomc_status_t loomc_target_environment_create_from_provider_set(
    const loom_target_provider_set_t* provider_set, loomc_allocator_t allocator,
    loomc_target_environment_t** out_target_environment) {
  if (out_target_environment == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_target_environment must not be NULL");
  }
  *out_target_environment = NULL;
  if (provider_set == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "provider_set must not be NULL");
  }

  loomc_target_environment_t* target_environment = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_allocator_malloc(
      allocator, sizeof(*target_environment), (void**)&target_environment));
  memset(target_environment, 0, sizeof(*target_environment));
  iree_atomic_ref_count_init(&target_environment->ref_count);
  target_environment->allocator = allocator;

  loomc_status_t status =
      loomc_status_from_iree(loom_target_environment_initialize(
          provider_set, &target_environment->environment));
  if (loomc_status_is_ok(status)) {
    status = loomc_target_pass_environment_initialize(
        target_environment, &target_environment->pass_environment);
  }
  if (loomc_status_is_ok(status)) {
    *out_target_environment = target_environment;
  } else {
    loomc_target_pass_environment_deinitialize(
        &target_environment->pass_environment);
    loom_target_environment_deinitialize(&target_environment->environment);
    loomc_allocator_free(allocator, target_environment);
  }
  return status;
}

const loom_target_environment_t*
loomc_target_environment_loom_target_environment(
    const loomc_target_environment_t* target_environment) {
  return target_environment ? &target_environment->environment : NULL;
}

const loomc_target_pass_environment_t*
loomc_target_environment_pass_environment(
    const loomc_target_environment_t* target_environment) {
  return target_environment ? &target_environment->pass_environment : NULL;
}

loomc_status_t loomc_target_environment_register_context(
    const loomc_target_environment_t* target_environment,
    loom_context_t* context) {
  if (target_environment == NULL || context == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "target_environment and context must not be NULL");
  }
  return loomc_status_from_iree(loom_target_environment_register_context(
      &target_environment->environment, context));
}

loomc_status_t loomc_context_target_options_resolve(
    const loomc_context_options_t* options,
    loomc_target_environment_t** out_target_environment) {
  *out_target_environment = NULL;
  const void* next = options ? options->next : NULL;
  while (next != NULL) {
    const loomc_descriptor_prefix_t* prefix =
        (const loomc_descriptor_prefix_t*)next;
    switch (prefix->type) {
      case LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS: {
        if (*out_target_environment != NULL) {
          return loomc_make_status(
              LOOMC_STATUS_INVALID_ARGUMENT,
              "context options contain duplicate target options");
        }
        const loomc_context_target_options_t* target_options =
            (const loomc_context_target_options_t*)next;
        LOOMC_RETURN_IF_ERROR(
            loomc_context_target_options_validate(target_options));
        *out_target_environment = target_options->target_environment;
        next = target_options->next;
        break;
      }
      case LOOMC_STRUCTURE_TYPE_NONE:
        return loomc_make_status(
            LOOMC_STATUS_INVALID_ARGUMENT,
            "context option extension is missing a structure type");
      default:
        return loomc_make_status(
            LOOMC_STATUS_UNIMPLEMENTED,
            "context option extension type is not supported");
    }
  }
  return loomc_ok_status();
}

loomc_status_t loomc_target_selection_options_resolve(
    const void* next, loomc_target_selection_t** out_target_selection) {
  if (out_target_selection == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_target_selection must not be NULL");
  }
  *out_target_selection = NULL;
  while (next != NULL) {
    const loomc_descriptor_prefix_t* prefix =
        (const loomc_descriptor_prefix_t*)next;
    switch (prefix->type) {
      case LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS: {
        if (*out_target_selection != NULL) {
          return loomc_make_status(
              LOOMC_STATUS_INVALID_ARGUMENT,
              "option chain contains duplicate target selection options");
        }
        const loomc_target_selection_options_t* target_options =
            (const loomc_target_selection_options_t*)next;
        LOOMC_RETURN_IF_ERROR(
            loomc_target_selection_options_validate(target_options));
        *out_target_selection = target_options->target_selection;
        next = target_options->next;
        break;
      }
      case LOOMC_STRUCTURE_TYPE_NONE:
        return loomc_make_status(
            LOOMC_STATUS_INVALID_ARGUMENT,
            "option extension is missing a structure type");
      default:
        return loomc_make_status(LOOMC_STATUS_UNIMPLEMENTED,
                                 "option extension type is not supported");
    }
  }
  return loomc_ok_status();
}

loomc_status_t loomc_target_selection_validate_environment(
    const loomc_target_selection_t* target_selection,
    const loomc_target_environment_t* target_environment) {
  if (target_selection == NULL || target_selection->profile == NULL) {
    return loomc_ok_status();
  }
  const loomc_target_environment_t* profile_environment =
      target_selection->profile->target_environment;
  if (loomc_target_environment_is_compatible(target_environment,
                                             profile_environment)) {
    return loomc_ok_status();
  }
  return loomc_make_status(
      LOOMC_STATUS_INVALID_ARGUMENT,
      "target selection was created for an incompatible target environment");
}

loom_target_selection_t loomc_target_selection_loom_target_selection(
    const loomc_target_selection_t* target_selection) {
  return target_selection ? target_selection->selection
                          : loom_target_selection_empty();
}

loomc_status_t loomc_target_pass_registry_initialize(
    const loomc_target_environment_t* target_environment,
    loom_pass_registry_storage_t* out_storage,
    const loom_pass_registry_t** out_registry) {
  if (out_storage == NULL || out_registry == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_storage and out_registry must not be NULL");
  }
  *out_storage = (loom_pass_registry_storage_t){0};
  *out_registry = NULL;
  const loom_pass_registry_t* registries[2] = {
      loom_pass_builtin_registry(),
  };
  iree_host_size_t registry_count = 1;
  if (target_environment != NULL) {
    registries[registry_count++] =
        loom_target_environment_pass_registry(&target_environment->environment);
  }
  LOOMC_RETURN_IF_ERROR(loomc_status_from_iree(
      loom_pass_registry_storage_initialize_from_registries(
          registries, registry_count, out_storage)));
  *out_registry = loom_pass_registry_storage_registry(out_storage);
  return loomc_ok_status();
}

loom_pass_environment_t
loomc_target_pass_environment_make_loom_pass_environment(
    const loomc_target_pass_environment_t* environment,
    loom_target_selection_t target_selection, loom_symbol_ref_t target_ref,
    loom_low_pass_environment_storage_t* out_storage) {
  return loom_low_pass_environment_storage_initialize(
      &environment->low_descriptor_registry.registry,
      &environment->low_lower_policy_registry,
      &environment->low_legality_provider_list,
      &environment->legalizer_provider_list, &environment->math_policy_registry,
      NULL, target_selection, target_ref, out_storage);
}

void loomc_target_pass_environment_initialize_text_asm_environment(
    const loomc_target_pass_environment_t* environment,
    loom_text_low_asm_environment_t* out_environment) {
  if (out_environment == NULL) {
    return;
  }
  if (environment == NULL) {
    *out_environment = (loom_text_low_asm_environment_t){0};
    return;
  }
  loom_low_descriptor_text_asm_environment_initialize(
      &environment->low_descriptor_registry.registry, out_environment);
}

void loomc_target_environment_retain(
    loomc_target_environment_t* target_environment) {
  if (target_environment == NULL) {
    return;
  }
  iree_atomic_ref_count_inc(&target_environment->ref_count);
}

void loomc_target_environment_release(
    loomc_target_environment_t* target_environment) {
  if (target_environment == NULL) {
    return;
  }
  if (iree_atomic_ref_count_dec(&target_environment->ref_count) != 1) {
    return;
  }
  loomc_allocator_t allocator = target_environment->allocator;
  loomc_target_pass_environment_deinitialize(
      &target_environment->pass_environment);
  loom_target_environment_deinitialize(&target_environment->environment);
  loomc_allocator_free(allocator, target_environment);
}

loomc_status_t loomc_target_profile_create_empty(
    loomc_target_environment_t* target_environment,
    const loomc_target_profile_options_t* options, loomc_allocator_t allocator,
    loomc_target_profile_t** out_profile) {
  return loomc_target_profile_create_from_selection(
      target_environment, options, loom_target_selection_empty(), NULL, NULL,
      NULL, allocator, out_profile);
}

loomc_status_t loomc_target_profile_create_from_selection(
    loomc_target_environment_t* target_environment,
    const loomc_target_profile_options_t* options,
    loom_target_selection_t selection, const void* payload_type, void* payload,
    loomc_target_profile_payload_deinitialize_fn_t deinitialize,
    loomc_allocator_t allocator, loomc_target_profile_t** out_profile) {
  void* pending_payload = payload;
  loomc_target_profile_t* profile = NULL;
  loomc_status_t status = loomc_ok_status();

  if (out_profile == NULL) {
    status = loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                               "out_profile must not be NULL");
  } else {
    *out_profile = NULL;
  }
  if (loomc_status_is_ok(status) && target_environment == NULL) {
    status = loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                               "target_environment must not be NULL");
  }
  if (loomc_status_is_ok(status) && pending_payload != NULL &&
      (payload_type == NULL || deinitialize == NULL)) {
    status = loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "profile payload requires a payload type and deinitializer");
  }
  if (loomc_status_is_ok(status)) {
    status = loomc_target_profile_options_validate(options);
  }
  if (loomc_status_is_ok(status)) {
    status =
        loomc_allocator_malloc(allocator, sizeof(*profile), (void**)&profile);
  }
  if (loomc_status_is_ok(status)) {
    memset(profile, 0, sizeof(*profile));
    iree_atomic_ref_count_init(&profile->ref_count);
    profile->allocator = allocator;
    profile->target_environment = target_environment;
    loomc_target_environment_retain(profile->target_environment);
    profile->selection = selection;
    profile->payload_type = payload_type;
    profile->payload = pending_payload;
    profile->payload_deinitialize = deinitialize;
    pending_payload = NULL;
    status = loomc_string_view_clone(
        options ? options->identifier : loomc_string_view_empty(), allocator,
        &profile->identifier);
  }
  if (loomc_status_is_ok(status) && out_profile != NULL) {
    *out_profile = profile;
    profile = NULL;
  } else {
    loomc_target_profile_release(profile);
    loomc_target_profile_deinitialize_payload(pending_payload, deinitialize,
                                              allocator);
  }
  return status;
}

loom_target_selection_t loomc_target_profile_loom_target_selection(
    const loomc_target_profile_t* profile) {
  return profile ? profile->selection : loom_target_selection_empty();
}

loomc_target_environment_t* loomc_target_profile_target_environment(
    const loomc_target_profile_t* profile) {
  return profile ? profile->target_environment : NULL;
}

loomc_string_view_t loomc_target_profile_identifier(
    const loomc_target_profile_t* profile) {
  return profile ? profile->identifier : loomc_string_view_empty();
}

const void* loomc_target_profile_payload(const loomc_target_profile_t* profile,
                                         const void* payload_type) {
  if (profile == NULL || profile->payload_type != payload_type) {
    return NULL;
  }
  return profile->payload;
}

void loomc_target_profile_retain(loomc_target_profile_t* profile) {
  if (profile == NULL) {
    return;
  }
  iree_atomic_ref_count_inc(&profile->ref_count);
}

void loomc_target_profile_release(loomc_target_profile_t* profile) {
  if (profile == NULL) {
    return;
  }
  if (iree_atomic_ref_count_dec(&profile->ref_count) != 1) {
    return;
  }
  loomc_allocator_t allocator = profile->allocator;
  if (profile->payload != NULL && profile->payload_deinitialize != NULL) {
    profile->payload_deinitialize(profile->payload, allocator);
  }
  loomc_target_environment_release(profile->target_environment);
  loomc_allocator_free(allocator, (void*)profile->identifier.data);
  loomc_allocator_free(allocator, profile);
}

loomc_status_t loomc_target_selection_create_empty(
    loomc_allocator_t allocator,
    loomc_target_selection_t** out_target_selection) {
  if (out_target_selection == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_target_selection must not be NULL");
  }
  loomc_target_selection_t* target_selection = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_target_selection_allocate(allocator, &target_selection));
  target_selection->selection = loom_target_selection_empty();
  *out_target_selection = target_selection;
  return loomc_ok_status();
}

loomc_status_t loomc_target_selection_create_from_profile(
    loomc_target_profile_t* profile, loomc_allocator_t allocator,
    loomc_target_selection_t** out_target_selection) {
  if (out_target_selection == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_target_selection must not be NULL");
  }
  *out_target_selection = NULL;
  if (profile == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "profile must not be NULL");
  }
  loomc_target_selection_t* target_selection = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_target_selection_allocate(allocator, &target_selection));
  target_selection->profile = profile;
  loomc_target_profile_retain(target_selection->profile);
  target_selection->selection = profile->selection;
  *out_target_selection = target_selection;
  return loomc_ok_status();
}

void loomc_target_selection_retain(loomc_target_selection_t* target_selection) {
  if (target_selection == NULL) {
    return;
  }
  iree_atomic_ref_count_inc(&target_selection->ref_count);
}

void loomc_target_selection_release(
    loomc_target_selection_t* target_selection) {
  if (target_selection == NULL) {
    return;
  }
  if (iree_atomic_ref_count_dec(&target_selection->ref_count) != 1) {
    return;
  }
  loomc_allocator_t allocator = target_selection->allocator;
  loomc_target_profile_release(target_selection->profile);
  loomc_allocator_free(allocator, target_selection);
}
