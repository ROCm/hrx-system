// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "target.h"

#include <string.h>

#include "iree/base/internal/atomics.h"
#include "loom/codegen/low/repr.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/pass/builtin_registry.h"
#include "loomc/iree.h"
#include "option_chain.h"
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

  // Target-family profile carrying structured compiler target facts.
  loom_target_profile_t* target_profile;

  // Unique product contract exposed by the profile's target environment, or
  // NULL when product selection remains external.
  const loom_target_product_contract_t* product_contract;

  // Releases target_profile when the final public reference is released.
  loomc_target_profile_deinitialize_fn_t target_profile_deinitialize;
};

typedef struct loomc_descriptor_prefix_t {
  // Structure type identifying the descriptor.
  loomc_structure_type_t type;

  // Size of the descriptor in bytes.
  loomc_host_size_t structure_size;

  // Next descriptor in the option extension chain.
  const void* next;
} loomc_descriptor_prefix_t;

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

loomc_status_t loomc_target_specialization_options_validate(
    const loomc_target_specialization_options_t* options) {
  if (options->type != LOOMC_STRUCTURE_TYPE_TARGET_SPECIALIZATION_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "target specialization options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "target specialization options structure_size is too small");
  }
  if (options->specialization_count != 0 && options->specializations == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "target specialization count is nonzero but specializations is NULL");
  }
  if (options->target_binding_count != 0 && options->target_bindings == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "target binding count is nonzero but target_bindings is NULL");
  }
  for (loomc_host_size_t i = 0; i < options->specialization_count; ++i) {
    const loomc_target_specialization_t* specialization =
        &options->specializations[i];
    if ((specialization->function_symbol.data == NULL &&
         specialization->function_symbol.size != 0) ||
        loomc_string_view_is_empty(specialization->function_symbol)) {
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "target specialization function symbol must not be empty");
    }
    if (specialization->target_profile == NULL) {
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "target specialization requires a target profile");
    }
  }
  for (loomc_host_size_t i = 0; i < options->target_binding_count; ++i) {
    const loomc_target_binding_t* binding = &options->target_bindings[i];
    if ((binding->target_symbol.data == NULL &&
         binding->target_symbol.size != 0) ||
        loomc_string_view_is_empty(binding->target_symbol)) {
      return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                               "target binding symbol must not be empty");
    }
    if (binding->target_profile == NULL) {
      return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                               "target binding requires a target profile");
    }
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

static loomc_status_t loomc_target_specialization_validate_profile_environment(
    const loomc_target_environment_t* target_environment,
    const loomc_target_profile_t* profile, const char* incompatible_message,
    const char* incomplete_message) {
  if (!loomc_target_environment_is_compatible(target_environment,
                                              profile->target_environment)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             incompatible_message);
  }
  if (profile->target_profile == NULL ||
      profile->target_profile->target_bundle == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT, incomplete_message);
  }
  return loomc_ok_status();
}

static bool loomc_target_environment_supports_profile_type(
    const loomc_target_environment_t* target_environment,
    const loom_target_profile_type_t* profile_type) {
  const loom_target_provider_set_t* provider_set =
      loomc_target_environment_provider_set(target_environment);
  if (provider_set == NULL || profile_type == NULL) {
    return false;
  }
  for (iree_host_size_t i = 0; i < provider_set->provider_count; ++i) {
    if (provider_set->providers[i]->profile_type == profile_type) {
      return true;
    }
  }
  return false;
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
  out_environment->target_environment = internal_environment;
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
  LOOMC_RETURN_IF_ERROR(
      loomc_status_from_iree(loom_low_legalizer_registry_storage_initialize(
          loom_target_environment_legalizer_provider_list(internal_environment),
          iree_allocator_from_loomc(target_environment->allocator),
          &out_environment->legalizer_registry_storage)));
  return loomc_ok_status();
}

static void loomc_target_pass_environment_deinitialize(
    loomc_target_pass_environment_t* environment) {
  if (environment == NULL) {
    return;
  }
  loom_target_legalizer_registry_storage_deinitialize(
      &environment->legalizer_registry_storage);
  *environment = (loomc_target_pass_environment_t){0};
}

static void loomc_target_profile_deinitialize_target_profile(
    loom_target_profile_t* target_profile,
    loomc_target_profile_deinitialize_fn_t target_profile_deinitialize,
    loomc_allocator_t allocator) {
  if (target_profile == NULL || target_profile_deinitialize == NULL) {
    return;
  }
  target_profile_deinitialize(target_profile, allocator);
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

loomc_status_t loomc_target_specialization_options_resolve(
    const void* next,
    const loomc_target_specialization_options_t** out_options) {
  if (out_options == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_options must not be NULL");
  }
  loomc_option_chain_t options = {0};
  LOOMC_RETURN_IF_ERROR(loomc_option_chain_resolve(
      next, LOOMC_OPTION_CHAIN_ALLOW_TARGET_SPECIALIZATION, &options));
  *out_options = options.target_specialization;
  return loomc_ok_status();
}

loomc_status_t loomc_target_specialization_options_validate_environment(
    const loomc_target_specialization_options_t* options,
    const loomc_target_environment_t* target_environment) {
  if (options == NULL || (options->specialization_count == 0 &&
                          options->target_binding_count == 0)) {
    return loomc_ok_status();
  }
  if (target_environment == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_FAILED_PRECONDITION,
        "target specialization requires a context target environment");
  }
  for (loomc_host_size_t i = 0; i < options->specialization_count; ++i) {
    LOOMC_RETURN_IF_ERROR(
        loomc_target_specialization_validate_profile_environment(
            target_environment, options->specializations[i].target_profile,
            "target specialization profile was created for an incompatible "
            "target environment",
            "target specialization contains an incomplete target profile"));
  }
  for (loomc_host_size_t i = 0; i < options->target_binding_count; ++i) {
    LOOMC_RETURN_IF_ERROR(
        loomc_target_specialization_validate_profile_environment(
            target_environment, options->target_bindings[i].target_profile,
            "target binding profile was created for an incompatible target "
            "environment",
            "target binding contains an incomplete target profile"));
  }
  return loomc_ok_status();
}

loomc_status_t loomc_target_specialization_options_make_lists(
    const loomc_target_specialization_options_t* options,
    loomc_target_specialization_list_flags_t flags,
    iree_arena_allocator_t* arena,
    loom_target_specialization_request_list_t* out_requests,
    loom_target_declaration_binding_list_t* out_bindings) {
  *out_requests = (loom_target_specialization_request_list_t){0};
  *out_bindings = (loom_target_declaration_binding_list_t){0};
  if (options == NULL) {
    return loomc_ok_status();
  }

  if (options->specialization_count != 0) {
    loom_target_specialization_request_t* requests = NULL;
    LOOMC_RETURN_IF_ERROR(loomc_status_from_iree(
        iree_arena_allocate_array(arena, options->specialization_count,
                                  sizeof(*requests), (void**)&requests)));
    for (loomc_host_size_t i = 0; i < options->specialization_count; ++i) {
      const loomc_target_specialization_t* specialization =
          &options->specializations[i];
      requests[i] = (loom_target_specialization_request_t){
          .function_name =
              iree_string_view_from_loomc(specialization->function_symbol),
          .target_profile = loomc_target_profile_loom_target_profile(
              specialization->target_profile),
          .product_contract =
              iree_any_bit_set(
                  flags,
                  LOOMC_TARGET_SPECIALIZATION_LIST_FLAG_APPLY_PRODUCT_CONTRACT)
                  ? specialization->target_profile->product_contract
                  : NULL,
      };
    }
    *out_requests = (loom_target_specialization_request_list_t){
        .values = requests,
        .count = options->specialization_count,
    };
  }

  if (options->target_binding_count != 0) {
    loom_target_declaration_binding_t* bindings = NULL;
    LOOMC_RETURN_IF_ERROR(loomc_status_from_iree(
        iree_arena_allocate_array(arena, options->target_binding_count,
                                  sizeof(*bindings), (void**)&bindings)));
    for (loomc_host_size_t i = 0; i < options->target_binding_count; ++i) {
      const loomc_target_binding_t* binding = &options->target_bindings[i];
      bindings[i] = (loom_target_declaration_binding_t){
          .target_name = iree_string_view_from_loomc(binding->target_symbol),
          .target_profile =
              loomc_target_profile_loom_target_profile(binding->target_profile),
          .product_contract =
              iree_any_bit_set(
                  flags,
                  LOOMC_TARGET_SPECIALIZATION_LIST_FLAG_APPLY_PRODUCT_CONTRACT)
                  ? binding->target_profile->product_contract
                  : NULL,
      };
    }
    *out_bindings = (loom_target_declaration_binding_list_t){
        .values = bindings,
        .count = options->target_binding_count,
    };
  }
  return loomc_ok_status();
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
    loom_function_version_owner_t* function_version_owner,
    loom_low_pass_environment_storage_t* out_storage) {
  return loom_low_pass_environment_storage_initialize_mutable(
      &environment->low_descriptor_registry.registry,
      &environment->low_lower_policy_registry,
      &environment->low_legality_provider_list,
      loom_target_legalizer_registry_storage_registry(
          &environment->legalizer_registry_storage),
      &environment->math_policy_registry, /*compile_report=*/NULL,
      environment->target_environment, function_version_owner, out_storage);
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

void loomc_target_pass_environment_initialize_low_repr_environment(
    const loomc_target_pass_environment_t* environment,
    loom_low_repr_environment_t* out_environment) {
  if (environment == NULL) {
    // Target-free contexts can still read and write generic modules. The
    // format boundary rejects this empty codec if it encounters a scoped Low
    // representation value.
    *out_environment = (loom_low_repr_environment_t){0};
    return;
  }
  loom_low_repr_environment_initialize(
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

loomc_status_t loomc_target_profile_create(
    loomc_target_environment_t* target_environment,
    loomc_string_view_t identifier, loom_target_profile_t* target_profile,
    loomc_target_profile_deinitialize_fn_t deinitialize,
    loomc_allocator_t allocator, loomc_target_profile_t** out_profile) {
  loom_target_profile_t* pending_target_profile = target_profile;
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
  if (loomc_status_is_ok(status) && pending_target_profile == NULL) {
    status = loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                               "target_profile must not be NULL");
  }
  if (loomc_status_is_ok(status) && pending_target_profile->type == NULL) {
    status = loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "target_profile requires a target-family profile type");
  }
  if (loomc_status_is_ok(status) &&
      !loomc_target_environment_supports_profile_type(
          target_environment, pending_target_profile->type)) {
    status = loomc_status_from_iree(iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target environment does not support profile family '%.*s'",
        (int)pending_target_profile->type->name.size,
        pending_target_profile->type->name.data));
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
    profile->target_profile = pending_target_profile;
    const loom_target_emitter_list_t emitters =
        loom_target_environment_emitter_list(&target_environment->environment);
    for (iree_host_size_t i = 0; i < emitters.count; ++i) {
      const loom_target_product_contract_t* product_contract =
          emitters.values[i]->product_contract;
      if (product_contract == NULL) {
        continue;
      }
      if (profile->product_contract != NULL &&
          profile->product_contract != product_contract) {
        profile->product_contract = NULL;
        break;
      }
      profile->product_contract = product_contract;
    }
    profile->target_profile_deinitialize = deinitialize;
    pending_target_profile = NULL;
    status =
        loomc_string_view_clone(identifier, allocator, &profile->identifier);
  }
  if (loomc_status_is_ok(status) && out_profile != NULL) {
    *out_profile = profile;
    profile = NULL;
  } else {
    loomc_target_profile_release(profile);
    loomc_target_profile_deinitialize_target_profile(pending_target_profile,
                                                     deinitialize, allocator);
  }
  return status;
}

const loom_target_profile_t* loomc_target_profile_loom_target_profile(
    const loomc_target_profile_t* profile) {
  return profile ? profile->target_profile : NULL;
}

loomc_target_environment_t* loomc_target_profile_target_environment(
    const loomc_target_profile_t* profile) {
  return profile ? profile->target_environment : NULL;
}

loomc_string_view_t loomc_target_profile_identifier(
    const loomc_target_profile_t* profile) {
  return profile ? profile->identifier : loomc_string_view_empty();
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
  loomc_target_profile_deinitialize_target_profile(
      profile->target_profile, profile->target_profile_deinitialize, allocator);
  loomc_target_environment_release(profile->target_environment);
  loomc_allocator_free(allocator, (void*)profile->identifier.data);
  loomc_allocator_free(allocator, profile);
}
