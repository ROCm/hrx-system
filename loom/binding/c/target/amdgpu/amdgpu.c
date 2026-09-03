// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/amdgpu.h"

#include <string.h>

#include "iree/base/api.h"
#include "loom/error/emitter.h"
#include "loom/target/arch/amdgpu/amdhsa_target_id.h"
#include "loom/target/arch/amdgpu/artifact_key.h"
#include "loom/target/arch/amdgpu/profile.h"
#include "loom/target/arch/amdgpu/provider.h"
#include "loom/target/arch/amdgpu/records/target_records.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/target/emit/native/amdgpu/hal_kernel_library.h"
#include "loom/target/emit/native/amdgpu/product_contract.h"
#include "loomc/iree.h"
#include "target.h"

static void loomc_amdgpu_target_profile_deinitialize(
    loom_target_profile_t* target_profile, loomc_allocator_t allocator) {
  loomc_allocator_free(allocator,
                       (loom_amdgpu_target_profile_t*)target_profile);
}

static loomc_status_t loomc_amdgpu_validate_string_view(
    loomc_string_view_t value) {
  if (value.data == NULL && value.size != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "string view has length but no data");
  }
  return loomc_ok_status();
}

static loomc_status_t loomc_amdgpu_profile_options_validate(
    const loomc_amdgpu_profile_options_t* options) {
  if (options == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "AMDGPU profile options are required");
  }
  if (options->type != LOOMC_STRUCTURE_TYPE_NONE &&
      options->type != LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "AMDGPU profile options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "AMDGPU profile options structure_size is too small");
  }
  if (options->next != NULL) {
    return loomc_make_status(
        LOOMC_STATUS_UNIMPLEMENTED,
        "AMDGPU profile option extensions are not supported");
  }
  LOOMC_RETURN_IF_ERROR(loomc_amdgpu_validate_string_view(options->identifier));
  LOOMC_RETURN_IF_ERROR(
      loomc_amdgpu_validate_string_view(options->identity.target));
  if (loomc_string_view_is_empty(options->identity.target)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "AMDGPU profile options require target");
  }
  switch (options->identity.amdhsa_features.sramecc) {
    case LOOMC_AMDGPU_TARGET_FEATURE_ANY:
    case LOOMC_AMDGPU_TARGET_FEATURE_UNSUPPORTED:
    case LOOMC_AMDGPU_TARGET_FEATURE_OFF:
    case LOOMC_AMDGPU_TARGET_FEATURE_ON:
      break;
    default:
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "AMDGPU profile options contain an unknown sramecc state");
  }
  switch (options->identity.amdhsa_features.xnack) {
    case LOOMC_AMDGPU_TARGET_FEATURE_ANY:
    case LOOMC_AMDGPU_TARGET_FEATURE_UNSUPPORTED:
    case LOOMC_AMDGPU_TARGET_FEATURE_OFF:
    case LOOMC_AMDGPU_TARGET_FEATURE_ON:
      break;
    default:
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "AMDGPU profile options contain an unknown xnack state");
  }
  return loomc_ok_status();
}

static loom_amdgpu_target_feature_state_t
loomc_amdgpu_target_feature_state_to_internal(
    loomc_amdgpu_target_feature_state_t state) {
  switch (state) {
    case LOOMC_AMDGPU_TARGET_FEATURE_UNSUPPORTED:
      return LOOM_AMDGPU_TARGET_FEATURE_UNSUPPORTED;
    case LOOMC_AMDGPU_TARGET_FEATURE_OFF:
      return LOOM_AMDGPU_TARGET_FEATURE_OFF;
    case LOOMC_AMDGPU_TARGET_FEATURE_ON:
      return LOOM_AMDGPU_TARGET_FEATURE_ON;
    default:
      return LOOM_AMDGPU_TARGET_FEATURE_ANY;
  }
}

static loomc_amdgpu_target_feature_state_t
loomc_amdgpu_target_feature_state_from_internal(
    loom_amdgpu_target_feature_state_t state) {
  switch (state) {
    case LOOM_AMDGPU_TARGET_FEATURE_UNSUPPORTED:
      return LOOMC_AMDGPU_TARGET_FEATURE_UNSUPPORTED;
    case LOOM_AMDGPU_TARGET_FEATURE_OFF:
      return LOOMC_AMDGPU_TARGET_FEATURE_OFF;
    case LOOM_AMDGPU_TARGET_FEATURE_ON:
      return LOOMC_AMDGPU_TARGET_FEATURE_ON;
    default:
      return LOOMC_AMDGPU_TARGET_FEATURE_ANY;
  }
}

static void loomc_amdgpu_target_identity_to_internal(
    const loomc_amdgpu_target_identity_t* identity,
    const loom_amdgpu_target_info_t* target,
    loom_amdgpu_target_identity_t* out_identity) {
  loom_amdgpu_target_identity_initialize(target, out_identity);
  out_identity->amdhsa_features = (loom_amdgpu_amdhsa_feature_states_t){
      .sramecc = loomc_amdgpu_target_feature_state_to_internal(
          identity->amdhsa_features.sramecc),
      .xnack = loomc_amdgpu_target_feature_state_to_internal(
          identity->amdhsa_features.xnack),
  };
}

static void loomc_amdgpu_target_identity_from_internal(
    const loom_amdgpu_target_identity_t* identity,
    loomc_amdgpu_target_identity_t* out_identity) {
  *out_identity = (loomc_amdgpu_target_identity_t){
      .target = loomc_string_view_from_iree(identity->target->name),
      .amdhsa_features =
          {
              .sramecc = loomc_amdgpu_target_feature_state_from_internal(
                  identity->amdhsa_features.sramecc),
              .xnack = loomc_amdgpu_target_feature_state_from_internal(
                  identity->amdhsa_features.xnack),
          },
  };
}

loomc_status_t loomc_amdgpu_target_identity_parse_artifact_key(
    loomc_string_view_t artifact_key,
    loomc_amdgpu_target_identity_t* out_identity) {
  if (out_identity == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_identity must not be NULL");
  }
  *out_identity = (loomc_amdgpu_target_identity_t){0};
  LOOMC_RETURN_IF_ERROR(loomc_amdgpu_validate_string_view(artifact_key));

  loom_amdgpu_target_identity_t identity = {0};
  LOOMC_RETURN_IF_ERROR(loomc_status_from_iree(loom_amdgpu_artifact_key_parse(
      iree_string_view_from_loomc(artifact_key), &identity)));
  loomc_amdgpu_target_identity_from_internal(&identity, out_identity);
  return loomc_ok_status();
}

typedef struct loomc_amdgpu_emit_artifact_storage_t {
  // Allocator owning this storage.
  iree_allocator_t allocator;

  // Artifact manifest sidecar descriptor.
  loom_target_emit_sidecar_artifact_t artifact_manifest;
} loomc_amdgpu_emit_artifact_storage_t;

typedef struct loomc_amdgpu_emit_option_prefix_t {
  // Structure type identifying the descriptor.
  loomc_structure_type_t type;

  // Size of this structure in bytes.
  loomc_host_size_t structure_size;

  // Next descriptor in the option extension chain.
  const void* next;
} loomc_amdgpu_emit_option_prefix_t;

static void loomc_amdgpu_emit_artifact_storage_release(void* storage) {
  loomc_amdgpu_emit_artifact_storage_t* artifact_storage =
      (loomc_amdgpu_emit_artifact_storage_t*)storage;
  iree_allocator_free(artifact_storage->allocator, artifact_storage);
}

static iree_status_t loomc_amdgpu_forward_diagnostic(
    void* user_data, const loom_diagnostic_t* diagnostic) {
  iree_diagnostic_emitter_t* emitter = (iree_diagnostic_emitter_t*)user_data;
  if (diagnostic == NULL || diagnostic->error == NULL) {
    return iree_ok_status();
  }
  const loom_diagnostic_emission_t emission = {
      .error = diagnostic->error,
      .params = diagnostic->params,
      .param_count = diagnostic->param_count,
  };
  return iree_diagnostic_emit(*emitter, &emission);
}

static iree_status_t loomc_amdgpu_emit_validate_runtime_globals(
    loomc_amdgpu_runtime_global_flags_t flags) {
  if ((flags & ~LOOMC_AMDGPU_RUNTIME_GLOBALS_KNOWN) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU emit options contain unknown runtime global bits");
  }
  return iree_ok_status();
}

static loom_amdgpu_runtime_global_flags_t loomc_amdgpu_emit_map_runtime_globals(
    loomc_amdgpu_runtime_global_flags_t flags) {
  loom_amdgpu_runtime_global_flags_t runtime_globals =
      LOOM_AMDGPU_RUNTIME_GLOBAL_NONE;
  if (iree_any_bit_set(flags, LOOMC_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG)) {
    runtime_globals |= LOOM_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG;
  }
  if (iree_any_bit_set(flags, LOOMC_AMDGPU_RUNTIME_GLOBAL_ASAN_CONFIG)) {
    runtime_globals |= LOOM_AMDGPU_RUNTIME_GLOBAL_ASAN_CONFIG;
  }
  return runtime_globals;
}

static iree_status_t loomc_amdgpu_emit_resolve_runtime_globals(
    const void* option_chain,
    loom_amdgpu_runtime_global_flags_t* out_runtime_globals) {
  *out_runtime_globals = LOOM_AMDGPU_RUNTIME_GLOBAL_NONE;
  const void* node = option_chain;
  bool has_amdgpu_options = false;
  while (node != NULL) {
    const loomc_amdgpu_emit_option_prefix_t* prefix =
        (const loomc_amdgpu_emit_option_prefix_t*)node;
    if (prefix->type != LOOMC_STRUCTURE_TYPE_AMDGPU_EMIT_OPTIONS) {
      if (prefix->structure_size != 0 &&
          prefix->structure_size < sizeof(*prefix)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "AMDGPU emit option extension structure_size is too small");
      }
      node = prefix->next;
      continue;
    }
    if (has_amdgpu_options) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU emit option chain contains duplicate AMDGPU emit options");
    }
    if (prefix->structure_size != 0 &&
        prefix->structure_size < sizeof(loomc_amdgpu_emit_options_t)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU emit options structure_size is too small");
    }

    const loomc_amdgpu_emit_options_t* options =
        (const loomc_amdgpu_emit_options_t*)node;
    const loomc_amdgpu_runtime_global_flags_t runtime_globals =
        options->runtime_globals;
    IREE_RETURN_IF_ERROR(
        loomc_amdgpu_emit_validate_runtime_globals(runtime_globals));
    *out_runtime_globals =
        loomc_amdgpu_emit_map_runtime_globals(runtime_globals);
    has_amdgpu_options = true;
    node = options->next;
  }
  return iree_ok_status();
}

static iree_status_t loomc_amdgpu_emit_module_artifact(
    const loom_target_emit_request_t* request,
    loom_target_emit_artifact_t* out_artifact) {
  *out_artifact = (loom_target_emit_artifact_t){0};

  loom_amdgpu_runtime_global_flags_t runtime_globals =
      LOOM_AMDGPU_RUNTIME_GLOBAL_NONE;
  IREE_RETURN_IF_ERROR(loomc_amdgpu_emit_resolve_runtime_globals(
      request->option_chain, &runtime_globals));
  iree_diagnostic_emitter_t diagnostic_emitter = request->diagnostic_emitter;
  const loom_amdgpu_hal_kernel_library_options_t library_options = {
      .function_versions = request->function_versions,
      .runtime_globals = runtime_globals,
      .diagnostic_sink =
          {
              .fn = loomc_amdgpu_forward_diagnostic,
              .user_data = &diagnostic_emitter,
          },
      .max_errors = 20,
      .report = request->compile_report,
      .artifact_name = request->identifier,
      .artifact_manifest_identifier = request->artifact_manifest.identifier,
      .artifact_manifest =
          {
              .mode = request->artifact_manifest.mode,
          },
  };
  bool emitted = false;
  loom_amdgpu_hal_kernel_library_t library = {0};
  iree_status_t status = loom_amdgpu_emit_hal_kernel_library(
      request->module, &library_options, request->allocator, &emitted,
      &library);
  if (iree_status_is_ok(status) && !emitted) {
    status =
        iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                         "AMDGPU HSACO emission produced no executable bytes");
  }
  if (iree_status_is_ok(status) && library.artifact_manifest.contents == NULL) {
    out_artifact->target_artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF;
    out_artifact->contents = library.hsaco_data;
    library.hsaco_data = NULL;
  } else if (iree_status_is_ok(status)) {
    loomc_amdgpu_emit_artifact_storage_t* storage = NULL;
    status = iree_allocator_malloc(request->allocator, sizeof(*storage),
                                   (void**)&storage);
    if (iree_status_is_ok(status)) {
      *storage = (loomc_amdgpu_emit_artifact_storage_t){
          .allocator = request->allocator,
          .artifact_manifest = library.artifact_manifest,
      };
      out_artifact->target_artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF;
      out_artifact->contents = library.hsaco_data;
      out_artifact->sidecars = &storage->artifact_manifest;
      out_artifact->sidecar_count = 1;
      out_artifact->storage = storage;
      out_artifact->release_storage =
          loomc_amdgpu_emit_artifact_storage_release;
      library.hsaco_data = NULL;
      library.artifact_manifest = (loom_target_emit_sidecar_artifact_t){0};
    }
  }
  loom_amdgpu_hal_kernel_library_deinitialize(&library, request->allocator);
  return status;
}

static const loom_target_emitter_t loomc_amdgpu_hsaco_emitter = {
    .name = {"amdgpu-hsaco", 12},
    .public_artifact_format = {LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO,
                               sizeof(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO) - 1},
    .default_identifier = {"module.hsaco", 12},
    .target_artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF,
    .product_contract = &loom_amdgpu_hsaco_kernel_product_contract,
    .emit = loomc_amdgpu_emit_module_artifact,
};

static const loom_target_emitter_t* const kLoomcAmdgpuEmitters[] = {
    &loomc_amdgpu_hsaco_emitter,
};

static const loom_target_provider_t loomc_amdgpu_emit_target_provider = {
    .emitter_list =
        {
            .values = kLoomcAmdgpuEmitters,
            .count = IREE_ARRAYSIZE(kLoomcAmdgpuEmitters),
        },
};

static const loom_target_provider_t* const kLoomcAmdgpuTargetProviders[] = {
    &loom_amdgpu_target_provider,
    &loomc_amdgpu_emit_target_provider,
};

static const loom_target_provider_set_t loomc_amdgpu_target_provider_set = {
    .providers = kLoomcAmdgpuTargetProviders,
    .provider_count = IREE_ARRAYSIZE(kLoomcAmdgpuTargetProviders),
};

loomc_status_t loomc_target_environment_create_amdgpu(
    loomc_allocator_t allocator,
    loomc_target_environment_t** out_target_environment) {
  return loomc_target_environment_create_from_provider_set(
      &loomc_amdgpu_target_provider_set, allocator, out_target_environment);
}

loomc_status_t loomc_target_profile_create_amdgpu(
    loomc_target_environment_t* target_environment,
    const loomc_amdgpu_profile_options_t* options, loomc_allocator_t allocator,
    loomc_target_profile_t** out_profile) {
  if (out_profile == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_profile must not be NULL");
  }
  *out_profile = NULL;
  if (target_environment == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "target_environment must not be NULL");
  }
  LOOMC_RETURN_IF_ERROR(loomc_amdgpu_profile_options_validate(options));

  const loom_amdgpu_target_info_t* target = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_status_from_iree(loom_amdgpu_target_info_lookup_target(
          iree_string_view_from_loomc(options->identity.target), &target)));
  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_info_target_processor(target);
  IREE_ASSERT(processor != NULL);
  if (!loom_amdgpu_processor_properties_support_hsaco(&processor->properties)) {
    return loomc_make_status(LOOMC_STATUS_UNAVAILABLE,
                             "AMDGPU target cannot be emitted as HSACO");
  }

  loom_amdgpu_target_profile_t* target_profile = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_allocator_malloc(
      allocator, sizeof(*target_profile), (void**)&target_profile));
  loom_amdgpu_target_identity_t identity = {0};
  loomc_amdgpu_target_identity_to_internal(&options->identity, target,
                                           &identity);
  loomc_status_t status = loomc_status_from_iree(
      loom_amdgpu_target_profile_initialize(&identity, target_profile));
  if (!loomc_status_is_ok(status)) {
    loomc_allocator_free(allocator, target_profile);
    return status;
  }

  const loomc_string_view_t identifier =
      loomc_string_view_is_empty(options->identifier) ? options->identity.target
                                                      : options->identifier;
  return loomc_target_profile_create(
      target_environment, identifier, &target_profile->base,
      loomc_amdgpu_target_profile_deinitialize, allocator, out_profile);
}

loomc_status_t loomc_amdgpu_target_profile_query_identity(
    const loomc_target_profile_t* profile,
    loomc_amdgpu_target_identity_t* out_identity) {
  if (out_identity == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_identity must not be NULL");
  }
  *out_identity = (loomc_amdgpu_target_identity_t){0};
  if (profile == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "profile must not be NULL");
  }
  const loom_amdgpu_target_profile_t* target_profile =
      loom_amdgpu_target_profile_cast(
          loomc_target_profile_loom_target_profile(profile));
  if (target_profile == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "profile is not an AMDGPU profile");
  }
  loomc_amdgpu_target_identity_from_internal(&target_profile->identity,
                                             out_identity);
  return loomc_ok_status();
}

loomc_status_t loomc_amdgpu_target_identity_from_hsa_isa_name(
    loomc_string_view_t hsa_isa_name, uint32_t asic_revision,
    loomc_amdgpu_target_identity_t* out_identity) {
  if (out_identity == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_identity must not be NULL");
  }
  *out_identity = (loomc_amdgpu_target_identity_t){0};
  LOOMC_RETURN_IF_ERROR(loomc_amdgpu_validate_string_view(hsa_isa_name));

  loom_amdgpu_amdhsa_target_id_t target_id = {0};
  LOOMC_RETURN_IF_ERROR(
      loomc_status_from_iree(loom_amdgpu_amdhsa_target_id_parse(
          iree_string_view_from_loomc(hsa_isa_name), &target_id)));
  const loom_amdgpu_target_info_t* target = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_status_from_iree(loom_amdgpu_target_info_lookup_physical_target(
          target_id.processor, asic_revision, &target)));
  const loom_amdgpu_target_identity_t internal_identity = {
      .target = target,
      .amdhsa_features = target_id.features,
  };
  loomc_amdgpu_target_identity_from_internal(&internal_identity, out_identity);
  return loomc_ok_status();
}
