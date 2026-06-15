// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loomc/target/amdgpu.h"

#include <string.h>

#include "iree/base/api.h"
#include "loom/error/emitter.h"
#include "loom/target/arch/amdgpu/provider.h"
#include "loom/target/arch/amdgpu/records/target_records.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/target/emit/native/amdgpu/hal_kernel_library.h"
#include "loomc/iree.h"
#include "target.h"

static char loomc_amdgpu_profile_payload_type;

static void loomc_amdgpu_profile_payload_deinitialize(
    void* payload, loomc_allocator_t allocator) {
  (void)payload;
  (void)allocator;
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
  LOOMC_RETURN_IF_ERROR(loomc_amdgpu_validate_string_view(options->processor));
  if (loomc_string_view_is_empty(options->processor)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "AMDGPU profile options require processor");
  }
  return loomc_ok_status();
}

static void loomc_amdgpu_emit_artifact_release(void* storage,
                                               iree_allocator_t allocator) {
  iree_allocator_free(allocator, storage);
}

typedef struct loomc_amdgpu_emit_library_storage_t {
  // AMDGPU kernel library storage.
  loom_amdgpu_hal_kernel_library_t library;
} loomc_amdgpu_emit_library_storage_t;

static void loomc_amdgpu_emit_library_release(void* storage,
                                              iree_allocator_t allocator) {
  loomc_amdgpu_emit_library_storage_t* library_storage =
      (loomc_amdgpu_emit_library_storage_t*)storage;
  loom_amdgpu_hal_kernel_library_deinitialize(&library_storage->library,
                                              allocator);
  iree_allocator_free(allocator, library_storage);
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

static iree_status_t loomc_amdgpu_emit_module_artifact(
    const loom_target_emit_request_t* request,
    loom_target_emit_artifact_t* out_artifact) {
  *out_artifact = (loom_target_emit_artifact_t){0};

  const loom_amdgpu_processor_info_t* processor =
      (const loom_amdgpu_processor_info_t*)request->target_selection.data;
  iree_diagnostic_emitter_t diagnostic_emitter = request->diagnostic_emitter;
  const loom_amdgpu_hal_kernel_library_options_t library_options = {
      .processor = processor ? processor->name : iree_string_view_empty(),
      .target_selection = request->target_selection,
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
  if (iree_status_is_ok(status) &&
      library.artifact_manifest.contents.data == NULL) {
    out_artifact->target_artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF;
    out_artifact->contents = iree_make_const_byte_span(
        library.hsaco_data, library.hsaco_data_length);
    out_artifact->storage = library.hsaco_data;
    out_artifact->release = loomc_amdgpu_emit_artifact_release;
    library.hsaco_data = NULL;
    library.hsaco_data_length = 0;
  } else if (iree_status_is_ok(status)) {
    loomc_amdgpu_emit_library_storage_t* storage = NULL;
    status = iree_allocator_malloc(request->allocator, sizeof(*storage),
                                   (void**)&storage);
    if (iree_status_is_ok(status)) {
      *storage = (loomc_amdgpu_emit_library_storage_t){
          .library = library,
      };
      out_artifact->target_artifact_format = LOOM_TARGET_ARTIFACT_FORMAT_ELF;
      out_artifact->contents = iree_make_const_byte_span(
          storage->library.hsaco_data, storage->library.hsaco_data_length);
      out_artifact->sidecars = &storage->library.artifact_manifest;
      out_artifact->sidecar_count = 1;
      out_artifact->storage = storage;
      out_artifact->release = loomc_amdgpu_emit_library_release;
      library = (loom_amdgpu_hal_kernel_library_t){0};
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

  const loom_amdgpu_processor_info_t* processor = NULL;
  LOOMC_RETURN_IF_ERROR(
      loomc_status_from_iree(loom_amdgpu_target_info_lookup_processor(
          iree_string_view_from_loomc(options->processor), &processor)));
  bool hsaco_supported = false;
  LOOMC_RETURN_IF_ERROR(
      loomc_status_from_iree(loom_amdgpu_target_info_processor_supports_hsaco(
          processor, &hsaco_supported)));
  if (!hsaco_supported) {
    return loomc_make_status(LOOMC_STATUS_UNAVAILABLE,
                             "AMDGPU processor cannot be emitted as HSACO");
  }
  const loom_target_bundle_t* target_bundle =
      loom_amdgpu_target_bundle_for_descriptor_set(
          processor->descriptor_set.ordinal);
  if (target_bundle == NULL) {
    return loomc_make_status(
        LOOMC_STATUS_UNAVAILABLE,
        "AMDGPU processor has no Loom target bundle for its descriptor set");
  }

  const loom_target_selection_t selection = {
      .bundle = target_bundle,
      .data = (void*)processor,
  };
  const loomc_target_profile_options_t profile_options = {
      .type = LOOMC_STRUCTURE_TYPE_TARGET_PROFILE_OPTIONS,
      .structure_size = sizeof(profile_options),
      .identifier = loomc_string_view_is_empty(options->identifier)
                        ? options->processor
                        : options->identifier,
  };
  return loomc_target_profile_create_from_selection(
      target_environment, &profile_options, selection,
      &loomc_amdgpu_profile_payload_type, (void*)processor,
      loomc_amdgpu_profile_payload_deinitialize, allocator, out_profile);
}

loomc_string_view_t loomc_amdgpu_target_profile_processor(
    const loomc_target_profile_t* profile) {
  const loom_amdgpu_processor_info_t* processor =
      (const loom_amdgpu_processor_info_t*)loomc_target_profile_payload(
          profile, &loomc_amdgpu_profile_payload_type);
  return processor ? loomc_string_view_from_iree(processor->name)
                   : loomc_string_view_empty();
}

loomc_status_t loomc_amdgpu_processor_from_hsa_isa_name(
    loomc_string_view_t hsa_isa_name, loomc_string_view_t* out_processor) {
  if (out_processor == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_processor must not be NULL");
  }
  *out_processor = loomc_string_view_empty();
  LOOMC_RETURN_IF_ERROR(loomc_amdgpu_validate_string_view(hsa_isa_name));

  loom_amdgpu_amdhsa_target_id_t target_id = {0};
  LOOMC_RETURN_IF_ERROR(
      loomc_status_from_iree(loom_amdgpu_target_info_parse_amdhsa_target_id(
          iree_string_view_from_loomc(hsa_isa_name), &target_id)));
  *out_processor = loomc_string_view_from_iree(target_id.processor->name);
  return loomc_ok_status();
}
