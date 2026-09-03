// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/compile/artifact_product.h"

#include "loom/product/kernel.h"

static iree_status_t loom_artifact_product_validate_candidate(
    const loom_product_format_provider_t* product_provider,
    const loom_artifact_candidate_t* candidate) {
  if (!candidate->compiled) return iree_ok_status();
  const loom_artifact_t* artifact = &candidate->artifact;
  if (artifact->target_bundle == NULL || artifact->executable_data == NULL ||
      artifact->target_artifact_data == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "artifact provider '%.*s' returned incomplete kernel metadata",
        (int)candidate->provider->name.size, candidate->provider->name.data);
  }
  if (artifact->target_artifact_data != artifact->executable_data) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "artifact provider '%.*s' returned distinct target-native and "
        "executable payloads for product format '%.*s'",
        (int)candidate->provider->name.size, candidate->provider->name.data,
        (int)product_provider->format->name.size,
        product_provider->format->name.data);
  }
  if ((artifact->target_listing_data == NULL) !=
      iree_string_view_is_empty(artifact->target_listing_format)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "artifact provider '%.*s' returned an incomplete target listing",
        (int)candidate->provider->name.size, candidate->provider->name.data);
  }
  for (iree_host_size_t i = 0; i < artifact->sidecar_count; ++i) {
    const loom_target_emit_sidecar_artifact_t* sidecar = &artifact->sidecars[i];
    if (sidecar->kind !=
            LOOM_TARGET_EMIT_SIDECAR_ARTIFACT_KIND_ARTIFACT_MANIFEST ||
        iree_string_view_is_empty(sidecar->identifier) ||
        sidecar->contents == NULL) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "artifact provider '%.*s' returned an unsupported or incomplete "
          "sidecar artifact",
          (int)candidate->provider->name.size, candidate->provider->name.data);
    }
  }
  return iree_ok_status();
}

static void loom_artifact_product_populate_artifacts(
    const loom_product_format_provider_t* product_provider,
    const loom_product_build_request_t* request,
    const loom_artifact_t* source_artifact,
    loom_product_artifact_t* out_artifacts) {
  iree_host_size_t artifact_ordinal = 0;
  out_artifacts[artifact_ordinal++] = (loom_product_artifact_t){
      .role = IREE_SVL(LOOM_PRODUCT_ARTIFACT_ROLE_KERNEL),
      .format = product_provider->format->name,
      .identifier = iree_string_view_is_empty(request->artifact_identifier)
                        ? product_provider->format->name
                        : request->artifact_identifier,
      .contents = source_artifact->executable_data,
  };
  if (source_artifact->target_listing_data != NULL) {
    out_artifacts[artifact_ordinal++] = (loom_product_artifact_t){
        .role = IREE_SVL(LOOM_PRODUCT_ARTIFACT_ROLE_LISTING),
        .format = source_artifact->target_listing_format,
        .identifier = source_artifact->target_listing_format,
        .contents = source_artifact->target_listing_data,
    };
  }
  for (iree_host_size_t i = 0; i < source_artifact->sidecar_count; ++i) {
    const loom_target_emit_sidecar_artifact_t* sidecar =
        &source_artifact->sidecars[i];
    out_artifacts[artifact_ordinal++] = (loom_product_artifact_t){
        .role = IREE_SVL(LOOM_PRODUCT_ARTIFACT_ROLE_ARTIFACT_MANIFEST),
        .format = IREE_SVL(LOOM_PRODUCT_ARTIFACT_FORMAT_JSON),
        .identifier = sidecar->identifier,
        .contents = sidecar->contents,
    };
  }
}

iree_status_t loom_artifact_provider_build_kernel_product(
    const loom_product_format_provider_t* product_provider,
    const loom_artifact_provider_t* artifact_provider,
    const loom_product_build_request_t* request, loom_product_t** out_product) {
  IREE_ASSERT_ARGUMENT(out_product);
  *out_product = NULL;
  if (product_provider == NULL ||
      product_provider->operation != &loom_kernel_product_operation ||
      product_provider->format == NULL || artifact_provider == NULL ||
      request == NULL || request->module == NULL ||
      request->compile_options == NULL ||
      iree_allocator_is_null(request->allocator)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel product adapter inputs are incomplete");
  }
  if (artifact_provider->target_profile_type !=
          product_provider->target_profile_type ||
      artifact_provider->product_contract !=
          product_provider->target_product_contract) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "product and artifact providers disagree on their target contract");
  }

  loom_artifact_candidate_t candidate = {0};
  const loom_artifact_target_t target = {
      .target_profile = request->target_profile,
      .target_key = request->target_key,
  };
  iree_status_t status =
      request->target_profile != NULL
          ? loom_artifact_candidate_emit_target(
                artifact_provider, &target, request->module,
                request->compile_options, request->allocator, &candidate)
          : loom_artifact_candidate_emit_module_target(
                artifact_provider, request->module, request->compile_options,
                request->allocator, &candidate);
  if (iree_status_is_ok(status)) {
    status =
        loom_artifact_product_validate_candidate(product_provider, &candidate);
  }

  loom_product_artifact_t* artifacts = NULL;
  iree_host_size_t artifact_count = 0;
  if (iree_status_is_ok(status) && candidate.compiled) {
    artifact_count = 1;
    if (candidate.artifact.target_listing_data != NULL) ++artifact_count;
    if (!iree_host_size_checked_add(artifact_count,
                                    candidate.artifact.sidecar_count,
                                    &artifact_count)) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "kernel product has too many artifacts");
    }
  }
  if (iree_status_is_ok(status) && candidate.compiled) {
    status =
        iree_allocator_malloc_array(request->allocator, artifact_count,
                                    sizeof(*artifacts), (void**)&artifacts);
  }
  if (iree_status_is_ok(status) && candidate.compiled) {
    loom_artifact_product_populate_artifacts(product_provider, request,
                                             &candidate.artifact, artifacts);
    iree_string_view_t target_key = candidate.artifact.target_key;
    if (iree_string_view_is_empty(target_key)) target_key = request->target_key;
    if (iree_string_view_is_empty(target_key)) {
      target_key = candidate.artifact.target_bundle->name;
    }
    const loom_kernel_product_options_t options = {
        .target_key = target_key,
        .target_bundle = candidate.artifact.target_bundle,
        .artifacts = artifacts,
        .artifact_count = artifact_count,
        .loadable_artifact_ordinal = 0,
        .export_count = request->export_count,
        .requirement_count = 0,
    };
    status =
        loom_kernel_product_create(&options, request->allocator, out_product);
    if (iree_status_is_ok(status) && request->compile_options->report != NULL) {
      request->compile_options->report->target_key =
          loom_kernel_product_target_key(*out_product);
    }
  }

  iree_allocator_free(request->allocator, artifacts);
  loom_artifact_candidate_deinitialize(&candidate);
  return status;
}
