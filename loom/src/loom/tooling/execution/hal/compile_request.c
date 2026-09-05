// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/execution/hal/compile_request.h"

iree_status_t loom_run_hal_compile_request_resolve(
    const loom_run_hal_compile_resolve_options_t* options,
    loom_run_hal_compile_request_t* out_request) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(options->module);
  IREE_ASSERT_ARGUMENT(options->target_environment);
  IREE_ASSERT_ARGUMENT(options->device_provider);
  IREE_ASSERT_ARGUMENT(options->runtime);
  IREE_ASSERT_ARGUMENT(out_request);
  *out_request = (loom_run_hal_compile_request_t){0};

  const loom_artifact_provider_t* artifact_provider =
      options->device_provider->artifact_provider;
  if (artifact_provider == NULL ||
      artifact_provider->target_profile_type == NULL ||
      artifact_provider->target_profile_type->fact_type == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "selected device provider has no complete "
                            "artifact target family");
  }
  const loom_artifact_provider_t* artifact_providers[] = {
      artifact_provider,
  };
  const loom_artifact_provider_registry_t artifact_registry = {
      .providers = artifact_providers,
      .provider_count = IREE_ARRAYSIZE(artifact_providers),
  };

  loom_compile_request_t compile_request = {0};
  IREE_RETURN_IF_ERROR(loom_compile_request_resolve(
      options->module, &options->compile, &artifact_registry,
      options->target_environment,
      artifact_provider->target_profile_type->fact_type, &compile_request));
  if (compile_request.product != LOOM_COMPILE_PRODUCT_KERNEL) {
    const iree_string_view_t product_name =
        loom_compile_product_name(compile_request.product);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "HAL execution requires product 'kernel'; selected roots infer "
        "product '%.*s'",
        (int)product_name.size, product_name.data);
  }
  if (compile_request.producer.kind != LOOM_COMPILE_PRODUCER_ARTIFACT ||
      compile_request.producer.value.artifact_provider != artifact_provider) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "format '%.*s' is not loadable by the selected --device provider",
        (int)compile_request.format.size, compile_request.format.data);
  }
  if (options->target_requirement != NULL &&
      compile_request.target_fact_type !=
          options->target_requirement->fact_type) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "selected kernel root target facts do not match its compile request");
  }

  iree_status_t status = iree_ok_status();
  if (compile_request.explicit_target.target_profile != NULL) {
    status = loom_device_provider_select_profile_target(
        options->device_provider, options->runtime,
        compile_request.explicit_target.target_profile,
        compile_request.explicit_target.target_key,
        &out_request->device_target);
  } else {
    status = loom_device_provider_select_compatible_target(
        options->device_provider, options->runtime, options->target_requirement,
        &out_request->device_target);
  }
  if (iree_status_is_ok(status)) {
    out_request->compile = compile_request;
  } else {
    *out_request = (loom_run_hal_compile_request_t){0};
  }
  return status;
}

iree_status_t loom_run_hal_compile_request_target_profile(
    const loom_run_hal_compile_request_t* request,
    const loom_device_provider_t* device_provider,
    const loom_run_hal_runtime_t* runtime,
    loom_device_target_profile_t* out_device_profile,
    const loom_target_profile_t** out_profile) {
  IREE_ASSERT_ARGUMENT(request);
  IREE_ASSERT_ARGUMENT(device_provider);
  IREE_ASSERT_ARGUMENT(runtime);
  IREE_ASSERT_ARGUMENT(out_device_profile);
  IREE_ASSERT_ARGUMENT(out_profile);
  *out_device_profile = (loom_device_target_profile_t){0};
  *out_profile = request->compile.explicit_target.target_profile;
  if (*out_profile != NULL) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_device_target_profile_initialize(
      device_provider, runtime, &request->device_target, out_device_profile));
  *out_profile = &out_device_profile->base;
  return iree_ok_status();
}
