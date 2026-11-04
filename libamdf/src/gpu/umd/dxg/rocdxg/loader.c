// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define _GNU_SOURCE
#include "libamdf/src/gpu/umd/dxg/rocdxg/loader.h"

#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "libamdf/src/allocator.h"

#define AMDF_DXG_BRIDGE_PATH_ENVIRONMENT_VARIABLE "AMDF_DXG_BRIDGE_PATH"
#define AMDF_ROCDXG_BRIDGE_FILE_NAME "libamdf_rocdxg_bridge.so"

static const char amdf_gpu_dxg_rocdxg_module_anchor = 0;

static amdf_status_t amdf_gpu_dxg_rocdxg_allocate_path(
    amdf_allocator_t host_allocator, const char* directory_path,
    size_t directory_length, const char* file_name, char** out_path) {
  const size_t file_name_length = strlen(file_name);
  if (directory_length > SIZE_MAX - file_name_length - 1) {
    return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  }
  const size_t path_length = directory_length + file_name_length;
  char* path = NULL;
  amdf_status_t status = amdf_malloc(host_allocator, path_length + 1,
                                     amdf_alignof(char), (void**)&path);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  memcpy(path, directory_path, directory_length);
  memcpy(path + directory_length, file_name, file_name_length + 1);
  *out_path = path;
  return AMDF_STATUS_OK;
}

static amdf_status_t amdf_gpu_dxg_rocdxg_allocate_bridge_path(
    amdf_allocator_t host_allocator, char** out_path) {
  const char* environment_path =
      getenv(AMDF_DXG_BRIDGE_PATH_ENVIRONMENT_VARIABLE);
  if (environment_path != NULL) {
    if (environment_path[0] == '\0') {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
    return amdf_gpu_dxg_rocdxg_allocate_path(host_allocator, environment_path,
                                             strlen(environment_path), "",
                                             out_path);
  }

  Dl_info owner_info = {0};
  if (dladdr((const void*)&amdf_gpu_dxg_rocdxg_module_anchor, &owner_info) ==
          0 ||
      owner_info.dli_fname == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  const char* file_name = strrchr(owner_info.dli_fname, '/');
  const size_t directory_length =
      file_name == NULL ? 0 : (size_t)(file_name + 1 - owner_info.dli_fname);
  return amdf_gpu_dxg_rocdxg_allocate_path(
      host_allocator, owner_info.dli_fname, directory_length,
      AMDF_ROCDXG_BRIDGE_FILE_NAME, out_path);
}

static bool amdf_gpu_dxg_rocdxg_api_is_complete(
    const amdf_dxg_bridge_api_t* api) {
  return api != NULL && api->structure_size >= sizeof(*api) &&
         api->abi_version == AMDF_DXG_BRIDGE_ABI_VERSION_13 &&
         api->gpu_adapter_open != NULL && api->gpu_adapter_close != NULL &&
         api->gpu_allocation_query_layout != NULL &&
         api->gpu_allocation_create != NULL &&
         api->gpu_kernel_queue_create != NULL &&
         api->gpu_kernel_queue_submit != NULL &&
         api->gpu_kernel_queue_destroy != NULL &&
         api->gpu_buffer_prepare_import != NULL &&
         api->gpu_clock_sample != NULL;
}

amdf_status_t amdf_gpu_dxg_rocdxg_loader_initialize(
    amdf_allocator_t host_allocator, amdf_gpu_dxg_rocdxg_loader_t* out_loader) {
  char* bridge_path = NULL;
  amdf_status_t status =
      amdf_gpu_dxg_rocdxg_allocate_bridge_path(host_allocator, &bridge_path);
  amdf_gpu_dxg_rocdxg_loader_t loader = {0};
  if (amdf_status_is_ok(status)) {
    loader.module = dlopen(bridge_path, RTLD_NOW | RTLD_LOCAL);
    if (loader.module == NULL) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
    }
  }
  amdf_free(host_allocator, bridge_path);
  if (amdf_status_is_ok(status)) {
    *out_loader = loader;
  }
  return status;
}

amdf_status_t amdf_gpu_dxg_rocdxg_loader_query_api(
    const amdf_gpu_dxg_rocdxg_loader_t* loader,
    const amdf_dxg_bridge_api_t** out_api) {
  (void)dlerror();
  void* query_symbol = dlsym(loader->module, "amdf_dxg_bridge_query_api");
  const char* query_error = dlerror();
  if (query_error != NULL || query_symbol == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }

  amdf_dxg_bridge_query_api_fn_t query_api = NULL;
  _Static_assert(sizeof(query_api) == sizeof(query_symbol),
                 "ELF data and function pointers must have equal size");
  memcpy(&query_api, &query_symbol, sizeof(query_api));
  const amdf_dxg_bridge_api_t* api = NULL;
  const amdf_dxg_bridge_result_t result = query_api(
      AMDF_DXG_BRIDGE_ABI_VERSION_13, AMDF_DXG_BRIDGE_ABI_VERSION_13, &api);
  if (result == AMDF_DXG_BRIDGE_RESULT_VERSION_MISMATCH) {
    return amdf_make_api_status(AMDF_STATUS_CODE_VERSION_MISMATCH);
  }
  if (result != AMDF_DXG_BRIDGE_RESULT_SUCCESS ||
      !amdf_gpu_dxg_rocdxg_api_is_complete(api)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  *out_api = api;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_gpu_dxg_rocdxg_loader_deinitialize(
    amdf_gpu_dxg_rocdxg_loader_t* loader) {
  if (dlclose(loader->module) != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  *loader = (amdf_gpu_dxg_rocdxg_loader_t){0};
  return AMDF_STATUS_OK;
}
