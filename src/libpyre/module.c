// Copyright 2026 The Pyre Authors
// SPDX-License-Identifier: Apache-2.0
//
// Module loading: wraps IREE executable cache to load GPU fat binaries.
// Ported from iree-hal-streaming's module.c.

#include "pyre_internal.h"

#include <string.h>

#include "iree/io/file_handle.h"

pyre_status_t pyre_module_load_data(pyre_device_t device,
                                    const void* data, size_t size,
                                    pyre_module_t* out_module) {
  if (!device || !data || !size || !out_module) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "NULL argument");
  }
  *out_module = NULL;

  iree_allocator_t alloc = iree_allocator_system();

  pyre_module_s* module = NULL;
  iree_status_t status =
      iree_allocator_malloc(alloc, sizeof(*module), (void**)&module);
  if (!iree_status_is_ok(status)) {
    return pyre_status_from_iree(status);
  }
  memset(module, 0, sizeof(*module));
  iree_atomic_ref_count_init(&module->ref_count);
  module->device = device;
  module->host_allocator = alloc;

  // Create executable cache for this device.
  status = iree_hal_executable_cache_create(
      device->hal_device, iree_make_cstring_view("pyre"),
      &module->cache);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(alloc, module);
    return pyre_status_from_iree(status);
  }

  // Infer the executable format from the binary data.
  iree_const_byte_span_t image = {(const uint8_t*)data, size};
  char executable_format[64];
  iree_device_size_t inferred_size = size;
  status = iree_hal_executable_cache_infer_format(
      module->cache,
      IREE_HAL_EXECUTABLE_CACHING_MODE_ALLOW_PERSISTENT_CACHING,
      image, sizeof(executable_format), executable_format, &inferred_size);
  if (!iree_status_is_ok(status)) {
    iree_hal_executable_cache_release(module->cache);
    iree_allocator_free(alloc, module);
    return pyre_status_from_iree(status);
  }
  image.data_length = inferred_size;

  // Load executable from binary data.
  iree_hal_executable_params_t params;
  iree_hal_executable_params_initialize(&params);
  params.caching_mode =
      IREE_HAL_EXECUTABLE_CACHING_MODE_ALLOW_PERSISTENT_CACHING;
  params.executable_format = iree_make_cstring_view(executable_format);
  params.executable_data = image;
  status = iree_hal_executable_cache_prepare_executable(
      module->cache, &params, &module->executable);
  if (!iree_status_is_ok(status)) {
    iree_hal_executable_cache_release(module->cache);
    iree_allocator_free(alloc, module);
    return pyre_status_from_iree(status);
  }

  module->export_count = iree_hal_executable_export_count(module->executable);
  *out_module = module;
  return pyre_ok_status();
}

pyre_status_t pyre_module_load_file(pyre_device_t device, const char* path,
                                    pyre_module_t* out_module) {
  if (!device || !path || !out_module) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "NULL argument");
  }
  *out_module = NULL;

  iree_allocator_t alloc = iree_allocator_system();

  iree_io_file_handle_t* file_handle = NULL;
  iree_status_t status = iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ, iree_make_cstring_view(path), alloc,
      &file_handle);
  if (!iree_status_is_ok(status)) {
    return pyre_status_from_iree(status);
  }

  iree_io_file_mapping_t* mapping = NULL;
  status = iree_io_file_map_view(
      file_handle, IREE_IO_FILE_ACCESS_READ, 0, IREE_HOST_SIZE_MAX,
      IREE_IO_FILE_MAPPING_FLAG_NONE, alloc, &mapping);
  iree_io_file_handle_release(file_handle);
  if (!iree_status_is_ok(status)) {
    return pyre_status_from_iree(status);
  }

  iree_const_byte_span_t contents = iree_io_file_mapping_contents_ro(mapping);
  pyre_status_t pstatus =
      pyre_module_load_data(device, contents.data, contents.data_length,
                            out_module);
  iree_io_file_mapping_release(mapping);
  return pstatus;
}

pyre_status_t pyre_module_retain(pyre_module_t module) {
  if (!module) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "module is NULL");
  }
  iree_atomic_ref_count_inc(&module->ref_count);
  return pyre_ok_status();
}

pyre_status_t pyre_module_release(pyre_module_t module) {
  if (!module) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "module is NULL");
  }
  if (iree_atomic_ref_count_dec(&module->ref_count) == 1) {
    if (module->executable) {
      iree_hal_executable_release(module->executable);
    }
    if (module->cache) {
      iree_hal_executable_cache_release(module->cache);
    }
    iree_allocator_free(module->host_allocator, module);
  }
  return pyre_ok_status();
}

pyre_status_t pyre_module_lookup_function(pyre_module_t module,
                                          const char* name,
                                          pyre_executable_t* executable,
                                          uint32_t* ordinal) {
  if (!module || !name) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "NULL argument");
  }

  iree_string_view_t name_view = iree_make_cstring_view(name);

  for (uint32_t i = 0; i < module->export_count; ++i) {
    iree_hal_executable_export_info_t info;
    iree_status_t status = iree_hal_executable_export_info(
        module->executable, i, &info);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      continue;
    }
    if (iree_string_view_equal(info.name, name_view)) {
      if (executable) *executable = (pyre_executable_t)module;
      if (ordinal) *ordinal = i;
      return pyre_ok_status();
    }
  }

  return pyre_make_status(PYRE_STATUS_NOT_FOUND,
                          "function not found in module");
}

pyre_status_t pyre_module_lookup_global(pyre_module_t module, const char* name,
                                        void** device_ptr, size_t* size) {
  if (!module || !name || !device_ptr) {
    return pyre_make_status(PYRE_STATUS_INVALID_ARGUMENT, "NULL argument");
  }
  *device_ptr = NULL;
  if (size) *size = 0;

  uint64_t addr = 0;
  iree_device_size_t iree_size = 0;
  iree_status_t status = iree_hal_executable_lookup_global(
      module->executable, iree_make_cstring_view(name),
      IREE_HAL_QUEUE_AFFINITY_ANY, &addr, &iree_size);
  if (!iree_status_is_ok(status)) {
    return pyre_status_from_iree(status);
  }

  *device_ptr = (void*)(uintptr_t)addr;
  if (size) *size = (size_t)iree_size;
  return pyre_ok_status();
}
