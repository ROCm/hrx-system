// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/pipeline/kernel_library.h"

#include <stdint.h>
#include <string.h>

#include "iree/base/internal/atomics.h"

struct id4_pipeline_kernel_library_t {
  // Reference count for shared library ownership.
  iree_atomic_ref_count_t ref_count;
  // Allocator used for library storage.
  iree_allocator_t host_allocator;
  // Number of module entries.
  iree_host_size_t module_count;
  // Module entries owned by this library.
  id4_pipeline_kernel_module_t* modules;
};

static iree_status_t id4_pipeline_kernel_library_validate_options_size(
    iree_host_size_t actual_size, iree_host_size_t expected_size,
    iree_string_view_t options_name) {
  if (actual_size >= expected_size) return iree_ok_status();
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "%.*s options structure size %" PRIhsz
                          " is smaller than expected %" PRIhsz,
                          (int)options_name.size, options_name.data,
                          actual_size, expected_size);
}

static iree_status_t id4_pipeline_kernel_library_copy_string(
    iree_string_view_t source, iree_allocator_t host_allocator,
    iree_string_view_t* out_target) {
  *out_target = iree_string_view_empty();
  if (iree_string_view_is_empty(source)) return iree_ok_status();
  if (source.size == IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "kernel library string is too large to copy");
  }
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      host_allocator, source.size + 1, sizeof(storage[0]), (void**)&storage));
  memcpy(storage, source.data, source.size);
  storage[source.size] = 0;
  *out_target = iree_make_string_view(storage, source.size);
  return iree_ok_status();
}

static void id4_pipeline_kernel_library_free_string(
    iree_string_view_t* value, iree_allocator_t host_allocator) {
  if (!value) return;
  iree_allocator_free(host_allocator, (void*)value->data);
  memset(value, 0, sizeof(*value));
}

static iree_status_t id4_pipeline_kernel_library_copy_bytes(
    iree_const_byte_span_t source, iree_allocator_t host_allocator,
    iree_const_byte_span_t* out_target) {
  memset(out_target, 0, sizeof(*out_target));
  if (source.data_length == 0) return iree_ok_status();
  uint8_t* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc_array(host_allocator, source.data_length,
                                  sizeof(storage[0]), (void**)&storage));
  memcpy(storage, source.data, source.data_length);
  *out_target = iree_make_const_byte_span(storage, source.data_length);
  return iree_ok_status();
}

static void id4_pipeline_kernel_library_free_bytes(
    iree_const_byte_span_t* value, iree_allocator_t host_allocator) {
  if (!value) return;
  iree_allocator_free(host_allocator, (void*)value->data);
  memset(value, 0, sizeof(*value));
}

static iree_status_t id4_pipeline_kernel_library_validate_module(
    const id4_pipeline_kernel_module_t* module, iree_host_size_t index) {
  if (iree_string_view_is_empty(module->module_path)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel module %" PRIhsz " module path is required",
                            index);
  }
  if (iree_string_view_is_empty(module->source_identifier)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel module %.*s source identifier is required",
                            (int)module->module_path.size,
                            module->module_path.data);
  }
  if (!module->source_contents.data ||
      module->source_contents.data_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel module %.*s source contents are required",
                            (int)module->module_path.size,
                            module->module_path.data);
  }
  return iree_ok_status();
}

static iree_status_t id4_pipeline_kernel_library_validate_create_options(
    const id4_pipeline_kernel_library_create_options_t* options) {
  if (!options) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel library create options are required");
  }
  IREE_RETURN_IF_ERROR(id4_pipeline_kernel_library_validate_options_size(
      options->structure_size, sizeof(*options),
      IREE_SV("kernel library create")));
  if (options->next) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "kernel library create extension structures are not supported");
  }
  if (options->module_count != 0 && !options->modules) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel library module array is required");
  }
  for (iree_host_size_t i = 0; i < options->module_count; ++i) {
    IREE_RETURN_IF_ERROR(
        id4_pipeline_kernel_library_validate_module(&options->modules[i], i));
    for (iree_host_size_t j = 0; j < i; ++j) {
      if (iree_string_view_equal(options->modules[i].module_path,
                                 options->modules[j].module_path)) {
        return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                                "kernel module path %.*s is duplicated",
                                (int)options->modules[i].module_path.size,
                                options->modules[i].module_path.data);
      }
    }
  }
  return iree_ok_status();
}

static void id4_pipeline_kernel_library_destroy(
    id4_pipeline_kernel_library_t* library) {
  iree_allocator_t host_allocator = library->host_allocator;
  for (iree_host_size_t i = 0; i < library->module_count; ++i) {
    id4_pipeline_kernel_library_free_bytes(&library->modules[i].source_contents,
                                           host_allocator);
    id4_pipeline_kernel_library_free_string(
        &library->modules[i].source_identifier, host_allocator);
    id4_pipeline_kernel_library_free_string(&library->modules[i].module_path,
                                            host_allocator);
  }
  iree_allocator_free(host_allocator, library->modules);
  iree_allocator_free(host_allocator, library);
}

iree_status_t id4_pipeline_kernel_library_create(
    const id4_pipeline_kernel_library_create_options_t* options,
    iree_allocator_t host_allocator,
    id4_pipeline_kernel_library_t** out_library) {
  IREE_ASSERT_ARGUMENT(out_library);
  *out_library = NULL;
  IREE_RETURN_IF_ERROR(
      id4_pipeline_kernel_library_validate_create_options(options));

  id4_pipeline_kernel_library_t* library = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*library),
                                             (void**)&library));
  memset(library, 0, sizeof(*library));
  iree_atomic_ref_count_init(&library->ref_count);
  library->host_allocator = host_allocator;
  library->module_count = options->module_count;

  iree_status_t status = iree_ok_status();
  if (options->module_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, options->module_count,
                                         sizeof(library->modules[0]),
                                         (void**)&library->modules);
  }
  if (iree_status_is_ok(status)) {
    memset(library->modules, 0,
           options->module_count * sizeof(library->modules[0]));
    for (iree_host_size_t i = 0;
         i < options->module_count && iree_status_is_ok(status); ++i) {
      status = id4_pipeline_kernel_library_copy_string(
          options->modules[i].module_path, host_allocator,
          &library->modules[i].module_path);
      if (iree_status_is_ok(status)) {
        status = id4_pipeline_kernel_library_copy_string(
            options->modules[i].source_identifier, host_allocator,
            &library->modules[i].source_identifier);
      }
      if (iree_status_is_ok(status)) {
        status = id4_pipeline_kernel_library_copy_bytes(
            options->modules[i].source_contents, host_allocator,
            &library->modules[i].source_contents);
      }
    }
  }
  if (iree_status_is_ok(status)) {
    *out_library = library;
  } else {
    id4_pipeline_kernel_library_destroy(library);
  }
  return status;
}

static iree_status_t id4_pipeline_kernel_library_source_file_to_module(
    const id4_pipeline_kernel_source_file_t* source_file,
    iree_host_size_t index, id4_pipeline_kernel_module_t* out_module) {
  memset(out_module, 0, sizeof(*out_module));
  if (!source_file) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel source file %" PRIhsz " is NULL", index);
  }
  if (iree_string_view_is_empty(source_file->source_identifier)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "kernel source file %" PRIhsz " source identifier is required", index);
  }
  if (!iree_string_view_ends_with(source_file->source_identifier,
                                  IREE_SV(".loom"))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "kernel source file %.*s must be named <module_path>.loom",
        (int)source_file->source_identifier.size,
        source_file->source_identifier.data);
  }
  if (!source_file->source_contents.data ||
      source_file->source_contents.data_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel source file %.*s contents are required",
                            (int)source_file->source_identifier.size,
                            source_file->source_identifier.data);
  }
  out_module->module_path = iree_string_view_strip_suffix(
      source_file->source_identifier, IREE_SV(".loom"));
  out_module->source_identifier = source_file->source_identifier;
  out_module->source_contents = source_file->source_contents;
  return iree_ok_status();
}

iree_status_t id4_pipeline_kernel_library_create_from_source_files(
    iree_host_size_t source_file_count,
    const id4_pipeline_kernel_source_file_t* source_files,
    iree_allocator_t host_allocator,
    id4_pipeline_kernel_library_t** out_library) {
  IREE_ASSERT_ARGUMENT(out_library);
  *out_library = NULL;
  if (source_file_count != 0 && !source_files) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel source file array is required");
  }

  id4_pipeline_kernel_module_t* modules = NULL;
  iree_status_t status = iree_ok_status();
  if (source_file_count != 0) {
    status = iree_allocator_malloc_array(host_allocator, source_file_count,
                                         sizeof(modules[0]), (void**)&modules);
  }
  for (iree_host_size_t i = 0;
       i < source_file_count && iree_status_is_ok(status); ++i) {
    status = id4_pipeline_kernel_library_source_file_to_module(&source_files[i],
                                                               i, &modules[i]);
  }
  if (iree_status_is_ok(status)) {
    id4_pipeline_kernel_library_create_options_t options;
    memset(&options, 0, sizeof(options));
    options.structure_size = sizeof(options);
    options.module_count = source_file_count;
    options.modules = modules;
    status = id4_pipeline_kernel_library_create(&options, host_allocator,
                                                out_library);
  }
  iree_allocator_free(host_allocator, modules);
  return status;
}

void id4_pipeline_kernel_library_retain(
    id4_pipeline_kernel_library_t* library) {
  if (!library) return;
  iree_atomic_ref_count_inc(&library->ref_count);
}

void id4_pipeline_kernel_library_release(
    id4_pipeline_kernel_library_t* library) {
  if (library && iree_atomic_ref_count_dec(&library->ref_count) == 1) {
    id4_pipeline_kernel_library_destroy(library);
  }
}

iree_host_size_t id4_pipeline_kernel_library_module_count(
    const id4_pipeline_kernel_library_t* library) {
  return library ? library->module_count : 0;
}

const id4_pipeline_kernel_module_t* id4_pipeline_kernel_library_module_at(
    const id4_pipeline_kernel_library_t* library, iree_host_size_t index) {
  if (!library || index >= library->module_count) return NULL;
  return &library->modules[index];
}

iree_status_t id4_pipeline_kernel_library_lookup(
    const id4_pipeline_kernel_library_t* library,
    iree_string_view_t module_path,
    const id4_pipeline_kernel_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = NULL;
  if (!library) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel library is required");
  }
  if (iree_string_view_is_empty(module_path)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel module path is required");
  }
  for (iree_host_size_t i = 0; i < library->module_count; ++i) {
    if (iree_string_view_equal(library->modules[i].module_path, module_path)) {
      *out_module = &library->modules[i];
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "kernel module path %.*s was not found",
                          (int)module_path.size, module_path.data);
}
