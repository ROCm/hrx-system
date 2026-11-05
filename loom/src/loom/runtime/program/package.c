// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/program/package.h"

#include <stdlib.h>
#include <string.h>

#include "iree/base/internal/unicode.h"
#include "loomc/iree.h"

struct loom_runtime_package_t {
  // Shared by configuration owners and asynchronous compilation inputs.
  iree_atomic_ref_count_t ref_count;
  // Allocator owning the table, directory and object.
  iree_allocator_t host_allocator;
  // Owned document directory, including its trailing '/' when nonempty.
  iree_string_view_t directory;
  // Number of populated source references, including partial construction.
  iree_host_size_t count;
  // Owned sources sorted by their copied package-relative identifiers.
  loomc_source_t** files;
};

static iree_status_t loom_runtime_package_check_path(iree_string_view_t path,
                                                     bool canonical) {
  if (!path.size || path.data[0] == '/' || !iree_unicode_utf8_validate(path)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "package filenames must be relative UTF-8 paths");
  }
  for (iree_host_size_t i = 0; i < path.size; ++i) {
    const uint8_t c = (uint8_t)path.data[i];
    if (c < 0x20 || c == 0x7F || c == '\\' || c == ':') {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "package filenames require portable separators");
    }
  }
  iree_string_view_t remaining = path;
  while (remaining.size) {
    iree_string_view_t component;
    iree_string_view_split(remaining, '/', &component, &remaining);
    if (!component.size ||
        (canonical && (iree_string_view_equal(component, IREE_SV(".")) ||
                       iree_string_view_equal(component, IREE_SV(".."))))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "package filenames require canonical components");
    }
  }
  if (path.data[path.size - 1] == '/') {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "package member must name a file");
  }
  return iree_ok_status();
}

void loom_runtime_package_retain(loom_runtime_package_t* package) {
  if (package) {
    iree_atomic_ref_count_inc(&package->ref_count);
  }
}

void loom_runtime_package_release(loom_runtime_package_t* package) {
  if (!package) {
    return;
  }
  if (iree_atomic_ref_count_dec(&package->ref_count) != 1) {
    return;
  }
  for (iree_host_size_t i = 0; i < package->count; ++i) {
    loomc_source_release(package->files[i]);
  }
  iree_allocator_free(package->host_allocator, package->files);
  iree_allocator_free(package->host_allocator, (void*)package->directory.data);
  iree_allocator_free(package->host_allocator, package);
}

static int loom_runtime_package_compare(const void* lhs, const void* rhs) {
  return iree_string_view_compare(
      iree_string_view_from_loomc(
          loomc_source_identifier(*(loomc_source_t* const*)lhs)),
      iree_string_view_from_loomc(
          loomc_source_identifier(*(loomc_source_t* const*)rhs)));
}

iree_status_t loom_runtime_package_create(
    iree_string_view_t document, iree_host_size_t file_count,
    const loom_runtime_package_file_t* files, iree_allocator_t host_allocator,
    loom_runtime_package_t** out_package) {
  *out_package = NULL;
  IREE_RETURN_IF_ERROR(loom_runtime_package_check_path(document, true));
  loom_runtime_package_t* package = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*package),
                                             (void**)&package));
  iree_atomic_ref_count_init(&package->ref_count);
  package->host_allocator = host_allocator;
  iree_host_size_t directory_length = 0;
  for (iree_host_size_t i = 0; i < document.size; ++i) {
    if (document.data[i] == '/') {
      directory_length = i + 1;
    }
  }
  char* directory = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, directory_length + 1, (void**)&directory);
  if (iree_status_is_ok(status)) {
    memcpy(directory, document.data, directory_length);
    directory[directory_length] = 0;
    package->directory = iree_make_string_view(directory, directory_length);
    if (file_count) {
      status = iree_allocator_malloc_array(host_allocator, file_count,
                                           sizeof(*package->files),
                                           (void**)&package->files);
    }
  }
  for (iree_host_size_t i = 0; i < file_count && iree_status_is_ok(status);
       ++i) {
    status = loom_runtime_package_check_path(files[i].path, true);
    if (iree_status_is_ok(status)) {
      const loomc_source_options_t options = {
          .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
          .structure_size = sizeof(options),
          .identifier = loomc_string_view_from_iree(files[i].path),
          .contents = loomc_byte_span_from_iree(files[i].contents),
          .storage = LOOMC_SOURCE_STORAGE_COPY,
      };
      status = iree_status_from_loomc(loomc_source_create(
          &options, loomc_allocator_from_iree(host_allocator),
          &package->files[package->count]));
      if (iree_status_is_ok(status)) {
        ++package->count;
      }
    }
  }
  if (iree_status_is_ok(status) && package->count > 1) {
    qsort(package->files, package->count, sizeof(*package->files),
          loom_runtime_package_compare);
    for (iree_host_size_t i = 1;
         i < package->count && iree_status_is_ok(status); ++i) {
      if (!loom_runtime_package_compare(&package->files[i - 1],
                                        &package->files[i])) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "duplicate package filename");
      }
    }
  }
  if (iree_status_is_ok(status)) {
    *out_package = package;
  } else {
    loom_runtime_package_release(package);
  }
  return status;
}

iree_status_t loom_runtime_package_open(loom_runtime_package_t* package,
                                        iree_string_view_t locator,
                                        loomc_source_t** out_source) {
  *out_source = NULL;
  IREE_RETURN_IF_ERROR(loom_runtime_package_check_path(locator, false));
  iree_string_builder_t path;
  iree_string_builder_initialize(package->host_allocator, &path);
  iree_status_t status =
      iree_string_builder_append_string(&path, package->directory);
  iree_string_view_t remaining = locator;
  while (remaining.size && iree_status_is_ok(status)) {
    iree_string_view_t component;
    iree_string_view_split(remaining, '/', &component, &remaining);
    if (iree_string_view_equal(component, IREE_SV("."))) {
      continue;
    }
    if (iree_string_view_equal(component, IREE_SV(".."))) {
      if (!path.size) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "source locator escapes its package");
      } else {
        // Each accumulated component has a trailing separator.
        --path.size;
        while (path.size && path.buffer[path.size - 1] != '/') {
          --path.size;
        }
        path.buffer[path.size] = 0;
      }
    } else {
      status = iree_string_builder_append_string(&path, component);
      if (iree_status_is_ok(status)) {
        status = iree_string_builder_append_cstring(&path, "/");
      }
    }
  }
  if (iree_status_is_ok(status)) {
    iree_string_view_t name = iree_string_builder_view(&path);
    if (name.size) {
      --name.size;
    }
    iree_host_size_t begin = 0, end = package->count;
    while (begin < end) {
      const iree_host_size_t middle = begin + (end - begin) / 2;
      const int order = iree_string_view_compare(
          name, iree_string_view_from_loomc(
                    loomc_source_identifier(package->files[middle])));
      if (order > 0) {
        begin = middle + 1;
      } else if (order < 0) {
        end = middle;
      } else {
        *out_source = package->files[middle];
        loomc_source_retain(*out_source);
        break;
      }
    }
    if (!*out_source) {
      status = iree_make_status(IREE_STATUS_NOT_FOUND,
                                "package member '%.*s' is missing",
                                (int)name.size, name.data);
    }
  }
  iree_string_builder_deinitialize(&path);
  return status;
}
