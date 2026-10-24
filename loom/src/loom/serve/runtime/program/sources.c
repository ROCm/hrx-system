// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/program/sources.h"

#include <string.h>

#include "iree/base/internal/path.h"
#include "loomc/iree.h"

static iree_string_view_t loom_serve_sources_copy_string(
    iree_string_view_t value, char** cursor) {
  if (!value.size) {
    return iree_string_view_empty();
  }
  memcpy(*cursor, value.data, value.size);
  iree_string_view_t copy = iree_make_string_view(*cursor, value.size);
  *cursor += value.size;
  return copy;
}

iree_status_t loom_serve_sources_clone(const loom_serve_sources_t* sources,
                                       iree_allocator_t host_allocator,
                                       loom_serve_sources_t** out_sources) {
  *out_sources = NULL;
  iree_host_size_t size = 0;
  bool representable = iree_host_size_checked_mul_add(
      sizeof(*sources), sources->count, sizeof(*sources->values), &size);
  const iree_string_view_t paths[] = {sources->base_directory,
                                      sources->prepare,
                                      sources->control,
                                      sources->reference.path,
                                      sources->package_directory,
                                      sources->reference.selector.preset,
                                      sources->reference.selector.service,
                                      sources->reference.selector.variant,
                                      sources->reference.selector.programs,
                                      sources->reference.selector.partition};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(paths) && representable;
       ++i) {
    representable = iree_host_size_checked_add(size, paths[i].size, &size);
  }
  for (iree_host_size_t i = 0; i < sources->count && representable; ++i) {
    representable =
        iree_host_size_checked_add(size, sources->values[i].size, &size);
  }
  if (!representable) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "source namespace storage is not representable");
  }
  loom_serve_sources_t* copy = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, size, (void**)&copy));
  iree_string_view_t* values = (iree_string_view_t*)(copy + 1);
  char* cursor = (char*)(values + sources->count);
  copy->base_directory =
      loom_serve_sources_copy_string(sources->base_directory, &cursor);
  copy->prepare = loom_serve_sources_copy_string(sources->prepare, &cursor);
  copy->control = loom_serve_sources_copy_string(sources->control, &cursor);
  copy->reference.path =
      loom_serve_sources_copy_string(sources->reference.path, &cursor);
  copy->package_directory =
      loom_serve_sources_copy_string(sources->package_directory, &cursor);
  copy->reference.selector.preset = loom_serve_sources_copy_string(
      sources->reference.selector.preset, &cursor);
  copy->reference.selector.service = loom_serve_sources_copy_string(
      sources->reference.selector.service, &cursor);
  copy->reference.selector.variant = loom_serve_sources_copy_string(
      sources->reference.selector.variant, &cursor);
  copy->reference.selector.programs = loom_serve_sources_copy_string(
      sources->reference.selector.programs, &cursor);
  copy->reference.selector.partition = loom_serve_sources_copy_string(
      sources->reference.selector.partition, &cursor);
  copy->count = sources->count;
  copy->values = sources->count ? values : NULL;
  for (iree_host_size_t i = 0; i < sources->count; ++i) {
    values[i] = loom_serve_sources_copy_string(sources->values[i], &cursor);
  }
  copy->package = sources->package;
  loom_serve_package_retain(copy->package);
  copy->card = sources->card;
  copy->component = sources->component;
  loom_model_card_retain(copy->card);
  *out_sources = copy;
  return iree_ok_status();
}

void loom_serve_sources_free(loom_serve_sources_t* sources,
                             iree_allocator_t host_allocator) {
  if (sources) {
    loom_serve_package_release(sources->package);
    loom_model_card_release(sources->card);
    iree_allocator_free(host_allocator, sources);
  }
}

iree_status_t loom_serve_sources_resolve(const loom_serve_sources_t* sources,
                                         iree_allocator_t host_allocator,
                                         loom_serve_sources_t** out_sources) {
  *out_sources = NULL;
  if (!sources->reference.path.size) {
    return loom_serve_sources_clone(sources, host_allocator, out_sources);
  }
  loomc_source_t* document = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_sources_open(sources, sources->reference.path,
                                               host_allocator, &document));
  const loomc_byte_span_t bytes = loomc_source_contents(document);
  loom_model_card_t* card = NULL;
  iree_status_t status = loom_model_card_parse(
      iree_make_string_view((const char*)bytes.data, bytes.data_length),
      &sources->reference.selector, host_allocator, &card);
  loomc_source_release(document);
  char* path = NULL;
  if (iree_status_is_ok(status) &&
      (sources->base_directory.size ||
       loom_serve_path_is_absolute(sources->reference.path))) {
    status =
        loom_serve_path_resolve(sources->base_directory,
                                sources->reference.path, host_allocator, &path);
  }
  char* package_path = NULL;
  if (iree_status_is_ok(status) && sources->package) {
    status =
        iree_file_path_join(sources->package_directory, sources->reference.path,
                            host_allocator, &package_path);
  }
  if (iree_status_is_ok(status)) {
    const loom_serve_sources_t selected = {
        .base_directory =
            path ? iree_file_path_dirname(iree_make_cstring_view(path))
                 : iree_string_view_empty(),
        .package = sources->package,
        .package_directory =
            package_path
                ? iree_file_path_dirname(iree_make_cstring_view(package_path))
                : iree_string_view_empty(),
        .card = card};
    status = loom_serve_sources_clone(&selected, host_allocator, out_sources);
  }
  iree_allocator_free(host_allocator, package_path);
  iree_allocator_free(host_allocator, path);
  loom_model_card_release(card);
  return status;
}

iree_status_t loom_serve_sources_open(const loom_serve_sources_t* sources,
                                      iree_string_view_t locator,
                                      iree_allocator_t host_allocator,
                                      loomc_source_t** out_source) {
  *out_source = NULL;
  if (sources->package && !loom_serve_path_is_absolute(locator)) {
    char* path = NULL;
    IREE_RETURN_IF_ERROR(iree_file_path_join(sources->package_directory,
                                             locator, host_allocator, &path));
    iree_status_t status = loom_serve_package_open(
        sources->package, iree_make_cstring_view(path), out_source);
    iree_allocator_free(host_allocator, path);
    return status;
  }
  char* path = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_path_resolve(sources->base_directory, locator,
                                               host_allocator, &path));
  iree_status_t status = iree_status_from_loomc(loomc_source_create_from_path(
      loomc_make_cstring_view(path), NULL,
      loomc_allocator_from_iree(host_allocator), out_source));
  iree_allocator_free(host_allocator, path);
  return status;
}

static void loom_serve_sources_release_view(void* user_data,
                                            loomc_byte_span_t contents) {
  loomc_source_release(user_data);
}

iree_status_t loom_serve_sources_open_provider(
    const loom_serve_sources_t* sources, const loom_model_provider_t* provider,
    iree_allocator_t host_allocator, loomc_source_t** out_source) {
  *out_source = NULL;
  const iree_string_view_t format = provider->file.format;
  if (provider->options.count) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "Loom providers do not accept importer options");
  }
  loomc_source_format_t source_format = LOOMC_SOURCE_FORMAT_UNKNOWN;
  if (iree_string_view_equal(format, IREE_SV("loom"))) {
    source_format = LOOMC_SOURCE_FORMAT_TEXT;
  } else if (iree_string_view_equal(format, IREE_SV("loombc"))) {
    source_format = LOOMC_SOURCE_FORMAT_BYTECODE;
  } else if (format.size) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "model provider format '%.*s' is not supported by "
        "this linker; supported formats are loom and loombc",
        (int)format.size, format.data);
  }
  if (source_format == LOOMC_SOURCE_FORMAT_UNKNOWN) {
    return loom_serve_sources_open(sources, provider->file.path, host_allocator,
                                   out_source);
  }
  loomc_source_t* storage = NULL;
  IREE_RETURN_IF_ERROR(loom_serve_sources_open(sources, provider->file.path,
                                               host_allocator, &storage));
  const loomc_source_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .structure_size = sizeof(options),
      .format = source_format,
      .identifier = loomc_source_identifier(storage),
      .contents = loomc_source_contents(storage),
      .storage = LOOMC_SOURCE_STORAGE_EXTERNAL,
      .release = loom_serve_sources_release_view,
      .release_user_data = storage};
  iree_status_t status = iree_status_from_loomc(loomc_source_create(
      &options, loomc_allocator_from_iree(host_allocator), out_source));
  if (!iree_status_is_ok(status)) {
    loomc_source_release(storage);
  }
  return status;
}

bool loom_serve_path_is_absolute(iree_string_view_t path) {
#if defined(IREE_PLATFORM_WINDOWS)
  return (path.size >= 2 && ((path.data[0] == '\\' && path.data[1] == '\\') ||
                             (path.data[0] == '/' && path.data[1] == '/'))) ||
         (path.size >= 3 && path.data[1] == ':' &&
          ((path.data[0] >= 'A' && path.data[0] <= 'Z') ||
           (path.data[0] >= 'a' && path.data[0] <= 'z')) &&
          (path.data[2] == '/' || path.data[2] == '\\'));
#else
  return path.size && path.data[0] == '/';
#endif
}

iree_status_t loom_serve_path_resolve(iree_string_view_t base,
                                      iree_string_view_t path,
                                      iree_allocator_t host_allocator,
                                      char** out_path) {
  *out_path = NULL;
  if (loom_serve_path_is_absolute(path)) {
    return iree_file_path_join(iree_string_view_empty(), path, host_allocator,
                               out_path);
  }
#if defined(IREE_PLATFORM_WINDOWS)
  // A root-relative or drive-relative name depends on process drive state,
  // not the model document. It is neither a portable relative locator nor a
  // fully qualified local path.
  if ((path.size && (path.data[0] == '/' || path.data[0] == '\\')) ||
      (path.size >= 2 && path.data[1] == ':')) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "model paths must be relative or fully qualified");
  }
#endif
  if (iree_string_view_is_empty(base)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "relative model path '%.*s' requires a base directory", (int)path.size,
        path.data);
  }
  return iree_file_path_join(base, path, host_allocator, out_path);
}
