// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/parameter/sources.h"

#include "iree/io/parameter_index_provider.h"
#include "iree/tooling/parameter_util.h"

typedef struct loom_serve_parameter_source_t {
  // Owned canonical keys, metadata, and open checkpoint files.
  iree_io_parameter_index_t* index;
  // Owned imported-file cache shared by every consumer of this source.
  iree_io_parameter_provider_t* provider;
} loom_serve_parameter_source_t;

struct loom_serve_parameter_sources_t {
  // Model and request-choice catalog ownership.
  iree_atomic_ref_count_t ref_count;
  // Allocator owning this catalog and its trailing source records.
  iree_allocator_t host_allocator;
  // Retained immutable sources preceding the owned records, or NULL.
  loom_serve_parameter_sources_t* prefix;
  // First ordinal owned by this catalog, equal to the prefix's source count.
  uint32_t first;
  // Owned records, including partial construction on failure.
  uint32_t count;
  // Independent indexes/providers in declaration order.
  loom_serve_parameter_source_t values[];
};

iree_status_t loom_serve_parameter_sources_create(
    loom_serve_parameter_sources_t* prefix, const loom_serve_sources_t* sources,
    uint32_t count, const loom_model_file_list_t* paths,
    iree_allocator_t host_allocator,
    loom_serve_parameter_sources_t** out_sources) {
  *out_sources = NULL;
  if (!count) {
    loom_serve_parameter_sources_retain(prefix);
    *out_sources = prefix;
    return iree_ok_status();
  }
  loom_serve_parameter_sources_t* catalog = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_struct_array(
      host_allocator, sizeof(*catalog), count, sizeof(*catalog->values),
      (void**)&catalog));
  iree_atomic_ref_count_init(&catalog->ref_count);
  catalog->host_allocator = host_allocator;
  catalog->prefix = prefix;
  loom_serve_parameter_sources_retain(prefix);
  catalog->first = loom_serve_parameter_sources_count(prefix);
  iree_status_t status = iree_ok_status();
  for (uint32_t s = 0; s < count && iree_status_is_ok(status); ++s) {
    loom_serve_parameter_source_t* source = &catalog->values[s];
    ++catalog->count;
    status = iree_io_parameter_index_create(host_allocator, &source->index);
    for (iree_host_size_t p = 0;
         p < paths[s].count && iree_status_is_ok(status); ++p) {
      char* path = NULL;
      status = loom_serve_path_resolve(sources->base_directory,
                                       paths[s].values[p].path, host_allocator,
                                       &path);
      if (iree_status_is_ok(status)) {
        status = iree_tooling_append_parameter_file_to_index_with_format(
            iree_make_cstring_view(path), paths[s].values[p].format,
            source->index, host_allocator);
      }
      iree_allocator_free(host_allocator, path);
    }
    if (iree_status_is_ok(status)) {
      status = iree_io_parameter_index_provider_create(
          iree_string_view_empty(), source->index, IREE_HOST_SIZE_MAX,
          host_allocator, &source->provider);
    }
  }
  if (iree_status_is_ok(status)) {
    *out_sources = catalog;
  } else {
    loom_serve_parameter_sources_release(catalog);
  }
  return status;
}

void loom_serve_parameter_sources_retain(
    loom_serve_parameter_sources_t* sources) {
  if (sources) {
    iree_atomic_ref_count_inc(&sources->ref_count);
  }
}

void loom_serve_parameter_sources_release(
    loom_serve_parameter_sources_t* sources) {
  if (sources && iree_atomic_ref_count_dec(&sources->ref_count) == 1) {
    for (uint32_t i = 0; i < sources->count; ++i) {
      iree_io_parameter_provider_release(sources->values[i].provider);
      iree_io_parameter_index_release(sources->values[i].index);
    }
    loom_serve_parameter_sources_release(sources->prefix);
    iree_allocator_free(sources->host_allocator, sources);
  }
}

uint32_t loom_serve_parameter_sources_count(
    const loom_serve_parameter_sources_t* sources) {
  return sources ? sources->first + sources->count : 0;
}

iree_io_parameter_index_t* loom_serve_parameter_sources_index(
    const loom_serve_parameter_sources_t* sources, uint32_t ordinal) {
  if (ordinal < sources->first) {
    return loom_serve_parameter_sources_index(sources->prefix, ordinal);
  }
  return sources->values[ordinal - sources->first].index;
}

iree_io_parameter_provider_t* loom_serve_parameter_sources_provider(
    const loom_serve_parameter_sources_t* sources, uint32_t ordinal) {
  if (ordinal < sources->first) {
    return loom_serve_parameter_sources_provider(sources->prefix, ordinal);
  }
  return sources->values[ordinal - sources->first].provider;
}
