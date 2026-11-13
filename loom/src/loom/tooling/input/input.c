// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/input/input.h"

#include <string.h>

#include "loom/error/source.h"
#include "loom/format/bytecode/reader.h"
#include "loom/ir/module.h"

static iree_status_t loom_input_text_load(const loom_input_request_t* request,
                                          loom_input_source_capture_t capture,
                                          loom_context_t* context,
                                          iree_arena_block_pool_t* block_pool,
                                          iree_allocator_t host_allocator,
                                          loom_module_t** out_module) {
  if (!iree_string_view_is_empty(request->options)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Loom text input does not accept input options");
  }
  return loom_text_parse(request->source, request->path, context, block_pool,
                         &request->parse_options, out_module);
}

const loom_input_provider_t loom_input_text_provider = {
    .format = &loom_input_text_format,
    .load = loom_input_text_load,
};

static iree_status_t loom_input_bytecode_load(
    const loom_input_request_t* request, loom_input_source_capture_t capture,
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    iree_allocator_t host_allocator, loom_module_t** out_module) {
  if (!iree_string_view_is_empty(request->options)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Loom bytecode input does not accept input options");
  }
  const loom_bytecode_read_options_t read_options = {
      .diagnostic_sink = request->parse_options.diagnostic_sink,
      .low_repr_environment = request->low_repr_environment,
  };
  loom_bytecode_read_result_t read_result = {0};
  return loom_bytecode_read_module(
      iree_make_const_byte_span(request->source.data, request->source.size),
      request->path, context, block_pool, &read_options, host_allocator,
      &read_result, out_module);
}

const loom_input_provider_t loom_input_bytecode_provider = {
    .format = &loom_input_bytecode_format,
    .load = loom_input_bytecode_load,
};

iree_status_t loom_input_provider_select(
    loom_input_provider_list_t providers, iree_string_view_t name,
    iree_string_view_t path, const loom_input_provider_t** out_provider) {
  *out_provider = NULL;
  // The final application supplies this small, immutable provider list.
  const loom_input_format_t** formats =
      iree_alloca(providers.count * sizeof(*formats));
  for (iree_host_size_t i = 0; i < providers.count; ++i) {
    formats[i] = providers.values[i]->format;
  }
  const loom_input_format_t* format = NULL;
  iree_host_size_t ordinal = 0;
  IREE_RETURN_IF_ERROR(loom_input_format_select(
      (loom_input_format_list_t){formats, providers.count}, name, path, &format,
      &ordinal));
  *out_provider = ordinal == 0   ? &loom_input_text_provider
                  : ordinal == 1 ? &loom_input_bytecode_provider
                                 : providers.values[ordinal - 2];
  return iree_ok_status();
}

typedef struct loom_input_source_path_t {
  // Next source path retained by this invocation.
  struct loom_input_source_path_t* next;
  // Physical identity used by the frontend and include lookup.
  iree_string_view_t path;
  // Logical identity used by diagnostics and the output module.
  iree_string_view_t filename;
  // True when a frontend admitted source bytes under this physical identity.
  bool is_admitted;
} loom_input_source_path_t;

typedef struct loom_input_capture_t {
  // Output owning captured source snapshots.
  loom_input_module_t* input;
  // Caller options borrowed during admission.
  const loom_input_request_t* request;
  // Physical-to-logical source names already resolved.
  loom_input_source_path_t* paths;
  // Temporary path-remapping and diagnostic storage.
  iree_arena_allocator_t* scratch_arena;
} loom_input_capture_t;

static iree_status_t loom_input_copy_arena_string(
    iree_arena_allocator_t* arena, iree_string_view_t value,
    iree_string_view_t* out_value) {
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, value.size + 1, (void**)&storage));
  if (value.size) {
    memcpy(storage, value.data, value.size);
  }
  storage[value.size] = 0;
  *out_value = iree_make_string_view(storage, value.size);
  return iree_ok_status();
}

static iree_status_t loom_input_clone_string(iree_allocator_t allocator,
                                             iree_string_view_t value,
                                             iree_string_view_t* out_value) {
  iree_host_size_t allocation_size = 0;
  if (!iree_host_size_checked_add(value.size, 1, &allocation_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "input filename length overflow");
  }
  char* storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
      allocator, allocation_size, (void**)&storage));
  if (value.size) {
    memcpy(storage, value.data, value.size);
  }
  storage[value.size] = 0;
  *out_value = iree_make_string_view(storage, value.size);
  return iree_ok_status();
}

static iree_status_t loom_input_remap_path(loom_input_capture_t* capture,
                                           iree_string_view_t path,
                                           iree_string_view_t* out_filename) {
  if (iree_string_view_is_empty(path)) {
    *out_filename = iree_string_view_empty();
    return iree_ok_status();
  }
  for (loom_input_source_path_t* source_path = capture->paths;
       source_path != NULL; source_path = source_path->next) {
    if (iree_string_view_equal(source_path->path, path)) {
      *out_filename = source_path->filename;
      return iree_ok_status();
    }
  }
  iree_string_view_t filename = iree_string_view_empty();
  char* filename_storage = NULL;
  IREE_RETURN_IF_ERROR(loom_tooling_source_path_remap(
      path, &capture->request->source_path_options,
      iree_arena_allocator(capture->scratch_arena), &filename,
      &filename_storage));
  iree_arena_allocator_t* arena = capture->scratch_arena;
  loom_input_source_path_t* source_path = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*source_path), (void**)&source_path));
  IREE_RETURN_IF_ERROR(
      loom_input_copy_arena_string(arena, path, &source_path->path));
  if (filename_storage != NULL) {
    source_path->filename = filename;
  } else {
    IREE_RETURN_IF_ERROR(
        loom_input_copy_arena_string(arena, filename, &source_path->filename));
  }
  source_path->is_admitted = false;
  source_path->next = capture->paths;
  capture->paths = source_path;
  *out_filename = source_path->filename;
  return iree_ok_status();
}

static iree_status_t loom_input_admit_source_path(
    loom_input_capture_t* capture, iree_string_view_t path,
    iree_string_view_t* out_filename) {
  IREE_RETURN_IF_ERROR(loom_input_remap_path(capture, path, out_filename));
  if (iree_string_view_is_empty(path)) {
    return iree_ok_status();
  }
  loom_input_source_path_t* admitted_path = NULL;
  for (loom_input_source_path_t* source_path = capture->paths;
       source_path != NULL; source_path = source_path->next) {
    if (iree_string_view_equal(source_path->path, path)) {
      admitted_path = source_path;
    } else if (source_path->is_admitted &&
               iree_string_view_equal(source_path->filename, *out_filename)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "source path remapping gives distinct sources '%.*s' and '%.*s' "
          "the same filename '%.*s'",
          (int)source_path->path.size, source_path->path.data, (int)path.size,
          path.data, (int)out_filename->size, out_filename->data);
    }
  }
  IREE_ASSERT(admitted_path);
  admitted_path->is_admitted = true;
  return iree_ok_status();
}

static iree_status_t loom_input_capture_source(void* user_data,
                                               loom_source_id_t source_id,
                                               iree_string_view_t path,
                                               iree_string_view_t source) {
  loom_input_capture_t* capture = (loom_input_capture_t*)user_data;
  iree_string_view_t filename = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_input_admit_source_path(capture, path, &filename));
  return loom_source_storage_insert(&capture->input->sources, source_id,
                                    filename, source);
}

static iree_status_t loom_input_capture_diagnostic(
    void* user_data, const loom_diagnostic_t* diagnostic) {
  loom_input_capture_t* capture = (loom_input_capture_t*)user_data;
  loom_diagnostic_t remapped = *diagnostic;
  IREE_RETURN_IF_ERROR(loom_input_remap_path(
      capture, diagnostic->origin.filename, &remapped.origin.filename));
  IREE_RETURN_IF_ERROR(
      loom_input_remap_path(capture, diagnostic->source_location.filename,
                            &remapped.source_location.filename));
  if (diagnostic->related_location_count) {
    loom_diagnostic_related_location_t* related = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        capture->scratch_arena, diagnostic->related_location_count,
        sizeof(*related), (void**)&related));
    for (iree_host_size_t i = 0; i < diagnostic->related_location_count; ++i) {
      related[i] = diagnostic->related_locations[i];
      IREE_RETURN_IF_ERROR(
          loom_input_remap_path(capture, related[i].source_location.filename,
                                &related[i].source_location.filename));
    }
    remapped.related_locations = related;
  }
  loom_diagnostic_sink_t sink = capture->request->parse_options.diagnostic_sink;
  return sink.fn(sink.user_data, &remapped);
}

static iree_status_t loom_input_bind_sources(loom_input_capture_t* capture) {
  loom_input_module_t* input = capture->input;
  loom_module_t* module = input->module;
  input->sources.table.module = module;
  for (iree_host_size_t i = 0; i < module->sources.count; ++i) {
    iree_string_view_t path = module->sources.entries[i];
    iree_string_view_t filename = iree_string_view_empty();
    IREE_RETURN_IF_ERROR(loom_input_remap_path(capture, path, &filename));
    if (iree_string_view_equal(path, capture->request->path)) {
      IREE_RETURN_IF_ERROR(
          loom_source_storage_insert(&input->sources, (loom_source_id_t)i,
                                     filename, capture->request->source));
    }
    // Locations keep their IDs; only the source table's displayed names change.
    IREE_RETURN_IF_ERROR(loom_input_copy_arena_string(
        &module->arena, filename, &module->sources.entries[i]));
  }
  return iree_ok_status();
}

iree_status_t loom_input_module_load(const loom_input_provider_t* provider,
                                     const loom_input_request_t* request,
                                     loom_context_t* context,
                                     iree_arena_block_pool_t* block_pool,
                                     iree_allocator_t host_allocator,
                                     loom_input_module_t* out_input) {
  *out_input = (loom_input_module_t){.host_allocator = host_allocator};
  loom_source_storage_initialize(host_allocator, &out_input->sources);
  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(block_pool, &scratch_arena);
  loom_input_capture_t capture = {
      .input = out_input,
      .request = request,
      .scratch_arena = &scratch_arena,
  };
  iree_string_view_t filename = iree_string_view_empty();
  iree_status_t status =
      loom_input_admit_source_path(&capture, request->path, &filename);
  if (iree_status_is_ok(status)) {
    status =
        loom_input_clone_string(host_allocator, filename, &out_input->filename);
  }
  loom_input_request_t frontend_request = *request;
  if (request->parse_options.diagnostic_sink.fn) {
    frontend_request.parse_options.diagnostic_sink = (loom_diagnostic_sink_t){
        .fn = loom_input_capture_diagnostic, .user_data = &capture};
  }
  if (iree_status_is_ok(status)) {
    status = provider->load(
        &frontend_request,
        (loom_input_source_capture_t){.fn = loom_input_capture_source,
                                      .user_data = &capture},
        context, block_pool, host_allocator, &out_input->module);
  }
  if (iree_status_is_ok(status) && out_input->module) {
    status = loom_input_bind_sources(&capture);
  }
  iree_arena_deinitialize(&scratch_arena);
  return status;
}

loom_source_resolver_t loom_input_module_source_resolver(
    loom_input_module_t* input) {
  return loom_source_storage_resolver(&input->sources);
}

void loom_input_module_deinitialize(loom_input_module_t* input) {
  loom_module_free(input->module);
  loom_source_storage_deinitialize(&input->sources);
  iree_allocator_free(input->host_allocator, (void*)input->filename.data);
  memset(input, 0, sizeof(*input));
}
