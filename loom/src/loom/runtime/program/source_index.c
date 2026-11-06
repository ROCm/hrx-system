// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/program/source_index.h"

#include "iree/base/internal/arena.h"
#include "loom/tooling/input/loomc_configured.h"
#include "loomc/iree.h"

enum {
  LOOM_RUNTIME_SOURCE_INDEX_BLOCK_SIZE = 128 * 1024,
};

typedef struct loom_runtime_source_index_import_t {
  // Mutable compiler workspace created only when a foreign provider is used.
  loomc_workspace_t* workspace;
  // Import-option scratch returned after each provider admission.
  iree_arena_block_pool_t block_pool;
  // Whether block_pool requires deinitialization.
  bool block_pool_initialized;
} loom_runtime_source_index_import_t;

static void loom_runtime_source_index_identity_update(
    loom_sha256_state_t* identity, const void* data, iree_host_size_t length) {
  if (!identity) {
    return;
  }
  uint8_t encoded_length[8];
  iree_unaligned_store_le_u64(encoded_length, length);
  loom_sha256_update(identity, iree_make_const_byte_span(
                                   encoded_length, sizeof(encoded_length)));
  loom_sha256_update(identity, iree_make_const_byte_span(data, length));
}

static void loom_runtime_source_index_identity_update_view(
    loom_sha256_state_t* identity, iree_string_view_t value) {
  loom_runtime_source_index_identity_update(identity, value.data, value.size);
}

static void loom_runtime_source_index_identity_update_u64(
    loom_sha256_state_t* identity, uint64_t value) {
  uint8_t encoded[8];
  iree_unaligned_store_le_u64(encoded, value);
  loom_runtime_source_index_identity_update(identity, encoded, sizeof(encoded));
}

static void loom_runtime_source_index_identity_update_source(
    loom_sha256_state_t* identity, const loomc_source_t* source) {
  if (!identity) {
    return;
  }
  const loomc_byte_span_t contents = loomc_source_contents(source);
  loom_runtime_source_index_identity_update_u64(
      identity, (uint64_t)loomc_source_format(source));
  loom_runtime_source_index_identity_update_view(
      identity, iree_string_view_from_loomc(loomc_source_identifier(source)));
  loom_runtime_source_index_identity_update(
      identity, contents.data, (iree_host_size_t)contents.data_length);
}

static void loom_runtime_source_index_identity_update_direct(
    loom_sha256_state_t* identity, loomc_link_provider_role_t role,
    iree_string_view_t provider_name, const loomc_source_t* source) {
  if (!identity) {
    return;
  }
  loom_runtime_source_index_identity_update_u64(identity, 0);
  loom_runtime_source_index_identity_update_u64(identity, (uint64_t)role);
  loom_runtime_source_index_identity_update_view(identity, provider_name);
  loom_runtime_source_index_identity_update_source(identity, source);
}

static iree_status_t loom_runtime_source_index_identity_update_provider(
    loom_sha256_state_t* identity, const loom_runtime_sources_t* sources,
    const loom_model_provider_t* provider,
    const loom_input_format_t* selected_format, loomc_link_provider_role_t role,
    const loomc_source_t* source, iree_allocator_t host_allocator) {
  if (!identity) {
    return iree_ok_status();
  }
  loom_runtime_source_index_identity_update_u64(identity, 1);
  loom_runtime_source_index_identity_update_u64(identity, (uint64_t)role);
  loom_runtime_source_index_identity_update_view(identity, provider->file.path);
  loom_runtime_source_index_identity_update_view(identity,
                                                 selected_format->name);
  loom_runtime_source_index_identity_update_u64(identity,
                                                provider->options.count);
  for (iree_host_size_t i = 0; i < provider->options.count; ++i) {
    loom_runtime_source_index_identity_update_view(identity,
                                                   provider->options.values[i]);
  }
  loom_runtime_source_index_identity_update_source(identity, source);
  loom_runtime_source_index_identity_update_u64(identity, provider->data.count);
  for (iree_host_size_t i = 0; i < provider->data.count; ++i) {
    const loom_model_file_t file = provider->data.values[i];
    loomc_source_t* data_source = NULL;
    iree_status_t status = loom_runtime_sources_open(
        sources, file.path, host_allocator, &data_source);
    if (iree_status_is_ok(status)) {
      loom_runtime_source_index_identity_update_view(identity, file.path);
      loom_runtime_source_index_identity_update_view(identity, file.format);
      loom_runtime_source_index_identity_update_source(identity, data_source);
    }
    loomc_source_release(data_source);
    IREE_RETURN_IF_ERROR(status);
  }
  return iree_ok_status();
}

static void LOOMC_API_PTR loom_runtime_source_index_release_source(
    void* user_data, loomc_byte_span_t contents) {
  (void)contents;
  loomc_source_release((loomc_source_t*)user_data);
}

static iree_status_t loom_runtime_source_index_format_source(
    loomc_source_t* source, loomc_source_format_t format,
    iree_allocator_t host_allocator, loomc_source_t** out_source) {
  *out_source = NULL;
  if (loomc_source_format(source) == format) {
    *out_source = source;
    loomc_source_retain(source);
    return iree_ok_status();
  }
  loomc_source_retain(source);
  const loomc_source_options_t options = {
      .type = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS,
      .structure_size = sizeof(options),
      .format = format,
      .identifier = loomc_source_identifier(source),
      .contents = loomc_source_contents(source),
      .storage = LOOMC_SOURCE_STORAGE_EXTERNAL,
      .release = loom_runtime_source_index_release_source,
      .release_user_data = source,
  };
  iree_status_t status = iree_status_from_loomc(loomc_source_create(
      &options, loomc_allocator_from_iree(host_allocator), out_source));
  if (!iree_status_is_ok(status)) {
    loomc_source_release(source);
  }
  return status;
}

static iree_status_t loom_runtime_source_index_import_initialize(
    loom_runtime_source_index_import_t* import,
    iree_allocator_t host_allocator) {
  if (import->workspace) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_status_from_loomc(loomc_workspace_create(
      NULL, loomc_allocator_from_iree(host_allocator), &import->workspace)));
  iree_arena_block_pool_initialize(LOOM_RUNTIME_SOURCE_INDEX_BLOCK_SIZE,
                                   host_allocator, &import->block_pool);
  import->block_pool_initialized = true;
  return iree_ok_status();
}

static void loom_runtime_source_index_import_deinitialize(
    loom_runtime_source_index_import_t* import) {
  loomc_workspace_release(import->workspace);
  if (import->block_pool_initialized) {
    iree_arena_block_pool_deinitialize(&import->block_pool);
  }
}

static iree_status_t loom_runtime_source_index_add_direct(
    loomc_link_index_builder_t* builder, loomc_source_t* source,
    loomc_link_provider_role_t role, iree_string_view_t provider_name,
    loom_sha256_state_t* identity) {
  const loomc_link_index_source_options_t options = {
      .provider_name = loomc_string_view_from_iree(provider_name),
      .role = role,
  };
  loom_runtime_source_index_identity_update_direct(identity, role,
                                                   provider_name, source);
  return iree_status_from_loomc(
      loomc_link_index_builder_add_source(builder, source, &options, NULL));
}

static iree_status_t loom_runtime_source_index_add_provider(
    loomc_context_t* context, loomc_link_index_builder_t* builder,
    const loom_runtime_sources_t* sources,
    const loom_model_provider_t* provider,
    const loom_input_format_t* selected_format, loomc_source_t* source,
    loomc_link_provider_role_t role, loom_runtime_source_index_import_t* import,
    loom_runtime_compilation_report_t* report, iree_allocator_t host_allocator,
    loom_sha256_state_t* identity) {
  iree_status_t status = loom_runtime_source_index_identity_update_provider(
      identity, sources, provider, selected_format, role, source,
      host_allocator);
  const loomc_link_index_source_options_t link_options = {
      .provider_name = loomc_string_view_from_iree(provider->file.path),
      .role = role,
  };
  if (iree_status_is_ok(status) &&
      (iree_string_view_equal(selected_format->name,
                              loom_input_text_format.name) ||
       iree_string_view_equal(selected_format->name,
                              loom_input_bytecode_format.name))) {
    iree_string_view_t input_options = iree_string_view_empty();
    status = loom_input_options_for_provider(
        provider->options, selected_format->name, &input_options);
    if (iree_status_is_ok(status) &&
        !iree_string_view_is_empty(input_options)) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "%.*s model providers do not accept input options",
          (int)selected_format->name.size, selected_format->name.data);
    }
    loomc_source_t* formatted_source = NULL;
    if (iree_status_is_ok(status)) {
      const loomc_source_format_t format =
          iree_string_view_equal(selected_format->name,
                                 loom_input_bytecode_format.name)
              ? LOOMC_SOURCE_FORMAT_BYTECODE
              : LOOMC_SOURCE_FORMAT_TEXT;
      status = loom_runtime_source_index_format_source(
          source, format, host_allocator, &formatted_source);
    }
    if (iree_status_is_ok(status)) {
      status = iree_status_from_loomc(loomc_link_index_builder_add_source(
          builder, formatted_source, &link_options, NULL));
    }
    loomc_source_release(formatted_source);
  } else if (iree_status_is_ok(status)) {
    status =
        loom_runtime_source_index_import_initialize(import, host_allocator);
    loomc_module_t* module = NULL;
    loomc_result_t* result = NULL;
    if (iree_status_is_ok(status)) {
      const loomc_byte_span_t contents = loomc_source_contents(source);
      const loom_tooling_loomc_input_options_t admission_options = {
          .input =
              {
                  .format = selected_format->name,
                  .provider_options = provider->options,
              },
          .path = iree_string_view_from_loomc(loomc_source_identifier(source)),
          .source =
              iree_make_string_view((const char*)contents.data,
                                    (iree_host_size_t)contents.data_length),
          .import = loom_configured_input_loomc_importer(),
      };
      status = loom_tooling_input_admit_loomc_module(
          &admission_options, context, import->workspace, &import->block_pool,
          host_allocator, &module, &result);
      status = loom_runtime_compilation_result(loomc_status_from_iree(status),
                                               result, report);
    }
    if (iree_status_is_ok(status)) {
      status = iree_status_from_loomc(loomc_link_index_builder_add_module(
          builder, module, &link_options, NULL));
    }
    loomc_module_release(module);
  }
  return status;
}

iree_status_t loom_runtime_source_index_create(
    loomc_context_t* context, const loom_runtime_sources_t* sources,
    loomc_source_t* primary, loom_runtime_compilation_report_t* report,
    iree_allocator_t host_allocator, loom_sha256_t* out_identity,
    loomc_link_index_t** out_index) {
  *out_index = NULL;
  if (out_identity) {
    *out_identity = (loom_sha256_t){0};
  }
  loom_sha256_state_t identity_storage;
  loom_sha256_state_t* identity = out_identity ? &identity_storage : NULL;
  if (identity) {
    loom_sha256_initialize(identity);
    loom_runtime_source_index_identity_update_view(
        identity, IREE_SV("loom.serve.source-index.v1"));
  }

  loomc_link_index_builder_t* builder = NULL;
  iree_status_t status =
      iree_status_from_loomc(loomc_link_index_builder_allocate(
          context, NULL, loomc_allocator_from_iree(host_allocator), &builder));
  iree_string_view_t primary_name = iree_string_view_empty();
  if (iree_status_is_ok(status) && primary) {
    primary_name =
        iree_string_view_from_loomc(loomc_source_identifier(primary));
    status = loom_runtime_source_index_add_direct(
        builder, primary, LOOMC_LINK_PROVIDER_ROLE_INPUT, primary_name,
        identity);
  }
  for (iree_host_size_t i = 0; i < sources->count && iree_status_is_ok(status);
       ++i) {
    loomc_source_t* source = NULL;
    status = loom_runtime_sources_open(sources, sources->values[i],
                                       host_allocator, &source);
    if (iree_status_is_ok(status)) {
      const iree_string_view_t name =
          iree_string_view_from_loomc(loomc_source_identifier(source));
      if (!primary || !iree_string_view_equal(name, primary_name)) {
        status = loom_runtime_source_index_add_direct(
            builder, source, LOOMC_LINK_PROVIDER_ROLE_LIBRARY,
            sources->values[i], identity);
      }
    }
    loomc_source_release(source);
  }

  loom_runtime_source_index_import_t import = {0};
  if (sources->card) {
    const loom_model_selection_t* model =
        loom_model_card_selection(sources->card);
    const loom_model_provider_list_t lists[] = {model->inputs,
                                                model->libraries};
    for (iree_host_size_t l = 0;
         l < IREE_ARRAYSIZE(lists) && iree_status_is_ok(status); ++l) {
      for (iree_host_size_t i = 0;
           i < lists[l].count && iree_status_is_ok(status); ++i) {
        const loom_model_provider_t* provider = &lists[l].values[i];
        const loom_input_format_t* selected_format = NULL;
        status = loom_input_format_select(
            loom_configured_input_loomc_formats(), provider->file.format,
            provider->file.path, &selected_format, NULL);
        loomc_source_t* source = NULL;
        if (iree_status_is_ok(status)) {
          status = loom_runtime_sources_open(sources, provider->file.path,
                                             host_allocator, &source);
        }
        const bool duplicate =
            iree_status_is_ok(status) && primary &&
            iree_string_view_equal(
                iree_string_view_from_loomc(loomc_source_identifier(source)),
                primary_name);
        if (iree_status_is_ok(status) && !duplicate) {
          status = loom_runtime_source_index_add_provider(
              context, builder, sources, provider, selected_format, source,
              l == 0 ? LOOMC_LINK_PROVIDER_ROLE_INPUT
                     : LOOMC_LINK_PROVIDER_ROLE_LIBRARY,
              &import, report, host_allocator, identity);
        }
        loomc_source_release(source);
      }
    }
  }
  loom_runtime_source_index_import_deinitialize(&import);

  if (iree_status_is_ok(status)) {
    loomc_result_t* result = NULL;
    const loomc_status_t operation =
        loomc_link_index_builder_finish(builder, out_index, &result);
    status = loom_runtime_compilation_result(operation, result, report);
  }
  if (iree_status_is_ok(status) && identity) {
    loom_sha256_finalize(identity, out_identity);
  }
  loomc_link_index_builder_free(builder);
  return status;
}
