// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/binding/cli/lora_set.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "experimental/id4/tooling/runtime.h"
#include "iree/io/parameter_index.h"
#include "iree/io/parameter_index_provider.h"
#include "iree/tooling/parameter_util.h"

struct id4_cli_lora_set_t {
  // Allocator owning this set and its arrays.
  iree_allocator_t host_allocator;
  // Number of ordered adapters in the set.
  iree_host_size_t adapter_count;
  // Issue-time strengths indexed by adapter ordinal.
  float* strengths;
  // Providers indexed by adapter ordinal.
  iree_io_parameter_provider_t** providers;
  // Immutable topology copied from all imported catalogs.
  id4_ideogram4_lora_topology_t* topology;
};

void id4_cli_lora_set_release(id4_cli_lora_set_t* lora_set) {
  if (!lora_set) return;
  const iree_allocator_t host_allocator = lora_set->host_allocator;
  id4_ideogram4_lora_topology_release(lora_set->topology);
  if (lora_set->providers) {
    for (iree_host_size_t i = 0; i < lora_set->adapter_count; ++i) {
      iree_io_parameter_provider_release(lora_set->providers[i]);
    }
  }
  iree_allocator_free(host_allocator, lora_set->providers);
  iree_allocator_free(host_allocator, lora_set->strengths);
  iree_allocator_free(host_allocator, lora_set);
}

static iree_status_t id4_cli_lora_set_parse_strengths(
    iree_string_view_list_t paths, iree_string_view_list_t strength_strings,
    float* out_strengths) {
  if (paths.count == 0 && strength_strings.count != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--lora_strength requires at least one --lora");
  }
  if (strength_strings.count != 0 && strength_strings.count != paths.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--lora_strength must be omitted or repeated once per --lora; got "
        "%" PRIhsz " strengths for %" PRIhsz " adapters",
        strength_strings.count, paths.count);
  }
  for (iree_host_size_t i = 0; i < paths.count; ++i) {
    float strength = 1.0f;
    if (strength_strings.count != 0) {
      const iree_string_view_t value = strength_strings.values[i];
      char storage[32] = {0};
      if (iree_string_view_is_empty(value) || value.size >= sizeof(storage)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "--lora_strength value %" PRIhsz " must be a finite F32 number", i);
      }
      memcpy(storage, value.data, value.size);
      errno = 0;
      char* end = NULL;
      strength = strtof(storage, &end);
      if (end != storage + value.size || errno == ERANGE ||
          !isfinite(strength)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "--lora_strength value %" PRIhsz " must be a finite F32 number", i);
      }
    }
    out_strengths[i] = strength;
  }
  return iree_ok_status();
}

static iree_status_t id4_cli_lora_set_format_scope(
    iree_host_size_t adapter_ordinal, char* buffer,
    iree_host_size_t buffer_capacity, iree_string_view_t* out_scope) {
  const int length =
      snprintf(buffer, buffer_capacity, "lora_%" PRIhsz, adapter_ordinal);
  if (length < 0 || (iree_host_size_t)length >= buffer_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "LoRA provider scope overflow");
  }
  *out_scope = iree_make_string_view(buffer, (iree_host_size_t)length);
  return iree_ok_status();
}

static iree_status_t id4_cli_lora_set_import_adapter(
    id4_cli_lora_set_t* lora_set, const id4_ideogram4_dit_model_config_t* model,
    iree_host_size_t adapter_ordinal, iree_string_view_t path,
    id4_ideogram4_lora_t** out_lora) {
  *out_lora = NULL;
  if (iree_string_view_is_empty(path)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--lora value %" PRIhsz " is empty",
                            adapter_ordinal);
  }

  char scope_buffer[32];
  iree_string_view_t scope = iree_string_view_empty();
  iree_io_parameter_index_t* index = NULL;
  iree_status_t status = id4_cli_lora_set_format_scope(
      adapter_ordinal, scope_buffer, sizeof(scope_buffer), &scope);
  if (iree_status_is_ok(status)) {
    status = iree_io_parameter_index_create(lora_set->host_allocator, &index);
  }
  if (iree_status_is_ok(status)) {
    status = iree_tooling_append_parameter_file_to_index(
        path, index, lora_set->host_allocator);
  }
  if (iree_status_is_ok(status) && iree_io_parameter_index_count(index) == 0) {
    status = iree_make_status(IREE_STATUS_NOT_FOUND,
                              "LoRA file `%.*s` contained no parameters",
                              (int)path.size, path.data);
  }
  if (iree_status_is_ok(status)) {
    id4_ideogram4_lora_import_options_t import_options;
    memset(&import_options, 0, sizeof(import_options));
    import_options.structure_size = sizeof(import_options);
    import_options.model = *model;
    import_options.parameter_index = index;
    import_options.source_scope = scope;
    status = id4_ideogram4_lora_import(&import_options,
                                       lora_set->host_allocator, out_lora);
  }
  if (iree_status_is_ok(status)) {
    status = iree_io_parameter_index_provider_create(
        scope, index,
        IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS,
        lora_set->host_allocator, &lora_set->providers[adapter_ordinal]);
  }
  iree_io_parameter_index_release(index);
  if (!iree_status_is_ok(status)) {
    id4_ideogram4_lora_release(*out_lora);
    *out_lora = NULL;
  }
  return status;
}

iree_status_t id4_cli_lora_set_create(
    const id4_ideogram4_dit_model_config_t* model,
    iree_string_view_list_t paths, iree_string_view_list_t strength_strings,
    iree_allocator_t host_allocator, id4_cli_lora_set_t** out_lora_set) {
  IREE_ASSERT_ARGUMENT(out_lora_set);
  *out_lora_set = NULL;
  if (!model) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LoRA model configuration is required");
  }
  if ((paths.count != 0 && !paths.values) ||
      (strength_strings.count != 0 && !strength_strings.values)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "LoRA path and strength lists must have storage");
  }
  if (paths.count == 0) {
    return id4_cli_lora_set_parse_strengths(paths, strength_strings, NULL);
  }

  id4_cli_lora_set_t* lora_set = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*lora_set), (void**)&lora_set);
  if (iree_status_is_ok(status)) {
    memset(lora_set, 0, sizeof(*lora_set));
    lora_set->host_allocator = host_allocator;
    lora_set->adapter_count = paths.count;
    status = iree_allocator_malloc_array(host_allocator, paths.count,
                                         sizeof(lora_set->strengths[0]),
                                         (void**)&lora_set->strengths);
  }
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, paths.count,
                                         sizeof(lora_set->providers[0]),
                                         (void**)&lora_set->providers);
  }
  if (iree_status_is_ok(status)) {
    memset(lora_set->providers, 0,
           paths.count * sizeof(lora_set->providers[0]));
    status = id4_cli_lora_set_parse_strengths(paths, strength_strings,
                                              lora_set->strengths);
  }

  id4_ideogram4_lora_t** loras = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, paths.count,
                                         sizeof(loras[0]), (void**)&loras);
  }
  if (iree_status_is_ok(status)) {
    memset(loras, 0, paths.count * sizeof(loras[0]));
  }
  for (iree_host_size_t i = 0; i < paths.count && iree_status_is_ok(status);
       ++i) {
    status = id4_cli_lora_set_import_adapter(lora_set, model, i,
                                             paths.values[i], &loras[i]);
  }
  if (iree_status_is_ok(status)) {
    id4_ideogram4_lora_topology_create_options_t topology_options;
    memset(&topology_options, 0, sizeof(topology_options));
    topology_options.structure_size = sizeof(topology_options);
    topology_options.lora_count = paths.count;
    topology_options.loras = loras;
    status = id4_ideogram4_lora_topology_create(
        &topology_options, host_allocator, &lora_set->topology);
  }
  for (iree_host_size_t i = 0; i < paths.count; ++i) {
    id4_ideogram4_lora_release(loras ? loras[i] : NULL);
  }
  iree_allocator_free(host_allocator, loras);

  if (iree_status_is_ok(status)) {
    *out_lora_set = lora_set;
  } else {
    id4_cli_lora_set_release(lora_set);
  }
  return status;
}

id4_ideogram4_lora_topology_t* id4_cli_lora_set_topology(
    const id4_cli_lora_set_t* lora_set) {
  return lora_set ? lora_set->topology : NULL;
}

iree_host_size_t id4_cli_lora_set_adapter_count(
    const id4_cli_lora_set_t* lora_set) {
  return lora_set ? lora_set->adapter_count : 0;
}

const float* id4_cli_lora_set_strengths(const id4_cli_lora_set_t* lora_set) {
  return lora_set ? lora_set->strengths : NULL;
}

iree_status_t id4_cli_lora_set_create_conditioned_provider(
    const id4_cli_lora_set_t* lora_set, iree_string_view_t base_scope,
    iree_io_parameter_provider_t* base_provider,
    iree_allocator_t host_allocator,
    iree_io_parameter_provider_t** out_provider) {
  IREE_ASSERT_ARGUMENT(out_provider);
  *out_provider = NULL;
  if (!base_provider) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "conditioned-DiT base provider is required");
  }
  if (!lora_set) {
    iree_io_parameter_provider_retain(base_provider);
    *out_provider = base_provider;
    return iree_ok_status();
  }

  const iree_host_size_t entry_count = lora_set->adapter_count + 1;
  id4_tooling_parameter_provider_set_entry_t* entries = NULL;
  iree_status_t status = iree_allocator_malloc_array(
      host_allocator, entry_count, sizeof(entries[0]), (void**)&entries);
  if (iree_status_is_ok(status)) {
    memset(entries, 0, entry_count * sizeof(entries[0]));
    entries[0].scope = base_scope;
    entries[0].provider = base_provider;
    for (iree_host_size_t i = 0; i < lora_set->adapter_count; ++i) {
      entries[i + 1].scope = id4_ideogram4_lora_topology_adapter_source_scope(
          lora_set->topology, i);
      entries[i + 1].provider = lora_set->providers[i];
    }
    status = id4_tooling_create_parameter_provider_set(
        entry_count, entries, host_allocator, out_provider);
  }
  iree_allocator_free(host_allocator, entries);
  return status;
}
