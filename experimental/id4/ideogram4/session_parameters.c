// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/id4/ideogram4/session_parameters.h"

#include <stddef.h>
#include <string.h>

typedef struct id4_ideogram4_generation_parameter_source_descriptor_t {
  // Coarse stage receiving this parameter source.
  id4_ideogram4_generation_stage_ordinal_t stage_ordinal;
  // Human-readable model component used in diagnostics.
  iree_string_view_t name;
  // Byte offset of the source in the generation source catalog.
  iree_host_size_t source_offset;
} id4_ideogram4_generation_parameter_source_descriptor_t;

static const id4_ideogram4_generation_parameter_source_descriptor_t
    id4_ideogram4_generation_parameter_source_descriptors[] = {
        {ID4_IDEOGRAM4_GENERATION_STAGE_QWEN, IREE_SVL("Qwen"),
         offsetof(id4_ideogram4_generation_parameter_sources_t, qwen)},
        {ID4_IDEOGRAM4_GENERATION_STAGE_DIT_CONDITIONED,
         IREE_SVL("conditioned DiT"),
         offsetof(id4_ideogram4_generation_parameter_sources_t,
                  dit_conditioned)},
        {ID4_IDEOGRAM4_GENERATION_STAGE_DIT_UNCONDITIONED,
         IREE_SVL("unconditioned DiT"),
         offsetof(id4_ideogram4_generation_parameter_sources_t,
                  dit_unconditioned)},
        {ID4_IDEOGRAM4_GENERATION_STAGE_DECODE, IREE_SVL("VAE"),
         offsetof(id4_ideogram4_generation_parameter_sources_t, vae)},
};

static const id4_pipeline_parameter_source_t*
id4_ideogram4_generation_parameter_source_at(
    const id4_ideogram4_generation_parameter_sources_t* sources,
    const id4_ideogram4_generation_parameter_source_descriptor_t* descriptor) {
  return (const id4_pipeline_parameter_source_t*)((const uint8_t*)sources +
                                                  descriptor->source_offset);
}

static id4_pipeline_parameter_source_t*
id4_ideogram4_generation_mutable_parameter_source_at(
    id4_ideogram4_generation_parameter_sources_t* sources,
    const id4_ideogram4_generation_parameter_source_descriptor_t* descriptor) {
  return (id4_pipeline_parameter_source_t*)((uint8_t*)sources +
                                            descriptor->source_offset);
}

static iree_status_t id4_ideogram4_generation_parameter_source_validate(
    const id4_pipeline_parameter_source_t* source, iree_string_view_t name) {
  switch (source->kind) {
    case ID4_PIPELINE_PARAMETER_SOURCE_KIND_CHECKPOINT:
      if (!source->storage.checkpoint.provider) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Ideogram 4 generation %.*s checkpoint provider is required",
            (int)name.size, name.data);
      }
      return iree_ok_status();
    case ID4_PIPELINE_PARAMETER_SOURCE_KIND_EXECUTION_LAYOUT:
      if (!source->storage.execution_layout.index ||
          !source->storage.execution_layout.provider ||
          iree_string_view_is_empty(source->storage.execution_layout.scope) ||
          !iree_io_parameter_provider_query_support(
              source->storage.execution_layout.provider,
              source->storage.execution_layout.scope)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Ideogram 4 generation %.*s execution-layout source requires an "
            "index, provider, and supported scope",
            (int)name.size, name.data);
      }
      return iree_ok_status();
    case ID4_PIPELINE_PARAMETER_SOURCE_KIND_RESIDENT:
      if (!source->storage.resident.slabs) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Ideogram 4 generation %.*s resident parameter slabs are "
            "required",
            (int)name.size, name.data);
      }
      return iree_ok_status();
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram 4 generation %.*s parameter source kind %u is invalid",
          (int)name.size, name.data, source->kind);
  }
}

static void id4_ideogram4_generation_parameter_source_retain(
    const id4_pipeline_parameter_source_t* source) {
  switch (source->kind) {
    case ID4_PIPELINE_PARAMETER_SOURCE_KIND_CHECKPOINT:
      iree_io_parameter_provider_retain(source->storage.checkpoint.provider);
      break;
    case ID4_PIPELINE_PARAMETER_SOURCE_KIND_EXECUTION_LAYOUT:
      iree_io_parameter_index_retain(source->storage.execution_layout.index);
      iree_io_parameter_provider_retain(
          source->storage.execution_layout.provider);
      break;
    case ID4_PIPELINE_PARAMETER_SOURCE_KIND_RESIDENT:
      id4_pipeline_parameter_slab_set_retain(source->storage.resident.slabs);
      break;
    default:
      break;
  }
}

static void id4_ideogram4_generation_parameter_source_release(
    const id4_pipeline_parameter_source_t* source) {
  switch (source->kind) {
    case ID4_PIPELINE_PARAMETER_SOURCE_KIND_CHECKPOINT:
      iree_io_parameter_provider_release(source->storage.checkpoint.provider);
      break;
    case ID4_PIPELINE_PARAMETER_SOURCE_KIND_EXECUTION_LAYOUT:
      iree_io_parameter_provider_release(
          source->storage.execution_layout.provider);
      iree_io_parameter_index_release(source->storage.execution_layout.index);
      break;
    case ID4_PIPELINE_PARAMETER_SOURCE_KIND_RESIDENT:
      id4_pipeline_parameter_slab_set_release(source->storage.resident.slabs);
      break;
    default:
      break;
  }
}

iree_status_t id4_ideogram4_generation_parameter_sources_validate(
    const id4_ideogram4_generation_parameter_sources_t* sources) {
  if (!sources) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Ideogram 4 generation parameter sources are required");
  }
  id4_pipeline_parameter_source_kind_t common_kind =
      ID4_PIPELINE_PARAMETER_SOURCE_KIND_INVALID;
  for (iree_host_size_t i = 0;
       i <
       IREE_ARRAYSIZE(id4_ideogram4_generation_parameter_source_descriptors);
       ++i) {
    const id4_ideogram4_generation_parameter_source_descriptor_t* descriptor =
        &id4_ideogram4_generation_parameter_source_descriptors[i];
    const id4_pipeline_parameter_source_t* source =
        id4_ideogram4_generation_parameter_source_at(sources, descriptor);
    IREE_RETURN_IF_ERROR(id4_ideogram4_generation_parameter_source_validate(
        source, descriptor->name));
    if (source->kind == ID4_PIPELINE_PARAMETER_SOURCE_KIND_RESIDENT) {
      continue;
    }
    if (common_kind == ID4_PIPELINE_PARAMETER_SOURCE_KIND_INVALID) {
      common_kind = source->kind;
    } else if (source->kind != common_kind) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Ideogram 4 generation checkpoint and execution-layout sources "
          "cannot be mixed");
    }
  }
  return iree_ok_status();
}

iree_status_t id4_ideogram4_generation_parameter_sources_clone(
    const id4_ideogram4_generation_parameter_sources_t* sources,
    iree_allocator_t host_allocator,
    id4_ideogram4_generation_parameter_sources_t* out_sources,
    char** out_scope_storage) {
  memset(out_sources, 0, sizeof(*out_sources));
  *out_scope_storage = NULL;
  IREE_RETURN_IF_ERROR(
      id4_ideogram4_generation_parameter_sources_validate(sources));

  iree_host_size_t scope_storage_length = 0;
  for (iree_host_size_t i = 0;
       i <
       IREE_ARRAYSIZE(id4_ideogram4_generation_parameter_source_descriptors);
       ++i) {
    const id4_pipeline_parameter_source_t* source =
        id4_ideogram4_generation_parameter_source_at(
            sources, &id4_ideogram4_generation_parameter_source_descriptors[i]);
    if (source->kind != ID4_PIPELINE_PARAMETER_SOURCE_KIND_EXECUTION_LAYOUT) {
      continue;
    }
    if (!iree_host_size_checked_add(scope_storage_length,
                                    source->storage.execution_layout.scope.size,
                                    &scope_storage_length)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "Ideogram 4 generation parameter source scope storage overflows");
    }
  }

  char* scope_storage = NULL;
  if (scope_storage_length != 0) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
        host_allocator, scope_storage_length, (void**)&scope_storage));
  }
  *out_sources = *sources;
  char* scope_cursor = scope_storage;
  for (iree_host_size_t i = 0;
       i <
       IREE_ARRAYSIZE(id4_ideogram4_generation_parameter_source_descriptors);
       ++i) {
    const id4_ideogram4_generation_parameter_source_descriptor_t* descriptor =
        &id4_ideogram4_generation_parameter_source_descriptors[i];
    id4_pipeline_parameter_source_t* source =
        id4_ideogram4_generation_mutable_parameter_source_at(out_sources,
                                                             descriptor);
    if (source->kind == ID4_PIPELINE_PARAMETER_SOURCE_KIND_EXECUTION_LAYOUT) {
      const iree_string_view_t source_scope =
          source->storage.execution_layout.scope;
      memcpy(scope_cursor, source_scope.data, source_scope.size);
      source->storage.execution_layout.scope =
          iree_make_string_view(scope_cursor, source_scope.size);
      scope_cursor += source_scope.size;
    }
    id4_ideogram4_generation_parameter_source_retain(source);
  }
  *out_scope_storage = scope_storage;
  return iree_ok_status();
}

void id4_ideogram4_generation_parameter_sources_deinitialize(
    id4_ideogram4_generation_parameter_sources_t* sources, char* scope_storage,
    iree_allocator_t host_allocator) {
  for (iree_host_size_t i = 0;
       i <
       IREE_ARRAYSIZE(id4_ideogram4_generation_parameter_source_descriptors);
       ++i) {
    const id4_pipeline_parameter_source_t* source =
        id4_ideogram4_generation_parameter_source_at(
            sources, &id4_ideogram4_generation_parameter_source_descriptors[i]);
    id4_ideogram4_generation_parameter_source_release(source);
  }
  iree_allocator_free(host_allocator, scope_storage);
  memset(sources, 0, sizeof(*sources));
}

const id4_pipeline_parameter_source_t*
id4_ideogram4_generation_parameter_source_for_stage(
    const id4_ideogram4_generation_parameter_sources_t* sources,
    id4_ideogram4_generation_stage_ordinal_t stage_ordinal) {
  for (iree_host_size_t i = 0;
       i <
       IREE_ARRAYSIZE(id4_ideogram4_generation_parameter_source_descriptors);
       ++i) {
    const id4_ideogram4_generation_parameter_source_descriptor_t* descriptor =
        &id4_ideogram4_generation_parameter_source_descriptors[i];
    if (descriptor->stage_ordinal == stage_ordinal) {
      return id4_ideogram4_generation_parameter_source_at(sources, descriptor);
    }
  }
  return NULL;
}

bool id4_ideogram4_resident_parameter_cache_entry_matches(
    const id4_ideogram4_resident_parameter_cache_entry_t* entry,
    const id4_pipeline_parameter_source_t* source,
    const id4_pipeline_plan_t* plan) {
  if (!entry->slabs || !source || entry->source.kind != source->kind ||
      !id4_pipeline_plan_matches_resident_parameter_slabs(plan, entry->slabs)) {
    return false;
  }
  switch (source->kind) {
    case ID4_PIPELINE_PARAMETER_SOURCE_KIND_CHECKPOINT:
      return entry->source.storage.checkpoint.provider ==
             source->storage.checkpoint.provider;
    case ID4_PIPELINE_PARAMETER_SOURCE_KIND_EXECUTION_LAYOUT:
      return entry->source.storage.execution_layout.index ==
                 source->storage.execution_layout.index &&
             entry->source.storage.execution_layout.provider ==
                 source->storage.execution_layout.provider &&
             iree_string_view_equal(
                 entry->source.storage.execution_layout.scope,
                 source->storage.execution_layout.scope);
    default:
      return false;
  }
}

void id4_ideogram4_resident_parameter_cache_entry_deinitialize(
    id4_ideogram4_resident_parameter_cache_entry_t* entry,
    iree_allocator_t host_allocator) {
  id4_pipeline_parameter_slab_set_release(entry->slabs);
  id4_ideogram4_generation_parameter_source_release(&entry->source);
  iree_allocator_free(host_allocator, entry->execution_layout_scope_storage);
  memset(entry, 0, sizeof(*entry));
}

iree_status_t id4_ideogram4_resident_parameter_cache_entry_assign(
    id4_ideogram4_resident_parameter_cache_entry_t* entry,
    const id4_pipeline_parameter_source_t* source,
    id4_pipeline_parameter_slab_set_t* slabs, iree_allocator_t host_allocator) {
  if (!source || !slabs) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "resident parameter cache source and slabs are required");
  }
  if (source->kind == ID4_PIPELINE_PARAMETER_SOURCE_KIND_RESIDENT) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "resident parameter cache only accepts materializable sources");
  }
  IREE_RETURN_IF_ERROR(id4_ideogram4_generation_parameter_source_validate(
      source, IREE_SV("cached stage")));

  id4_ideogram4_resident_parameter_cache_entry_t new_entry;
  memset(&new_entry, 0, sizeof(new_entry));
  new_entry.source = *source;
  if (source->kind == ID4_PIPELINE_PARAMETER_SOURCE_KIND_EXECUTION_LAYOUT) {
    const iree_string_view_t scope = source->storage.execution_layout.scope;
    IREE_RETURN_IF_ERROR(iree_allocator_clone(
        host_allocator, iree_make_const_byte_span(scope.data, scope.size),
        (void**)&new_entry.execution_layout_scope_storage));
    new_entry.source.storage.execution_layout.scope = iree_make_string_view(
        new_entry.execution_layout_scope_storage, scope.size);
  }
  id4_ideogram4_generation_parameter_source_retain(&new_entry.source);
  new_entry.slabs = slabs;
  id4_pipeline_parameter_slab_set_retain(new_entry.slabs);

  id4_ideogram4_resident_parameter_cache_entry_deinitialize(entry,
                                                            host_allocator);
  *entry = new_entry;
  return iree_ok_status();
}
