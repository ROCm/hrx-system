// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/binding/c/target/cmd/program.h"

#include <string.h>

#include "iree/base/alignment.h"
#include "iree/base/internal/atomics.h"
#include "loomc/iree.h"

struct loomc_cmd_program_package_t {
  // Atomic reference count for immutable shared ownership.
  iree_atomic_ref_count_t ref_count;

  // Allocator used for this coallocated package object.
  loomc_allocator_t allocator;

  // Optional release callback owning externally stored package bytes.
  loomc_artifact_release_fn_t release;

  // Caller value passed to |release|.
  void* release_user_data;

  // Original or package-owned canonical bytes.
  loomc_byte_span_t contents;

  // Validated package view borrowing |contents|.
  loom_cmd_program_package_t parsed;

  // Decoded exports borrowing |contents|.
  loom_cmd_program_package_export_t* exports;
};

typedef struct loomc_cmd_program_info_prefix_t {
  // Structure type identifying the output descriptor.
  loomc_structure_type_t type;

  // Size of the complete output descriptor in bytes.
  loomc_host_size_t structure_size;

  // Reserved extension chain.
  void* next;
} loomc_cmd_program_info_prefix_t;

static bool loomc_cmd_program_string_view_is_well_formed(
    loomc_string_view_t value) {
  return value.data != NULL || value.size == 0;
}

static loomc_status_t loomc_cmd_program_validate_output_info(
    const void* info, loomc_structure_type_t expected_type,
    loomc_host_size_t expected_size) {
  if (info == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_info must not be NULL");
  }
  const loomc_cmd_program_info_prefix_t* prefix =
      (const loomc_cmd_program_info_prefix_t*)info;
  if (prefix->type != LOOMC_STRUCTURE_TYPE_NONE &&
      prefix->type != expected_type) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "command program output has an unknown structure type");
  }
  if (prefix->structure_size != 0 && prefix->structure_size < expected_size) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "command program output structure_size is too small");
  }
  if (prefix->next != NULL) {
    return loomc_make_status(
        LOOMC_STATUS_UNIMPLEMENTED,
        "command program output extensions are not supported");
  }
  return loomc_ok_status();
}

static void loomc_cmd_program_package_rebase(
    const uint8_t* source_base, uint8_t* target_base,
    loom_cmd_program_package_t* package) {
  package->storage =
      iree_make_const_byte_span(target_base, package->storage.data_length);
  package->export_table = target_base + (package->export_table - source_base);
  package->entry_table = target_base + (package->entry_table - source_base);
  package->strings.data = target_base + (package->strings.data - source_base);
  package->payloads.data = target_base + (package->payloads.data - source_base);
}

static loomc_status_t loomc_cmd_program_package_load_impl(
    const loomc_artifact_t* artifact, loomc_artifact_release_fn_t release,
    void* release_user_data, loomc_allocator_t allocator,
    loomc_cmd_program_package_t** out_package) {
  if (out_package == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_package must not be NULL");
  }
  *out_package = NULL;
  if (artifact == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "artifact must not be NULL");
  }
  if (artifact->kind != LOOMC_ARTIFACT_KIND_EXECUTABLE) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "artifact is not a command program package");
  }
  if (!loomc_cmd_program_string_view_is_well_formed(artifact->format) ||
      !loomc_cmd_program_string_view_is_well_formed(artifact->identifier)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "artifact string view is malformed");
  }
  if (!loomc_string_view_equal(
          artifact->format,
          loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_COMMAND_PACKAGE))) {
    return loomc_make_status(LOOMC_STATUS_UNIMPLEMENTED,
                             "artifact is not in command package format");
  }
  if (artifact->contents.data == NULL && artifact->contents.data_length != 0) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "artifact contents have length but no data");
  }

  const iree_const_byte_span_t source_data =
      iree_const_byte_span_from_loomc(artifact->contents);
  loom_cmd_program_package_t parsed = {0};
  LOOMC_RETURN_IF_ERROR(loomc_status_from_iree(
      loom_cmd_program_package_parse(source_data, &parsed)));

  const iree_host_size_t copied_byte_length =
      release == NULL ? source_data.data_length : 0;
  iree_host_size_t total_size = 0;
  iree_host_size_t exports_offset = 0;
  iree_host_size_t contents_offset = 0;
  LOOMC_RETURN_IF_ERROR(loomc_status_from_iree(IREE_STRUCT_LAYOUT(
      sizeof(loomc_cmd_program_package_t), &total_size,
      IREE_STRUCT_FIELD_ALIGNED(
          parsed.export_count, loom_cmd_program_package_export_t,
          iree_alignof(loom_cmd_program_package_export_t), &exports_offset),
      IREE_STRUCT_FIELD(copied_byte_length, uint8_t, &contents_offset))));

  loomc_cmd_program_package_t* package = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_allocator_malloc_uninitialized(
      allocator, total_size, (void**)&package));
  iree_atomic_ref_count_init(&package->ref_count);
  package->allocator = allocator;
  package->release = release;
  package->release_user_data = release_user_data;
  package->parsed = parsed;
  package->exports =
      (loom_cmd_program_package_export_t*)((uint8_t*)package + exports_offset);

  if (release == NULL) {
    uint8_t* copied_data = (uint8_t*)package + contents_offset;
    memcpy(copied_data, source_data.data, source_data.data_length);
    package->contents =
        loomc_make_byte_span(copied_data, source_data.data_length);
    loomc_cmd_program_package_rebase(source_data.data, copied_data,
                                     &package->parsed);
  } else {
    package->contents = artifact->contents;
  }
  for (uint32_t i = 0; i < package->parsed.export_count; ++i) {
    package->exports[i] =
        loom_cmd_program_package_export_at(&package->parsed, i);
  }

  *out_package = package;
  return loomc_ok_status();
}

loomc_status_t loomc_cmd_program_package_load(
    const loomc_artifact_t* artifact, loomc_artifact_release_fn_t release,
    void* release_user_data, loomc_allocator_t allocator,
    loomc_cmd_program_package_t** out_package) {
  const loomc_byte_span_t contents =
      artifact != NULL ? artifact->contents : loomc_byte_span_empty();
  loomc_status_t status = loomc_cmd_program_package_load_impl(
      artifact, release, release_user_data, allocator, out_package);
  if (!loomc_status_is_ok(status) && release != NULL && artifact != NULL) {
    release(release_user_data, contents);
  }
  return status;
}

void loomc_cmd_program_package_retain(loomc_cmd_program_package_t* package) {
  if (package == NULL) return;
  iree_atomic_ref_count_inc(&package->ref_count);
}

void loomc_cmd_program_package_release(loomc_cmd_program_package_t* package) {
  if (package == NULL) return;
  if (iree_atomic_ref_count_dec(&package->ref_count) == 1) {
    const loomc_allocator_t allocator = package->allocator;
    if (package->release != NULL) {
      package->release(package->release_user_data, package->contents);
    }
    loomc_allocator_free(allocator, package);
  }
}

loomc_host_size_t loomc_cmd_program_package_export_count(
    const loomc_cmd_program_package_t* package) {
  return package != NULL ? package->parsed.export_count : 0;
}

loomc_cmd_program_export_t loomc_cmd_program_package_export_at(
    const loomc_cmd_program_package_t* package, loomc_host_size_t index) {
  if (package == NULL || index >= package->parsed.export_count) {
    return loomc_cmd_program_export_invalid();
  }
  return (loomc_cmd_program_export_t){.value = index};
}

const loom_cmd_program_package_t* loomc_cmd_program_package_parsed(
    const loomc_cmd_program_package_t* package) {
  return package != NULL ? &package->parsed : NULL;
}

const loom_cmd_program_package_export_t*
loomc_cmd_program_package_resolve_export(
    const loomc_cmd_program_package_t* package,
    loomc_cmd_program_export_t program_export) {
  if (package == NULL || program_export.value >= package->parsed.export_count) {
    return NULL;
  }
  return &package->exports[program_export.value];
}

loomc_status_t loomc_cmd_program_package_lookup_export(
    const loomc_cmd_program_package_t* package, loomc_string_view_t name,
    loomc_cmd_program_export_t* out_export) {
  if (out_export == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_export must not be NULL");
  }
  *out_export = loomc_cmd_program_export_invalid();
  if (package == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "package must not be NULL");
  }
  if (!loomc_cmd_program_string_view_is_well_formed(name) || name.size == 0 ||
      name.data[0] == '@') {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "name must be a non-empty export name without a leading '@'");
  }
  for (uint32_t i = 0; i < package->parsed.export_count; ++i) {
    if (iree_string_view_equal(package->exports[i].name,
                               iree_string_view_from_loomc(name))) {
      *out_export = (loomc_cmd_program_export_t){.value = i};
      return loomc_ok_status();
    }
  }
  return loomc_make_status(LOOMC_STATUS_NOT_FOUND,
                           "command program export was not found");
}

loomc_status_t loomc_cmd_program_package_export_info(
    const loomc_cmd_program_package_t* package,
    loomc_cmd_program_export_t program_export,
    loomc_cmd_program_info_t* out_info) {
  LOOMC_RETURN_IF_ERROR(loomc_cmd_program_validate_output_info(
      out_info, LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_INFO, sizeof(*out_info)));
  const loom_cmd_program_package_export_t* package_export =
      loomc_cmd_program_package_resolve_export(package, program_export);
  if (package_export == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "command program export token is invalid");
  }
  const loom_cmd_program_t* program = &package_export->program;
  *out_info = (loomc_cmd_program_info_t){
      .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_INFO,
      .structure_size = sizeof(*out_info),
      .name = loomc_string_view_from_iree(package_export->name),
      .fixed_buffer_count = program->requirements.fixed_buffer_count,
      .rebindable_binding_count =
          program->requirements.rebindable_binding_count,
      .executable_count = program->requirements.executable_count,
      .entry_count = program->requirements.entry_count,
      .parameter_root_count = program->parameter_roots.count,
      .parameter_count = program->parameters.count,
      .command_count = program->commands.count,
      .transient =
          {
              .binding_index = program->requirements.transient.binding_index,
              .required_byte_length =
                  program->requirements.transient.required_byte_length,
              .minimum_alignment =
                  program->requirements.transient.minimum_alignment,
          },
      .config =
          {
              .binding_index =
                  program->requirements.launch_counts.binding_index,
              .required_byte_length =
                  program->requirements.launch_counts.required_byte_length,
              .minimum_alignment =
                  program->requirements.launch_counts.minimum_alignment,
          },
  };
  return loomc_ok_status();
}

loomc_status_t loomc_cmd_program_package_entry_info(
    const loomc_cmd_program_package_t* package,
    loomc_cmd_program_export_t program_export, loomc_host_size_t index,
    loomc_cmd_program_entry_info_t* out_info) {
  LOOMC_RETURN_IF_ERROR(loomc_cmd_program_validate_output_info(
      out_info, LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_ENTRY_INFO,
      sizeof(*out_info)));
  const loom_cmd_program_package_export_t* package_export =
      loomc_cmd_program_package_resolve_export(package, program_export);
  if (package_export == NULL || index >= package_export->entry_count) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "command program entry index is out of range");
  }
  const loom_cmd_program_package_entry_t entry =
      loom_cmd_program_package_export_entry_at(&package->parsed, package_export,
                                               (uint32_t)index);
  *out_info = (loomc_cmd_program_entry_info_t){
      .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_ENTRY_INFO,
      .structure_size = sizeof(*out_info),
      .executable_index = entry.executable_index,
      .name = loomc_string_view_from_iree(entry.name),
  };
  return loomc_ok_status();
}

loomc_status_t loomc_cmd_program_package_parameter_root_info(
    const loomc_cmd_program_package_t* package,
    loomc_cmd_program_export_t program_export, loomc_host_size_t index,
    loomc_cmd_program_parameter_root_info_t* out_info) {
  LOOMC_RETURN_IF_ERROR(loomc_cmd_program_validate_output_info(
      out_info, LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PARAMETER_ROOT_INFO,
      sizeof(*out_info)));
  const loom_cmd_program_package_export_t* package_export =
      loomc_cmd_program_package_resolve_export(package, program_export);
  if (package_export == NULL ||
      index >= package_export->program.parameter_roots.count) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "command program parameter-root index is out of range");
  }
  const loom_cmd_program_parameter_root_t root =
      loom_cmd_program_parameter_root_at(&package_export->program,
                                         (uint32_t)index);
  *out_info = (loomc_cmd_program_parameter_root_info_t){
      .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PARAMETER_ROOT_INFO,
      .structure_size = sizeof(*out_info),
      .fixed_buffer_index = root.fixed_buffer_index,
      .required_byte_length = root.required_byte_length,
      .minimum_alignment = root.minimum_alignment,
  };
  return loomc_ok_status();
}

loomc_status_t loomc_cmd_program_package_parameter_info(
    const loomc_cmd_program_package_t* package,
    loomc_cmd_program_export_t program_export, loomc_host_size_t index,
    loomc_cmd_program_parameter_info_t* out_info) {
  LOOMC_RETURN_IF_ERROR(loomc_cmd_program_validate_output_info(
      out_info, LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PARAMETER_INFO,
      sizeof(*out_info)));
  const loom_cmd_program_package_export_t* package_export =
      loomc_cmd_program_package_resolve_export(package, program_export);
  if (package_export == NULL ||
      index >= package_export->program.parameters.count) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "command program parameter index is out of range");
  }
  const loom_cmd_program_parameter_t parameter =
      loom_cmd_program_parameter_at(&package_export->program, (uint32_t)index);
  *out_info = (loomc_cmd_program_parameter_info_t){
      .type = LOOMC_STRUCTURE_TYPE_CMD_PROGRAM_PARAMETER_INFO,
      .structure_size = sizeof(*out_info),
      .key = loomc_string_view_from_iree(parameter.key),
      .fixed_buffer_index = parameter.fixed_buffer_index,
      .byte_offset = parameter.byte_offset,
      .byte_length = parameter.byte_length,
      .minimum_alignment = parameter.minimum_alignment,
  };
  return loomc_ok_status();
}
