// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/program/parameter_layout.h"

#include <string.h>

struct loom_serve_parameter_layout_t {
  // Allocator owning this header, retained rows, and trailing key bytes.
  iree_allocator_t host_allocator;
  // Canonical root footprint and alignment; its original binding is immaterial.
  loom_cmd_program_parameter_root_t root;
  // Number of retained parameter rows in compiler order within the domain.
  uint32_t count;
  // Canonical rows followed by owned, non-NUL-terminated key bytes.
  loom_cmd_program_parameter_t parameters[];
};

static bool loom_serve_parameter_placement_equal(
    loom_cmd_program_parameter_t lhs, loom_cmd_program_parameter_t rhs) {
  return iree_string_view_equal(lhs.key, rhs.key) &&
         lhs.byte_offset == rhs.byte_offset &&
         lhs.byte_length == rhs.byte_length &&
         lhs.minimum_alignment == rhs.minimum_alignment;
}

iree_status_t loom_serve_parameter_layout_check(
    const loom_cmd_program_t* baseline, const loom_cmd_program_t* candidate) {
  if (baseline->requirements.fixed_buffer_count !=
          candidate->requirements.fixed_buffer_count ||
      baseline->parameter_roots.count != candidate->parameter_roots.count ||
      baseline->parameters.count != candidate->parameters.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "stage parameter counts differ");
  }
  for (uint32_t i = 0; i < baseline->parameter_roots.count; ++i) {
    const loom_cmd_program_parameter_root_t lhs =
        loom_cmd_program_parameter_root_at(baseline, i);
    const loom_cmd_program_parameter_root_t rhs =
        loom_cmd_program_parameter_root_at(candidate, i);
    if (lhs.fixed_buffer_index != rhs.fixed_buffer_index ||
        lhs.required_byte_length != rhs.required_byte_length ||
        lhs.minimum_alignment != rhs.minimum_alignment) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "stage parameter root %u differs", i);
    }
  }
  for (uint32_t i = 0; i < baseline->parameters.count; ++i) {
    const loom_cmd_program_parameter_t lhs =
        loom_cmd_program_parameter_at(baseline, i);
    const loom_cmd_program_parameter_t rhs =
        loom_cmd_program_parameter_at(candidate, i);
    if (lhs.fixed_buffer_index != rhs.fixed_buffer_index ||
        !loom_serve_parameter_placement_equal(lhs, rhs)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "stage parameter placement %u differs", i);
    }
  }
  return iree_ok_status();
}

iree_status_t loom_serve_parameter_layout_allocate_domain(
    const loom_cmd_program_t* program, uint32_t domain,
    iree_allocator_t host_allocator,
    loom_serve_parameter_layout_t** out_layout) {
  *out_layout = NULL;
  uint32_t count = 0;
  iree_host_size_t size = sizeof(loom_serve_parameter_layout_t);
  bool fits = true;
  for (uint32_t i = 0; i < program->parameters.count && fits; ++i) {
    const loom_cmd_program_parameter_t parameter =
        loom_cmd_program_parameter_at(program, i);
    if (parameter.fixed_buffer_index != domain) {
      continue;
    }
    ++count;
    fits = iree_host_size_checked_add(size, parameter.key.size, &size);
  }
  fits &= iree_host_size_checked_mul_add(
      size, count, sizeof(loom_cmd_program_parameter_t), &size);
  if (!fits) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "parameter layout storage is not representable");
  }
  loom_serve_parameter_layout_t* layout = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, size, (void**)&layout));
  layout->host_allocator = host_allocator;
  layout->root = loom_cmd_program_parameter_root_at(program, domain);
  char* bytes = (char*)(layout->parameters + count);
  for (uint32_t i = 0; i < program->parameters.count; ++i) {
    loom_cmd_program_parameter_t parameter =
        loom_cmd_program_parameter_at(program, i);
    if (parameter.fixed_buffer_index != domain) {
      continue;
    }
    memcpy(bytes, parameter.key.data, parameter.key.size);
    parameter.key = iree_make_string_view(bytes, parameter.key.size);
    bytes += parameter.key.size;
    layout->parameters[layout->count++] = parameter;
  }
  *out_layout = layout;
  return iree_ok_status();
}

iree_status_t loom_serve_parameter_layout_check_domain(
    const loom_serve_parameter_layout_t* layout,
    const loom_cmd_program_t* program, uint32_t domain) {
  const loom_cmd_program_parameter_root_t root =
      loom_cmd_program_parameter_root_at(program, domain);
  if (root.required_byte_length != layout->root.required_byte_length ||
      root.minimum_alignment != layout->root.minimum_alignment) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "parameter arena bank footprint differs");
  }
  uint32_t cursor = 0;
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0;
       i < program->parameters.count && iree_status_is_ok(status); ++i) {
    const loom_cmd_program_parameter_t parameter =
        loom_cmd_program_parameter_at(program, i);
    if (parameter.fixed_buffer_index != domain) {
      continue;
    }
    if (cursor == layout->count || !loom_serve_parameter_placement_equal(
                                       layout->parameters[cursor], parameter)) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "parameter arena bank placement differs");
    }
    ++cursor;
  }
  if (iree_status_is_ok(status) && cursor != layout->count) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "parameter arena bank is missing rows");
  }
  return status;
}

void loom_serve_parameter_layout_free(loom_serve_parameter_layout_t* layout) {
  if (layout) {
    iree_allocator_free(layout->host_allocator, layout);
  }
}
