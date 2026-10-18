// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/testing/test_diagnostic.h"

#include <string.h>

static int loom_test_diagnostic_find_param_index(
    const loom_test_diagnostic_t* diagnostic, iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < diagnostic->param_count; ++i) {
    if (iree_string_view_equal(name, diagnostic->params[i].name)) {
      return (int)i;
    }
  }
  return -1;
}

bool loom_test_diagnostic_matches_annotation(
    const loom_test_diagnostic_t* diagnostic,
    const loom_test_annotation_t* annotation, iree_string_view_t filename) {
  if (!iree_string_view_equal(diagnostic->origin.filename, filename)) {
    return false;
  }
  if (diagnostic->severity != annotation->severity) {
    return false;
  }
  if (annotation->domain != LOOM_ERROR_DOMAIN_COUNT_ &&
      diagnostic->domain != annotation->domain) {
    return false;
  }
  if (annotation->code != 0 && diagnostic->code != annotation->code) {
    return false;
  }
  if (diagnostic->origin.line != (uint32_t)annotation->target_line) {
    return false;
  }
  for (uint8_t i = 0; i < annotation->message_substring_count; ++i) {
    if (iree_string_view_find(diagnostic->message,
                              annotation->message_substrings[i],
                              0) == IREE_STRING_VIEW_NPOS) {
      return false;
    }
  }
  for (uint8_t i = 0; i < annotation->param_match_count; ++i) {
    const loom_test_annotation_param_match_t* match =
        &annotation->param_matches[i];
    const int param_index =
        loom_test_diagnostic_find_param_index(diagnostic, match->name);
    if (param_index < 0 ||
        !iree_string_view_equal(diagnostic->params[param_index].value,
                                match->value)) {
      return false;
    }
  }
  return true;
}

typedef struct loom_test_diagnostic_match_context_t {
  // Diagnostics whose matched bits are updated by the assignment.
  loom_test_diagnostic_t* diagnostics;
  // Number of collected diagnostics.
  iree_host_size_t diagnostic_count;
  // Annotation constraints in source order.
  const loom_test_annotation_t* annotations;
  // Number of annotation constraints.
  iree_host_size_t annotation_count;
  // Main source identity to which unqualified annotations refer.
  iree_string_view_t filename;
  // Diagnostic assigned to each annotation, or IREE_HOST_SIZE_MAX.
  iree_host_size_t* annotation_to_diagnostic;
  // Annotations visited by the current augmenting-path search.
  bool* visited_annotations;
} loom_test_diagnostic_match_context_t;

static bool loom_test_diagnostic_match_augment(
    loom_test_diagnostic_match_context_t* context,
    iree_host_size_t diagnostic_index) {
  for (iree_host_size_t annotation_index = 0;
       annotation_index < context->annotation_count; ++annotation_index) {
    if (context->visited_annotations[annotation_index] ||
        !loom_test_diagnostic_matches_annotation(
            &context->diagnostics[diagnostic_index],
            &context->annotations[annotation_index], context->filename)) {
      continue;
    }
    context->visited_annotations[annotation_index] = true;
    const iree_host_size_t previous_diagnostic =
        context->annotation_to_diagnostic[annotation_index];
    if (previous_diagnostic == IREE_HOST_SIZE_MAX ||
        loom_test_diagnostic_match_augment(context, previous_diagnostic)) {
      context->annotation_to_diagnostic[annotation_index] = diagnostic_index;
      return true;
    }
  }
  return false;
}

iree_status_t loom_test_diagnostics_match_annotations(
    loom_test_diagnostic_t* diagnostics, iree_host_size_t diagnostic_count,
    const loom_test_annotation_t* annotations,
    iree_host_size_t annotation_count, iree_string_view_t filename,
    iree_arena_allocator_t* arena,
    iree_host_size_t** out_annotation_to_diagnostic) {
  *out_annotation_to_diagnostic = NULL;
  for (iree_host_size_t i = 0; i < diagnostic_count; ++i) {
    diagnostics[i].matched = false;
  }
  if (annotation_count == 0) {
    return iree_ok_status();
  }

  iree_host_size_t* annotation_to_diagnostic = NULL;
  bool* visited_annotations = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, annotation_count, sizeof(*annotation_to_diagnostic),
      (void**)&annotation_to_diagnostic));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, annotation_count,
                                                 sizeof(*visited_annotations),
                                                 (void**)&visited_annotations));
  for (iree_host_size_t i = 0; i < annotation_count; ++i) {
    annotation_to_diagnostic[i] = IREE_HOST_SIZE_MAX;
  }

  loom_test_diagnostic_match_context_t context = {
      .diagnostics = diagnostics,
      .diagnostic_count = diagnostic_count,
      .annotations = annotations,
      .annotation_count = annotation_count,
      .filename = filename,
      .annotation_to_diagnostic = annotation_to_diagnostic,
      .visited_annotations = visited_annotations,
  };
  for (iree_host_size_t diagnostic_index = 0;
       diagnostic_index < diagnostic_count; ++diagnostic_index) {
    memset(visited_annotations, 0,
           annotation_count * sizeof(*visited_annotations));
    loom_test_diagnostic_match_augment(&context, diagnostic_index);
  }
  for (iree_host_size_t annotation_index = 0;
       annotation_index < annotation_count; ++annotation_index) {
    const iree_host_size_t diagnostic_index =
        annotation_to_diagnostic[annotation_index];
    if (diagnostic_index != IREE_HOST_SIZE_MAX) {
      diagnostics[diagnostic_index].matched = true;
    }
  }

  *out_annotation_to_diagnostic = annotation_to_diagnostic;
  return iree_ok_status();
}
