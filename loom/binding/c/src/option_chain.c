// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "option_chain.h"

#include "loomc/iree.h"
#include "target.h"

static loomc_status_t loomc_sanitizer_reporting_mode_to_internal(
    loomc_sanitizer_reporting_mode_t reporting_mode,
    loom_sanitizer_reporting_mode_t* out_reporting_mode) {
  *out_reporting_mode = LOOM_SANITIZER_REPORTING_MODE_DEFAULT;
  switch (reporting_mode) {
    case LOOMC_SANITIZER_REPORTING_MODE_TRAP:
      *out_reporting_mode = LOOM_SANITIZER_REPORTING_MODE_TRAP;
      return loomc_ok_status();
    case LOOMC_SANITIZER_REPORTING_MODE_REPORT_ONLY:
      *out_reporting_mode = LOOM_SANITIZER_REPORTING_MODE_REPORT_ONLY;
      return loomc_ok_status();
    case LOOMC_SANITIZER_REPORTING_MODE_DEFAULT:
      return loomc_ok_status();
    default:
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "sanitizer options contain unknown reporting mode");
  }
}

typedef struct loomc_descriptor_prefix_t {
  // Structure type identifying the descriptor.
  loomc_structure_type_t type;

  // Size of the descriptor in bytes.
  loomc_host_size_t structure_size;

  // Next descriptor in the option extension chain.
  const void* next;
} loomc_descriptor_prefix_t;

loomc_status_t loomc_sanitizer_options_resolve(
    const loomc_sanitizer_options_t* options,
    loom_sanitizer_options_t* out_options) {
  *out_options = (loom_sanitizer_options_t){0};
  if (options == NULL) {
    return loomc_ok_status();
  }
  if (options->type != LOOMC_STRUCTURE_TYPE_SANITIZER_OPTIONS) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "sanitizer options have an unknown structure type");
  }
  if (options->structure_size != 0 &&
      options->structure_size < sizeof(*options)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "sanitizer options structure_size is too small");
  }

  loom_sanitizer_reporting_mode_t reporting_mode =
      LOOM_SANITIZER_REPORTING_MODE_DEFAULT;
  LOOMC_RETURN_IF_ERROR(loomc_sanitizer_reporting_mode_to_internal(
      options->reporting_mode, &reporting_mode));
  const loom_sanitizer_options_t resolved_options = {
      .checks = options->checks,
      .flags = options->flags,
      .reporting_mode = reporting_mode,
  };
  LOOMC_RETURN_IF_ERROR(loomc_status_from_iree(
      loom_sanitizer_options_validate(&resolved_options)));
  *out_options = resolved_options;
  return loomc_ok_status();
}

loomc_status_t loomc_option_chain_resolve(
    const void* next, loomc_option_chain_allowed_t allowed_options,
    loomc_option_chain_t* out_options) {
  if (out_options == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_options must not be NULL");
  }
  *out_options = (loomc_option_chain_t){0};
  while (next != NULL) {
    const loomc_descriptor_prefix_t* prefix =
        (const loomc_descriptor_prefix_t*)next;
    switch (prefix->type) {
      case LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS: {
        if (!iree_all_bits_set(allowed_options,
                               LOOMC_OPTION_CHAIN_ALLOW_TARGET_SELECTION)) {
          return loomc_make_status(
              LOOMC_STATUS_UNIMPLEMENTED,
              "target selection option extension is not supported here");
        }
        if (out_options->target_selection != NULL) {
          return loomc_make_status(
              LOOMC_STATUS_INVALID_ARGUMENT,
              "option chain contains duplicate target selection options");
        }
        const loomc_target_selection_options_t* target_options =
            (const loomc_target_selection_options_t*)next;
        LOOMC_RETURN_IF_ERROR(
            loomc_target_selection_options_validate(target_options));
        out_options->target_selection = target_options->target_selection;
        next = target_options->next;
        break;
      }
      case LOOMC_STRUCTURE_TYPE_SANITIZER_OPTIONS: {
        if (!iree_all_bits_set(allowed_options,
                               LOOMC_OPTION_CHAIN_ALLOW_SANITIZER)) {
          return loomc_make_status(
              LOOMC_STATUS_UNIMPLEMENTED,
              "sanitizer option extension is not supported here");
        }
        if (out_options->has_sanitizer) {
          return loomc_make_status(
              LOOMC_STATUS_INVALID_ARGUMENT,
              "option chain contains duplicate sanitizer options");
        }
        const loomc_sanitizer_options_t* sanitizer_options =
            (const loomc_sanitizer_options_t*)next;
        LOOMC_RETURN_IF_ERROR(loomc_sanitizer_options_resolve(
            sanitizer_options, &out_options->sanitizer));
        out_options->has_sanitizer = true;
        next = sanitizer_options->next;
        break;
      }
      case LOOMC_STRUCTURE_TYPE_NONE:
        return loomc_make_status(
            LOOMC_STATUS_INVALID_ARGUMENT,
            "option extension is missing a structure type");
      default:
        return loomc_make_status(LOOMC_STATUS_UNIMPLEMENTED,
                                 "option extension type is not supported");
    }
  }
  return loomc_ok_status();
}
