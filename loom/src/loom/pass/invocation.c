// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/invocation.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/pass/ops.h"

static const char* loom_pass_kind_display_name(loom_pass_kind_t kind) {
  switch (kind) {
    case LOOM_PASS_MODULE:
      return "module";
    case LOOM_PASS_FUNCTION:
      return "function";
    default:
      return "unknown";
  }
}

static iree_status_t loom_pass_string_from_id(const loom_module_t* module,
                                              loom_string_id_t string_id,
                                              const char* label,
                                              iree_string_view_t* out_string) {
  if (!module || !out_string || string_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid %s string id", label);
  }
  *out_string = module->strings.entries[string_id];
  return iree_ok_status();
}

iree_status_t loom_pass_invocation_resolve_run_op(
    const loom_module_t* module, const loom_pass_registry_t* registry,
    const loom_op_t* op, loom_pass_kind_t required_kind,
    iree_arena_allocator_t* arena, loom_pass_invocation_t* out_invocation) {
  memset(out_invocation, 0, sizeof(*out_invocation));
  if (!loom_pass_run_isa(op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected pass.run op");
  }
  iree_string_view_t key = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(loom_pass_string_from_id(module, loom_pass_run_key(op),
                                                "pass key", &key));

  const loom_pass_descriptor_t* descriptor = NULL;
  IREE_RETURN_IF_ERROR(loom_pass_registry_lookup(registry, key, &descriptor));
  if (!descriptor) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "unknown pass '%.*s'",
                            (int)key.size, key.data);
  }
  if (!loom_pass_descriptor_is_available(descriptor)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "pass '%.*s' is unavailable: %.*s",
                            (int)descriptor->key.size, descriptor->key.data,
                            (int)descriptor->unavailable_reason.size,
                            descriptor->unavailable_reason.data);
  }

  const loom_pass_info_t* info = descriptor->info ? descriptor->info() : NULL;
  if (!info) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pass descriptor '%.*s' returned no info",
                            (int)descriptor->key.size, descriptor->key.data);
  }
  if (required_kind != LOOM_PASS_COUNT_ && info->kind != required_kind) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pass '%.*s' requires %s anchor but current pipeline anchor is %s",
        (int)descriptor->key.size, descriptor->key.data,
        loom_pass_kind_display_name(info->kind),
        loom_pass_kind_display_name(required_kind));
  }

  loom_pass_decoded_options_t decoded_options = {0};
  IREE_RETURN_IF_ERROR(loom_pass_descriptor_decode_attr_options(
      descriptor, module, loom_pass_run_options(op), arena, &decoded_options));

  *out_invocation = (loom_pass_invocation_t){
      .source_op = op,
      .key = key,
      .descriptor = descriptor,
      .info = info,
      .decoded_options = decoded_options,
  };
  return iree_ok_status();
}
