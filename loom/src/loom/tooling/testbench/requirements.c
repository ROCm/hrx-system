// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/testbench/requirements.h"

#include <string.h>

#include "loom/ops/check/ops.h"

static iree_string_view_t loom_testbench_requirement_module_string(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

iree_string_view_t loom_testbench_requirement_op_kind_name(
    loom_testbench_requirement_op_kind_t op_kind) {
  switch (op_kind) {
    case LOOM_TESTBENCH_REQUIREMENT_OP_KIND_NONE:
      return IREE_SV("none");
    case LOOM_TESTBENCH_REQUIREMENT_OP_KIND_REQUIRES:
      return IREE_SV("requires");
    case LOOM_TESTBENCH_REQUIREMENT_OP_KIND_SKIP_IF:
      return IREE_SV("skip_if");
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_testbench_requirement_skip_code_name(
    loom_testbench_requirement_skip_code_t code) {
  switch (code) {
    case LOOM_TESTBENCH_REQUIREMENT_SKIP_CODE_NONE:
      return IREE_SV("none");
    case LOOM_TESTBENCH_REQUIREMENT_SKIP_CODE_PROVIDER_NOT_REGISTERED:
      return IREE_SV("provider_not_registered");
    case LOOM_TESTBENCH_REQUIREMENT_SKIP_CODE_PROVIDER_UNAVAILABLE:
      return IREE_SV("provider_unavailable");
    case LOOM_TESTBENCH_REQUIREMENT_SKIP_CODE_REQUIREMENT_NOT_SATISFIED:
      return IREE_SV("requirement_not_satisfied");
    case LOOM_TESTBENCH_REQUIREMENT_SKIP_CODE_SKIP_PREDICATE_MATCHED:
      return IREE_SV("skip_predicate_matched");
    default:
      return IREE_SV("unknown");
  }
}

void loom_testbench_requirement_provider_registry_initialize(
    const loom_testbench_requirement_provider_t* providers,
    iree_host_size_t provider_count,
    loom_testbench_requirement_provider_registry_t* out_registry) {
  IREE_ASSERT(provider_count == 0 || providers);
  IREE_ASSERT_ARGUMENT(out_registry);
  *out_registry = (loom_testbench_requirement_provider_registry_t){
      .providers = provider_count != 0 ? providers : NULL,
      .provider_count = provider_count,
  };
}

const loom_named_attr_t* loom_testbench_requirement_find_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name) {
  IREE_ASSERT_ARGUMENT(module);
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* attr = &attrs.entries[i];
    if (iree_string_view_equal(
            loom_testbench_requirement_module_string(module, attr->name_id),
            name)) {
      return attr;
    }
  }
  return NULL;
}

iree_status_t loom_testbench_requirement_read_string_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name, iree_string_view_t* out_value) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(out_value);
  const loom_named_attr_t* attr =
      loom_testbench_requirement_find_attr(module, attrs, name);
  if (!attr) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "requirement attr '%.*s' is required",
                            (int)name.size, name.data);
  }
  if (attr->value.kind != LOOM_ATTR_STRING) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "requirement attr '%.*s' must be string",
                            (int)name.size, name.data);
  }
  *out_value = loom_testbench_requirement_module_string(
      module, loom_attr_as_string_id(attr->value));
  return iree_ok_status();
}

iree_status_t loom_testbench_requirement_read_optional_i64_attr(
    const loom_module_t* module, loom_named_attr_slice_t attrs,
    iree_string_view_t name, bool* out_present, int64_t* out_value) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(out_present);
  IREE_ASSERT_ARGUMENT(out_value);
  *out_present = false;
  *out_value = 0;
  const loom_named_attr_t* attr =
      loom_testbench_requirement_find_attr(module, attrs, name);
  if (!attr) {
    return iree_ok_status();
  }
  if (attr->value.kind != LOOM_ATTR_I64) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "requirement attr '%.*s' must be i64",
                            (int)name.size, name.data);
  }
  *out_present = true;
  *out_value = loom_attr_as_i64(attr->value);
  return iree_ok_status();
}

static const loom_testbench_requirement_provider_t*
loom_testbench_requirement_find_provider(
    const loom_testbench_requirement_provider_registry_t* registry,
    iree_string_view_t name) {
  for (iree_host_size_t i = 0; i < registry->provider_count; ++i) {
    const loom_testbench_requirement_provider_t* provider =
        &registry->providers[i];
    if (iree_string_view_equal(provider->name, name)) {
      return provider;
    }
  }
  return NULL;
}

static iree_status_t loom_testbench_requirement_evaluate_provider(
    const loom_module_t* module,
    const loom_testbench_requirement_provider_registry_t* registry,
    iree_string_view_t provider_name, loom_named_attr_slice_t attrs,
    loom_testbench_requirement_provider_result_t* out_provider_result) {
  const loom_testbench_requirement_provider_t* provider =
      loom_testbench_requirement_find_provider(registry, provider_name);
  if (!provider || provider->query == NULL) {
    *out_provider_result = (loom_testbench_requirement_provider_result_t){
        .state = LOOM_TESTBENCH_REQUIREMENT_PROVIDER_STATE_NOT_REGISTERED,
    };
    return iree_ok_status();
  }
  return provider->query(provider->user_data, module, attrs,
                         out_provider_result);
}

static iree_string_view_t loom_testbench_requirement_display_message(
    const loom_module_t* module, const loom_op_t* op,
    iree_string_view_t provider_display_message) {
  if (loom_check_skip_if_isa(op)) {
    loom_attribute_t reason_attr =
        loom_op_attrs(op)[loom_check_skip_if_reason_ATTR_INDEX];
    if (reason_attr.kind == LOOM_ATTR_STRING) {
      return loom_testbench_requirement_module_string(
          module, loom_attr_as_string_id(reason_attr));
    }
  }
  return provider_display_message;
}

static loom_testbench_requirement_skip_code_t
loom_testbench_requirement_skip_code(
    loom_testbench_requirement_op_kind_t op_kind,
    const loom_testbench_requirement_provider_result_t* provider_result) {
  if (provider_result->state ==
      LOOM_TESTBENCH_REQUIREMENT_PROVIDER_STATE_NOT_REGISTERED) {
    return LOOM_TESTBENCH_REQUIREMENT_SKIP_CODE_PROVIDER_NOT_REGISTERED;
  }
  if (provider_result->state ==
      LOOM_TESTBENCH_REQUIREMENT_PROVIDER_STATE_UNAVAILABLE) {
    return LOOM_TESTBENCH_REQUIREMENT_SKIP_CODE_PROVIDER_UNAVAILABLE;
  }
  if (op_kind == LOOM_TESTBENCH_REQUIREMENT_OP_KIND_REQUIRES) {
    return LOOM_TESTBENCH_REQUIREMENT_SKIP_CODE_REQUIREMENT_NOT_SATISFIED;
  }
  return LOOM_TESTBENCH_REQUIREMENT_SKIP_CODE_SKIP_PREDICATE_MATCHED;
}

static iree_status_t loom_testbench_requirement_evaluate_op(
    const loom_module_t* module,
    const loom_testbench_requirement_provider_registry_t* registry,
    const loom_op_t* op, loom_testbench_requirement_result_t* result) {
  iree_string_view_t provider_name = iree_string_view_empty();
  loom_named_attr_slice_t attrs = loom_named_attr_slice_empty();
  loom_testbench_requirement_op_kind_t op_kind =
      LOOM_TESTBENCH_REQUIREMENT_OP_KIND_NONE;
  if (loom_check_requires_isa(op)) {
    op_kind = LOOM_TESTBENCH_REQUIREMENT_OP_KIND_REQUIRES;
    provider_name = loom_testbench_requirement_module_string(
        module, loom_check_requires_provider(op));
    attrs = loom_check_requires_attrs(op);
  } else if (loom_check_skip_if_isa(op)) {
    op_kind = LOOM_TESTBENCH_REQUIREMENT_OP_KIND_SKIP_IF;
    provider_name = loom_testbench_requirement_module_string(
        module, loom_check_skip_if_provider(op));
    attrs = loom_check_skip_if_attrs(op);
  } else {
    return iree_ok_status();
  }

  loom_testbench_requirement_provider_result_t provider_result = {0};
  IREE_RETURN_IF_ERROR(loom_testbench_requirement_evaluate_provider(
      module, registry, provider_name, attrs, &provider_result));
  if (provider_result.state ==
      LOOM_TESTBENCH_REQUIREMENT_PROVIDER_STATE_UNSPECIFIED) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "requirement provider '%.*s' returned no predicate state",
        (int)provider_name.size, provider_name.data);
  }

  const bool provider_available =
      provider_result.state !=
          LOOM_TESTBENCH_REQUIREMENT_PROVIDER_STATE_UNAVAILABLE &&
      provider_result.state !=
          LOOM_TESTBENCH_REQUIREMENT_PROVIDER_STATE_NOT_REGISTERED;
  const bool predicate_satisfied =
      provider_result.state ==
      LOOM_TESTBENCH_REQUIREMENT_PROVIDER_STATE_SATISFIED;
  const bool predicate_skip =
      op_kind == LOOM_TESTBENCH_REQUIREMENT_OP_KIND_REQUIRES
          ? !predicate_satisfied
          : predicate_satisfied;
  const bool should_skip = !provider_available || predicate_skip;
  if (should_skip) {
    *result = (loom_testbench_requirement_result_t){
        .skipped = true,
        .op = op,
        .op_kind = op_kind,
        .code = loom_testbench_requirement_skip_code(op_kind, &provider_result),
        .provider = provider_name,
        .provider_code = provider_result.provider_code,
        .display_message = loom_testbench_requirement_display_message(
            module, op, provider_result.display_message),
    };
  }
  return iree_ok_status();
}

iree_status_t loom_testbench_evaluate_case_requirements(
    const loom_module_t* module, const loom_testbench_case_plan_t* case_plan,
    const loom_testbench_requirement_provider_registry_t* registry,
    loom_testbench_requirement_result_t* out_result) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(case_plan);
  IREE_ASSERT_ARGUMENT(registry);
  IREE_ASSERT_ARGUMENT(out_result);
  memset(out_result, 0, sizeof(*out_result));

  loom_region_t* body = loom_check_case_body(case_plan->op);
  loom_block_t* block = NULL;
  loom_region_for_each_block(body, block) {
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      IREE_RETURN_IF_ERROR(loom_testbench_requirement_evaluate_op(
          module, registry, op, out_result));
      if (out_result->skipped) {
        return iree_ok_status();
      }
    }
  }
  return iree_ok_status();
}
