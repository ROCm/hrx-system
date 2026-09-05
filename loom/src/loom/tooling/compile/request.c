// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/compile/request.h"

#include "loom/ops/op_defs.h"
#include "loom/target/entry_selection.h"
#include "loom/target/projection.h"
#include "loom/target/selection.h"

iree_string_view_t loom_compile_product_name(loom_compile_product_t product) {
  switch (product) {
    case LOOM_COMPILE_PRODUCT_INVALID:
      break;
    case LOOM_COMPILE_PRODUCT_KERNEL:
      return IREE_SV("kernel");
    case LOOM_COMPILE_PRODUCT_COMMAND:
      return IREE_SV("command");
    case LOOM_COMPILE_PRODUCT_MODULE:
      return IREE_SV("module");
  }
  return IREE_SV("unknown");
}

static iree_status_t loom_compile_request_lookup_root(
    const loom_module_t* module, iree_string_view_t root_name,
    const loom_symbol_t** out_symbol) {
  *out_symbol = NULL;
  root_name = loom_target_entry_normalize_symbol_name(root_name);
  if (iree_string_view_is_empty(root_name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "root symbol name must not be empty");
  }
  const loom_string_id_t name_id = loom_module_lookup_string(module, root_name);
  const loom_symbol_id_t symbol_id =
      name_id != LOOM_STRING_ID_INVALID
          ? loom_module_find_symbol(module, name_id)
          : LOOM_SYMBOL_ID_INVALID;
  if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "root symbol '@%.*s' was not found",
                            (int)root_name.size, root_name.data);
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_id];
  if (symbol->defining_op == NULL) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "root symbol '@%.*s' has no materialized definition or declaration",
        (int)root_name.size, root_name.data);
  }
  *out_symbol = symbol;
  return iree_ok_status();
}

static iree_status_t loom_compile_request_classify_symbol(
    const loom_module_t* module, const loom_symbol_t* symbol,
    loom_compile_product_t* out_product) {
  if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_COMMAND_PROGRAM)) {
    *out_product = LOOM_COMPILE_PRODUCT_COMMAND;
    return iree_ok_status();
  }
  if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_KERNEL)) {
    *out_product = LOOM_COMPILE_PRODUCT_KERNEL;
    return iree_ok_status();
  }
  if (loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE)) {
    *out_product = LOOM_COMPILE_PRODUCT_MODULE;
    return iree_ok_status();
  }
  const iree_string_view_t symbol_name =
      module->strings.entries[symbol->name_id];
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "root symbol '@%.*s' does not define a compilable product",
      (int)symbol_name.size, symbol_name.data);
}

static iree_status_t loom_compile_request_kernel_target_fact_type(
    const loom_module_t* module, const loom_symbol_t* symbol,
    const loom_target_fact_type_t** out_fact_type) {
  *out_fact_type = NULL;
  const loom_func_like_t function =
      loom_func_like_const_cast(module, symbol->defining_op);
  if (!loom_func_like_isa(function)) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "kernel root does not implement FuncLike");
  }
  const loom_symbol_ref_t target_ref = loom_func_like_target(function);
  if (!loom_symbol_ref_is_valid(target_ref)) {
    return iree_ok_status();
  }
  if (target_ref.module_id != 0 ||
      target_ref.symbol_id >= module->symbols.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel root has an invalid target reference");
  }
  const loom_symbol_t* target_symbol =
      &module->symbols.entries[target_ref.symbol_id];
  const loom_target_like_t target =
      loom_target_like_cast(module, target_symbol->defining_op);
  const loom_target_like_descriptor_t* descriptor =
      loom_target_like_descriptor(target);
  if (descriptor == NULL || descriptor->fact_type == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel root target is not a target definition");
  }
  *out_fact_type = descriptor->fact_type;
  return iree_ok_status();
}

typedef struct loom_compile_root_summary_t {
  // Product shared by every selected root.
  loom_compile_product_t product;
  // Number of selected product roots.
  iree_host_size_t root_count;
  // Common authored kernel target type.
  const loom_target_fact_type_t* target_fact_type;
  // Number of selected kernel roots without an authored target.
  iree_host_size_t untargeted_kernel_count;
} loom_compile_root_summary_t;

static iree_status_t loom_compile_request_merge_kernel_target(
    const loom_module_t* module, const loom_symbol_t* symbol,
    loom_compile_root_summary_t* summary) {
  const loom_target_fact_type_t* fact_type = NULL;
  IREE_RETURN_IF_ERROR(
      loom_compile_request_kernel_target_fact_type(module, symbol, &fact_type));
  if (fact_type == NULL) {
    ++summary->untargeted_kernel_count;
    return iree_ok_status();
  }
  if (summary->target_fact_type != NULL &&
      summary->target_fact_type != fact_type) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "selected kernel roots use multiple target families ('%.*s' and "
        "'%.*s')",
        (int)summary->target_fact_type->name.size,
        summary->target_fact_type->name.data, (int)fact_type->name.size,
        fact_type->name.data);
  }
  summary->target_fact_type = fact_type;
  return iree_ok_status();
}

static iree_status_t loom_compile_request_merge_root(
    const loom_module_t* module, const loom_symbol_t* symbol,
    loom_compile_root_summary_t* summary) {
  loom_compile_product_t product = LOOM_COMPILE_PRODUCT_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_compile_request_classify_symbol(module, symbol, &product));
  if (summary->root_count != 0 && summary->product != product) {
    const iree_string_view_t symbol_name =
        module->strings.entries[symbol->name_id];
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "selected roots mix '%.*s' and '%.*s' products at '@%.*s'",
        (int)loom_compile_product_name(summary->product).size,
        loom_compile_product_name(summary->product).data,
        (int)loom_compile_product_name(product).size,
        loom_compile_product_name(product).data, (int)symbol_name.size,
        symbol_name.data);
  }
  summary->product = product;
  ++summary->root_count;
  if (product == LOOM_COMPILE_PRODUCT_KERNEL) {
    IREE_RETURN_IF_ERROR(
        loom_compile_request_merge_kernel_target(module, symbol, summary));
  }
  return iree_ok_status();
}

static bool loom_compile_request_is_concrete_public_command(
    const loom_symbol_t* symbol) {
  return symbol->defining_op != NULL &&
         loom_symbol_implements(symbol,
                                LOOM_SYMBOL_INTERFACE_COMMAND_PROGRAM) &&
         !loom_symbol_definition_is_declaration(symbol->definition) &&
         iree_any_bit_set(symbol->flags,
                          LOOM_SYMBOL_FLAG_PUBLIC | LOOM_SYMBOL_FLAG_RETAIN);
}

static bool loom_compile_request_is_concrete_kernel_entry(
    const loom_symbol_t* symbol) {
  return symbol->defining_op != NULL &&
         loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_KERNEL_ENTRY) &&
         !loom_symbol_definition_is_declaration(symbol->definition);
}

static void loom_compile_request_collect_implicit_commands(
    const loom_module_t* module, loom_compile_root_summary_t* summary) {
  summary->product = LOOM_COMPILE_PRODUCT_COMMAND;
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_symbol_t* symbol = &module->symbols.entries[i];
    if (loom_compile_request_is_concrete_public_command(symbol)) {
      ++summary->root_count;
    }
  }
}

static iree_status_t loom_compile_request_collect_implicit_kernels(
    const loom_module_t* module, loom_compile_root_summary_t* summary) {
  summary->product = LOOM_COMPILE_PRODUCT_KERNEL;
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_symbol_t* symbol = &module->symbols.entries[i];
    if (!loom_compile_request_is_concrete_kernel_entry(symbol)) {
      continue;
    }
    ++summary->root_count;
    IREE_RETURN_IF_ERROR(
        loom_compile_request_merge_kernel_target(module, symbol, summary));
  }
  return iree_ok_status();
}

static iree_status_t loom_compile_request_validate_product(
    loom_compile_product_t inferred_product,
    loom_compile_product_t explicit_product) {
  if (explicit_product == LOOM_COMPILE_PRODUCT_INVALID ||
      explicit_product == inferred_product) {
    return iree_ok_status();
  }
  const iree_string_view_t inferred_name =
      loom_compile_product_name(inferred_product);
  const iree_string_view_t explicit_name =
      loom_compile_product_name(explicit_product);
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "selected roots infer product '%.*s'; --product='%.*s' cannot "
      "reinterpret them",
      (int)inferred_name.size, inferred_name.data, (int)explicit_name.size,
      explicit_name.data);
}

static iree_status_t loom_compile_request_select_roots(
    const loom_module_t* module, iree_string_view_list_t roots,
    loom_compile_product_t explicit_product,
    loom_compile_root_summary_t* out_summary) {
  *out_summary = (loom_compile_root_summary_t){0};
  if (roots.count != 0 && roots.values == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "root count is nonzero but roots are NULL");
  }
  for (iree_host_size_t i = 0; i < roots.count; ++i) {
    const loom_symbol_t* symbol = NULL;
    IREE_RETURN_IF_ERROR(
        loom_compile_request_lookup_root(module, roots.values[i], &symbol));
    IREE_RETURN_IF_ERROR(
        loom_compile_request_merge_root(module, symbol, out_summary));
  }
  if (out_summary->root_count != 0) {
    return loom_compile_request_validate_product(out_summary->product,
                                                 explicit_product);
  }

  if (explicit_product == LOOM_COMPILE_PRODUCT_COMMAND) {
    loom_compile_request_collect_implicit_commands(module, out_summary);
    if (out_summary->root_count == 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "product 'command' requires at least one public or retained command "
          "root");
    }
    return iree_ok_status();
  }
  if (explicit_product == LOOM_COMPILE_PRODUCT_KERNEL) {
    IREE_RETURN_IF_ERROR(
        loom_compile_request_collect_implicit_kernels(module, out_summary));
    if (out_summary->root_count == 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "product 'kernel' requires at least one kernel entry root");
    }
    return iree_ok_status();
  }
  if (explicit_product == LOOM_COMPILE_PRODUCT_MODULE) {
    out_summary->product = LOOM_COMPILE_PRODUCT_MODULE;
    return iree_ok_status();
  }

  loom_compile_request_collect_implicit_commands(module, out_summary);
  if (out_summary->root_count != 0) {
    return iree_ok_status();
  }

  *out_summary = (loom_compile_root_summary_t){0};
  IREE_RETURN_IF_ERROR(
      loom_compile_request_collect_implicit_kernels(module, out_summary));
  if (out_summary->root_count != 0) {
    return iree_ok_status();
  }

  out_summary->product = LOOM_COMPILE_PRODUCT_MODULE;
  return iree_ok_status();
}

static iree_status_t loom_compile_request_select_explicit_target(
    iree_string_view_t target_value,
    const loom_target_environment_t* target_environment,
    loom_artifact_target_t* out_target) {
  *out_target = (loom_artifact_target_t){0};
  target_value = iree_string_view_trim(target_value);
  if (iree_string_view_is_empty(target_value)) {
    return iree_ok_status();
  }
  loom_target_specification_t specification = {0};
  IREE_RETURN_IF_ERROR(
      loom_target_specification_parse(target_value, &specification));
  const loom_target_profile_t* profile = NULL;
  IREE_RETURN_IF_ERROR(loom_target_environment_select_profile(
      target_environment, &specification, &profile));
  *out_target = (loom_artifact_target_t){
      .target_profile = profile,
      .target_key = specification.selector,
  };
  return iree_ok_status();
}

static iree_status_t loom_compile_request_parse_product(
    iree_string_view_t value, loom_compile_product_t* out_product) {
  *out_product = LOOM_COMPILE_PRODUCT_INVALID;
  value = iree_string_view_trim(value);
  if (iree_string_view_is_empty(value)) {
    return iree_ok_status();
  }
  for (loom_compile_product_t product = LOOM_COMPILE_PRODUCT_KERNEL;
       product <= LOOM_COMPILE_PRODUCT_MODULE; ++product) {
    if (iree_string_view_equal(value, loom_compile_product_name(product))) {
      *out_product = product;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown --product='%.*s'; expected 'kernel', "
                          "'command', or 'module'",
                          (int)value.size, value.data);
}

static iree_status_t loom_compile_request_validate_artifact_provider(
    const loom_artifact_provider_t* provider,
    const loom_target_fact_type_t* target_fact_type) {
  if (provider->target_profile_type == NULL ||
      provider->target_profile_type->fact_type == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "artifact provider '%.*s' has no target profile type",
        (int)provider->name.size, provider->name.data);
  }
  if (provider->public_artifact_format.data == NULL ||
      iree_string_view_is_empty(provider->public_artifact_format)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "artifact provider '%.*s' has no public artifact format",
        (int)provider->name.size, provider->name.data);
  }
  if (target_fact_type != provider->target_profile_type->fact_type) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "format '%.*s' requires target family '%.*s'; selected roots use "
        "'%.*s'",
        (int)provider->public_artifact_format.size,
        provider->public_artifact_format.data,
        (int)provider->target_profile_type->name.size,
        provider->target_profile_type->name.data,
        (int)target_fact_type->name.size, target_fact_type->name.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_compile_request_select_named_format(
    loom_compile_product_t product, iree_string_view_t format,
    const loom_target_fact_type_t* target_fact_type,
    const loom_artifact_provider_registry_t* artifact_provider_registry,
    const loom_target_environment_t* target_environment,
    loom_compile_producer_t* out_producer) {
  *out_producer = (loom_compile_producer_t){0};
  iree_host_size_t match_count = 0;
  if (iree_string_view_equal(format, IREE_SV("loom-command"))) {
    out_producer->kind = LOOM_COMPILE_PRODUCER_COMMAND;
    ++match_count;
  }
  for (iree_host_size_t i = 0; i < artifact_provider_registry->provider_count;
       ++i) {
    const loom_artifact_provider_t* provider =
        artifact_provider_registry->providers[i];
    if (provider != NULL &&
        iree_string_view_equal(provider->public_artifact_format, format)) {
      out_producer->kind = LOOM_COMPILE_PRODUCER_ARTIFACT;
      out_producer->value.artifact_provider = provider;
      ++match_count;
    }
  }
  const loom_target_emitter_list_t emitters =
      loom_target_environment_emitter_list(target_environment);
  for (iree_host_size_t i = 0; i < emitters.count; ++i) {
    const loom_target_emitter_t* emitter = emitters.values[i];
    if (emitter != NULL &&
        iree_string_view_equal(emitter->public_artifact_format, format)) {
      out_producer->kind = LOOM_COMPILE_PRODUCER_TARGET_EMITTER;
      out_producer->value.target_emitter = emitter;
      ++match_count;
    }
  }
  if (match_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "format '%.*s' is not available in this binary",
                            (int)format.size, format.data);
  }
  if (match_count != 1) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "format '%.*s' has multiple configured producers",
                            (int)format.size, format.data);
  }

  const iree_string_view_t product_name = loom_compile_product_name(product);
  switch (out_producer->kind) {
    case LOOM_COMPILE_PRODUCER_COMMAND:
      if (product != LOOM_COMPILE_PRODUCT_COMMAND) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "format 'loom-command' cannot emit product '%.*s'",
            (int)product_name.size, product_name.data);
      }
      return iree_ok_status();
    case LOOM_COMPILE_PRODUCER_ARTIFACT:
      if (product != LOOM_COMPILE_PRODUCT_KERNEL) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "format '%.*s' cannot emit product '%.*s'",
                                (int)format.size, format.data,
                                (int)product_name.size, product_name.data);
      }
      return loom_compile_request_validate_artifact_provider(
          out_producer->value.artifact_provider, target_fact_type);
    case LOOM_COMPILE_PRODUCER_TARGET_EMITTER:
      if (product == LOOM_COMPILE_PRODUCT_COMMAND) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "format '%.*s' cannot emit product 'command'",
                                (int)format.size, format.data);
      }
      return iree_ok_status();
    case LOOM_COMPILE_PRODUCER_INVALID:
      break;
  }
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "format producer selection is invalid");
}

static iree_status_t loom_compile_request_select_canonical_kernel_format(
    const loom_target_fact_type_t* target_fact_type,
    const loom_artifact_provider_registry_t* artifact_provider_registry,
    iree_string_view_t* out_format, loom_compile_producer_t* out_producer) {
  *out_format = iree_string_view_empty();
  *out_producer = (loom_compile_producer_t){0};
  iree_host_size_t match_count = 0;
  for (iree_host_size_t i = 0; i < artifact_provider_registry->provider_count;
       ++i) {
    const loom_artifact_provider_t* provider =
        artifact_provider_registry->providers[i];
    if (provider == NULL ||
        !iree_any_bit_set(provider->flags,
                          LOOM_ARTIFACT_PROVIDER_FLAG_CANONICAL) ||
        provider->target_profile_type == NULL ||
        provider->target_profile_type->fact_type != target_fact_type) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_compile_request_validate_artifact_provider(
        provider, target_fact_type));
    *out_format = provider->public_artifact_format;
    out_producer->kind = LOOM_COMPILE_PRODUCER_ARTIFACT;
    out_producer->value.artifact_provider = provider;
    ++match_count;
  }
  if (match_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "no canonical kernel format is configured for target family '%.*s'",
        (int)target_fact_type->name.size, target_fact_type->name.data);
  }
  if (match_count != 1) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "target family '%.*s' has multiple canonical kernel formats; pass "
        "--format to select one",
        (int)target_fact_type->name.size, target_fact_type->name.data);
  }
  return iree_ok_status();
}

static iree_status_t loom_compile_request_select_format(
    loom_compile_product_t product, iree_string_view_t explicit_format,
    const loom_target_fact_type_t* target_fact_type,
    const loom_artifact_provider_registry_t* artifact_provider_registry,
    const loom_target_environment_t* target_environment,
    iree_string_view_t* out_format, loom_compile_producer_t* out_producer) {
  explicit_format = iree_string_view_trim(explicit_format);
  if (!iree_string_view_is_empty(explicit_format)) {
    IREE_RETURN_IF_ERROR(loom_compile_request_select_named_format(
        product, explicit_format, target_fact_type, artifact_provider_registry,
        target_environment, out_producer));
    *out_format = explicit_format;
    return iree_ok_status();
  }
  switch (product) {
    case LOOM_COMPILE_PRODUCT_INVALID:
      break;
    case LOOM_COMPILE_PRODUCT_KERNEL:
      return loom_compile_request_select_canonical_kernel_format(
          target_fact_type, artifact_provider_registry, out_format,
          out_producer);
    case LOOM_COMPILE_PRODUCT_COMMAND:
      *out_format = IREE_SV("loom-command");
      out_producer->kind = LOOM_COMPILE_PRODUCER_COMMAND;
      return iree_ok_status();
    case LOOM_COMPILE_PRODUCT_MODULE:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "module product has no canonical format; pass --format");
  }
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "compile product selection is invalid");
}

iree_status_t loom_compile_request_resolve(
    const loom_module_t* module, const loom_compile_request_options_t* options,
    const loom_artifact_provider_registry_t* artifact_provider_registry,
    const loom_target_environment_t* target_environment,
    loom_compile_request_t* out_request) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(artifact_provider_registry);
  IREE_ASSERT_ARGUMENT(target_environment);
  IREE_ASSERT_ARGUMENT(out_request);
  *out_request = (loom_compile_request_t){0};

  loom_compile_product_t explicit_product = LOOM_COMPILE_PRODUCT_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_compile_request_parse_product(options->product, &explicit_product));
  loom_compile_root_summary_t root_summary = {0};
  IREE_RETURN_IF_ERROR(loom_compile_request_select_roots(
      module, options->roots, explicit_product, &root_summary));

  loom_artifact_target_t explicit_target = {0};
  IREE_RETURN_IF_ERROR(loom_compile_request_select_explicit_target(
      options->target, target_environment, &explicit_target));
  if (explicit_target.target_profile != NULL &&
      root_summary.product != LOOM_COMPILE_PRODUCT_KERNEL) {
    const iree_string_view_t product_name =
        loom_compile_product_name(root_summary.product);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "--target is not valid for product '%.*s'",
                            (int)product_name.size, product_name.data);
  }
  if (explicit_target.target_profile != NULL &&
      root_summary.target_fact_type != NULL &&
      explicit_target.target_profile->type->fact_type !=
          root_summary.target_fact_type) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "--target family '%.*s' cannot specialize roots authored for target "
        "family '%.*s'",
        (int)explicit_target.target_profile->type->name.size,
        explicit_target.target_profile->type->name.data,
        (int)root_summary.target_fact_type->name.size,
        root_summary.target_fact_type->name.data);
  }
  const loom_target_fact_type_t* target_fact_type =
      explicit_target.target_profile != NULL
          ? explicit_target.target_profile->type->fact_type
          : root_summary.target_fact_type;
  if (root_summary.product == LOOM_COMPILE_PRODUCT_KERNEL &&
      explicit_target.target_profile == NULL &&
      root_summary.untargeted_kernel_count != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "kernel product requires --target when %u selected kernel root%s "
        "omit target(...) attrs",
        (unsigned)root_summary.untargeted_kernel_count,
        root_summary.untargeted_kernel_count == 1 ? "" : "s");
  }
  if (root_summary.product == LOOM_COMPILE_PRODUCT_KERNEL &&
      target_fact_type == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "kernel product requires a target");
  }
  if (target_fact_type != NULL &&
      loom_target_environment_lookup_fact_provider(target_environment,
                                                   target_fact_type) == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target family '%.*s' is not available in this binary",
        (int)target_fact_type->name.size, target_fact_type->name.data);
  }

  loom_compile_request_t request = {
      .product = root_summary.product,
      .roots = options->roots,
      .explicit_target = explicit_target,
      .target_fact_type = target_fact_type,
  };
  IREE_RETURN_IF_ERROR(loom_compile_request_select_format(
      request.product, options->format, request.target_fact_type,
      artifact_provider_registry, target_environment, &request.format,
      &request.producer));
  *out_request = request;
  return iree_ok_status();
}
