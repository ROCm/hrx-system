// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/product/registry.h"

#include <inttypes.h>

static bool loom_product_registry_contains_operation(
    const loom_product_registry_t* registry,
    const loom_product_operation_t* operation) {
  for (iree_host_size_t i = 0; i < registry->operations.count; ++i) {
    if (registry->operations.values[i] == operation) return true;
  }
  return false;
}

static bool loom_product_registry_contains_format(
    const loom_product_registry_t* registry,
    const loom_product_format_t* format) {
  for (iree_host_size_t i = 0; i < registry->formats.count; ++i) {
    if (registry->formats.values[i] == format) return true;
  }
  return false;
}

static iree_status_t loom_product_operation_validate(
    const loom_product_operation_t* operation) {
  if (operation == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "product operation table contains a NULL entry");
  }
  if (iree_string_view_is_empty(operation->name)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "product operation name must not be empty");
  }
  if (operation->product_descriptor == NULL ||
      operation->product_descriptor->destroy == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "product operation '%.*s' has no usable product descriptor",
        (int)operation->name.size, operation->name.data);
  }
  if (!iree_string_view_equal(operation->name,
                              operation->product_descriptor->name)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "product operation '%.*s' descriptor is named '%.*s'",
        (int)operation->name.size, operation->name.data,
        (int)operation->product_descriptor->name.size,
        operation->product_descriptor->name.data);
  }
  if (operation->root_operation_name_count == 0 ||
      operation->root_operation_names == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "product operation '%.*s' has no durable root operations",
        (int)operation->name.size, operation->name.data);
  }
  for (iree_host_size_t i = 0; i < operation->root_operation_name_count; ++i) {
    const iree_string_view_t root_name = operation->root_operation_names[i];
    if (iree_string_view_is_empty(root_name)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "product operation '%.*s' has an empty durable root operation",
          (int)operation->name.size, operation->name.data);
    }
    for (iree_host_size_t j = 0; j < i; ++j) {
      if (iree_string_view_equal(operation->root_operation_names[j],
                                 root_name)) {
        return iree_make_status(
            IREE_STATUS_ALREADY_EXISTS,
            "product operation '%.*s' owns durable root operation '%.*s' "
            "more than once",
            (int)operation->name.size, operation->name.data,
            (int)root_name.size, root_name.data);
      }
    }
  }
  return iree_ok_status();
}

static const loom_product_artifact_schema_t*
loom_product_format_lookup_artifact_schema(const loom_product_format_t* format,
                                           iree_string_view_t role) {
  for (iree_host_size_t i = 0; i < format->artifact_schema_count; ++i) {
    const loom_product_artifact_schema_t* schema = &format->artifact_schemas[i];
    if (iree_string_view_equal(schema->role, role)) return schema;
  }
  return NULL;
}

static iree_status_t loom_product_format_validate(
    const loom_product_format_t* format) {
  if (format == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "product format table contains a NULL entry");
  }
  if (format->operation == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "product format has no product operation");
  }
  if (iree_string_view_is_empty(format->name)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "product operation '%.*s' has a format with an empty name",
        (int)format->operation->name.size, format->operation->name.data);
  }
  if (format->artifact_schema_count == 0 || format->artifact_schemas == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "product format '%.*s/%.*s' has no artifact schema",
                            (int)format->operation->name.size,
                            format->operation->name.data,
                            (int)format->name.size, format->name.data);
  }
  for (iree_host_size_t i = 0; i < format->artifact_schema_count; ++i) {
    const loom_product_artifact_schema_t* schema = &format->artifact_schemas[i];
    if (iree_string_view_is_empty(schema->role) ||
        iree_string_view_is_empty(schema->format)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "product format '%.*s/%.*s' has an artifact schema with an empty "
          "role or format",
          (int)format->operation->name.size, format->operation->name.data,
          (int)format->name.size, format->name.data);
    }
    if (schema->maximum_count == 0 ||
        schema->minimum_count > schema->maximum_count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "product format '%.*s/%.*s' role '%.*s' has invalid cardinality "
          "%" PRIhsz "..%" PRIhsz,
          (int)format->operation->name.size, format->operation->name.data,
          (int)format->name.size, format->name.data, (int)schema->role.size,
          schema->role.data, schema->minimum_count, schema->maximum_count);
    }
    for (iree_host_size_t j = 0; j < i; ++j) {
      if (iree_string_view_equal(format->artifact_schemas[j].role,
                                 schema->role)) {
        return iree_make_status(
            IREE_STATUS_ALREADY_EXISTS,
            "product format '%.*s/%.*s' declares artifact role '%.*s' more "
            "than once",
            (int)format->operation->name.size, format->operation->name.data,
            (int)format->name.size, format->name.data, (int)schema->role.size,
            schema->role.data);
      }
    }
  }

  switch (format->persistence) {
    case LOOM_PRODUCT_PERSISTENCE_SINGLE_FILE: {
      if (iree_string_view_is_empty(format->single_file.role) ||
          iree_string_view_is_empty(format->single_file.extension) ||
          !iree_string_view_starts_with_char(format->single_file.extension,
                                             '.')) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "single-file product format '%.*s/%.*s' requires a payload role "
            "and a period-prefixed extension",
            (int)format->operation->name.size, format->operation->name.data,
            (int)format->name.size, format->name.data);
      }
      const loom_product_artifact_schema_t* payload_schema =
          loom_product_format_lookup_artifact_schema(format,
                                                     format->single_file.role);
      if (payload_schema == NULL || payload_schema->minimum_count != 1 ||
          payload_schema->maximum_count != 1 ||
          !iree_string_view_equal(payload_schema->format, format->name)) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "single-file product format '%.*s/%.*s' payload role '%.*s' "
            "must have cardinality 1..1 and use the public format name",
            (int)format->operation->name.size, format->operation->name.data,
            (int)format->name.size, format->name.data,
            (int)format->single_file.role.size, format->single_file.role.data);
      }
      break;
    }
    case LOOM_PRODUCT_PERSISTENCE_ARTIFACT_SET:
      if (!iree_string_view_is_empty(format->single_file.role) ||
          !iree_string_view_is_empty(format->single_file.extension)) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "artifact-set product format '%.*s/%.*s' contains a single-file "
            "contract",
            (int)format->operation->name.size, format->operation->name.data,
            (int)format->name.size, format->name.data);
      }
      break;
    default:
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "product format '%.*s/%.*s' has unknown persistence shape %d",
          (int)format->operation->name.size, format->operation->name.data,
          (int)format->name.size, format->name.data, (int)format->persistence);
  }
  return iree_ok_status();
}

static iree_status_t loom_product_format_provider_validate(
    const loom_product_registry_t* registry,
    const loom_product_format_provider_t* provider) {
  if (provider == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "product provider table contains a NULL entry");
  }
  if (iree_string_view_is_empty(provider->name)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "product provider name must not be empty");
  }
  if (!loom_product_registry_contains_operation(registry,
                                                provider->operation)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "product provider '%.*s' references an unregistered operation",
        (int)provider->name.size, provider->name.data);
  }
  if (!loom_product_registry_contains_format(registry, provider->format)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "product provider '%.*s' references an unregistered format",
        (int)provider->name.size, provider->name.data);
  }
  if (provider->format->operation != provider->operation) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "product provider '%.*s' operation does not own format '%.*s'",
        (int)provider->name.size, provider->name.data,
        (int)provider->format->name.size, provider->format->name.data);
  }
  if (provider->build == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "product provider '%.*s' has no build implementation",
        (int)provider->name.size, provider->name.data);
  }
  if (provider->target_profile_type == NULL &&
      provider->accepts_target != NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "target-neutral product provider '%.*s' has a target predicate",
        (int)provider->name.size, provider->name.data);
  }
  if (provider->target_profile_type != NULL) {
    if (iree_string_view_is_empty(provider->target_profile_type->name)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "product provider '%.*s' has an unnamed target profile type",
          (int)provider->name.size, provider->name.data);
    }
    IREE_RETURN_IF_ERROR(loom_target_product_contract_validate(
        provider->target_product_contract));
    if (!iree_string_view_equal(provider->target_product_contract->name,
                                provider->format->name)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "product provider '%.*s' target contract '%.*s' does not match "
          "format '%.*s'",
          (int)provider->name.size, provider->name.data,
          (int)provider->target_product_contract->name.size,
          provider->target_product_contract->name.data,
          (int)provider->format->name.size, provider->format->name.data);
    }
  }
  return iree_ok_status();
}

iree_status_t loom_product_registry_validate(
    const loom_product_registry_t* registry) {
  if (registry == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "product registry is NULL");
  }
  if ((registry->operations.count != 0 &&
       registry->operations.values == NULL) ||
      (registry->formats.count != 0 && registry->formats.values == NULL) ||
      (registry->providers.count != 0 && registry->providers.values == NULL)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "product registry contains a NULL table with a nonzero count");
  }

  for (iree_host_size_t i = 0; i < registry->operations.count; ++i) {
    const loom_product_operation_t* operation = registry->operations.values[i];
    IREE_RETURN_IF_ERROR(loom_product_operation_validate(operation));
    for (iree_host_size_t j = 0; j < i; ++j) {
      const loom_product_operation_t* prior = registry->operations.values[j];
      if (iree_string_view_equal(prior->name, operation->name)) {
        return iree_make_status(
            IREE_STATUS_ALREADY_EXISTS,
            "product operation '%.*s' is registered more than once",
            (int)operation->name.size, operation->name.data);
      }
      for (iree_host_size_t root_i = 0;
           root_i < operation->root_operation_name_count; ++root_i) {
        for (iree_host_size_t root_j = 0;
             root_j < prior->root_operation_name_count; ++root_j) {
          if (iree_string_view_equal(operation->root_operation_names[root_i],
                                     prior->root_operation_names[root_j])) {
            const iree_string_view_t root_name =
                operation->root_operation_names[root_i];
            return iree_make_status(
                IREE_STATUS_ALREADY_EXISTS,
                "durable root operation '%.*s' is owned by products '%.*s' "
                "and '%.*s'",
                (int)root_name.size, root_name.data, (int)prior->name.size,
                prior->name.data, (int)operation->name.size,
                operation->name.data);
          }
        }
      }
    }
  }

  for (iree_host_size_t i = 0; i < registry->formats.count; ++i) {
    const loom_product_format_t* format = registry->formats.values[i];
    IREE_RETURN_IF_ERROR(loom_product_format_validate(format));
    if (!loom_product_registry_contains_operation(registry,
                                                  format->operation)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "product format '%.*s' references an unregistered operation",
          (int)format->name.size, format->name.data);
    }
    for (iree_host_size_t j = 0; j < i; ++j) {
      const loom_product_format_t* prior = registry->formats.values[j];
      if (prior->operation == format->operation &&
          iree_string_view_equal(prior->name, format->name)) {
        return iree_make_status(
            IREE_STATUS_ALREADY_EXISTS,
            "product format '%.*s/%.*s' is registered more than once",
            (int)format->operation->name.size, format->operation->name.data,
            (int)format->name.size, format->name.data);
      }
    }
  }

  for (iree_host_size_t i = 0; i < registry->providers.count; ++i) {
    const loom_product_format_provider_t* provider =
        registry->providers.values[i];
    IREE_RETURN_IF_ERROR(
        loom_product_format_provider_validate(registry, provider));
    for (iree_host_size_t j = 0; j < i; ++j) {
      const loom_product_format_provider_t* prior =
          registry->providers.values[j];
      if (iree_string_view_equal(prior->name, provider->name)) {
        return iree_make_status(
            IREE_STATUS_ALREADY_EXISTS,
            "product provider '%.*s' is registered more than once",
            (int)provider->name.size, provider->name.data);
      }
      if (prior->operation == provider->operation &&
          prior->format == provider->format &&
          prior->target_profile_type == provider->target_profile_type &&
          prior->accepts_target == NULL && provider->accepts_target == NULL) {
        return iree_make_status(
            IREE_STATUS_ALREADY_EXISTS,
            "product providers '%.*s' and '%.*s' unconditionally overlap for "
            "'%.*s/%.*s'",
            (int)prior->name.size, prior->name.data, (int)provider->name.size,
            provider->name.data, (int)provider->operation->name.size,
            provider->operation->name.data, (int)provider->format->name.size,
            provider->format->name.data);
      }
      if (prior->operation == provider->operation &&
          prior->target_profile_type == provider->target_profile_type &&
          iree_all_bits_set(prior->flags,
                            LOOM_PRODUCT_FORMAT_PROVIDER_CANONICAL) &&
          iree_all_bits_set(provider->flags,
                            LOOM_PRODUCT_FORMAT_PROVIDER_CANONICAL) &&
          (prior->accepts_target == NULL || provider->accepts_target == NULL)) {
        return iree_make_status(
            IREE_STATUS_ALREADY_EXISTS,
            "canonical product providers '%.*s' and '%.*s' unconditionally "
            "overlap for product '%.*s'",
            (int)prior->name.size, prior->name.data, (int)provider->name.size,
            provider->name.data, (int)provider->operation->name.size,
            provider->operation->name.data);
      }
    }
  }

  for (iree_host_size_t i = 0; i < registry->providers.count; ++i) {
    const loom_product_format_provider_t* provider =
        registry->providers.values[i];
    bool has_canonical = false;
    for (iree_host_size_t j = 0; j < registry->providers.count; ++j) {
      const loom_product_format_provider_t* candidate =
          registry->providers.values[j];
      if (candidate->operation == provider->operation &&
          candidate->target_profile_type == provider->target_profile_type &&
          iree_all_bits_set(candidate->flags,
                            LOOM_PRODUCT_FORMAT_PROVIDER_CANONICAL)) {
        has_canonical = true;
        break;
      }
    }
    if (!has_canonical) {
      const iree_string_view_t target_family =
          provider->target_profile_type != NULL
              ? provider->target_profile_type->name
              : IREE_SV("<target-neutral>");
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "product '%.*s' has no canonical provider for target family '%.*s'",
          (int)provider->operation->name.size, provider->operation->name.data,
          (int)target_family.size, target_family.data);
    }
  }

  for (iree_host_size_t i = 0; i < registry->operations.count; ++i) {
    const loom_product_operation_t* operation = registry->operations.values[i];
    bool implemented = false;
    for (iree_host_size_t j = 0; j < registry->providers.count; ++j) {
      implemented |= registry->providers.values[j]->operation == operation;
    }
    if (!implemented) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "product operation '%.*s' has no registered provider",
          (int)operation->name.size, operation->name.data);
    }
  }
  for (iree_host_size_t i = 0; i < registry->formats.count; ++i) {
    const loom_product_format_t* format = registry->formats.values[i];
    bool implemented = false;
    for (iree_host_size_t j = 0; j < registry->providers.count; ++j) {
      implemented |= registry->providers.values[j]->format == format;
    }
    if (!implemented) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "product format '%.*s/%.*s' has no registered provider",
          (int)format->operation->name.size, format->operation->name.data,
          (int)format->name.size, format->name.data);
    }
  }
  return iree_ok_status();
}

const loom_product_operation_t* loom_product_registry_lookup_operation(
    const loom_product_registry_t* registry, iree_string_view_t name) {
  if (registry == NULL || iree_string_view_is_empty(name)) return NULL;
  for (iree_host_size_t i = 0; i < registry->operations.count; ++i) {
    const loom_product_operation_t* operation = registry->operations.values[i];
    if (iree_string_view_equal(operation->name, name)) return operation;
  }
  return NULL;
}

const loom_product_operation_t* loom_product_registry_lookup_root_operation(
    const loom_product_registry_t* registry,
    iree_string_view_t defining_op_name) {
  if (registry == NULL || iree_string_view_is_empty(defining_op_name)) {
    return NULL;
  }
  for (iree_host_size_t i = 0; i < registry->operations.count; ++i) {
    const loom_product_operation_t* operation = registry->operations.values[i];
    for (iree_host_size_t j = 0; j < operation->root_operation_name_count;
         ++j) {
      if (iree_string_view_equal(operation->root_operation_names[j],
                                 defining_op_name)) {
        return operation;
      }
    }
  }
  return NULL;
}

const loom_product_format_t* loom_product_registry_lookup_format(
    const loom_product_registry_t* registry,
    const loom_product_operation_t* operation, iree_string_view_t name) {
  if (registry == NULL || operation == NULL ||
      iree_string_view_is_empty(name)) {
    return NULL;
  }
  for (iree_host_size_t i = 0; i < registry->formats.count; ++i) {
    const loom_product_format_t* format = registry->formats.values[i];
    if (format->operation == operation &&
        iree_string_view_equal(format->name, name)) {
      return format;
    }
  }
  return NULL;
}

static bool loom_product_format_provider_accepts_target(
    const loom_product_format_provider_t* provider,
    const loom_target_profile_t* target_profile) {
  if (provider->target_profile_type == NULL) {
    return target_profile == NULL;
  }
  if (!loom_target_profile_has_type(target_profile,
                                    provider->target_profile_type)) {
    return false;
  }
  return provider->accepts_target == NULL ||
         provider->accepts_target(provider, target_profile);
}

iree_status_t loom_product_registry_select_provider(
    const loom_product_registry_t* registry,
    const loom_product_operation_t* operation, iree_string_view_t format_name,
    const loom_target_profile_t* target_profile,
    const loom_product_format_provider_t** out_provider) {
  IREE_ASSERT_ARGUMENT(out_provider);
  *out_provider = NULL;
  if (registry == NULL || operation == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "product provider selection requires a registry and operation");
  }
  if (!loom_product_registry_contains_operation(registry, operation)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "product operation is not registered");
  }

  format_name = iree_string_view_trim(format_name);
  const loom_product_format_t* explicit_format = NULL;
  if (!iree_string_view_is_empty(format_name)) {
    explicit_format =
        loom_product_registry_lookup_format(registry, operation, format_name);
    if (explicit_format == NULL) {
      return iree_make_status(IREE_STATUS_NOT_FOUND,
                              "product '%.*s' has no registered format '%.*s'",
                              (int)operation->name.size, operation->name.data,
                              (int)format_name.size, format_name.data);
    }
  }

  const loom_product_format_provider_t* selected_provider = NULL;
  for (iree_host_size_t i = 0; i < registry->providers.count; ++i) {
    const loom_product_format_provider_t* provider =
        registry->providers.values[i];
    if (provider->operation != operation ||
        (explicit_format != NULL && provider->format != explicit_format) ||
        (explicit_format == NULL &&
         !iree_all_bits_set(provider->flags,
                            LOOM_PRODUCT_FORMAT_PROVIDER_CANONICAL)) ||
        !loom_product_format_provider_accepts_target(provider,
                                                     target_profile)) {
      continue;
    }
    if (selected_provider != NULL) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "product providers '%.*s' and '%.*s' both match '%.*s%s%.*s'",
          (int)selected_provider->name.size, selected_provider->name.data,
          (int)provider->name.size, provider->name.data,
          (int)operation->name.size, operation->name.data,
          explicit_format != NULL ? "/" : "",
          explicit_format != NULL ? (int)explicit_format->name.size : 0,
          explicit_format != NULL ? explicit_format->name.data : "");
    }
    selected_provider = provider;
  }
  if (selected_provider == NULL) {
    const iree_string_view_t target_family =
        target_profile != NULL && target_profile->type != NULL
            ? target_profile->type->name
            : IREE_SV("<target-neutral>");
    if (explicit_format != NULL) {
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "no provider implements product/format '%.*s/%.*s' for target "
          "family '%.*s'",
          (int)operation->name.size, operation->name.data,
          (int)explicit_format->name.size, explicit_format->name.data,
          (int)target_family.size, target_family.data);
    }
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "product '%.*s' has no canonical format for target family '%.*s'",
        (int)operation->name.size, operation->name.data,
        (int)target_family.size, target_family.data);
  }

  *out_provider = selected_provider;
  return iree_ok_status();
}

iree_status_t loom_product_format_validate_product(
    const loom_product_format_t* format, const loom_product_t* product) {
  if (format == NULL || product == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "product format validation requires a format and product");
  }
  IREE_RETURN_IF_ERROR(loom_product_format_validate(format));
  if (!loom_product_isa(product, format->operation->product_descriptor)) {
    const loom_product_descriptor_t* actual_descriptor =
        loom_product_descriptor(product);
    const iree_string_view_t actual_name = actual_descriptor != NULL
                                               ? actual_descriptor->name
                                               : IREE_SV("<unknown>");
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "format '%.*s/%.*s' cannot persist product '%.*s'",
                            (int)format->operation->name.size,
                            format->operation->name.data,
                            (int)format->name.size, format->name.data,
                            (int)actual_name.size, actual_name.data);
  }

  const iree_host_size_t artifact_count = loom_product_artifact_count(product);
  for (iree_host_size_t i = 0; i < artifact_count; ++i) {
    const loom_product_artifact_t* artifact =
        loom_product_artifact_at(product, i);
    if (iree_string_view_is_empty(artifact->role) ||
        iree_string_view_is_empty(artifact->format) ||
        iree_string_view_is_empty(artifact->identifier) ||
        artifact->contents == NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "product artifact %" PRIhsz
                              " has an empty role, format, identifier, or "
                              "contents",
                              i);
    }
    const loom_product_artifact_schema_t* schema =
        loom_product_format_lookup_artifact_schema(format, artifact->role);
    if (schema == NULL) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "product format '%.*s/%.*s' does not declare artifact role '%.*s'",
          (int)format->operation->name.size, format->operation->name.data,
          (int)format->name.size, format->name.data, (int)artifact->role.size,
          artifact->role.data);
    }
    if (!iree_string_view_equal(schema->format, artifact->format)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "product format '%.*s/%.*s' role '%.*s' requires artifact format "
          "'%.*s', got '%.*s'",
          (int)format->operation->name.size, format->operation->name.data,
          (int)format->name.size, format->name.data, (int)artifact->role.size,
          artifact->role.data, (int)schema->format.size, schema->format.data,
          (int)artifact->format.size, artifact->format.data);
    }
  }

  for (iree_host_size_t schema_i = 0; schema_i < format->artifact_schema_count;
       ++schema_i) {
    const loom_product_artifact_schema_t* schema =
        &format->artifact_schemas[schema_i];
    iree_host_size_t matching_count = 0;
    for (iree_host_size_t artifact_i = 0; artifact_i < artifact_count;
         ++artifact_i) {
      const loom_product_artifact_t* artifact =
          loom_product_artifact_at(product, artifact_i);
      if (iree_string_view_equal(artifact->role, schema->role)) {
        ++matching_count;
      }
    }
    if (matching_count < schema->minimum_count ||
        matching_count > schema->maximum_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "product format '%.*s/%.*s' role '%.*s' requires cardinality "
          "%" PRIhsz "..%" PRIhsz ", got %" PRIhsz,
          (int)format->operation->name.size, format->operation->name.data,
          (int)format->name.size, format->name.data, (int)schema->role.size,
          schema->role.data, schema->minimum_count, schema->maximum_count,
          matching_count);
    }
  }
  return iree_ok_status();
}
