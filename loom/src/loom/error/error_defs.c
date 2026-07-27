// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/error_defs.h"

#include "loom/error/error_catalog.h"
#include "loom/error/error_defs_tables.inl"

const char* loom_diagnostic_severity_name(loom_diagnostic_severity_t severity) {
  if (severity < IREE_ARRAYSIZE(loom_diagnostic_severity_names)) {
    const char* name = loom_diagnostic_severity_names[severity];
    if (name != NULL) {
      return name;
    }
  }
  return "unknown";
}

const char* loom_error_domain_name(loom_error_domain_t domain) {
  if (domain < IREE_ARRAYSIZE(loom_error_domain_names)) {
    const char* name = loom_error_domain_names[domain];
    if (name != NULL) {
      return name;
    }
  }
  return "UNKNOWN";
}

bool loom_error_domain_from_name(iree_string_view_t name,
                                 loom_error_domain_t* out_domain) {
  if (out_domain == NULL) {
    return false;
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(loom_error_domain_names);
       ++i) {
    const char* domain_name = loom_error_domain_names[i];
    if (domain_name != NULL &&
        iree_string_view_equal(name, iree_make_cstring_view(domain_name))) {
      *out_domain = (loom_error_domain_t)i;
      return true;
    }
  }
  return false;
}

const char* loom_emitter_name(loom_emitter_t emitter) {
  if (emitter < IREE_ARRAYSIZE(loom_emitter_names)) {
    const char* name = loom_emitter_names[emitter];
    if (name != NULL) {
      return name;
    }
  }
  return "unknown";
}

const loom_error_def_t* loom_error_catalog_lookup(
    const loom_error_catalog_t* catalog, loom_error_domain_t domain,
    uint16_t code) {
  if (catalog == NULL || domain >= LOOM_ERROR_DOMAIN_COUNT_) {
    return NULL;
  }
  const loom_error_domain_catalog_t* domain_catalog = catalog->domains[domain];
  if (domain_catalog == NULL || code >= domain_catalog->code_count) {
    return loom_error_catalog_lookup(catalog->fallback_catalog, domain, code);
  }
  const loom_error_def_t* error = domain_catalog->errors_by_code[code];
  if (error != NULL) {
    return error;
  }
  return loom_error_catalog_lookup(catalog->fallback_catalog, domain, code);
}

const loom_error_def_t* loom_error_catalog_lookup_ref(
    const loom_error_catalog_t* catalog, loom_error_ref_t ref) {
  if (!loom_error_ref_is_set(ref)) {
    return NULL;
  }
  return loom_error_catalog_lookup(catalog, loom_error_ref_domain(ref),
                                   loom_error_ref_code(ref));
}

const loom_error_def_t* loom_error_def_lookup(loom_error_domain_t domain,
                                              uint16_t code) {
  return loom_error_catalog_lookup(&loom_error_catalog_core, domain, code);
}

const loom_error_def_t* loom_error_def_lookup_ref(loom_error_ref_t ref) {
  return loom_error_catalog_lookup_ref(&loom_error_catalog_core, ref);
}
