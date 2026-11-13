// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/tool/amdf-info/selection.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

static bool amdf_info_parse_hex_word(const char* text, uint64_t* out_value) {
  uint64_t value = 0;
  for (uint32_t i = 0; i < 16; ++i) {
    const unsigned char c = (unsigned char)text[i];
    uint32_t digit;
    if (c >= '0' && c <= '9') {
      digit = c - '0';
    } else if (c >= 'a' && c <= 'f') {
      digit = c - 'a' + 10;
    } else if (c >= 'A' && c <= 'F') {
      digit = c - 'A' + 10;
    } else {
      return false;
    }
    value = (value << 4) | digit;
  }
  *out_value = value;
  return true;
}

const char* amdf_info_selector_parse(const char* text,
                                     amdf_info_selector_t* out_selector) {
  amdf_info_selector_t selector = {0};
  if (!text) {
    *out_selector = selector;
    return NULL;
  }
  const char* path = strchr(text, ':');
  if (!path && strchr(text, '/')) {
    selector.kind = AMDF_INFO_SELECTOR_NATIVE_PATH;
    selector.path = text;
    *out_selector = selector;
    return NULL;
  }
  const size_t family_length = path ? (size_t)(path - text) : strlen(text);
  if (family_length == 4 && strncmp(text, "rdna", 4) == 0) {
    selector.engine = AMDF_ENGINE_KIND_GPU;
    selector.architecture = AMDF_GPU_ARCHITECTURE_RDNA;
  } else if (family_length == 4 && strncmp(text, "cdna", 4) == 0) {
    selector.engine = AMDF_ENGINE_KIND_GPU;
    selector.architecture = AMDF_GPU_ARCHITECTURE_CDNA;
  } else if (family_length == 6 && strncmp(text, "amdgpu", 6) == 0) {
    selector.engine = AMDF_ENGINE_KIND_GPU;
  } else if (family_length == 4 && strncmp(text, "xdna", 4) == 0) {
    selector.engine = AMDF_ENGINE_KIND_XDNA;
  } else {
    return "device must name rdna, cdna, amdgpu, xdna, or a native device path";
  }
  selector.kind = AMDF_INFO_SELECTOR_ORDINAL;
  if (!path) {
    *out_selector = selector;
    return NULL;
  }
  ++path;
  if (strncmp(path, "//", 2) == 0) {
    path += 2;
  }
  if (!path[0]) {
    return "device URI requires an ordinal or identity path";
  }
  if (strchr(path, '?') || strchr(path, '#')) {
    return "device URI parameters are execution policy, not diagnostic options";
  }
  if (strncmp(path, "id/", 3) == 0) {
    selector.kind = AMDF_INFO_SELECTOR_ENDPOINT_ID;
    path += 3;
    if (strlen(path) != 32 ||
        !amdf_info_parse_hex_word(path, &selector.endpoint_id.words[0]) ||
        !amdf_info_parse_hex_word(path + 16, &selector.endpoint_id.words[1])) {
      return "endpoint identity requires exactly 32 hexadecimal digits";
    }
  } else if (strncmp(path, "GPU-", 4) == 0) {
    selector.kind = AMDF_INFO_SELECTOR_GPU_ID;
    path += 4;
    if (selector.engine != AMDF_ENGINE_KIND_GPU || strlen(path) != 16 ||
        !amdf_info_parse_hex_word(path, &selector.gpu_id) || !selector.gpu_id) {
      return "GPU identity requires a GPU family and a nonzero 16-digit "
             "hexadecimal value";
    }
  } else {
    for (const char* c = path; *c; ++c) {
      if (*c < '0' || *c > '9' ||
          selector.ordinal > (UINT32_MAX - (uint32_t)(*c - '0')) / 10) {
        return "device path must be an unsigned ordinal, GPU-<id>, or id/<id>";
      }
      selector.ordinal = selector.ordinal * 10 + (uint32_t)(*c - '0');
    }
  }
  *out_selector = selector;
  return NULL;
}

typedef enum amdf_info_match_e {
  AMDF_INFO_MATCH_NO,
  AMDF_INFO_MATCH_YES,
  AMDF_INFO_MATCH_UNKNOWN,
} amdf_info_match_t;

static amdf_info_match_t amdf_info_match_engine(amdf_engine_kind_t actual,
                                                amdf_engine_kind_t required) {
  if (required == AMDF_ENGINE_KIND_UNKNOWN) {
    return AMDF_INFO_MATCH_YES;
  }
  if (actual == AMDF_ENGINE_KIND_UNKNOWN) {
    return AMDF_INFO_MATCH_UNKNOWN;
  }
  return actual == required ? AMDF_INFO_MATCH_YES : AMDF_INFO_MATCH_NO;
}

static amdf_info_match_t amdf_info_match_family(
    const amdf_info_selector_t* selector,
    const amdf_info_endpoint_t* endpoint) {
  const amdf_engine_kind_t engine = endpoint->stage >= AMDF_INFO_ENDPOINT_INFO
                                        ? endpoint->info.engine_kind
                                        : endpoint->summary.engine_kind;
  const amdf_info_match_t match =
      amdf_info_match_engine(engine, selector->engine);
  if (match != AMDF_INFO_MATCH_YES ||
      selector->architecture == AMDF_GPU_ARCHITECTURE_UNKNOWN) {
    return match;
  }
  if (endpoint->stage != AMDF_INFO_ENDPOINT_TARGET) {
    return AMDF_INFO_MATCH_UNKNOWN;
  }
  return amdf_gpu_endpoint_architecture(&endpoint->target.gpu) ==
                 selector->architecture
             ? AMDF_INFO_MATCH_YES
             : AMDF_INFO_MATCH_NO;
}

static amdf_info_match_t amdf_info_match_identity(
    const amdf_info_selector_t* selector,
    const amdf_info_endpoint_t* endpoint) {
  if (selector->kind == AMDF_INFO_SELECTOR_ENDPOINT_ID) {
    return amdf_endpoint_id_is_equal(&selector->endpoint_id,
                                     &endpoint->summary.id)
               ? AMDF_INFO_MATCH_YES
               : AMDF_INFO_MATCH_NO;
  }
  if (selector->kind == AMDF_INFO_SELECTOR_GPU_ID) {
    const amdf_engine_kind_t engine = endpoint->summary.engine_kind;
    if (engine != AMDF_ENGINE_KIND_UNKNOWN && engine != AMDF_ENGINE_KIND_GPU) {
      return AMDF_INFO_MATCH_NO;
    }
    if (endpoint->stage != AMDF_INFO_ENDPOINT_TARGET) {
      return AMDF_INFO_MATCH_UNKNOWN;
    }
    return endpoint->info.engine_kind == AMDF_ENGINE_KIND_GPU &&
                   endpoint->target.gpu.unique_id == selector->gpu_id
               ? AMDF_INFO_MATCH_YES
               : AMDF_INFO_MATCH_NO;
  }
  if (selector->kind == AMDF_INFO_SELECTOR_NATIVE_PATH) {
    if (endpoint->stage < AMDF_INFO_ENDPOINT_INFO) {
      return AMDF_INFO_MATCH_UNKNOWN;
    }
    const amdf_endpoint_native_identity_t* a = &selector->native_identity;
    const amdf_endpoint_native_identity_t* b = &endpoint->info.native_identity;
    return a->type == AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_LINUX_DEVICE &&
                   b->type == a->type &&
                   a->value.linux_device.major == b->value.linux_device.major &&
                   a->value.linux_device.minor == b->value.linux_device.minor
               ? AMDF_INFO_MATCH_YES
               : AMDF_INFO_MATCH_NO;
  }
  return AMDF_INFO_MATCH_YES;
}

amdf_info_exit_code_t amdf_info_selection_apply(
    const amdf_info_options_t* options, const amdf_info_selector_t* selector,
    uint32_t count, amdf_info_endpoint_t* endpoints) {
  uint32_t family_ordinal = 0;
  uint32_t selected_count = 0;
  bool uncertain = false;
  bool ordinal_uncertain = false;
  for (uint32_t i = 0; i < count; ++i) {
    amdf_info_endpoint_t* endpoint = &endpoints[i];
    endpoint->selected = false;
    const amdf_info_match_t family = amdf_info_match_family(selector, endpoint);
    const amdf_info_match_t identity =
        amdf_info_match_identity(selector, endpoint);
    bool ordinal_matches = true;
    if (selector->kind == AMDF_INFO_SELECTOR_ORDINAL) {
      if (family == AMDF_INFO_MATCH_UNKNOWN) {
        // A later unqualified endpoint cannot change an already resolved
        // ordinal. An earlier one can shift every following family member.
        if (family_ordinal > selector->ordinal) {
          continue;
        }
        ordinal_uncertain = true;
      }
      if (family == AMDF_INFO_MATCH_YES) {
        ordinal_matches = family_ordinal++ == selector->ordinal;
      }
    }
    const amdf_engine_kind_t engine = endpoint->stage >= AMDF_INFO_ENDPOINT_INFO
                                          ? endpoint->info.engine_kind
                                          : endpoint->summary.engine_kind;
    const amdf_info_match_t engine_match =
        amdf_info_match_engine(engine, options->engine);
    if (family == AMDF_INFO_MATCH_NO || identity == AMDF_INFO_MATCH_NO ||
        engine_match == AMDF_INFO_MATCH_NO || !ordinal_matches) {
      continue;
    }
    amdf_info_match_t target = AMDF_INFO_MATCH_YES;
    if (options->target) {
      char target_name[AMDF_XDNA_TARGET_ID_CAPACITY];
      amdf_info_endpoint_target(endpoint, target_name);
      target = endpoint->stage != AMDF_INFO_ENDPOINT_TARGET
                   ? AMDF_INFO_MATCH_UNKNOWN
               : strcmp(target_name, options->target) == 0 ? AMDF_INFO_MATCH_YES
                                                           : AMDF_INFO_MATCH_NO;
    }
    if (target == AMDF_INFO_MATCH_NO) {
      continue;
    }
    if (family == AMDF_INFO_MATCH_UNKNOWN ||
        identity == AMDF_INFO_MATCH_UNKNOWN ||
        engine_match == AMDF_INFO_MATCH_UNKNOWN ||
        target == AMDF_INFO_MATCH_UNKNOWN) {
      uncertain = true;
      // Retain the identity and its failed qualification in the diagnostic.
      endpoint->selected = true;
      continue;
    }
    endpoint->selected = true;
    ++selected_count;
  }
  if (uncertain || ordinal_uncertain) {
    return AMDF_INFO_EXIT_FAILURE;
  }
  const bool scalar = options->select || options->device;
  if (!selected_count && (scalar || options->activate)) {
    return AMDF_INFO_EXIT_NOT_FOUND;
  }
  if (scalar && selected_count > 1) {
    return AMDF_INFO_EXIT_AMBIGUOUS;
  }
  return AMDF_INFO_EXIT_SUCCESS;
}

void amdf_info_selection_uri(const amdf_info_endpoint_t* endpoint,
                             char out_uri[48]) {
  if (strcmp(amdf_info_endpoint_scheme(endpoint), "unknown") == 0) {
    out_uri[0] = 0;
    return;
  }
  snprintf(out_uri, 48, "%s://id/%016" PRIx64 "%016" PRIx64,
           amdf_info_endpoint_scheme(endpoint), endpoint->summary.id.words[0],
           endpoint->summary.id.words[1]);
}
