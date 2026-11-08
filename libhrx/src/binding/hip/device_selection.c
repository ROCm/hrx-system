// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/device_selection.h"

#include <stdlib.h>
#include <string.h>

iree_hip_device_selection_t iree_hip_device_selection_from_environment(void) {
  return (iree_hip_device_selection_t){
      .rocr_visible_devices = getenv("ROCR_VISIBLE_DEVICES"),
      .hip_visible_devices = getenv("HIP_VISIBLE_DEVICES"),
      .cuda_visible_devices = getenv("CUDA_VISIBLE_DEVICES"),
  };
}

static const char* iree_hip_device_selection_frontend_filter(
    const iree_hip_device_selection_t* selection) {
  return selection->hip_visible_devices && selection->hip_visible_devices[0]
             ? selection->hip_visible_devices
             : selection->cuda_visible_devices;
}

bool iree_hip_device_selection_selects_none(
    const iree_hip_device_selection_t* selection) {
  if (selection->rocr_visible_devices) {
    const iree_string_view_t value = iree_string_view_trim(
        iree_make_cstring_view(selection->rocr_visible_devices));
    if (iree_string_view_is_empty(value) ||
        iree_string_view_equal(value, IREE_SV("-1"))) {
      return true;
    }
  }
  const char* frontend = iree_hip_device_selection_frontend_filter(selection);
  return frontend && strcmp(frontend, "-1") == 0;
}

// ROCr accepts strtol-style base-0 ordinals. CLR accepts only the canonical
// nonnegative decimal spelling, with no whitespace, sign, or leading zeros.
static bool iree_hip_device_selection_parse_ordinal(
    iree_string_view_t token, bool rocr, iree_host_size_t* out_ordinal) {
  uint32_t base = 10;
  bool negative = false;
  if (rocr) {
    negative = iree_string_view_consume_prefix(&token, IREE_SV("-"));
    if (!negative) {
      iree_string_view_consume_prefix(&token, IREE_SV("+"));
    }
    if (iree_string_view_consume_prefix(&token, IREE_SV("0x")) ||
        iree_string_view_consume_prefix(&token, IREE_SV("0X"))) {
      base = 16;
    } else if (token.size > 1 && token.data[0] == '0') {
      base = 8;
    }
  } else if (token.size > 1 && token.data[0] == '0') {
    return false;
  }
  if (iree_string_view_is_empty(token)) {
    return false;
  }
  iree_host_size_t ordinal = 0;
  for (iree_host_size_t i = 0; i < token.size; ++i) {
    const char character = token.data[i];
    uint32_t digit = UINT32_MAX;
    if (character >= '0' && character <= '9') {
      digit = (uint32_t)(character - '0');
    } else if (character >= 'a' && character <= 'f') {
      digit = (uint32_t)(character - 'a') + 10;
    } else if (character >= 'A' && character <= 'F') {
      digit = (uint32_t)(character - 'A') + 10;
    }
    if (digit >= base || ordinal > (IREE_HOST_SIZE_MAX - digit) / base) {
      return false;
    }
    ordinal = ordinal * base + digit;
  }
  if (negative && ordinal != 0) {
    return false;
  }
  *out_ordinal = ordinal;
  return true;
}

static iree_host_size_t iree_hip_device_selection_resolve(
    iree_string_view_t token, bool rocr, iree_host_size_t device_count,
    const iree_hal_device_info_t* device_infos) {
  const bool by_uuid =
      token.size >= 4 &&
      (rocr ? iree_string_view_equal_case(iree_string_view_substr(token, 0, 4),
                                          IREE_SV("GPU-"))
            : iree_string_view_starts_with(token, IREE_SV("GPU-")));
  if (!by_uuid) {
    iree_host_size_t ordinal = 0;
    return iree_hip_device_selection_parse_ordinal(token, rocr, &ordinal) &&
                   ordinal < device_count
               ? ordinal
               : IREE_HOST_SIZE_MAX;
  }
  if (token.size > 20 || (rocr && token.size < 5)) {
    return IREE_HOST_SIZE_MAX;
  }
  iree_host_size_t match = IREE_HOST_SIZE_MAX;
  for (iree_host_size_t i = 0; i < device_count; ++i) {
    const iree_string_view_t path = device_infos[i].path;
    if (path.size != 20 ||
        !iree_string_view_starts_with(path, IREE_SV("GPU-")) ||
        iree_string_view_equal(path, IREE_SV("GPU-0000000000000000"))) {
      continue;
    }
    const iree_string_view_t prefix =
        iree_string_view_substr(path, 0, token.size);
    if (!(rocr ? iree_string_view_equal_case(prefix, token)
               : iree_string_view_equal(prefix, token))) {
      continue;
    }
    // CLR selects the first UUID prefix match. ROCr requires uniqueness.
    if (!rocr) {
      return i;
    }
    if (match != IREE_HOST_SIZE_MAX) {
      return IREE_HOST_SIZE_MAX;
    }
    match = i;
  }
  return match;
}

static iree_host_size_t iree_hip_device_selection_filter(
    const char* filter, bool rocr, iree_host_size_t device_count,
    const iree_hal_device_info_t* device_infos,
    iree_hal_device_info_t* selected_infos) {
  iree_host_size_t selected_count = 0;
  iree_host_size_t token_count = 0;
  iree_string_view_t remaining = iree_make_cstring_view(filter);
  while (!iree_string_view_is_empty(remaining) &&
         (!rocr || token_count < device_count)) {
    ++token_count;
    iree_string_view_t token;
    iree_string_view_split(remaining, ',', &token, &remaining);
    if (rocr) {
      token = iree_string_view_trim(token);
    }
    const iree_host_size_t ordinal = iree_hip_device_selection_resolve(
        token, rocr, device_count, device_infos);
    if (ordinal == IREE_HOST_SIZE_MAX) {
      break;
    }
    const iree_hal_device_info_t* info = &device_infos[ordinal];
    bool duplicate = false;
    for (iree_host_size_t i = 0; i < selected_count; ++i) {
      if (selected_infos[i].device_id == info->device_id) {
        duplicate = true;
        break;
      }
    }
    if (duplicate && rocr) {
      break;
    }
    if (!duplicate) {
      selected_infos[selected_count++] = *info;
    }
  }
  return selected_count;
}

iree_status_t iree_hip_device_selection_apply(
    void* self, iree_string_view_t driver_name, iree_allocator_t host_allocator,
    iree_host_size_t* device_count, iree_hal_device_info_t* device_infos) {
  const iree_hip_device_selection_t* selection = self;
  (void)driver_name;
  const char* rocr = selection->rocr_visible_devices;
  const char* frontend = iree_hip_device_selection_frontend_filter(selection);
  if (frontend && !frontend[0]) {
    frontend = NULL;
  }
  if ((!rocr && !frontend) || !*device_count) {
    return iree_ok_status();
  }
  if (rocr && !rocr[0]) {
    *device_count = 0;
    return iree_ok_status();
  }

  // Preserve the input order while each stage writes its compact output.
  // Strings remain in the original enumeration allocation owned by HRX.
  iree_hal_device_info_t* source_infos = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, *device_count * sizeof(*source_infos),
      (void**)&source_infos));
  if (rocr) {
    memcpy(source_infos, device_infos, *device_count * sizeof(*source_infos));
    *device_count = iree_hip_device_selection_filter(
        rocr, /*rocr=*/true, *device_count, source_infos, device_infos);
  }
  if (frontend && *device_count) {
    memcpy(source_infos, device_infos, *device_count * sizeof(*source_infos));
    *device_count = iree_hip_device_selection_filter(
        frontend, /*rocr=*/false, *device_count, source_infos, device_infos);
  }
  iree_allocator_free(host_allocator, source_infos);
  return iree_ok_status();
}
