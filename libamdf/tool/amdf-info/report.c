// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/tool/amdf-info/report.h"

#include <errno.h>
#include <inttypes.h>

static const amdf_info_name_t amdf_info_status_domains[] = {
    {AMDF_STATUS_DOMAIN_API, "api"},
    {AMDF_STATUS_DOMAIN_NTSTATUS, "ntstatus"},
    {AMDF_STATUS_DOMAIN_FIRMWARE, "firmware"},
    {AMDF_STATUS_DOMAIN_ERRNO, "errno"},
    {AMDF_STATUS_DOMAIN_WIN32, "win32"},
    {AMDF_STATUS_DOMAIN_HRESULT, "hresult"},
    {0, NULL},
};

static const amdf_info_name_t amdf_info_api_codes[] = {
    {AMDF_STATUS_CODE_OK, "ok"},
    {AMDF_STATUS_CODE_INVALID_ARGUMENT, "invalid argument"},
    {AMDF_STATUS_CODE_OUT_OF_RANGE, "out of range"},
    {AMDF_STATUS_CODE_UNSUPPORTED, "unsupported"},
    {AMDF_STATUS_CODE_NOT_FOUND, "not found"},
    {AMDF_STATUS_CODE_RESOURCE_EXHAUSTED, "resource exhausted"},
    {AMDF_STATUS_CODE_BUSY, "busy"},
    {AMDF_STATUS_CODE_DEADLINE_EXCEEDED, "deadline exceeded"},
    {AMDF_STATUS_CODE_PERMISSION_DENIED, "access denied"},
    {AMDF_STATUS_CODE_DEVICE_LOST, "device lost"},
    {AMDF_STATUS_CODE_VERSION_MISMATCH, "version mismatch"},
    {AMDF_STATUS_CODE_BUFFER_TOO_SMALL, "buffer too small"},
    {AMDF_STATUS_CODE_FAILED_PRECONDITION, "failed precondition"},
    {AMDF_STATUS_CODE_INTERNAL, "internal error"},
    {0, NULL},
};

static bool amdf_info_status_is_denied(amdf_status_t status) {
  const uint32_t code = amdf_status_code(status);
  switch (amdf_status_domain(status)) {
    case AMDF_STATUS_DOMAIN_API:
      return code == AMDF_STATUS_CODE_PERMISSION_DENIED;
    case AMDF_STATUS_DOMAIN_ERRNO:
      return code == EACCES || code == EPERM;
    case AMDF_STATUS_DOMAIN_NTSTATUS:
      // STATUS_ACCESS_DENIED, also returned by the Linux DXG provider.
      return code == UINT32_C(0xc0000022);
    case AMDF_STATUS_DOMAIN_WIN32:
      return code == 5;  // ERROR_ACCESS_DENIED.
    case AMDF_STATUS_DOMAIN_HRESULT:
      return code == UINT32_C(0x80070005);  // E_ACCESSDENIED.
    default:
      return false;
  }
}

bool amdf_info_status_is_optional(amdf_status_t status) {
  return amdf_info_status_is_denied(status) ||
         (amdf_status_domain(status) == AMDF_STATUS_DOMAIN_API &&
          amdf_status_code(status) == AMDF_STATUS_CODE_UNSUPPORTED);
}

static const char* amdf_info_status_description(amdf_status_t status) {
  if (amdf_info_status_is_denied(status)) {
    return "access denied";
  }
  if (amdf_status_domain(status) == AMDF_STATUS_DOMAIN_API) {
    const char* name =
        amdf_info_name(amdf_status_code(status), amdf_info_api_codes);
    if (name) {
      return name;
    }
  }
  return "native operation failed";
}

void amdf_info_report_error(amdf_info_output_t* output, const char* field,
                            const char* operation, amdf_status_t status) {
  if (output->format == AMDF_INFO_FORMAT_JSON) {
    amdf_info_json_begin(output, field, '{');
    amdf_info_json_begin(output, "error", '{');
    amdf_info_json_string(output, "operation", operation);
    amdf_info_json_enum(output, "domain", amdf_status_domain(status),
                        amdf_info_status_domains);
    amdf_info_json_u32(output, "code", amdf_status_code(status));
    amdf_info_json_string(output, "message",
                          amdf_info_status_description(status));
    amdf_info_json_end(output, '}');
    amdf_info_json_end(output, '}');
  } else {
    const char* domain =
        amdf_info_name(amdf_status_domain(status), amdf_info_status_domains);
    amdf_info_output_print(output, "  Error       ");
    amdf_info_output_text(output, operation);
    amdf_info_output_print(output, ": %s (",
                           amdf_info_status_description(status));
    if (domain) {
      amdf_info_output_print(output, "%s", domain);
    } else {
      amdf_info_output_print(output, "domain %" PRIu32,
                             amdf_status_domain(status));
    }
    amdf_info_output_print(output, ":%" PRIu32 ", 0x%08" PRIx32 ")\n",
                           amdf_status_code(status), amdf_status_code(status));
  }
}

bool amdf_info_report_query(amdf_info_output_t* output,
                            amdf_info_result_t* result, const char* field,
                            const char* operation, amdf_status_t status,
                            amdf_info_query_requirement_t requirement) {
  if (amdf_status_is_ok(status)) {
    return true;
  }
  result->complete = false;
  if (requirement == AMDF_INFO_QUERY_REQUIRED ||
      !amdf_info_status_is_optional(status)) {
    result->exit_code = AMDF_INFO_EXIT_FAILURE;
  }
  amdf_info_report_error(output, field, operation, status);
  return false;
}

void amdf_info_report_selection_error(amdf_info_output_t* output,
                                      amdf_info_exit_code_t exit_code) {
  const char* message =
      exit_code == AMDF_INFO_EXIT_NOT_FOUND
          ? "no endpoint matches the requested selection"
      : exit_code == AMDF_INFO_EXIT_AMBIGUOUS
          ? "multiple endpoints match; use an exact device identity"
          : "selection requires identity or target information that could not "
            "be queried";
  if (output->format == AMDF_INFO_FORMAT_JSON) {
    amdf_info_json_begin(output, "selection", '{');
    amdf_info_json_string(output, "error", message);
    amdf_info_json_end(output, '}');
  } else {
    amdf_info_output_print(output, "\nSelection error: %s\n", message);
  }
}

void amdf_info_report_begin(amdf_info_output_t* output, const char* platform) {
  if (output->format == AMDF_INFO_FORMAT_JSON) {
    amdf_info_json_begin(output, NULL, '{');
    amdf_info_json_u32(output, "schema_version", 1);
    amdf_info_json_string(output, "platform", platform);
  } else {
    amdf_info_output_heading(output, "libamdf");
    amdf_info_output_print(output, " on %s\n", platform);
  }
}

static void amdf_info_report_extension(amdf_info_output_t* output,
                                       const char* name, uint32_t version,
                                       amdf_status_t status) {
  amdf_info_json_begin(output, name, '{');
  amdf_info_json_bool(output, "available", amdf_status_is_ok(status));
  if (amdf_status_is_ok(status)) {
    amdf_info_json_u32(output, "version", version);
  } else {
    amdf_info_report_error(output, "query", "query_extension", status);
  }
  amdf_info_json_end(output, '}');
}

void amdf_info_report_library(amdf_info_output_t* output,
                              const amdf_info_library_t* library) {
  if (output->format == AMDF_INFO_FORMAT_JSON) {
    amdf_info_json_begin(output, "library", '{');
    amdf_info_json_u32(output, "abi_version", library->api->abi_version);
    amdf_info_json_begin(output, "extensions", '{');
    amdf_info_report_extension(
        output, "gpu", library->gpu ? library->gpu->extension_version : 0,
        library->gpu_status);
    amdf_info_report_extension(
        output, "xdna", library->xdna ? library->xdna->extension_version : 0,
        library->xdna_status);
    amdf_info_json_end(output, '}');
    amdf_info_json_end(output, '}');
  } else {
    if (output->details) {
      amdf_info_output_print(output, "  ABI         %" PRIu32 "\n",
                             library->api->abi_version);
    }
    if (library->gpu && output->details) {
      amdf_info_output_print(output, "  GPU API     %" PRIu32 "\n",
                             library->gpu->extension_version);
    } else if (!library->gpu &&
               (output->details ||
                !amdf_info_status_is_optional(library->gpu_status))) {
      amdf_info_report_error(output, "gpu", "query_extension(GPU)",
                             library->gpu_status);
    }
    if (library->xdna && output->details) {
      amdf_info_output_print(output, "  XDNA API    %" PRIu32 "\n",
                             library->xdna->extension_version);
    } else if (!library->xdna &&
               (output->details ||
                !amdf_info_status_is_optional(library->xdna_status))) {
      amdf_info_report_error(output, "xdna", "query_extension(XDNA)",
                             library->xdna_status);
    }
  }
}

void amdf_info_report_end(amdf_info_output_t* output, bool complete,
                          amdf_info_exit_code_t exit_code, bool activated) {
  if (output->format == AMDF_INFO_FORMAT_JSON) {
    amdf_info_json_begin(output, "result", '{');
    amdf_info_json_bool(output, "complete", complete);
    amdf_info_json_u32(output, "exit_code", (uint32_t)exit_code);
    amdf_info_json_end(output, '}');
    amdf_info_json_end(output, '}');
    amdf_info_output_print(output, "\n");
  } else if (activated) {
    amdf_info_output_print(
        output, "\nActivation: %s\n",
        exit_code == AMDF_INFO_EXIT_SUCCESS ? "OK" : "FAILED");
  }
}
