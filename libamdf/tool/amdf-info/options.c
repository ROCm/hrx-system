// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/tool/amdf-info/options.h"

#include <string.h>

const char* amdf_info_options_parse(int argument_count,
                                    const char* const* arguments,
                                    amdf_info_options_t* out_options) {
  amdf_info_options_t options = {0};
  bool format_seen = false;
  bool color_seen = false;
  for (int i = 1; i < argument_count; ++i) {
    const char* argument = arguments[i];
    const char* value = strchr(argument, '=');
    const size_t name_length =
        value ? (size_t)(value - argument) : strlen(argument);
    if (strcmp(argument, "select") == 0 && i == 1) {
      options.select = true;
      continue;
    }
    if (strcmp(argument, "--help") == 0 || strcmp(argument, "-h") == 0) {
      options.help = true;
      continue;
    }
    if (strcmp(argument, "--details") == 0) {
      options.details = true;
      continue;
    }
    if (strcmp(argument, "--activate") == 0) {
      options.activate = true;
      continue;
    }
    enum { DEVICE, TARGET, ENGINE, FORMAT, COLOR } option;
    if (name_length == 8 && strncmp(argument, "--device", 8) == 0) {
      option = DEVICE;
    } else if (name_length == 8 && strncmp(argument, "--target", 8) == 0) {
      option = TARGET;
    } else if (name_length == 8 && strncmp(argument, "--engine", 8) == 0) {
      option = ENGINE;
    } else if (name_length == 8 && strncmp(argument, "--format", 8) == 0) {
      option = FORMAT;
    } else if (name_length == 7 && strncmp(argument, "--color", 7) == 0) {
      option = COLOR;
    } else {
      return "unknown argument; use --help for supported options";
    }
    if (value) {
      ++value;
    } else if (i + 1 < argument_count) {
      value = arguments[++i];
    } else {
      return "option requires a value";
    }
    if (!value[0] || value[0] == '-') {
      return "option requires a nonempty value";
    }
    switch (option) {
      case DEVICE:
        if (options.device) {
          return "--device may be specified only once";
        }
        options.device = value;
        break;
      case TARGET:
        if (options.target) {
          return "--target may be specified only once";
        }
        options.target = value;
        break;
      case ENGINE:
        if (options.engine != AMDF_ENGINE_KIND_UNKNOWN) {
          return "--engine may be specified only once";
        }
        if (strcmp(value, "gpu") == 0) {
          options.engine = AMDF_ENGINE_KIND_GPU;
        } else if (strcmp(value, "xdna") == 0) {
          options.engine = AMDF_ENGINE_KIND_XDNA;
        } else {
          return "--engine must be gpu or xdna";
        }
        break;
      case FORMAT:
        if (format_seen) {
          return "--format may be specified only once";
        }
        format_seen = true;
        if (strcmp(value, "text") == 0) {
          options.format = AMDF_INFO_FORMAT_TEXT;
        } else if (strcmp(value, "json") == 0) {
          options.format = AMDF_INFO_FORMAT_JSON;
        } else {
          return "--format must be text or json";
        }
        break;
      case COLOR:
        if (color_seen) {
          return "--color may be specified only once";
        }
        color_seen = true;
        if (strcmp(value, "auto") == 0) {
          options.color = AMDF_INFO_COLOR_AUTO;
        } else if (strcmp(value, "always") == 0) {
          options.color = AMDF_INFO_COLOR_ALWAYS;
        } else if (strcmp(value, "never") == 0) {
          options.color = AMDF_INFO_COLOR_NEVER;
        } else {
          return "--color must be auto, always, or never";
        }
        break;
    }
  }
  *out_options = options;
  return NULL;
}

void amdf_info_options_print_usage(FILE* file) {
  fputs(
      "Usage: amdf-info [select] [options]\n"
      "\n"
      "Report AMD GPUs, NPUs, system memory, and native capabilities.\n"
      "\n"
      "  --device=URI       Select rdna, cdna, amdgpu, or xdna; optional\n"
      "                     ://ORDINAL, ://GPU-<16 hex>, or ://id/<32 hex>.\n"
      "                     Linux also accepts a character-device path.\n"
      "  --engine=gpu|xdna  Filter by execution domain.\n"
      "  --target=NAME      Filter by exact compiler target identifier.\n"
      "  --details          Expand text capabilities and exact identities.\n"
      "  --format=text|json Write readable text or lossless JSON to stdout.\n"
      "  --color=POLICY     Text emphasis: auto (default), always, or never.\n"
      "  --activate         Check device creation, queries, and teardown.\n"
      "  --help             Show this help without querying hardware.\n"
      "\n"
      "Inventory and --details do not activate devices. --activate submits\n"
      "no workload and creates no queue. select requires exactly one match.\n"
      "HAL/ROCr visibility variables do not filter this inventory; use exact\n"
      "identity URIs to carry a selection between tools. Ordinals are local\n"
      "to this invocation's unmasked family view.\n"
      "\n"
      "Exit codes: 0 success, 1 operation failure, 2 invalid arguments,\n"
      "3 no match, 4 ambiguous selection. Denied optional inventory queries\n"
      "remain visible without failing independent queries.\n",
      file);
}
