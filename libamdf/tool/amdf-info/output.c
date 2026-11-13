// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/tool/amdf-info/output.h"

#include <assert.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

void amdf_info_output_initialize(FILE* file, const amdf_info_options_t* options,
                                 bool terminal_color,
                                 amdf_info_output_t* out_output) {
  const char* term = getenv("TERM");
  const bool auto_color = terminal_color && !getenv("NO_COLOR") &&
                          (!term || strcmp(term, "dumb") != 0);
  *out_output = (amdf_info_output_t){
      .file = file,
      .format = options->format,
      .details = options->details,
      .color = options->format == AMDF_INFO_FORMAT_TEXT &&
               (options->color == AMDF_INFO_COLOR_ALWAYS ||
                (options->color == AMDF_INFO_COLOR_AUTO && auto_color)),
      .empty = {true},
  };
}

void amdf_info_output_print(amdf_info_output_t* output, const char* format,
                            ...) {
  if (output->failed) {
    return;
  }
  va_list arguments;
  va_start(arguments, format);
  if (vfprintf(output->file, format, arguments) < 0) {
    output->failed = true;
  }
  va_end(arguments);
}

static void amdf_info_output_character(amdf_info_output_t* output, int c) {
  if (!output->failed && fputc(c, output->file) == EOF) {
    output->failed = true;
  }
}

void amdf_info_output_text(amdf_info_output_t* output, const char* value) {
  for (const unsigned char* c = (const unsigned char*)value;
       *c && !output->failed; ++c) {
    if (*c == 0xc2 && c[1] >= 0x80 && c[1] <= 0x9f) {
      amdf_info_output_print(output, "\\u%04x", (unsigned)*++c);
    } else if (*c < 0x20 || *c == 0x7f) {
      amdf_info_output_print(output, "\\x%02x", (unsigned)*c);
    } else {
      amdf_info_output_character(output, *c);
    }
  }
}

void amdf_info_output_heading(amdf_info_output_t* output, const char* value) {
  if (output->color) {
    amdf_info_output_print(output, "\x1b[1m");
  }
  amdf_info_output_text(output, value);
  if (output->color) {
    amdf_info_output_print(output, "\x1b[0m");
  }
}

static void amdf_info_json_quoted(amdf_info_output_t* output,
                                  const char* value) {
  amdf_info_output_character(output, '"');
  for (const unsigned char* c = (const unsigned char*)value;
       *c && !output->failed; ++c) {
    if (*c == '"' || *c == '\\') {
      amdf_info_output_character(output, '\\');
      amdf_info_output_character(output, *c);
    } else if (*c < 0x20 || *c == 0x7f) {
      amdf_info_output_print(output, "\\u%04x", (unsigned)*c);
    } else {
      amdf_info_output_character(output, *c);
    }
  }
  amdf_info_output_character(output, '"');
}

static void amdf_info_json_prefix(amdf_info_output_t* output, const char* key) {
  if (!output->empty[output->depth]) {
    amdf_info_output_character(output, ',');
  }
  if (output->depth) {
    amdf_info_output_print(output, "\n%*s", (int)(output->depth * 2), "");
  }
  output->empty[output->depth] = false;
  if (key) {
    amdf_info_json_quoted(output, key);
    amdf_info_output_print(output, ": ");
  }
}

void amdf_info_json_begin(amdf_info_output_t* output, const char* key,
                          char delimiter) {
  // Schema changes must retain the bounded stack or deliberately resize it.
  assert(output->depth + 1 < sizeof(output->empty) / sizeof(output->empty[0]));
  amdf_info_json_prefix(output, key);
  amdf_info_output_character(output, delimiter);
  output->empty[++output->depth] = true;
}

void amdf_info_json_end(amdf_info_output_t* output, char delimiter) {
  const bool empty = output->empty[output->depth--];
  if (!empty) {
    amdf_info_output_print(output, "\n%*s", (int)(output->depth * 2), "");
  }
  amdf_info_output_character(output, delimiter);
}

void amdf_info_json_string(amdf_info_output_t* output, const char* key,
                           const char* value) {
  amdf_info_json_prefix(output, key);
  if (value) {
    amdf_info_json_quoted(output, value);
  } else {
    amdf_info_output_print(output, "null");
  }
}

void amdf_info_json_u32(amdf_info_output_t* output, const char* key,
                        uint32_t value) {
  amdf_info_json_prefix(output, key);
  amdf_info_output_print(output, "%" PRIu32, value);
}

void amdf_info_json_u64(amdf_info_output_t* output, const char* key,
                        uint64_t value) {
  amdf_info_json_prefix(output, key);
  amdf_info_output_print(output, "\"%" PRIu64 "\"", value);
}

void amdf_info_json_hex(amdf_info_output_t* output, const char* key,
                        uint64_t value) {
  amdf_info_json_prefix(output, key);
  amdf_info_output_print(output, "\"0x%016" PRIx64 "\"", value);
}

void amdf_info_json_bool(amdf_info_output_t* output, const char* key,
                         bool value) {
  amdf_info_json_prefix(output, key);
  amdf_info_output_print(output, "%s", value ? "true" : "false");
}

const char* amdf_info_name(uint64_t value, const amdf_info_name_t* names) {
  for (; names->name; ++names) {
    if (names->value == value) {
      return names->name;
    }
  }
  return NULL;
}

void amdf_info_json_enum(amdf_info_output_t* output, const char* key,
                         uint32_t value, const amdf_info_name_t* names) {
  amdf_info_json_begin(output, key, '{');
  amdf_info_json_u32(output, "value", value);
  amdf_info_json_string(output, "name", amdf_info_name(value, names));
  amdf_info_json_end(output, '}');
}

void amdf_info_json_flags(amdf_info_output_t* output, const char* key,
                          uint64_t value, const amdf_info_name_t* names) {
  amdf_info_json_begin(output, key, '{');
  amdf_info_json_hex(output, "value", value);
  amdf_info_json_begin(output, "names", '[');
  for (; names->name; ++names) {
    if (value & names->value) {
      amdf_info_json_string(output, NULL, names->name);
    }
  }
  amdf_info_json_end(output, ']');
  amdf_info_json_end(output, '}');
}

void amdf_info_output_flags(amdf_info_output_t* output, uint64_t value,
                            const amdf_info_name_t* names) {
  bool first = true;
  for (; names->name; ++names) {
    if (!(value & names->value)) {
      continue;
    }
    amdf_info_output_print(output, "%s%s", first ? "" : ", ", names->name);
    value &= ~names->value;
    first = false;
  }
  if (value) {
    amdf_info_output_print(output, "%s0x%" PRIx64, first ? "" : ", ", value);
  } else if (first) {
    amdf_info_output_print(output, "none");
  }
}
