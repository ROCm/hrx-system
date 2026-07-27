// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// B-strings: length-prefixed byte strings for compile-time .rodata tables.
//
// A B-string is stored as [length_byte][char_data...]. The first byte
// encodes the string length (0-255 characters). The character data
// follows immediately with no required NUL terminator, though compile-
// time literals include one from C string concatenation.
//
// B-strings are the standard representation for compile-time-known
// strings in loom .rodata tables: op names, keyword strings, enum case
// names, type names, instance flag names. They encode length without
// runtime strlen, cost 1 byte of overhead per string, and support
// direct comparison against iree_string_view_t parser tokens.
//
// Typical usage in generated tables:
//
//   static const uint8_t my_keyword[] = LOOM_BSTRING_LITERAL(5, "hello");
//   if (loom_bstring_equal(my_keyword, token.text)) { ... }
//
// Large generated descriptor databases should use loom_bstring_table_t:
// descriptor rows store 32-bit offsets into a packed B-string byte table while
// consumers still recover ordinary loom_bstring_t values.
//
// Literal helpers take the byte length as a decimal integer. The macro
// expands that length to a complete one-byte string literal before adjacent
// string-literal concatenation, keeping generated code readable without using
// compound literals or non-portable static initializer tricks.

#ifndef LOOM_UTIL_BSTRING_H_
#define LOOM_UTIL_BSTRING_H_

#include <string.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// A pointer to a length-prefixed byte string: [length][data...].
typedef const uint8_t* loom_bstring_t;

// Offset into a packed B-string table.
typedef uint32_t loom_bstring_table_offset_t;

// Sentinel for absent B-string table offsets.
#define LOOM_BSTRING_TABLE_OFFSET_NONE UINT32_MAX

typedef struct loom_bstring_table_t {
  // Packed B-string bytes stored as repeated [length][data...] entries.
  const uint8_t* data;
  // Total number of bytes in data.
  uint32_t data_length;
} loom_bstring_table_t;

// Returns the character count of a B-string.
static inline uint8_t loom_bstring_length(loom_bstring_t bstring) {
  return bstring[0];
}

// Returns a pointer to the character data (past the length byte).
static inline const char* loom_bstring_data(loom_bstring_t bstring) {
  return (const char*)(bstring + 1);
}

// Converts a B-string to an iree_string_view_t. Zero-cost: no strlen,
// just struct construction from the embedded length.
static inline iree_string_view_t loom_bstring_view(loom_bstring_t bstring) {
  return iree_make_string_view((const char*)(bstring + 1), bstring[0]);
}

// Returns true if a B-string equals an iree_string_view_t.
// Checks length first (single byte comparison), then memcmp only
// when lengths match. This is the hot-path replacement for
// iree_string_view_equal(view, iree_make_cstring_view(cstring))
// which would call strlen on every comparison.
static inline bool loom_bstring_equal(loom_bstring_t bstring,
                                      iree_string_view_t view) {
  uint8_t length = bstring[0];
  if (view.size != length) return false;
  return length == 0 || memcmp(view.data, bstring + 1, length) == 0;
}

// Returns whether |offset| names a complete B-string inside |table|.
static inline bool loom_bstring_table_contains(
    const loom_bstring_table_t* table, loom_bstring_table_offset_t offset) {
  if (offset == LOOM_BSTRING_TABLE_OFFSET_NONE || table == NULL ||
      table->data == NULL || offset >= table->data_length) {
    return false;
  }
  const uint32_t remaining = table->data_length - offset;
  return remaining > 0 && table->data[offset] < remaining;
}

// Returns the B-string at |offset|. The offset must be valid in |table|.
static inline loom_bstring_t loom_bstring_table_get(
    const loom_bstring_table_t* table, loom_bstring_table_offset_t offset) {
  IREE_ASSERT(loom_bstring_table_contains(table, offset));
  return table->data + offset;
}

// Returns true and sets |out_bstring| when |offset| names a complete B-string.
static inline bool loom_bstring_table_try_get(
    const loom_bstring_table_t* table, loom_bstring_table_offset_t offset,
    loom_bstring_t* out_bstring) {
  if (out_bstring != NULL) *out_bstring = NULL;
  if (!loom_bstring_table_contains(table, offset)) return false;
  if (out_bstring != NULL) *out_bstring = loom_bstring_table_get(table, offset);
  return true;
}

//===----------------------------------------------------------------------===//
// Literal helpers
//===----------------------------------------------------------------------===//

// One-byte string literal tokens used by LOOM_BSTRING_LITERAL. Keep each
// escape in its own complete string literal so hexadecimal escapes cannot
// consume bytes from the following data literal during concatenation.
#define LOOM_BSTRING_BYTE_0 "\x00"
#define LOOM_BSTRING_BYTE_1 "\x01"
#define LOOM_BSTRING_BYTE_2 "\x02"
#define LOOM_BSTRING_BYTE_3 "\x03"
#define LOOM_BSTRING_BYTE_4 "\x04"
#define LOOM_BSTRING_BYTE_5 "\x05"
#define LOOM_BSTRING_BYTE_6 "\x06"
#define LOOM_BSTRING_BYTE_7 "\x07"
#define LOOM_BSTRING_BYTE_8 "\x08"
#define LOOM_BSTRING_BYTE_9 "\x09"
#define LOOM_BSTRING_BYTE_10 "\x0a"
#define LOOM_BSTRING_BYTE_11 "\x0b"
#define LOOM_BSTRING_BYTE_12 "\x0c"
#define LOOM_BSTRING_BYTE_13 "\x0d"
#define LOOM_BSTRING_BYTE_14 "\x0e"
#define LOOM_BSTRING_BYTE_15 "\x0f"
#define LOOM_BSTRING_BYTE_16 "\x10"
#define LOOM_BSTRING_BYTE_17 "\x11"
#define LOOM_BSTRING_BYTE_18 "\x12"
#define LOOM_BSTRING_BYTE_19 "\x13"
#define LOOM_BSTRING_BYTE_20 "\x14"
#define LOOM_BSTRING_BYTE_21 "\x15"
#define LOOM_BSTRING_BYTE_22 "\x16"
#define LOOM_BSTRING_BYTE_23 "\x17"
#define LOOM_BSTRING_BYTE_24 "\x18"
#define LOOM_BSTRING_BYTE_25 "\x19"
#define LOOM_BSTRING_BYTE_26 "\x1a"
#define LOOM_BSTRING_BYTE_27 "\x1b"
#define LOOM_BSTRING_BYTE_28 "\x1c"
#define LOOM_BSTRING_BYTE_29 "\x1d"
#define LOOM_BSTRING_BYTE_30 "\x1e"
#define LOOM_BSTRING_BYTE_31 "\x1f"
#define LOOM_BSTRING_BYTE_32 "\x20"
#define LOOM_BSTRING_BYTE_33 "\x21"
#define LOOM_BSTRING_BYTE_34 "\x22"
#define LOOM_BSTRING_BYTE_35 "\x23"
#define LOOM_BSTRING_BYTE_36 "\x24"
#define LOOM_BSTRING_BYTE_37 "\x25"
#define LOOM_BSTRING_BYTE_38 "\x26"
#define LOOM_BSTRING_BYTE_39 "\x27"
#define LOOM_BSTRING_BYTE_40 "\x28"
#define LOOM_BSTRING_BYTE_41 "\x29"
#define LOOM_BSTRING_BYTE_42 "\x2a"
#define LOOM_BSTRING_BYTE_43 "\x2b"
#define LOOM_BSTRING_BYTE_44 "\x2c"
#define LOOM_BSTRING_BYTE_45 "\x2d"
#define LOOM_BSTRING_BYTE_46 "\x2e"
#define LOOM_BSTRING_BYTE_47 "\x2f"
#define LOOM_BSTRING_BYTE_48 "\x30"
#define LOOM_BSTRING_BYTE_49 "\x31"
#define LOOM_BSTRING_BYTE_50 "\x32"
#define LOOM_BSTRING_BYTE_51 "\x33"
#define LOOM_BSTRING_BYTE_52 "\x34"
#define LOOM_BSTRING_BYTE_53 "\x35"
#define LOOM_BSTRING_BYTE_54 "\x36"
#define LOOM_BSTRING_BYTE_55 "\x37"
#define LOOM_BSTRING_BYTE_56 "\x38"
#define LOOM_BSTRING_BYTE_57 "\x39"
#define LOOM_BSTRING_BYTE_58 "\x3a"
#define LOOM_BSTRING_BYTE_59 "\x3b"
#define LOOM_BSTRING_BYTE_60 "\x3c"
#define LOOM_BSTRING_BYTE_61 "\x3d"
#define LOOM_BSTRING_BYTE_62 "\x3e"
#define LOOM_BSTRING_BYTE_63 "\x3f"
#define LOOM_BSTRING_BYTE_64 "\x40"
#define LOOM_BSTRING_BYTE_65 "\x41"
#define LOOM_BSTRING_BYTE_66 "\x42"
#define LOOM_BSTRING_BYTE_67 "\x43"
#define LOOM_BSTRING_BYTE_68 "\x44"
#define LOOM_BSTRING_BYTE_69 "\x45"
#define LOOM_BSTRING_BYTE_70 "\x46"
#define LOOM_BSTRING_BYTE_71 "\x47"
#define LOOM_BSTRING_BYTE_72 "\x48"
#define LOOM_BSTRING_BYTE_73 "\x49"
#define LOOM_BSTRING_BYTE_74 "\x4a"
#define LOOM_BSTRING_BYTE_75 "\x4b"
#define LOOM_BSTRING_BYTE_76 "\x4c"
#define LOOM_BSTRING_BYTE_77 "\x4d"
#define LOOM_BSTRING_BYTE_78 "\x4e"
#define LOOM_BSTRING_BYTE_79 "\x4f"
#define LOOM_BSTRING_BYTE_80 "\x50"
#define LOOM_BSTRING_BYTE_81 "\x51"
#define LOOM_BSTRING_BYTE_82 "\x52"
#define LOOM_BSTRING_BYTE_83 "\x53"
#define LOOM_BSTRING_BYTE_84 "\x54"
#define LOOM_BSTRING_BYTE_85 "\x55"
#define LOOM_BSTRING_BYTE_86 "\x56"
#define LOOM_BSTRING_BYTE_87 "\x57"
#define LOOM_BSTRING_BYTE_88 "\x58"
#define LOOM_BSTRING_BYTE_89 "\x59"
#define LOOM_BSTRING_BYTE_90 "\x5a"
#define LOOM_BSTRING_BYTE_91 "\x5b"
#define LOOM_BSTRING_BYTE_92 "\x5c"
#define LOOM_BSTRING_BYTE_93 "\x5d"
#define LOOM_BSTRING_BYTE_94 "\x5e"
#define LOOM_BSTRING_BYTE_95 "\x5f"
#define LOOM_BSTRING_BYTE_96 "\x60"
#define LOOM_BSTRING_BYTE_97 "\x61"
#define LOOM_BSTRING_BYTE_98 "\x62"
#define LOOM_BSTRING_BYTE_99 "\x63"
#define LOOM_BSTRING_BYTE_100 "\x64"
#define LOOM_BSTRING_BYTE_101 "\x65"
#define LOOM_BSTRING_BYTE_102 "\x66"
#define LOOM_BSTRING_BYTE_103 "\x67"
#define LOOM_BSTRING_BYTE_104 "\x68"
#define LOOM_BSTRING_BYTE_105 "\x69"
#define LOOM_BSTRING_BYTE_106 "\x6a"
#define LOOM_BSTRING_BYTE_107 "\x6b"
#define LOOM_BSTRING_BYTE_108 "\x6c"
#define LOOM_BSTRING_BYTE_109 "\x6d"
#define LOOM_BSTRING_BYTE_110 "\x6e"
#define LOOM_BSTRING_BYTE_111 "\x6f"
#define LOOM_BSTRING_BYTE_112 "\x70"
#define LOOM_BSTRING_BYTE_113 "\x71"
#define LOOM_BSTRING_BYTE_114 "\x72"
#define LOOM_BSTRING_BYTE_115 "\x73"
#define LOOM_BSTRING_BYTE_116 "\x74"
#define LOOM_BSTRING_BYTE_117 "\x75"
#define LOOM_BSTRING_BYTE_118 "\x76"
#define LOOM_BSTRING_BYTE_119 "\x77"
#define LOOM_BSTRING_BYTE_120 "\x78"
#define LOOM_BSTRING_BYTE_121 "\x79"
#define LOOM_BSTRING_BYTE_122 "\x7a"
#define LOOM_BSTRING_BYTE_123 "\x7b"
#define LOOM_BSTRING_BYTE_124 "\x7c"
#define LOOM_BSTRING_BYTE_125 "\x7d"
#define LOOM_BSTRING_BYTE_126 "\x7e"
#define LOOM_BSTRING_BYTE_127 "\x7f"
#define LOOM_BSTRING_BYTE_128 "\x80"
#define LOOM_BSTRING_BYTE_129 "\x81"
#define LOOM_BSTRING_BYTE_130 "\x82"
#define LOOM_BSTRING_BYTE_131 "\x83"
#define LOOM_BSTRING_BYTE_132 "\x84"
#define LOOM_BSTRING_BYTE_133 "\x85"
#define LOOM_BSTRING_BYTE_134 "\x86"
#define LOOM_BSTRING_BYTE_135 "\x87"
#define LOOM_BSTRING_BYTE_136 "\x88"
#define LOOM_BSTRING_BYTE_137 "\x89"
#define LOOM_BSTRING_BYTE_138 "\x8a"
#define LOOM_BSTRING_BYTE_139 "\x8b"
#define LOOM_BSTRING_BYTE_140 "\x8c"
#define LOOM_BSTRING_BYTE_141 "\x8d"
#define LOOM_BSTRING_BYTE_142 "\x8e"
#define LOOM_BSTRING_BYTE_143 "\x8f"
#define LOOM_BSTRING_BYTE_144 "\x90"
#define LOOM_BSTRING_BYTE_145 "\x91"
#define LOOM_BSTRING_BYTE_146 "\x92"
#define LOOM_BSTRING_BYTE_147 "\x93"
#define LOOM_BSTRING_BYTE_148 "\x94"
#define LOOM_BSTRING_BYTE_149 "\x95"
#define LOOM_BSTRING_BYTE_150 "\x96"
#define LOOM_BSTRING_BYTE_151 "\x97"
#define LOOM_BSTRING_BYTE_152 "\x98"
#define LOOM_BSTRING_BYTE_153 "\x99"
#define LOOM_BSTRING_BYTE_154 "\x9a"
#define LOOM_BSTRING_BYTE_155 "\x9b"
#define LOOM_BSTRING_BYTE_156 "\x9c"
#define LOOM_BSTRING_BYTE_157 "\x9d"
#define LOOM_BSTRING_BYTE_158 "\x9e"
#define LOOM_BSTRING_BYTE_159 "\x9f"
#define LOOM_BSTRING_BYTE_160 "\xa0"
#define LOOM_BSTRING_BYTE_161 "\xa1"
#define LOOM_BSTRING_BYTE_162 "\xa2"
#define LOOM_BSTRING_BYTE_163 "\xa3"
#define LOOM_BSTRING_BYTE_164 "\xa4"
#define LOOM_BSTRING_BYTE_165 "\xa5"
#define LOOM_BSTRING_BYTE_166 "\xa6"
#define LOOM_BSTRING_BYTE_167 "\xa7"
#define LOOM_BSTRING_BYTE_168 "\xa8"
#define LOOM_BSTRING_BYTE_169 "\xa9"
#define LOOM_BSTRING_BYTE_170 "\xaa"
#define LOOM_BSTRING_BYTE_171 "\xab"
#define LOOM_BSTRING_BYTE_172 "\xac"
#define LOOM_BSTRING_BYTE_173 "\xad"
#define LOOM_BSTRING_BYTE_174 "\xae"
#define LOOM_BSTRING_BYTE_175 "\xaf"
#define LOOM_BSTRING_BYTE_176 "\xb0"
#define LOOM_BSTRING_BYTE_177 "\xb1"
#define LOOM_BSTRING_BYTE_178 "\xb2"
#define LOOM_BSTRING_BYTE_179 "\xb3"
#define LOOM_BSTRING_BYTE_180 "\xb4"
#define LOOM_BSTRING_BYTE_181 "\xb5"
#define LOOM_BSTRING_BYTE_182 "\xb6"
#define LOOM_BSTRING_BYTE_183 "\xb7"
#define LOOM_BSTRING_BYTE_184 "\xb8"
#define LOOM_BSTRING_BYTE_185 "\xb9"
#define LOOM_BSTRING_BYTE_186 "\xba"
#define LOOM_BSTRING_BYTE_187 "\xbb"
#define LOOM_BSTRING_BYTE_188 "\xbc"
#define LOOM_BSTRING_BYTE_189 "\xbd"
#define LOOM_BSTRING_BYTE_190 "\xbe"
#define LOOM_BSTRING_BYTE_191 "\xbf"
#define LOOM_BSTRING_BYTE_192 "\xc0"
#define LOOM_BSTRING_BYTE_193 "\xc1"
#define LOOM_BSTRING_BYTE_194 "\xc2"
#define LOOM_BSTRING_BYTE_195 "\xc3"
#define LOOM_BSTRING_BYTE_196 "\xc4"
#define LOOM_BSTRING_BYTE_197 "\xc5"
#define LOOM_BSTRING_BYTE_198 "\xc6"
#define LOOM_BSTRING_BYTE_199 "\xc7"
#define LOOM_BSTRING_BYTE_200 "\xc8"
#define LOOM_BSTRING_BYTE_201 "\xc9"
#define LOOM_BSTRING_BYTE_202 "\xca"
#define LOOM_BSTRING_BYTE_203 "\xcb"
#define LOOM_BSTRING_BYTE_204 "\xcc"
#define LOOM_BSTRING_BYTE_205 "\xcd"
#define LOOM_BSTRING_BYTE_206 "\xce"
#define LOOM_BSTRING_BYTE_207 "\xcf"
#define LOOM_BSTRING_BYTE_208 "\xd0"
#define LOOM_BSTRING_BYTE_209 "\xd1"
#define LOOM_BSTRING_BYTE_210 "\xd2"
#define LOOM_BSTRING_BYTE_211 "\xd3"
#define LOOM_BSTRING_BYTE_212 "\xd4"
#define LOOM_BSTRING_BYTE_213 "\xd5"
#define LOOM_BSTRING_BYTE_214 "\xd6"
#define LOOM_BSTRING_BYTE_215 "\xd7"
#define LOOM_BSTRING_BYTE_216 "\xd8"
#define LOOM_BSTRING_BYTE_217 "\xd9"
#define LOOM_BSTRING_BYTE_218 "\xda"
#define LOOM_BSTRING_BYTE_219 "\xdb"
#define LOOM_BSTRING_BYTE_220 "\xdc"
#define LOOM_BSTRING_BYTE_221 "\xdd"
#define LOOM_BSTRING_BYTE_222 "\xde"
#define LOOM_BSTRING_BYTE_223 "\xdf"
#define LOOM_BSTRING_BYTE_224 "\xe0"
#define LOOM_BSTRING_BYTE_225 "\xe1"
#define LOOM_BSTRING_BYTE_226 "\xe2"
#define LOOM_BSTRING_BYTE_227 "\xe3"
#define LOOM_BSTRING_BYTE_228 "\xe4"
#define LOOM_BSTRING_BYTE_229 "\xe5"
#define LOOM_BSTRING_BYTE_230 "\xe6"
#define LOOM_BSTRING_BYTE_231 "\xe7"
#define LOOM_BSTRING_BYTE_232 "\xe8"
#define LOOM_BSTRING_BYTE_233 "\xe9"
#define LOOM_BSTRING_BYTE_234 "\xea"
#define LOOM_BSTRING_BYTE_235 "\xeb"
#define LOOM_BSTRING_BYTE_236 "\xec"
#define LOOM_BSTRING_BYTE_237 "\xed"
#define LOOM_BSTRING_BYTE_238 "\xee"
#define LOOM_BSTRING_BYTE_239 "\xef"
#define LOOM_BSTRING_BYTE_240 "\xf0"
#define LOOM_BSTRING_BYTE_241 "\xf1"
#define LOOM_BSTRING_BYTE_242 "\xf2"
#define LOOM_BSTRING_BYTE_243 "\xf3"
#define LOOM_BSTRING_BYTE_244 "\xf4"
#define LOOM_BSTRING_BYTE_245 "\xf5"
#define LOOM_BSTRING_BYTE_246 "\xf6"
#define LOOM_BSTRING_BYTE_247 "\xf7"
#define LOOM_BSTRING_BYTE_248 "\xf8"
#define LOOM_BSTRING_BYTE_249 "\xf9"
#define LOOM_BSTRING_BYTE_250 "\xfa"
#define LOOM_BSTRING_BYTE_251 "\xfb"
#define LOOM_BSTRING_BYTE_252 "\xfc"
#define LOOM_BSTRING_BYTE_253 "\xfd"
#define LOOM_BSTRING_BYTE_254 "\xfe"
#define LOOM_BSTRING_BYTE_255 "\xff"

#define LOOM_BSTRING_BYTE_IMPL(length) LOOM_BSTRING_BYTE_##length
#define LOOM_BSTRING_BYTE(length) LOOM_BSTRING_BYTE_IMPL(length)

// Emits one B-string literal fragment for packed compile-time tables.
// |data| must be a string literal whose byte length matches |length|.
#define LOOM_BSTRING_LITERAL(length, data) LOOM_BSTRING_BYTE(length) data

// Emits a B-string pointer to static string-literal storage.
#define LOOM_BSTRING_REF(length, data) \
  ((loom_bstring_t)LOOM_BSTRING_LITERAL(length, data))

// Emits an op-name literal fragment: [length][namespace_length][data...].
#define LOOM_OP_NAME_LITERAL(length, namespace_length, data) \
  LOOM_BSTRING_BYTE(length) LOOM_BSTRING_BYTE(namespace_length) data

// Emits an op-name pointer to static string-literal storage.
#define LOOM_OP_NAME_REF(length, namespace_length, data) \
  ((loom_bstring_t)LOOM_OP_NAME_LITERAL(length, namespace_length, data))

#ifdef __cplusplus
}
#endif

#endif  // LOOM_UTIL_BSTRING_H_
