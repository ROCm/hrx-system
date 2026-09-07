// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/encoding/storage.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/base/internal/math.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/encoding/auxiliary.h"
#include "loom/ops/encoding/families.h"
#include "loom/ops/encoding/operand.h"
#include "loom/ops/encoding/ops.h"
#include "loom/util/numeric_format.h"

namespace loom {
namespace {

static const loom_encoding_record_field_t* FindRecordField(
    const loom_encoding_record_layout_t* layout,
    loom_encoding_record_field_role_t role, uint8_t hierarchy_level) {
  for (uint8_t i = 0; i < layout->field_count; ++i) {
    const loom_encoding_record_field_t* field = &layout->fields[i];
    if (field->role == role && field->hierarchy_level == hierarchy_level) {
      return field;
    }
  }
  return nullptr;
}

static std::vector<uint64_t> ProjectRecordField(
    const loom_encoding_record_layout_t* layout,
    const loom_encoding_record_field_t* field,
    const std::vector<uint8_t>& record) {
  std::vector<uint64_t> values(field->element_count, 0);
  for (uint8_t i = 0; i < field->mapping_count; ++i) {
    const loom_encoding_record_mapping_t* mapping =
        &layout->mappings[field->first_mapping_index + i];
    for (uint16_t element = 0; element < mapping->element_count; ++element) {
      uint64_t mapped_bits = 0;
      const uint32_t source_bit_offset =
          mapping->record_bit_offset + element * mapping->record_bit_stride;
      for (uint8_t bit = 0; bit < mapping->bit_count; ++bit) {
        const uint32_t source_bit = source_bit_offset + bit;
        mapped_bits |=
            uint64_t{(record[source_bit / 8] >> (source_bit % 8)) & 1u} << bit;
      }
      values[mapping->field_element_offset + element] |=
          mapped_bits << mapping->field_bit_offset;
    }
  }
  return values;
}

static void ExpectRecordField(const loom_encoding_record_layout_t* layout,
                              loom_encoding_record_field_role_t role,
                              uint8_t hierarchy_level,
                              loom_encoding_numeric_format_t numeric_format,
                              const std::vector<uint8_t>& record,
                              const std::vector<uint64_t>& expected) {
  const loom_encoding_record_field_t* field =
      FindRecordField(layout, role, hierarchy_level);
  ASSERT_NE(field, nullptr);
  EXPECT_EQ(field->numeric_format, numeric_format);
  ASSERT_EQ(field->element_count, expected.size());
  for (uint8_t i = 0; i < field->mapping_count; ++i) {
    const loom_encoding_record_mapping_t* mapping =
        &layout->mappings[field->first_mapping_index + i];
    ASSERT_LE(mapping->bit_count, 64u);
    ASSERT_LE(mapping->field_bit_offset + mapping->bit_count, 64u);
    for (uint16_t element = 0; element < mapping->element_count; ++element) {
      const uint32_t source_bit_offset =
          mapping->record_bit_offset + element * mapping->record_bit_stride;
      for (uint8_t bit = 0; bit < mapping->bit_count; ++bit) {
        const uint32_t source_bit = source_bit_offset + bit;
        ASSERT_LT(source_bit, record.size() * 8);
      }
    }
  }
  EXPECT_EQ(ProjectRecordField(layout, field, record), expected);
}

static void ExpectGgmlRecordTopology(
    const loom_encoding_family_descriptor_t* descriptor,
    const loom_encoding_record_layout_t* layout) {
  std::vector<uint8_t> record(layout->geometry.storage_byte_count);
  for (iree_host_size_t i = 0; i < record.size(); ++i) {
    record[i] = static_cast<uint8_t>(i * 37 + 11);
  }
  auto u16_at = [&](iree_host_size_t byte_offset) {
    return uint64_t{record[byte_offset]} |
           (uint64_t{record[byte_offset + 1]} << 8);
  };

  if (descriptor == &loom_encoding_ggml_q4_0_family_descriptor) {
    ExpectRecordField(layout, LOOM_ENCODING_RECORD_FIELD_SCALE, 0,
                      LOOM_ENCODING_NUMERIC_FORMAT_F16, record, {u16_at(0)});
    std::vector<uint64_t> payload(32);
    for (iree_host_size_t i = 0; i < payload.size(); ++i) {
      const uint8_t packed = record[2 + i % 16];
      payload[i] = (packed >> (i / 16 * 4)) & 0xFu;
    }
    ExpectRecordField(layout, LOOM_ENCODING_RECORD_FIELD_PAYLOAD, 0,
                      LOOM_ENCODING_NUMERIC_FORMAT_QUANT_I4, record, payload);
    return;
  }

  if (descriptor == &loom_encoding_ggml_q8_0_family_descriptor) {
    ExpectRecordField(layout, LOOM_ENCODING_RECORD_FIELD_SCALE, 0,
                      LOOM_ENCODING_NUMERIC_FORMAT_F16, record, {u16_at(0)});
    std::vector<uint64_t> payload(record.begin() + 2, record.end());
    ExpectRecordField(layout, LOOM_ENCODING_RECORD_FIELD_PAYLOAD, 0,
                      LOOM_ENCODING_NUMERIC_FORMAT_QUANT_I8, record, payload);
    return;
  }

  auto expect_k_scale_minimum_fields = [&]() {
    ExpectRecordField(layout, LOOM_ENCODING_RECORD_FIELD_SCALE, 0,
                      LOOM_ENCODING_NUMERIC_FORMAT_F16, record, {u16_at(0)});
    ExpectRecordField(layout, LOOM_ENCODING_RECORD_FIELD_MINIMUM, 0,
                      LOOM_ENCODING_NUMERIC_FORMAT_F16, record, {u16_at(2)});
    std::vector<uint64_t> scales(8);
    std::vector<uint64_t> minimums(8);
    for (iree_host_size_t i = 0; i < 4; ++i) {
      scales[i] = record[4 + i] & 0x3Fu;
      minimums[i] = record[8 + i] & 0x3Fu;
      scales[4 + i] = (record[12 + i] & 0xFu) | ((record[4 + i] >> 6) << 4);
      minimums[4 + i] = (record[12 + i] >> 4) | ((record[8 + i] >> 6) << 4);
    }
    ExpectRecordField(layout, LOOM_ENCODING_RECORD_FIELD_SCALE, 1,
                      LOOM_ENCODING_NUMERIC_FORMAT_U6, record, scales);
    ExpectRecordField(layout, LOOM_ENCODING_RECORD_FIELD_MINIMUM, 1,
                      LOOM_ENCODING_NUMERIC_FORMAT_U6, record, minimums);
  };

  if (descriptor == &loom_encoding_ggml_q4_k_family_descriptor) {
    expect_k_scale_minimum_fields();
    std::vector<uint64_t> payload(256);
    for (iree_host_size_t i = 0; i < payload.size(); ++i) {
      const iree_host_size_t group = i / 32;
      const uint8_t packed = record[16 + group / 2 * 32 + i % 32];
      payload[i] = (packed >> (group % 2 * 4)) & 0xFu;
    }
    ExpectRecordField(layout, LOOM_ENCODING_RECORD_FIELD_PAYLOAD, 0,
                      LOOM_ENCODING_NUMERIC_FORMAT_U4, record, payload);
    return;
  }

  if (descriptor == &loom_encoding_ggml_q5_k_family_descriptor) {
    expect_k_scale_minimum_fields();
    std::vector<uint64_t> payload(256);
    for (iree_host_size_t i = 0; i < payload.size(); ++i) {
      const iree_host_size_t group = i / 32;
      const uint8_t packed = record[48 + group / 2 * 32 + i % 32];
      const uint8_t high_bit = (record[16 + i % 32] >> group) & 1u;
      payload[i] = ((packed >> (group % 2 * 4)) & 0xFu) | (high_bit << 4);
    }
    ExpectRecordField(layout, LOOM_ENCODING_RECORD_FIELD_PAYLOAD, 0,
                      LOOM_ENCODING_NUMERIC_FORMAT_U5, record, payload);
    return;
  }

  if (descriptor == &loom_encoding_ggml_q6_k_family_descriptor) {
    std::vector<uint64_t> payload(256);
    for (iree_host_size_t i = 0; i < payload.size(); ++i) {
      const iree_host_size_t half = i / 128;
      const iree_host_size_t quarter = i % 128 / 32;
      const iree_host_size_t lane = i % 32;
      const uint8_t low = (record[half * 64 + quarter % 2 * 32 + lane] >>
                           (quarter < 2 ? 0 : 4)) &
                          0xFu;
      const uint8_t high =
          (record[128 + half * 32 + lane] >> (quarter * 2)) & 0x3u;
      payload[i] = low | (high << 4);
    }
    ExpectRecordField(layout, LOOM_ENCODING_RECORD_FIELD_PAYLOAD, 0,
                      LOOM_ENCODING_NUMERIC_FORMAT_QUANT_I6, record, payload);
    ExpectRecordField(layout, LOOM_ENCODING_RECORD_FIELD_SCALE, 0,
                      LOOM_ENCODING_NUMERIC_FORMAT_F16, record, {u16_at(208)});
    std::vector<uint64_t> scales(record.begin() + 192, record.begin() + 208);
    ExpectRecordField(layout, LOOM_ENCODING_RECORD_FIELD_SCALE, 1,
                      LOOM_ENCODING_NUMERIC_FORMAT_I8, record, scales);
    return;
  }

  ASSERT_EQ(descriptor, &loom_encoding_ggml_q8_1_x4_family_descriptor);
  std::vector<uint64_t> scales(4);
  std::vector<uint64_t> sums(4);
  for (iree_host_size_t i = 0; i < 4; ++i) {
    scales[i] = u16_at(i * 4);
    sums[i] = u16_at(i * 4 + 2);
  }
  ExpectRecordField(layout, LOOM_ENCODING_RECORD_FIELD_SCALE, 0,
                    LOOM_ENCODING_NUMERIC_FORMAT_F16, record, scales);
  ExpectRecordField(layout, LOOM_ENCODING_RECORD_FIELD_SUM_CORRECTION, 0,
                    LOOM_ENCODING_NUMERIC_FORMAT_F16, record, sums);
  std::vector<uint64_t> payload(record.begin() + 16, record.end());
  ExpectRecordField(layout, LOOM_ENCODING_RECORD_FIELD_PAYLOAD, 0,
                    LOOM_ENCODING_NUMERIC_FORMAT_QUANT_I8, record, payload);
}

enum class GgmlOracleKind {
  kQ4_0,
  kQ8_0,
  kQ4K,
  kQ5K,
  kQ6K,
  kQ8_1X4,
};

enum class GgmlOraclePattern {
  kRandom,
  kZeroCodes,
  kMaximumCodes,
  kAlternatingPlanes,
  kBoundaryMetadataAndScales,
};

struct GgmlOracleCase {
  // Display name identifying this format in a failing oracle case.
  const char* name;
  // Independent GGML layout used to compute reference values.
  GgmlOracleKind kind;
  // Generated encoding contract whose reconstruction is under test.
  const loom_encoding_family_descriptor_t* descriptor;
};

static uint32_t NextOracleRandom(uint32_t* state) {
  *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
  return *state;
}

static void WriteF16Bits(std::vector<uint8_t>* record, iree_host_size_t offset,
                         uint16_t bits) {
  (*record)[offset] = static_cast<uint8_t>(bits);
  (*record)[offset + 1] = static_cast<uint8_t>(bits >> 8);
}

static void WriteF16(std::vector<uint8_t>* record, iree_host_size_t offset,
                     float value) {
  WriteF16Bits(record, offset, iree_math_f32_to_f16(value));
}

static double ReadF16(const std::vector<uint8_t>& record,
                      iree_host_size_t offset) {
  const uint16_t bits = static_cast<uint16_t>(record[offset]) |
                        (static_cast<uint16_t>(record[offset + 1]) << 8);
  return iree_math_f16_to_f64(bits);
}

static int8_t ReadI8(const std::vector<uint8_t>& record,
                     iree_host_size_t offset) {
  return static_cast<int8_t>(record[offset]);
}

static std::vector<uint8_t> MakeGgmlOracleRecord(
    GgmlOracleKind kind, uint16_t storage_byte_count, uint32_t seed,
    GgmlOraclePattern pattern = GgmlOraclePattern::kRandom) {
  std::vector<uint8_t> record(storage_byte_count);
  uint32_t random = seed;
  for (uint8_t& byte : record) {
    byte = static_cast<uint8_t>(NextOracleRandom(&random) >> 24);
  }

  static const float kScales[] = {0.25f, 0.5f, 1.0f, 1.5f, 2.0f};
  auto next_scale = [&]() {
    return kScales[NextOracleRandom(&random) % IREE_ARRAYSIZE(kScales)];
  };
  switch (kind) {
    case GgmlOracleKind::kQ4_0:
    case GgmlOracleKind::kQ8_0:
      WriteF16(&record, 0, next_scale());
      break;
    case GgmlOracleKind::kQ4K:
    case GgmlOracleKind::kQ5K:
      WriteF16(&record, 0, next_scale());
      WriteF16(&record, 2, next_scale());
      break;
    case GgmlOracleKind::kQ6K:
      WriteF16(&record, 208, next_scale());
      break;
    case GgmlOracleKind::kQ8_1X4:
      for (iree_host_size_t group = 0; group < 4; ++group) {
        const float scale = next_scale();
        for (iree_host_size_t lane = 0; lane < 32; ++lane) {
          const int8_t value =
              static_cast<int8_t>(NextOracleRandom(&random) % 31) - 15;
          record[16 + group * 32 + lane] = static_cast<uint8_t>(value);
        }
        WriteF16(&record, group * 4, scale);
      }
      break;
  }

  auto fill_alternating = [&](iree_host_size_t begin, iree_host_size_t end) {
    for (iree_host_size_t i = begin; i < end; ++i) {
      record[i] = static_cast<uint8_t>((i - begin) & 1 ? 0xAA : 0x55);
    }
  };
  switch (pattern) {
    case GgmlOraclePattern::kRandom:
      break;
    case GgmlOraclePattern::kZeroCodes:
      switch (kind) {
        case GgmlOracleKind::kQ4_0:
        case GgmlOracleKind::kQ8_0:
          std::fill(record.begin() + 2, record.end(), 0);
          break;
        case GgmlOracleKind::kQ4K:
          std::fill(record.begin() + 16, record.end(), 0);
          break;
        case GgmlOracleKind::kQ5K:
          std::fill(record.begin() + 16, record.end(), 0);
          break;
        case GgmlOracleKind::kQ6K:
          std::fill(record.begin(), record.begin() + 192, 0);
          break;
        case GgmlOracleKind::kQ8_1X4:
          std::fill(record.begin() + 16, record.end(), 0);
          break;
      }
      break;
    case GgmlOraclePattern::kMaximumCodes:
      switch (kind) {
        case GgmlOracleKind::kQ4_0:
          std::fill(record.begin() + 2, record.end(), 0xFF);
          break;
        case GgmlOracleKind::kQ8_0:
          std::fill(record.begin() + 2, record.end(), 0x7F);
          break;
        case GgmlOracleKind::kQ4K:
          std::fill(record.begin() + 16, record.end(), 0xFF);
          break;
        case GgmlOracleKind::kQ5K:
          std::fill(record.begin() + 16, record.end(), 0xFF);
          break;
        case GgmlOracleKind::kQ6K:
          std::fill(record.begin(), record.begin() + 192, 0xFF);
          break;
        case GgmlOracleKind::kQ8_1X4:
          std::fill(record.begin() + 16, record.end(), 0x7F);
          break;
      }
      break;
    case GgmlOraclePattern::kAlternatingPlanes:
      switch (kind) {
        case GgmlOracleKind::kQ4_0:
        case GgmlOracleKind::kQ8_0:
          fill_alternating(2, record.size());
          break;
        case GgmlOracleKind::kQ4K:
          fill_alternating(16, record.size());
          break;
        case GgmlOracleKind::kQ5K:
          fill_alternating(16, 48);
          fill_alternating(48, record.size());
          break;
        case GgmlOracleKind::kQ6K:
          fill_alternating(0, 128);
          fill_alternating(128, 192);
          break;
        case GgmlOracleKind::kQ8_1X4:
          fill_alternating(16, record.size());
          break;
      }
      break;
    case GgmlOraclePattern::kBoundaryMetadataAndScales:
      switch (kind) {
        case GgmlOracleKind::kQ4_0:
          WriteF16Bits(&record, 0, 0xBC00);
          break;
        case GgmlOracleKind::kQ8_0:
          WriteF16Bits(&record, 0, 0x0001);
          break;
        case GgmlOracleKind::kQ4K:
        case GgmlOracleKind::kQ5K:
          WriteF16Bits(&record, 0, 0x0001);
          WriteF16Bits(&record, 2, 0x7BFF);
          for (iree_host_size_t i = 0; i < 4; ++i) {
            record[4 + i] = static_cast<uint8_t>(i & 1 ? 0xFF : 0x00);
            record[8 + i] = static_cast<uint8_t>(i & 1 ? 0x00 : 0xFF);
            record[12 + i] = static_cast<uint8_t>(i & 1 ? 0x0F : 0xF0);
          }
          break;
        case GgmlOracleKind::kQ6K:
          WriteF16Bits(&record, 208, 0x7BFF);
          for (iree_host_size_t i = 0; i < 16; ++i) {
            static const uint8_t kBoundaryScales[] = {0x80, 0x00, 0x7F};
            record[192 + i] =
                kBoundaryScales[i % IREE_ARRAYSIZE(kBoundaryScales)];
          }
          break;
        case GgmlOracleKind::kQ8_1X4: {
          static const uint16_t kBoundaryScales[] = {0x0000, 0x0001, 0x7BFF,
                                                     0xBC00};
          for (iree_host_size_t group = 0; group < 4; ++group) {
            WriteF16Bits(&record, group * 4, kBoundaryScales[group]);
          }
          break;
        }
      }
      break;
  }

  if (kind == GgmlOracleKind::kQ8_1X4) {
    for (iree_host_size_t group = 0; group < 4; ++group) {
      int32_t sum = 0;
      for (iree_host_size_t lane = 0; lane < 32; ++lane) {
        sum += ReadI8(record, 16 + group * 32 + lane);
      }
      WriteF16(&record, group * 4 + 2,
               static_cast<float>(ReadF16(record, group * 4) * sum));
    }
  }
  return record;
}

static int64_t SignExtendRecordField(uint64_t value, uint8_t bit_count) {
  const uint64_t sign_bit = UINT64_C(1) << (bit_count - 1);
  return static_cast<int64_t>((value ^ sign_bit) - sign_bit);
}

static double InterpretRecordFieldValue(
    const loom_encoding_record_field_t* field, uint64_t value) {
  const auto numeric_format =
      static_cast<loom_encoding_numeric_format_t>(field->numeric_format);
  const loom_numeric_format_info_t* info = nullptr;
  loom_numeric_format_info(loom_encoding_numeric_format_fact(numeric_format),
                           &info);
  if (info->kind == LOOM_NUMERIC_FORMAT_KIND_FLOAT) {
    return iree_math_f16_to_f64(static_cast<uint16_t>(value));
  }
  if (iree_any_bit_set(info->flags, LOOM_NUMERIC_FORMAT_FLAG_OFFSET_BINARY)) {
    return static_cast<double>(value) + info->integer_decode_bias;
  }
  if (iree_any_bit_set(info->flags, LOOM_NUMERIC_FORMAT_FLAG_SIGNED)) {
    return static_cast<double>(
        SignExtendRecordField(value, field->element_bit_count));
  }
  return static_cast<double>(value);
}

struct ProjectedRecordField {
  const loom_encoding_record_field_t* descriptor;
  std::vector<uint64_t> values;
};

static std::vector<double> ReconstructRecordValues(
    const loom_encoding_record_layout_t* layout,
    const std::vector<uint8_t>& record) {
  std::vector<ProjectedRecordField> fields;
  fields.reserve(layout->field_count);
  const ProjectedRecordField* payload = nullptr;
  for (uint8_t i = 0; i < layout->field_count; ++i) {
    fields.push_back({&layout->fields[i],
                      ProjectRecordField(layout, &layout->fields[i], record)});
    if (layout->fields[i].role == LOOM_ENCODING_RECORD_FIELD_PAYLOAD) {
      payload = &fields.back();
    }
  }

  std::vector<double> decoded(layout->geometry.logical_element_count);
  for (uint16_t logical_index = 0;
       logical_index < layout->geometry.logical_element_count;
       ++logical_index) {
    const uint16_t payload_index = loom_encoding_record_field_element_index(
        layout, payload->descriptor, logical_index);
    double value = InterpretRecordFieldValue(payload->descriptor,
                                             payload->values[payload_index]);
    double scale = 1.0;
    double minimum = 1.0;
    bool has_minimum = false;
    for (const ProjectedRecordField& field : fields) {
      const uint16_t field_index = loom_encoding_record_field_element_index(
          layout, field.descriptor, logical_index);
      if (field.descriptor->role == LOOM_ENCODING_RECORD_FIELD_SCALE) {
        scale *= InterpretRecordFieldValue(field.descriptor,
                                           field.values[field_index]);
      } else if (field.descriptor->role == LOOM_ENCODING_RECORD_FIELD_MINIMUM) {
        minimum *= InterpretRecordFieldValue(field.descriptor,
                                             field.values[field_index]);
        has_minimum = true;
      }
    }
    value *= scale;
    decoded[logical_index] = has_minimum ? value - minimum : value;
  }
  return decoded;
}

static void ReadKScaleMinimum(const std::vector<uint8_t>& record,
                              iree_host_size_t group, uint8_t* out_scale,
                              uint8_t* out_minimum) {
  if (group < 4) {
    *out_scale = record[4 + group] & 0x3Fu;
    *out_minimum = record[8 + group] & 0x3Fu;
    return;
  }
  const iree_host_size_t lane = group - 4;
  *out_scale = (record[12 + lane] & 0xFu) |
               static_cast<uint8_t>((record[4 + lane] >> 6) << 4);
  *out_minimum = (record[12 + lane] >> 4) |
                 static_cast<uint8_t>((record[8 + lane] >> 6) << 4);
}

static uint8_t ReadQ4KCode(const std::vector<uint8_t>& record,
                           iree_host_size_t logical_index,
                           iree_host_size_t payload_offset) {
  const iree_host_size_t group = logical_index / 32;
  const uint8_t packed =
      record[payload_offset + group / 2 * 32 + logical_index % 32];
  return static_cast<uint8_t>((packed >> (group % 2 * 4)) & 0xFu);
}

static uint8_t ReadQ5KCode(const std::vector<uint8_t>& record,
                           iree_host_size_t logical_index) {
  const iree_host_size_t group = logical_index / 32;
  const uint8_t low = ReadQ4KCode(record, logical_index, 48);
  const uint8_t high =
      static_cast<uint8_t>((record[16 + logical_index % 32] >> group) & 1u);
  return static_cast<uint8_t>(low | (high << 4));
}

// These independent byte formulas follow ggml-quants.c and ggml-common.h at
// llama.cpp commit 6a1a922d269908a29cbd4b49c27e6a8e7fd10fae. They are kept
// separate from Loom's generated record mappings so agreement proves the
// tables instead of restating them through the same implementation.
static std::vector<double> DecodeGgmlOracle(
    GgmlOracleKind kind, const std::vector<uint8_t>& record) {
  iree_host_size_t logical_element_count = 0;
  switch (kind) {
    case GgmlOracleKind::kQ4_0:
    case GgmlOracleKind::kQ8_0:
      logical_element_count = 32;
      break;
    case GgmlOracleKind::kQ4K:
    case GgmlOracleKind::kQ5K:
    case GgmlOracleKind::kQ6K:
      logical_element_count = 256;
      break;
    case GgmlOracleKind::kQ8_1X4:
      logical_element_count = 128;
      break;
  }
  std::vector<double> decoded(logical_element_count);
  for (iree_host_size_t i = 0; i < logical_element_count; ++i) {
    switch (kind) {
      case GgmlOracleKind::kQ4_0: {
        const uint8_t packed = record[2 + i % 16];
        const int32_t code = ((packed >> (i / 16 * 4)) & 0xFu) - 8;
        decoded[i] = code * ReadF16(record, 0);
        break;
      }
      case GgmlOracleKind::kQ8_0:
        decoded[i] = ReadI8(record, 2 + i) * ReadF16(record, 0);
        break;
      case GgmlOracleKind::kQ4K:
      case GgmlOracleKind::kQ5K: {
        const iree_host_size_t group = i / 32;
        uint8_t scale = 0;
        uint8_t minimum = 0;
        ReadKScaleMinimum(record, group, &scale, &minimum);
        const uint8_t code = kind == GgmlOracleKind::kQ4K
                                 ? ReadQ4KCode(record, i, 16)
                                 : ReadQ5KCode(record, i);
        decoded[i] =
            code * ReadF16(record, 0) * scale - ReadF16(record, 2) * minimum;
        break;
      }
      case GgmlOracleKind::kQ6K: {
        const iree_host_size_t half = i / 128;
        const iree_host_size_t quarter = i % 128 / 32;
        const iree_host_size_t lane = i % 32;
        const uint8_t low =
            static_cast<uint8_t>(record[half * 64 + quarter % 2 * 32 + lane] >>
                                 (quarter < 2 ? 0 : 4)) &
            0xFu;
        const uint8_t high = static_cast<uint8_t>(
            (record[128 + half * 32 + lane] >> (quarter * 2)) & 0x3u);
        const int32_t code = (low | (high << 4)) - 32;
        decoded[i] = code * ReadF16(record, 208) * ReadI8(record, 192 + i / 16);
        break;
      }
      case GgmlOracleKind::kQ8_1X4:
        decoded[i] = ReadI8(record, 16 + i) * ReadF16(record, i / 32 * 4);
        break;
    }
  }
  return decoded;
}

static double DotOracleValues(const std::vector<double>& values,
                              uint32_t seed) {
  double result = 0.0;
  for (iree_host_size_t i = 0; i < values.size(); ++i) {
    const int32_t activation =
        static_cast<int32_t>((i * 17 + seed * 13) % 29) - 14;
    result += values[i] * (activation * 0.125);
  }
  return result;
}

static double DotKRecordWithQ8_1X4(GgmlOracleKind weight_kind,
                                   const std::vector<uint8_t>& weight,
                                   const std::vector<uint8_t> activations[2]) {
  double result = 0.0;
  for (iree_host_size_t group = 0; group < 8; ++group) {
    const std::vector<uint8_t>& activation = activations[group / 4];
    const iree_host_size_t activation_group = group % 4;
    int32_t integer_dot = 0;
    for (iree_host_size_t lane = 0; lane < 32; ++lane) {
      const iree_host_size_t weight_index = group * 32 + lane;
      const uint8_t weight_code = weight_kind == GgmlOracleKind::kQ4K
                                      ? ReadQ4KCode(weight, weight_index, 16)
                                      : ReadQ5KCode(weight, weight_index);
      integer_dot +=
          weight_code * ReadI8(activation, 16 + activation_group * 32 + lane);
    }
    uint8_t scale = 0;
    uint8_t minimum = 0;
    ReadKScaleMinimum(weight, group, &scale, &minimum);
    result += integer_dot * ReadF16(weight, 0) * scale *
                  ReadF16(activation, activation_group * 4) -
              ReadF16(weight, 2) * minimum *
                  ReadF16(activation, activation_group * 4 + 2);
  }
  return result;
}

static const loom_encoding_family_fixed_metadata_t kFixedRecordMetadata = {
    /*.operand_summary=*/{},
    /*.required_auxiliary_keys=*/{},
    /*.record=*/
    {
        /*.geometry=*/
        {
            /*.logical_element_count=*/32,
            /*.storage_byte_count=*/18,
            /*.required_alignment=*/2,
        },
    },
};
static const loom_encoding_family_descriptor_t kFixedRecordDescriptor = {
    /*.name=*/LOOM_BSTRING_REF(17, "test.fixed_record"),
    /*.role=*/LOOM_ENCODING_ROLE_STORAGE_SCHEMA,
    /*.family_flags=*/{},
    /*.parameter_count=*/{},
    /*.parameter_descriptors=*/{},
    /*.dynamic_parameter_count=*/{},
    /*.dynamic_parameter_descriptors=*/{},
    /*.fixed_metadata=*/&kFixedRecordMetadata,
};
static const loom_encoding_vtable_t kFixedRecordVtable = {
    /*.descriptor=*/&kFixedRecordDescriptor,
};

class EncodingStorageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_encoding_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_ENCODING, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_context_register_builtin_encoding_vtables(&context_));
    IREE_ASSERT_OK(
        loom_context_register_encoding_vtable(&context_, &kFixedRecordVtable));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_module_t* Parse(iree_string_view_t source) {
    loom_text_parse_options_t options = {};
    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(loom_text_parse(source, IREE_SV("storage_test.loom"),
                                   &context_, &block_pool_, &options, &module));
    return module;
  }

  uint16_t FirstSpecId(loom_module_t* module) {
    const loom_block_t* body = loom_module_block(module);
    const loom_op_t* op = loom_block_const_op(body, 0);
    EXPECT_TRUE(loom_encoding_define_isa(op));
    return loom_encoding_define_spec(op);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

TEST(EncodingStorageQueryTest, AbsentShapedAttachmentIsDense) {
  const loom_type_kind_t shaped_kinds[] = {
      LOOM_TYPE_TILE,
      LOOM_TYPE_TENSOR,
      LOOM_TYPE_VIEW,
  };
  for (loom_type_kind_t shaped_kind : shaped_kinds) {
    const loom_type_t type = loom_type_shaped_1d(
        shaped_kind, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(16), 0);
    loom_value_fact_address_layout_t layout = {};
    EXPECT_TRUE(loom_encoding_query_type_address_layout(
        /*context=*/nullptr, /*module=*/nullptr, type,
        /*stride_storage=*/nullptr, /*stride_capacity=*/0, &layout));
    EXPECT_EQ(layout.kind, LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE);

    loom_value_fact_storage_schema_t storage_schema = {};
    EXPECT_FALSE(loom_encoding_query_type_storage_schema(
        /*context=*/nullptr, /*module=*/nullptr, type, &storage_schema));
  }

  const loom_type_t vector_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(16), 0);
  loom_value_fact_address_layout_t vector_layout = {};
  EXPECT_FALSE(loom_encoding_query_type_address_layout(
      /*context=*/nullptr, /*module=*/nullptr, vector_type,
      /*stride_storage=*/nullptr, /*stride_capacity=*/0, &vector_layout));
}

TEST_F(EncodingStorageTest, InternExactComposedStorageSummary) {
  loom_module_t* module = Parse(
      IREE_SV("%schema = encoding.define #ggml.q5_k : encoding<schema>\n"));
  ASSERT_NE(module, nullptr);
  const uint16_t schema_encoding_id = FirstSpecId(module);

  loom_value_facts_t stride_facts[] = {
      loom_value_facts_exact_i64(352),
      loom_value_facts_exact_i64(704),
      loom_value_facts_exact_i64(176),
      loom_value_facts_exact_i64(1),
  };
  const loom_value_fact_encoding_summary_t summary = {
      /*.role=*/LOOM_ENCODING_ROLE_PHYSICAL_STORAGE,
      /*.static_spec_encoding_id=*/0,
      /*.address_layout=*/
      {
          /*.kind=*/LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED,
          /*.rank=*/IREE_ARRAYSIZE(stride_facts),
          /*.strides=*/stride_facts,
      },
      /*.storage_schema=*/
      {
          /*.static_spec_encoding_id=*/schema_encoding_id,
      },
  };
  uint16_t storage_encoding_id = 0;
  IREE_ASSERT_OK(loom_encoding_intern_exact_summary(module, &summary,
                                                    &storage_encoding_id));
  ASSERT_NE(storage_encoding_id, 0);

  loom_value_facts_t decoded_stride_facts[IREE_ARRAYSIZE(stride_facts)] = {};
  loom_value_fact_address_layout_t decoded_layout = {};
  ASSERT_TRUE(loom_encoding_query_static_address_layout(
      module, storage_encoding_id, decoded_stride_facts,
      IREE_ARRAYSIZE(decoded_stride_facts), &decoded_layout));
  ASSERT_EQ(decoded_layout.kind, LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED);
  ASSERT_EQ(decoded_layout.rank, IREE_ARRAYSIZE(stride_facts));
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(stride_facts); ++i) {
    int64_t stride = -1;
    ASSERT_TRUE(
        loom_value_facts_as_exact_i64(decoded_layout.strides[i], &stride));
    EXPECT_EQ(stride, stride_facts[i].range_lo);
  }

  loom_value_fact_storage_schema_t decoded_schema = {};
  ASSERT_TRUE(loom_encoding_query_static_storage_schema(
      module, storage_encoding_id, &decoded_schema));
  EXPECT_EQ(decoded_schema.static_spec_encoding_id, schema_encoding_id);

  uint16_t duplicate_encoding_id = 0;
  IREE_ASSERT_OK(loom_encoding_intern_exact_summary(module, &summary,
                                                    &duplicate_encoding_id));
  EXPECT_EQ(duplicate_encoding_id, storage_encoding_id);
  loom_module_free(module);
}

TEST_F(EncodingStorageTest, FixedRecordGeometry) {
  loom_module_t* module =
      Parse(IREE_SV("%schema = encoding.define #test.fixed_record : "
                    "encoding<schema>\n"));
  ASSERT_NE(module, nullptr);

  loom_encoding_record_geometry_t geometry;
  ASSERT_TRUE(loom_encoding_query_static_record_geometry(
      module, FirstSpecId(module), &geometry));
  EXPECT_EQ(geometry.logical_element_count, 32u);
  EXPECT_EQ(geometry.storage_byte_count, 18u);
  EXPECT_EQ(geometry.required_alignment, 2u);

  loom_module_free(module);
}

TEST_F(EncodingStorageTest, FixedGgmlSchemasExposeCanonicalContracts) {
  struct SchemaExpectation {
    // Complete source for one exact provider schema.
    iree_string_view_t source;

    // Generated descriptor carrying the fixed provider metadata.
    const loom_encoding_family_descriptor_t* descriptor;

    // Exact serialized record geometry.
    loom_encoding_record_geometry_t record;

    // Target-independent logical operand semantics.
    loom_encoding_operand_summary_t operand;

    // Explicit values needed after extracting the packed payload.
    loom_encoding_auxiliary_key_flags_t required_auxiliary_keys;
  };

  const SchemaExpectation expectations[] = {
      {
          /*.source=*/IREE_SV(
              "%schema = encoding.define #ggml.q4_0 : encoding<schema>\n"),
          /*.descriptor=*/&loom_encoding_ggml_q4_0_family_descriptor,
          /*.record=*/{32, 18, 2},
          /*.operand=*/
          {
              /*.element_format=*/LOOM_VALUE_FACT_NUMERIC_FORMAT_QUANT_I4,
              /*.scale_format=*/LOOM_VALUE_FACT_NUMERIC_FORMAT_F16,
              /*.secondary_scale_format=*/{},
              /*.payload_packing=*/
              LOOM_VALUE_FACT_PAYLOAD_PACKING_LITTLE_ENDIAN_NIBBLES,
              /*.scale_topology=*/LOOM_VALUE_FACT_SCALE_TOPOLOGY_BLOCK_1D,
              /*.affine_policy=*/LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_ONLY,
              /*.rounding_policy=*/{},
              /*.codebook_policy=*/{},
              /*.sparsity_policy=*/{},
              /*.flags=*/{},
              /*.sparsity_group=*/{},
              /*.payload_register_count=*/{},
              /*.payload_element_count=*/32,
              /*.scale_group=*/{32, {32}},
              /*.scale_operand_count=*/1,
          },
          /*.required_auxiliary_keys=*/
          1ull << LOOM_ENCODING_AUXILIARY_KEY_SCALE,
      },
      {
          /*.source=*/IREE_SV(
              "%schema = encoding.define #ggml.q8_0 : encoding<schema>\n"),
          /*.descriptor=*/&loom_encoding_ggml_q8_0_family_descriptor,
          /*.record=*/{32, 34, 2},
          /*.operand=*/
          {
              /*.element_format=*/LOOM_VALUE_FACT_NUMERIC_FORMAT_QUANT_I8,
              /*.scale_format=*/LOOM_VALUE_FACT_NUMERIC_FORMAT_F16,
              /*.secondary_scale_format=*/{},
              /*.payload_packing=*/LOOM_VALUE_FACT_PAYLOAD_PACKING_DENSE_LANES,
              /*.scale_topology=*/LOOM_VALUE_FACT_SCALE_TOPOLOGY_BLOCK_1D,
              /*.affine_policy=*/LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_ONLY,
              /*.rounding_policy=*/{},
              /*.codebook_policy=*/{},
              /*.sparsity_policy=*/{},
              /*.flags=*/{},
              /*.sparsity_group=*/{},
              /*.payload_register_count=*/{},
              /*.payload_element_count=*/32,
              /*.scale_group=*/{32, {32}},
              /*.scale_operand_count=*/1,
          },
          /*.required_auxiliary_keys=*/
          1ull << LOOM_ENCODING_AUXILIARY_KEY_SCALE,
      },
      {
          /*.source=*/IREE_SV(
              "%schema = encoding.define #ggml.q4_k : encoding<schema>\n"),
          /*.descriptor=*/&loom_encoding_ggml_q4_k_family_descriptor,
          /*.record=*/{256, 144, 2},
          /*.operand=*/
          {
              /*.element_format=*/LOOM_VALUE_FACT_NUMERIC_FORMAT_U4,
              /*.scale_format=*/LOOM_VALUE_FACT_NUMERIC_FORMAT_F16,
              /*.secondary_scale_format=*/LOOM_VALUE_FACT_NUMERIC_FORMAT_U6,
              /*.payload_packing=*/LOOM_VALUE_FACT_PAYLOAD_PACKING_MULTI_STREAM,
              /*.scale_topology=*/LOOM_VALUE_FACT_SCALE_TOPOLOGY_HIERARCHICAL,
              /*.affine_policy=*/LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_PLUS_MIN,
              /*.rounding_policy=*/{},
              /*.codebook_policy=*/{},
              /*.sparsity_policy=*/{},
              /*.flags=*/{},
              /*.sparsity_group=*/{},
              /*.payload_register_count=*/{},
              /*.payload_element_count=*/256,
              /*.scale_group=*/{32, {32}},
              /*.scale_operand_count=*/2,
          },
          /*.required_auxiliary_keys=*/
          (1ull << LOOM_ENCODING_AUXILIARY_KEY_SCALE) |
              (1ull << LOOM_ENCODING_AUXILIARY_KEY_SECONDARY_SCALE) |
              (1ull << LOOM_ENCODING_AUXILIARY_KEY_MINIMUM),
      },
      {
          /*.source=*/IREE_SV(
              "%schema = encoding.define #ggml.q5_k : encoding<schema>\n"),
          /*.descriptor=*/&loom_encoding_ggml_q5_k_family_descriptor,
          /*.record=*/{256, 176, 2},
          /*.operand=*/
          {
              /*.element_format=*/LOOM_VALUE_FACT_NUMERIC_FORMAT_U5,
              /*.scale_format=*/LOOM_VALUE_FACT_NUMERIC_FORMAT_F16,
              /*.secondary_scale_format=*/LOOM_VALUE_FACT_NUMERIC_FORMAT_U6,
              /*.payload_packing=*/LOOM_VALUE_FACT_PAYLOAD_PACKING_MULTI_STREAM,
              /*.scale_topology=*/LOOM_VALUE_FACT_SCALE_TOPOLOGY_HIERARCHICAL,
              /*.affine_policy=*/LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_PLUS_MIN,
              /*.rounding_policy=*/{},
              /*.codebook_policy=*/{},
              /*.sparsity_policy=*/{},
              /*.flags=*/{},
              /*.sparsity_group=*/{},
              /*.payload_register_count=*/{},
              /*.payload_element_count=*/256,
              /*.scale_group=*/{32, {32}},
              /*.scale_operand_count=*/2,
          },
          /*.required_auxiliary_keys=*/
          (1ull << LOOM_ENCODING_AUXILIARY_KEY_SCALE) |
              (1ull << LOOM_ENCODING_AUXILIARY_KEY_SECONDARY_SCALE) |
              (1ull << LOOM_ENCODING_AUXILIARY_KEY_MINIMUM),
      },
      {
          /*.source=*/IREE_SV(
              "%schema = encoding.define #ggml.q6_k : encoding<schema>\n"),
          /*.descriptor=*/&loom_encoding_ggml_q6_k_family_descriptor,
          /*.record=*/{256, 210, 2},
          /*.operand=*/
          {
              /*.element_format=*/LOOM_VALUE_FACT_NUMERIC_FORMAT_QUANT_I6,
              /*.scale_format=*/LOOM_VALUE_FACT_NUMERIC_FORMAT_F16,
              /*.secondary_scale_format=*/LOOM_VALUE_FACT_NUMERIC_FORMAT_I8,
              /*.payload_packing=*/LOOM_VALUE_FACT_PAYLOAD_PACKING_MULTI_STREAM,
              /*.scale_topology=*/LOOM_VALUE_FACT_SCALE_TOPOLOGY_HIERARCHICAL,
              /*.affine_policy=*/
              LOOM_VALUE_FACT_AFFINE_POLICY_SUPER_SCALE_TIMES_SUBSCALE,
              /*.rounding_policy=*/{},
              /*.codebook_policy=*/{},
              /*.sparsity_policy=*/{},
              /*.flags=*/{},
              /*.sparsity_group=*/{},
              /*.payload_register_count=*/{},
              /*.payload_element_count=*/256,
              /*.scale_group=*/{16, {16}},
              /*.scale_operand_count=*/2,
          },
          /*.required_auxiliary_keys=*/
          (1ull << LOOM_ENCODING_AUXILIARY_KEY_SCALE) |
              (1ull << LOOM_ENCODING_AUXILIARY_KEY_SECONDARY_SCALE),
      },
      {
          /*.source=*/IREE_SV(
              "%schema = encoding.define #ggml.q8_1_x4 : encoding<schema>\n"),
          /*.descriptor=*/&loom_encoding_ggml_q8_1_x4_family_descriptor,
          /*.record=*/{128, 144, 16},
          /*.operand=*/
          {
              /*.element_format=*/LOOM_VALUE_FACT_NUMERIC_FORMAT_QUANT_I8,
              /*.scale_format=*/LOOM_VALUE_FACT_NUMERIC_FORMAT_F16,
              /*.secondary_scale_format=*/{},
              /*.payload_packing=*/
              LOOM_VALUE_FACT_PAYLOAD_PACKING_SEPARATE_SCALE_PAYLOAD,
              /*.scale_topology=*/LOOM_VALUE_FACT_SCALE_TOPOLOGY_BLOCK_1D,
              /*.affine_policy=*/LOOM_VALUE_FACT_AFFINE_POLICY_SUM_CORRECTION,
              /*.rounding_policy=*/{},
              /*.codebook_policy=*/{},
              /*.sparsity_policy=*/{},
              /*.flags=*/{},
              /*.sparsity_group=*/{},
              /*.payload_register_count=*/{},
              /*.payload_element_count=*/128,
              /*.scale_group=*/{32, {32}},
              /*.scale_operand_count=*/1,
          },
          /*.required_auxiliary_keys=*/
          (1ull << LOOM_ENCODING_AUXILIARY_KEY_SCALE) |
              (1ull << LOOM_ENCODING_AUXILIARY_KEY_SUM_CORRECTION),
      },
  };

  for (const SchemaExpectation& expectation : expectations) {
    loom_module_t* module = Parse(expectation.source);
    ASSERT_NE(module, nullptr);
    const uint16_t encoding_id = FirstSpecId(module);

    const loom_encoding_record_layout_t* layout = nullptr;
    ASSERT_TRUE(
        loom_encoding_query_static_record_layout(module, encoding_id, &layout));
    ASSERT_NE(layout, nullptr);
    ExpectGgmlRecordTopology(expectation.descriptor, layout);

    loom_encoding_record_geometry_t record;
    ASSERT_TRUE(loom_encoding_query_static_record_geometry(module, encoding_id,
                                                           &record));
    EXPECT_EQ(record.logical_element_count,
              expectation.record.logical_element_count);
    EXPECT_EQ(record.storage_byte_count, expectation.record.storage_byte_count);
    EXPECT_EQ(record.required_alignment, expectation.record.required_alignment);

    loom_value_fact_storage_schema_t schema;
    ASSERT_TRUE(loom_encoding_query_static_storage_schema(module, encoding_id,
                                                          &schema));
    EXPECT_TRUE(loom_value_fact_encoded_operand_schema_equal(
        schema.encoded_operand, expectation.operand));

    loom_encoding_auxiliary_key_flags_t required_auxiliary_keys = 0;
    ASSERT_TRUE(loom_encoding_auxiliary_required_keys_from_schema(
        schema.encoded_operand, &required_auxiliary_keys, nullptr));
    EXPECT_EQ(required_auxiliary_keys, expectation.required_auxiliary_keys);
    EXPECT_EQ(loom_encoding_record_embedded_auxiliary_keys(layout),
              expectation.required_auxiliary_keys);
    ASSERT_NE(expectation.descriptor->fixed_metadata, nullptr);
    EXPECT_EQ(expectation.descriptor->fixed_metadata->required_auxiliary_keys,
              expectation.required_auxiliary_keys);

    loom_module_free(module);
  }
}

TEST(EncodingStorageQueryTest, FixedGgmlRecordsMatchCompleteNumericalOracles) {
  const GgmlOracleCase cases[] = {
      {"q4_0", GgmlOracleKind::kQ4_0,
       &loom_encoding_ggml_q4_0_family_descriptor},
      {"q8_0", GgmlOracleKind::kQ8_0,
       &loom_encoding_ggml_q8_0_family_descriptor},
      {"q4_k", GgmlOracleKind::kQ4K,
       &loom_encoding_ggml_q4_k_family_descriptor},
      {"q5_k", GgmlOracleKind::kQ5K,
       &loom_encoding_ggml_q5_k_family_descriptor},
      {"q6_k", GgmlOracleKind::kQ6K,
       &loom_encoding_ggml_q6_k_family_descriptor},
      {"q8_1_x4", GgmlOracleKind::kQ8_1X4,
       &loom_encoding_ggml_q8_1_x4_family_descriptor},
  };
  struct PatternCase {
    const char* name;
    GgmlOraclePattern pattern;
    uint32_t seed_count;
  };
  const PatternCase patterns[] = {
      {"random", GgmlOraclePattern::kRandom, 8},
      {"zero codes", GgmlOraclePattern::kZeroCodes, 1},
      {"maximum codes", GgmlOraclePattern::kMaximumCodes, 1},
      {"alternating planes", GgmlOraclePattern::kAlternatingPlanes, 1},
      {"boundary metadata and scales",
       GgmlOraclePattern::kBoundaryMetadataAndScales, 1},
  };

  for (const GgmlOracleCase& test_case : cases) {
    ASSERT_NE(test_case.descriptor->fixed_metadata, nullptr);
    const loom_encoding_record_layout_t* layout =
        &test_case.descriptor->fixed_metadata->record;
    for (const PatternCase& pattern : patterns) {
      for (uint32_t seed = 1; seed <= pattern.seed_count; ++seed) {
        SCOPED_TRACE(::testing::Message() << test_case.name << " "
                                          << pattern.name << " seed " << seed);
        const std::vector<uint8_t> record = MakeGgmlOracleRecord(
            test_case.kind, layout->geometry.storage_byte_count, seed,
            pattern.pattern);
        const std::vector<double> reconstructed =
            ReconstructRecordValues(layout, record);
        const std::vector<double> oracle =
            DecodeGgmlOracle(test_case.kind, record);
        ASSERT_EQ(reconstructed.size(), oracle.size());
        for (iree_host_size_t i = 0; i < oracle.size(); ++i) {
          EXPECT_DOUBLE_EQ(reconstructed[i], oracle[i])
              << "logical index " << i;
        }
        EXPECT_DOUBLE_EQ(DotOracleValues(reconstructed, seed),
                         DotOracleValues(oracle, seed));

        if (test_case.kind == GgmlOracleKind::kQ8_1X4) {
          const loom_encoding_record_field_t* sum_field = FindRecordField(
              layout, LOOM_ENCODING_RECORD_FIELD_SUM_CORRECTION, 0);
          ASSERT_NE(sum_field, nullptr);
          const std::vector<uint64_t> sums =
              ProjectRecordField(layout, sum_field, record);
          for (iree_host_size_t group = 0; group < 4; ++group) {
            int32_t payload_sum = 0;
            for (iree_host_size_t lane = 0; lane < 32; ++lane) {
              payload_sum += ReadI8(record, 16 + group * 32 + lane);
            }
            const double rounded_sum =
                iree_math_f16_to_f64(iree_math_f32_to_f16(static_cast<float>(
                    ReadF16(record, group * 4) * payload_sum)));
            EXPECT_DOUBLE_EQ(InterpretRecordFieldValue(sum_field, sums[group]),
                             rounded_sum);
          }
        }
      }
    }
  }
}

TEST(EncodingStorageQueryTest, KQuantDotUsesQ8_1SumCorrection) {
  const GgmlOracleCase weight_cases[] = {
      {"q4_k", GgmlOracleKind::kQ4K,
       &loom_encoding_ggml_q4_k_family_descriptor},
      {"q5_k", GgmlOracleKind::kQ5K,
       &loom_encoding_ggml_q5_k_family_descriptor},
  };
  const loom_encoding_record_layout_t* activation_layout =
      &loom_encoding_ggml_q8_1_x4_family_descriptor.fixed_metadata->record;

  for (const GgmlOracleCase& test_case : weight_cases) {
    const loom_encoding_record_layout_t* weight_layout =
        &test_case.descriptor->fixed_metadata->record;
    for (uint32_t seed = 1; seed <= 8; ++seed) {
      SCOPED_TRACE(::testing::Message() << test_case.name << " seed " << seed);
      const std::vector<uint8_t> weight = MakeGgmlOracleRecord(
          test_case.kind, weight_layout->geometry.storage_byte_count, seed);
      const std::vector<uint8_t> activations[2] = {
          MakeGgmlOracleRecord(GgmlOracleKind::kQ8_1X4,
                               activation_layout->geometry.storage_byte_count,
                               seed * 2),
          MakeGgmlOracleRecord(GgmlOracleKind::kQ8_1X4,
                               activation_layout->geometry.storage_byte_count,
                               seed * 2 + 1),
      };
      const std::vector<double> decoded_weight =
          DecodeGgmlOracle(test_case.kind, weight);
      const std::vector<double> decoded_activation0 =
          DecodeGgmlOracle(GgmlOracleKind::kQ8_1X4, activations[0]);
      const std::vector<double> decoded_activation1 =
          DecodeGgmlOracle(GgmlOracleKind::kQ8_1X4, activations[1]);
      double reference = 0.0;
      for (iree_host_size_t i = 0; i < decoded_weight.size(); ++i) {
        const std::vector<double>& activation =
            i < 128 ? decoded_activation0 : decoded_activation1;
        reference += decoded_weight[i] * activation[i % 128];
      }

      const double corrected =
          DotKRecordWithQ8_1X4(test_case.kind, weight, activations);
      const double tolerance = std::max(1e-9, std::abs(reference) * 1e-12);
      EXPECT_NEAR(corrected, reference, tolerance);
    }
  }
}

TEST_F(EncodingStorageTest, ShapedTypeResolvesFixedRecordLayout) {
  loom_module_t* module =
      Parse(IREE_SV("%schema = encoding.define #ggml.q5_k : "
                    "encoding<schema>\n"));
  ASSERT_NE(module, nullptr);
  const uint16_t encoding_id = FirstSpecId(module);
  const loom_type_t view_type =
      loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I8,
                          loom_dim_pack_static(256), encoding_id);

  const loom_encoding_record_layout_t* layout = nullptr;
  ASSERT_TRUE(loom_encoding_query_type_record_layout(
      /*context=*/nullptr, module, view_type, &layout));
  ASSERT_NE(layout, nullptr);
  EXPECT_EQ(layout->geometry.logical_element_count, 256u);
  EXPECT_EQ(layout->geometry.storage_byte_count, 176u);

  loom_module_free(module);
}

TEST_F(EncodingStorageTest, OperandSummaryHasNoFixedGeometry) {
  loom_module_t* module =
      Parse(IREE_SV("%schema = encoding.define "
                    "#encoding.operand<element_format=f8e4m3fn, "
                    "payload_elements=1, payload_packing=dense_lanes, "
                    "rounding=finite_only> : encoding<schema>\n"));
  ASSERT_NE(module, nullptr);
  const uint16_t encoding_id = FirstSpecId(module);

  loom_encoding_record_geometry_t geometry = {1, 1, 1};
  EXPECT_FALSE(loom_encoding_query_static_record_geometry(module, encoding_id,
                                                          &geometry));
  EXPECT_EQ(geometry.logical_element_count, 0u);
  EXPECT_EQ(geometry.storage_byte_count, 0u);
  EXPECT_EQ(geometry.required_alignment, 0u);

  loom_value_fact_storage_schema_t schema;
  ASSERT_TRUE(
      loom_encoding_query_static_storage_schema(module, encoding_id, &schema));
  EXPECT_EQ(schema.encoded_operand.element_format,
            LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN);
  EXPECT_EQ(schema.encoded_operand.payload_packing,
            LOOM_VALUE_FACT_PAYLOAD_PACKING_DENSE_LANES);
  EXPECT_EQ(schema.encoded_operand.payload_element_count, 1u);
  EXPECT_EQ(schema.encoded_operand.rounding_policy,
            LOOM_VALUE_FACT_ROUNDING_POLICY_FINITE_ONLY);

  loom_module_free(module);
}

TEST_F(EncodingStorageTest, ParameterizedOperandSummary) {
  loom_module_t* module = Parse(IREE_SV(
      "%schema = encoding.define #encoding.operand<affine=scale_only, "
      "codebook=static_builtin_table, element_format=f4e2m1, "
      "payload_elements=128, payload_packing=multi_stream, "
      "payload_registers=16, rounding=finite_only, scale_format=f8e4m3, "
      "scale_group_shape=[8, 16], scale_operands=1, "
      "scale_topology=block_2d, secondary_scale_format=f32, "
      "sparsity=n_m_structured, sparsity_group_elements=4, "
      "sparsity_group_nonzero_elements=2, zero_scale_fallback=true> : "
      "encoding<schema>\n"));
  ASSERT_NE(module, nullptr);

  loom_value_fact_storage_schema_t schema;
  ASSERT_TRUE(loom_encoding_query_static_storage_schema(
      module, FirstSpecId(module), &schema));
  const loom_value_fact_encoded_operand_schema_t& operand =
      schema.encoded_operand;
  EXPECT_EQ(operand.element_format, LOOM_VALUE_FACT_NUMERIC_FORMAT_F4_E2M1);
  EXPECT_EQ(operand.payload_packing,
            LOOM_VALUE_FACT_PAYLOAD_PACKING_MULTI_STREAM);
  EXPECT_EQ(operand.payload_element_count, 128u);
  EXPECT_EQ(operand.payload_register_count, 16u);
  EXPECT_EQ(operand.rounding_policy,
            LOOM_VALUE_FACT_ROUNDING_POLICY_FINITE_ONLY);
  EXPECT_EQ(operand.scale_format, LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3);
  EXPECT_EQ(operand.secondary_scale_format, LOOM_VALUE_FACT_NUMERIC_FORMAT_F32);
  EXPECT_EQ(operand.scale_topology, LOOM_VALUE_FACT_SCALE_TOPOLOGY_BLOCK_2D);
  EXPECT_EQ(operand.scale_group.element_count, 128u);
  EXPECT_EQ(operand.scale_group.shape[0], 8u);
  EXPECT_EQ(operand.scale_group.shape[1], 16u);
  EXPECT_EQ(operand.scale_operand_count, 1u);
  EXPECT_EQ(operand.affine_policy, LOOM_VALUE_FACT_AFFINE_POLICY_SCALE_ONLY);
  EXPECT_EQ(operand.codebook_policy,
            LOOM_VALUE_FACT_CODEBOOK_POLICY_STATIC_BUILTIN_TABLE);
  EXPECT_EQ(operand.sparsity_policy,
            LOOM_VALUE_FACT_SPARSITY_POLICY_N_M_STRUCTURED);
  EXPECT_EQ(operand.sparsity_group.element_count, 4u);
  EXPECT_EQ(operand.sparsity_group.nonzero_element_count, 2u);
  EXPECT_TRUE(iree_all_bits_set(
      operand.flags, LOOM_VALUE_FACT_ENCODED_OPERAND_FLAG_ZERO_SCALE_FALLBACK));

  loom_module_free(module);
}

}  // namespace
}  // namespace loom
