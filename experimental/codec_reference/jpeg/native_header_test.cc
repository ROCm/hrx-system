#include "native_header.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include "iree/base/internal/debugging.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "jpeglib.h"
#include "native_color.h"
#include "native_metadata.h"

namespace {
struct HeaderCase {
  // Independent input identity used by failure diagnostics.
  const char* name;
  // Complete header bytes through SOS, with no body allocation.
  const uint8_t* bytes;
  // Exact readable header extent.
  size_t length;
  // Independent semantic fields, canonical tables, and block geometry.
  const uint32_t* expected;
  // Number of semantic words in expected.
  size_t expected_count;
  // Independently packed device layout, geometry, capacities, and quantizers.
  const uint32_t* metadata;
  // Number of live packed metadata words.
  size_t metadata_count;
  // Canonical oracle codebook identity for each device table slot.
  const uint8_t* table_indices;
  // Color kind in a preserved GPU-qualified mixed fixture, or zero otherwise.
  unsigned fixture_color_kind;
};
struct InvalidCase {
  // Header mutation identifying the violated format contract.
  const char* name;
  // External malformed or explicitly unsupported header bytes.
  const uint8_t* bytes;
  // Complete supplied extent, including the mutated scan header.
  size_t length;
  // Exact expected IREE status code, separate from partial output contents.
  unsigned status;
};

#include "native_header_cases.inc"

std::vector<uint32_t> SemanticWords(const jpeg_header_t& header) {
  std::vector<uint32_t> words = {
      header.width,           header.height,        header.process,
      header.component_count, header.entropy_start, header.restart_interval,
      header.mcu_columns,     header.mcu_rows,      header.blocks_per_mcu};
  for (unsigned i = 0; i < header.component_count; ++i) {
    const auto& component = header.components[i];
    words.insert(words.end(),
                 {component.identifier, component.horizontal,
                  component.vertical, component.quantization, component.dc,
                  component.ac, component.block_columns, component.block_rows});
    for (uint32_t value : header.quantization[component.quantization]) {
      words.push_back(value);
    }
    for (unsigned kind = 0; kind < 2; ++kind) {
      const auto& table =
          header.huffman[kind][kind ? component.ac : component.dc];
      words.push_back(table.symbol_count);
      unsigned code = 0, ordinal = 0;
      for (unsigned length = 1; length <= 16; ++length) {
        for (unsigned entry = 0; entry < table.counts[length - 1]; ++entry) {
          words.insert(words.end(), {length, code++, table.symbols[ordinal++]});
        }
        code <<= 1;
      }
    }
  }
  for (unsigned i = 0; i < header.blocks_per_mcu; ++i) {
    const auto& block = header.blocks[i];
    words.insert(words.end(), {block.component, block.column, block.row});
  }
  return words;
}

TEST(NativeHeaderTest, IndependentSemanticFactsAndUnreadBody) {
  for (const auto& test : positive_cases) {
    SCOPED_TRACE(test.name);
    // Poison the actual following allocation so a body read fails even though
    // the input span includes it. Header length need not be ASAN-granule
    // aligned.
    const size_t aligned = iree_host_align(test.length, 8);
    std::unique_ptr<uint8_t[]> allocation(new uint8_t[aligned + 64]);
    uint8_t* data = allocation.get() + aligned - test.length;
    std::memcpy(data, test.bytes, test.length);
    std::memset(data + test.length, 0x7f, 64);
    IREE_ASAN_POISON_MEMORY_REGION(data + test.length, 64);
    jpeg_header_t header;
    iree_status_t status = jpeg_header_parse(
        iree_make_const_byte_span(data, test.length + 64), &header);
    IREE_ASAN_UNPOISON_MEMORY_REGION(data + test.length, 64);
    IREE_ASSERT_OK(status);
    auto words = SemanticWords(header);
    ASSERT_EQ(words.size(), test.expected_count);
    for (size_t i = 0; i < words.size(); ++i) {
      ASSERT_EQ(words[i], test.expected[i]) << "semantic word " << i;
    }
    IREE_ASSERT_OK(jpeg_header_parse(
        iree_make_const_byte_span(test.bytes, test.length), &header));
    ASSERT_EQ(SemanticWords(header), words);
  }
}

TEST(NativeHeaderTest, EveryHeaderTruncation) {
  // The first twenty cases cover independently encoded corpus files. Synthetic
  // sampling variants share their framing and need no duplicated truncations.
  for (size_t index = 0; index < 20; ++index) {
    const auto& test = positive_cases[index];
    SCOPED_TRACE(test.name);
    for (size_t length = 0; length < test.length; ++length) {
      std::unique_ptr<uint8_t[]> bytes(new uint8_t[length]);
      if (length) {
        std::memcpy(bytes.get(), test.bytes, length);
      }
      jpeg_header_t header;
      iree_status_t status = jpeg_header_parse(
          iree_make_const_byte_span(bytes.get(), length), &header);
      EXPECT_EQ(iree_status_code(status), IREE_STATUS_INVALID_ARGUMENT)
          << length;
      iree_status_ignore(status);
    }
  }
}

TEST(NativeHeaderTest, MalformedAndDistinctProcesses) {
  for (const auto& test : negative_cases) {
    SCOPED_TRACE(test.name);
    jpeg_header_t header;
    iree_status_t status = jpeg_header_parse(
        iree_make_const_byte_span(test.bytes, test.length), &header);
    EXPECT_EQ(iree_status_code(status), test.status);
    iree_status_ignore(status);
  }
}

TEST(NativeHeaderTest, DeviceOffsetExtentBeforeInputAccess) {
  if (sizeof(iree_host_size_t) <= 4) {
    return;
  }
  jpeg_header_t header;
  iree_status_t status = jpeg_header_parse(
      iree_make_const_byte_span(nullptr, uint64_t{UINT32_MAX} + 1), &header);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_RESOURCE_EXHAUSTED);
  iree_status_ignore(status);
}

TEST(NativeHeaderTest, ExactDeviceMetadataAndCodebooks) {
  for (const auto& test : positive_cases) {
    SCOPED_TRACE(test.name);
    jpeg_header_t header;
    IREE_ASSERT_OK(jpeg_header_parse(
        iree_make_const_byte_span(test.bytes, test.length), &header));
    jpeg_device_metadata_t metadata;
    jpeg_device_metadata_initialize(&header, &metadata);
    std::vector<uint32_t> words = {
        metadata.table_count,    metadata.block_count,
        metadata.pixel_words,    metadata.plane_words[0],
        metadata.plane_words[1], metadata.plane_words[2]};
    words.insert(words.end(), metadata.layout, metadata.layout + 4);
    words.insert(words.end(), metadata.geometry,
                 metadata.geometry + header.blocks_per_mcu * 8);
    words.insert(words.end(), metadata.quantization,
                 metadata.quantization + header.component_count * 64);
    ASSERT_EQ(words.size(), test.metadata_count);
    for (size_t i = 0; i < words.size(); ++i) {
      ASSERT_EQ(words[i], test.metadata[i]) << "device metadata word " << i;
    }
    std::unique_ptr<uint32_t[]> tables(
        new uint32_t[metadata.table_count * 2048]);
    jpeg_device_metadata_write_tables(&header, &metadata, tables.get());
    for (unsigned index = 0; index < metadata.table_count; ++index) {
      const uint32_t* expected = expected_tables[test.table_indices[index]];
      for (unsigned word = 0; word < 2048; ++word) {
        ASSERT_EQ(tables[index * 2048 + word], expected[word])
            << "table " << index << " word " << word;
      }
    }
  }
}
TEST(NativeHeaderTest, AliasedSymbolsPublicDecoder) {
  jpeg_decompress_struct decoder = {};
  jpeg_error_mgr error = {};
  decoder.err = jpeg_std_error(&error);
  jpeg_create_decompress(&decoder);
  std::unique_ptr<jpeg_decompress_struct, decltype(&jpeg_destroy_decompress)>
      owner(&decoder, jpeg_destroy_decompress);
  jpeg_mem_src(&decoder, aliased_image, sizeof(aliased_image));
  ASSERT_EQ(jpeg_read_header(&decoder, TRUE), JPEG_HEADER_OK);
  ASSERT_TRUE(jpeg_start_decompress(&decoder));
  ASSERT_EQ(decoder.output_width, 8);
  ASSERT_EQ(decoder.output_height, 16);
  ASSERT_EQ(decoder.output_components, 1);
  while (decoder.output_scanline < decoder.output_height) {
    uint8_t row[8];
    JSAMPROW rows[] = {row};
    ASSERT_EQ(jpeg_read_scanlines(&decoder, rows, 1), 1);
    for (uint8_t value : row) {
      ASSERT_EQ(value, 128);
    }
  }
  ASSERT_TRUE(jpeg_finish_decompress(&decoder));
  ASSERT_EQ(error.num_warnings, 0);
}

TEST(NativeHeaderTest, PublicColorInterpretationAndDeviceProfiles) {
  // Existing color leaf profiles: [kind, color space, H0, V0, H1, V1, H2, V2].
  // Ratios are output pixels per component sample, rather than SOF factors.
  constexpr unsigned profiles[][8] = {
      {1, JCS_GRAYSCALE, 1, 1, 1, 1, 1, 1}, {2, JCS_RGB, 1, 1, 1, 1, 1, 1},
      {3, JCS_YCbCr, 1, 1, 1, 1, 1, 1},     {4, JCS_YCbCr, 1, 1, 2, 1, 2, 1},
      {5, JCS_YCbCr, 1, 1, 2, 2, 2, 2},     {6, JCS_YCbCr, 1, 1, 1, 2, 1, 2},
      {7, JCS_YCbCr, 1, 1, 4, 1, 4, 1},     {8, JCS_YCbCr, 1, 1, 4, 2, 4, 2},
  };
  for (const auto& test : positive_cases) {
    SCOPED_TRACE(test.name);
    jpeg_header_t header;
    IREE_ASSERT_OK(jpeg_header_parse(
        iree_make_const_byte_span(test.bytes, test.length), &header));
    if (header.width > JPEG_MAX_DIMENSION ||
        header.height > JPEG_MAX_DIMENSION) {
      // These three generated endpoint fixtures retain the baseline's JFIF
      // 4:4:4 interpretation. libjpeg caps dimensions below the u16 format
      // limit; the native semantic/geometry tests still cover their complete
      // headers.
      uint32_t kind = 0;
      IREE_ASSERT_OK(jpeg_device_color_kind(&header, &kind));
      EXPECT_EQ(kind, 3);
      continue;
    }
    jpeg_decompress_struct reference = {};
    jpeg_error_mgr error = {};
    reference.err = jpeg_std_error(&error);
    jpeg_create_decompress(&reference);
    std::unique_ptr<jpeg_decompress_struct, decltype(&jpeg_destroy_decompress)>
        owner(&reference, jpeg_destroy_decompress);
    jpeg_mem_src(&reference, test.bytes, test.length);
    ASSERT_EQ(jpeg_read_header(&reference, TRUE), JPEG_HEADER_OK);
    ASSERT_EQ(header.jfif, reference.saw_JFIF_marker != 0);
    ASSERT_EQ(header.adobe_transform,
              reference.saw_Adobe_marker ? reference.Adobe_transform : -1);
    unsigned expected = 0;
    for (const auto& profile : profiles) {
      if (reference.jpeg_color_space != profile[1]) {
        continue;
      }
      bool matches = true;
      for (int component = 0; component < reference.num_components;
           ++component) {
        matches &= reference.comp_info[component].h_samp_factor *
                       profile[2 + component * 2] ==
                   reference.max_h_samp_factor;
        matches &= reference.comp_info[component].v_samp_factor *
                       profile[3 + component * 2] ==
                   reference.max_v_samp_factor;
      }
      if (matches) {
        expected = profile[0];
      }
    }
    // The bounded device path rejects unknown Adobe transforms instead of
    // adopting the public decoder's warning-and-assume-YCbCr policy.
    if (header.component_count == 3 && !header.jfif &&
        header.adobe_transform > 1) {
      expected = 0;
    }
    uint32_t kind = 0;
    iree_status_t status = jpeg_device_color_kind(&header, &kind);
    EXPECT_EQ(kind, expected);
    if (test.fixture_color_kind) {
      EXPECT_EQ(kind, test.fixture_color_kind);
    }
    EXPECT_EQ(iree_status_code(status),
              expected ? IREE_STATUS_OK : IREE_STATUS_UNIMPLEMENTED);
    iree_status_ignore(status);
  }
}
}  // namespace
