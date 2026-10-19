#include "native_metadata.h"

#include <string.h>

static bool jpeg_equal_codebooks(const jpeg_header_huffman_t* left,
                                 const jpeg_header_huffman_t* right) {
  return left->symbol_count == right->symbol_count &&
         memcmp(left->counts, right->counts, 16) == 0 &&
         memcmp(left->symbols, right->symbols, left->symbol_count) == 0;
}

void jpeg_device_metadata_initialize(const jpeg_header_t* header,
                                     jpeg_device_metadata_t* metadata) {
  memset(metadata, 0, sizeof(*metadata));
  unsigned selections[3][2];
  uint32_t plane_offsets[3];
  for (unsigned component = 0; component < header->component_count;
       ++component) {
    const jpeg_header_component_t* source = &header->components[component];
    plane_offsets[component] = metadata->pixel_words;
    // Three components with sixteen-bit frame dimensions fit in u32 word
    // offsets. Byte capacities remain the packet owner's checked size domain.
    metadata->plane_words[component] =
        source->block_columns * source->block_rows * 16;
    metadata->pixel_words += metadata->plane_words[component];
    memcpy(metadata->quantization + component * 64,
           header->quantization[source->quantization], 64 * sizeof(uint32_t));
    for (unsigned kind = 0; kind < 2; ++kind) {
      unsigned destination = kind ? source->ac : source->dc;
      const jpeg_header_huffman_t* table = &header->huffman[kind][destination];
      unsigned selected = 0;
      for (; selected < metadata->table_count; ++selected) {
        unsigned key = metadata->table_keys[selected];
        if (jpeg_equal_codebooks(table, &header->huffman[key / 4][key % 4])) {
          break;
        }
      }
      if (selected == metadata->table_count) {
        metadata->table_keys[metadata->table_count++] = kind * 4 + destination;
      }
      selections[component][kind] = selected;
    }
  }
  metadata->block_count =
      header->mcu_columns * header->mcu_rows * header->blocks_per_mcu;
  metadata->layout[3] = header->blocks_per_mcu;
  for (unsigned block = 0; block < header->blocks_per_mcu; ++block) {
    const jpeg_header_block_t* descriptor = &header->blocks[block];
    unsigned component = descriptor->component;
    const jpeg_header_component_t* source = &header->components[component];
    unsigned packed = component | (selections[component][0] << 2) |
                      (selections[component][1] << 5);
    metadata->layout[block / 4] |= packed << ((block % 4) * 8);
    uint32_t* geometry = metadata->geometry + block * 8;
    geometry[0] = plane_offsets[component];
    geometry[1] = source->block_columns;
    geometry[2] = source->block_rows;
    geometry[3] = header->component_count == 1 ? 1 : source->horizontal;
    geometry[4] = header->component_count == 1 ? 1 : source->vertical;
    geometry[5] = descriptor->column;
    geometry[6] = descriptor->row;
    geometry[7] = 0;
  }
}

static void jpeg_write_codebook(const jpeg_header_huffman_t* table,
                                uint32_t* words) {
  memset(words, 0, 2048 * sizeof(uint32_t));
  uint8_t secondary_width[512] = {0};
  unsigned code = 0;
  for (unsigned length = 1; length <= 16; ++length) {
    unsigned count = table->counts[length - 1];
    if (length > 9) {
      for (unsigned entry = 0; entry < count; ++entry) {
        secondary_width[(code + entry) >> (length - 9)] = length - 9;
      }
    }
    code = (code + count) << 1;
  }
  unsigned used = 512;
  for (unsigned prefix = 0; prefix < 512; ++prefix) {
    unsigned width = secondary_width[prefix];
    if (width) {
      words[prefix] = 0x80000000u | ((23 - width) << 24) |
                      (((1u << width) - 1) << 16) | used;
      used += 1u << width;
    }
  }
  // Canonical ordering permits one partial nine-bit prefix at each long
  // length; other suffix entries correspond to symbols. The complete extent
  // is at most 512 + 256 + 7*127, within the fixed device representation.
  IREE_ASSERT(used <= 2048);
  code = 0;
  unsigned ordinal = 0;
  for (unsigned length = 1; length <= 16; ++length) {
    for (unsigned entry = 0; entry < table->counts[length - 1];
         ++entry, ++code) {
      unsigned value = (length << 8) | table->symbols[ordinal++];
      if (length <= 9) {
        unsigned begin = code << (9 - length);
        unsigned end = (code + 1) << (9 - length);
        for (unsigned index = begin; index < end; ++index) {
          words[index] = value;
        }
      } else {
        unsigned prefix = code >> (length - 9);
        unsigned shift = secondary_width[prefix] - (length - 9);
        unsigned suffix = code & ((1u << (length - 9)) - 1);
        unsigned offset = words[prefix] & 2047;
        unsigned begin = suffix << shift;
        unsigned end = (suffix + 1) << shift;
        for (unsigned index = begin; index < end; ++index) {
          words[offset + index] = value;
        }
      }
    }
    code <<= 1;
  }
}

void jpeg_device_metadata_write_tables(const jpeg_header_t* header,
                                       const jpeg_device_metadata_t* metadata,
                                       uint32_t* tables) {
  for (unsigned index = 0; index < metadata->table_count; ++index) {
    unsigned key = metadata->table_keys[index];
    jpeg_write_codebook(&header->huffman[key / 4][key % 4],
                        tables + index * 2048);
  }
}
