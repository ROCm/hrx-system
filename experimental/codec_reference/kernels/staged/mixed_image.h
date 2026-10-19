#pragma once

// Header preparation owns these immutable per-image facts. Body-derived counts
// and validity live in the separate device-produced segment and status arrays.
struct Image {
  // Encoded byte extent, including the terminal marker.
  unsigned compressed_bytes;
  // Byte offset immediately after the validated SOS header.
  unsigned entropy_start;
  // Complete MCU block count required by the encoded dimensions and sampling.
  unsigned block_count;
  // First u32 of this image's disjoint component output.
  unsigned component_word_offset;
  // First u32 of the guarded original-byte input allocation.
  unsigned compressed_word_offset;
  // First u32 of the bounded unstuffed allocation, with lookahead padding.
  unsigned entropy_word_offset;
  // First 4096-byte tile record and prefix owned by this image.
  unsigned tile_offset;
  // Number of 4096-byte original-input tiles, excluding lookahead padding.
  unsigned tile_count;
  // First u32 of this image's validated two-level Huffman table pool.
  unsigned table_word_offset;
  // First u32 of its per-MCU-block reconstruction geometry.
  unsigned geometry_word_offset;
  // First u32 of its component quantization tables in natural coefficient
  // order.
  unsigned quantization_word_offset;
  // MCU columns used to map a decoded block ordinal to component coordinates.
  unsigned mcu_columns;
  // Encoded image width in pixels, before orientation or model preprocessing.
  unsigned width;
  // Encoded image height in pixels, before orientation or model preprocessing.
  unsigned height;
  // First byte of this image's disjoint RGB destination.
  unsigned rgb_byte_offset;
  // Allocated bytes in the first block-tiled component plane.
  unsigned first_plane_bytes;
  // Allocated bytes in the second block-tiled component plane, zero for gray.
  unsigned second_plane_bytes;
  // Validated bounded color/sampling case consumed by the color leaf.
  unsigned color_kind;
};

struct BlockLayout {
  // Packed component/DC-table/AC-table descriptors for blocks zero through
  // three.
  unsigned first;
  // Packed descriptors for blocks four through seven.
  unsigned second;
  // Packed descriptors for blocks eight and nine.
  unsigned third;
  // Number of blocks in this scan MCU, between one and ten.
  unsigned count;
};
