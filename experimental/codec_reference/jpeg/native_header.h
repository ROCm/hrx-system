#ifndef CODEC_EXPERIMENT_NATIVE_HEADER_H_
#define CODEC_EXPERIMENT_NATIVE_HEADER_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// JPEG scan coefficient ordinal to the natural 8x8 coefficient position.
extern const uint8_t jpeg_zigzag[64];

typedef struct jpeg_header_huffman_t {
  // Canonical code counts for lengths one through sixteen.
  uint8_t counts[16];
  // Symbols in canonical code order; only symbol_count entries are live.
  uint8_t symbols[256];
  // Zero denotes an undefined destination.
  uint16_t symbol_count;
} jpeg_header_huffman_t;

typedef struct jpeg_header_component_t {
  // Frame-defined component identity, independent of its ordinal.
  uint8_t identifier;
  // Horizontal sampling factor from SOF.
  uint8_t horizontal;
  // Vertical sampling factor from SOF.
  uint8_t vertical;
  // Destination of the selected quantization table.
  uint8_t quantization;
  // Destination of the selected DC Huffman table.
  uint8_t dc;
  // Destination of the selected AC Huffman table.
  uint8_t ac;
  // Component columns in units of eight samples, including edge padding.
  uint32_t block_columns;
  // Component rows in units of eight samples, including edge padding.
  uint32_t block_rows;
} jpeg_header_component_t;

typedef struct jpeg_header_block_t {
  // Ordinal of the frame component producing this scan block.
  uint8_t component;
  // Block column inside that component's MCU rectangle.
  uint8_t column;
  // Block row inside that component's MCU rectangle.
  uint8_t row;
} jpeg_header_block_t;

typedef struct jpeg_header_t {
  // SOF marker selecting baseline or extended sequential Huffman DCT.
  uint8_t process;
  // Number of frame components; this experiment admits one or three.
  uint8_t component_count;
  // Bit mask of defined quantization destinations.
  uint8_t quantization_mask;
  // Bit mask of quantization destinations encoded with sixteen-bit entries.
  uint8_t wide_quantization_mask;
  // JFIF APP0 identifier with its complete fixed fields; implies YCbCr.
  bool jfif;
  // Last complete Adobe APP14 transform byte, or -1 when none was present.
  int16_t adobe_transform;
  // Encoded sample columns, before display orientation.
  uint32_t width;
  // Encoded sample rows, before display orientation.
  uint32_t height;
  // First byte after the validated first SOS; no subsequent byte is read.
  uint32_t entropy_start;
  // Header-declared restart interval in MCUs; zero disables restart markers.
  uint32_t restart_interval;
  // Number of MCU columns in the scan.
  uint32_t mcu_columns;
  // Number of MCU rows in the scan.
  uint32_t mcu_rows;
  // Live scan-block descriptors in blocks, between one and ten.
  uint32_t blocks_per_mcu;
  // Frame order and selected scan table destinations.
  jpeg_header_component_t components[3];
  // Interleaved scan order, or one descriptor for a grayscale scan.
  jpeg_header_block_t blocks[10];
  // Natural-order quantization entries, indexed by original destination.
  uint32_t quantization[4][64];
  // Canonical Huffman definitions indexed by class (DC/AC), then destination.
  jpeg_header_huffman_t huffman[2][4];
} jpeg_header_t;

// External bytes are checked through the first SOS. Output is consumed only on
// success; input is borrowed only during the call. Successful parsing allocates
// no storage and never accesses entropy. Errors use ordinary owned statuses;
// explicitly unsupported processes return UNIMPLEMENTED.
iree_status_t jpeg_header_parse(iree_const_byte_span_t input,
                                jpeg_header_t* output);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // CODEC_EXPERIMENT_NATIVE_HEADER_H_
