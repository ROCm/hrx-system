#ifndef CODEC_EXPERIMENT_NATIVE_METADATA_H_
#define CODEC_EXPERIMENT_NATIVE_METADATA_H_

#include "native_header.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct jpeg_device_metadata_t {
  // Number of distinct selected codebooks in first-use scan order.
  uint32_t table_count;
  // Original class/destination for each codebook: class * 4 + destination.
  uint8_t table_keys[6];
  // Complete scan block count, including interleaved edge blocks.
  uint32_t block_count;
  // Total component-plane capacity in packed four-sample words.
  uint32_t pixel_words;
  // Per-component plane capacities in packed four-sample words.
  uint32_t plane_words[3];
  // Three packed block descriptor words, followed by blocks per MCU.
  uint32_t layout[4];
  // Eight words per live MCU block, matching the reconstruction leaf.
  uint32_t geometry[80];
  // Natural-order quantizers in frame-component order.
  uint32_t quantization[192];
} jpeg_device_metadata_t;

// Consumes successful header parsing. Computes and retains all codebook reuse
// and packed metadata facts once; no input bytes are consulted again.
void jpeg_device_metadata_initialize(const jpeg_header_t* header,
                                     jpeg_device_metadata_t* metadata);

// Writes table_count consecutive 2048-word codebooks directly into caller-owned
// storage. The validated canonical tree bounds every secondary lookup.
void jpeg_device_metadata_write_tables(const jpeg_header_t* header,
                                       const jpeg_device_metadata_t* metadata,
                                       uint32_t* tables);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // CODEC_EXPERIMENT_NATIVE_METADATA_H_
