#ifndef CODEC_EXPERIMENT_NATIVE_UPLOAD_H_
#define CODEC_EXPERIMENT_NATIVE_UPLOAD_H_

#include "device_upload.h"
#include "native_metadata.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct jpeg_upload_image_t {
  // Borrowed original bytes, retained by the caller through upload creation.
  iree_const_byte_span_t encoded;
  // Validated marker facts; no entropy was accessed to produce them.
  jpeg_header_t header;
  // Retained selected tables and reconstruction geometry.
  jpeg_device_metadata_t metadata;
  // Qualified grayscale, RGB, or YCbCr sampling leaf, numbered one to eight.
  uint32_t color_kind;
} jpeg_upload_image_t;

typedef struct jpeg_upload_t {
  // Allocator owning the single immutable host upload.
  iree_allocator_t host_allocator;
  // Owned directory, tables, descriptors, and original compressed bytes.
  iree_byte_span_t input;
  // Required device allocation extent; no host output storage is allocated.
  uint32_t output_length;
  // Checked byte offsets, identical to the directory at input byte zero.
  uint32_t offsets[JPEG_BUFFER_COUNT];
  // Complete guarded byte extent for each qualified binding.
  uint32_t lengths[JPEG_BUFFER_COUNT];
} jpeg_upload_t;

// Per-image external-input boundary. The caller retains successful records and
// rejects failed images independently before selecting a cohort. Successful
// preparation allocates nothing. Output is consumed only after success.
iree_status_t jpeg_upload_image_initialize(iree_const_byte_span_t encoded,
                                           jpeg_upload_image_t* image);

// Packs 1..256 successful image records in stable ordinal order. Owns one exact
// upload allocation; only header-derived inputs are initialized. The external
// sizes are checked before allocation, and errors leave output empty. Original
// spans may be released after success. There is no CPU entropy inspection.
iree_status_t jpeg_upload_create(iree_host_size_t count,
                                 const jpeg_upload_image_t* images,
                                 iree_allocator_t host_allocator,
                                 jpeg_upload_t* output);

// Releases host ownership; zero-initialized or already released uploads work.
void jpeg_upload_deinitialize(jpeg_upload_t* upload);

// Root ownership is fixed by the decoder contract, not by input bytes.
bool jpeg_upload_is_input(enum JpegBuffer buffer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // CODEC_EXPERIMENT_NATIVE_UPLOAD_H_
