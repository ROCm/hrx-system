#include "native_upload.h"

#include <string.h>

#include "native_color.h"

bool jpeg_upload_is_input(enum JpegBuffer buffer) {
  switch (buffer) {
    case JPEG_TABLES:
    case JPEG_BLOCKS:
    case JPEG_GEOMETRY:
    case JPEG_QUANTIZATION:
    case JPEG_ZIGZAG:
    case JPEG_COMPRESSED:
    case JPEG_HEADERS:
    case JPEG_COLOR_TILES:
    case JPEG_COLOR_SCHEDULE:
      return true;
    default:
      return false;
  }
}

iree_status_t jpeg_upload_image_initialize(iree_const_byte_span_t encoded,
                                           jpeg_upload_image_t* image) {
  if (encoded.data_length > 1024u * 4096u) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "JPEG input exceeds the device tile capacity");
  }
  IREE_RETURN_IF_ERROR(jpeg_header_parse(encoded, &image->header));
  if (image->header.restart_interval) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "JPEG restart segments require a connected device path");
  }
  IREE_RETURN_IF_ERROR(
      jpeg_device_color_kind(&image->header, &image->color_kind));
  jpeg_device_metadata_initialize(&image->header, &image->metadata);
  image->encoded = encoded;
  return iree_ok_status();
}

static uint64_t jpeg_align(uint64_t value, unsigned alignment) {
  return (value + alignment - 1) & ~((uint64_t)alignment - 1);
}

// Only successful image records reach this transform. The frame domain is
// u16 by u16 and count is at most 256, so the u64 sums cannot overflow. Their
// representability in the current device offset domain is checked by create.
static void jpeg_upload_lengths(iree_host_size_t count,
                                const jpeg_upload_image_t* images,
                                uint64_t* lengths) {
  uint64_t tiles = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    const jpeg_upload_image_t* image = &images[i];
    const jpeg_header_t* header = &image->header;
    const jpeg_device_metadata_t* metadata = &image->metadata;
    const uint64_t image_tiles = (image->encoded.data_length + 4095) / 4096;
    tiles += image_tiles;
    lengths[JPEG_COMPRESSED] += image_tiles * 4096 + 16;
    lengths[JPEG_ENTROPY] += image_tiles * 4096 + (32 + 8) * 4;
    lengths[JPEG_TABLES] += metadata->table_count * 8192;
    lengths[JPEG_GEOMETRY] += header->blocks_per_mcu * 32;
    lengths[JPEG_QUANTIZATION] += header->component_count * 256;
    lengths[JPEG_PIXELS] += (uint64_t)metadata->pixel_words * 4 + 32;
    lengths[JPEG_RGB] +=
        jpeg_align((uint64_t)header->width * header->height * 3, 4) + 32;
    lengths[JPEG_COLOR_TILES] += (uint64_t)((header->width + 63) / 64) *
                                 ((header->height + 15) / 16) * 12;
  }
  const uint64_t jobs = tiles * 4;
  lengths[JPEG_BLOCKS] = count * 16;
  lengths[JPEG_EXPECTED] = jobs * 32;
  lengths[JPEG_JOBS] = jobs * 16;
  lengths[JPEG_SEGMENTS] = count * 32;
  lengths[JPEG_BRIDGES] = jobs * 32;
  lengths[JPEG_SUFFIXES] = (jobs + count) * 32;
  lengths[JPEG_RANGES] = (jobs + count) * 32;
  lengths[JPEG_INTERVALS] = (jobs * 5 + count) * 32;
  lengths[JPEG_ZIGZAG] = 256;
  lengths[JPEG_STATUS] = (jobs * 5 + count) * 4;
  lengths[JPEG_POINTS] = jobs * 3 * 32;
  lengths[JPEG_RECORDS] = tiles * 16;
  lengths[JPEG_PREFIXES] = tiles * 4;
  lengths[JPEG_ERRORS] = count * 16;
  lengths[JPEG_HEADERS] = count * 72;
  lengths[JPEG_SCHEDULE] = 32;
  lengths[JPEG_COLOR_SCHEDULE] = 4;
  lengths[JPEG_VALIDITY] = count * 4;
  // Preserve the existing qualified binding guards at this ownership join.
  // Entropy, component, and RGB allocations already carry per-image guards.
  for (unsigned i = 0; i < JPEG_BUFFER_COUNT; ++i) {
    if (i != JPEG_ENTROPY && i != JPEG_PIXELS && i != JPEG_RGB) {
      lengths[i] += 32;
    }
  }
}

static void jpeg_store_words(uint8_t* target, const uint32_t* words,
                             unsigned count) {
  for (unsigned i = 0; i < count; ++i) {
    iree_unaligned_store_le_u32(target + i * 4, words[i]);
  }
}

// Writes only the immutable inputs. Device-owned scratch and terminal state
// have no host allocation, initialization, or body-derived expected values.
static void jpeg_upload_write(iree_host_size_t count,
                              const jpeg_upload_image_t* images,
                              jpeg_upload_t* upload) {
  uint8_t* pointers[JPEG_BUFFER_COUNT];
  uint64_t position[JPEG_BUFFER_COUNT] = {0};
  for (unsigned buffer = 0; buffer < JPEG_BUFFER_COUNT; ++buffer) {
    pointers[buffer] = jpeg_upload_is_input((enum JpegBuffer)buffer)
                           ? upload->input.data + upload->offsets[buffer]
                           : NULL;
  }
  jpeg_store_words(upload->input.data, upload->offsets, JPEG_BUFFER_COUNT);
  uint32_t tiles = 0;
  uint32_t color_tiles = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    const jpeg_upload_image_t* image = &images[i];
    const jpeg_header_t* header = &image->header;
    const jpeg_device_metadata_t* metadata = &image->metadata;
    const uint32_t image_tiles =
        (uint32_t)((image->encoded.data_length + 4095) / 4096);
    const uint32_t fields[18] = {(uint32_t)image->encoded.data_length,
                                 header->entropy_start,
                                 metadata->block_count,
                                 (uint32_t)(position[JPEG_PIXELS] / 4),
                                 (uint32_t)(position[JPEG_COMPRESSED] / 4),
                                 (uint32_t)(position[JPEG_ENTROPY] / 4),
                                 tiles,
                                 image_tiles,
                                 (uint32_t)(position[JPEG_TABLES] / 4),
                                 (uint32_t)(position[JPEG_GEOMETRY] / 4),
                                 (uint32_t)(position[JPEG_QUANTIZATION] / 4),
                                 header->mcu_columns,
                                 header->width,
                                 header->height,
                                 (uint32_t)position[JPEG_RGB],
                                 metadata->plane_words[0] * 4,
                                 metadata->plane_words[1] * 4,
                                 image->color_kind};
    jpeg_store_words(pointers[JPEG_HEADERS] + i * 72, fields, 18);
    jpeg_store_words(pointers[JPEG_BLOCKS] + i * 16, metadata->layout, 4);
    jpeg_store_words(pointers[JPEG_GEOMETRY] + position[JPEG_GEOMETRY],
                     metadata->geometry, header->blocks_per_mcu * 8);
    jpeg_store_words(pointers[JPEG_QUANTIZATION] + position[JPEG_QUANTIZATION],
                     metadata->quantization, header->component_count * 64);
    uint32_t* table_words =
        (uint32_t*)(pointers[JPEG_TABLES] + position[JPEG_TABLES]);
    jpeg_device_metadata_write_tables(header, metadata, table_words);
    // The materializer consumes native words; the upload representation is LE.
    for (unsigned word = 0; word < metadata->table_count * 2048; ++word) {
      iree_unaligned_store_le_u32((uint8_t*)table_words + word * 4,
                                  table_words[word]);
    }
    memcpy(pointers[JPEG_COMPRESSED] + position[JPEG_COMPRESSED],
           image->encoded.data, image->encoded.data_length);
    memset(pointers[JPEG_COMPRESSED] + position[JPEG_COMPRESSED] +
               image->encoded.data_length,
           0, image_tiles * 4096 + 16 - image->encoded.data_length);
    for (unsigned y = 0; y < header->height; y += 16) {
      for (unsigned x = 0; x < header->width; x += 64) {
        const uint32_t tile[3] = {(uint32_t)i, x, y};
        jpeg_store_words(pointers[JPEG_COLOR_TILES] + color_tiles * 12, tile,
                         3);
        ++color_tiles;
      }
    }
    tiles += image_tiles;
    position[JPEG_COMPRESSED] += image_tiles * 4096 + 16;
    position[JPEG_ENTROPY] += image_tiles * 4096 + (32 + 8) * 4;
    position[JPEG_TABLES] += metadata->table_count * 8192;
    position[JPEG_GEOMETRY] += header->blocks_per_mcu * 32;
    position[JPEG_QUANTIZATION] += header->component_count * 256;
    position[JPEG_PIXELS] += (uint64_t)metadata->pixel_words * 4 + 32;
    position[JPEG_RGB] +=
        jpeg_align((uint64_t)header->width * header->height * 3, 4) + 32;
  }
  for (unsigned i = 0; i < 64; ++i) {
    iree_unaligned_store_le_u32(pointers[JPEG_ZIGZAG] + i * 4, jpeg_zigzag[i]);
  }
  iree_unaligned_store_le_u32(pointers[JPEG_COLOR_SCHEDULE], color_tiles);
}

iree_status_t jpeg_upload_create(iree_host_size_t count,
                                 const jpeg_upload_image_t* images,
                                 iree_allocator_t host_allocator,
                                 jpeg_upload_t* output) {
  memset(output, 0, sizeof(*output));
  if (!count || count > 256) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "JPEG cohort requires 1..256 images");
  }
  uint64_t lengths[JPEG_BUFFER_COUNT] = {0};
  jpeg_upload_lengths(count, images, lengths);
  if ((lengths[JPEG_COLOR_TILES] - 32) / 12 > 1048576) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "JPEG cohort exceeds the color launch capacity");
  }
  uint64_t input_length = sizeof(struct JpegUploadDirectory);
  uint64_t output_length = 0;
  uint64_t offsets[JPEG_BUFFER_COUNT];
  for (unsigned buffer = 0; buffer < JPEG_BUFFER_COUNT; ++buffer) {
    uint64_t* length = jpeg_upload_is_input((enum JpegBuffer)buffer)
                           ? &input_length
                           : &output_length;
    offsets[buffer] = jpeg_align(*length, 16);
    *length = offsets[buffer] + lengths[buffer];
  }
  input_length = jpeg_align(input_length, 16);
  output_length = jpeg_align(output_length, 16);
  if (input_length > UINT32_MAX || output_length > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "JPEG cohort exceeds 32-bit device root extents");
  }
  uint8_t* allocation = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
      host_allocator, (iree_host_size_t)input_length, (void**)&allocation));
  // Initialize only alignment and metadata guards. Live metadata is written
  // once below, and compressed-vector padding is separately zeroed.
  uint64_t filled = sizeof(struct JpegUploadDirectory);
  for (unsigned buffer = 0; buffer < JPEG_BUFFER_COUNT; ++buffer) {
    if (!jpeg_upload_is_input((enum JpegBuffer)buffer)) {
      continue;
    }
    for (uint64_t offset = filled; offset < offsets[buffer]; offset += 4) {
      iree_unaligned_store_le_u32(allocation + offset, 17);
    }
    filled = offsets[buffer] + lengths[buffer];
    for (uint64_t offset = filled - 32; offset < filled; offset += 4) {
      iree_unaligned_store_le_u32(allocation + offset, 17);
    }
  }
  for (uint64_t offset = filled; offset < input_length; offset += 4) {
    iree_unaligned_store_le_u32(allocation + offset, 17);
  }
  output->host_allocator = host_allocator;
  output->input =
      iree_make_byte_span(allocation, (iree_host_size_t)input_length);
  output->output_length = (uint32_t)output_length;
  for (unsigned buffer = 0; buffer < JPEG_BUFFER_COUNT; ++buffer) {
    output->offsets[buffer] = (uint32_t)offsets[buffer];
    output->lengths[buffer] = (uint32_t)lengths[buffer];
  }
  jpeg_upload_write(count, images, output);
  return iree_ok_status();
}

void jpeg_upload_deinitialize(jpeg_upload_t* upload) {
  iree_allocator_free(upload->host_allocator, upload->input.data);
  memset(upload, 0, sizeof(*upload));
}
