#pragma once

// Directory ordinals retain the qualified decoder's buffer contracts. The
// owning root is fixed for each ordinal; offsets are bytes relative to it.
enum JpegBuffer {
  JPEG_ENTROPY,
  JPEG_TABLES,
  JPEG_BLOCKS,
  JPEG_EXPECTED,
  JPEG_JOBS,
  JPEG_SEGMENTS,
  JPEG_BRIDGES,
  JPEG_SUFFIXES,
  JPEG_RANGES,
  JPEG_INTERVALS,
  JPEG_GEOMETRY,
  JPEG_QUANTIZATION,
  JPEG_ZIGZAG,
  JPEG_PIXELS,
  JPEG_STATUS,
  JPEG_POINTS,
  JPEG_RGB,
  JPEG_COMPRESSED,
  JPEG_RECORDS,
  JPEG_PREFIXES,
  JPEG_ERRORS,
  JPEG_HEADERS,
  JPEG_SCHEDULE,
  JPEG_COLOR_TILES,
  JPEG_COLOR_SCHEDULE,
  JPEG_VALIDITY,
  JPEG_BUFFER_COUNT
};

struct JpegUploadDirectory {
  // Byte offsets in the immutable input or disjoint device-output root.
  unsigned offsets[JPEG_BUFFER_COUNT];
};
