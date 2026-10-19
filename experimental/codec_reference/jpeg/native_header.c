#include "native_header.h"

#include <string.h>

const uint8_t jpeg_zigzag[64] = {
    0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,  7,  14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63};

static unsigned jpeg_word(const uint8_t* source) {
  return ((unsigned)source[0] << 8) | source[1];
}

static uint32_t jpeg_divide_up(uint32_t value, uint32_t divisor) {
  return (value + divisor - 1) / divisor;
}

static iree_status_t jpeg_frame(iree_const_byte_span_t bytes, unsigned marker,
                                jpeg_header_t* header) {
  if (marker != 0xc0 && marker != 0xc1) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "JPEG process %02x is outside sequential DCT",
                            marker);
  }
  if (header->process || bytes.data_length < 6) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "duplicate or truncated JPEG frame");
  }
  if (bytes.data[0] != 8) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "JPEG sample precision %u is outside eight bits",
                            bytes.data[0]);
  }
  unsigned count = bytes.data[5];
  if (!count || bytes.data_length != 6 + count * 3 ||
      !jpeg_word(bytes.data + 3)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "invalid JPEG frame dimensions or component extent");
  }
  if (count != 1 && count != 3) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "JPEG frame has %u components", count);
  }
  if (!jpeg_word(bytes.data + 1)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "JPEG deferred height requires DNL");
  }
  header->process = marker;
  header->width = jpeg_word(bytes.data + 3);
  header->height = jpeg_word(bytes.data + 1);
  header->component_count = count;
  unsigned horizontal = 1, vertical = 1;
  for (unsigned i = 0; i < count; ++i) {
    const uint8_t* source = bytes.data + 6 + i * 3;
    jpeg_header_component_t* component = &header->components[i];
    component->identifier = source[0];
    component->horizontal = source[1] >> 4;
    component->vertical = source[1] & 15;
    component->quantization = source[2];
    if (!component->horizontal || component->horizontal > 4 ||
        !component->vertical || component->vertical > 4 ||
        component->quantization > 3) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "invalid JPEG sampling factor or quantizer selector");
    }
    for (unsigned previous = 0; previous < i; ++previous) {
      if (header->components[previous].identifier == component->identifier) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "duplicate JPEG component identifier");
      }
    }
    horizontal = iree_max(horizontal, component->horizontal);
    vertical = iree_max(vertical, component->vertical);
  }
  for (unsigned i = 0; i < count; ++i) {
    jpeg_header_component_t* component = &header->components[i];
    component->block_columns =
        jpeg_divide_up(header->width * component->horizontal, horizontal * 8);
    component->block_rows =
        jpeg_divide_up(header->height * component->vertical, vertical * 8);
  }
  header->mcu_columns = jpeg_divide_up(header->width, horizontal * 8);
  header->mcu_rows = jpeg_divide_up(header->height, vertical * 8);
  return iree_ok_status();
}

static iree_status_t jpeg_quantization(iree_const_byte_span_t bytes,
                                       jpeg_header_t* header) {
  if (!bytes.data_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "empty JPEG DQT");
  }
  iree_host_size_t position = 0;
  while (position < bytes.data_length) {
    unsigned descriptor = bytes.data[position++];
    unsigned destination = descriptor & 15;
    unsigned precision = descriptor >> 4;
    if (destination > 3 || precision > 1 ||
        bytes.data_length - position < 64 * (precision + 1)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid or truncated JPEG quantizer");
    }
    for (unsigned i = 0; i < 64; ++i) {
      unsigned value =
          precision ? jpeg_word(bytes.data + position) : bytes.data[position];
      if (!value) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "zero JPEG quantization entry");
      }
      header->quantization[destination][jpeg_zigzag[i]] = value;
      position += precision + 1;
    }
    header->quantization_mask |= 1u << destination;
    header->wide_quantization_mask =
        (header->wide_quantization_mask & ~(1u << destination)) |
        (precision << destination);
  }
  return iree_ok_status();
}

static iree_status_t jpeg_huffman(iree_const_byte_span_t bytes,
                                  jpeg_header_t* header) {
  if (!bytes.data_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "empty JPEG DHT");
  }
  iree_host_size_t position = 0;
  while (position < bytes.data_length) {
    if (bytes.data_length - position < 17) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "truncated JPEG Huffman counts");
    }
    unsigned descriptor = bytes.data[position++];
    unsigned kind = descriptor >> 4;
    unsigned destination = descriptor & 15;
    if (kind > 1 || destination > 3) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid JPEG Huffman destination");
    }
    jpeg_header_huffman_t* table = &header->huffman[kind][destination];
    unsigned code = 0, count = 0;
    for (unsigned length = 1; length <= 16; ++length) {
      unsigned entries = bytes.data[position++];
      if (code + entries >= (1u << length)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "overfull or all-ones JPEG Huffman code");
      }
      table->counts[length - 1] = entries;
      count += entries;
      code = (code + entries) << 1;
    }
    if (!count || count > 256 || count > bytes.data_length - position) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid JPEG Huffman symbol extent");
    }
    table->symbol_count = count;
    memcpy(table->symbols, bytes.data + position, count);
    position += count;
  }
  return iree_ok_status();
}

static iree_status_t jpeg_symbols(const jpeg_header_huffman_t* table,
                                  unsigned kind) {
  if (!table->symbol_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "undefined JPEG Huffman destination");
  }
  for (unsigned i = 0; i < table->symbol_count; ++i) {
    unsigned value = table->symbols[i];
    if ((!kind && value > 11) || (kind && value != 0 && value != 0xf0 &&
                                  (!(value & 15) || (value & 15) > 10))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "JPEG Huffman symbol is outside eight-bit DCT");
    }
  }
  return iree_ok_status();
}

static iree_status_t jpeg_scan(iree_const_byte_span_t bytes,
                               jpeg_header_t* header) {
  if (!header->process || bytes.data_length < 4) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "JPEG scan has no frame or complete header");
  }
  unsigned count = bytes.data[0];
  if (!count || count > 4 || bytes.data_length != 4 + count * 2 ||
      bytes.data[1 + count * 2] != 0 || bytes.data[2 + count * 2] != 63 ||
      bytes.data[3 + count * 2] != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid sequential JPEG scan parameters");
  }
  if (count != header->component_count) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "component-separated JPEG scans need separate plans");
  }
  for (unsigned i = 0; i < count; ++i) {
    jpeg_header_component_t* component = &header->components[i];
    unsigned selectors = bytes.data[2 + i * 2];
    unsigned dc = selectors >> 4, ac = selectors & 15;
    unsigned limit = header->process == 0xc0 ? 1 : 3;
    if (bytes.data[1 + i * 2] != component->identifier || dc > limit ||
        ac > limit ||
        !(header->quantization_mask & (1u << component->quantization))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid JPEG scan component or table selection");
    }
    if (header->process == 0xc0 &&
        (header->wide_quantization_mask & (1u << component->quantization))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "baseline JPEG requires eight-bit quantizers");
    }
    IREE_RETURN_IF_ERROR(jpeg_symbols(&header->huffman[0][dc], 0));
    IREE_RETURN_IF_ERROR(jpeg_symbols(&header->huffman[1][ac], 1));
    component->dc = dc;
    component->ac = ac;
    unsigned horizontal = count == 1 ? 1 : component->horizontal;
    unsigned vertical = count == 1 ? 1 : component->vertical;
    if (horizontal * vertical > 10 - header->blocks_per_mcu) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "JPEG interleaved MCU exceeds ten blocks");
    }
    for (unsigned row = 0; row < vertical; ++row) {
      for (unsigned column = 0; column < horizontal; ++column) {
        header->blocks[header->blocks_per_mcu++] =
            (jpeg_header_block_t){i, column, row};
      }
    }
  }
  if (count == 1) {
    header->mcu_columns = header->components[0].block_columns;
    header->mcu_rows = header->components[0].block_rows;
  }
  return iree_ok_status();
}

iree_status_t jpeg_header_parse(iree_const_byte_span_t input,
                                jpeg_header_t* output) {
  memset(output, 0, sizeof(*output));
  output->adobe_transform = -1;
  if (input.data_length > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "JPEG header offsets exceed 32-bit device extent");
  }
  if (input.data_length < 2 || input.data[0] != 0xff || input.data[1] != 0xd8) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "missing JPEG SOI");
  }
  iree_host_size_t position = 2;
  while (position < input.data_length) {
    if (input.data[position] != 0xff) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "missing JPEG marker");
    }
    while (position < input.data_length && input.data[position] == 0xff) {
      ++position;
    }
    if (position == input.data_length) {
      break;
    }
    unsigned marker = input.data[position++];
    if (marker == 0 || marker == 1 || (marker >= 0xd0 && marker <= 0xd9)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unexpected standalone JPEG marker");
    }
    if (input.data_length - position < 2) {
      break;
    }
    unsigned length = jpeg_word(input.data + position);
    if (length < 2 || length > input.data_length - position) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid JPEG marker extent");
    }
    iree_const_byte_span_t bytes =
        iree_make_const_byte_span(input.data + position + 2, length - 2);
    position += length;
    if ((marker >= 0xc0 && marker <= 0xcf) && marker != 0xc4 &&
        marker != 0xc8 && marker != 0xcc) {
      IREE_RETURN_IF_ERROR(jpeg_frame(bytes, marker, output));
    } else if (marker == 0xdb) {
      IREE_RETURN_IF_ERROR(jpeg_quantization(bytes, output));
    } else if (marker == 0xc4) {
      IREE_RETURN_IF_ERROR(jpeg_huffman(bytes, output));
    } else if (marker == 0xdd) {
      if (bytes.data_length != 2) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "invalid JPEG DRI extent");
      }
      output->restart_interval = jpeg_word(bytes.data);
    } else if (marker == 0xda) {
      IREE_RETURN_IF_ERROR(jpeg_scan(bytes, output));
      output->entropy_start = position;
      return iree_ok_status();
    } else if (marker == 0xe0 && bytes.data_length >= 14 &&
               memcmp(bytes.data, "JFIF\0", 5) == 0) {
      output->jfif = true;
    } else if (marker == 0xee && bytes.data_length >= 12 &&
               memcmp(bytes.data, "Adobe", 5) == 0) {
      output->adobe_transform = bytes.data[11];
    } else if (marker != 0xfe && !(marker >= 0xe0 && marker <= 0xef)) {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "JPEG marker %02x needs a different header plan",
                              marker);
    }
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "truncated JPEG scan header");
}
