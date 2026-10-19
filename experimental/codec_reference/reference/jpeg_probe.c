// Untimed public-libjpeg oracle. No Loom implementation is substituted here.
#include <setjmp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "jpeglib.h"

_Static_assert(sizeof(JCOEF) == sizeof(int16_t), "coefficient storage width");

typedef struct probe_error_t {
  // Public libjpeg error manager, first for callback recovery.
  struct jpeg_error_mgr base;
  // Recovery point whose mutable state is heap-owned.
  jmp_buf recovery;
  // Final fatal diagnostic, if any.
  char fatal[JMSG_LENGTH_MAX];
  // Recorded warning diagnostics; exceeding capacity fails this small probe.
  char warnings[16][JMSG_LENGTH_MAX];
  // Number of recorded warnings.
  size_t warning_count;
} probe_error_t;

typedef struct probe_case_t {
  // Stable numerical fixture name.
  const char* name;
  // Encoded color interpretation requested from the encoder.
  J_COLOR_SPACE color_space;
  // Luma horizontal sampling factor; other components use one.
  int horizontal;
  // Luma vertical sampling factor; other components use one.
  int vertical;
  // Source sample precision, in bits.
  int precision;
  // Encoder quality; one deliberately exercises 16-bit quantization tables.
  int quality;
  // Progressive DCT scan script selection.
  bool progressive;
  // Arithmetic instead of Huffman entropy coding.
  bool arithmetic;
  // Restart interval in scan MCUs, not pixel rows.
  unsigned restart_interval;
  // Predictive lossless process selection.
  bool lossless;
  // Sequential noninterleaved component scans.
  bool separate_scans;
} probe_case_t;

typedef struct encoder_state_t {
  // Libjpeg-owned compression state.
  struct jpeg_compress_struct codec;
  // Error recovery and diagnostics.
  probe_error_t error;
  // Whether destroy_compress owns initialized state.
  bool created;
  // Memory destination returned by libjpeg; caller eventually frees it.
  unsigned char* bytes;
  // Compressed destination length in bytes.
  unsigned long length;
  // Source row for the ordinary 8-bit entry point.
  JSAMPLE row8[64 * 4];
  // Source row for the 12-bit entry point.
  J12SAMPLE row12[64 * 4];
  // Source row for the 16-bit entry point.
  J16SAMPLE row16[64 * 4];
  // Optional sequential component scan script.
  jpeg_scan_info scans[4];
} encoder_state_t;

typedef struct decoder_state_t {
  // Libjpeg-owned decompression state.
  struct jpeg_decompress_struct codec;
  // Error recovery and diagnostics.
  probe_error_t error;
  // Whether destroy_decompress owns initialized state.
  bool created;
  // Whether the public header operation completed.
  bool header_ok;
  // Whether the requested numerical operation completed.
  bool complete;
  // Encoded width retained independently of decoder pool teardown.
  unsigned width;
  // Encoded height retained independently of decoder pool teardown.
  unsigned height;
  // Sample precision from the admitted header.
  int precision;
  // Encoded color interpretation inferred by the reference library.
  int color_space;
  // Whether the stream uses progressive DCT scans.
  bool progressive;
  // Whether the stream uses arithmetic entropy coding.
  bool arithmetic;
  // Restart interval in scan MCUs at the point of snapshot.
  unsigned restart_interval;
  // Selected public IDCT method enum.
  int dct_method;
  // Selected chroma interpolation flag.
  bool fancy;
  // Number of encoded components.
  int component_count;
  // Component identifiers from the SOF header.
  int identifiers[4];
  // Component horizontal sampling factors.
  int horizontal[4];
  // Component vertical sampling factors.
  int vertical[4];
  // Real component block widths, populated by coefficient decoding.
  unsigned block_widths[4];
  // Real component block heights, populated by coefficient decoding.
  unsigned block_heights[4];
  // Owned packed RGB8 result, undefined until complete.
  unsigned char* pixels;
  // Exact RGB8 result extent in bytes.
  size_t pixel_count;
  // Owned natural-order quantized coefficients for each real component block.
  int16_t* coefficients[4];
  // Coefficient element count for each component, excluding MCU dummy blocks.
  size_t coefficient_counts[4];
  // Quantization steps in natural order, latched for each component.
  uint16_t quantization[4][DCTSIZE2];
} decoder_state_t;

static void snapshot_metadata(decoder_state_t* state) {
  state->width = state->codec.image_width;
  state->height = state->codec.image_height;
  state->precision = state->codec.data_precision;
  state->color_space = state->codec.jpeg_color_space;
  state->progressive = state->codec.progressive_mode;
  state->arithmetic = state->codec.arith_code;
  state->restart_interval = state->codec.restart_interval;
  state->dct_method = state->codec.dct_method;
  state->fancy = state->codec.do_fancy_upsampling;
  state->component_count = state->codec.num_components;
  for (int i = 0; i < state->component_count; ++i) {
    const jpeg_component_info* component = &state->codec.comp_info[i];
    state->identifiers[i] = component->component_id;
    state->horizontal[i] = component->h_samp_factor;
    state->vertical[i] = component->v_samp_factor;
    state->block_widths[i] = component->width_in_blocks;
    state->block_heights[i] = component->height_in_blocks;
  }
}

static void* allocate(size_t count, size_t size) {
  void* result = calloc(count, size);
  if (!result) {
    perror("probe allocation");
    exit(1);
  }
  return result;
}

static void fatal_error(j_common_ptr common) {
  probe_error_t* error = (probe_error_t*)common->err;
  common->err->format_message(common, error->fatal);
  longjmp(error->recovery, 1);
}

static void emit_message(j_common_ptr common, int level) {
  if (level >= 0) {
    return;
  }
  probe_error_t* error = (probe_error_t*)common->err;
  if (error->warning_count == 16) {
    fputs("probe warning capacity exhausted\n", stderr);
    exit(1);
  }
  common->err->format_message(common, error->warnings[error->warning_count++]);
  ++error->base.num_warnings;
}

static void initialize_error(probe_error_t* error) {
  jpeg_std_error(&error->base);
  error->base.error_exit = fatal_error;
  error->base.emit_message = emit_message;
}

static void json_string(const char* value) {
  putchar('"');
  for (; *value; ++value) {
    unsigned char character = (unsigned char)*value;
    if (character == '"' || character == '\\') {
      putchar('\\');
    }
    if (character < 32) {
      printf("\\u%04x", character);
    } else {
      putchar(character);
    }
  }
  putchar('"');
}

static void json_hex(const unsigned char* bytes, size_t length) {
  putchar('"');
  for (size_t i = 0; i < length; ++i) {
    printf("%02x", bytes[i]);
  }
  putchar('"');
}

static void print_diagnostics(const probe_error_t* error) {
  printf(",\"error_code\":%d,\"error\":",
         error->fatal[0] ? error->base.msg_code : 0);
  json_string(error->fatal);
  fputs(",\"warnings\":[", stdout);
  for (size_t i = 0; i < error->warning_count; ++i) {
    if (i) {
      putchar(',');
    }
    json_string(error->warnings[i]);
  }
  putchar(']');
}

static unsigned sample(int x, int y, int channel, int precision) {
  unsigned value =
      (unsigned)(x * 17 + y * 37 + channel * 71 + ((x ^ y) & 7) * 19) & 255;
  if (precision == 8) {
    return value;
  }
  unsigned shift = (unsigned)precision - 8;
  return (value << shift) | ((unsigned)(x * y + channel) & ((1u << shift) - 1));
}

static bool encode(const probe_case_t* item, unsigned char** out_bytes,
                   unsigned long* out_length) {
  encoder_state_t* state = allocate(1, sizeof(*state));
  initialize_error(&state->error);
  state->codec.err = &state->error.base;
  if (!setjmp(state->error.recovery)) {
    jpeg_create_compress(&state->codec);
    state->created = true;
    jpeg_mem_dest(&state->codec, &state->bytes, &state->length);
    state->codec.image_width = 35;
    state->codec.image_height = 19;
    state->codec.in_color_space =
        item->color_space == JCS_GRAYSCALE ? JCS_GRAYSCALE
        : (item->color_space == JCS_CMYK || item->color_space == JCS_YCCK)
            ? JCS_CMYK
            : JCS_RGB;
    state->codec.input_components =
        state->codec.in_color_space == JCS_GRAYSCALE ? 1
        : state->codec.in_color_space == JCS_CMYK    ? 4
                                                     : 3;
    jpeg_set_defaults(&state->codec);
    jpeg_set_colorspace(&state->codec, item->color_space);
    state->codec.data_precision = item->precision;
    jpeg_set_quality(&state->codec, item->quality, FALSE);
    for (int i = 0; i < state->codec.num_components; ++i) {
      state->codec.comp_info[i].h_samp_factor = i ? 1 : item->horizontal;
      state->codec.comp_info[i].v_samp_factor = i ? 1 : item->vertical;
    }
    state->codec.arith_code = item->arithmetic;
    state->codec.restart_interval = item->restart_interval;
    if (item->lossless) {
      jpeg_enable_lossless(&state->codec, 1, 0);
    }
    if (item->progressive) {
      jpeg_simple_progression(&state->codec);
    }
    if (item->separate_scans) {
      for (int i = 0; i < state->codec.num_components; ++i) {
        state->scans[i].comps_in_scan = 1;
        state->scans[i].component_index[0] = i;
        state->scans[i].Se = 63;
      }
      state->codec.scan_info = state->scans;
      state->codec.num_scans = state->codec.num_components;
    }
    jpeg_start_compress(&state->codec, TRUE);
    while (state->codec.next_scanline < state->codec.image_height) {
      int y = (int)state->codec.next_scanline;
      for (int x = 0; x < 35; ++x) {
        for (int channel = 0; channel < state->codec.input_components;
             ++channel) {
          size_t index = (size_t)x * state->codec.input_components + channel;
          unsigned value = sample(x, y, channel, item->precision);
          state->row8[index] = (JSAMPLE)value;
          state->row12[index] = (J12SAMPLE)value;
          state->row16[index] = (J16SAMPLE)value;
        }
      }
      JDIMENSION count;
      if (item->precision <= 8) {
        JSAMPROW row = state->row8;
        count = jpeg_write_scanlines(&state->codec, &row, 1);
      } else if (item->precision <= 12) {
        J12SAMPROW row = state->row12;
        count = jpeg12_write_scanlines(&state->codec, &row, 1);
      } else {
        J16SAMPROW row = state->row16;
        count = jpeg16_write_scanlines(&state->codec, &row, 1);
      }
      if (count != 1) {
        fputs("non-suspending encoder stopped\n", stderr);
        exit(1);
      }
    }
    jpeg_finish_compress(&state->codec);
  }
  bool success = !state->error.fatal[0];
  fputs("{\"kind\":\"input\",\"name\":", stdout);
  json_string(item->name);
  printf(",\"success\":%s,\"precision\":%d,\"lossless\":%s,\"hex\":",
         success ? "true" : "false", item->precision,
         item->lossless ? "true" : "false");
  json_hex(state->bytes, success ? state->length : 0);
  print_diagnostics(&state->error);
  puts("}");
  if (state->created) {
    jpeg_destroy_compress(&state->codec);
  }
  *out_bytes = state->bytes;
  *out_length = success ? state->length : 0;
  free(state);
  return success;
}

// mode 0 retains coefficients; modes 1/2 use serving RGB8 with fancy on/off.
static void inspect_and_decode(const char* name, const unsigned char* bytes,
                               unsigned long length, int mode) {
  decoder_state_t* state = allocate(1, sizeof(*state));
  initialize_error(&state->error);
  state->codec.err = &state->error.base;
  if (!setjmp(state->error.recovery)) {
    jpeg_create_decompress(&state->codec);
    state->created = true;
    jpeg_mem_src(&state->codec, bytes, length);
    jpeg_read_header(&state->codec, TRUE);
    state->header_ok = true;
    if (state->codec.image_width > 256 || state->codec.image_height > 256 ||
        state->codec.num_components > 4) {
      fputs("input outside bounded probe geometry\n", stderr);
      exit(1);
    }
    snapshot_metadata(state);
    if (mode == 0) {
      jvirt_barray_ptr* arrays = jpeg_read_coefficients(&state->codec);
      for (int i = 0; i < state->codec.num_components; ++i) {
        jpeg_component_info* component = &state->codec.comp_info[i];
        size_t count = (size_t)component->width_in_blocks *
                       component->height_in_blocks * DCTSIZE2;
        state->coefficient_counts[i] = count;
        state->coefficients[i] = allocate(count, sizeof(int16_t));
        memcpy(state->quantization[i], component->quant_table->quantval,
               sizeof(state->quantization[i]));
        for (JDIMENSION row = 0; row < component->height_in_blocks; ++row) {
          JBLOCKARRAY blocks = state->codec.mem->access_virt_barray(
              (j_common_ptr)&state->codec, arrays[i], row, 1, FALSE);
          memcpy(state->coefficients[i] +
                     (size_t)row * component->width_in_blocks * DCTSIZE2,
                 blocks[0],
                 (size_t)component->width_in_blocks * sizeof(JBLOCK));
        }
      }
    } else {
      state->codec.out_color_space = JCS_RGB;
      if (mode == 2) {
        state->codec.do_fancy_upsampling = FALSE;
      }
      snapshot_metadata(state);
      jpeg_start_decompress(&state->codec);
      if (state->codec.output_components != 3) {
        fputs("unexpected RGB component count\n", stderr);
        exit(1);
      }
      size_t row_length = (size_t)state->codec.output_width * 3;
      state->pixel_count = row_length * state->codec.output_height;
      state->pixels = allocate(state->pixel_count, 1);
      while (state->codec.output_scanline < state->codec.output_height) {
        JSAMPROW row =
            state->pixels + (size_t)state->codec.output_scanline * row_length;
        if (jpeg_read_scanlines(&state->codec, &row, 1) != 1) {
          fputs("non-suspending decoder stopped\n", stderr);
          exit(1);
        }
      }
    }
    snapshot_metadata(state);
    if (!jpeg_finish_decompress(&state->codec)) {
      fputs("non-suspending decoder did not finish\n", stderr);
      exit(1);
    }
    state->complete = true;
  }
  fputs("{\"kind\":\"result\",\"name\":", stdout);
  json_string(name);
  printf(",\"mode\":%d,\"header_ok\":%s,\"complete\":%s", mode,
         state->header_ok ? "true" : "false",
         state->complete ? "true" : "false");
  if (state->header_ok) {
    printf(
        ",\"width\":%u,\"height\":%u,\"precision\":%d,\"color_space\":%d,"
        "\"progressive\":%s,\"arithmetic\":%s,\"restart_interval\":%u,"
        "\"dct_method\":%d,\"fancy\":%s,\"components\":[",
        state->width, state->height, state->precision, state->color_space,
        state->progressive ? "true" : "false",
        state->arithmetic ? "true" : "false", state->restart_interval,
        state->dct_method, state->fancy ? "true" : "false");
    for (int i = 0; i < state->component_count; ++i) {
      if (i) {
        putchar(',');
      }
      printf(
          "{\"id\":%d,\"horizontal\":%d,\"vertical\":%d,\"block_width\":%u,"
          "\"block_height\":%u}",
          state->identifiers[i], state->horizontal[i], state->vertical[i],
          state->block_widths[i], state->block_heights[i]);
    }
    putchar(']');
  }
  print_diagnostics(&state->error);
  if (state->complete && mode == 0) {
    fputs(",\"coefficients\":[", stdout);
    for (int i = 0; i < state->component_count; ++i) {
      if (i) {
        putchar(',');
      }
      fputs("{\"quantization\":[", stdout);
      for (size_t j = 0; j < DCTSIZE2; ++j) {
        if (j) {
          putchar(',');
        }
        printf("%u", state->quantization[i][j]);
      }
      fputs("],\"values\":[", stdout);
      for (size_t j = 0; j < state->coefficient_counts[i]; ++j) {
        if (j) {
          putchar(',');
        }
        printf("%d", state->coefficients[i][j]);
      }
      fputs("]}", stdout);
    }
    putchar(']');
  }
  if (state->complete && mode != 0) {
    fputs(",\"rgb_hex\":", stdout);
    json_hex(state->pixels, state->pixel_count);
  }
  puts("}");
  if (state->created) {
    jpeg_destroy_decompress(&state->codec);
  }
  free(state->pixels);
  for (int i = 0; i < 4; ++i) {
    free(state->coefficients[i]);
  }
  free(state);
}

int main(int argc, char** argv) {
  if (argc != 2) {
    fputs("expected existing fixture as hex\n", stderr);
    return 1;
  }
  printf(
      "{\"kind\":\"library\",\"jpeg_api_version\":%d,\"sample_bytes\":%zu}\n",
      JPEG_LIB_VERSION, sizeof(JSAMPLE));
  const probe_case_t cases[] = {
      {"ycbcr_444", JCS_YCbCr, 1, 1, 8, 90, false, false, 0, false, false},
      {"ycbcr_422", JCS_YCbCr, 2, 1, 8, 90, false, false, 0, false, false},
      {"ycbcr_420", JCS_YCbCr, 2, 2, 8, 90, false, false, 0, false, false},
      {"ycbcr_440", JCS_YCbCr, 1, 2, 8, 90, false, false, 0, false, false},
      {"ycbcr_411", JCS_YCbCr, 4, 1, 8, 90, false, false, 0, false, false},
      {"ycbcr_410", JCS_YCbCr, 4, 2, 8, 90, false, false, 0, false, false},
      {"ycbcr_420_restart_1", JCS_YCbCr, 2, 2, 8, 90, false, false, 1, false,
       false},
      {"ycbcr_420_restart_3", JCS_YCbCr, 2, 2, 8, 90, false, false, 3, false,
       false},
      {"ycbcr_420_progressive", JCS_YCbCr, 2, 2, 8, 90, true, false, 0, false,
       false},
      {"ycbcr_420_progressive_restart_1", JCS_YCbCr, 2, 2, 8, 90, true, false,
       1, false, false},
      {"ycbcr_420_separate_scans", JCS_YCbCr, 2, 2, 8, 90, false, false, 0,
       false, true},
      {"ycbcr_420_arithmetic", JCS_YCbCr, 2, 2, 8, 90, false, true, 0, false,
       false},
      {"ycbcr_420_arithmetic_progressive", JCS_YCbCr, 2, 2, 8, 90, true, true,
       0, false, false},
      {"ycbcr_444_extended_quantization", JCS_YCbCr, 1, 1, 8, 1, false, false,
       0, false, false},
      {"rgb_sequential", JCS_RGB, 1, 1, 8, 90, false, false, 0, false, false},
      {"grayscale", JCS_GRAYSCALE, 1, 1, 8, 90, false, false, 0, false, false},
      {"cmyk_sequential", JCS_CMYK, 1, 1, 8, 90, false, false, 0, false, false},
      {"ycck_sequential", JCS_YCCK, 1, 1, 8, 90, false, false, 0, false, false},
      {"rgb_lossless_8", JCS_RGB, 1, 1, 8, 90, false, false, 0, true, false},
      {"gray_lossless_8", JCS_GRAYSCALE, 1, 1, 8, 90, false, false, 0, true,
       false},
      {"gray_dct_12", JCS_GRAYSCALE, 1, 1, 12, 90, false, false, 0, false,
       false},
      {"rgb_lossless_12", JCS_RGB, 1, 1, 12, 90, false, false, 0, true, false},
      {"gray_lossless_16", JCS_GRAYSCALE, 1, 1, 16, 90, false, false, 0, true,
       false},
  };
  unsigned char* retained = NULL;
  unsigned long retained_length = 0;
  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    unsigned char* bytes = NULL;
    unsigned long length = 0;
    if (!encode(&cases[i], &bytes, &length)) {
      free(bytes);
      continue;
    }
    if (!cases[i].lossless) {
      inspect_and_decode(cases[i].name, bytes, length, 0);
    }
    inspect_and_decode(cases[i].name, bytes, length, 1);
    inspect_and_decode(cases[i].name, bytes, length, 2);
    if (!strcmp(cases[i].name, "ycbcr_420")) {
      retained = bytes;
      retained_length = length;
    } else {
      free(bytes);
    }
  }
  if (!retained) {
    return 1;
  }
  inspect_and_decode("missing_eoi", retained, retained_length - 2, 1);
  size_t position = 2;
  while (position + 4 < retained_length && retained[position + 1] != 0xda) {
    position +=
        2 + ((size_t)retained[position + 2] << 8) + retained[position + 3];
  }
  if (position + 4 >= retained_length) {
    return 1;
  }
  size_t entropy_start = position + 2 + ((size_t)retained[position + 2] << 8) +
                         retained[position + 3];
  inspect_and_decode("truncated_entropy", retained, entropy_start + 8, 1);
  free(retained);
  size_t fixture_length = strlen(argv[1]) / 2;
  unsigned char* fixture = allocate(fixture_length, 1);
  for (size_t i = 0; i < fixture_length; ++i) {
    unsigned value;
    if (sscanf(argv[1] + i * 2, "%2x", &value) != 1) {
      return 1;
    }
    fixture[i] = (unsigned char)value;
  }
  inspect_and_decode("existing_exif_fixture", fixture, fixture_length, 0);
  inspect_and_decode("existing_exif_fixture", fixture, fixture_length, 1);
  inspect_and_decode("existing_exif_fixture", fixture, fixture_length, 2);
  free(fixture);
  return 0;
}
