// Independent image oracle: public coefficients and a direct cosine matrix.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "float_factor.h"
#include "jpeglib.h"

namespace {
[[noreturn]] void Fail(const char* message) {
  std::fprintf(stderr, "%s\n", message);
  std::exit(1);
}
void Fatal(j_common_ptr codec) {
  codec->err->output_message(codec);
  Fail("JPEG reference failed");
}
void Message(j_common_ptr codec, int level) {
  if (level < 0) {
    Fatal(codec);
  }
}
using Bytes = std::vector<uint8_t>;
Bytes Read(const std::string& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream || stream.tellg() <= 0 || stream.tellg() > 64000000) {
    Fail("invalid JPEG extent");
  }
  Bytes data(static_cast<size_t>(stream.tellg()));
  stream.seekg(0);
  stream.read(reinterpret_cast<char*>(data.data()), data.size());
  if (!stream) {
    Fail("short JPEG read");
  }
  return data;
}
void Write(const std::filesystem::path& path, const Bytes& data) {
  std::ofstream stream(path, std::ios::binary);
  stream.write(reinterpret_cast<const char*>(data.data()), data.size());
  if (!stream) {
    Fail("short reference write");
  }
}
struct Plane {
  // Visible sample extent.
  int width = 0;
  // Visible sample height.
  int height = 0;
  // Number of full blocks in a row.
  int blocks = 0;
  // Integer horizontal sampling ratio relative to the output.
  int horizontal = 0;
  // Integer vertical sampling ratio relative to the output.
  int vertical = 0;
  // Complete block-major samples including edge-block padding.
  Bytes bytes;

  int At(int x, int y) const {
    x = std::clamp(x, 0, width - 1);
    y = std::clamp(y, 0, height - 1);
    return bytes[((y / 8) * blocks + x / 8) * 64 + (y % 8) * 8 + x % 8];
  }
  int Interpolate(int x, int y) const {
    int column = x / horizontal, row = y / vertical;
    int next_column = column + ((x & 1) ? 1 : -1);
    int next_row = row + ((y & 1) ? 1 : -1);
    int center = At(column, row);
    if (horizontal == 2 && vertical == 2 && width > 2) {
      return (9 * center + 3 * At(next_column, row) + 3 * At(column, next_row) +
              At(next_column, next_row) + ((x & 1) ? 7 : 8)) >>
             4;
    }
    if (horizontal == 2 && vertical == 1 && width > 2) {
      return (3 * center + At(next_column, row) + ((x & 1) ? 2 : 1)) >> 2;
    }
    if (horizontal == 1 && vertical == 2) {
      return (3 * center + At(column, next_row) + ((y & 1) ? 2 : 1)) >> 2;
    }
    return center;
  }
};
Bytes Color(const std::vector<Plane>& planes, int width, int height,
            J_COLOR_SPACE color) {
  Bytes rgb(size_t(width) * height * 3);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int r = planes[0].Interpolate(x, y), g = r, b = r;
      if (color == JCS_RGB) {
        g = planes[1].Interpolate(x, y);
        b = planes[2].Interpolate(x, y);
      } else if (color == JCS_YCbCr) {
        int cb = planes[1].Interpolate(x, y) - 128,
            cr = planes[2].Interpolate(x, y) - 128;
        r += (91881 * cr + 32768) >> 16;
        g += (-22554 * cb - 46802 * cr + 32768) >> 16;
        b += (116130 * cb + 32768) >> 16;
      } else if (color != JCS_GRAYSCALE) {
        Fail("unsupported reference RGB colorspace");
      }
      size_t index = (size_t(y) * width + x) * 3;
      rgb[index] = std::clamp(r, 0, 255);
      rgb[index + 1] = std::clamp(g, 0, 255);
      rgb[index + 2] = std::clamp(b, 0, 255);
    }
  }
  return rgb;
}
void Compare(const char* name, const Bytes& actual, const Bytes& expected) {
  if (actual.size() != expected.size()) {
    Fail("reference size differs");
  }
  uint64_t count = 0, absolute = 0, square = 0;
  unsigned maximum = 0;
  for (size_t index = 0; index < actual.size(); ++index) {
    unsigned difference = std::abs(int(actual[index]) - int(expected[index]));
    count += difference != 0;
    absolute += difference;
    square += difference * difference;
    maximum = std::max(maximum, difference);
  }
  std::printf(
      "{\"comparison\":\"%s\",\"samples\":%zu,\"different\":%llu,\"maximum\":%"
      "u,"
      "\"absolute_sum\":%llu,\"square_sum\":%llu}\n",
      name, actual.size(), (unsigned long long)count, maximum,
      (unsigned long long)absolute, (unsigned long long)square);
}
Bytes ReadFirstImage(const char* path, size_t length) {
  std::ifstream stream(path, std::ios::binary);
  std::array<uint8_t, 10> header{};
  stream.read(reinterpret_cast<char*>(header.data()), header.size());
  if (!stream || header[0] != 0x93 || header[6] != 1 || header[7] != 0) {
    Fail("invalid NPY header");
  }
  stream.seekg(10 + header[8] + size_t(header[9]) * 256);
  Bytes data(length);
  stream.read(reinterpret_cast<char*>(data.data()), data.size());
  if (!stream) {
    Fail("short baseline image read");
  }
  return data;
}

Bytes PublicRgb(const Bytes& encoded, J_DCT_METHOD method) {
  jpeg_decompress_struct codec{};
  jpeg_error_mgr error;
  jpeg_std_error(&error);
  error.error_exit = Fatal;
  error.emit_message = Message;
  codec.err = &error;
  jpeg_create_decompress(&codec);
  jpeg_mem_src(&codec, encoded.data(), encoded.size());
  if (jpeg_read_header(&codec, TRUE) != JPEG_HEADER_OK) {
    Fail("missing RGB header");
  }
  codec.dct_method = method;
  codec.out_color_space = JCS_RGB;
  if (!jpeg_start_decompress(&codec)) {
    Fail("short RGB start");
  }
  Bytes rgb(size_t(codec.output_width) * codec.output_height * 3);
  while (codec.output_scanline < codec.output_height) {
    JSAMPROW row =
        rgb.data() + size_t(codec.output_scanline) * codec.output_width * 3;
    if (jpeg_read_scanlines(&codec, &row, 1) != 1) {
      Fail("short RGB scanline");
    }
  }
  if (!jpeg_finish_decompress(&codec)) {
    Fail("short RGB finish");
  }
  jpeg_destroy_decompress(&codec);
  return rgb;
}

void Observe(const Bytes& encoded, const std::filesystem::path& directory,
             const char* baseline_pixels, const char* baseline_rgb) {
  jpeg_decompress_struct codec{};
  jpeg_error_mgr error;
  jpeg_std_error(&error);
  error.error_exit = Fatal;
  error.emit_message = Message;
  codec.err = &error;
  jpeg_create_decompress(&codec);
  jpeg_mem_src(&codec, encoded.data(), encoded.size());
  if (jpeg_read_header(&codec, TRUE) != JPEG_HEADER_OK) {
    Fail("missing header");
  }
  if (codec.data_precision != 8 || codec.num_components > 3 ||
      uint64_t(codec.image_width) * codec.image_height > 16000000) {
    Fail("unsupported reference geometry");
  }
  auto arrays = jpeg_read_coefficients(&codec);
  long double basis[8][8];
  for (int y = 0; y < 8; ++y) {
    for (int frequency = 0; frequency < 8; ++frequency) {
      basis[y][frequency] =
          frequency ? std::sqrt(2.0L) * std::cos((2 * y + 1) * frequency *
                                                 std::acos(-1.0L) / 16)
                    : 1;
    }
  }
  std::vector<Plane> model, mathematical;
  Bytes pixels, mathematical_pixels;
  long double maximum_unrounded_error = 0;
  for (int component_index = 0; component_index < codec.num_components;
       ++component_index) {
    const auto& component = codec.comp_info[component_index];
    if (codec.max_h_samp_factor % component.h_samp_factor ||
        codec.max_v_samp_factor % component.v_samp_factor) {
      Fail("non-integral sampling ratio");
    }
    Plane plane;
    plane.horizontal = codec.max_h_samp_factor / component.h_samp_factor;
    plane.vertical = codec.max_v_samp_factor / component.v_samp_factor;
    plane.width = (codec.image_width + plane.horizontal - 1) / plane.horizontal;
    plane.height = (codec.image_height + plane.vertical - 1) / plane.vertical;
    plane.blocks = component.width_in_blocks;
    plane.bytes.resize(size_t(component.width_in_blocks) *
                       component.height_in_blocks * 64);
    Plane exact = plane;
    for (JDIMENSION row = 0; row < component.height_in_blocks; ++row) {
      auto blocks =
          codec.mem->access_virt_barray(reinterpret_cast<j_common_ptr>(&codec),
                                        arrays[component_index], row, 1, FALSE);
      for (JDIMENSION column = 0; column < component.width_in_blocks;
           ++column) {
        const auto& block = blocks[0][column];
        int values[64];
        float intermediate[8][8];
        long double reference_columns[8][8];
        bool any_ac = false;
        for (int index = 0; index < 64; ++index) {
          values[index] =
              int(block[index]) * component.quant_table->quantval[index];
          if (index) {
            any_ac |= values[index] != 0;
          }
        }
        for (int x = 0; x < 8; ++x) {
          WideValues input{};
          for (int y = 0; y < 8; ++y) {
            input[y] = float(values[y * 8 + x]);
          }
          auto output = transform_block_axis(input);
          for (int y = 0; y < 8; ++y) {
            intermediate[y][x] = output[y];
            long double sum = 0;
            for (int frequency = 0; frequency < 8; ++frequency) {
              sum += values[frequency * 8 + x] * basis[y][frequency];
            }
            reference_columns[y][x] = sum;
          }
        }
        for (int y = 0; y < 8; ++y) {
          WideValues input{};
          std::copy_n(intermediate[y], 8, input.begin());
          auto output = transform_block_axis(input);
          for (int x = 0; x < 8; ++x) {
            long double sum = 0;
            for (int frequency = 0; frequency < 8; ++frequency) {
              sum += reference_columns[y][frequency] * basis[x][frequency];
            }
            sum /= 8;
            maximum_unrounded_error = std::max(
                maximum_unrounded_error, std::abs(sum - output[x] / 8.0L));
            int value =
                any_ac ? int(std::floor(std::fma(output[x], 0.125f, 128.5f)))
                       : ((values[0] + 4) >> 3) + 128;
            size_t index =
                (size_t(row) * plane.blocks + column) * 64 + y * 8 + x;
            plane.bytes[index] = std::clamp(value, 0, 255);
            exact.bytes[index] =
                std::clamp(int(std::floor(sum + 128.5L)), 0, 255);
          }
        }
      }
    }
    pixels.insert(pixels.end(), plane.bytes.begin(), plane.bytes.end());
    mathematical_pixels.insert(mathematical_pixels.end(), exact.bytes.begin(),
                               exact.bytes.end());
    model.push_back(std::move(plane));
    mathematical.push_back(std::move(exact));
  }
  Bytes rgb = Color(model, codec.image_width, codec.image_height,
                    codec.jpeg_color_space);
  Bytes mathematical_rgb = Color(mathematical, codec.image_width,
                                 codec.image_height, codec.jpeg_color_space);
  Compare("float32_vs_direct_cosine_components", pixels, mathematical_pixels);
  Compare("float32_vs_direct_cosine_rgb", rgb, mathematical_rgb);
  const J_DCT_METHOD methods[] = {JDCT_ISLOW, JDCT_FLOAT, JDCT_IFAST};
  const char* names[] = {"islow", "float", "ifast"};
  for (size_t index = 0; index < 3; ++index) {
    const auto public_rgb = PublicRgb(encoded, methods[index]);
    const std::string comparison =
        std::string("libjpeg_") + names[index] + "_vs_direct_cosine_rgb";
    Compare(comparison.c_str(), public_rgb, mathematical_rgb);
    Write(directory / (std::string(names[index]) + ".rgb"), public_rgb);
    if (methods[index] == JDCT_ISLOW && baseline_rgb) {
      Compare("libjpeg_islow_vs_integer_rgb", public_rgb,
              ReadFirstImage(baseline_rgb, public_rgb.size()));
    }
  }
  Write(directory / "cosine.rgb", mathematical_rgb);
  if (baseline_pixels && baseline_rgb) {
    Compare("float32_vs_integer_components", pixels,
            ReadFirstImage(baseline_pixels, pixels.size()));
    Compare("float32_vs_integer_rgb", rgb,
            ReadFirstImage(baseline_rgb, rgb.size()));
  }
  std::printf(
      "{\"maximum_unrounded_error\":%.17Lg,\"width\":%u,\"height\":%u,"
      "\"components\":%d,\"color_space\":%d}\n",
      maximum_unrounded_error, codec.image_width, codec.image_height,
      codec.num_components, static_cast<int>(codec.jpeg_color_space));
  Write(directory / "pixels.raw", pixels);
  Write(directory / "rgb.raw", rgb);
  if (!jpeg_finish_decompress(&codec)) {
    Fail("short JPEG finish");
  }
  jpeg_destroy_decompress(&codec);
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 3 && argc != 5) {
    Fail("usage: jpeg directory [integer-pixels.npy integer-rgb.npy]");
  }
  std::filesystem::create_directories(argv[2]);
  Observe(Read(argv[1]), argv[2], argc == 5 ? argv[3] : nullptr,
          argc == 5 ? argv[4] : nullptr);
}
