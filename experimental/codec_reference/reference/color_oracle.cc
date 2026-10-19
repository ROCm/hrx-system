// Public libjpeg numerical references for the independent color leaf.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "jpeglib.h"

namespace {
[[noreturn]] void Fail(const char* message) {
  std::fprintf(stderr, "%s\n", message);
  std::exit(1);
}
void Fatal(j_common_ptr codec) {
  codec->err->output_message(codec);
  Fail("public color reference failed");
}
void Message(j_common_ptr codec, int level) {
  if (level < 0) {
    Fatal(codec);
  }
}
void Hex(const std::vector<uint8_t>& bytes) {
  for (uint8_t value : bytes) {
    std::printf("%02x", value);
  }
}
std::vector<uint8_t> Read(const char* filename) {
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  if (!file || file.tellg() <= 0 || file.tellg() > 16000000) {
    Fail("invalid input");
  }
  std::vector<uint8_t> result(static_cast<size_t>(file.tellg()));
  file.seekg(0);
  file.read(reinterpret_cast<char*>(result.data()), result.size());
  if (!file) {
    Fail("short read");
  }
  return result;
}
void Write(const std::filesystem::path& path,
           const std::vector<uint8_t>& bytes) {
  std::ofstream file(path, std::ios::binary);
  file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  if (!file) {
    Fail("short write");
  }
}
std::vector<uint8_t> Encode(unsigned width, unsigned height,
                            unsigned horizontal, unsigned vertical,
                            J_COLOR_SPACE color_space) {
  jpeg_compress_struct codec{};
  jpeg_error_mgr error;
  jpeg_std_error(&error);
  error.error_exit = Fatal;
  error.emit_message = Message;
  codec.err = &error;
  jpeg_create_compress(&codec);
  unsigned char* data = nullptr;
  unsigned long size = 0;
  jpeg_mem_dest(&codec, &data, &size);
  codec.image_width = width;
  codec.image_height = height;
  codec.input_components = color_space == JCS_GRAYSCALE ? 1 : 3;
  codec.in_color_space = color_space == JCS_GRAYSCALE ? JCS_GRAYSCALE : JCS_RGB;
  jpeg_set_defaults(&codec);
  jpeg_set_colorspace(&codec, color_space);
  jpeg_set_quality(&codec, 89, TRUE);
  codec.comp_info[0].h_samp_factor = horizontal;
  codec.comp_info[0].v_samp_factor = vertical;
  for (int index = 1; index < codec.num_components; ++index) {
    codec.comp_info[index].h_samp_factor = 1;
    codec.comp_info[index].v_samp_factor = 1;
  }
  jpeg_start_compress(&codec, TRUE);
  std::vector<uint8_t> row(width * codec.input_components);
  while (codec.next_scanline < height) {
    unsigned y = codec.next_scanline;
    for (unsigned x = 0; x < width; ++x) {
      for (int component = 0; component < codec.input_components; ++component) {
        row[x * codec.input_components + component] =
            (x * 73 + y * 151 + x * y * 11 + component * (37 + x * 13)) & 255;
      }
    }
    JSAMPROW pointer = row.data();
    if (jpeg_write_scanlines(&codec, &pointer, 1) != 1) {
      Fail("short encode");
    }
  }
  jpeg_finish_compress(&codec);
  std::vector<uint8_t> result(data, data + size);
  jpeg_destroy_compress(&codec);
  std::free(data);
  return result;
}
void Open(const std::vector<uint8_t>& input, jpeg_decompress_struct* codec,
          jpeg_error_mgr* error) {
  jpeg_std_error(error);
  error->error_exit = Fatal;
  error->emit_message = Message;
  codec->err = error;
  jpeg_create_decompress(codec);
  jpeg_mem_src(codec, input.data(), input.size());
  if (jpeg_read_header(codec, TRUE) != JPEG_HEADER_OK) {
    Fail("missing header");
  }
  if (codec->data_precision != 8 || codec->num_components > 3 ||
      uint64_t(codec->image_width) * codec->image_height > 16000000) {
    Fail("outside bounded RGB oracle");
  }
}
void Observe(const std::string& name, const std::vector<uint8_t>& input,
             const std::filesystem::path& directory) {
  jpeg_decompress_struct codec{};
  jpeg_error_mgr error;
  Open(input, &codec, &error);
  std::printf("{\"name\":\"%s\",\"width\":%u,\"height\":%u,\"color_space\":%d",
              name.c_str(), codec.image_width, codec.image_height,
              codec.jpeg_color_space);
  codec.out_color_space = JCS_RGB;
  if (codec.dct_method != JDCT_ISLOW || !codec.do_fancy_upsampling) {
    Fail("default numerical boundary changed");
  }
  if (!jpeg_start_decompress(&codec)) {
    Fail("short start");
  }
  std::vector<uint8_t> rgb(size_t(codec.output_width) * codec.output_height *
                           3);
  while (codec.output_scanline < codec.output_height) {
    JSAMPROW row =
        rgb.data() + size_t(codec.output_scanline) * codec.output_width * 3;
    if (jpeg_read_scanlines(&codec, &row, 1) != 1) {
      Fail("short RGB read");
    }
  }
  if (!jpeg_finish_decompress(&codec)) {
    Fail("short finish");
  }
  jpeg_destroy_decompress(&codec);
  if (!directory.empty()) {
    Write(directory / (name + ".rgb"), rgb);
    std::printf(",\"rgb_bytes\":%zu}\n", rgb.size());
    return;
  }
  std::fputs(",\"input_hex\":\"", stdout);
  Hex(input);
  std::fputs("\",\"rgb_hex\":\"", stdout);
  Hex(rgb);
  std::fputs("\",\"planes\":[", stdout);
  codec = {};
  Open(input, &codec, &error);
  codec.raw_data_out = TRUE;
  codec.out_color_space = codec.jpeg_color_space;
  codec.do_fancy_upsampling = FALSE;
  if (!jpeg_start_decompress(&codec)) {
    Fail("short raw start");
  }
  std::vector<std::vector<uint8_t>> planes(codec.num_components);
  std::vector<std::vector<uint8_t>> storage(codec.num_components);
  std::vector<std::vector<JSAMPROW>> rows(codec.num_components);
  std::vector<JSAMPARRAY> image(codec.num_components);
  for (int index = 0; index < codec.num_components; ++index) {
    const auto& component = codec.comp_info[index];
    unsigned count = component.v_samp_factor * 8;
    unsigned pitch = component.width_in_blocks * 8;
    planes[index].resize(size_t(component.downsampled_width) *
                         component.downsampled_height);
    storage[index].resize(size_t(count) * pitch);
    rows[index].resize(count);
    for (unsigned row = 0; row < count; ++row) {
      rows[index][row] = storage[index].data() + row * pitch;
    }
    image[index] = rows[index].data();
  }
  size_t mcu_row = 0;
  unsigned height_per_mcu = codec.max_v_samp_factor * 8;
  while (codec.output_scanline < codec.output_height) {
    if (jpeg_read_raw_data(&codec, image.data(), height_per_mcu) !=
        height_per_mcu) {
      Fail("short raw read");
    }
    for (int index = 0; index < codec.num_components; ++index) {
      const auto& component = codec.comp_info[index];
      size_t first_row = mcu_row * rows[index].size();
      for (size_t row = 0; row < rows[index].size() &&
                           first_row + row < component.downsampled_height;
           ++row) {
        std::copy_n(rows[index][row], component.downsampled_width,
                    planes[index].data() +
                        (first_row + row) * component.downsampled_width);
      }
    }
    ++mcu_row;
  }
  for (int index = 0; index < codec.num_components; ++index) {
    const auto& component = codec.comp_info[index];
    std::printf("%s{\"width\":%u,\"height\":%u,\"hex\":\"", index ? "," : "",
                component.downsampled_width, component.downsampled_height);
    Hex(planes[index]);
    std::fputs("\"}", stdout);
  }
  std::puts("]}");
  if (!jpeg_finish_decompress(&codec)) {
    Fail("short raw finish");
  }
  jpeg_destroy_decompress(&codec);
}
}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "small") {
    const unsigned shapes[][2] = {{1, 1},   {2, 3},   {3, 2},   {4, 4},  {5, 7},
                                  {16, 16}, {63, 17}, {64, 16}, {65, 17}};
    const unsigned sampling[][2] = {{1, 1}, {2, 1}, {2, 2}, {1, 2}, {4, 1},
                                    {4, 2}, {3, 1}, {1, 3}, {2, 3}, {3, 2}};
    for (auto color_space : {JCS_GRAYSCALE, JCS_RGB, JCS_YCbCr}) {
      for (const auto& factors : sampling) {
        if (color_space != JCS_YCbCr && (factors[0] != 1 || factors[1] != 1)) {
          continue;
        }
        for (const auto& shape : shapes) {
          std::string name =
              "c" + std::to_string(color_space) + "-h" +
              std::to_string(factors[0]) + "-v" + std::to_string(factors[1]) +
              "-w" + std::to_string(shape[0]) + "-h" + std::to_string(shape[1]);
          Observe(
              name,
              Encode(shape[0], shape[1], factors[0], factors[1], color_space),
              {});
        }
      }
    }
  } else if (argc >= 4 && std::string(argv[1]) == "dump") {
    std::filesystem::create_directories(argv[2]);
    for (int index = 3; index < argc; ++index) {
      Observe(std::filesystem::path(argv[index]).filename().string(),
              Read(argv[index]), argv[2]);
    }
  } else {
    Fail("usage: small | dump output-directory images...");
  }
}
