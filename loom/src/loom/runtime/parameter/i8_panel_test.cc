// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/parameter/i8_panel.h"

#include <cmath>
#include <limits>
#include <vector>

#include "iree/base/internal/math.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

void Half(std::vector<uint8_t>& bytes, size_t at, float value) {
  uint16_t bits = iree_math_f32_to_f16(value);
  bytes[at] = bits & 255;
  bytes[at + 1] = bits >> 8;
}

TEST(I8PanelTest, DecodeQ8PreservesFp16BoundaryAndSignedCodes) {
  std::vector<uint8_t> raw(34);
  Half(raw, 0, 0.333251953125f);
  for (int i = 0; i < 32; ++i) {
    raw[2 + i] = static_cast<uint8_t>(i * 8 - 128);
  }
  float out[32];
  IREE_ASSERT_OK(loom_runtime_i8_panel_decode(
      8, iree_make_const_byte_span(raw.data(), raw.size()), 1, out));
  for (int i = 0; i < 32; ++i) {
    EXPECT_EQ(out[i],
              iree_math_round_to_nearest_f16(0.333251953125f * (i * 8 - 128)));
  }
}

TEST(I8PanelTest, DecodeKBlocksPreservesSubgroupScaleAndHighBits) {
  for (uint32_t type : {12u, 13u, 14u}) {
    std::vector<uint8_t> raw(type == 12 ? 144 : type == 13 ? 176 : 210, 0);
    if (type == 14) {
      Half(raw, 208, 0.5f);
      for (unsigned i = 0; i < 128; ++i) {
        raw[i] = 0x21;
      }
      for (unsigned i = 128; i < 192; ++i) {
        raw[i] = 0xe4;
      }
      for (unsigned i = 192; i < 208; ++i) {
        raw[i] = (i % 2) ? -3 : 2;
      }
    } else {
      Half(raw, 0, 0.5f);
      Half(raw, 2, 0.25f);
      for (unsigned i = 4; i < 12; ++i) {
        raw[i] = 1;
      }
      for (unsigned i = 12; i < 16; ++i) {
        raw[i] = 0x11;
      }
      unsigned start = type == 12 ? 16 : 48;
      for (unsigned i = start; i < raw.size(); ++i) {
        raw[i] = 0x21;
      }
      if (type == 13) {
        for (unsigned i = 16; i < 48; ++i) {
          raw[i] = 0xaa;
        }
      }
    }
    float out[256];
    IREE_ASSERT_OK(loom_runtime_i8_panel_decode(
        type, iree_make_const_byte_span(raw.data(), raw.size()), 1, out));
    for (unsigned i = 0; i < 256; ++i) {
      float expected;
      if (type == 14) {
        unsigned group = (i % 128) / 32;
        int code = (group < 2 ? 1 : 2) + 16 * group;
        float scale = (i / 16) % 2 ? -1.5f : 1.0f;
        expected = scale * (code - 32);
      } else {
        unsigned group = i / 32;
        int code = (group % 2 ? 2 : 1) + (type == 13 && group % 2 ? 16 : 0);
        expected = code * 0.5f - 0.25f;
      }
      EXPECT_EQ(out[i], expected) << type << " lane " << i;
    }
  }
}

TEST(I8PanelTest, RoundingZeroColumnsAndFragmentPermutation) {
  constexpr uint32_t n = 256, k = 256;
  std::vector<float> values(n * k, 0), scales(n);
  std::vector<int8_t> packed(n * k);
  for (unsigned row = 1; row < n; ++row) {
    for (unsigned c = 0; c < k; ++c) {
      values[row * k + c] =
          static_cast<float>(static_cast<int>((row * 17 + c * 31) % 255) - 127);
    }
  }
  values[k + 0] = 0.5f;
  values[k + 1] = 1.5f;
  values[k + 2] = -0.5f;
  values[k + 3] = -1.5f;
  IREE_ASSERT_OK(loom_runtime_i8_panel_pack(values.data(), n, k, packed.data(),
                                            scales.data()));
  size_t at = 0;
  for (unsigned nr = 0; nr < n; nr += 128) {
    for (unsigned kc = 0; kc < k; kc += 128) {
      for (unsigned nf = 0; nf < 16; ++nf) {
        for (unsigned kf = 0; kf < 16; ++kf) {
          for (unsigned ki = 0; ki < 8; ++ki) {
            for (unsigned ni = 0; ni < 8; ++ni) {
              unsigned row = nr + nf * 8 + ni, col = kc + kf * 8 + ki;
              EXPECT_EQ(
                  packed[at++],
                  static_cast<int>(std::nearbyint(values[row * k + col])));
            }
          }
        }
      }
    }
  }
  for (float scale : scales) {
    EXPECT_EQ(scale, 1.0f);
  }
}

TEST(I8PanelTest, Bf16ScaleRoundsTiesToEven) {
  std::vector<float> values(128 * 128, 0), scales(128);
  std::vector<int8_t> packed(values.size());
  values[0] = 127.0f * (1.0f + 1.0f / 256);
  values[128] = 127.0f * (1.0f + 3.0f / 256);
  IREE_ASSERT_OK(loom_runtime_i8_panel_pack(values.data(), 128, 128,
                                            packed.data(), scales.data()));
  EXPECT_EQ(scales[0], 1.0f);
  EXPECT_EQ(scales[1], 1.0f + 2.0f / 128);
  EXPECT_EQ(packed[0], 127);
}

TEST(I8PanelTest, RejectsUnsupportedBlocksGeometryAndNonfiniteValues) {
  uint8_t raw[210] = {0};
  float out[256];
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_runtime_i8_panel_decode(
          2, iree_make_const_byte_span(raw, sizeof(raw)), 1, out));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_runtime_i8_panel_decode(
                            14, iree_make_const_byte_span(raw, 209), 1, out));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_runtime_i8_panel_pack(nullptr, 128, 129, nullptr, nullptr));
  std::vector<float> values(128 * 128, 0), scales(128);
  std::vector<int8_t> packed(values.size());
  values[0] = std::numeric_limits<float>::infinity();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_runtime_i8_panel_pack(values.data(), 128, 128, packed.data(),
                                 scales.data()));
}

}  // namespace

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <sstream>

#include "iree/testing/temp_file.h"
#include "loom/util/digest.h"

namespace {
std::string Digest(const void* data, size_t size) {
  loom_sha256_t hash;
  loom_sha256_compute(iree_make_const_byte_span(data, size), &hash);
  std::string result;
  for (uint8_t b : hash.bytes) {
    result += "0123456789abcdef"[b >> 4];
    result += "0123456789abcdef"[b & 15];
  }
  return result;
}
std::string Read(const std::string& path) {
  std::ifstream f(path);
  return {std::istreambuf_iterator<char>(f), {}};
}
void Write(const std::string& path, const std::string& text) {
  std::ofstream(path) << text;
}
void Uint(std::string& data, uint64_t value, unsigned width) {
  for (unsigned i = 0; i < width; ++i) {
    data += char(value >> (i * 8));
  }
}
void String(std::string& data, const std::string& text) {
  Uint(data, text.size(), 8);
  data += text;
}
std::string Identity(const std::string& path, const std::string& sha) {
  struct stat st;
  EXPECT_EQ(stat(path.c_str(), &st), 0);
  return "{\"dev\":" + std::to_string(st.st_dev) +
         ",\"ino\":" + std::to_string(st.st_ino) +
         ",\"size\":" + std::to_string(st.st_size) + ",\"mtime_ns\":" +
         std::to_string(int64_t(st.st_mtim.tv_sec) * 1000000000 +
                        st.st_mtim.tv_nsec) +
         ",\"ctime_ns\":" +
         std::to_string(int64_t(st.st_ctim.tv_sec) * 1000000000 +
                        st.st_ctim.tv_nsec) +
         ",\"sha256\":\"" + sha + "\"}";
}

TEST(I8PanelCacheTest, StatIdentityInvalidationAndFullVerification) {
  namespace fs = std::filesystem;
  const std::string base = iree::testing::MakeTempFilePath("i8-panel-cache");
  ASSERT_TRUE(fs::create_directory(base));
  const std::string source = base + "/source.bin", root = base + "/cache";
  ASSERT_TRUE(fs::create_directory(root));
  std::string metadata, qmap = "[";
  Uint(metadata, 0x46554747, 4);
  Uint(metadata, 3, 4);
  Uint(metadata, 128, 8);
  Uint(metadata, 0, 8);
  for (unsigned t = 0; t < 128; ++t) {
    std::string name = "blk." + std::to_string(t / 2) +
                       (t % 2 ? ".ffn_up.weight" : ".ffn_gate.weight");
    String(metadata, name);
    Uint(metadata, 2, 4);
    Uint(metadata, 5120, 8);
    Uint(metadata, 17408, 8);
    Uint(metadata, 12, 4);
    Uint(metadata, 0, 8);
    qmap += (t ? "," : "") + std::string("{\"name\":\"") + name +
            "\",\"shape\":[5120,17408],\"type\":12}";
  }
  qmap += "]";
  metadata.resize((metadata.size() + 31) & ~size_t(31), 0);
  Write(source, metadata);
  const uint64_t source_size =
      metadata.size() + uint64_t(17408) * 5120 / 256 * 144;
  ASSERT_EQ(truncate(source.c_str(), source_size), 0);
  int fd = open(source.c_str(), O_RDONLY);
  ASSERT_GE(fd, 0);
  void* data = mmap(nullptr, source_size, PROT_READ, MAP_PRIVATE, fd, 0);
  ASSERT_NE(data, MAP_FAILED);
  std::string sha = Digest(data, source_size);
  munmap(data, source_size);
  close(fd);
  std::string material =
      "{\"columns\":10496,\"gguf_sha256\":\"" + sha +
      "\",\"k\":5120,\"kernel_version\":\"resident-full-k-i8-i32-bf16-column-"
      "scale-v1\","
      "\"layers\":64,\"layout_version\":\"full-k-colscale-n128-k128-nfrag-"
      "kfrag-k8-n8-v2\","
      "\"quant_type_map\":" +
      qmap + ",\"recipe_version\":\"loom-f16-dequant-bf16-absmax-rne-i8-v1\"}";
  std::string key = Digest(material.data(), material.size());
  std::string entry = root + "/" + key;
  ASSERT_TRUE(fs::create_directory(entry));
  std::string payloads[2] = {entry + "/panels_gu.i8",
                             entry + "/colscale_gu.f32"};
  const uint64_t sizes[2] = {uint64_t(64) * 2 * 10496 * 5120,
                             uint64_t(64) * 2 * 10496 * 4};
  std::string hashes[2];
  for (unsigned i = 0; i < 2; ++i) {
    fd = open(payloads[i].c_str(), O_CREAT | O_RDWR, 0600);
    ASSERT_GE(fd, 0);
    ASSERT_EQ(ftruncate(fd, sizes[i]), 0);
    data = mmap(nullptr, sizes[i], PROT_READ, MAP_PRIVATE, fd, 0);
    ASSERT_NE(data, MAP_FAILED);
    hashes[i] = Digest(data, sizes[i]);
    munmap(data, sizes[i]);
    close(fd);
  }
  auto manifest = [&] {
    Write(entry + "/cache.json",
          "{\"version\":2,\"cache_key\":\"" + key +
              "\",\"panels_gu.i8\":" + Identity(payloads[0], hashes[0]) +
              ",\"colscale_gu.f32\":" + Identity(payloads[1], hashes[1]) +
              ",\"complete\":true}\n");
  };
  manifest();
  std::string path = payloads[0];
  loom_model_file_t file = {};
  file.path = iree_make_cstring_view(path.c_str());
  file.transform.source = iree_make_cstring_view(source.c_str());
  file.transform.cache = iree_make_cstring_view(root.c_str());
  file.transform.output = LOOM_MODEL_TRANSFORM_I8_RESIDENT_HALVES;
  loom_runtime_sources_t sources = {};
  sources.base_directory = iree_make_cstring_view(base.c_str());
  std::string log;
  auto load = [&] {
    loom_runtime_i8_panel_cache_t* cache = nullptr;
    iree_io_parameter_index_t* index = nullptr;
    IREE_CHECK_OK(
        iree_io_parameter_index_create(iree_allocator_system(), &index));
    ::testing::internal::CaptureStderr();
    iree_status_t status = loom_runtime_i8_panel_append(
        &sources, &file, iree_allocator_system(), &cache, index);
    log = ::testing::internal::GetCapturedStderr();
    loom_runtime_i8_panel_cache_free(cache);
    iree_io_parameter_index_release(index);
    return status;
  };
  unsetenv("LOOM_DERIVE_VERIFY");
  IREE_ASSERT_OK(load());
  EXPECT_NE(log.find("source_hashes=1 payload_hashes=0"), std::string::npos);
  IREE_ASSERT_OK(load());
  EXPECT_NE(log.find("source_hashes=0 payload_hashes=0"), std::string::npos);
  std::string identity_path;
  for (const auto& f : fs::directory_iterator(root)) {
    if (f.path().filename().string().find("identity-") == 0 &&
        f.path().extension() == ".json") {
      identity_path = f.path();
    }
  }
  ASSERT_FALSE(identity_path.empty());
  std::string identity = Read(identity_path);
  ASSERT_EQ(utimensat(AT_FDCWD, source.c_str(), nullptr, 0), 0);
  IREE_ASSERT_OK(load());
  EXPECT_NE(log.find("source_hashes=1 payload_hashes=0"), std::string::npos);
  EXPECT_NE(Read(identity_path), identity);
  ASSERT_EQ(rename(source.c_str(), (base + "/old.bin").c_str()), 0);
  Write(source, metadata);
  ASSERT_EQ(truncate(source.c_str(), source_size), 0);
  fd = open(source.c_str(), O_WRONLY);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(pwrite(fd, "x", 1, source_size - 1), 1);
  close(fd);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, load());
  EXPECT_NE(log.find("source_hashes=1"), std::string::npos);
  ASSERT_EQ(unlink(source.c_str()), 0);
  ASSERT_EQ(rename((base + "/old.bin").c_str(), source.c_str()), 0);
  IREE_ASSERT_OK(load());
  std::string record = Read(identity_path);
  auto qpos = record.find("\"qmap_digest\":\"") + 15;
  ASSERT_LT(qpos, record.size());
  record[qpos] = record[qpos] == '0' ? '1' : '0';
  Write(identity_path, record);
  IREE_ASSERT_OK(load());
  EXPECT_NE(log.find("source_hashes=1"), std::string::npos);
  ASSERT_EQ(truncate(payloads[0].c_str(), sizes[0] - 1), 0);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, load());
  ASSERT_EQ(truncate(payloads[0].c_str(), sizes[0]), 0);
  manifest();
  struct stat st;
  ASSERT_EQ(stat(payloads[1].c_str(), &st), 0);
  fd = open(payloads[1].c_str(), O_WRONLY);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(pwrite(fd, "x", 1, 0), 1);
  close(fd);
  struct timespec times[2] = {st.st_atim, st.st_mtim};
  ASSERT_EQ(utimensat(AT_FDCWD, payloads[1].c_str(), times, 0), 0);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, load());
  EXPECT_NE(log.find("source_hashes=0 payload_hashes=1"), std::string::npos);
  fd = open(payloads[1].c_str(), O_WRONLY);
  ASSERT_GE(fd, 0);
  ASSERT_EQ(pwrite(fd, "\0", 1, 0), 1);
  close(fd);
  IREE_ASSERT_OK(load());
  EXPECT_NE(log.find("payload_hashes=1"), std::string::npos);
  IREE_ASSERT_OK(load());
  EXPECT_NE(log.find("payload_hashes=0"), std::string::npos);
  Write(entry + "/cache.json", "{\"cache_key\":\"" + key +
                                   "\",\"panels_gu.i8\":\"" + hashes[0] +
                                   "\",\"colscale_gu.f32\":\"" + hashes[1] +
                                   "\",\"complete\":true}\n");
  IREE_ASSERT_OK(load());
  EXPECT_NE(log.find("source_hashes=0 payload_hashes=2"), std::string::npos);
  EXPECT_NE(Read(entry + "/cache.json").find("\"version\":2"),
            std::string::npos);
  setenv("LOOM_DERIVE_VERIFY", "1", 1);
  IREE_ASSERT_OK(load());
  EXPECT_NE(log.find("source_hashes=1 payload_hashes=2 verify=1"),
            std::string::npos);
  unsetenv("LOOM_DERIVE_VERIFY");
}
}  // namespace
#endif
