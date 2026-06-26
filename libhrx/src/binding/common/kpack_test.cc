// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "common/kpack.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "common/fat_binary.h"
#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

#if defined(HRX_ENABLE_ZSTD)
#include <zstd.h>
#endif

namespace {

using ::iree::testing::status::StatusIs;

//===----------------------------------------------------------------------===//
// Test fixtures: MessagePack writer, kpack archive builder, minimal ELF
//===----------------------------------------------------------------------===//

// Minimal MessagePack encoder mirroring the subset the runtime reads. Used to
// build HIPK metadata and kpack TOCs in the same on-disk shape the reference
// rocm_kpack tooling emits.
struct Mp {
  std::vector<uint8_t> b;
  void Byte(uint8_t v) { b.push_back(v); }
  void Raw(const void* p, size_t n) {
    const uint8_t* d = static_cast<const uint8_t*>(p);
    b.insert(b.end(), d, d + n);
  }
  void Str(const std::string& s) {
    if (s.size() < 32) {
      Byte(0xa0 | static_cast<uint8_t>(s.size()));
    } else {
      Byte(0xd9);
      Byte(static_cast<uint8_t>(s.size()));
    }
    Raw(s.data(), s.size());
  }
  void UInt(uint64_t v) {
    if (v < 128) {
      Byte(static_cast<uint8_t>(v));
    } else if (v <= 0xff) {
      Byte(0xcc);
      Byte(static_cast<uint8_t>(v));
    } else if (v <= 0xffff) {
      Byte(0xcd);
      Byte(static_cast<uint8_t>(v >> 8));
      Byte(static_cast<uint8_t>(v));
    } else if (v <= 0xffffffffu) {
      Byte(0xce);
      for (int i = 3; i >= 0; --i) Byte(static_cast<uint8_t>(v >> (i * 8)));
    } else {
      Byte(0xcf);
      for (int i = 7; i >= 0; --i) Byte(static_cast<uint8_t>(v >> (i * 8)));
    }
  }
  void Map(uint32_t n) {
    if (n < 16) {
      Byte(0x80 | static_cast<uint8_t>(n));
    } else {
      Byte(0xde);
      Byte(static_cast<uint8_t>(n >> 8));
      Byte(static_cast<uint8_t>(n));
    }
  }
  void Array(uint32_t n) {
    if (n < 16) {
      Byte(0x90 | static_cast<uint8_t>(n));
    } else {
      Byte(0xdc);
      Byte(static_cast<uint8_t>(n >> 8));
      Byte(static_cast<uint8_t>(n));
    }
  }
};

std::vector<uint8_t> MakeMetadata(const std::string& kernel_name,
                                  const std::vector<std::string>& paths) {
  Mp mp;
  mp.Map(2);
  mp.Str("kernel_name");
  mp.Str(kernel_name);
  mp.Str("kpack_search_paths");
  mp.Array(static_cast<uint32_t>(paths.size()));
  for (const auto& p : paths) mp.Str(p);
  return mp.b;
}

void PutU32LE(std::vector<uint8_t>& v, uint32_t x) {
  for (int i = 0; i < 4; ++i) v.push_back(static_cast<uint8_t>(x >> (i * 8)));
}
void PutU64LE(std::vector<uint8_t>& v, uint64_t x) {
  for (int i = 0; i < 8; ++i) v.push_back(static_cast<uint8_t>(x >> (i * 8)));
}

// Builds a .kpack archive (NoOp, or zstd-per-kernel under HRX_ENABLE_ZSTD) in
// the documented binary layout: 16-byte header, padding to 64, blob, msgpack
// TOC. Kernels are assigned ordinals in insertion order.
class KpackBuilder {
 public:
  explicit KpackBuilder(bool zstd) : zstd_(zstd) {}

  KpackBuilder& Add(const std::string& binary_key, const std::string& arch,
                    std::vector<uint8_t> data) {
    kernels_.push_back({binary_key, arch, std::move(data)});
    return *this;
  }

  std::vector<uint8_t> Build() const {
    std::vector<uint8_t> blob;
    std::vector<std::pair<uint64_t, uint64_t>> blob_info;  // {abs offset, size}
    if (!zstd_) {
      for (const auto& k : kernels_) {
        uint64_t offset = 64 + blob.size();
        blob.insert(blob.end(), k.data.begin(), k.data.end());
        blob_info.push_back({offset, k.data.size()});
      }
    } else {
      PutU32LE(blob, static_cast<uint32_t>(kernels_.size()));
      for (const auto& k : kernels_) {
        std::vector<uint8_t> frame = Compress(k.data);
        PutU32LE(blob, static_cast<uint32_t>(frame.size()));
        blob.insert(blob.end(), frame.begin(), frame.end());
      }
    }

    std::set<std::string> arches;
    std::map<std::string, std::map<std::string, size_t>> by_binary;
    for (size_t i = 0; i < kernels_.size(); ++i) {
      arches.insert(kernels_[i].arch);
      by_binary[kernels_[i].binary_key][kernels_[i].arch] = i;
    }

    Mp toc;
    toc.Map(zstd_ ? 8 : 7);
    toc.Str("format_version");
    toc.UInt(1);
    toc.Str("group_name");
    toc.Str("test");
    toc.Str("gfx_arch_family");
    toc.Str("test");
    toc.Str("gfx_arches");
    toc.Array(static_cast<uint32_t>(arches.size()));
    for (const auto& a : arches) toc.Str(a);
    toc.Str("compression_scheme");
    toc.Str(zstd_ ? "zstd-per-kernel" : "none");
    toc.Str("toc");
    toc.Map(static_cast<uint32_t>(by_binary.size()));
    for (const auto& [binary, archmap] : by_binary) {
      toc.Str(binary);
      toc.Map(static_cast<uint32_t>(archmap.size()));
      for (const auto& [arch, ordinal] : archmap) {
        toc.Str(arch);
        toc.Map(3);
        toc.Str("type");
        toc.Str("hsaco");
        toc.Str("ordinal");
        toc.UInt(ordinal);
        toc.Str("original_size");
        toc.UInt(kernels_[ordinal].data.size());
      }
    }
    if (!zstd_) {
      toc.Str("blobs");
      toc.Array(static_cast<uint32_t>(blob_info.size()));
      for (const auto& bi : blob_info) {
        toc.Map(2);
        toc.Str("offset");
        toc.UInt(bi.first);
        toc.Str("size");
        toc.UInt(bi.second);
      }
    } else {
      toc.Str("zstd_offset");
      toc.UInt(64);
      toc.Str("zstd_size");
      toc.UInt(blob.size());
    }

    uint64_t toc_offset = 64 + blob.size();
    std::vector<uint8_t> out;
    out.insert(out.end(), {'K', 'P', 'A', 'K'});
    PutU32LE(out, 1);
    PutU64LE(out, toc_offset);
    out.resize(64, 0);  // pad to blob alignment
    out.insert(out.end(), blob.begin(), blob.end());
    out.insert(out.end(), toc.b.begin(), toc.b.end());
    return out;
  }

 private:
  struct Kernel {
    std::string binary_key;
    std::string arch;
    std::vector<uint8_t> data;
  };

#if defined(HRX_ENABLE_ZSTD)
  static std::vector<uint8_t> Compress(const std::vector<uint8_t>& data) {
    size_t bound = ZSTD_compressBound(data.size());
    std::vector<uint8_t> frame(bound);
    size_t n = ZSTD_compress(frame.data(), bound, data.data(), data.size(), 3);
    frame.resize(n);
    return frame;
  }
#else
  static std::vector<uint8_t> Compress(const std::vector<uint8_t>&) {
    return {};
  }
#endif

  bool zstd_;
  std::vector<Kernel> kernels_;
};

struct Elf64Header {
  uint8_t magic[4];
  uint8_t elf_class;
  uint8_t elf_data;
  uint8_t elf_version;
  uint8_t osabi;
  uint8_t abiversion;
  uint8_t padding[7];
  uint16_t type;
  uint16_t machine;
  uint32_t version;
  uint64_t entry;
  uint64_t phoff;
  uint64_t shoff;
  uint32_t flags;
  uint16_t ehsize;
  uint16_t phentsize;
  uint16_t phnum;
  uint16_t shentsize;
  uint16_t shnum;
  uint16_t shstrndx;
};
static_assert(sizeof(Elf64Header) == 64, "ELF64 header must be 64 bytes");

// Minimal AMDGPU HSACO ELF (matches fat_binary_test). machine 0x041 -> gfx1100.
std::vector<uint8_t> MakeMinimalAmdgpuElf(uint32_t machine = 0x041) {
  Elf64Header header = {};
  header.magic[0] = 0x7f;
  header.magic[1] = 'E';
  header.magic[2] = 'L';
  header.magic[3] = 'F';
  header.elf_class = 2;
  header.elf_data = 1;
  header.elf_version = 1;
  header.osabi = 64;
  header.abiversion = 4;
  header.machine = 224;
  header.version = 1;
  header.shoff = sizeof(Elf64Header);
  header.flags = machine;
  std::vector<uint8_t> elf(sizeof(header), 0);
  memcpy(elf.data(), &header, sizeof(header));
  return elf;
}

std::string WriteTempFile(const std::string& name,
                          const std::vector<uint8_t>& bytes) {
  static int counter = 0;
  std::string path = std::string(::testing::TempDir()) + "/kpack_test_" +
                     std::to_string(counter++) + "_" + name;
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  f.write(reinterpret_cast<const char*>(bytes.data()),
          static_cast<std::streamsize>(bytes.size()));
  f.close();
  return path;
}

std::vector<std::string> ExpandTargets(const char* isa) {
  std::vector<std::string> result;
  iree_hal_streaming_kpack_for_each_compatible_target(
      iree_make_cstring_view(isa),
      [](iree_string_view_t t, void* ud) -> bool {
        static_cast<std::vector<std::string>*>(ud)->emplace_back(t.data,
                                                                 t.size);
        return false;  // collect all
      },
      &result);
  return result;
}

std::string StripPrefix(const char* isa) {
  iree_string_view_t s =
      iree_hal_streaming_kpack_strip_target_prefix(iree_make_cstring_view(isa));
  return std::string(s.data, s.size);
}

void FreeBuffer(void* p) { iree_allocator_free(iree_allocator_system(), p); }

using Vec = std::vector<std::string>;

//===----------------------------------------------------------------------===//
// strip_target_prefix
//===----------------------------------------------------------------------===//

TEST(KpackTargetMatch, StripsAmdgcnPrefix) {
  EXPECT_EQ(StripPrefix("amdgcn-amd-amdhsa--gfx942"), "gfx942");
}
TEST(KpackTargetMatch, StripsAmdgcnPrefixWithFeatures) {
  EXPECT_EQ(StripPrefix("amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-"),
            "gfx942:sramecc+:xnack-");
}
TEST(KpackTargetMatch, StripNoOpForBareArch) {
  EXPECT_EQ(StripPrefix("gfx1201"), "gfx1201");
}

//===----------------------------------------------------------------------===//
// for_each_compatible_target (mirrors the reference isa_target_match suite)
//===----------------------------------------------------------------------===//

TEST(KpackTargetMatch, ConsumerCardBare) {
  EXPECT_EQ(ExpandTargets("gfx1201"), (Vec{"gfx1201"}));
}
TEST(KpackTargetMatch, ConsumerCardWithPrefix) {
  EXPECT_EQ(ExpandTargets("amdgcn-amd-amdhsa--gfx1100"), (Vec{"gfx1100"}));
}
TEST(KpackTargetMatch, TwoFeaturesMostSpecificFirst) {
  EXPECT_EQ(ExpandTargets("gfx942:sramecc+:xnack-"),
            (Vec{"gfx942:sramecc+:xnack-", "gfx942:sramecc+", "gfx942:xnack-",
                 "gfx942"}));
}
TEST(KpackTargetMatch, TwoFeaturesWithPrefix) {
  EXPECT_EQ(ExpandTargets("amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-"),
            (Vec{"gfx942:sramecc+:xnack-", "gfx942:sramecc+", "gfx942:xnack-",
                 "gfx942"}));
}
TEST(KpackTargetMatch, MixedFeatureValues) {
  EXPECT_EQ(ExpandTargets("gfx942:sramecc-:xnack+"),
            (Vec{"gfx942:sramecc-:xnack+", "gfx942:sramecc-", "gfx942:xnack+",
                 "gfx942"}));
}
TEST(KpackTargetMatch, SingleFeature) {
  EXPECT_EQ(ExpandTargets("gfx942:xnack+"), (Vec{"gfx942:xnack+", "gfx942"}));
}
TEST(KpackTargetMatch, GenericIsa) {
  EXPECT_EQ(ExpandTargets("gfx9-4-generic"), (Vec{"gfx9-4-generic"}));
}
TEST(KpackTargetMatch, EmptyString) { EXPECT_EQ(ExpandTargets(""), Vec{}); }

TEST(KpackTargetMatch, EarlyTerminationStopsAtFirstMatch) {
  std::vector<std::string> seen;
  struct Ctx {
    std::vector<std::string>* seen;
  } ctx{&seen};
  bool found = iree_hal_streaming_kpack_for_each_compatible_target(
      iree_make_cstring_view("gfx942:sramecc+:xnack-"),
      [](iree_string_view_t t, void* ud) -> bool {
        auto* c = static_cast<Ctx*>(ud);
        c->seen->emplace_back(t.data, t.size);
        return std::string(t.data, t.size) == "gfx942:xnack-";  // third
      },
      &ctx);
  EXPECT_TRUE(found);
  EXPECT_EQ(seen, (Vec{"gfx942:sramecc+:xnack-", "gfx942:sramecc+",
                       "gfx942:xnack-"}));
}

//===----------------------------------------------------------------------===//
// parse_metadata (mirrors the reference HIPKMetadata suite)
//===----------------------------------------------------------------------===//

TEST(KpackMetadata, ParsesValid) {
  auto bytes = MakeMetadata("lib/libfoo.so",
                            {"../.kpack/foo_@GFXARCH@.kpack", "bar.kpack"});
  iree_hal_streaming_kpack_metadata_t md;
  IREE_ASSERT_OK(iree_hal_streaming_kpack_parse_metadata(
      iree_make_const_byte_span(bytes.data(), bytes.size()), &md));
  EXPECT_EQ(std::string(md.kernel_name.data, md.kernel_name.size),
            "lib/libfoo.so");
  ASSERT_EQ(md.search_path_count, 2u);
  EXPECT_EQ(std::string(md.search_paths[0].data, md.search_paths[0].size),
            "../.kpack/foo_@GFXARCH@.kpack");
  EXPECT_EQ(std::string(md.search_paths[1].data, md.search_paths[1].size),
            "bar.kpack");
}

TEST(KpackMetadata, ParsesLongKernelNameStr8) {
  std::string long_name(60, 'x');  // forces str8 encoding (>31 bytes)
  auto bytes = MakeMetadata(long_name, {"a.kpack"});
  iree_hal_streaming_kpack_metadata_t md;
  IREE_ASSERT_OK(iree_hal_streaming_kpack_parse_metadata(
      iree_make_const_byte_span(bytes.data(), bytes.size()), &md));
  EXPECT_EQ(std::string(md.kernel_name.data, md.kernel_name.size), long_name);
}

TEST(KpackMetadata, MissingKernelName) {
  Mp mp;
  mp.Map(1);
  mp.Str("kpack_search_paths");
  mp.Array(1);
  mp.Str("a.kpack");
  iree_hal_streaming_kpack_metadata_t md;
  EXPECT_THAT(iree::Status(iree_hal_streaming_kpack_parse_metadata(
                  iree_make_const_byte_span(mp.b.data(), mp.b.size()), &md)),
              StatusIs(iree::StatusCode::kInvalidArgument));
}

TEST(KpackMetadata, MissingSearchPaths) {
  Mp mp;
  mp.Map(1);
  mp.Str("kernel_name");
  mp.Str("lib/x.so");
  iree_hal_streaming_kpack_metadata_t md;
  EXPECT_THAT(iree::Status(iree_hal_streaming_kpack_parse_metadata(
                  iree_make_const_byte_span(mp.b.data(), mp.b.size()), &md)),
              StatusIs(iree::StatusCode::kInvalidArgument));
}

TEST(KpackMetadata, EmptySearchPaths) {
  auto bytes = MakeMetadata("lib/x.so", {});
  iree_hal_streaming_kpack_metadata_t md;
  EXPECT_THAT(iree::Status(iree_hal_streaming_kpack_parse_metadata(
                  iree_make_const_byte_span(bytes.data(), bytes.size()), &md)),
              StatusIs(iree::StatusCode::kInvalidArgument));
}

TEST(KpackMetadata, WrongTypeKernelName) {
  Mp mp;
  mp.Map(2);
  mp.Str("kernel_name");
  mp.UInt(42);  // not a string
  mp.Str("kpack_search_paths");
  mp.Array(1);
  mp.Str("a.kpack");
  iree_hal_streaming_kpack_metadata_t md;
  EXPECT_THAT(iree::Status(iree_hal_streaming_kpack_parse_metadata(
                  iree_make_const_byte_span(mp.b.data(), mp.b.size()), &md)),
              StatusIs(iree::StatusCode::kInvalidArgument));
}

TEST(KpackMetadata, NotAMap) {
  Mp mp;
  mp.Array(2);
  mp.Str("kernel_name");
  mp.Str("lib/x.so");
  iree_hal_streaming_kpack_metadata_t md;
  EXPECT_THAT(iree::Status(iree_hal_streaming_kpack_parse_metadata(
                  iree_make_const_byte_span(mp.b.data(), mp.b.size()), &md)),
              StatusIs(iree::StatusCode::kInvalidArgument));
}

//===----------------------------------------------------------------------===//
// expand_gfxarch
//===----------------------------------------------------------------------===//

TEST(KpackPath, ExpandGfxArchReplaces) {
  char out[256];
  bool had = false;
  IREE_ASSERT_OK(iree_hal_streaming_kpack_expand_gfxarch(
      IREE_SV("../.kpack/foo_@GFXARCH@.kpack"), IREE_SV("gfx942"), out,
      sizeof(out), &had));
  EXPECT_TRUE(had);
  EXPECT_STREQ(out, "../.kpack/foo_gfx942.kpack");
}

TEST(KpackPath, ExpandGfxArchNoPlaceholder) {
  char out[256];
  bool had = true;
  IREE_ASSERT_OK(iree_hal_streaming_kpack_expand_gfxarch(
      IREE_SV("plain.kpack"), IREE_SV("gfx942"), out, sizeof(out), &had));
  EXPECT_FALSE(had);
  EXPECT_STREQ(out, "plain.kpack");
}

TEST(KpackPath, ExpandGfxArchBufferTooSmall) {
  char out[8];
  EXPECT_THAT(iree::Status(iree_hal_streaming_kpack_expand_gfxarch(
                  IREE_SV("foo_@GFXARCH@.kpack"), IREE_SV("gfx942"), out,
                  sizeof(out), nullptr)),
              StatusIs(iree::StatusCode::kOutOfRange));
}

//===----------------------------------------------------------------------===//
// resolve_relative_path
//===----------------------------------------------------------------------===//

std::string ResolveRel(const char* base, const char* rel) {
  char out[1024];
  IREE_EXPECT_OK(iree_hal_streaming_kpack_resolve_relative_path(
      iree_make_cstring_view(base), iree_make_cstring_view(rel), out,
      sizeof(out)));
  return std::string(out);
}

TEST(KpackPath, ResolveSiblingArchive) {
  EXPECT_EQ(ResolveRel("/usr/lib/libfoo.so", "../.kpack/foo.kpack"),
            "/usr/.kpack/foo.kpack");
}
TEST(KpackPath, ResolveSameDir) {
  EXPECT_EQ(ResolveRel("/opt/rocm/lib/libhip.so", "foo.kpack"),
            "/opt/rocm/lib/foo.kpack");
}
TEST(KpackPath, ResolveCollapsesDotSegments) {
  EXPECT_EQ(ResolveRel("/a/b/c.so", "./d/../e.kpack"), "/a/b/e.kpack");
}
TEST(KpackPath, ResolveAbsoluteRelativePassthrough) {
  EXPECT_EQ(ResolveRel("/a/b/c.so", "/abs/d.kpack"), "/abs/d.kpack");
}
TEST(KpackPath, ResolveAbsoluteWithDotDot) {
  EXPECT_EQ(ResolveRel("/a/b/c.so", "/abs/x/../d.kpack"), "/abs/d.kpack");
}

//===----------------------------------------------------------------------===//
// discover_binary_path
//===----------------------------------------------------------------------===//

// A symbol in the test binary's mapped image (not heap), so /proc/self/maps
// resolves it to a real file.
static const int kProbeSymbol = 0x5a5a;

TEST(KpackDiscover, ResolvesOwnBinary) {
  char path[1024];
  iree_host_size_t offset = 0;
  iree_status_t status = iree_hal_streaming_kpack_discover_binary_path(
      &kProbeSymbol, path, sizeof(path), &offset);
  // On Linux this should resolve to the running test executable/library.
  IREE_ASSERT_OK(status);
  EXPECT_GT(strlen(path), 0u);
}

TEST(KpackDiscover, NullAddress) {
  char path[64];
  EXPECT_THAT(iree::Status(iree_hal_streaming_kpack_discover_binary_path(
                  nullptr, path, sizeof(path), nullptr)),
              StatusIs(iree::StatusCode::kInvalidArgument));
}
TEST(KpackDiscover, ZeroCapacity) {
  char path[64];
  EXPECT_THAT(iree::Status(iree_hal_streaming_kpack_discover_binary_path(
                  &kProbeSymbol, path, 0, nullptr)),
              StatusIs(iree::StatusCode::kInvalidArgument));
}
TEST(KpackDiscover, BufferTooSmall) {
  char path[1];
  EXPECT_THAT(iree::Status(iree_hal_streaming_kpack_discover_binary_path(
                  &kProbeSymbol, path, sizeof(path), nullptr)),
              StatusIs(iree::StatusCode::kOutOfRange));
}

//===----------------------------------------------------------------------===//
// archive_open + archive_get_kernel (NoOp)
//===----------------------------------------------------------------------===//

TEST(KpackArchive, NoopGetKernel) {
  std::vector<uint8_t> k1 = {'A', 'B', 'C', 'D'};
  std::vector<uint8_t> k2(200, 0x42);
  std::vector<uint8_t> k3 = {'Z'};
  auto bytes = KpackBuilder(/*zstd=*/false)
                   .Add("lib/libtest.so#0", "gfx900", k1)
                   .Add("lib/libtest.so#0", "gfx906", k2)
                   .Add("bin/app#0", "gfx900", k3)
                   .Build();
  iree_hal_streaming_kpack_archive_t archive;
  IREE_ASSERT_OK(iree_hal_streaming_kpack_archive_open(
      iree_make_const_byte_span(bytes.data(), bytes.size()), &archive));

  void* out = nullptr;
  iree_host_size_t out_size = 0;
  IREE_ASSERT_OK(iree_hal_streaming_kpack_archive_get_kernel(
      &archive, IREE_SV("lib/libtest.so#0"), IREE_SV("gfx900"),
      iree_allocator_system(), &out, &out_size));
  ASSERT_EQ(out_size, k1.size());
  EXPECT_EQ(0, memcmp(out, k1.data(), k1.size()));
  FreeBuffer(out);

  IREE_ASSERT_OK(iree_hal_streaming_kpack_archive_get_kernel(
      &archive, IREE_SV("lib/libtest.so#0"), IREE_SV("gfx906"),
      iree_allocator_system(), &out, &out_size));
  ASSERT_EQ(out_size, k2.size());
  EXPECT_EQ(0, memcmp(out, k2.data(), k2.size()));
  FreeBuffer(out);
}

TEST(KpackArchive, NoopKernelNotFound) {
  auto bytes =
      KpackBuilder(false).Add("lib/x.so#0", "gfx900", {1, 2, 3}).Build();
  iree_hal_streaming_kpack_archive_t archive;
  IREE_ASSERT_OK(iree_hal_streaming_kpack_archive_open(
      iree_make_const_byte_span(bytes.data(), bytes.size()), &archive));
  void* out = nullptr;
  iree_host_size_t out_size = 0;
  EXPECT_THAT(iree::Status(iree_hal_streaming_kpack_archive_get_kernel(
                  &archive, IREE_SV("lib/x.so#0"), IREE_SV("gfx1100"),
                  iree_allocator_system(), &out, &out_size)),
              StatusIs(iree::StatusCode::kNotFound));
  EXPECT_THAT(iree::Status(iree_hal_streaming_kpack_archive_get_kernel(
                  &archive, IREE_SV("lib/missing.so#0"), IREE_SV("gfx900"),
                  iree_allocator_system(), &out, &out_size)),
              StatusIs(iree::StatusCode::kNotFound));
}

TEST(KpackArchive, BadMagic) {
  std::vector<uint8_t> bytes(64, 0);
  bytes[0] = 'N';
  iree_hal_streaming_kpack_archive_t archive;
  EXPECT_THAT(
      iree::Status(iree_hal_streaming_kpack_archive_open(
          iree_make_const_byte_span(bytes.data(), bytes.size()), &archive)),
      StatusIs(iree::StatusCode::kInvalidArgument));
}

TEST(KpackArchive, UnsupportedVersion) {
  auto bytes = KpackBuilder(false).Add("a#0", "gfx900", {1}).Build();
  bytes[4] = 2;  // version field
  iree_hal_streaming_kpack_archive_t archive;
  EXPECT_THAT(
      iree::Status(iree_hal_streaming_kpack_archive_open(
          iree_make_const_byte_span(bytes.data(), bytes.size()), &archive)),
      StatusIs(iree::StatusCode::kIncompatible));
}

TEST(KpackArchive, TruncatedHeader) {
  std::vector<uint8_t> bytes = {'K', 'P', 'A', 'K'};
  iree_hal_streaming_kpack_archive_t archive;
  EXPECT_THAT(
      iree::Status(iree_hal_streaming_kpack_archive_open(
          iree_make_const_byte_span(bytes.data(), bytes.size()), &archive)),
      StatusIs(iree::StatusCode::kInvalidArgument));
}

TEST(KpackArchive, TocOffsetOutOfRange) {
  auto bytes = KpackBuilder(false).Add("a#0", "gfx900", {1}).Build();
  // Corrupt the TOC offset (u64 at byte 8) to point past EOF.
  uint64_t huge = bytes.size() + 1000;
  for (int i = 0; i < 8; ++i)
    bytes[8 + i] = static_cast<uint8_t>(huge >> (i * 8));
  iree_hal_streaming_kpack_archive_t archive;
  EXPECT_THAT(
      iree::Status(iree_hal_streaming_kpack_archive_open(
          iree_make_const_byte_span(bytes.data(), bytes.size()), &archive)),
      StatusIs(iree::StatusCode::kInvalidArgument));
}

#if defined(HRX_ENABLE_ZSTD)
TEST(KpackArchive, ZstdGetKernelRoundTrips) {
  std::vector<uint8_t> k1;
  for (int i = 0; i < 1000; ++i) k1.push_back(static_cast<uint8_t>(i & 0x3f));
  std::vector<uint8_t> k2(500, 0x37);
  auto bytes = KpackBuilder(/*zstd=*/true)
                   .Add("lib/libhip.so#0", "gfx1100", k1)
                   .Add("lib/libhip.so#0", "gfx1101", k2)
                   .Build();
  iree_hal_streaming_kpack_archive_t archive;
  IREE_ASSERT_OK(iree_hal_streaming_kpack_archive_open(
      iree_make_const_byte_span(bytes.data(), bytes.size()), &archive));
  void* out = nullptr;
  iree_host_size_t out_size = 0;
  IREE_ASSERT_OK(iree_hal_streaming_kpack_archive_get_kernel(
      &archive, IREE_SV("lib/libhip.so#0"), IREE_SV("gfx1100"),
      iree_allocator_system(), &out, &out_size));
  ASSERT_EQ(out_size, k1.size());
  EXPECT_EQ(0, memcmp(out, k1.data(), k1.size()));
  FreeBuffer(out);

  IREE_ASSERT_OK(iree_hal_streaming_kpack_archive_get_kernel(
      &archive, IREE_SV("lib/libhip.so#0"), IREE_SV("gfx1101"),
      iree_allocator_system(), &out, &out_size));
  ASSERT_EQ(out_size, k2.size());
  EXPECT_EQ(0, memcmp(out, k2.data(), k2.size()));
  FreeBuffer(out);
}
#endif  // HRX_ENABLE_ZSTD

//===----------------------------------------------------------------------===//
// resolve_code_object (end-to-end through the filesystem)
//===----------------------------------------------------------------------===//

iree_status_t Resolve(const std::vector<uint8_t>& metadata, uint32_t co_index,
                      const std::vector<std::string>& targets, void** out,
                      iree_host_size_t* out_size) {
  std::vector<iree_string_view_t> tv;
  for (const auto& t : targets) tv.push_back(iree_make_cstring_view(t.c_str()));
  return iree_hal_streaming_kpack_resolve_code_object(
      metadata.data(), co_index, tv.size(), tv.data(), iree_allocator_system(),
      out, out_size);
}

TEST(KpackResolve, AbsoluteSearchPath) {
  auto elf = MakeMinimalAmdgpuElf();
  auto archive = KpackBuilder(false).Add("lib/x.so#0", "gfx900", elf).Build();
  std::string path = WriteTempFile("abs.kpack", archive);
  auto metadata = MakeMetadata("lib/x.so", {path});

  void* out = nullptr;
  iree_host_size_t out_size = 0;
  IREE_ASSERT_OK(Resolve(metadata, 0, {"gfx900"}, &out, &out_size));
  ASSERT_EQ(out_size, elf.size());
  EXPECT_EQ(0, memcmp(out, elf.data(), elf.size()));
  FreeBuffer(out);
}

TEST(KpackResolve, GfxArchPlaceholderExpansion) {
  auto elf = MakeMinimalAmdgpuElf();
  auto archive = KpackBuilder(false).Add("lib/x.so#0", "gfx900", elf).Build();
  // Write to a fixed dir so we can template the filename.
  std::string dir = std::string(::testing::TempDir());
  std::string concrete = dir + "/kpack_gfxexp_gfx900.kpack";
  {
    std::ofstream f(concrete, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(archive.data()),
            static_cast<std::streamsize>(archive.size()));
  }
  std::string pattern = dir + "/kpack_gfxexp_@GFXARCH@.kpack";
  auto metadata = MakeMetadata("lib/x.so", {pattern});

  void* out = nullptr;
  iree_host_size_t out_size = 0;
  IREE_ASSERT_OK(Resolve(metadata, 0, {"gfx900"}, &out, &out_size));
  EXPECT_EQ(out_size, elf.size());
  FreeBuffer(out);
}

TEST(KpackResolve, MultiTuCoIndex) {
  auto elf0 = MakeMinimalAmdgpuElf();
  std::vector<uint8_t> elf1 = MakeMinimalAmdgpuElf();
  elf1[63] = 0x11;  // make TU#1 distinguishable
  auto archive = KpackBuilder(false)
                     .Add("lib/x.so#0", "gfx900", elf0)
                     .Add("lib/x.so#1", "gfx900", elf1)
                     .Build();
  std::string path = WriteTempFile("multitu.kpack", archive);
  auto metadata = MakeMetadata("lib/x.so", {path});

  void* out = nullptr;
  iree_host_size_t out_size = 0;
  IREE_ASSERT_OK(Resolve(metadata, 1, {"gfx900"}, &out, &out_size));
  ASSERT_EQ(out_size, elf1.size());
  EXPECT_EQ(0, memcmp(out, elf1.data(), elf1.size()));
  FreeBuffer(out);
}

TEST(KpackResolve, ArchitecturePriorityPrefersFirst) {
  auto elf_a = MakeMinimalAmdgpuElf();
  auto elf_b = MakeMinimalAmdgpuElf();
  elf_b[63] = 0x22;
  auto archive = KpackBuilder(false)
                     .Add("lib/x.so#0", "gfx1100", elf_a)
                     .Add("lib/x.so#0", "gfx900", elf_b)
                     .Build();
  std::string path = WriteTempFile("prio.kpack", archive);
  auto metadata = MakeMetadata("lib/x.so", {path});

  // Both present; the higher-priority target (gfx1100) must win.
  void* out = nullptr;
  iree_host_size_t out_size = 0;
  IREE_ASSERT_OK(Resolve(metadata, 0, {"gfx1100", "gfx900"}, &out, &out_size));
  ASSERT_EQ(out_size, elf_a.size());
  EXPECT_EQ(0, memcmp(out, elf_a.data(), elf_a.size()));
  FreeBuffer(out);
}

TEST(KpackResolve, FeatureSubsetMatch) {
  // Archive keyed by bare gfx942 (release build); agent reports full features.
  auto elf = MakeMinimalAmdgpuElf();
  auto archive = KpackBuilder(false).Add("lib/x.so#0", "gfx942", elf).Build();
  std::string path = WriteTempFile("subset.kpack", archive);
  auto metadata = MakeMetadata("lib/x.so", {path});

  void* out = nullptr;
  iree_host_size_t out_size = 0;
  IREE_ASSERT_OK(
      Resolve(metadata, 0, {"gfx942:sramecc+:xnack-"}, &out, &out_size));
  EXPECT_EQ(out_size, elf.size());
  FreeBuffer(out);
}

TEST(KpackResolve, ArchNotFound) {
  auto elf = MakeMinimalAmdgpuElf();
  auto archive = KpackBuilder(false).Add("lib/x.so#0", "gfx900", elf).Build();
  std::string path = WriteTempFile("notfound.kpack", archive);
  auto metadata = MakeMetadata("lib/x.so", {path});
  void* out = nullptr;
  iree_host_size_t out_size = 0;
  EXPECT_THAT(iree::Status(Resolve(metadata, 0, {"gfx1100"}, &out, &out_size)),
              StatusIs(iree::StatusCode::kNotFound));
}

TEST(KpackResolve, EnvPathOverride) {
  auto elf = MakeMinimalAmdgpuElf();
  auto archive = KpackBuilder(false).Add("lib/x.so#0", "gfx900", elf).Build();
  std::string path = WriteTempFile("override.kpack", archive);
  // Metadata points nowhere useful; ROCM_KPACK_PATH overrides it.
  auto metadata = MakeMetadata("lib/x.so", {"/nonexistent/none.kpack"});

  setenv("ROCM_KPACK_PATH", path.c_str(), 1);
  void* out = nullptr;
  iree_host_size_t out_size = 0;
  iree_status_t status = Resolve(metadata, 0, {"gfx900"}, &out, &out_size);
  unsetenv("ROCM_KPACK_PATH");
  IREE_ASSERT_OK(status);
  EXPECT_EQ(out_size, elf.size());
  FreeBuffer(out);
}

TEST(KpackResolve, EnvDisable) {
  auto metadata = MakeMetadata("lib/x.so", {"/whatever.kpack"});
  setenv("ROCM_KPACK_DISABLE", "1", 1);
  void* out = nullptr;
  iree_host_size_t out_size = 0;
  iree_status_t status = Resolve(metadata, 0, {"gfx900"}, &out, &out_size);
  unsetenv("ROCM_KPACK_DISABLE");
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE, status);
}

TEST(KpackResolve, InvalidMetadata) {
  std::vector<uint8_t> garbage = {0x00, 0x01, 0x02, 0x03};
  void* out = nullptr;
  iree_host_size_t out_size = 0;
  EXPECT_THAT(iree::Status(Resolve(garbage, 0, {"gfx900"}, &out, &out_size)),
              StatusIs(iree::StatusCode::kInvalidArgument));
}

//===----------------------------------------------------------------------===//
// Full integration through the public fat-binary API (HIPK wrapper -> ELF)
//===----------------------------------------------------------------------===//

struct HipFatHeader {
  uint32_t magic;
  uint32_t version;
  void* binary;
  void* reserved;
};
static_assert(sizeof(HipFatHeader) == 24, "wrapper must be 24 bytes");

TEST(KpackIntegration, HipkWrapperResolvesToElf) {
  auto elf = MakeMinimalAmdgpuElf(/*machine=*/0x041);  // gfx1100
  auto archive =
      KpackBuilder(false).Add("lib/libhip.so#0", "gfx1100", elf).Build();
  std::string path = WriteTempFile("integration.kpack", archive);
  auto metadata = MakeMetadata("lib/libhip.so", {path});

  HipFatHeader header = {};
  header.magic = 0x4b504948u;  // "HIPK"
  header.version = 1;
  header.binary = metadata.data();
  header.reserved = reinterpret_cast<void*>(static_cast<uintptr_t>(0));

  const iree_hal_streaming_fat_binary_target_t targets[] = {
      {/*.value=*/IREE_SV("gfx1100")},
  };
  iree_hal_streaming_fat_binary_extract_t extract = {};
  IREE_ASSERT_OK(iree_hal_streaming_fat_binary_extract_for_targets(
      iree_make_const_byte_span(&header, sizeof(header)),
      IREE_ARRAYSIZE(targets), targets, iree_allocator_system(), &extract));
  ASSERT_EQ(extract.match_count, 1u);
  EXPECT_STREQ(extract.matches[0].executable_format, "gfx1100");
  EXPECT_EQ(extract.matches[0].data.data_length, elf.size());
  iree_hal_streaming_fat_binary_extract_reset(&extract);
}

TEST(KpackIntegration, HipkWrapperNoMatchReportsNotFound) {
  auto elf = MakeMinimalAmdgpuElf(/*machine=*/0x041);  // gfx1100
  auto archive =
      KpackBuilder(false).Add("lib/libhip.so#0", "gfx1100", elf).Build();
  std::string path = WriteTempFile("integration_nomatch.kpack", archive);
  auto metadata = MakeMetadata("lib/libhip.so", {path});

  HipFatHeader header = {};
  header.magic = 0x4b504948u;
  header.version = 1;
  header.binary = metadata.data();

  const iree_hal_streaming_fat_binary_target_t targets[] = {
      {/*.value=*/IREE_SV("gfx942")},  // not in the archive
  };
  iree_hal_streaming_fat_binary_extract_t extract = {};
  EXPECT_THAT(
      iree::Status(iree_hal_streaming_fat_binary_extract_for_targets(
          iree_make_const_byte_span(&header, sizeof(header)),
          IREE_ARRAYSIZE(targets), targets, iree_allocator_system(), &extract)),
      StatusIs(iree::StatusCode::kNotFound));
  iree_hal_streaming_fat_binary_extract_reset(&extract);
}

}  // namespace
