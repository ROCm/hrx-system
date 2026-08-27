// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "common/kpack.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "common/fat_binary.h"
#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

#if defined(IREE_HAVE_ZSTD)
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

// Builds a .kpack archive (NoOp, or zstd-per-kernel under IREE_HAVE_ZSTD) in
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

  // Omits the optional "compression_scheme" TOC field so the archive exercises
  // the runtime's absent-means-none default. Only meaningful for a NoOp
  // archive: a zstd archive whose scheme goes undeclared opens as NONE and
  // cannot decode.
  KpackBuilder& OmitCompressionScheme() {
    omit_compression_scheme_ = true;
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
    uint32_t toc_entry_count = zstd_ ? 8 : 7;
    if (omit_compression_scheme_) --toc_entry_count;
    toc.Map(toc_entry_count);
    toc.Str("format_version");
    toc.UInt(1);
    toc.Str("group_name");
    toc.Str("test");
    toc.Str("gfx_arch_family");
    toc.Str("test");
    toc.Str("gfx_arches");
    toc.Array(static_cast<uint32_t>(arches.size()));
    for (const auto& a : arches) toc.Str(a);
    if (!omit_compression_scheme_) {
      toc.Str("compression_scheme");
      toc.Str(zstd_ ? "zstd-per-kernel" : "none");
    }
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

#if defined(IREE_HAVE_ZSTD)
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
  // When true, Build() leaves the optional "compression_scheme" field out of
  // the TOC entirely (as opposed to writing "none").
  bool omit_compression_scheme_ = false;
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

void WriteFile(const std::string& path, const std::vector<uint8_t>& bytes) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(file.is_open()) << "cannot open " << path << " for writing";
  file.write(reinterpret_cast<const char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  file.close();  // sets failbit if the flush does not land
  ASSERT_TRUE(file.good()) << "cannot write " << bytes.size() << " bytes to "
                           << path;
}

std::string WriteTempFile(const std::string& name,
                          const std::vector<uint8_t>& bytes) {
  static int counter = 0;
  std::string path = std::string(::testing::TempDir()) + "/kpack_test_" +
                     std::to_string(counter++) + "_" + name;
  WriteFile(path, bytes);
  return path;
}

// Creates a directory, accepting one that an earlier run of the binary left
// behind in a TempDir() that is not per-run.
bool EnsureDirectory(const std::string& path) {
  return mkdir(path.c_str(), 0700) == 0 || errno == EEXIST;
}

std::string JoinPathList(const std::vector<std::string>& paths) {
  std::string joined;
  for (const auto& path : paths) {
    if (!joined.empty()) joined += ':';
    joined += path;
  }
  return joined;
}

std::vector<std::string> ExpandTargets(const char* isa) {
  std::vector<std::string> result;
  IREE_EXPECT_OK(iree_hal_streaming_kpack_for_each_compatible_target(
      iree_make_cstring_view(isa),
      [](iree_string_view_t t, void* ud) -> bool {
        static_cast<std::vector<std::string>*>(ud)->emplace_back(t.data,
                                                                 t.size);
        return false;  // collect all
      },
      &result));
  return result;
}

std::string StripPrefix(const char* isa) {
  iree_string_view_t s =
      iree_hal_streaming_kpack_strip_target_prefix(iree_make_cstring_view(isa));
  return std::string(s.data, s.size);
}

void FreeBuffer(void* p) { iree_allocator_free(iree_allocator_system(), p); }

using Vec = std::vector<std::string>;

// A mapping of |readable_pages| readable pages followed by one PROT_NONE page,
// giving the readable region a hard end: dereferencing guard() faults. Models
// an address at the end of a loaded segment, where the bytes beyond it belong
// to no readable mapping.
//
// The pages are anonymous unless |backing_path| is given, in which case they
// come from a file created there and the guard page is file-backed. That is the
// shape the ELF loader produces between a library's segments: /proc/self/maps
// names the file on the guard entry exactly as on the segments around it, so
// the path is no evidence that the bytes can be read.
class GuardedMapping {
 public:
  explicit GuardedMapping(size_t readable_pages,
                          const std::string& backing_path = "")
      : backing_path_(backing_path) {
    page_size_ = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    total_size_ = (readable_pages + 1) * page_size_;
    int fd = -1;
    if (!backing_path_.empty()) {
      fd = open(backing_path_.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
      if (fd < 0) return;
      if (ftruncate(fd, static_cast<off_t>(total_size_)) != 0) {
        close(fd);
        return;
      }
    }
    void* base =
        mmap(nullptr, total_size_, PROT_READ | PROT_WRITE,
             fd < 0 ? (MAP_PRIVATE | MAP_ANONYMOUS) : MAP_PRIVATE, fd, 0);
    if (fd >= 0) close(fd);  // the mapping keeps the file alive
    if (base == MAP_FAILED) return;
    // Splits the region into a readable mapping and an unreadable one, so the
    // readable extent is exactly |readable_pages| pages.
    uint8_t* guard = static_cast<uint8_t*>(base) + readable_pages * page_size_;
    if (mprotect(guard, page_size_, PROT_NONE) != 0) {
      munmap(base, total_size_);
      return;
    }
    base_ = static_cast<uint8_t*>(base);
    guard_ = guard;
  }
  ~GuardedMapping() {
    if (base_) munmap(base_, total_size_);
    // Only once unmapped: unlinking a mapped file would leave the kernel
    // rendering its entries as "(deleted)", a different case entirely.
    if (!backing_path_.empty()) unlink(backing_path_.c_str());
  }
  GuardedMapping(const GuardedMapping&) = delete;
  GuardedMapping& operator=(const GuardedMapping&) = delete;

  bool ok() const { return base_ != nullptr; }
  // First byte that cannot be read; the end of the readable mapping.
  uint8_t* guard() const { return guard_; }

 private:
  uint8_t* base_ = nullptr;
  uint8_t* guard_ = nullptr;
  size_t total_size_ = 0;
  size_t page_size_ = 0;
  // Path of the file backing the pages; empty when they are anonymous.
  std::string backing_path_;
};

// Returns the offset of the sole occurrence of |needle| within |hay|, requiring
// exactly one match so the byte-patch tests below target an unambiguous field.
size_t FindOnce(const std::vector<uint8_t>& hay,
                const std::vector<uint8_t>& needle) {
  auto first =
      std::search(hay.begin(), hay.end(), needle.begin(), needle.end());
  EXPECT_NE(first, hay.end());
  if (first == hay.end()) return 0;
  auto second = std::search(first + 1, hay.end(), needle.begin(), needle.end());
  EXPECT_EQ(second, hay.end());
  return static_cast<size_t>(first - hay.begin());
}

void AppendBytes(std::vector<uint8_t>& buffer, const void* data,
                 size_t length) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  buffer.insert(buffer.end(), bytes, bytes + length);
}

// __CLANG_OFFLOAD_BUNDLE__ entry descriptor (payload offset/size plus triple
// length), mirroring the on-disk layout the fat-binary extractor walks.
struct BundleEntry {
  uint64_t offset;
  uint64_t size;
  uint64_t triple_size;
};

// Builds an uncompressed Clang offload bundle from {triple, payload} entries,
// matching the format the fat-binary extractor consumes. Used to check that a
// kpack-resolved code object which is itself a bundle unpacks end-to-end.
std::vector<uint8_t> MakeBundle(
    std::vector<std::pair<std::string, std::vector<uint8_t>>> entries) {
  constexpr char kMagic[] = "__CLANG_OFFLOAD_BUNDLE__";
  uint64_t payload_offset = sizeof(kMagic) - 1 + sizeof(uint64_t);
  for (const auto& entry : entries) {
    payload_offset += sizeof(BundleEntry) + entry.first.size();
  }
  std::vector<uint8_t> bundle;
  AppendBytes(bundle, kMagic, sizeof(kMagic) - 1);
  const uint64_t entry_count = entries.size();
  AppendBytes(bundle, &entry_count, sizeof(entry_count));
  uint64_t next_payload_offset = payload_offset;
  for (const auto& entry : entries) {
    const BundleEntry bundle_entry = {
        /*.offset=*/next_payload_offset,
        /*.size=*/entry.second.size(),
        /*.triple_size=*/entry.first.size(),
    };
    AppendBytes(bundle, &bundle_entry, sizeof(bundle_entry));
    AppendBytes(bundle, entry.first.data(), entry.first.size());
    next_payload_offset += entry.second.size();
  }
  for (const auto& entry : entries) {
    AppendBytes(bundle, entry.second.data(), entry.second.size());
  }
  return bundle;
}

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
// Candidates rank strictly by descending feature count, so every two-feature
// subset precedes every one-feature subset. Real AMDGPU targets subset on two
// features, where descending count and descending bitmask coincide; this
// exercises the three-feature headroom the expansion accepts, where the two
// orders diverge (the one-feature mask 4 is numerically larger than the
// two-feature mask 3, yet must rank below it).
TEST(KpackTargetMatch, ThreeFeaturesRankByDescendingCardinality) {
  EXPECT_EQ(
      ExpandTargets("gfx900:a+:b+:c+"),
      (Vec{"gfx900:a+:b+:c+", "gfx900:a+:b+", "gfx900:a+:c+", "gfx900:b+:c+",
           "gfx900:a+", "gfx900:b+", "gfx900:c+", "gfx900"}));
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
  IREE_EXPECT_OK(iree_hal_streaming_kpack_for_each_compatible_target(
      iree_make_cstring_view("gfx942:sramecc+:xnack-"),
      [](iree_string_view_t t, void* ud) -> bool {
        auto* s = static_cast<std::vector<std::string>*>(ud);
        s->emplace_back(t.data, t.size);
        return std::string(t.data, t.size) == "gfx942:xnack-";  // third
      },
      &seen));
  // Iteration stopped at the third candidate instead of emitting the fourth
  // ("gfx942"), which is how a callback returning true is observed.
  EXPECT_EQ(seen, (Vec{"gfx942:sramecc+:xnack-", "gfx942:sramecc+",
                       "gfx942:xnack-"}));
}

// Feature subsetting only ever drops features; it must never flip a feature's
// sign. Expanding "xnack-" must not yield "xnack+" -- loading a code object
// that assumes a hardware feature the agent disabled is a correctness hazard.
TEST(KpackTargetMatch, FeatureSubsetNeverFlipsSign) {
  Vec expanded = ExpandTargets("gfx942:xnack-");
  EXPECT_EQ(expanded, (Vec{"gfx942:xnack-", "gfx942"}));
  EXPECT_EQ(std::find(expanded.begin(), expanded.end(), "gfx942:xnack+"),
            expanded.end());
}

// Collects every emitted candidate into a std::vector<std::string> passed as
// user data. Whether the target was representable is read from
// for_each_compatible_target's returned status, not from this callback.
bool CollectCandidate(iree_string_view_t target, void* user_data) {
  static_cast<std::vector<std::string>*>(user_data)->emplace_back(target.data,
                                                                  target.size);
  return false;
}

// A target carrying more subsettable features than the expansion holds is
// rejected, not silently truncated to the first few. Real AMDGPU target IDs
// subset on two features, so no valid target reaches this; expanding a partial
// feature set would generate the wrong candidates and miss the archive keyed on
// a dropped feature, a miss with no diagnostic.
TEST(KpackTargetMatch, RejectsTooManyFeatures) {
  std::vector<std::string> seen;
  EXPECT_THAT(iree::Status(iree_hal_streaming_kpack_for_each_compatible_target(
                  iree_make_cstring_view("gfx942:a+:b+:c+:d+:e+"),  // five
                  CollectCandidate, &seen)),
              StatusIs(iree::StatusCode::kOutOfRange));
  // The rejection lands before any candidate is emitted.
  EXPECT_TRUE(seen.empty());
}

// The processor prefixes every candidate, so a processor that cannot fit the
// target buffer leaves no candidate representable and is rejected rather than
// silently yielding nothing.
TEST(KpackTargetMatch, RejectsOverlongProcessor) {
  const std::string isa(IREE_HAL_STREAMING_KPACK_TARGET_CAPACITY + 8, 'x');
  std::vector<std::string> seen;
  EXPECT_THAT(
      iree::Status(iree_hal_streaming_kpack_for_each_compatible_target(
          iree_make_cstring_view(isa.c_str()), CollectCandidate, &seen)),
      StatusIs(iree::StatusCode::kOutOfRange));
  EXPECT_TRUE(seen.empty());
}

// The over-long-processor rejection fires ahead of the feature-expansion branch
// too: a feature-bearing target whose processor alone overruns the buffer is
// rejected before any candidate is built.
TEST(KpackTargetMatch, RejectsOverlongProcessorWithFeatures) {
  const std::string isa =
      std::string(IREE_HAL_STREAMING_KPACK_TARGET_CAPACITY + 8, 'x') +
      ":xnack+";
  std::vector<std::string> seen;
  EXPECT_THAT(
      iree::Status(iree_hal_streaming_kpack_for_each_compatible_target(
          iree_make_cstring_view(isa.c_str()), CollectCandidate, &seen)),
      StatusIs(iree::StatusCode::kOutOfRange));
  EXPECT_TRUE(seen.empty());
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

TEST(KpackMetadata, EmptyOrNullInput) {
  iree_hal_streaming_kpack_metadata_t md;
  EXPECT_THAT(iree::Status(iree_hal_streaming_kpack_parse_metadata(
                  iree_make_const_byte_span(nullptr, 0), &md)),
              StatusIs(iree::StatusCode::kInvalidArgument));
  std::vector<uint8_t> empty;
  EXPECT_THAT(iree::Status(iree_hal_streaming_kpack_parse_metadata(
                  iree_make_const_byte_span(empty.data(), 0), &md)),
              StatusIs(iree::StatusCode::kInvalidArgument));
}

TEST(KpackMetadata, SearchPathsWrongType) {
  Mp mp;
  mp.Map(2);
  mp.Str("kernel_name");
  mp.Str("lib/x.so");
  mp.Str("kpack_search_paths");
  mp.Str("not-an-array");  // a string where an array is required
  iree_hal_streaming_kpack_metadata_t md;
  EXPECT_THAT(iree::Status(iree_hal_streaming_kpack_parse_metadata(
                  iree_make_const_byte_span(mp.b.data(), mp.b.size()), &md)),
              StatusIs(iree::StatusCode::kInvalidArgument));
}

TEST(KpackMetadata, SearchPathsSkipsNonStringEntries) {
  Mp mp;
  mp.Map(2);
  mp.Str("kernel_name");
  mp.Str("lib/x.so");
  mp.Str("kpack_search_paths");
  mp.Array(2);
  mp.UInt(42);  // non-string entry is skipped, not fatal
  mp.Str("a.kpack");
  iree_hal_streaming_kpack_metadata_t md;
  IREE_ASSERT_OK(iree_hal_streaming_kpack_parse_metadata(
      iree_make_const_byte_span(mp.b.data(), mp.b.size()), &md));
  ASSERT_EQ(md.search_path_count, 1u);
  EXPECT_EQ(std::string(md.search_paths[0].data, md.search_paths[0].size),
            "a.kpack");
}

TEST(KpackMetadata, TooManySearchPaths) {
  std::vector<std::string> paths;
  for (int i = 0; i < 33; ++i) {
    paths.push_back("p" + std::to_string(i) + ".kpack");
  }
  auto bytes = MakeMetadata("lib/x.so", paths);
  iree_hal_streaming_kpack_metadata_t md;
  EXPECT_THAT(iree::Status(iree_hal_streaming_kpack_parse_metadata(
                  iree_make_const_byte_span(bytes.data(), bytes.size()), &md)),
              StatusIs(iree::StatusCode::kOutOfRange));
}

// A declared string length that runs past the end of the buffer must be
// rejected rather than read past it. The subtractive length checks in the
// MessagePack reader exist precisely for these truncated/oversized cases.
TEST(KpackMetadata, TruncatedStr8ValueRejected) {
  Mp mp;
  mp.Map(2);
  mp.Str("kernel_name");
  mp.Byte(0xd9);  // str8
  mp.Byte(200);   // declared length far exceeds what follows
  mp.Byte('x');
  mp.Byte('y');
  mp.Byte('z');  // only three bytes then EOF
  iree_hal_streaming_kpack_metadata_t md;
  EXPECT_THAT(iree::Status(iree_hal_streaming_kpack_parse_metadata(
                  iree_make_const_byte_span(mp.b.data(), mp.b.size()), &md)),
              StatusIs(iree::StatusCode::kInvalidArgument));
}

TEST(KpackMetadata, OversizeStr32ValueRejected) {
  Mp mp;
  mp.Map(2);
  mp.Str("kernel_name");
  mp.Byte(0xdb);  // str32
  mp.Byte(0xff);
  mp.Byte(0xff);
  mp.Byte(0xff);
  mp.Byte(0xff);  // declared length ~4 GiB in a near-empty buffer
  iree_hal_streaming_kpack_metadata_t md;
  EXPECT_THAT(iree::Status(iree_hal_streaming_kpack_parse_metadata(
                  iree_make_const_byte_span(mp.b.data(), mp.b.size()), &md)),
              StatusIs(iree::StatusCode::kInvalidArgument));
}

TEST(KpackMetadata, ValueEndingOnMarkerRejected) {
  // A uint64 (0xcf) marker with no trailing bytes exercises the big-endian
  // integer read bound; a float64 (0xcb) marker exercises the float bound.
  for (uint8_t marker : {uint8_t{0xcf}, uint8_t{0xcb}}) {
    Mp mp;
    mp.Map(2);
    mp.Str("kernel_name");
    mp.Byte(marker);  // EOF immediately after the marker
    iree_hal_streaming_kpack_metadata_t md;
    EXPECT_THAT(iree::Status(iree_hal_streaming_kpack_parse_metadata(
                    iree_make_const_byte_span(mp.b.data(), mp.b.size()), &md)),
                StatusIs(iree::StatusCode::kInvalidArgument));
  }
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
  // ".." is preserved for the kernel to resolve; only "." and separators are
  // simplified. "/usr/lib/.." physically names "/usr", so this reaches the
  // sibling .kpack directory when no symlink intervenes.
  EXPECT_EQ(ResolveRel("/usr/lib/libfoo.so", "../.kpack/foo.kpack"),
            "/usr/lib/../.kpack/foo.kpack");
}
TEST(KpackPath, ResolveSameDir) {
  EXPECT_EQ(ResolveRel("/opt/rocm/lib/libhip.so", "foo.kpack"),
            "/opt/rocm/lib/foo.kpack");
}
TEST(KpackPath, ResolveCollapsesDotKeepsDotDot) {
  // "." collapses; ".." is preserved (collapsing it would not commute with a
  // symlink at "d").
  EXPECT_EQ(ResolveRel("/a/b/c.so", "./d/../e.kpack"), "/a/b/d/../e.kpack");
}
TEST(KpackPath, ResolveAbsoluteRelativePassthrough) {
  EXPECT_EQ(ResolveRel("/a/b/c.so", "/abs/d.kpack"), "/abs/d.kpack");
}
TEST(KpackPath, ResolveAbsolutePreservesDotDot) {
  EXPECT_EQ(ResolveRel("/a/b/c.so", "/abs/x/../d.kpack"), "/abs/x/../d.kpack");
}
// An absolute spelling whose ".." follows a symlink still names the file the
// caller meant, because ".." is left for the kernel: realpath of the resolved
// spelling equals realpath of the archive it targets. A lexical ".." collapse
// would instead yield "base/sub/real/a.kpack", which does not exist.
TEST(KpackPath, ResolvePreservesDotDotAcrossSymlink) {
  const std::string base =
      std::string(::testing::TempDir()) + "/kpack_path_symlink";
  const std::string real = base + "/real";
  const std::string sub = base + "/sub";
  ASSERT_TRUE(EnsureDirectory(base));
  ASSERT_TRUE(EnsureDirectory(real));
  ASSERT_TRUE(EnsureDirectory(sub));
  WriteFile(real + "/a.kpack", MakeMinimalAmdgpuElf());
  const std::string link = sub + "/link";
  unlink(link.c_str());  // may survive an earlier run
  ASSERT_EQ(symlink(real.c_str(), link.c_str()), 0);

  const std::string spelling = link + "/../real/a.kpack";
  std::string resolved = ResolveRel("/ignored", spelling.c_str());

  char* real_resolved = realpath(resolved.c_str(), nullptr);
  char* real_target = realpath((real + "/a.kpack").c_str(), nullptr);
  ASSERT_NE(real_resolved, nullptr)
      << "resolved spelling names nothing: " << resolved;
  ASSERT_NE(real_target, nullptr);
  EXPECT_STREQ(real_resolved, real_target);
  free(real_resolved);
  free(real_target);
}

//===----------------------------------------------------------------------===//
// query_mapping
//===----------------------------------------------------------------------===//

// A symbol in the test binary's mapped image (not heap), so /proc/self/maps
// resolves it to a real file.
static const int kProbeSymbol = 0x5a5a;

TEST(KpackQueryMapping, ResolvesOwnBinary) {
  char path[1024];
  iree_hal_streaming_kpack_mapping_t mapping;
  // On Linux this resolves to the running test executable/library.
  IREE_ASSERT_OK(iree_hal_streaming_kpack_query_mapping(
      &kProbeSymbol, path, sizeof(path), &mapping));
  ASSERT_GT(mapping.path.size, 0u);
  EXPECT_EQ(mapping.path.data, static_cast<const char*>(path));
  EXPECT_GT(mapping.readable_bytes, 0u);
}

TEST(KpackQueryMapping, HeapPointerHasExtentButNoPath) {
  // A heap allocation is backed by an anonymous / [heap] mapping: a real extent
  // with no file behind it. Every test below hands resolution a heap pointer to
  // its metadata, so the query must report an extent for a mapping it cannot
  // name.
  const size_t kAllocationSize = 1 << 20;
  void* heap = malloc(kAllocationSize);
  ASSERT_NE(heap, nullptr);
  char path[1024];
  iree_hal_streaming_kpack_mapping_t mapping;
  iree_status_t status = iree_hal_streaming_kpack_query_mapping(
      heap, path, sizeof(path), &mapping);
  IREE_EXPECT_OK(status);
  EXPECT_EQ(mapping.path.size, 0u);
  // The whole allocation is readable from |heap|, so the extent covers it.
  EXPECT_GE(mapping.readable_bytes, kAllocationSize);
  free(heap);
}

TEST(KpackQueryMapping, DeletedBackingFileHasExtentButNoPath) {
  // An unlinked backing file is rendered as "<path> (deleted)", a spelling that
  // names no file that can be opened. The mapping is still readable, so it has
  // an extent; reporting the spelling as a path would resolve search paths
  // against a directory that no longer holds the file.
  const size_t kMappingSize = 4096;
  std::string path =
      WriteTempFile("deleted.kpack", std::vector<uint8_t>(kMappingSize, 0));
  int fd = open(path.c_str(), O_RDONLY);
  ASSERT_GE(fd, 0);
  void* region = mmap(nullptr, kMappingSize, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  ASSERT_NE(region, MAP_FAILED);
  ASSERT_EQ(unlink(path.c_str()), 0);

  char buffer[1024];
  iree_hal_streaming_kpack_mapping_t mapping;
  IREE_EXPECT_OK(iree_hal_streaming_kpack_query_mapping(
      region, buffer, sizeof(buffer), &mapping));
  EXPECT_EQ(mapping.path.size, 0u);
  EXPECT_EQ(mapping.readable_bytes, kMappingSize);
  munmap(region, kMappingSize);
}

TEST(KpackQueryMapping, UnmappedAddressNotFound) {
  char path[1024];
  iree_hal_streaming_kpack_mapping_t mapping;
  EXPECT_THAT(iree::Status(iree_hal_streaming_kpack_query_mapping(
                  reinterpret_cast<void*>(0x1), path, sizeof(path), &mapping)),
              StatusIs(iree::StatusCode::kNotFound));
}

TEST(KpackQueryMapping, ProtNoneMappingNotFound) {
  // A mapped but unreadable page faults on access exactly like an unmapped one,
  // so it must not be reported as an extent of readable bytes. The ELF loader
  // leaves such pages between a library's segments, where they are file-backed
  // and named after that library just as the segments around them are: reading
  // the permission bits is the only thing that separates them. Reporting this
  // page would hand back both an extent of bytes that fault and a path the
  // mapping cannot be read from.
  const std::string backing =
      std::string(::testing::TempDir()) + "/kpack_prot_none_backing.bin";
  GuardedMapping region(/*readable_pages=*/1, backing);
  ASSERT_TRUE(region.ok());
  char path[1024];
  iree_hal_streaming_kpack_mapping_t mapping;
  EXPECT_THAT(iree::Status(iree_hal_streaming_kpack_query_mapping(
                  region.guard(), path, sizeof(path), &mapping)),
              StatusIs(iree::StatusCode::kNotFound));
}

TEST(KpackQueryMapping, ExtentStopsAtMappingEnd) {
  // The extent ends at the mapping boundary rather than spilling into whatever
  // follows it.
  GuardedMapping region(/*readable_pages=*/2);
  ASSERT_TRUE(region.ok());
  const iree_host_size_t kTailBytes = 100;
  char path[1024];
  iree_hal_streaming_kpack_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_streaming_kpack_query_mapping(
      region.guard() - kTailBytes, path, sizeof(path), &mapping));
  EXPECT_EQ(mapping.readable_bytes, kTailBytes);
  EXPECT_EQ(mapping.path.size, 0u);
}

TEST(KpackQueryMapping, NullAddress) {
  char path[64];
  iree_hal_streaming_kpack_mapping_t mapping;
  EXPECT_THAT(iree::Status(iree_hal_streaming_kpack_query_mapping(
                  nullptr, path, sizeof(path), &mapping)),
              StatusIs(iree::StatusCode::kInvalidArgument));
}
TEST(KpackQueryMapping, ZeroCapacity) {
  char path[64];
  iree_hal_streaming_kpack_mapping_t mapping;
  EXPECT_THAT(iree::Status(iree_hal_streaming_kpack_query_mapping(
                  &kProbeSymbol, path, 0, &mapping)),
              StatusIs(iree::StatusCode::kInvalidArgument));
}
TEST(KpackQueryMapping, BufferTooSmall) {
  char path[1];
  iree_hal_streaming_kpack_mapping_t mapping;
  EXPECT_THAT(iree::Status(iree_hal_streaming_kpack_query_mapping(
                  &kProbeSymbol, path, sizeof(path), &mapping)),
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

TEST(KpackArchive, ZeroSizeBlobRejected) {
  auto bytes = KpackBuilder(false).Add("a#0", "gfx900", {}).Build();
  iree_hal_streaming_kpack_archive_t archive;
  IREE_ASSERT_OK(iree_hal_streaming_kpack_archive_open(
      iree_make_const_byte_span(bytes.data(), bytes.size()), &archive));
  void* out = nullptr;
  iree_host_size_t out_size = 0;
  EXPECT_THAT(iree::Status(iree_hal_streaming_kpack_archive_get_kernel(
                  &archive, IREE_SV("a#0"), IREE_SV("gfx900"),
                  iree_allocator_system(), &out, &out_size)),
              StatusIs(iree::StatusCode::kInvalidArgument));
}

TEST(KpackArchive, UnknownCompressionSchemeRejected) {
  auto bytes = KpackBuilder(false).Add("a#0", "gfx900", {1, 2, 3}).Build();
  // Rewrite the compression_scheme value (fixstr "none") to an unknown token;
  // an unrecognized scheme must fail closed at open time.
  size_t at = FindOnce(bytes, {0xa4, 'n', 'o', 'n', 'e'});
  const uint8_t kLzma[] = {'l', 'z', 'm', 'a'};
  memcpy(bytes.data() + at + 1, kLzma, sizeof(kLzma));
  iree_hal_streaming_kpack_archive_t archive;
  EXPECT_THAT(
      iree::Status(iree_hal_streaming_kpack_archive_open(
          iree_make_const_byte_span(bytes.data(), bytes.size()), &archive)),
      StatusIs(iree::StatusCode::kInvalidArgument));
}

TEST(KpackArchive, NonStringCompressionSchemeRejected) {
  auto bytes = KpackBuilder(false).Add("a#0", "gfx900", {1, 2, 3}).Build();
  // Overwrite the compression_scheme value (fixstr "none") with an equal-length
  // msgpack uint32, leaving the rest of the TOC layout intact. A scheme that is
  // present but not a string names no scheme at all, so it must fail closed
  // rather than apply the "none" default and decode the archive under a scheme
  // it never declared.
  size_t at = FindOnce(bytes, {0xa4, 'n', 'o', 'n', 'e'});
  const uint8_t kUInt32One[] = {0xce, 0x00, 0x00, 0x00, 0x01};
  memcpy(bytes.data() + at, kUInt32One, sizeof(kUInt32One));
  iree_hal_streaming_kpack_archive_t archive;
  EXPECT_THAT(
      iree::Status(iree_hal_streaming_kpack_archive_open(
          iree_make_const_byte_span(bytes.data(), bytes.size()), &archive)),
      StatusIs(iree::StatusCode::kInvalidArgument));
}

// The "compression_scheme" TOC field is optional: an archive that omits it
// opens as the uncompressed NONE scheme. A present-but-invalid scheme fails
// closed; the absent case is the documented default, so a NoOp archive built
// without the field must still open and decode its blobs.
TEST(KpackArchive, AbsentCompressionSchemeDefaultsToNone) {
  std::vector<uint8_t> k = {'A', 'B', 'C', 'D'};
  auto bytes = KpackBuilder(/*zstd=*/false)
                   .OmitCompressionScheme()
                   .Add("lib/x.so#0", "gfx900", k)
                   .Build();
  iree_hal_streaming_kpack_archive_t archive;
  IREE_ASSERT_OK(iree_hal_streaming_kpack_archive_open(
      iree_make_const_byte_span(bytes.data(), bytes.size()), &archive));
  EXPECT_EQ(archive.compression, IREE_HAL_STREAMING_KPACK_COMPRESSION_NONE);

  // The default is fully wired, not just the enum value: the blobs array is
  // consulted and the uncompressed bytes come back.
  void* out = nullptr;
  iree_host_size_t out_size = 0;
  IREE_ASSERT_OK(iree_hal_streaming_kpack_archive_get_kernel(
      &archive, IREE_SV("lib/x.so#0"), IREE_SV("gfx900"),
      iree_allocator_system(), &out, &out_size));
  ASSERT_EQ(out_size, k.size());
  EXPECT_EQ(0, memcmp(out, k.data(), k.size()));
  FreeBuffer(out);
}

TEST(KpackArchive, ExactArchMatchDoesNotFuzzMatch) {
  // get_kernel does an exact architecture-key match: a bare/less-specific
  // request must not resolve a feature-qualified key ("xnack+"). Loading a code
  // object that requires a hardware feature the agent lacks is unsafe.
  auto elf = MakeMinimalAmdgpuElf();
  auto bytes =
      KpackBuilder(false).Add("lib/x.so#0", "gfx942:xnack+", elf).Build();
  iree_hal_streaming_kpack_archive_t archive;
  IREE_ASSERT_OK(iree_hal_streaming_kpack_archive_open(
      iree_make_const_byte_span(bytes.data(), bytes.size()), &archive));
  void* out = nullptr;
  iree_host_size_t out_size = 0;
  EXPECT_THAT(iree::Status(iree_hal_streaming_kpack_archive_get_kernel(
                  &archive, IREE_SV("lib/x.so#0"), IREE_SV("gfx942"),
                  iree_allocator_system(), &out, &out_size)),
              StatusIs(iree::StatusCode::kNotFound));
}

TEST(KpackArchive, MissingTocKey) {
  auto bytes = KpackBuilder(false).Add("a#0", "gfx900", {1, 2, 3}).Build();
  // Rename the top-level "toc" key so the lookup misses it.
  size_t at = FindOnce(bytes, {0xa3, 't', 'o', 'c'});
  bytes[at + 1] = 'x';
  iree_hal_streaming_kpack_archive_t archive;
  IREE_ASSERT_OK(iree_hal_streaming_kpack_archive_open(
      iree_make_const_byte_span(bytes.data(), bytes.size()), &archive));
  void* out = nullptr;
  iree_host_size_t out_size = 0;
  EXPECT_THAT(iree::Status(iree_hal_streaming_kpack_archive_get_kernel(
                  &archive, IREE_SV("a#0"), IREE_SV("gfx900"),
                  iree_allocator_system(), &out, &out_size)),
              StatusIs(iree::StatusCode::kNotFound));
}

TEST(KpackArchive, ArchNodeMissingOrdinal) {
  auto bytes = KpackBuilder(false).Add("a#0", "gfx900", {1, 2, 3}).Build();
  // Rename the "ordinal" key inside the arch node.
  size_t at = FindOnce(bytes, {0xa7, 'o', 'r', 'd', 'i', 'n', 'a', 'l'});
  bytes[at + 1] = 'x';
  iree_hal_streaming_kpack_archive_t archive;
  IREE_ASSERT_OK(iree_hal_streaming_kpack_archive_open(
      iree_make_const_byte_span(bytes.data(), bytes.size()), &archive));
  void* out = nullptr;
  iree_host_size_t out_size = 0;
  EXPECT_THAT(iree::Status(iree_hal_streaming_kpack_archive_get_kernel(
                  &archive, IREE_SV("a#0"), IREE_SV("gfx900"),
                  iree_allocator_system(), &out, &out_size)),
              StatusIs(iree::StatusCode::kInvalidArgument));
}

TEST(KpackArchive, BlobsArrayShorterThanOrdinal) {
  auto bytes = KpackBuilder(false).Add("a#0", "gfx900", {1, 2, 3}).Build();
  // Shrink the single-element "blobs" fixarray (0x91) to empty (0x90); ordinal
  // 0 now indexes past its end.
  size_t at = FindOnce(bytes, {0xa5, 'b', 'l', 'o', 'b', 's', 0x91});
  bytes[at + 6] = 0x90;
  iree_hal_streaming_kpack_archive_t archive;
  IREE_ASSERT_OK(iree_hal_streaming_kpack_archive_open(
      iree_make_const_byte_span(bytes.data(), bytes.size()), &archive));
  void* out = nullptr;
  iree_host_size_t out_size = 0;
  EXPECT_THAT(iree::Status(iree_hal_streaming_kpack_archive_get_kernel(
                  &archive, IREE_SV("a#0"), IREE_SV("gfx900"),
                  iree_allocator_system(), &out, &out_size)),
              StatusIs(iree::StatusCode::kNotFound));
}

TEST(KpackArchive, BlobOffsetPastEof) {
  std::vector<uint8_t> big(200, 0xAA);
  auto bytes = KpackBuilder(false)
                   .Add("a#0", "gfx900", big)
                   .Add("b#0", "gfx900", {1, 2, 3})
                   .Build();
  // The second blob's offset is 64 + 200 = 264 (msgpack uint16 0xcd 0x01 0x08).
  // Rewrite it well past EOF so the bounds check rejects it.
  size_t at =
      FindOnce(bytes, {0xa6, 'o', 'f', 'f', 's', 'e', 't', 0xcd, 0x01, 0x08});
  bytes[at + 8] = 0xff;
  bytes[at + 9] = 0xff;
  iree_hal_streaming_kpack_archive_t archive;
  IREE_ASSERT_OK(iree_hal_streaming_kpack_archive_open(
      iree_make_const_byte_span(bytes.data(), bytes.size()), &archive));
  void* out = nullptr;
  iree_host_size_t out_size = 0;
  EXPECT_THAT(iree::Status(iree_hal_streaming_kpack_archive_get_kernel(
                  &archive, IREE_SV("b#0"), IREE_SV("gfx900"),
                  iree_allocator_system(), &out, &out_size)),
              StatusIs(iree::StatusCode::kInvalidArgument));
}

TEST(KpackArchive, TocOffsetInsideHeader) {
  auto bytes = KpackBuilder(false).Add("a#0", "gfx900", {1}).Build();
  // A TOC offset (u64 at byte 8) pointing inside the 16-byte header is below
  // the valid range [16, EOF) and must be rejected.
  for (int i = 0; i < 8; ++i) bytes[8 + i] = 0;
  bytes[8] = 8;
  iree_hal_streaming_kpack_archive_t archive;
  EXPECT_THAT(
      iree::Status(iree_hal_streaming_kpack_archive_open(
          iree_make_const_byte_span(bytes.data(), bytes.size()), &archive)),
      StatusIs(iree::StatusCode::kInvalidArgument));
}

TEST(KpackArchive, TocNotAMap) {
  auto bytes = KpackBuilder(false).Add("a#0", "gfx900", {1}).Build();
  uint64_t toc_offset = 0;
  memcpy(&toc_offset, bytes.data() + 8, sizeof(toc_offset));
  ASSERT_LT(toc_offset, bytes.size());
  bytes[toc_offset] = 0x00;  // positive fixint where a map header is required
  iree_hal_streaming_kpack_archive_t archive;
  EXPECT_THAT(
      iree::Status(iree_hal_streaming_kpack_archive_open(
          iree_make_const_byte_span(bytes.data(), bytes.size()), &archive)),
      StatusIs(iree::StatusCode::kInvalidArgument));
}

#if !defined(IREE_HAVE_ZSTD)
TEST(KpackArchive, ZstdWithoutSupportUnimplemented) {
  auto bytes =
      KpackBuilder(/*zstd=*/true).Add("a#0", "gfx900", {1, 2, 3}).Build();
  iree_hal_streaming_kpack_archive_t archive;
  IREE_ASSERT_OK(iree_hal_streaming_kpack_archive_open(
      iree_make_const_byte_span(bytes.data(), bytes.size()), &archive));
  void* out = nullptr;
  iree_host_size_t out_size = 0;
  EXPECT_THAT(iree::Status(iree_hal_streaming_kpack_archive_get_kernel(
                  &archive, IREE_SV("a#0"), IREE_SV("gfx900"),
                  iree_allocator_system(), &out, &out_size)),
              StatusIs(iree::StatusCode::kUnimplemented));
}
#endif  // !IREE_HAVE_ZSTD

#if defined(IREE_HAVE_ZSTD)
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

// A frame whose declared frame_size exceeds the bytes present in the blob must
// be rejected, not read past the blob end.
TEST(KpackArchive, ZstdFrameSizeTruncated) {
  std::vector<uint8_t> k(300, 0x5a);
  auto bytes = KpackBuilder(/*zstd=*/true).Add("a#0", "gfx900", k).Build();
  // Blob at byte 64: [num_kernels u32][frame_size u32][frame...]. Inflate the
  // first frame_size so it runs past the blob end.
  for (int i = 0; i < 4; ++i) bytes[68 + i] = 0xff;
  iree_hal_streaming_kpack_archive_t archive;
  IREE_ASSERT_OK(iree_hal_streaming_kpack_archive_open(
      iree_make_const_byte_span(bytes.data(), bytes.size()), &archive));
  void* out = nullptr;
  iree_host_size_t out_size = 0;
  EXPECT_THAT(iree::Status(iree_hal_streaming_kpack_archive_get_kernel(
                  &archive, IREE_SV("a#0"), IREE_SV("gfx900"),
                  iree_allocator_system(), &out, &out_size)),
              StatusIs(iree::StatusCode::kInvalidArgument));
}

// A corrupted zstd frame (mangled magic) must surface as DATA_LOSS.
TEST(KpackArchive, ZstdFrameCorruptedDataLoss) {
  std::vector<uint8_t> k(300, 0x5a);
  auto bytes = KpackBuilder(/*zstd=*/true).Add("a#0", "gfx900", k).Build();
  // The first frame begins at byte 72 (64 + num_kernels + frame_size). Flip a
  // magic byte so ZSTD_decompress errors.
  bytes[72] ^= 0xff;
  iree_hal_streaming_kpack_archive_t archive;
  IREE_ASSERT_OK(iree_hal_streaming_kpack_archive_open(
      iree_make_const_byte_span(bytes.data(), bytes.size()), &archive));
  void* out = nullptr;
  iree_host_size_t out_size = 0;
  EXPECT_THAT(iree::Status(iree_hal_streaming_kpack_archive_get_kernel(
                  &archive, IREE_SV("a#0"), IREE_SV("gfx900"),
                  iree_allocator_system(), &out, &out_size)),
              StatusIs(iree::StatusCode::kDataLoss));
}

// A recorded original_size that disagrees with the frame's true output size
// must surface as DATA_LOSS (size mismatch).
TEST(KpackArchive, ZstdOriginalSizeMismatchDataLoss) {
  std::vector<uint8_t> k(300, 0x5a);
  auto bytes = KpackBuilder(/*zstd=*/true).Add("a#0", "gfx900", k).Build();
  // original_size for the (only) arch node is 300 (msgpack uint16 0xcd 0x01
  // 0x2c). Enlarge it so the decompressed size no longer matches.
  size_t at = FindOnce(bytes, {0xad, 'o', 'r', 'i', 'g', 'i', 'n', 'a', 'l',
                               '_', 's', 'i', 'z', 'e', 0xcd, 0x01, 0x2c});
  bytes[at + 15] = 0xff;
  bytes[at + 16] = 0xff;
  iree_hal_streaming_kpack_archive_t archive;
  IREE_ASSERT_OK(iree_hal_streaming_kpack_archive_open(
      iree_make_const_byte_span(bytes.data(), bytes.size()), &archive));
  void* out = nullptr;
  iree_host_size_t out_size = 0;
  EXPECT_THAT(iree::Status(iree_hal_streaming_kpack_archive_get_kernel(
                  &archive, IREE_SV("a#0"), IREE_SV("gfx900"),
                  iree_allocator_system(), &out, &out_size)),
              StatusIs(iree::StatusCode::kDataLoss));
}

// An ordinal that points past the frame count must be NOT_FOUND.
TEST(KpackArchive, ZstdOrdinalOutOfRange) {
  std::vector<uint8_t> k(300, 0x5a);
  auto bytes = KpackBuilder(/*zstd=*/true).Add("a#0", "gfx900", k).Build();
  // The single arch node's ordinal is 0; point it past the one frame present.
  size_t at = FindOnce(bytes, {0xa7, 'o', 'r', 'd', 'i', 'n', 'a', 'l', 0x00});
  bytes[at + 8] = 0x05;
  iree_hal_streaming_kpack_archive_t archive;
  IREE_ASSERT_OK(iree_hal_streaming_kpack_archive_open(
      iree_make_const_byte_span(bytes.data(), bytes.size()), &archive));
  void* out = nullptr;
  iree_host_size_t out_size = 0;
  EXPECT_THAT(iree::Status(iree_hal_streaming_kpack_archive_get_kernel(
                  &archive, IREE_SV("a#0"), IREE_SV("gfx900"),
                  iree_allocator_system(), &out, &out_size)),
              StatusIs(iree::StatusCode::kNotFound));
}

// A recorded original_size beyond the code-object size cap must be rejected
// before the output buffer is allocated for it. The check runs before the frame
// is read, so no large data is needed.
TEST(KpackArchive, ZstdOriginalSizeCapEnforced) {
  std::vector<uint8_t> k(300, 0x5a);
  auto bytes = KpackBuilder(/*zstd=*/true).Add("a#0", "gfx900", k).Build();
  // original_size is encoded as a uint16 (0xcd 0x01 0x2c = 300). Widen it to a
  // uint32 (0xce) above KPACK_MAX_CODE_OBJECT_BYTES (256 MiB). Widening is
  // safe: the TOC is self-describing and sits after the unchanged blob, and
  // toc_offset points at the TOC start, which does not move.
  size_t at = FindOnce(bytes, {0xad, 'o', 'r', 'i', 'g', 'i', 'n', 'a', 'l',
                               '_', 's', 'i', 'z', 'e', 0xcd, 0x01, 0x2c});
  const std::vector<uint8_t> big_u32 = {0xce, 0x90, 0x00, 0x00, 0x00};  // 2.25G
  bytes.erase(bytes.begin() + at + 14, bytes.begin() + at + 17);
  bytes.insert(bytes.begin() + at + 14, big_u32.begin(), big_u32.end());
  iree_hal_streaming_kpack_archive_t archive;
  IREE_ASSERT_OK(iree_hal_streaming_kpack_archive_open(
      iree_make_const_byte_span(bytes.data(), bytes.size()), &archive));
  void* out = nullptr;
  iree_host_size_t out_size = 0;
  EXPECT_THAT(iree::Status(iree_hal_streaming_kpack_archive_get_kernel(
                  &archive, IREE_SV("a#0"), IREE_SV("gfx900"),
                  iree_allocator_system(), &out, &out_size)),
              StatusIs(iree::StatusCode::kInvalidArgument));
}
#endif  // IREE_HAVE_ZSTD

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

TEST(KpackResolve, MetadataAtMappingEndIsBounded) {
  // The HIPK metadata pointer carries no length, so the only thing that makes
  // the msgpack walk memory-safe is the extent of the mapping holding it. This
  // blob sits flush against the end of the readable mapping and declares a
  // string whose payload therefore begins on the guard page: a reader bounded
  // by anything other than that extent accepts the declared length, skips the
  // payload, and reads the next key from unreadable memory.
  GuardedMapping region(/*readable_pages=*/1);
  ASSERT_TRUE(region.ok());

  // The first key is not the one the parser looks for, so it moves on to a
  // second pair rather than stopping at the first. The declared length is kept
  // well inside the guard page so the fault is that page rather than whatever
  // happens to lie beyond the region.
  const std::vector<uint8_t> blob = {
      0x82,                  // map, 2 pairs
      0xa3, 'z',  'z', 'z',  // key: "zzz"
      0xd9, 0x64,            // value: str8 declaring 100 bytes
  };
  uint8_t* metadata = region.guard() - blob.size();
  memcpy(metadata, blob.data(), blob.size());

  const iree_string_view_t target = IREE_SV("gfx900");
  void* out = nullptr;
  iree_host_size_t out_size = 0;
  EXPECT_THAT(
      iree::Status(iree_hal_streaming_kpack_resolve_code_object(
          metadata, 0, 1, &target, iree_allocator_system(), &out, &out_size)),
      StatusIs(iree::StatusCode::kInvalidArgument));
  EXPECT_EQ(out, nullptr);
}

// A metadata pointer that lies in no mapping at all has no readable bytes to
// bound the parse, so resolution reports it rather than dereferencing it.
TEST(KpackResolve, UnmappedMetadataRejected) {
  const iree_string_view_t target = IREE_SV("gfx900");
  void* out = nullptr;
  iree_host_size_t out_size = 0;
  EXPECT_THAT(iree::Status(iree_hal_streaming_kpack_resolve_code_object(
                  reinterpret_cast<void*>(0x1), 0, 1, &target,
                  iree_allocator_system(), &out, &out_size)),
              StatusIs(iree::StatusCode::kNotFound));
  EXPECT_EQ(out, nullptr);
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

// Resolution probes every archive on the search path and returns the first that
// holds the requested code object.
TEST(KpackResolve, SearchesMultipleArchives) {
  auto other =
      KpackBuilder(false).Add("lib/y.so#0", "gfx900", MakeMinimalAmdgpuElf());
  auto elf = MakeMinimalAmdgpuElf();
  auto match = KpackBuilder(false).Add("lib/x.so#0", "gfx900", elf);
  std::string p_other = WriteTempFile("multi_other.kpack", other.Build());
  std::string p_match = WriteTempFile("multi_match.kpack", match.Build());
  auto metadata = MakeMetadata("lib/x.so", {p_other, p_match});

  void* out = nullptr;
  iree_host_size_t out_size = 0;
  IREE_ASSERT_OK(Resolve(metadata, 0, {"gfx900"}, &out, &out_size));
  ASSERT_EQ(out_size, elf.size());
  EXPECT_EQ(0, memcmp(out, elf.data(), elf.size()));
  FreeBuffer(out);
}

// A file that maps but is not a valid archive does not stop the search: a valid
// archive later on the search path still resolves, and the diagnostic about the
// bad one is discarded because the search succeeded.
TEST(KpackResolve, UnparseableArchiveDoesNotBlockLaterArchive) {
  std::vector<uint8_t> garbage(32, 0xAB);  // maps, but not a "KPAK" archive
  std::string p_bad = WriteTempFile("multi_bad.kpack", garbage);
  auto elf = MakeMinimalAmdgpuElf();
  auto good = KpackBuilder(false).Add("lib/x.so#0", "gfx900", elf).Build();
  std::string p_good = WriteTempFile("multi_good.kpack", good);

  void* out = nullptr;
  iree_host_size_t out_size = 0;
  auto with_good = MakeMetadata("lib/x.so", {p_bad, p_good});
  IREE_ASSERT_OK(Resolve(with_good, 0, {"gfx900"}, &out, &out_size));
  ASSERT_EQ(out_size, elf.size());
  FreeBuffer(out);
}

// When the only candidate is a file that is present but not an archive, the
// caller learns the file is malformed rather than that nothing was found: the
// file being there is a fact they cannot otherwise discover.
TEST(KpackResolve, UnparseableArchiveReportedWhenNothingMatches) {
  std::vector<uint8_t> garbage(32, 0xAB);
  std::string p_bad = WriteTempFile("only_bad.kpack", garbage);

  void* out = nullptr;
  iree_host_size_t out_size = 0;
  auto only_bad = MakeMetadata("lib/x.so", {p_bad});
  EXPECT_THAT(iree::Status(Resolve(only_bad, 0, {"gfx900"}, &out, &out_size)),
              StatusIs(iree::StatusCode::kInvalidArgument));
}

// An archive reachable only through a spelling whose ".." follows a symlink
// must resolve. Collapsing ".." lexically does not commute with symlink
// resolution, so the normalized spelling of such a path names a different file
// than the caller did, or none at all.
TEST(KpackResolve, SymlinkedParentSpellingResolves) {
  auto elf = MakeMinimalAmdgpuElf();
  auto archive = KpackBuilder(false).Add("lib/x.so#0", "gfx900", elf).Build();
  const std::string base = std::string(::testing::TempDir()) + "/kpack_symlink";
  const std::string real = base + "/real";
  const std::string sub = base + "/sub";
  ASSERT_TRUE(EnsureDirectory(base));
  ASSERT_TRUE(EnsureDirectory(real));
  ASSERT_TRUE(EnsureDirectory(sub));
  WriteFile(real + "/a.kpack", archive);
  const std::string link = sub + "/link";
  unlink(link.c_str());  // may survive an earlier run
  ASSERT_EQ(symlink(real.c_str(), link.c_str()), 0);

  // "sub/link/.." is |base|, so this names the archive; collapsing it lexically
  // yields "sub/real/a.kpack", where nothing exists.
  const std::string spelling = link + "/../real/a.kpack";
  ASSERT_EQ(access(spelling.c_str(), R_OK), 0);
  auto metadata = MakeMetadata("lib/x.so", {"/nonexistent/none.kpack"});
  setenv("ROCM_KPACK_PATH", spelling.c_str(), 1);
  void* out = nullptr;
  iree_host_size_t out_size = 0;
  iree_status_t status = Resolve(metadata, 0, {"gfx900"}, &out, &out_size);
  unsetenv("ROCM_KPACK_PATH");
  IREE_ASSERT_OK(status);
  EXPECT_EQ(out_size, elf.size());
  FreeBuffer(out);
}

// The embedded search path flows through resolve_relative_path rather than the
// ROCM_KPACK_PATH override, so an embedded spelling whose ".." follows a
// symlink must reach the real archive. Collapsing ".." lexically would name a
// nonexistent file and the resolve would miss. The spelling is absolute because
// heap-allocated test metadata has no backing file, so a relative embedded path
// is rejected before resolution; an absolute one still traverses the same
// normalize as a relative path would.
TEST(KpackResolve, EmbeddedSymlinkedParentSpellingResolves) {
  auto elf = MakeMinimalAmdgpuElf();
  auto archive = KpackBuilder(false).Add("lib/x.so#0", "gfx900", elf).Build();
  const std::string base =
      std::string(::testing::TempDir()) + "/kpack_embedded_symlink";
  const std::string real = base + "/real";
  const std::string sub = base + "/sub";
  ASSERT_TRUE(EnsureDirectory(base));
  ASSERT_TRUE(EnsureDirectory(real));
  ASSERT_TRUE(EnsureDirectory(sub));
  WriteFile(real + "/a.kpack", archive);
  const std::string link = sub + "/link";
  unlink(link.c_str());  // may survive an earlier run
  ASSERT_EQ(symlink(real.c_str(), link.c_str()), 0);

  // "sub/link/.." is |base|, so this names the archive; collapsing it lexically
  // yields "sub/real/a.kpack", where nothing exists.
  const std::string spelling = link + "/../real/a.kpack";
  ASSERT_EQ(access(spelling.c_str(), R_OK), 0);
  auto metadata = MakeMetadata("lib/x.so", {spelling});

  void* out = nullptr;
  iree_host_size_t out_size = 0;
  IREE_ASSERT_OK(Resolve(metadata, 0, {"gfx900"}, &out, &out_size));
  EXPECT_EQ(out_size, elf.size());
  FreeBuffer(out);
}

// A directory on the search path names something, so it is reported rather than
// skipped. Mapping a directory fails with the same NOT_FOUND that a path naming
// nothing produces, so the code alone cannot carry the diagnosis and the
// message has to name the offending path.
TEST(KpackResolve, DirectoryOnSearchPathIsReported) {
  const std::string directory =
      std::string(::testing::TempDir()) + "/kpack_directory_entry";
  ASSERT_TRUE(EnsureDirectory(directory));
  auto metadata = MakeMetadata("lib/x.so", {"/nonexistent/none.kpack"});
  setenv("ROCM_KPACK_PATH", directory.c_str(), 1);
  void* out = nullptr;
  iree_host_size_t out_size = 0;
  iree::Status status(Resolve(metadata, 0, {"gfx900"}, &out, &out_size));
  unsetenv("ROCM_KPACK_PATH");
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr(directory));
}

// A relative search path is formed against the binary owning the metadata, so a
// mapping with no backing file leaves it unformable. Dropping it silently would
// leave a caller whose search matched nothing with no way to learn why.
TEST(KpackResolve, RelativeSearchPathWithoutOwningBinaryIsReported) {
  // The metadata here lives in a heap allocation, whose mapping is anonymous.
  auto metadata = MakeMetadata("lib/x.so", {"sibling.kpack"});
  void* out = nullptr;
  iree_host_size_t out_size = 0;
  iree::Status status(Resolve(metadata, 0, {"gfx900"}, &out, &out_size));
  EXPECT_EQ(status.code(), iree::StatusCode::kFailedPrecondition);
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("sibling.kpack"));
}

// Once archives are opened the search runs to completion, so a miss is a target
// mismatch and that is the answer. A candidate rejected along the way is
// attached as context rather than displacing it: reporting the junk file alone
// would describe a file that has nothing to do with the miss.
TEST(KpackResolve, TargetMismatchOutranksRejectedCandidate) {
  std::vector<uint8_t> garbage(32, 0xAB);  // maps, but not a "KPAK" archive
  std::string p_bad = WriteTempFile("mismatch_bad.kpack", garbage);
  auto elf = MakeMinimalAmdgpuElf();
  auto good = KpackBuilder(false).Add("lib/x.so#0", "gfx900", elf).Build();
  std::string p_good = WriteTempFile("mismatch_good.kpack", good);
  auto metadata = MakeMetadata("lib/x.so", {p_bad, p_good});

  void* out = nullptr;
  iree_host_size_t out_size = 0;
  iree::Status status(Resolve(metadata, 0, {"gfx1100"}, &out, &out_size));
  EXPECT_EQ(status.code(), iree::StatusCode::kNotFound);
  // The explanation the completed search established comes first...
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("matches target 'gfx1100'"));
  // ...and the rejected candidate survives alongside it.
  EXPECT_THAT(status.ToString(), ::testing::HasSubstr("bad magic"));
}

// Only one cause can be returned, so every other one reaches the user on the
// debug channel or not at all. A user who fixes the reported archive and
// re-runs would otherwise meet the next one only after the round trip, one run
// per bad archive. Each line names its own path: "malformed" without a path is
// not something the user can act on.
TEST(KpackResolve, EveryRejectedCandidateIsLogged) {
  std::string first =
      WriteTempFile("logged_first.kpack", std::vector<uint8_t>(32, 0xAB));
  std::string second =
      WriteTempFile("logged_second.kpack", std::vector<uint8_t>(32, 0xCD));
  auto metadata = MakeMetadata("lib/x.so", {"/nonexistent/none.kpack"});

  setenv("ROCM_KPACK_DEBUG", "1", 1);
  setenv("ROCM_KPACK_PATH", JoinPathList({first, second}).c_str(), 1);
  ::testing::internal::CaptureStderr();
  void* out = nullptr;
  iree_host_size_t out_size = 0;
  iree_status_t status = Resolve(metadata, 0, {"gfx900"}, &out, &out_size);
  const std::string log = ::testing::internal::GetCapturedStderr();
  unsetenv("ROCM_KPACK_PATH");
  unsetenv("ROCM_KPACK_DEBUG");

  // The first cause is the one returned; the second exists only in the log.
  EXPECT_THAT(iree::Status(std::move(status)),
              StatusIs(iree::StatusCode::kInvalidArgument));
  EXPECT_THAT(log, ::testing::HasSubstr(first));
  EXPECT_THAT(log, ::testing::HasSubstr(second));
}

// A target the expansion cannot represent fails the whole resolve with
// OUT_OF_RANGE rather than searching a truncated candidate set and reporting an
// unexplained miss. This exercises the rejection end-to-end, including its
// propagation out of the candidate-collection stage.
TEST(KpackResolve, TooManyFeaturesRejected) {
  auto metadata = MakeMetadata("lib/x.so", {"/nonexistent/none.kpack"});
  void* out = nullptr;
  iree_host_size_t out_size = 0;
  iree::Status status(
      Resolve(metadata, 0, {"gfx942:a+:b+:c+:d+:e+"}, &out, &out_size));
  EXPECT_EQ(status.code(), iree::StatusCode::kOutOfRange);
  EXPECT_EQ(out, nullptr);
}

// More compatible target candidates than the ranked list holds keep the
// highest-ranked ones and record that the rest were dropped. The list is ranked
// most-specific-first, so a match on a kept candidate still resolves; when
// nothing matches, the recorded truncation surfaces instead of an unexplained
// not-found. Each distinct bare processor expands to one candidate, so 65 of
// them overrun the list's capacity of 64, and the single unreachable search
// path leaves the truncation as the sole explanation for the miss.
TEST(KpackResolve, TooManyCandidatesReportsTruncationOnMiss) {
  std::vector<std::string> targets;
  for (int i = 0; i < 65; ++i)
    targets.push_back("gfx" + std::to_string(9000 + i));
  auto metadata = MakeMetadata("lib/x.so", {"/nonexistent/none.kpack"});
  void* out = nullptr;
  iree_host_size_t out_size = 0;
  iree::Status status(Resolve(metadata, 0, targets, &out, &out_size));
  EXPECT_EQ(status.code(), iree::StatusCode::kResourceExhausted);
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("compatible target candidates"));
  EXPECT_EQ(out, nullptr);
}

// The candidate cap defends the ranked search, not the resolve: overflowing it
// records a deferred truncation cause but must not fail a resolve that a kept,
// higher-ranked candidate still satisfies. Here 65 distinct processors overrun
// the list of 64, dropping the lowest-ranked fallback and recording the
// truncation, while an archive keyed on the highest-ranked candidate matches.
// The resolve returns that code object and the pending truncation is discarded,
// never promoted to a terminal RESOURCE_EXHAUSTED.
TEST(KpackResolve, CandidateOverflowStillResolvesWhenKeptCandidateMatches) {
  std::vector<std::string> targets;
  for (int i = 0; i < 65; ++i)
    targets.push_back("gfx" + std::to_string(9000 + i));
  auto elf = MakeMinimalAmdgpuElf();
  // The first target expands to the highest-ranked candidate, which is kept;
  // keying the archive on it lands the match on a kept slot, not a dropped one.
  auto archive = KpackBuilder(false).Add("lib/x.so#0", "gfx9000", elf).Build();
  std::string path = WriteTempFile("overflow_hit.kpack", archive);
  auto metadata = MakeMetadata("lib/x.so", {path});

  void* out = nullptr;
  iree_host_size_t out_size = 0;
  IREE_ASSERT_OK(Resolve(metadata, 0, targets, &out, &out_size));
  ASSERT_EQ(out_size, elf.size());
  EXPECT_EQ(0, memcmp(out, elf.data(), elf.size()));
  FreeBuffer(out);
}

// When the candidate list overflows and archives are opened but none holds a
// matching code object, the search still runs to completion, so the miss is a
// target mismatch reported as NOT_FOUND. The recorded truncation is joined onto
// it as context rather than displacing it or being lost, so a caller sees both
// that the search found nothing and that the lowest-ranked candidates past the
// cap were never searched.
TEST(KpackResolve, CandidateOverflowTruncationJoinsMissWhenArchivesSearched) {
  std::vector<std::string> targets;
  for (int i = 0; i < 65; ++i)
    targets.push_back("gfx" + std::to_string(9000 + i));
  // The archive opens and is searched, but holds no arch any candidate asks
  // for.
  auto archive = KpackBuilder(false)
                     .Add("lib/x.so#0", "gfx1100", MakeMinimalAmdgpuElf())
                     .Build();
  std::string path = WriteTempFile("overflow_miss.kpack", archive);
  auto metadata = MakeMetadata("lib/x.so", {path});

  void* out = nullptr;
  iree_host_size_t out_size = 0;
  iree::Status status(Resolve(metadata, 0, targets, &out, &out_size));
  EXPECT_EQ(status.code(), iree::StatusCode::kNotFound);
  EXPECT_THAT(status.ToString(),
              ::testing::HasSubstr("compatible target candidates"));
  EXPECT_EQ(out, nullptr);
}

// A deferred truncation cause is only ever a fallback explanation for a miss; a
// requested target that fails the resolve outright outranks it. Overflowing the
// candidate list records the truncation, then a later target carrying more
// subsettable features than the expansion holds fails collection with
// OUT_OF_RANGE. That terminal failure is what the caller sees, and the recorded
// truncation is discarded rather than surfaced in its place.
TEST(KpackResolve,
     CandidateOverflowTruncationDiscardedWhenLaterTargetRejected) {
  std::vector<std::string> targets;
  for (int i = 0; i < 65; ++i)
    targets.push_back("gfx" + std::to_string(9000 + i));
  targets.push_back("gfx942:a+:b+:c+:d+:e+");  // five features: over the cap
  auto metadata = MakeMetadata("lib/x.so", {"/nonexistent/none.kpack"});

  void* out = nullptr;
  iree_host_size_t out_size = 0;
  iree::Status status(Resolve(metadata, 0, targets, &out, &out_size));
  EXPECT_EQ(status.code(), iree::StatusCode::kOutOfRange);
  EXPECT_EQ(out, nullptr);
}

// A path that simply holds no file is not a member of the search space, so it
// stays a plain NOT_FOUND rather than being reported as a problem.
TEST(KpackResolve, MissingArchiveStaysNotFound) {
  void* out = nullptr;
  iree_host_size_t out_size = 0;
  auto metadata = MakeMetadata("lib/x.so", {"/nonexistent/none.kpack"});
  EXPECT_THAT(iree::Status(Resolve(metadata, 0, {"gfx900"}, &out, &out_size)),
              StatusIs(iree::StatusCode::kNotFound));
}

// A zero-length file is present but cannot be an archive, so it is reported
// rather than skipped as an absent one. Mapping it fails with INVALID_ARGUMENT
// because a zero-length mapping is required to fail (EINVAL); what the rule
// depends on is only that it is not NOT_FOUND.
TEST(KpackResolve, EmptyArchiveReportedWhenNothingMatches) {
  std::string path = WriteTempFile("empty.kpack", {});
  auto metadata = MakeMetadata("lib/x.so", {path});

  void* out = nullptr;
  iree_host_size_t out_size = 0;
  EXPECT_THAT(iree::Status(Resolve(metadata, 0, {"gfx900"}, &out, &out_size)),
              StatusIs(iree::StatusCode::kInvalidArgument));
}

// An archive the process cannot read is reported as such. Without this the user
// has no way to tell a permissions problem from a missing archive.
TEST(KpackResolve, UnreadableArchiveReportsPermissionDenied) {
  if (geteuid() == 0) {
    GTEST_SKIP() << "root bypasses the mode bits this test relies on";
  }
  auto elf = MakeMinimalAmdgpuElf();
  auto archive = KpackBuilder(false).Add("lib/x.so#0", "gfx900", elf).Build();
  std::string path = WriteTempFile("noperm.kpack", archive);
  ASSERT_EQ(chmod(path.c_str(), 0), 0);
  auto metadata = MakeMetadata("lib/x.so", {path});

  void* out = nullptr;
  iree_host_size_t out_size = 0;
  iree_status_t status = Resolve(metadata, 0, {"gfx900"}, &out, &out_size);
  chmod(path.c_str(), 0600);
  EXPECT_THAT(iree::Status(std::move(status)),
              StatusIs(iree::StatusCode::kPermissionDenied));
}

// An unreadable archive is a diagnostic, not a hard stop: one stale file on a
// speculative search path must not break an application that resolves from a
// later path.
TEST(KpackResolve, UnreadableArchiveDoesNotBlockLaterArchive) {
  if (geteuid() == 0) {
    GTEST_SKIP() << "root bypasses the mode bits this test relies on";
  }
  auto elf = MakeMinimalAmdgpuElf();
  auto archive = KpackBuilder(false).Add("lib/x.so#0", "gfx900", elf).Build();
  std::string p_bad = WriteTempFile("noperm_first.kpack", archive);
  ASSERT_EQ(chmod(p_bad.c_str(), 0), 0);
  std::string p_good = WriteTempFile("perm_second.kpack", archive);
  auto metadata = MakeMetadata("lib/x.so", {p_bad, p_good});

  void* out = nullptr;
  iree_host_size_t out_size = 0;
  iree_status_t status = Resolve(metadata, 0, {"gfx900"}, &out, &out_size);
  chmod(p_bad.c_str(), 0600);
  IREE_ASSERT_OK(status);
  EXPECT_EQ(out_size, elf.size());
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

// The arch candidate list is ranked most-specific-first, so a two-feature
// archive key wins over a one-feature key even though the one-feature mask is
// numerically larger. gfx900:a+:b+:c+ expands with gfx900:b+:c+ (two features)
// ahead of gfx900:a+ (one feature), so the two-feature payload is selected.
TEST(KpackResolve, MoreSpecificFeatureSubsetWinsOverLessSpecific) {
  auto two = MakeMinimalAmdgpuElf(0x042);
  auto one = MakeMinimalAmdgpuElf(0x041);
  ASSERT_NE(two, one);  // distinct payloads distinguish which key was selected
  auto archive = KpackBuilder(false)
                     .Add("lib/x.so#0", "gfx900:b+:c+", two)
                     .Add("lib/x.so#0", "gfx900:a+", one)
                     .Build();
  std::string path = WriteTempFile("feature_rank.kpack", archive);
  auto metadata = MakeMetadata("lib/x.so", {path});

  void* out = nullptr;
  iree_host_size_t out_size = 0;
  IREE_ASSERT_OK(Resolve(metadata, 0, {"gfx900:a+:b+:c+"}, &out, &out_size));
  ASSERT_EQ(out_size, two.size());
  EXPECT_EQ(0, memcmp(out, two.data(), two.size()));
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

TEST(KpackResolve, EnvPathPrefix) {
  auto elf = MakeMinimalAmdgpuElf();
  auto archive = KpackBuilder(false).Add("lib/x.so#0", "gfx900", elf).Build();
  std::string path = WriteTempFile("prefix.kpack", archive);

  // The embedded search path does not resolve; ROCM_KPACK_PATH_PREFIX supplies
  // the archive. Unlike ROCM_KPACK_PATH (which replaces the embedded paths),
  // the prefix is prepended, so the archive is reached only if the prefix is
  // consulted.
  auto metadata = MakeMetadata("lib/x.so", {"/nonexistent/none.kpack"});
  setenv("ROCM_KPACK_PATH_PREFIX", path.c_str(), 1);
  void* out = nullptr;
  iree_host_size_t out_size = 0;
  iree_status_t status = Resolve(metadata, 0, {"gfx900"}, &out, &out_size);
  unsetenv("ROCM_KPACK_PATH_PREFIX");
  IREE_ASSERT_OK(status);
  EXPECT_EQ(out_size, elf.size());
  FreeBuffer(out);

  // With a bogus prefix but a resolvable embedded (absolute) path, resolution
  // still succeeds -- confirming the prefix prepends rather than replaces, the
  // distinction from ROCM_KPACK_PATH.
  auto metadata2 = MakeMetadata("lib/x.so", {path});
  setenv("ROCM_KPACK_PATH_PREFIX", "/nonexistent/prefix.kpack", 1);
  out = nullptr;
  out_size = 0;
  status = Resolve(metadata2, 0, {"gfx900"}, &out, &out_size);
  unsetenv("ROCM_KPACK_PATH_PREFIX");
  IREE_ASSERT_OK(status);
  EXPECT_EQ(out_size, elf.size());
  FreeBuffer(out);
}

// Writes |count| valid archives to distinct paths. The one at |match_index|
// holds "lib/x.so#0"; the rest hold an unrelated key, so every archive parses
// and occupies a slot but only one can satisfy the search.
std::vector<std::string> WriteArchiveSet(const std::string& name_prefix,
                                         size_t count, size_t match_index,
                                         const std::vector<uint8_t>& elf) {
  std::vector<std::string> paths;
  paths.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    const std::string key = i == match_index ? "lib/x.so#0" : "lib/other.so#0";
    auto archive = KpackBuilder(false).Add(key, "gfx900", elf).Build();
    paths.push_back(
        WriteTempFile(name_prefix + std::to_string(i) + ".kpack", archive));
  }
  return paths;
}

// The cap counts archives that fit, so exactly that many resolve. The match
// sits in the last one, pinning that every slot is searched.
TEST(KpackResolve, ExactlyMaxArchivesSucceeds) {
  auto elf = MakeMinimalAmdgpuElf();
  const size_t count = IREE_HAL_STREAMING_KPACK_MAX_OPEN_ARCHIVES;
  auto paths =
      WriteArchiveSet("cap_exact_", count, /*match_index=*/count - 1, elf);
  auto metadata = MakeMetadata("lib/x.so", {"/nonexistent/none.kpack"});

  setenv("ROCM_KPACK_PATH", JoinPathList(paths).c_str(), 1);
  void* out = nullptr;
  iree_host_size_t out_size = 0;
  iree_status_t status = Resolve(metadata, 0, {"gfx900"}, &out, &out_size);
  unsetenv("ROCM_KPACK_PATH");
  IREE_ASSERT_OK(status);
  EXPECT_EQ(out_size, elf.size());
  FreeBuffer(out);
}

// One archive past the cap. The match is in the first archive, so a truncated
// search would find it and succeed by luck; because the ranked space cannot be
// evaluated, a match that happens to be reachable is not an honest answer.
TEST(KpackResolve, TooManyArchivesFailsLoudly) {
  auto elf = MakeMinimalAmdgpuElf();
  const size_t count = IREE_HAL_STREAMING_KPACK_MAX_OPEN_ARCHIVES + 1;
  auto paths = WriteArchiveSet("cap_over_", count, /*match_index=*/0, elf);
  auto metadata = MakeMetadata("lib/x.so", {"/nonexistent/none.kpack"});

  setenv("ROCM_KPACK_PATH", JoinPathList(paths).c_str(), 1);
  void* out = nullptr;
  iree_host_size_t out_size = 0;
  iree_status_t status = Resolve(metadata, 0, {"gfx900"}, &out, &out_size);
  unsetenv("ROCM_KPACK_PATH");
  EXPECT_THAT(iree::Status(std::move(status)),
              StatusIs(iree::StatusCode::kResourceExhausted));
  EXPECT_EQ(out, nullptr);
}

// The cap counts archives, not candidate paths. With the archive set already
// exactly full, a further path that holds nothing (or holds a file that is not
// an archive) is still not a member of the search space, so it neither occupies
// a slot nor reports the set as overfull.
TEST(KpackResolve, NonArchivePathsDoNotConsumeArchiveSlots) {
  auto elf = MakeMinimalAmdgpuElf();
  const size_t count = IREE_HAL_STREAMING_KPACK_MAX_OPEN_ARCHIVES;
  auto entries = WriteArchiveSet("slots_", count, /*match_index=*/0, elf);
  entries.push_back("/nonexistent/none.kpack");
  entries.push_back(
      WriteTempFile("slots_garbage.kpack", std::vector<uint8_t>(32, 0xAB)));
  auto metadata = MakeMetadata("lib/x.so", {"/nonexistent/none.kpack"});

  setenv("ROCM_KPACK_PATH", JoinPathList(entries).c_str(), 1);
  void* out = nullptr;
  iree_host_size_t out_size = 0;
  iree_status_t status = Resolve(metadata, 0, {"gfx900"}, &out, &out_size);
  unsetenv("ROCM_KPACK_PATH");
  IREE_ASSERT_OK(status);
  EXPECT_EQ(out_size, elf.size());
  FreeBuffer(out);
}

// One spelling repeated names one archive, and dedup is observable through the
// archive cap: more repetitions than the cap allows still resolve, because they
// occupy one slot between them. Opening each repetition would yield a slot each
// and exceed the cap.
TEST(KpackResolve, RepeatedSearchPathOpensArchiveOnce) {
  auto elf = MakeMinimalAmdgpuElf();
  auto archive = KpackBuilder(false).Add("lib/x.so#0", "gfx900", elf).Build();
  std::string path = WriteTempFile("dedup.kpack", archive);
  std::vector<std::string> entries(
      IREE_HAL_STREAMING_KPACK_MAX_OPEN_ARCHIVES + 1, path);
  auto metadata = MakeMetadata("lib/x.so", {"/nonexistent/none.kpack"});

  setenv("ROCM_KPACK_PATH", JoinPathList(entries).c_str(), 1);
  void* out = nullptr;
  iree_host_size_t out_size = 0;
  iree_status_t status = Resolve(metadata, 0, {"gfx900"}, &out, &out_size);
  unsetenv("ROCM_KPACK_PATH");
  IREE_ASSERT_OK(status);
  EXPECT_EQ(out_size, elf.size());
  FreeBuffer(out);
}

// Two spellings that collapse to one string lexically can still name different
// files, so both must be searched. Collapsing ".." does not commute with
// symlinks: "B/link/.." is not "B" when "link" points elsewhere. Treating the
// collapsed spelling as an identity would drop the second archive as a
// duplicate of the first and lose the code object it holds.
TEST(KpackResolve, LexicallyCollidingSpellingsOfDifferentFilesAreBothSearched) {
  auto elf = MakeMinimalAmdgpuElf();
  // Only the archive under "A" answers the search; the one under "B" parses and
  // occupies a slot, so a dropped "A" surfaces as a miss rather than an error.
  auto match = KpackBuilder(false).Add("lib/x.so#0", "gfx900", elf).Build();
  auto other = KpackBuilder(false).Add("lib/other.so#0", "gfx900", elf).Build();

  // TempDir() may already end in '/'; a doubled separator would leave the
  // spellings differing from their normalized form for a reason unrelated to
  // the collision, masking it.
  std::string base = std::string(::testing::TempDir());
  while (!base.empty() && base.back() == '/') base.pop_back();
  base += "/kpack_collision";
  const std::string directory_a = base + "/A";
  const std::string directory_b = base + "/B";
  ASSERT_TRUE(EnsureDirectory(base));
  ASSERT_TRUE(EnsureDirectory(directory_a));
  ASSERT_TRUE(EnsureDirectory(directory_a + "/sub"));
  ASSERT_TRUE(EnsureDirectory(directory_b));
  WriteFile(directory_a + "/x.kpack", match);
  WriteFile(directory_b + "/x.kpack", other);
  const std::string link = directory_b + "/link";
  unlink(link.c_str());  // may survive an earlier run
  ASSERT_EQ(symlink((directory_a + "/sub").c_str(), link.c_str()), 0);

  // "B/link/.." is "A", so this second spelling physically names "A/x.kpack".
  // A naive lexical ".." collapse would fold it to "B/x.kpack" — the first
  // spelling exactly — yet they are distinct strings naming different files.
  // The resolver preserves "..", so dedup on the verbatim spelling opens and
  // searches both; folding |spelling_a| onto |spelling_b| would lose A's
  // answer.
  const std::string spelling_b = directory_b + "/x.kpack";
  const std::string spelling_a = link + "/../x.kpack";
  ASSERT_NE(spelling_a, spelling_b);
  char* real_a = realpath(spelling_a.c_str(), nullptr);
  char* real_match = realpath((directory_a + "/x.kpack").c_str(), nullptr);
  ASSERT_NE(real_a, nullptr);
  ASSERT_NE(real_match, nullptr);
  EXPECT_STREQ(real_a, real_match);
  free(real_a);
  free(real_match);

  auto metadata = MakeMetadata("lib/x.so", {"/nonexistent/none.kpack"});
  setenv("ROCM_KPACK_PATH", JoinPathList({spelling_b, spelling_a}).c_str(), 1);
  void* out = nullptr;
  iree_host_size_t out_size = 0;
  iree_status_t status = Resolve(metadata, 0, {"gfx900"}, &out, &out_size);
  unsetenv("ROCM_KPACK_PATH");
  IREE_ASSERT_OK(status);
  EXPECT_EQ(out_size, elf.size());
  FreeBuffer(out);
}

// A relative search path stays relative; the open resolves it against the
// process working directory.
TEST(KpackResolve, RelativeEnvPathResolves) {
  auto elf = MakeMinimalAmdgpuElf();
  auto archive = KpackBuilder(false).Add("lib/x.so#0", "gfx900", elf).Build();
  std::string path = WriteTempFile("relative.kpack", archive);
  size_t slash = path.rfind('/');
  std::string directory = path.substr(0, slash);
  std::string basename = path.substr(slash + 1);

  char previous_cwd[4096];
  ASSERT_NE(getcwd(previous_cwd, sizeof(previous_cwd)), nullptr);
  ASSERT_EQ(chdir(directory.c_str()), 0);
  auto metadata = MakeMetadata("lib/x.so", {"/nonexistent/none.kpack"});
  setenv("ROCM_KPACK_PATH", basename.c_str(), 1);
  void* out = nullptr;
  iree_host_size_t out_size = 0;
  iree_status_t status = Resolve(metadata, 0, {"gfx900"}, &out, &out_size);
  unsetenv("ROCM_KPACK_PATH");
  // Restore the working directory before consuming |status|; the restore must
  // run on every path and IREE_ASSERT_OK aborts the test on an error status.
  const int restore_result = chdir(previous_cwd);
  IREE_ASSERT_OK(status);
  EXPECT_EQ(restore_result, 0);
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

// Presents a bare AMDGPU target key the way a HAL device spec hands one to the
// fat-binary extractor, so the HIPK integration tests drive the same target
// validation and key projection that production uses.
static iree_hal_executable_target_t MakeAmdgpuDeviceTarget(
    iree_string_view_t target_key) {
  iree_hal_executable_target_t target = {};
  target.family = IREE_SV("amdgpu");
  target.target_key = target_key;
  target.physical_device_affinity = 1;
  return target;
}

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

  const iree_hal_executable_target_t executable_target =
      MakeAmdgpuDeviceTarget(IREE_SV("gfx1100"));
  const iree_hal_streaming_fat_binary_target_t targets[] = {
      {/*.executable_target=*/&executable_target},
  };
  iree_hal_streaming_fat_binary_extract_t extract = {};
  IREE_ASSERT_OK(iree_hal_streaming_fat_binary_extract_for_targets(
      iree_make_const_byte_span(&header, sizeof(header)),
      IREE_ARRAYSIZE(targets), targets, iree_allocator_system(), &extract));
  ASSERT_EQ(extract.match_count, 1u);
  EXPECT_STREQ(extract.matches[0].code_object_target_key, "gfx1100");
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

  const iree_hal_executable_target_t executable_target =
      MakeAmdgpuDeviceTarget(IREE_SV("gfx942"));  // not in the archive
  const iree_hal_streaming_fat_binary_target_t targets[] = {
      {/*.executable_target=*/&executable_target},
  };
  iree_hal_streaming_fat_binary_extract_t extract = {};
  EXPECT_THAT(
      iree::Status(iree_hal_streaming_fat_binary_extract_for_targets(
          iree_make_const_byte_span(&header, sizeof(header)),
          IREE_ARRAYSIZE(targets), targets, iree_allocator_system(), &extract)),
      StatusIs(iree::StatusCode::kNotFound));
  iree_hal_streaming_fat_binary_extract_reset(&extract);
}

TEST(KpackIntegration, HipkResolvesBundleCodeObject) {
  // The resolved kpack code object is itself a Clang offload bundle; the
  // fat-binary extractor must unpack it to the matching ELF.
  auto elf = MakeMinimalAmdgpuElf(/*machine=*/0x041);  // gfx1100
  auto bundle = MakeBundle({{"hipv4-amdgcn-amd-amdhsa--gfx1100", elf}});
  auto archive =
      KpackBuilder(false).Add("lib/libhip.so#0", "gfx1100", bundle).Build();
  std::string path = WriteTempFile("integration_bundle.kpack", archive);
  auto metadata = MakeMetadata("lib/libhip.so", {path});

  HipFatHeader header = {};
  header.magic = 0x4b504948u;  // "HIPK"
  header.version = 1;
  header.binary = metadata.data();

  const iree_hal_executable_target_t executable_target =
      MakeAmdgpuDeviceTarget(IREE_SV("gfx1100"));
  const iree_hal_streaming_fat_binary_target_t targets[] = {
      {/*.executable_target=*/&executable_target},
  };
  iree_hal_streaming_fat_binary_extract_t extract = {};
  IREE_ASSERT_OK(iree_hal_streaming_fat_binary_extract_for_targets(
      iree_make_const_byte_span(&header, sizeof(header)),
      IREE_ARRAYSIZE(targets), targets, iree_allocator_system(), &extract));
  ASSERT_EQ(extract.match_count, 1u);
  EXPECT_STREQ(extract.matches[0].code_object_target_key, "gfx1100");
  EXPECT_EQ(extract.matches[0].data.data_length, elf.size());
  iree_hal_streaming_fat_binary_extract_reset(&extract);
}

TEST(KpackIntegration, HipkResolvedGarbageRejected) {
  // A resolved object that is neither an ELF nor an offload bundle must be
  // rejected (fail closed), not handed to a code-object loader.
  std::vector<uint8_t> garbage(32, 0xAB);  // no ELF or bundle magic
  auto archive =
      KpackBuilder(false).Add("lib/libhip.so#0", "gfx1100", garbage).Build();
  std::string path = WriteTempFile("integration_garbage.kpack", archive);
  auto metadata = MakeMetadata("lib/libhip.so", {path});

  HipFatHeader header = {};
  header.magic = 0x4b504948u;
  header.version = 1;
  header.binary = metadata.data();

  const iree_hal_executable_target_t executable_target =
      MakeAmdgpuDeviceTarget(IREE_SV("gfx1100"));
  const iree_hal_streaming_fat_binary_target_t targets[] = {
      {/*.executable_target=*/&executable_target},
  };
  iree_hal_streaming_fat_binary_extract_t extract = {};
  EXPECT_THAT(
      iree::Status(iree_hal_streaming_fat_binary_extract_for_targets(
          iree_make_const_byte_span(&header, sizeof(header)),
          IREE_ARRAYSIZE(targets), targets, iree_allocator_system(), &extract)),
      StatusIs(iree::StatusCode::kInvalidArgument));
  iree_hal_streaming_fat_binary_extract_reset(&extract);
}

TEST(KpackIntegration, HipkReservedSelectsCoIndex) {
  // The wrapper's |reserved| field is the multi-TU code-object index; a value
  // of 1 must select the "#1" TOC entry.
  auto elf = MakeMinimalAmdgpuElf(/*machine=*/0x041);  // gfx1100
  auto archive =
      KpackBuilder(false).Add("lib/libhip.so#1", "gfx1100", elf).Build();
  std::string path = WriteTempFile("integration_coindex.kpack", archive);
  auto metadata = MakeMetadata("lib/libhip.so", {path});

  HipFatHeader header = {};
  header.magic = 0x4b504948u;
  header.version = 1;
  header.binary = metadata.data();
  header.reserved = reinterpret_cast<void*>(static_cast<uintptr_t>(1));

  const iree_hal_executable_target_t executable_target =
      MakeAmdgpuDeviceTarget(IREE_SV("gfx1100"));
  const iree_hal_streaming_fat_binary_target_t targets[] = {
      {/*.executable_target=*/&executable_target},
  };
  iree_hal_streaming_fat_binary_extract_t extract = {};
  IREE_ASSERT_OK(iree_hal_streaming_fat_binary_extract_for_targets(
      iree_make_const_byte_span(&header, sizeof(header)),
      IREE_ARRAYSIZE(targets), targets, iree_allocator_system(), &extract));
  ASSERT_EQ(extract.match_count, 1u);
  EXPECT_EQ(extract.matches[0].data.data_length, elf.size());
  iree_hal_streaming_fat_binary_extract_reset(&extract);
}

}  // namespace
