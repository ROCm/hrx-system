// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/image/hsaco.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

#include "iree/base/alignment.h"
#include "iree/base/byte_sequence.h"
#include "iree/hal/drivers/amd/gpu/image/hip_images.h"
#include "iree/hal/drivers/amd/gpu/image/image.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amd::gpu {
namespace {

// A real immutable source with callback-scoped segment storage. Poisoning that
// storage after each callback catches accidental retention of enumeration
// views.
struct TestSequence {
  // Refcounted sequence base.
  iree_byte_sequence_t base;
  // Owned artifact bytes.
  std::vector<uint8_t> bytes;
  // Zero for a contiguous view; otherwise callback span length.
  size_t segment_length;
  // Number of full enumerations, used to verify batched materialization.
  size_t enumeration_count = 0;
  // Optional caller-owned flag observing final release.
  bool* destroyed = nullptr;

  static void Destroy(iree_byte_sequence_t* base) {
    auto* self = reinterpret_cast<TestSequence*>(base);
    if (self->destroyed) {
      *self->destroyed = true;
    }
    delete self;
  }
  static iree_status_t Enumerate(
      const iree_byte_sequence_t* base,
      iree_byte_sequence_segment_callback_t callback) {
    auto* self =
        const_cast<TestSequence*>(reinterpret_cast<const TestSequence*>(base));
    ++self->enumeration_count;
    if (!self->segment_length) {
      return callback.fn(
          callback.user_data,
          iree_make_const_byte_span(self->bytes.data(), self->bytes.size()));
    }
    std::array<uint8_t, 257> temporary;
    for (size_t offset = 0; offset < self->bytes.size();) {
      size_t length =
          std::min(self->segment_length, self->bytes.size() - offset);
      memcpy(temporary.data(), self->bytes.data() + offset, length);
      IREE_RETURN_IF_ERROR(
          callback.fn(callback.user_data,
                      iree_make_const_byte_span(temporary.data(), length)));
      std::fill(temporary.begin(), temporary.end(), 0xdd);
      offset += length;
    }
    return iree_ok_status();
  }
  static bool Contiguous(const iree_byte_sequence_t* base,
                         iree_const_byte_span_t* out_span) {
    const auto* self = reinterpret_cast<const TestSequence*>(base);
    if (self->segment_length) {
      return false;
    }
    *out_span =
        iree_make_const_byte_span(self->bytes.data(), self->bytes.size());
    return true;
  }
  static TestSequence* Allocate(std::vector<uint8_t> bytes,
                                size_t segment_length = 0,
                                bool* destroyed = nullptr) {
    static const iree_byte_sequence_vtable_t vtable = {
        .destroy = Destroy,
        .enumerate = Enumerate,
        .try_get_contiguous_span = Contiguous,
    };
    auto* self = new TestSequence{};
    self->bytes = std::move(bytes);
    self->segment_length = segment_length;
    self->destroyed = destroyed;
    iree_byte_sequence_initialize(&vtable, self->bytes.size(), &self->base);
    return self;
  }
};

struct Image {
  // Plain initialized image owner.
  iree_hal_amd_gpu_image_t value = {};
  ~Image() { iree_hal_amd_gpu_image_deinitialize(&value); }
};

// Mutable ELF navigation is confined to tests of the serialized boundary. The
// input is always an actual HIP compiler artifact, never a hand-written ELF.
uint16_t Read16(const std::vector<uint8_t>& bytes, size_t offset) {
  return iree_unaligned_load_le_u16(bytes.data() + offset);
}
uint32_t Read32(const std::vector<uint8_t>& bytes, size_t offset) {
  return iree_unaligned_load_le_u32(bytes.data() + offset);
}
uint64_t Read64(const std::vector<uint8_t>& bytes, size_t offset) {
  return iree_unaligned_load_le_u64(bytes.data() + offset);
}
void Write16(std::vector<uint8_t>& bytes, size_t offset, uint16_t value) {
  iree_unaligned_store_le_u16(bytes.data() + offset, value);
}
void Write32(std::vector<uint8_t>& bytes, size_t offset, uint32_t value) {
  iree_unaligned_store_le_u32(bytes.data() + offset, value);
}
void Write64(std::vector<uint8_t>& bytes, size_t offset, uint64_t value) {
  iree_unaligned_store_le_u64(bytes.data() + offset, value);
}

size_t FindProgram(const std::vector<uint8_t>& bytes, uint32_t type) {
  size_t offset = Read64(bytes, 32);
  for (uint16_t i = 0; i < Read16(bytes, 56); ++i, offset += 56) {
    if (Read32(bytes, offset) == type) {
      return offset;
    }
  }
  return 0;
}

size_t FindDynamic(const std::vector<uint8_t>& bytes, uint64_t tag) {
  size_t program = FindProgram(bytes, 2);
  size_t offset = Read64(bytes, program + 8);
  size_t end = offset + Read64(bytes, program + 32);
  for (; offset < end; offset += 16) {
    if (Read64(bytes, offset) == tag) {
      return offset;
    }
  }
  return 0;
}

size_t FileOffset(const std::vector<uint8_t>& bytes, uint64_t address) {
  size_t offset = Read64(bytes, 32);
  for (uint16_t i = 0; i < Read16(bytes, 56); ++i, offset += 56) {
    uint64_t begin = Read64(bytes, offset + 16);
    if (Read32(bytes, offset) == 1 && address >= begin &&
        address - begin < Read64(bytes, offset + 32)) {
      return Read64(bytes, offset + 8) + address - begin;
    }
  }
  return 0;
}

size_t FindSymbol(const std::vector<uint8_t>& bytes, const char* name) {
  size_t strings = FileOffset(bytes, Read64(bytes, FindDynamic(bytes, 5) + 8));
  size_t symbols = FileOffset(bytes, Read64(bytes, FindDynamic(bytes, 6) + 8));
  size_t hash = FileOffset(bytes, Read64(bytes, FindDynamic(bytes, 4) + 8));
  for (uint32_t i = 1; i < Read32(bytes, hash + 4); ++i) {
    size_t offset = symbols + i * 24;
    if (std::strcmp(reinterpret_cast<const char*>(bytes.data() + strings +
                                                  Read32(bytes, offset)),
                    name) == 0) {
      return offset;
    }
  }
  return 0;
}

class HsacoTest : public ::testing::TestWithParam<size_t> {
 protected:
  void SetUp() override {
    if (!iree_hal_amd_gpu_hip_images_size()) {
      GTEST_SKIP() << "HIP compiler fixtures were not built";
    }
    const auto& entry = iree_hal_amd_gpu_hip_images_create()[GetParam()];
    bytes_.assign(entry.data, entry.data + entry.size);
    ASSERT_GT(bytes_.size(), 64u);
    RecordProperty("target", entry.name);
  }

  iree_status_t Initialize(std::vector<uint8_t> bytes, Image* image,
                           size_t segment_length = 0) {
    auto* source = TestSequence::Allocate(std::move(bytes), segment_length);
    iree_status_t status = iree_hal_amd_gpu_hsaco_image_initialize(
        &source->base, iree_allocator_system(), &image->value);
    iree_byte_sequence_release(&source->base);
    return status;
  }

  std::vector<iree_byte_span_t> Map(const Image& image,
                                    std::vector<uint8_t>* storage) {
    storage->assign(image.value.byte_length, 0xa5);
    std::vector<iree_byte_span_t> mappings;
    for (size_t i = 0; i < image.value.segment_count; ++i) {
      const auto& segment = image.value.segments[i];
      mappings.push_back(iree_make_byte_span(storage->data() + segment.offset,
                                             segment.length));
    }
    return mappings;
  }

  std::vector<uint8_t> bytes_;
};

TEST_P(HsacoTest, LoadsRealHipGlobalsAndDescriptors) {
  for (size_t segment_length : {size_t{0}, size_t{7}, size_t{257}}) {
    SCOPED_TRACE(segment_length);
    Image image;
    IREE_ASSERT_OK(Initialize(bytes_, &image, segment_length));
    EXPECT_EQ(image.value.metadata.kernel_count, 2u);
    EXPECT_EQ(image.value.import_count, 0u);
    EXPECT_GT(image.value.relocation_count, 0u);
    EXPECT_GE(image.value.segment_count, 3u);
    EXPECT_GE(image.value.alignment, 256u);
    const auto* value = iree_hal_amd_gpu_image_find_symbol(
        &image.value, IREE_SV("image_value"));
    const auto* pointer = iree_hal_amd_gpu_image_find_symbol(
        &image.value, IREE_SV("image_pointer"));
    const auto* zero =
        iree_hal_amd_gpu_image_find_symbol(&image.value, IREE_SV("image_zero"));
    ASSERT_NE(value, nullptr);
    ASSERT_NE(pointer, nullptr);
    ASSERT_NE(zero, nullptr);
    EXPECT_EQ(value->length, 4u);
    EXPECT_EQ(value->access, IREE_HAL_AMD_GPU_IMAGE_ACCESS_READ |
                                 IREE_HAL_AMD_GPU_IMAGE_ACCESS_WRITE);
    EXPECT_EQ(pointer->length, 8u);
    EXPECT_EQ(pointer->access, value->access);
    EXPECT_EQ(zero->length, 17u * 4);
    EXPECT_EQ(zero->access, value->access);
    EXPECT_EQ(
        iree_hal_amd_gpu_image_find_symbol(&image.value, IREE_SV("absent")),
        nullptr);

    std::vector<uint8_t> storage;
    auto mappings = Map(image, &storage);
    auto* source = reinterpret_cast<TestSequence*>(image.value.source);
    size_t enumeration_count = source->enumeration_count;
    constexpr uint64_t kBase = UINT64_C(0x1200000000);
    IREE_ASSERT_OK(iree_hal_amd_gpu_image_load(
        &image.value, kBase, 0, nullptr, mappings.size(), mappings.data()));
    EXPECT_EQ(source->enumeration_count, enumeration_count + 1);
    EXPECT_EQ(Read32(storage, value->address.offset), 19u);
    EXPECT_EQ(Read64(storage, pointer->address.offset),
              kBase + value->address.offset);
    for (size_t i = 0; i < zero->length; ++i) {
      EXPECT_EQ(storage[zero->address.offset + i], 0u);
    }
    // Linked VA gaps must not be repacked or overwritten by materialization.
    for (size_t i = 1; i < image.value.segment_count; ++i) {
      const auto& previous = image.value.segments[i - 1];
      for (uint64_t j = previous.offset + previous.length;
           j < image.value.segments[i].offset; ++j) {
        ASSERT_EQ(storage[j], 0xa5);
      }
    }
    for (size_t i = 0; i < image.value.metadata.kernel_count; ++i) {
      const auto& kernel = image.value.kernels[i];
      EXPECT_EQ(kernel.descriptor_offset % 64, 0u);
      EXPECT_EQ(kernel.entry_offset % 256, 0u);
      EXPECT_EQ(kernel.descriptor_offset +
                    Read64(storage, kernel.descriptor_offset + 16),
                kernel.entry_offset);
      EXPECT_EQ(Read32(storage, kernel.descriptor_offset + 48),
                kernel.compute_pgm_rsrc1);
    }
  }
}

TEST_P(HsacoTest, SourceOutlivesCallerAndCallbackSpans) {
  bool destroyed = false;
  auto* source = TestSequence::Allocate(bytes_, 3, &destroyed);
  Image image;
  iree_status_t status = iree_hal_amd_gpu_hsaco_image_initialize(
      &source->base, iree_allocator_system(), &image.value);
  iree_byte_sequence_release(&source->base);
  IREE_ASSERT_OK(status);
  EXPECT_FALSE(destroyed);
  EXPECT_NE(
      iree_hal_amd_gpu_image_find_symbol(&image.value, IREE_SV("image_value")),
      nullptr);
  EXPECT_EQ(image.value.metadata.kernel_count, 2u);
  for (size_t i = 0; i < image.value.metadata.kernel_count; ++i) {
    EXPECT_TRUE(iree_string_view_starts_with(
        image.value.metadata.kernels[i].name, IREE_SV("image_")));
  }
  iree_hal_amd_gpu_image_deinitialize(&image.value);
  EXPECT_TRUE(destroyed);
}

TEST_P(HsacoTest, DoesNotRequireSectionHeaders) {
  Write64(bytes_, 40, 0);
  Write16(bytes_, 58, 0);
  Write16(bytes_, 60, 0);
  Write16(bytes_, 62, 0);
  Image image;
  IREE_ASSERT_OK(Initialize(bytes_, &image, 11));
  EXPECT_EQ(image.value.metadata.kernel_count, 2u);
  EXPECT_GT(image.value.relocation_count, 0u);
}

TEST_P(HsacoTest, SupportsGnuHashWithoutSystemVHash) {
  size_t hash = FindDynamic(bytes_, 4);
  ASSERT_NE(hash, 0u);
  ASSERT_NE(FindDynamic(bytes_, 0x6ffffef5), 0u);
  // DT_DEBUG carries no loading requirement and preserves the table extent.
  Write64(bytes_, hash, 21);
  Write64(bytes_, hash + 8, 0);
  Image image;
  IREE_ASSERT_OK(Initialize(bytes_, &image, 5));
  EXPECT_NE(iree_hal_amd_gpu_image_find_symbol(&image.value,
                                               IREE_SV("image_pointer")),
            nullptr);
  EXPECT_EQ(image.value.metadata.kernel_count, 2u);
}

TEST_P(HsacoTest, ValidatesMappingsBeforeWriting) {
  Image image;
  IREE_ASSERT_OK(Initialize(bytes_, &image));
  std::vector<uint8_t> storage;
  auto mappings = Map(image, &storage);
  const auto untouched = storage;
  --mappings.back().data_length;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_amd_gpu_image_load(&image.value, 0x100000, 0, nullptr,
                                  mappings.size(), mappings.data()));
  EXPECT_EQ(storage, untouched);
  ++mappings.back().data_length;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amd_gpu_image_load(&image.value, 1, 0, nullptr, mappings.size(),
                                  mappings.data()));
  EXPECT_EQ(storage, untouched);
}

TEST_P(HsacoTest, RejectsTruncatedArtifactWithEmptyOutput) {
  size_t programs = Read64(bytes_, 32);
  bytes_.resize(programs + 13);
  bool destroyed = false;
  auto* source = TestSequence::Allocate(bytes_, 3, &destroyed);
  Image image;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_amd_gpu_hsaco_image_initialize(
          &source->base, iree_allocator_system(), &image.value));
  EXPECT_EQ(image.value.storage, nullptr);
  EXPECT_EQ(image.value.source, nullptr);
  iree_byte_sequence_release(&source->base);
  EXPECT_TRUE(destroyed);
}

TEST_P(HsacoTest, RejectsOverlappingVirtualSegments) {
  size_t first = FindProgram(bytes_, 1);
  ASSERT_NE(first, 0u);
  size_t second = first + 56;
  ASSERT_EQ(Read32(bytes_, second), 1u);
  Write64(bytes_, second + 16, Read64(bytes_, first + 16));
  Write64(bytes_, second + 48, 1);
  Image image;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Initialize(bytes_, &image));
}

TEST_P(HsacoTest, RejectsStaticRelocationsInDynamicTable) {
  size_t dynamic = FindDynamic(bytes_, 7);
  ASSERT_NE(dynamic, 0u);
  size_t relocations = FileOffset(bytes_, Read64(bytes_, dynamic + 8));
  ASSERT_NE(relocations, 0u);
  const auto original = bytes_;
  // Complete family of static-only, reserved, and unknown encodings.
  for (uint32_t type : {4u, 5u, 7u, 8u, 9u, 10u, 11u, 12u, 14u, 0xffffu}) {
    SCOPED_TRACE(type);
    bytes_ = original;
    Write32(bytes_, relocations + 8, type);
    Image image;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                          Initialize(bytes_, &image));
  }
}

TEST_P(HsacoTest, RejectsNonAddressSymbolSections) {
  const size_t symbol = FindSymbol(bytes_, "image_value");
  ASSERT_NE(symbol, 0u);
  // LDS, OS-specific semantics, COMMON and extended section indices must not
  // accidentally become GPU virtual addresses merely because st_value fits.
  for (uint16_t section : {0xff00u, 0xff20u, 0xfff2u, 0xffffu}) {
    SCOPED_TRACE(section);
    Write16(bytes_, symbol + 6, section);
    Image image;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                          Initialize(bytes_, &image));
  }
}

TEST_P(HsacoTest, RelocatesEveryDynamicAddressEncoding) {
  size_t dynamic = FindDynamic(bytes_, 7);
  ASSERT_NE(dynamic, 0u);
  size_t relocations = FileOffset(bytes_, Read64(bytes_, dynamic + 8));
  ASSERT_NE(relocations, 0u);
  uint64_t target_offset = Read64(bytes_, relocations);
  const auto original = bytes_;
  constexpr uint64_t kValue = UINT64_C(0x123456789abcdef0);
  for (uint32_t type : {1u, 2u, 3u, 6u, 13u}) {
    SCOPED_TRACE(type);
    bytes_ = original;
    // Symbol zero denotes the absolute zero base; the addend supplies the
    // encoded value. RELATIVE64 instead uses the loaded image base.
    Write64(bytes_, relocations + 8, type);
    uint64_t addend = type == 6 ? 0x12345678 : kValue;
    Write64(bytes_, relocations + 16, addend);
    Image image;
    IREE_ASSERT_OK(Initialize(bytes_, &image, 7));
    std::vector<uint8_t> storage;
    auto mappings = Map(image, &storage);
    constexpr uint64_t kBase = 0x100000000ull;
    IREE_ASSERT_OK(iree_hal_amd_gpu_image_load(
        &image.value, kBase, 0, nullptr, mappings.size(), mappings.data()));
    if (type == 3 || type == 13) {
      EXPECT_EQ(Read64(storage, target_offset),
                type == 13 ? kBase + addend : addend);
    } else {
      EXPECT_EQ(Read32(storage, target_offset),
                static_cast<uint32_t>(type == 2 ? addend >> 32 : addend));
    }
  }
}

TEST_P(HsacoTest, NormalizesNonzeroLinkedVirtualBase) {
  Image baseline;
  IREE_ASSERT_OK(Initialize(bytes_, &baseline));
  size_t symbols =
      FileOffset(bytes_, Read64(bytes_, FindDynamic(bytes_, 6) + 8));
  size_t hash = FileOffset(bytes_, Read64(bytes_, FindDynamic(bytes_, 4) + 8));
  uint32_t symbol_count = Read32(bytes_, hash + 4);
  size_t relocations =
      FileOffset(bytes_, Read64(bytes_, FindDynamic(bytes_, 7) + 8));
  size_t relocation_count = Read64(bytes_, FindDynamic(bytes_, 8) + 8) / 24;
  constexpr uint64_t kBias = 0x400000;
  size_t program = Read64(bytes_, 32);
  for (uint16_t i = 0; i < Read16(bytes_, 56); ++i, program += 56) {
    if (Read32(bytes_, program) == 1 || Read64(bytes_, program + 16)) {
      Write64(bytes_, program + 16, Read64(bytes_, program + 16) + kBias);
    }
  }
  for (uint32_t i = 1; i < symbol_count; ++i) {
    size_t symbol = symbols + i * 24;
    uint16_t section = Read16(bytes_, symbol + 6);
    if (section && section != 0xfff1) {
      Write64(bytes_, symbol + 8, Read64(bytes_, symbol + 8) + kBias);
    }
  }
  for (size_t i = 0; i < relocation_count; ++i) {
    size_t relocation = relocations + i * 24;
    Write64(bytes_, relocation, Read64(bytes_, relocation) + kBias);
    if (Read32(bytes_, relocation + 8) == 13) {
      Write64(bytes_, relocation + 16, Read64(bytes_, relocation + 16) + kBias);
    }
  }
  for (uint64_t tag : {4u, 5u, 6u, 7u, 0x6ffffef5u}) {
    size_t entry = FindDynamic(bytes_, tag);
    ASSERT_NE(entry, 0u);
    Write64(bytes_, entry + 8, Read64(bytes_, entry + 8) + kBias);
  }
  Image image;
  IREE_ASSERT_OK(Initialize(bytes_, &image, 7));
  EXPECT_EQ(image.value.byte_length, baseline.value.byte_length);
  EXPECT_EQ(image.value.segment_count, baseline.value.segment_count);
  for (size_t i = 0; i < image.value.segment_count; ++i) {
    EXPECT_EQ(image.value.segments[i].offset,
              baseline.value.segments[i].offset);
  }
  std::vector<uint8_t> storage;
  auto mappings = Map(image, &storage);
  constexpr uint64_t kBase = 0x100000000ull;
  IREE_ASSERT_OK(iree_hal_amd_gpu_image_load(&image.value, kBase, 0, nullptr,
                                             mappings.size(), mappings.data()));
  const auto* value =
      iree_hal_amd_gpu_image_find_symbol(&image.value, IREE_SV("image_value"));
  const auto* pointer = iree_hal_amd_gpu_image_find_symbol(
      &image.value, IREE_SV("image_pointer"));
  ASSERT_NE(value, nullptr);
  ASSERT_NE(pointer, nullptr);
  EXPECT_EQ(Read32(storage, value->address.offset), 19u);
  EXPECT_EQ(Read64(storage, pointer->address.offset),
            kBase + value->address.offset);
}

TEST_P(HsacoTest, RejectsOverlappingRelocationFields) {
  size_t relocation_size = FindDynamic(bytes_, 8);
  ASSERT_GE(Read64(bytes_, relocation_size + 8), 48u);
  size_t relocations =
      FileOffset(bytes_, Read64(bytes_, FindDynamic(bytes_, 7) + 8));
  Write64(bytes_, relocations + 24, Read64(bytes_, relocations) + 4);
  Image image;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Initialize(bytes_, &image));
}

TEST_P(HsacoTest, RejectsNarrowRelocationOverflowBeforeWrites) {
  size_t relocations =
      FileOffset(bytes_, Read64(bytes_, FindDynamic(bytes_, 7) + 8));
  Write64(bytes_, relocations + 8, 6);
  Write64(bytes_, relocations + 16, UINT64_C(1) << 32);
  Image image;
  IREE_ASSERT_OK(Initialize(bytes_, &image));
  std::vector<uint8_t> storage;
  auto mappings = Map(image, &storage);
  const auto untouched = storage;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_amd_gpu_image_load(&image.value, 0x100000000ull, 0, nullptr,
                                  mappings.size(), mappings.data()));
  EXPECT_EQ(storage, untouched);
}

TEST_P(HsacoTest, ResolvesExplicitStrongAndWeakImports) {
  size_t symbol_offset = FindSymbol(bytes_, "image_value");
  ASSERT_NE(symbol_offset, 0u);
  Write16(bytes_, symbol_offset + 6, 0);  // SHN_UNDEF.
  Write64(bytes_, symbol_offset + 8, 0);
  Write64(bytes_, symbol_offset + 16, 0);
  for (bool weak : {false, true}) {
    SCOPED_TRACE(weak);
    bytes_[symbol_offset + 4] = weak ? 0x21 : 0x11;
    Image image;
    IREE_ASSERT_OK(Initialize(bytes_, &image, 5));
    ASSERT_EQ(image.value.import_count, 1u);
    EXPECT_TRUE(iree_string_view_equal(image.value.imports[0].name,
                                       IREE_SV("image_value")));
    EXPECT_EQ(image.value.imports[0].weak, weak);
    const auto* pointer = iree_hal_amd_gpu_image_find_symbol(
        &image.value, IREE_SV("image_pointer"));
    ASSERT_NE(pointer, nullptr);
    std::vector<uint8_t> storage;
    auto mappings = Map(image, &storage);
    const auto untouched = storage;
    iree_hal_amd_gpu_image_import_binding_t binding = {};
    auto load = [&]() {
      return iree_hal_amd_gpu_image_load(&image.value, 0x100000000ull, 1,
                                         &binding, mappings.size(),
                                         mappings.data());
    };
    if (weak) {
      IREE_ASSERT_OK(load());
      EXPECT_EQ(Read64(storage, pointer->address.offset), 0u);
    } else {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, load());
      EXPECT_EQ(storage, untouched);
    }
    binding = {.address = 0x1234567812345678ull, .resolved = true};
    IREE_ASSERT_OK(load());
    EXPECT_EQ(Read64(storage, pointer->address.offset), binding.address);
    binding.address = 0;
    IREE_ASSERT_OK(load());
    EXPECT_EQ(Read64(storage, pointer->address.offset), 0u);
  }
}

TEST_P(HsacoTest, RejectsUnsupportedLoaderServices) {
  const auto original = bytes_;
  size_t slot = FindDynamic(bytes_, 4);
  ASSERT_NE(slot, 0u);
  // Host init/fini, implicit addends, PLT, TLS, and packed relocations cannot
  // silently disappear merely because ordinary kernels do not use them.
  for (uint64_t tag :
       {12u, 13u, 17u, 18u, 19u, 20u, 23u, 25u, 26u, 32u, 35u, 36u, 37u}) {
    SCOPED_TRACE(tag);
    bytes_ = original;
    Write64(bytes_, slot, tag);
    Image image;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                          Initialize(bytes_, &image));
  }
}

TEST_P(HsacoTest, RollbackOwnsEveryIntermediateAllocation) {
  struct Allocator {
    // Next allocation ordinal in this admission attempt.
    size_t next = 0;
    // Allocation ordinal refused by the allocator, or SIZE_MAX for none.
    size_t fail_at;
    // Number of allocations still owned by the image or reader.
    size_t live = 0;
    static iree_status_t Control(void* self, iree_allocator_command_t command,
                                 const void* parameters, void** pointer) {
      auto* state = static_cast<Allocator*>(self);
      if (command != IREE_ALLOCATOR_COMMAND_FREE &&
          state->next++ == state->fail_at) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "test allocation capacity exhausted");
      }
      if (command == IREE_ALLOCATOR_COMMAND_FREE) {
        if (*pointer) {
          --state->live;
        }
      }
      const auto allocator = iree_allocator_system();
      iree_status_t status =
          allocator.ctl(allocator.self, command, parameters, pointer);
      if (command != IREE_ALLOCATOR_COMMAND_FREE && iree_status_is_ok(status)) {
        ++state->live;
      }
      return status;
    }
  };
  for (size_t segment_length : {size_t{0}, size_t{7}}) {
    size_t allocation_count = 0;
    {
      Allocator state = {.fail_at = SIZE_MAX};
      auto* source = TestSequence::Allocate(bytes_, segment_length);
      iree_hal_amd_gpu_image_t image = {};
      auto status = iree_hal_amd_gpu_hsaco_image_initialize(
          &source->base, {&state, Allocator::Control}, &image);
      iree_byte_sequence_release(&source->base);
      IREE_ASSERT_OK(status);
      allocation_count = state.next;
      iree_hal_amd_gpu_image_deinitialize(&image);
      EXPECT_EQ(state.live, 0u);
    }
    for (size_t failure = 0; failure < allocation_count; ++failure) {
      SCOPED_TRACE(failure);
      Allocator state = {.fail_at = failure};
      auto* source = TestSequence::Allocate(bytes_, segment_length);
      iree_hal_amd_gpu_image_t image = {};
      auto status = iree_hal_amd_gpu_hsaco_image_initialize(
          &source->base, {&state, Allocator::Control}, &image);
      iree_byte_sequence_release(&source->base);
      IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
      EXPECT_EQ(image.source, nullptr);
      EXPECT_EQ(image.storage, nullptr);
      EXPECT_EQ(state.live, 0u);
      iree_hal_amd_gpu_image_deinitialize(&image);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    HipCompilerArtifacts, HsacoTest,
    ::testing::Range(size_t{0},
                     std::max(size_t{1}, iree_hal_amd_gpu_hip_images_size())));

}  // namespace
}  // namespace iree::hal::amd::gpu
