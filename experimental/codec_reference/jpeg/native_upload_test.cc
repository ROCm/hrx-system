#include "native_upload.h"

#include <array>
#include <cstring>
#include <memory>
#include <vector>

#include "iree/base/internal/debugging.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct UploadBinding {
  // Name of the independently qualified binding, in directory order.
  const char* name;
  // Immutable-input root ownership; other ranges are device output.
  bool input;
  // Exact expected byte extent, including the original binding guards.
  size_t length;
  // Preserved immutable bytes, or null for device-produced results.
  const uint8_t* bytes;
};

struct UploadImage {
  // Source image identity, independent of its ordinal in the cohort.
  const char* name;
  // Byte offset in the frozen compressed allocation.
  size_t offset;
  // Original encoded byte count, without vector-read padding.
  size_t length;
  // First body byte; preparation cannot read at or past this position.
  size_t entropy_start;
};

#include "native_upload_cases.inc"

class NativeUploadTest : public ::testing::Test {
 protected:
  void SetUp() override {
    for (size_t i = 0; i < prepared_.size(); ++i) {
      IREE_ASSERT_OK(jpeg_upload_image_initialize(
          iree_make_const_byte_span(input_compressed + images[i].offset,
                                    images[i].length),
          &prepared_[i]));
    }
  }

  void TearDown() override { jpeg_upload_deinitialize(&upload_); }

  // Retained native header plans, with no copied or decoded image pixels.
  std::array<jpeg_upload_image_t, IREE_ARRAYSIZE(images)> prepared_;
  // Owned packed upload; device output is described but never host-allocated.
  jpeg_upload_t upload_ = {};
};

TEST_F(NativeUploadTest, ExactQualifiedInputsAndDisjointRanges) {
  static_assert(IREE_ARRAYSIZE(bindings) == JPEG_BUFFER_COUNT);
  IREE_ASSERT_OK(jpeg_upload_create(prepared_.size(), prepared_.data(),
                                    iree_allocator_system(), &upload_));
  size_t input_end = sizeof(JpegUploadDirectory);
  size_t output_end = 0;
  for (unsigned i = 0; i < JPEG_BUFFER_COUNT; ++i) {
    SCOPED_TRACE(bindings[i].name);
    EXPECT_EQ(jpeg_upload_is_input(static_cast<JpegBuffer>(i)),
              bindings[i].input);
    ASSERT_EQ(upload_.lengths[i], bindings[i].length);
    EXPECT_EQ(upload_.offsets[i] % 16, 0u);
    EXPECT_EQ(iree_unaligned_load_le_u32(upload_.input.data + i * 4),
              upload_.offsets[i]);
    size_t& end = bindings[i].input ? input_end : output_end;
    ASSERT_GE(upload_.offsets[i], end);
    end = static_cast<size_t>(upload_.offsets[i]) + upload_.lengths[i];
    ASSERT_LE(end, bindings[i].input ? upload_.input.data_length
                                     : upload_.output_length);
    if (bindings[i].input) {
      const uint8_t* actual = upload_.input.data + upload_.offsets[i];
      for (size_t byte = 0; byte < bindings[i].length; ++byte) {
        ASSERT_EQ(actual[byte], bindings[i].bytes[byte]) << "byte " << byte;
      }
    }
  }
  EXPECT_EQ(upload_.input.data_length, iree_host_align(input_end, 16));
  EXPECT_EQ(upload_.output_length, iree_host_align(output_end, 16));
}

TEST_F(NativeUploadTest, PreparationLeavesEntropyUnread) {
  for (const auto& image : images) {
    SCOPED_TRACE(image.name);
    const size_t aligned = iree_host_align(image.entropy_start, 8);
    std::unique_ptr<uint8_t[]> storage(new uint8_t[aligned + image.length]);
    uint8_t* data = storage.get() + aligned - image.entropy_start;
    std::memcpy(data, input_compressed + image.offset, image.entropy_start);
    IREE_ASAN_POISON_MEMORY_REGION(data + image.entropy_start,
                                   image.length - image.entropy_start);
    jpeg_upload_image_t prepared;
    auto status = jpeg_upload_image_initialize(
        iree_make_const_byte_span(data, image.length), &prepared);
    IREE_ASAN_UNPOISON_MEMORY_REGION(data + image.entropy_start,
                                     image.length - image.entropy_start);
    IREE_ASSERT_OK(status);
  }
}

TEST_F(NativeUploadTest, CopiesMalformedBodyAndOwnsOriginalBytes) {
  std::vector<uint8_t> original(input_compressed,
                                input_compressed + images[0].length);
  original.back() = 0xd8;  // Replace EOI with SOI; header remains valid.
  jpeg_upload_image_t prepared;
  IREE_ASSERT_OK(jpeg_upload_image_initialize(
      iree_make_const_byte_span(original.data(), original.size()), &prepared));
  IREE_ASSERT_OK(
      jpeg_upload_create(1, &prepared, iree_allocator_system(), &upload_));
  const std::vector<uint8_t> expected = original;
  std::fill(original.begin(), original.end(), 0);
  original.clear();
  original.shrink_to_fit();
  const uint8_t* copied = upload_.input.data + upload_.offsets[JPEG_COMPRESSED];
  EXPECT_EQ(std::memcmp(copied, expected.data(), expected.size()), 0);
  for (size_t i = expected.size(); i < 4096 + 16; ++i) {
    EXPECT_EQ(copied[i], 0);
  }
}

struct AllocationObserver {
  // Number of attempted allocation operations, excluding frees.
  size_t attempts = 0;
  // Sum of requested bytes; only one immutable root may be allocated.
  size_t bytes = 0;
  // Number of releases made through this owner.
  size_t frees = 0;
  // Whether allocation should fail before taking ownership.
  bool fail = false;

  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* parameters, void** pointer) {
    auto& observer = *static_cast<AllocationObserver*>(self);
    if (command == IREE_ALLOCATOR_COMMAND_FREE) {
      ++observer.frees;
    } else {
      ++observer.attempts;
      observer.bytes +=
          static_cast<const iree_allocator_alloc_params_t*>(parameters)
              ->byte_length;
      if (observer.fail) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED);
      }
    }
    const auto allocator = iree_allocator_system();
    return allocator.ctl(allocator.self, command, parameters, pointer);
  }

  iree_allocator_t allocator() { return {this, Control}; }
};

TEST_F(NativeUploadTest, SingleInputAllocationAndFailureOwnership) {
  AllocationObserver observer;
  IREE_ASSERT_OK(jpeg_upload_create(prepared_.size(), prepared_.data(),
                                    observer.allocator(), &upload_));
  EXPECT_EQ(observer.attempts, 1u);
  EXPECT_EQ(observer.bytes, upload_.input.data_length);
  jpeg_upload_deinitialize(&upload_);
  EXPECT_EQ(observer.frees, 1u);
  jpeg_upload_deinitialize(&upload_);
  EXPECT_EQ(observer.frees, 1u);
  observer.fail = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        jpeg_upload_create(prepared_.size(), prepared_.data(),
                                           observer.allocator(), &upload_));
  EXPECT_EQ(upload_.input.data, nullptr);
  EXPECT_EQ(upload_.input.data_length, 0u);
  EXPECT_EQ(upload_.output_length, 0u);
  jpeg_upload_deinitialize(&upload_);
  EXPECT_EQ(observer.frees, 1u);
}

TEST_F(NativeUploadTest, ExtentAndLaunchBoundsBeforeAllocation) {
  AllocationObserver observer;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      jpeg_upload_create(0, nullptr, observer.allocator(), &upload_));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      jpeg_upload_create(257, nullptr, observer.allocator(), &upload_));
  jpeg_upload_image_t image;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      jpeg_upload_image_initialize(
          iree_make_const_byte_span(nullptr, 1024u * 4096u + 1), &image));
  IREE_ASSERT_OK(jpeg_upload_image_initialize(
      iree_make_const_byte_span(huge_header, sizeof(huge_header)), &image));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      jpeg_upload_create(1, &image, observer.allocator(), &upload_));
  IREE_ASSERT_OK(jpeg_upload_image_initialize(
      iree_make_const_byte_span(tall_header, sizeof(tall_header)), &image));
  std::vector<jpeg_upload_image_t> many(129, image);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        jpeg_upload_create(many.size(), many.data(),
                                           observer.allocator(), &upload_));
  EXPECT_EQ(observer.attempts, 0u);
  EXPECT_EQ(upload_.input.data, nullptr);
}

TEST_F(NativeUploadTest, RestartInputFailsAtImageBoundaryAndPeerSurvives) {
  jpeg_upload_image_t image;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNIMPLEMENTED,
      jpeg_upload_image_initialize(
          iree_make_const_byte_span(restart_header, sizeof(restart_header)),
          &image));
  IREE_ASSERT_OK(
      jpeg_upload_create(1, &prepared_[0], iree_allocator_system(), &upload_));
  EXPECT_GT(upload_.input.data_length, 0u);
}

TEST_F(NativeUploadTest, TileEdgesMaximumInputAndMaximumCohort) {
  // These retain a valid header while varying an opaque body extent. Only the
  // GPU is permitted to decide whether the extended entropy is a valid image.
  for (size_t length : {4095u, 4096u, 4097u, 1024u * 4096u}) {
    SCOPED_TRACE(length);
    std::vector<uint8_t> bytes(length, 0x5a);
    std::memcpy(bytes.data(), input_compressed, images[0].length);
    jpeg_upload_image_t image;
    IREE_ASSERT_OK(jpeg_upload_image_initialize(
        iree_make_const_byte_span(bytes.data(), bytes.size()), &image));
    IREE_ASSERT_OK(
        jpeg_upload_create(1, &image, iree_allocator_system(), &upload_));
    const uint32_t tiles = static_cast<uint32_t>((length + 4095) / 4096);
    const uint8_t* header = upload_.input.data + upload_.offsets[JPEG_HEADERS];
    EXPECT_EQ(iree_unaligned_load_le_u32(header), length);
    EXPECT_EQ(iree_unaligned_load_le_u32(header + 7 * 4), tiles);
    EXPECT_EQ(upload_.lengths[JPEG_COMPRESSED], tiles * 4096 + 16 + 32);
    EXPECT_EQ(upload_.lengths[JPEG_ENTROPY], tiles * 4096 + 160);
    EXPECT_EQ(upload_.lengths[JPEG_JOBS], tiles * 4 * 16 + 32);
    EXPECT_EQ(std::memcmp(upload_.input.data + upload_.offsets[JPEG_COMPRESSED],
                          bytes.data(), bytes.size()),
              0);
    jpeg_upload_deinitialize(&upload_);
  }
  std::vector<jpeg_upload_image_t> cohort(256, prepared_[0]);
  IREE_ASSERT_OK(jpeg_upload_create(cohort.size(), cohort.data(),
                                    iree_allocator_system(), &upload_));
  EXPECT_EQ(upload_.lengths[JPEG_HEADERS], 256u * 72 + 32);
  EXPECT_EQ(upload_.lengths[JPEG_BLOCKS], 256u * 16 + 32);
  EXPECT_EQ(upload_.lengths[JPEG_VALIDITY], 256u * 4 + 32);
  const uint8_t* last =
      upload_.input.data + upload_.offsets[JPEG_HEADERS] + 255 * 72;
  EXPECT_EQ(iree_unaligned_load_le_u32(last + 6 * 4), 255u);
}

}  // namespace
