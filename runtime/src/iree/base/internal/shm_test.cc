// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/shm.h"

#include <cstring>
#include <string>
#include <vector>

#if defined(IREE_PLATFORM_WINDOWS)
#include <process.h>
#elif !defined(IREE_PLATFORM_ANDROID)
#include <unistd.h>
#endif

#if !defined(IREE_PLATFORM_WINDOWS) && !defined(IREE_PLATFORM_ANDROID)
#include <sys/mman.h>  // shm_unlink for test cleanup
#endif

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class ShmTest : public ::testing::Test {};

class ShmReservationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    granularity_ = iree_shm_required_size(0);
    IREE_ASSERT_OK(iree_shm_create(nullptr, 4 * granularity_, &source_));
    IREE_ASSERT_OK(iree_shm_reservation_initialize(
        4 * granularity_, 16 * granularity_, granularity_, &reservation_));
  }

  void TearDown() override {
    iree_status_t status = iree_shm_reservation_deinitialize(&reservation_);
    if (iree_status_is_ok(status)) {
      iree_shm_close(&source_);
    }
    IREE_EXPECT_OK(status);
  }

  iree_shm_mapping_update_t Mapping(iree_host_size_t target_offset,
                                    iree_host_size_t source_offset,
                                    iree_host_size_t byte_length) {
    return {target_offset, byte_length, &source_, source_offset,
            IREE_SHM_ACCESS_READ | IREE_SHM_ACCESS_WRITE};
  }

  // Native page unit used by this host.
  iree_host_size_t granularity_ = 0;
  // Original physical storage and independently observable mapping.
  iree_shm_mapping_t source_ = {};
  // Explicit virtual address owner.
  iree_shm_reservation_t reservation_ = {};
};

TEST_F(ShmReservationTest, ReplacesSubrangesWithoutCopyingBacking) {
  EXPECT_EQ(
      reinterpret_cast<uintptr_t>(reservation_.base) % (16 * granularity_), 0u);
  auto* source = static_cast<uint8_t*>(source_.base);
  for (iree_host_size_t i = 0; i < 4; ++i) {
    std::memset(source + i * granularity_, static_cast<int>(i + 1),
                granularity_);
  }
  auto initial = Mapping(0, 0, 4 * granularity_);
  IREE_ASSERT_OK(iree_shm_reservation_update(&reservation_, 1, &initial));

  // Replace the middle of an occupied range and alias one backing page twice.
  const iree_shm_mapping_update_t replacements[] = {
      Mapping(granularity_, 3 * granularity_, granularity_),
      Mapping(2 * granularity_, 0, granularity_),
  };
  IREE_ASSERT_OK(iree_shm_reservation_update(&reservation_, 2, replacements));
  const uint8_t expected[] = {1, 4, 1, 4};
  for (iree_host_size_t i = 0; i < 4; ++i) {
    for (iree_host_size_t j = 0; j < granularity_; ++j) {
      ASSERT_EQ(reservation_.base[i * granularity_ + j], expected[i]);
    }
  }
  reservation_.base[granularity_ + 7] = 91;
  EXPECT_EQ(source[3 * granularity_ + 7], 91);
  EXPECT_EQ(reservation_.base[3 * granularity_ + 7], 91);
  source[13] = 73;
  EXPECT_EQ(reservation_.base[13], 73);
  EXPECT_EQ(reservation_.base[2 * granularity_ + 13], 73);

  // Detach across old mapping boundaries, then reuse the same address range.
  const iree_shm_mapping_update_t detach = {granularity_, 2 * granularity_};
  IREE_ASSERT_OK(iree_shm_reservation_update(&reservation_, 1, &detach));
  auto replacement = Mapping(granularity_, granularity_, 2 * granularity_);
  IREE_ASSERT_OK(iree_shm_reservation_update(&reservation_, 1, &replacement));
  EXPECT_EQ(reservation_.base[granularity_], 2);
  EXPECT_EQ(reservation_.base[2 * granularity_], 3);
  EXPECT_EQ(reservation_.base[13], 73);
  EXPECT_EQ(reservation_.base[3 * granularity_ + 7], 91);
}

TEST_F(ShmReservationTest, ValidatesCompleteVectorBeforeMutation) {
  auto* source = static_cast<uint8_t*>(source_.base);
  std::memset(source, 11, granularity_);
  std::memset(source + granularity_, 22, granularity_);
  auto initial = Mapping(0, 0, granularity_);
  IREE_ASSERT_OK(iree_shm_reservation_update(&reservation_, 1, &initial));
  const iree_shm_mapping_update_t updates[] = {
      Mapping(0, granularity_, granularity_),
      Mapping(granularity_, source_.size, granularity_),
  };
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_shm_reservation_update(&reservation_, 2, updates));
  EXPECT_EQ(reservation_.base[0], 11);
  auto valid = Mapping(0, granularity_, granularity_);
  IREE_ASSERT_OK(iree_shm_reservation_update(&reservation_, 1, &valid));
  EXPECT_EQ(reservation_.base[0], 22);
}

TEST_F(ShmReservationTest, PopulatesUntouchedSubrangesInAnyOrder) {
  auto* source = static_cast<uint8_t*>(source_.base);
  for (iree_host_size_t i = 0; i < 4; ++i) {
    std::memset(source + i * granularity_, static_cast<int>(i + 1),
                granularity_);
  }

  // Split both sides of an untouched placeholder, then its tail, its remaining
  // prefix, and finally the first granule. No earlier mapping prepares a split.
  const iree_host_size_t positions[] = {2, 3, 1, 0};
  for (iree_host_size_t position : positions) {
    auto update = Mapping(position * granularity_,
                          (3 - position) * granularity_, granularity_);
    IREE_ASSERT_OK(iree_shm_reservation_update(&reservation_, 1, &update));
    for (iree_host_size_t i = 0; i < granularity_; ++i) {
      ASSERT_EQ(reservation_.base[position * granularity_ + i], 4 - position);
    }
    reservation_.base[position * granularity_ + 7] = 73;
    EXPECT_EQ(source[(3 - position) * granularity_ + 7], 73);
  }
}

TEST_F(ShmReservationTest, IndependentReservationsShareOriginalStorage) {
  iree_shm_reservation_t other = {};
  IREE_ASSERT_OK(iree_shm_reservation_initialize(2 * granularity_, 0,
                                                 2 * granularity_, &other));
  auto update = Mapping(0, granularity_, 2 * granularity_);
  IREE_ASSERT_OK(iree_shm_reservation_update(&reservation_, 1, &update));
  IREE_ASSERT_OK(iree_shm_reservation_update(&other, 1, &update));
  reservation_.base[5] = 39;
  EXPECT_EQ(other.base[5], 39);
  EXPECT_EQ(static_cast<uint8_t*>(source_.base)[granularity_ + 5], 39);
  IREE_ASSERT_OK(iree_shm_reservation_deinitialize(&reservation_));
  other.base[7] = 42;
  EXPECT_EQ(static_cast<uint8_t*>(source_.base)[granularity_ + 7], 42);
  IREE_ASSERT_OK(iree_shm_reservation_deinitialize(&other));
}

TEST_F(ShmReservationTest, DetachesBackingWhileRetainingAddressesForReuse) {
  const auto original = reservation_;
  for (uint8_t generation : {17, 29}) {
    std::memset(source_.base, generation, source_.size);
    // Leave holes on both sides of a bound interval. Full detach must remove
    // the installed views and preserve ownership of the untouched addresses.
    const auto mapping = Mapping(granularity_, 0, 2 * granularity_);
    IREE_ASSERT_OK(iree_shm_reservation_update(&reservation_, 1, &mapping));
    EXPECT_EQ(reservation_.base[granularity_ + 7], generation);
    reservation_.base[2 * granularity_ + 13] = generation + 1;
    EXPECT_EQ(static_cast<uint8_t*>(source_.base)[granularity_ + 13],
              generation + 1);

    IREE_ASSERT_OK(iree_shm_reservation_detach(&reservation_));
    EXPECT_EQ(reservation_.base, original.base);
    EXPECT_EQ(reservation_.byte_length, original.byte_length);
    EXPECT_EQ(reservation_.granularity, original.granularity);
    EXPECT_EQ(reservation_.allocation_base, original.allocation_base);
    EXPECT_EQ(reservation_.allocation_byte_length,
              original.allocation_byte_length);
    IREE_ASSERT_OK(iree_shm_reservation_detach(&reservation_));

    // Checked removal authorizes releasing this generation's storage while
    // the address owner remains live for the next generation.
    iree_shm_close(&source_);
    IREE_ASSERT_OK(iree_shm_create(nullptr, 4 * granularity_, &source_));
  }
}

TEST_F(ShmReservationTest, CoarseUnitsPreserveShortTailAndNeighbors) {
  IREE_ASSERT_OK(iree_shm_reservation_deinitialize(&reservation_));
  iree_shm_close(&source_);
  IREE_ASSERT_OK(iree_shm_create(nullptr, 9 * granularity_, &source_));
  const iree_host_size_t unit = 4 * granularity_;
  // Cover both a reservation smaller than one unit and a tail following full
  // units. Each initial update spans the complete reservation in one call.
  for (iree_host_size_t page_count : {1, 9}) {
    SCOPED_TRACE(page_count);
    const iree_host_size_t length = page_count * granularity_;
    IREE_ASSERT_OK(
        iree_shm_reservation_initialize(length, 0, unit, &reservation_));
    EXPECT_EQ(reservation_.byte_length, length);
    EXPECT_EQ(reservation_.granularity, unit);
    auto* source = static_cast<uint8_t*>(source_.base);
    for (iree_host_size_t i = 0; i < page_count; ++i) {
      std::memset(source + i * granularity_, static_cast<int>(i + 1),
                  granularity_);
    }
    auto initial = Mapping(0, 0, length);
    IREE_ASSERT_OK(iree_shm_reservation_update(&reservation_, 1, &initial));
    for (iree_host_size_t i = 0; i < length; ++i) {
      ASSERT_EQ(reservation_.base[i], i / granularity_ + 1);
    }
    const iree_host_size_t tail_offset = length - granularity_;
    auto replacement = Mapping(tail_offset, 0, granularity_);
    IREE_ASSERT_OK(iree_shm_reservation_update(&reservation_, 1, &replacement));
    reservation_.base[tail_offset + 7] = 93;
    EXPECT_EQ(source[7], 93);
    if (page_count > 1) {
      // A short interior update remains invalid, including when an earlier
      // valid update in the same batch could otherwise change the neighbor.
      const iree_shm_mapping_update_t invalid[] = {
          Mapping(0, unit, unit),
          Mapping(unit, 0, granularity_),
      };
      IREE_EXPECT_STATUS_IS(
          IREE_STATUS_OUT_OF_RANGE,
          iree_shm_reservation_update(&reservation_, 2, invalid));
      EXPECT_EQ(reservation_.base[7], 93);
      EXPECT_EQ(reservation_.base[unit + 7], 5);
      const auto full_unit = Mapping(0, unit, unit);
      IREE_ASSERT_OK(iree_shm_reservation_update(&reservation_, 1, &full_unit));
      EXPECT_EQ(reservation_.base[7], 5);
      EXPECT_EQ(reservation_.base[tail_offset + 7], 93);
    }
    const iree_shm_mapping_update_t detach = {tail_offset, granularity_};
    IREE_ASSERT_OK(iree_shm_reservation_update(&reservation_, 1, &detach));
    replacement.source_offset = granularity_;
    IREE_ASSERT_OK(iree_shm_reservation_update(&reservation_, 1, &replacement));
    reservation_.base[length - 1] = 71;
    EXPECT_EQ(source[2 * granularity_ - 1], 71);
    if (page_count > 1) {
      EXPECT_EQ(reservation_.base[unit + 7], 5);
    }
    IREE_ASSERT_OK(iree_shm_reservation_detach(&reservation_));
    IREE_ASSERT_OK(iree_shm_reservation_update(&reservation_, 1, &initial));
    IREE_ASSERT_OK(iree_shm_reservation_deinitialize(&reservation_));
  }
}

TEST_F(ShmTest, ShortReservationStillRequiresWholeNativePages) {
  const iree_host_size_t page = iree_shm_required_size(0);
  iree_shm_reservation_t reservation = {.byte_length = 17};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_shm_reservation_initialize(5 * page - 1, 0, 4 * page, &reservation));
  EXPECT_EQ(reservation.byte_length, 17u);
  EXPECT_EQ(reservation.base, nullptr);
}

TEST_F(ShmTest, RequiredSizeZeroReturnsOnePage) {
  iree_host_size_t page = iree_shm_required_size(0);
  EXPECT_GT(page, 0);
  // Must be a power of two (all known page sizes are).
  EXPECT_EQ(page & (page - 1), 0);
}

TEST_F(ShmTest, RequiredSizeOneReturnsOnePage) {
  iree_host_size_t page = iree_shm_required_size(0);
  EXPECT_EQ(iree_shm_required_size(1), page);
}

TEST_F(ShmTest, RequiredSizeExactPageUnchanged) {
  iree_host_size_t page = iree_shm_required_size(0);
  EXPECT_EQ(iree_shm_required_size(page), page);
}

TEST_F(ShmTest, RequiredSizeRoundsUpToNextPage) {
  iree_host_size_t page = iree_shm_required_size(0);
  EXPECT_EQ(iree_shm_required_size(page + 1), page * 2);
}

TEST_F(ShmTest, InvalidHandleSentinel) {
  iree_shm_handle_t handle = IREE_SHM_HANDLE_INVALID;
  EXPECT_FALSE(iree_shm_handle_is_valid(handle));
}

TEST_F(ShmTest, HandleCloseOnInvalid) {
  iree_shm_handle_t handle = IREE_SHM_HANDLE_INVALID;
  // Must not crash.
  iree_shm_handle_close(&handle);
  EXPECT_FALSE(iree_shm_handle_is_valid(handle));
}

TEST_F(ShmTest, HandleCloseOnNull) {
  // Must not crash.
  iree_shm_handle_close(NULL);
}

TEST_F(ShmTest, CloseZeroedMapping) {
  iree_shm_mapping_t mapping;
  memset(&mapping, 0, sizeof(mapping));
  mapping.handle = IREE_SHM_HANDLE_INVALID;
  // Must not crash.
  iree_shm_close(&mapping);
}

TEST_F(ShmTest, CloseNull) {
  // Must not crash.
  iree_shm_close(NULL);
}

TEST_F(ShmTest, CreateZeroSizeFails) {
  iree_shm_mapping_t mapping;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_shm_create(NULL, 0, &mapping));
}

TEST_F(ShmTest, CreateAnonymous) {
  iree_shm_mapping_t mapping;
  IREE_ASSERT_OK(iree_shm_create(NULL, 4096, &mapping));

  EXPECT_NE(mapping.base, nullptr);
  EXPECT_GE(mapping.size, (iree_host_size_t)4096);
  EXPECT_TRUE(iree_shm_handle_is_valid(mapping.handle));

  // Verify read/write works.
  memset(mapping.base, 0xAB, mapping.size);
  EXPECT_EQ(((uint8_t*)mapping.base)[0], 0xAB);
  EXPECT_EQ(((uint8_t*)mapping.base)[mapping.size - 1], 0xAB);

  iree_shm_close(&mapping);

  // After close, mapping is zeroed.
  EXPECT_EQ(mapping.base, nullptr);
  EXPECT_EQ(mapping.size, 0);
  EXPECT_FALSE(iree_shm_handle_is_valid(mapping.handle));
}

TEST_F(ShmTest, CreateAnonymousSubPageSize) {
  // Requesting less than a page should still get a full page.
  iree_shm_mapping_t mapping;
  IREE_ASSERT_OK(iree_shm_create(NULL, 1, &mapping));

  iree_host_size_t page_size = iree_shm_required_size(0);
  EXPECT_GE(mapping.size, page_size);
  EXPECT_NE(mapping.base, nullptr);

  iree_shm_close(&mapping);
}

TEST_F(ShmTest, HandleDup) {
  iree_shm_mapping_t mapping;
  IREE_ASSERT_OK(iree_shm_create(NULL, 4096, &mapping));

  iree_shm_handle_t dup_handle = IREE_SHM_HANDLE_INVALID;
  IREE_ASSERT_OK(iree_shm_handle_dup(mapping.handle, &dup_handle));
  EXPECT_TRUE(iree_shm_handle_is_valid(dup_handle));

  // Duplicated handle is independent; closing the original mapping doesn't
  // invalidate the dup.
  iree_shm_close(&mapping);

  // The dup handle is still valid and can be used to open a mapping.
  iree_shm_handle_close(&dup_handle);
  EXPECT_FALSE(iree_shm_handle_is_valid(dup_handle));
}

TEST_F(ShmTest, HandleDupInvalidFails) {
  iree_shm_handle_t dup_handle;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_shm_handle_dup(IREE_SHM_HANDLE_INVALID, &dup_handle));
}

TEST_F(ShmTest, OpenHandle) {
  // Create a region and write a pattern.
  iree_shm_mapping_t creator;
  IREE_ASSERT_OK(iree_shm_create(NULL, 4096, &creator));
  memset(creator.base, 0xCD, creator.size);

  // Duplicate the handle (simulates passing to another process).
  iree_shm_handle_t shared_handle = IREE_SHM_HANDLE_INVALID;
  IREE_ASSERT_OK(iree_shm_handle_dup(creator.handle, &shared_handle));

  // Open a second mapping from the duplicated handle.
  iree_shm_mapping_t opener;
  IREE_ASSERT_OK(iree_shm_open_handle(shared_handle, creator.size, &opener));
  EXPECT_NE(opener.base, nullptr);
  EXPECT_EQ(opener.size, creator.size);

  // Both mappings must see the same data.
  EXPECT_EQ(memcmp(creator.base, opener.base, creator.size), 0);

  // Write through the opener and verify the creator sees it.
  memset(opener.base, 0xEF, opener.size);
  EXPECT_EQ(((uint8_t*)creator.base)[0], 0xEF);

  iree_shm_handle_close(&shared_handle);
  iree_shm_close(&opener);
  iree_shm_close(&creator);
}

TEST_F(ShmTest, OpenHandleInvalidFails) {
  iree_shm_mapping_t mapping;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_shm_open_handle(IREE_SHM_HANDLE_INVALID, 4096, &mapping));
}

TEST_F(ShmTest, OpenHandleZeroSizeFails) {
  iree_shm_mapping_t creator;
  IREE_ASSERT_OK(iree_shm_create(NULL, 4096, &creator));

  iree_shm_mapping_t opener;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_shm_open_handle(creator.handle, 0, &opener));

  iree_shm_close(&creator);
}

TEST_F(ShmTest, OpenHandleSizeTooLargeFails) {
  // Create a one-page region, then try to open it with a much larger size.
  // POSIX: our fstat check catches this before mmap (avoids SIGBUS).
  // Windows: MapViewOfFile fails with an error for the same case.
  iree_shm_mapping_t creator;
  IREE_ASSERT_OK(iree_shm_create(NULL, 4096, &creator));

  iree_shm_handle_t shared_handle = IREE_SHM_HANDLE_INVALID;
  IREE_ASSERT_OK(iree_shm_handle_dup(creator.handle, &shared_handle));

  iree_shm_mapping_t opener;
  iree_status_t status =
      iree_shm_open_handle(shared_handle, creator.size * 1024, &opener);
  IREE_EXPECT_NOT_OK(status);

  iree_shm_handle_close(&shared_handle);
  iree_shm_close(&creator);
}

// Named shared memory tests.
//
// Android's bionic libc lacks shm_open/shm_unlink, so the named SHM API
// returns UNAVAILABLE there. These tests are only meaningful on platforms
// that support named shared memory.
#if !defined(IREE_PLATFORM_ANDROID)

static std::string MakeTestShmName(const char* suffix) {
  // POSIX shm_open names live in a global namespace and can outlive a failed
  // process. Include the process id so concurrent or previously failed CI jobs
  // do not collide on fixed names.
#if defined(IREE_PLATFORM_WINDOWS)
  return "iree_" + std::to_string(_getpid()) + "_" + suffix;
#else
  return "/iree_" + std::to_string(getpid()) + "_" + suffix;
#endif  // IREE_PLATFORM_WINDOWS
}

class ShmNamedTest : public ShmTest {
 protected:
  void TearDown() override {
    // Clean up any named regions left by failed tests.
    for (const auto& name : names_to_cleanup_) {
#if !defined(IREE_PLATFORM_WINDOWS)
      shm_unlink(name.c_str());
#endif
    }
  }
  void TrackName(const char* name) { names_to_cleanup_.push_back(name); }

 private:
  std::vector<std::string> names_to_cleanup_;
};

TEST_F(ShmNamedTest, CreateAndOpenNamed) {
  std::string name = MakeTestShmName("create_open");
  TrackName(name.c_str());

  iree_shm_mapping_t creator;
  IREE_ASSERT_OK(iree_shm_create_named(iree_make_cstring_view(name.c_str()),
                                       NULL, 4096, &creator));
  EXPECT_NE(creator.base, nullptr);
  EXPECT_GE(creator.size, (iree_host_size_t)4096);

  // Write a pattern.
  memset(creator.base, 0x42, creator.size);

  // Open the same region by name.
  iree_shm_mapping_t opener;
  IREE_ASSERT_OK(iree_shm_open_named(iree_make_cstring_view(name.c_str()),
                                     creator.size, &opener));
  EXPECT_NE(opener.base, nullptr);

  // Both mappings see the same data.
  EXPECT_EQ(memcmp(creator.base, opener.base, creator.size), 0);

  iree_shm_close(&opener);
  iree_shm_close(&creator);

  // Clean up the name on POSIX.
#if !defined(IREE_PLATFORM_WINDOWS)
  shm_unlink(name.c_str());
#endif
}

TEST_F(ShmNamedTest, CreateNamedDuplicateFails) {
  std::string name = MakeTestShmName("dup_fail");
  TrackName(name.c_str());

  iree_shm_mapping_t first;
  IREE_ASSERT_OK(iree_shm_create_named(iree_make_cstring_view(name.c_str()),
                                       NULL, 4096, &first));

  // Creating a second region with the same name must fail.
  iree_shm_mapping_t second;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ALREADY_EXISTS,
      iree_shm_create_named(iree_make_cstring_view(name.c_str()), NULL, 4096,
                            &second));

  iree_shm_close(&first);

#if !defined(IREE_PLATFORM_WINDOWS)
  shm_unlink(name.c_str());
#endif
}

TEST_F(ShmNamedTest, OpenNamedNonexistentFails) {
  std::string name = MakeTestShmName("nonexistent");
  iree_shm_mapping_t mapping;
  iree_status_t status =
      iree_shm_open_named(iree_make_cstring_view(name.c_str()), 4096, &mapping);
  IREE_EXPECT_NOT_OK(status);
}

TEST_F(ShmNamedTest, CreateNamedZeroSizeFails) {
  std::string name = MakeTestShmName("zero_create");
  iree_shm_mapping_t mapping;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_shm_create_named(iree_make_cstring_view(name.c_str()), NULL, 0,
                            &mapping));
}

TEST_F(ShmNamedTest, OpenNamedZeroSizeFails) {
  std::string name = MakeTestShmName("zero_open");
  iree_shm_mapping_t mapping;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_shm_open_named(iree_make_cstring_view(name.c_str()), 0, &mapping));
}

TEST_F(ShmNamedTest, CreateNamedEmptyNameFails) {
  iree_shm_mapping_t mapping;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_shm_create_named(iree_make_string_view("", 0),
                                              NULL, 4096, &mapping));
}

TEST_F(ShmNamedTest, CreateNamedNameTooLongFails) {
  // Build a name one character beyond the platform limit.
  char long_name[IREE_SHM_MAX_NAME_LENGTH + 2];
  long_name[0] = '/';
  memset(long_name + 1, 'x', IREE_SHM_MAX_NAME_LENGTH + 1);
  iree_string_view_t name =
      iree_make_string_view(long_name, IREE_SHM_MAX_NAME_LENGTH + 1);
  iree_shm_mapping_t mapping;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_shm_create_named(name, NULL, 4096, &mapping));
}

TEST_F(ShmNamedTest, CreateNamedMaxLengthSucceeds) {
  // A name at exactly the platform limit must succeed.
  std::string max_name = MakeTestShmName("max_length");
  ASSERT_LT(max_name.size(), IREE_SHM_MAX_NAME_LENGTH);
  max_name.resize(IREE_SHM_MAX_NAME_LENGTH, 'y');
  iree_string_view_t name =
      iree_make_string_view(max_name.data(), IREE_SHM_MAX_NAME_LENGTH);
  TrackName(max_name.c_str());
  iree_shm_mapping_t mapping;
  IREE_ASSERT_OK(iree_shm_create_named(name, NULL, 4096, &mapping));
  EXPECT_NE(mapping.base, nullptr);
  iree_shm_close(&mapping);
#if !defined(IREE_PLATFORM_WINDOWS)
  shm_unlink(max_name.c_str());
#endif
}

#endif  // !IREE_PLATFORM_ANDROID

TEST_F(ShmTest, QuerySealsInitiallyNone) {
  iree_shm_mapping_t mapping;
  IREE_ASSERT_OK(iree_shm_create(NULL, 4096, &mapping));

  iree_shm_seal_flags_t seals = iree_shm_query_seals(&mapping);
#if defined(IREE_PLATFORM_LINUX)
  // Anonymous (memfd) regions get SHRINK|GROW seals during creation.
  EXPECT_TRUE(seals & IREE_SHM_SEAL_SHRINK);
  EXPECT_TRUE(seals & IREE_SHM_SEAL_GROW);
  EXPECT_FALSE(seals & IREE_SHM_SEAL_WRITE);
  EXPECT_FALSE(seals & IREE_SHM_SEAL_SEAL);
#else
  EXPECT_EQ(seals, IREE_SHM_SEAL_NONE);
#endif

  iree_shm_close(&mapping);
}

TEST_F(ShmTest, QuerySealsNullMapping) {
  EXPECT_EQ(iree_shm_query_seals(NULL), IREE_SHM_SEAL_NONE);
}

TEST_F(ShmTest, QuerySealsUnmappedRegion) {
  iree_shm_mapping_t mapping;
  memset(&mapping, 0, sizeof(mapping));
  mapping.handle = IREE_SHM_HANDLE_INVALID;
  EXPECT_EQ(iree_shm_query_seals(&mapping), IREE_SHM_SEAL_NONE);
}

TEST_F(ShmTest, SealNoneFlagsSucceeds) {
  iree_shm_mapping_t mapping;
  IREE_ASSERT_OK(iree_shm_create(NULL, 4096, &mapping));
  IREE_EXPECT_OK(iree_shm_seal(&mapping, IREE_SHM_SEAL_NONE));
  iree_shm_close(&mapping);
}

TEST_F(ShmTest, SealNullMappingFails) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_shm_seal(NULL, IREE_SHM_SEAL_WRITE));
}

TEST_F(ShmTest, SealWrite) {
  iree_shm_mapping_t mapping;
  IREE_ASSERT_OK(iree_shm_create(NULL, 4096, &mapping));

  // Write data before sealing.
  memset(mapping.base, 0xAA, mapping.size);
  EXPECT_EQ(((uint8_t*)mapping.base)[0], 0xAA);

#if defined(IREE_PLATFORM_LINUX)
  // Linux supports sealing on anonymous (memfd) regions.
  IREE_ASSERT_OK(iree_shm_seal(&mapping, IREE_SHM_SEAL_WRITE));
  iree_shm_seal_flags_t seals = iree_shm_query_seals(&mapping);
  EXPECT_TRUE(seals & IREE_SHM_SEAL_WRITE);
  // Data is still readable after sealing.
  EXPECT_EQ(((uint8_t*)mapping.base)[0], 0xAA);
#elif defined(IREE_PLATFORM_APPLE)
  // macOS does not support sealing.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_shm_seal(&mapping, IREE_SHM_SEAL_WRITE));
#elif defined(IREE_PLATFORM_WINDOWS)
  // Windows uses VirtualProtect for write sealing.
  IREE_ASSERT_OK(iree_shm_seal(&mapping, IREE_SHM_SEAL_WRITE));
  iree_shm_seal_flags_t seals = iree_shm_query_seals(&mapping);
  EXPECT_TRUE(seals & IREE_SHM_SEAL_WRITE);
  // Data is still readable after sealing.
  EXPECT_EQ(((uint8_t*)mapping.base)[0], 0xAA);
#endif

  iree_shm_close(&mapping);
}

TEST_F(ShmTest, SealWriteIdempotent) {
  // Sealing the same flag twice must succeed (no-op on second call).
  iree_shm_mapping_t mapping;
  IREE_ASSERT_OK(iree_shm_create(NULL, 4096, &mapping));
  memset(mapping.base, 0xBB, mapping.size);

#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_WINDOWS)
  IREE_ASSERT_OK(iree_shm_seal(&mapping, IREE_SHM_SEAL_WRITE));
  IREE_ASSERT_OK(iree_shm_seal(&mapping, IREE_SHM_SEAL_WRITE));
  EXPECT_TRUE(iree_shm_query_seals(&mapping) & IREE_SHM_SEAL_WRITE);
#endif

  iree_shm_close(&mapping);
}

#if defined(IREE_PLATFORM_LINUX)
TEST_F(ShmTest, SealWriteFailsWithSecondWritableMapping) {
  // When a second writable mapping exists, F_SEAL_WRITE fails with EBUSY.
  // Verify the rollback restores the original mapping so it's still usable.
  iree_shm_mapping_t creator;
  IREE_ASSERT_OK(iree_shm_create(NULL, 4096, &creator));
  memset(creator.base, 0xDD, creator.size);

  // Open a second writable mapping of the same fd.
  iree_shm_handle_t dup_handle = IREE_SHM_HANDLE_INVALID;
  IREE_ASSERT_OK(iree_shm_handle_dup(creator.handle, &dup_handle));
  iree_shm_mapping_t second;
  IREE_ASSERT_OK(iree_shm_open_handle(dup_handle, creator.size, &second));

  // Sealing the creator must fail because the second mapping is writable.
  iree_status_t status = iree_shm_seal(&creator, IREE_SHM_SEAL_WRITE);
  IREE_EXPECT_NOT_OK(status);

  // The creator mapping must still be valid and readable after rollback.
  EXPECT_NE(creator.base, nullptr);
  EXPECT_EQ(((uint8_t*)creator.base)[0], 0xDD);
  EXPECT_TRUE(iree_shm_handle_is_valid(creator.handle));

  // The write seal must NOT have been applied.
  EXPECT_FALSE(iree_shm_query_seals(&creator) & IREE_SHM_SEAL_WRITE);

  iree_shm_handle_close(&dup_handle);
  iree_shm_close(&second);
  iree_shm_close(&creator);
}

TEST_F(ShmTest, SealSealPreventsNewSeals) {
  iree_shm_mapping_t mapping;
  IREE_ASSERT_OK(iree_shm_create(NULL, 4096, &mapping));

  // Apply SEAL_SEAL — no more seals can be added after this.
  IREE_ASSERT_OK(iree_shm_seal(&mapping, IREE_SHM_SEAL_SEAL));
  EXPECT_TRUE(iree_shm_query_seals(&mapping) & IREE_SHM_SEAL_SEAL);

  // Attempting to add SEAL_WRITE after SEAL_SEAL must fail.
  iree_status_t status = iree_shm_seal(&mapping, IREE_SHM_SEAL_WRITE);
  IREE_EXPECT_NOT_OK(status);

  iree_shm_close(&mapping);
}
#endif  // IREE_PLATFORM_LINUX

TEST_F(ShmTest, SealWriteVisibleToSecondMapping) {
  // Seal via the creator, verify the opener sees the sealed data.
  iree_shm_mapping_t creator;
  IREE_ASSERT_OK(iree_shm_create(NULL, 4096, &creator));
  memset(creator.base, 0xCC, creator.size);

  iree_shm_handle_t shared_handle = IREE_SHM_HANDLE_INVALID;
  IREE_ASSERT_OK(iree_shm_handle_dup(creator.handle, &shared_handle));

  iree_shm_mapping_t opener;
  IREE_ASSERT_OK(iree_shm_open_handle(shared_handle, creator.size, &opener));

#if defined(IREE_PLATFORM_LINUX)
  // Both mappings must be made read-only for F_SEAL_WRITE to succeed.
  // Seal via the opener mapping to verify it works from either side.
  // The creator's mapping is still writable, so we need to seal via the
  // creator (which will mprotect the creator's mapping, but the kernel also
  // requires no other writable VMAs — meaning we need to close the opener's
  // writable mapping first, then seal, then reopen as read-only).
  //
  // For simplicity, close the opener, seal via creator, then reopen.
  iree_shm_close(&opener);
  IREE_ASSERT_OK(iree_shm_seal(&creator, IREE_SHM_SEAL_WRITE));

  // Reopen — the new mapping inherits the seal; the kernel won't allow
  // PROT_WRITE since F_SEAL_WRITE is set.
  IREE_ASSERT_OK(iree_shm_open_handle(shared_handle, creator.size, &opener));
  // The sealed data is readable through both mappings.
  EXPECT_EQ(((uint8_t*)creator.base)[0], 0xCC);
  EXPECT_EQ(((uint8_t*)opener.base)[0], 0xCC);
  // Both mappings report the seal.
  EXPECT_TRUE(iree_shm_query_seals(&creator) & IREE_SHM_SEAL_WRITE);
  EXPECT_TRUE(iree_shm_query_seals(&opener) & IREE_SHM_SEAL_WRITE);
#elif defined(IREE_PLATFORM_WINDOWS)
  IREE_ASSERT_OK(iree_shm_seal(&creator, IREE_SHM_SEAL_WRITE));
  EXPECT_TRUE(iree_shm_query_seals(&creator) & IREE_SHM_SEAL_WRITE);
  // On Windows, VirtualProtect is per-view — the opener's view is unaffected.
  EXPECT_EQ(((uint8_t*)opener.base)[0], 0xCC);
#endif

  iree_shm_handle_close(&shared_handle);
  iree_shm_close(&opener);
  iree_shm_close(&creator);
}

TEST_F(ShmTest, WriteReadCoherence) {
  // Write via one mapping, read via another opened from a dup'd handle.
  iree_shm_mapping_t writer;
  IREE_ASSERT_OK(iree_shm_create(NULL, 8192, &writer));

  iree_shm_handle_t reader_handle = IREE_SHM_HANDLE_INVALID;
  IREE_ASSERT_OK(iree_shm_handle_dup(writer.handle, &reader_handle));

  iree_shm_mapping_t reader;
  IREE_ASSERT_OK(iree_shm_open_handle(reader_handle, writer.size, &reader));

  // Write a structured pattern.
  for (iree_host_size_t i = 0; i < writer.size; ++i) {
    ((uint8_t*)writer.base)[i] = (uint8_t)(i & 0xFF);
  }

  // Verify the reader sees it.
  for (iree_host_size_t i = 0; i < reader.size; ++i) {
    ASSERT_EQ(((uint8_t*)reader.base)[i], (uint8_t)(i & 0xFF))
        << "at byte " << i;
  }

  iree_shm_handle_close(&reader_handle);
  iree_shm_close(&reader);
  iree_shm_close(&writer);
}

//===----------------------------------------------------------------------===//
// Placement options tests (huge pages, NUMA, THP)
//===----------------------------------------------------------------------===//

// These tests verify that placement options are accepted and the allocation
// succeeds. The specific backing (huge pages, THP, NUMA node) depends on
// system configuration and privileges; the tests validate the fallback
// behavior: even without huge pages or NUMA support, creation must succeed
// with normal pages.

TEST_F(ShmTest, CreateWithExplicitHugePages) {
  iree_numa_alloc_options_t options = iree_numa_alloc_options_default();
  options.flags = IREE_MEMORY_PLACEMENT_FLAG_EXPLICIT_HUGE_PAGES;
  options.huge_page_size = 2 * 1024 * 1024;  // 2MB.

  // Allocate 4MB (two huge pages).
  iree_shm_mapping_t mapping;
  IREE_ASSERT_OK(iree_shm_create(&options, 4 * 1024 * 1024, &mapping));
  EXPECT_NE(mapping.base, nullptr);
  EXPECT_GE(mapping.size, (iree_host_size_t)(4 * 1024 * 1024));

  // Write and verify a pattern.
  memset(mapping.base, 0xAA, mapping.size);
  EXPECT_EQ(((uint8_t*)mapping.base)[0], 0xAA);
  EXPECT_EQ(((uint8_t*)mapping.base)[mapping.size - 1], 0xAA);

  iree_shm_close(&mapping);
}

TEST_F(ShmTest, CreateWithTransparentHugePages) {
  iree_numa_alloc_options_t options = iree_numa_alloc_options_default();
  options.flags = IREE_MEMORY_PLACEMENT_FLAG_TRANSPARENT_HUGE_PAGES;

  // Allocate 4MB (2x the typical huge page size).
  iree_shm_mapping_t mapping;
  IREE_ASSERT_OK(iree_shm_create(&options, 4 * 1024 * 1024, &mapping));
  EXPECT_NE(mapping.base, nullptr);
  EXPECT_GE(mapping.size, (iree_host_size_t)(4 * 1024 * 1024));

  memset(mapping.base, 0xBB, mapping.size);
  EXPECT_EQ(((uint8_t*)mapping.base)[0], 0xBB);

  iree_shm_close(&mapping);
}

TEST_F(ShmTest, CreateWithNumaNode) {
  iree_numa_alloc_options_t options = iree_numa_alloc_options_default();
  options.node_id = 0;  // Node 0 is always valid.

  iree_shm_mapping_t mapping;
  IREE_ASSERT_OK(iree_shm_create(&options, 4096, &mapping));
  EXPECT_NE(mapping.base, nullptr);

  memset(mapping.base, 0xCC, mapping.size);
  EXPECT_EQ(((uint8_t*)mapping.base)[0], 0xCC);

  iree_shm_close(&mapping);
}

TEST_F(ShmTest, CreateWithAllPlacementOptions) {
  iree_numa_alloc_options_t options = iree_numa_alloc_options_default();
  options.node_id = 0;
  options.flags = IREE_MEMORY_PLACEMENT_FLAG_EXPLICIT_HUGE_PAGES |
                  IREE_MEMORY_PLACEMENT_FLAG_TRANSPARENT_HUGE_PAGES;
  options.huge_page_size = 2 * 1024 * 1024;

  iree_shm_mapping_t mapping;
  IREE_ASSERT_OK(iree_shm_create(&options, 4 * 1024 * 1024, &mapping));
  EXPECT_NE(mapping.base, nullptr);
  EXPECT_GE(mapping.size, (iree_host_size_t)(4 * 1024 * 1024));

  memset(mapping.base, 0xDD, mapping.size);
  EXPECT_EQ(((uint8_t*)mapping.base)[0], 0xDD);

  iree_shm_close(&mapping);
}

TEST_F(ShmTest, CreateNullOptionsUsesDefaults) {
  // NULL options is equivalent to default options (no NUMA, no huge pages).
  iree_shm_mapping_t mapping;
  IREE_ASSERT_OK(iree_shm_create(NULL, 4096, &mapping));
  EXPECT_NE(mapping.base, nullptr);

  memset(mapping.base, 0xEE, mapping.size);
  EXPECT_EQ(((uint8_t*)mapping.base)[0], 0xEE);

  iree_shm_close(&mapping);
}

#if !defined(IREE_PLATFORM_ANDROID)
TEST_F(ShmNamedTest, CreateNamedWithHugePagesFallsBack) {
  // Named SHM on Linux uses shm_open (tmpfs), which doesn't support explicit
  // huge pages. The implementation should silently fall back to THP or normal
  // pages. On Windows, named mappings can support large pages if privileged.
  std::string name = MakeTestShmName("hp_fallback");
  TrackName(name.c_str());

  iree_numa_alloc_options_t options = iree_numa_alloc_options_default();
  options.flags = IREE_MEMORY_PLACEMENT_FLAG_EXPLICIT_HUGE_PAGES;
  options.huge_page_size = 2 * 1024 * 1024;

  iree_shm_mapping_t mapping;
  IREE_ASSERT_OK(iree_shm_create_named(iree_make_cstring_view(name.c_str()),
                                       &options, 4 * 1024 * 1024, &mapping));
  EXPECT_NE(mapping.base, nullptr);
  EXPECT_GE(mapping.size, (iree_host_size_t)(4 * 1024 * 1024));

  memset(mapping.base, 0xFF, mapping.size);
  EXPECT_EQ(((uint8_t*)mapping.base)[0], 0xFF);

  iree_shm_close(&mapping);

#if !defined(IREE_PLATFORM_WINDOWS)
  shm_unlink(name.c_str());
#endif
}
#endif  // !IREE_PLATFORM_ANDROID

}  // namespace
