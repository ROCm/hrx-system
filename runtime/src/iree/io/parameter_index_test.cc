// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/io/parameter_index.h"

#include <array>
#include <string>
#include <thread>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace {

static iree_status_t AddEntry(iree_io_parameter_index_t* index,
                              iree_string_view_t key, uint64_t length) {
  iree_io_parameter_index_entry_t entry = {
      .key = key,
      .length = length,
      .type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT,
      .storage = {.splat = {.pattern_length = 1}},
  };
  return iree_io_parameter_index_add(index, &entry);
}

static std::string Key(size_t i) {
  return "layers." + std::to_string(i / 256) + ".weights." + std::to_string(i);
}

class ParameterIndexTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(
        iree_io_parameter_index_create(iree_allocator_system(), &index_));
  }
  void TearDown() override { iree_io_parameter_index_release(index_); }

  // Index owned by this fixture until all readers have finished.
  iree_io_parameter_index_t* index_ = nullptr;
};

TEST_F(ParameterIndexTest, EmptyAndReserved) {
  const iree_io_parameter_index_entry_t* entry = nullptr;
  EXPECT_EQ(iree_io_parameter_index_count(index_), 0);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      iree_io_parameter_index_lookup(index_, IREE_SV("key"), &entry));
  EXPECT_EQ(entry, nullptr);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_io_parameter_index_get(index_, 0, &entry));
  IREE_ASSERT_OK(iree_io_parameter_index_reserve(index_, 1024));
  EXPECT_EQ(iree_io_parameter_index_count(index_), 0);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      iree_io_parameter_index_lookup(index_, IREE_SV("key"), &entry));
  IREE_ASSERT_OK(AddEntry(index_, IREE_SV("key"), 42));
  IREE_ASSERT_OK(
      iree_io_parameter_index_lookup(index_, IREE_SV("key"), &entry));
  EXPECT_EQ(entry->length, 42);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_io_parameter_index_reserve(index_, SIZE_MAX));
  const iree_io_parameter_index_entry_t* after = nullptr;
  IREE_ASSERT_OK(iree_io_parameter_index_get(index_, 0, &after));
  EXPECT_EQ(entry, after);
}

TEST_F(ParameterIndexTest, BinaryKeysAndCopiedMetadata) {
  char key[] = {'a', '\0', 'b', '\xFF'};
  uint8_t metadata[] = {1, 2, 3};
  iree_io_parameter_index_entry_t source = {
      .key = iree_make_string_view(key, sizeof(key)),
      .metadata = iree_make_const_byte_span(metadata, sizeof(metadata)),
      .length = 17,
      .type = IREE_IO_PARAMETER_INDEX_ENTRY_STORAGE_TYPE_SPLAT,
      .storage = {.splat = {.pattern_length = 1, .pattern = {0xAB}}},
  };
  IREE_ASSERT_OK(iree_io_parameter_index_add(index_, &source));
  key[0] = 'z';
  metadata[0] = 99;
  IREE_ASSERT_OK(AddEntry(index_, iree_string_view_empty(), 18));
  IREE_ASSERT_OK(AddEntry(index_, IREE_SV("a"), 19));

  const iree_io_parameter_index_entry_t* entry = nullptr;
  IREE_ASSERT_OK(iree_io_parameter_index_lookup(
      index_, iree_make_string_view("a\0b\xFF", 4), &entry));
  EXPECT_EQ(entry->length, 17);
  ASSERT_EQ(entry->metadata.data_length, sizeof(metadata));
  EXPECT_EQ(entry->metadata.data[0], 1);
  EXPECT_EQ(entry->storage.splat.pattern[0], 0xAB);
  IREE_ASSERT_OK(
      iree_io_parameter_index_lookup(index_, iree_string_view_empty(), &entry));
  EXPECT_EQ(entry->length, 18);
  IREE_ASSERT_OK(iree_io_parameter_index_lookup(index_, IREE_SV("a"), &entry));
  EXPECT_EQ(entry->length, 19);
}

TEST_F(ParameterIndexTest, GrowthPreservesPointersOrderAndFirstDuplicate) {
  std::vector<const iree_io_parameter_index_entry_t*> entries;
  for (size_t i = 0; i < 8192; ++i) {
    std::string key = Key(i);
    IREE_ASSERT_OK(
        AddEntry(index_, iree_make_string_view(key.data(), key.size()), i));
    const iree_io_parameter_index_entry_t* entry = nullptr;
    IREE_ASSERT_OK(iree_io_parameter_index_get(index_, i * 2, &entry));
    entries.push_back(entry);
    IREE_ASSERT_OK(
        AddEntry(index_, iree_make_string_view(key.data(), key.size()), i + 1));
  }
  IREE_ASSERT_OK(iree_io_parameter_index_reserve(index_, 65536));
  EXPECT_EQ(iree_io_parameter_index_count(index_), entries.size() * 2);
  for (size_t i = entries.size(); i > 0; --i) {
    size_t ordinal = i - 1;
    std::string key = Key(ordinal);
    const iree_io_parameter_index_entry_t* entry = nullptr;
    IREE_ASSERT_OK(iree_io_parameter_index_lookup(
        index_, iree_make_string_view(key.data(), key.size()), &entry));
    EXPECT_EQ(entry, entries[ordinal]);
    EXPECT_EQ(entry->length, ordinal);
    IREE_ASSERT_OK(iree_io_parameter_index_get(index_, ordinal * 2, &entry));
    EXPECT_EQ(entry, entries[ordinal]);
    IREE_ASSERT_OK(
        iree_io_parameter_index_get(index_, ordinal * 2 + 1, &entry));
    EXPECT_EQ(entry->length, ordinal + 1);
  }
  const iree_io_parameter_index_entry_t* missing = entries.front();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_NOT_FOUND,
      iree_io_parameter_index_lookup(
          index_, IREE_SV("layers.0.weights.missing"), &missing));
  EXPECT_EQ(missing, nullptr);
}

TEST_F(ParameterIndexTest, ConcurrentInsertionAndLookup) {
  IREE_ASSERT_OK(AddEntry(index_, IREE_SV("anchor"), 123));
  const iree_io_parameter_index_entry_t* anchor = nullptr;
  IREE_ASSERT_OK(
      iree_io_parameter_index_lookup(index_, IREE_SV("anchor"), &anchor));
  auto reader = [this, anchor] {
    for (size_t i = 0; i < 8192; ++i) {
      const iree_io_parameter_index_entry_t* entry = nullptr;
      IREE_ASSERT_OK(
          iree_io_parameter_index_lookup(index_, IREE_SV("anchor"), &entry));
      EXPECT_EQ(entry, anchor);
      EXPECT_EQ(entry->length, 123);
    }
  };
  std::array<std::thread, 2> readers = {std::thread(reader),
                                        std::thread(reader)};
  for (size_t i = 0; i < 8192; ++i) {
    std::string key = Key(i);
    IREE_EXPECT_OK(
        AddEntry(index_, iree_make_string_view(key.data(), key.size()), i));
  }
  for (auto& thread : readers) {
    thread.join();
  }
  EXPECT_EQ(iree_io_parameter_index_count(index_), 8193);
}

struct FailingAllocator {
  // Successful allocation commands remaining before the next command fails.
  // SIZE_MAX disables failure injection.
  size_t remaining = SIZE_MAX;

  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* parameters, void** inout_pointer) {
    auto* allocator = static_cast<FailingAllocator*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE &&
        allocator->remaining != SIZE_MAX) {
      if (!allocator->remaining) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "injected allocation failure");
      }
      --allocator->remaining;
    }
    auto delegate = iree_allocator_system();
    return delegate.ctl(delegate.self, command, parameters, inout_pointer);
  }
};

TEST(ParameterIndexFailureTest, FailedGrowthPreservesPublishedEntries) {
  for (bool reserve : {false, true}) {
    for (size_t prefix_count : {0, 16, 32}) {
      bool succeeded = false;
      for (size_t failure_ordinal = 0; !succeeded; ++failure_ordinal) {
        FailingAllocator allocator;
        iree_io_parameter_index_t* index = nullptr;
        IREE_ASSERT_OK(iree_io_parameter_index_create(
            {&allocator, FailingAllocator::Control}, &index));
        for (size_t i = 0; i < prefix_count; ++i) {
          std::string key = Key(i);
          IREE_ASSERT_OK(AddEntry(
              index, iree_make_string_view(key.data(), key.size()), i));
        }
        allocator.remaining = failure_ordinal;
        iree_status_t status =
            reserve ? iree_io_parameter_index_reserve(index, 1024)
                    : AddEntry(index, IREE_SV("new"), 1234);
        succeeded = iree_status_is_ok(status);
        if (!succeeded) {
          IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
        }
        EXPECT_EQ(iree_io_parameter_index_count(index),
                  prefix_count + (succeeded && !reserve ? 1 : 0));
        for (size_t i = 0; i < prefix_count; ++i) {
          std::string key = Key(i);
          const iree_io_parameter_index_entry_t* entry = nullptr;
          IREE_ASSERT_OK(iree_io_parameter_index_lookup(
              index, iree_make_string_view(key.data(), key.size()), &entry));
          EXPECT_EQ(entry->length, i);
        }
        allocator.remaining = SIZE_MAX;
        if (!succeeded || reserve) {
          IREE_ASSERT_OK(AddEntry(index, IREE_SV("new"), 1234));
        }
        const iree_io_parameter_index_entry_t* entry = nullptr;
        IREE_ASSERT_OK(
            iree_io_parameter_index_lookup(index, IREE_SV("new"), &entry));
        EXPECT_EQ(entry->length, 1234);
        iree_io_parameter_index_release(index);
      }
    }
  }
}

}  // namespace
}  // namespace iree
