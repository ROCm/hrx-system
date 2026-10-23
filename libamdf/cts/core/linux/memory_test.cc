// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <linux/mempolicy.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <climits>
#include <cstring>
#include <vector>

#include "amdf/amdf.h"
#include "gtest/gtest.h"
#include "util/device_cache.h"
#include "util/mapped_memory.h"
#include "util/provider.h"

namespace {

class HostNumaMemoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(amdf_cts_provider_query_api()(AMDF_ABI_VERSION_LATEST,
                                            AMDF_ABI_VERSION_LATEST, &api_),
              AMDF_STATUS_OK);
    ASSERT_EQ(GetCtsDeviceCache().GetInstance(&instance_), AMDF_STATUS_OK);
    uint32_t count = 0;
    ASSERT_EQ(amdf_status_code(api_->instance_enumerate_memory_scopes(
                  instance_, 0, nullptr, &count)),
              AMDF_STATUS_CODE_BUFFER_TOO_SMALL);
    std::vector<amdf_memory_scope_t*> scopes(count);
    ASSERT_EQ(api_->instance_enumerate_memory_scopes(instance_, count,
                                                     scopes.data(), &count),
              AMDF_STATUS_OK);
    for (auto* scope : scopes) {
      amdf_memory_scope_info_t info = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
          .structure_size = sizeof(info),
      };
      ASSERT_EQ(api_->memory_scope_query_info(scope, &info), AMDF_STATUS_OK);
      if (info.kind != AMDF_MEMORY_SCOPE_KIND_SYSTEM) {
        continue;
      }
      for (uint32_t ordinal = 0; ordinal < info.memory_profile_count;
           ++ordinal) {
        amdf_memory_profile_t profile = {
            .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
            .structure_size = sizeof(profile),
        };
        const auto status = api_->memory_scope_query_device_profile(
            scope, ordinal, 0, nullptr, &profile, nullptr);
        if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
          continue;
        }
        ASSERT_EQ(status, AMDF_STATUS_OK);
        if ((profile.roles & AMDF_MEMORY_PROFILE_ROLE_CREATE) != 0 &&
            (profile.allocation.placement_kinds &
             (UINT32_C(1) << AMDF_MEMORY_PLACEMENT_KIND_HOST_NUMA)) != 0) {
          scope_ = scope;
          profile_ = profile;
          break;
        }
      }
      if (scope_ != nullptr) {
        break;
      }
    }
    ASSERT_NE(scope_, nullptr);
    const long page_size = sysconf(_SC_PAGESIZE);
    ASSERT_GT(page_size, 0);
    page_size_ = static_cast<size_t>(page_size);
  }

  void TearDown() override {
    if (alias_ != nullptr) {
      ASSERT_EQ(api_->host_mapping_destroy(alias_), AMDF_STATUS_OK);
    }
    if (reservation_ != nullptr) {
      ASSERT_EQ(api_->memory_reservation_destroy(reservation_), AMDF_STATUS_OK);
    }
    EXPECT_TRUE(memory_.Release(api_));
  }

  void ExpectPlacement(void* pointer, uint32_t expected_node) {
    for (size_t offset = 0; offset < memory_.info.byte_length;
         offset += page_size_) {
      int node = -1;
      ASSERT_EQ(syscall(SYS_get_mempolicy, &node, nullptr, 0,
                        static_cast<uint8_t*>(pointer) + offset,
                        MPOL_F_NODE | MPOL_F_ADDR),
                0)
          << std::strerror(errno);
      EXPECT_EQ(node, static_cast<int>(expected_node))
          << "page offset " << offset;
    }
  }

  // Public provider and instance borrowed from the corpus cache.
  const amdf_api_t* api_ = nullptr;
  // CPU-only construction never opens or creates an accelerator.
  amdf_instance_t* instance_ = nullptr;
  // Borrowed scope admitting the selected platform allocation profile.
  amdf_memory_scope_t* scope_ = nullptr;
  // Exact allocation capabilities selected through public queries.
  amdf_memory_profile_t profile_ = {};
  // OS page size used to inspect every physical page.
  size_t page_size_ = 0;
  // Owned storage retained until the alias has been removed.
  CtsMappedMemory memory_;
  // Optional host virtual range borrowing the allocation's remapping source.
  amdf_memory_reservation_t* reservation_ = nullptr;
  // Optional CPU view of the virtual range.
  amdf_host_mapping_t* alias_ = nullptr;
};

TEST_F(HostNumaMemoryTest, PlacesEveryPageAndPreservesHostAliases) {
  constexpr size_t kWordBits = sizeof(unsigned long) * CHAR_BIT;
  std::vector<unsigned long> allowed(page_size_ / sizeof(unsigned long));
  const size_t node_count = allowed.size() * kWordBits;
  const long result = syscall(SYS_get_mempolicy, nullptr, allowed.data(),
                              node_count + 1, nullptr, MPOL_F_MEMS_ALLOWED);
  if (result == -1 && (errno == EPERM || errno == ENOSYS)) {
    GTEST_SKIP() << "NUMA policy queries unavailable: " << std::strerror(errno);
  }
  ASSERT_EQ(result, 0) << std::strerror(errno);
  uint32_t tested_nodes = 0;
  for (uint32_t node = 0; node < node_count; ++node) {
    if ((allowed[node / kWordBits] & (1UL << (node % kWordBits))) == 0) {
      continue;
    }
    ++tested_nodes;
    for (amdf_memory_flags_t flags : std::array<amdf_memory_flags_t, 2>{
             0, AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE}) {
      SCOPED_TRACE(::testing::Message()
                   << "node=" << node << " flags=" << flags);
      const amdf_memory_create_info_t create_info = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
          .structure_size = sizeof(create_info),
          .memory_profile_ordinal = profile_.ordinal,
          .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE | flags,
          .byte_length = 4 * page_size_,
          .placement = {AMDF_MEMORY_PLACEMENT_KIND_HOST_NUMA, node},
      };
      ASSERT_NO_FATAL_FAILURE(memory_.Create(api_, scope_, create_info));
      EXPECT_EQ(memory_.info.placement.kind, create_info.placement.kind);
      EXPECT_EQ(memory_.info.placement.node, node);
      ASSERT_NO_FATAL_FAILURE(ExpectPlacement(memory_.host.pointer, node));
      auto bytes = memory_.bytes();
      for (size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = (i * 17 + node) & 0xFF;
      }
      if (flags != 0) {
        amdf_memory_reservation_capabilities_t capabilities = {
            .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CAPABILITIES,
            .structure_size = sizeof(capabilities),
        };
        ASSERT_EQ(api_->instance_query_host_memory_reservation_capabilities(
                      instance_, &capabilities),
                  AMDF_STATUS_OK);
        const amdf_memory_reservation_access_t access = {
            .address_domain_ordinal = capabilities.address_domain_ordinal,
            .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
            .mapping_granularity = capabilities.mapping_granularity,
        };
        const amdf_memory_reservation_create_info_t reserve_info = {
            .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CREATE_INFO,
            .structure_size = sizeof(reserve_info),
            .byte_length = bytes.size(),
            .access_count = 1,
            .accesses = &access,
        };
        ASSERT_EQ(api_->memory_reservation_create(instance_, &reserve_info,
                                                  &reservation_),
                  AMDF_STATUS_OK);
        const amdf_memory_mapping_update_t update = {
            .byte_length = bytes.size(),
            .memory = memory_.memory,
            .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
        };
        ASSERT_EQ(api_->memory_reservation_update(reservation_, 0, 1, &update),
                  AMDF_STATUS_OK);
        const amdf_memory_map_info_t map_info = {
            .type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO,
            .structure_size = sizeof(map_info),
            .byte_length = bytes.size(),
            .flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
        };
        ASSERT_EQ(
            api_->memory_reservation_map(reservation_, 0, &map_info, &alias_),
            AMDF_STATUS_OK);
        amdf_host_mapping_info_t alias_info = {
            .type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO,
            .structure_size = sizeof(alias_info),
        };
        ASSERT_EQ(api_->host_mapping_query_info(alias_, &alias_info),
                  AMDF_STATUS_OK);
        ASSERT_NO_FATAL_FAILURE(ExpectPlacement(alias_info.pointer, node));
        EXPECT_EQ(std::memcmp(bytes.data(), alias_info.pointer, bytes.size()),
                  0);
        std::memset(alias_info.pointer, 0xA5, bytes.size());
        for (uint8_t value : bytes) {
          EXPECT_EQ(value, 0xA5);
        }
        ASSERT_EQ(api_->host_mapping_destroy(alias_), AMDF_STATUS_OK);
        alias_ = nullptr;
        ASSERT_EQ(api_->memory_reservation_destroy(reservation_),
                  AMDF_STATUS_OK);
        reservation_ = nullptr;
      }
      ASSERT_NO_FATAL_FAILURE(ExpectPlacement(bytes.data(), node));
      ASSERT_TRUE(memory_.Release(api_));
    }
  }
  EXPECT_GT(tested_nodes, 0u);
  RecordProperty("numa_nodes", tested_nodes);
}

TEST_F(HostNumaMemoryTest, RejectsInvalidNodeWithoutPublishingAnOwner) {
  const amdf_memory_create_info_t create_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
      .structure_size = sizeof(create_info),
      .memory_profile_ordinal = profile_.ordinal,
      .byte_length = page_size_,
      .placement = {AMDF_MEMORY_PLACEMENT_KIND_HOST_NUMA, UINT32_MAX},
  };
  auto* const sentinel = reinterpret_cast<amdf_memory_t*>(uintptr_t{1});
  amdf_memory_t* memory = sentinel;
  EXPECT_EQ(amdf_status_code(
                api_->memory_create(scope_, &create_info, &memory, nullptr)),
            AMDF_STATUS_CODE_OUT_OF_RANGE);
  EXPECT_EQ(memory, sentinel);
}

}  // namespace
