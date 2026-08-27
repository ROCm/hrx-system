// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/context_cache.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include "iree/testing/gtest.h"

namespace {

iree_const_byte_span_t Bytes(const std::vector<uint8_t>& data) {
  return iree_make_const_byte_span(data.data(), data.size());
}

iree_hal_amdxdna_context_cache_key_t Key(
    iree_const_byte_span_t pdi, const char* kernel_name) {
  iree_hal_amdxdna_context_cache_key_t key = {};
  key.pdi = pdi;
  key.xclbin = iree_const_byte_span_empty();
  key.kernel_name = iree_make_cstring_view(kernel_name);
  return key;
}

struct FakeContextRef {
  explicit FakeContextRef(uint8_t key) : key(key) {}

  uint8_t key;
  std::atomic<int> reference_count{1};
};

struct ScriptedCreateResult {
  iree_status_code_t status_code;
  bool context_pool_exhausted;
};

class FakeContextFactory {
 public:
  FakeContextFactory() {
    for (auto& value : create_counts) value.store(0);
    for (auto& value : destroy_counts) value.store(0);
  }

  iree_hal_amdxdna_context_cache_ops_t Ops() {
    iree_hal_amdxdna_context_cache_ops_t ops = {};
    ops.create_context = Create;
    ops.retain_context = Retain;
    ops.release_context = Release;
    return ops;
  }

  void ScriptFailure(iree_status_code_t status_code,
                     bool context_pool_exhausted) {
    scripted_results.push_back({status_code, context_pool_exhausted});
  }

  void ReleaseCaller(iree_hal_amdxdna_native_context_ref_t* context_ref) {
    Release(this, context_ref);
  }

  std::atomic<int> create_calls{0};
  std::array<std::atomic<int>, 256> create_counts;
  std::array<std::atomic<int>, 256> destroy_counts;

 private:
  static iree_status_t Create(
      void* user_data,
      const iree_hal_amdxdna_native_c_context_image_t* context_image,
      bool* out_context_pool_exhausted,
      iree_hal_amdxdna_native_context_ref_t** out_context_ref) {
    auto* self = static_cast<FakeContextFactory*>(user_data);
    self->create_calls.fetch_add(1);
    *out_context_pool_exhausted = false;
    *out_context_ref = nullptr;
    if (self->next_scripted_result < self->scripted_results.size()) {
      const ScriptedCreateResult result =
          self->scripted_results[self->next_scripted_result++];
      if (result.status_code != IREE_STATUS_OK) {
        *out_context_pool_exhausted = result.context_pool_exhausted;
        return iree_make_status(result.status_code,
                                "scripted native context creation failure");
      }
    }
    const uint8_t key = context_image->pdi.data[0];
    self->create_counts[key].fetch_add(1);
    auto context_ref = std::make_unique<FakeContextRef>(key);
    FakeContextRef* raw_context_ref = context_ref.get();
    self->contexts.push_back(std::move(context_ref));
    *out_context_ref = reinterpret_cast<iree_hal_amdxdna_native_context_ref_t*>(
        raw_context_ref);
    return iree_ok_status();
  }

  static iree_hal_amdxdna_native_context_ref_t* Retain(
      void* user_data, iree_hal_amdxdna_native_context_ref_t* context_ref) {
    (void)user_data;
    auto* fake_context_ref = reinterpret_cast<FakeContextRef*>(context_ref);
    fake_context_ref->reference_count.fetch_add(1);
    return context_ref;
  }

  static void Release(void* user_data,
                      iree_hal_amdxdna_native_context_ref_t* context_ref) {
    if (!context_ref) return;
    auto* self = static_cast<FakeContextFactory*>(user_data);
    auto* fake_context_ref = reinterpret_cast<FakeContextRef*>(context_ref);
    if (fake_context_ref->reference_count.fetch_sub(1) == 1) {
      self->destroy_counts[fake_context_ref->key].fetch_add(1);
    }
  }

  std::vector<ScriptedCreateResult> scripted_results;
  size_t next_scripted_result = 0;
  std::vector<std::unique_ptr<FakeContextRef>> contexts;
};

class ContextCachePolicyTest : public ::testing::Test {
 protected:
  void CreateCache(iree_host_size_t capacity) {
    auto ops = factory_.Ops();
    cache_ = iree_hal_amdxdna_device_context_cache_create_with_ops(
        iree_allocator_system(), capacity, &ops, &factory_);
    ASSERT_NE(cache_, nullptr);
  }

  iree_status_t Get(uint8_t key,
                    iree_hal_amdxdna_native_context_ref_t** out_context_ref) {
    return iree_hal_amdxdna_context_cache_get_or_create(
        cache_, nullptr, IREE_HAL_AMDXDNA_NATIVE_C_CONTEXT_IMAGE_MODEL_PDI,
        iree_make_const_byte_span(&key, 1), iree_const_byte_span_empty(),
        IREE_SV("MLIR_AIE"), out_context_ref);
  }

  iree_hal_amdxdna_native_context_ref_t* GetAndExpectOk(uint8_t key) {
    iree_hal_amdxdna_native_context_ref_t* context_ref = nullptr;
    iree_status_t status = Get(key, &context_ref);
    const bool ok = iree_status_is_ok(status);
    EXPECT_TRUE(ok);
    iree_status_ignore(status);
    if (!ok) return nullptr;
    EXPECT_NE(context_ref, nullptr);
    return context_ref;
  }

  void TearDown() override {
    iree_hal_amdxdna_device_context_cache_destroy(cache_);
    cache_ = nullptr;
  }

  FakeContextFactory factory_;
  iree_hal_amdxdna_device_context_cache_t* cache_ = nullptr;
};

// Cross-executable reuse: two distinct executables that reference byte-for-byte
// identical native context-image inputs through independent backing storage must
// be treated as the same context so the cache is shared.
TEST(ContextCacheKeyTest, IdenticalContentThroughDistinctStorageMatches) {
  std::vector<uint8_t> pdi_a = {1, 2, 3, 4};
  std::vector<uint8_t> pdi_b = {1, 2, 3, 4};  // distinct storage, same bytes

  auto lhs = Key(Bytes(pdi_a), "MLIR_AIE");
  auto rhs = Key(Bytes(pdi_b), "MLIR_AIE");

  EXPECT_TRUE(iree_hal_amdxdna_context_cache_key_equal(&lhs, &rhs));
}

// Dispatch control code is not hardware-context identity. Linux KMQ creates a
// native hwctx from PDI + CU name only; command/run-list differences are handled
// by the dispatch command cache and must not create duplicate hwctx objects.
TEST(ContextCacheKeyTest, SameImageIgnoresDispatchControlCodeByConstruction) {
  std::vector<uint8_t> pdi = {9, 8, 7, 6};

  auto lhs = Key(Bytes(pdi), "MLIR_AIE");
  auto rhs = Key(Bytes(pdi), "MLIR_AIE");

  EXPECT_TRUE(iree_hal_amdxdna_context_cache_key_equal(&lhs, &rhs));
}

TEST(ContextCacheKeyTest, DifferentPdiMisses) {
  std::vector<uint8_t> pdi_a = {1, 2, 3, 4};
  std::vector<uint8_t> pdi_b = {1, 2, 3, 5};

  auto lhs = Key(Bytes(pdi_a), "MLIR_AIE");
  auto rhs = Key(Bytes(pdi_b), "MLIR_AIE");

  EXPECT_FALSE(iree_hal_amdxdna_context_cache_key_equal(&lhs, &rhs));
}

TEST(ContextCacheKeyTest, DifferentKernelNameMisses) {
  std::vector<uint8_t> pdi = {2, 4, 6};

  auto lhs = Key(Bytes(pdi), "kernel_a");
  auto rhs = Key(Bytes(pdi), "kernel_b");

  EXPECT_FALSE(iree_hal_amdxdna_context_cache_key_equal(&lhs, &rhs));
}

TEST_F(ContextCachePolicyTest, HitPromotesEntryBeforeLruEviction) {
  CreateCache(2);
  auto* one = GetAndExpectOk(1);
  factory_.ReleaseCaller(one);
  auto* two = GetAndExpectOk(2);
  factory_.ReleaseCaller(two);

  // Promote key 1 from the LRU tail to the MRU head.
  one = GetAndExpectOk(1);
  factory_.ReleaseCaller(one);
  EXPECT_EQ(factory_.create_counts[1].load(), 1);

  // Inserting key 3 must now evict key 2, not the recently-hit key 1.
  auto* three = GetAndExpectOk(3);
  factory_.ReleaseCaller(three);
  EXPECT_EQ(factory_.destroy_counts[1].load(), 0);
  EXPECT_EQ(factory_.destroy_counts[2].load(), 1);
  EXPECT_EQ(factory_.destroy_counts[3].load(), 0);

  one = GetAndExpectOk(1);
  factory_.ReleaseCaller(one);
  EXPECT_EQ(factory_.create_counts[1].load(), 1);
}

TEST_F(ContextCachePolicyTest, PoolExhaustionEvictsLruAndRetries) {
  CreateCache(3);
  auto* one = GetAndExpectOk(1);
  factory_.ReleaseCaller(one);
  auto* two = GetAndExpectOk(2);
  factory_.ReleaseCaller(two);

  // The explicit native classification, not the generic status code, controls
  // retry. Windows may preserve an INTERNAL status while still identifying an
  // unambiguous context-ID exhaustion condition.
  factory_.ScriptFailure(IREE_STATUS_INTERNAL, true);
  auto* three = GetAndExpectOk(3);
  factory_.ReleaseCaller(three);

  EXPECT_EQ(factory_.create_calls.load(), 4);
  EXPECT_EQ(factory_.destroy_counts[1].load(), 1);
  EXPECT_EQ(factory_.destroy_counts[2].load(), 0);
  EXPECT_EQ(factory_.create_counts[3].load(), 1);
}

TEST_F(ContextCachePolicyTest, NonPoolFailureDoesNotEvictOrRetry) {
  CreateCache(3);
  auto* one = GetAndExpectOk(1);
  factory_.ReleaseCaller(one);
  auto* two = GetAndExpectOk(2);
  factory_.ReleaseCaller(two);

  // Generic resource exhaustion (for example, a PDI BO allocation failure)
  // must not evict healthy cached contexts without the native pool signal.
  factory_.ScriptFailure(IREE_STATUS_RESOURCE_EXHAUSTED, false);
  iree_hal_amdxdna_native_context_ref_t* three = nullptr;
  iree_status_t status = Get(3, &three);
  EXPECT_EQ(iree_status_code(status), IREE_STATUS_RESOURCE_EXHAUSTED);
  iree_status_ignore(status);
  EXPECT_EQ(three, nullptr);
  EXPECT_EQ(factory_.create_calls.load(), 3);
  EXPECT_EQ(factory_.destroy_counts[1].load(), 0);
  EXPECT_EQ(factory_.destroy_counts[2].load(), 0);

  one = GetAndExpectOk(1);
  factory_.ReleaseCaller(one);
  two = GetAndExpectOk(2);
  factory_.ReleaseCaller(two);
  EXPECT_EQ(factory_.create_calls.load(), 3);
}

TEST_F(ContextCachePolicyTest, EvictionPreservesLiveCallerReference) {
  CreateCache(1);
  auto* one = GetAndExpectOk(1);

  auto* two = GetAndExpectOk(2);
  factory_.ReleaseCaller(two);
  EXPECT_EQ(factory_.destroy_counts[1].load(), 0)
      << "eviction must release only the cache-owned reference";

  factory_.ReleaseCaller(one);
  EXPECT_EQ(factory_.destroy_counts[1].load(), 1);
}

TEST_F(ContextCachePolicyTest, ConcurrentCallersPinEvictedContext) {
  CreateCache(1);
  auto* initial = GetAndExpectOk(1);
  factory_.ReleaseCaller(initial);

  constexpr int kThreadCount = 4;
  std::atomic<int> ready{0};
  std::atomic<bool> release_callers{false};
  std::atomic<int> failures{0};
  std::vector<std::thread> callers;
  callers.reserve(kThreadCount);
  for (int i = 0; i < kThreadCount; ++i) {
    callers.emplace_back([&]() {
      iree_hal_amdxdna_native_context_ref_t* context_ref = nullptr;
      iree_status_t status = Get(1, &context_ref);
      if (!iree_status_is_ok(status) || !context_ref) {
        failures.fetch_add(1);
        iree_status_ignore(status);
        return;
      }
      ready.fetch_add(1);
      while (!release_callers.load()) std::this_thread::yield();
      factory_.ReleaseCaller(context_ref);
    });
  }
  while (ready.load() != kThreadCount && failures.load() == 0) {
    std::this_thread::yield();
  }
  if (failures.load() != 0) {
    release_callers.store(true);
    for (auto& caller : callers) caller.join();
    FAIL() << "a concurrent cache lookup failed";
  }

  auto* two = GetAndExpectOk(2);
  factory_.ReleaseCaller(two);
  EXPECT_EQ(factory_.destroy_counts[1].load(), 0);

  release_callers.store(true);
  for (auto& caller : callers) caller.join();
  EXPECT_EQ(factory_.destroy_counts[1].load(), 1);
}

}  // namespace
