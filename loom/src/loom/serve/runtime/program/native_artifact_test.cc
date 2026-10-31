// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/program/native_artifact.h"

#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loomc/iree.h"
#include "loomc/target/vm.h"

namespace {

// Accounts actual requested storage and injects one allocator failure. The
// production cache, byte sequences, and diagnostics all use this allocator.
class Storage {
 public:
  iree_allocator_t allocator() { return {this, Control}; }
  size_t bytes() {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t result = 0;
    for (const auto& allocation : allocations_) {
      result += allocation.second;
    }
    return result;
  }
  void FailAfter(size_t count) { remaining_ = count; }

 private:
  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** inout_ptr) {
    auto* storage = static_cast<Storage*>(self);
    std::lock_guard<std::mutex> lock(storage->mutex_);
    const auto allocator = iree_allocator_system();
    if (command == IREE_ALLOCATOR_COMMAND_FREE) {
      EXPECT_EQ(storage->allocations_.erase(*inout_ptr), 1u);
      return allocator.ctl(allocator.self, command, params, inout_ptr);
    }
    if (storage->remaining_ != SIZE_MAX && storage->remaining_-- == 0) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "injected native storage failure");
    }
    void* previous =
        command == IREE_ALLOCATOR_COMMAND_REALLOC ? *inout_ptr : nullptr;
    auto status = allocator.ctl(allocator.self, command, params, inout_ptr);
    if (iree_status_is_ok(status)) {
      if (previous) {
        storage->allocations_.erase(previous);
      }
      storage->allocations_[*inout_ptr] =
          static_cast<const iree_allocator_alloc_params_t*>(params)
              ->byte_length;
    }
    return status;
  }

  // Protects allocations made or released by concurrent producers/consumers.
  std::mutex mutex_;
  // Requested bytes for each live allocation, not allocator usable size.
  std::unordered_map<void*, size_t> allocations_;
  // Number of successful allocation commands before one injected failure.
  size_t remaining_ = SIZE_MAX;
};

class NativeArtifactTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_status_from_loomc(loomc_target_environment_create_vm(
        loomc_allocator_system(), &environment_)));
    IREE_ASSERT_OK(iree_status_from_loomc(loomc_target_profile_select(
        environment_, loomc_make_cstring_view("vm:core"),
        loomc_allocator_system(), &options_.profile)));
    options_.artifact_format = IREE_SV("vm");
    IREE_ASSERT_OK(
        loom_serve_native_cache_allocate(storage_.allocator(), &cache_));
    IREE_ASSERT_OK(
        loom_serve_native_domain_acquire(cache_, &options_, &domain_));
  }

  void TearDown() override {
    loom_serve_native_domain_release(domain_);
    loom_serve_native_cache_free(cache_);
    EXPECT_EQ(storage_.bytes(), 0u);
    loomc_target_profile_release(options_.profile);
    loomc_target_environment_release(environment_);
  }

  static iree_status_t Acquire(loom_serve_native_domain_t* domain,
                               iree_string_view_t bytes,
                               loom_serve_native_subscription_t* subscription,
                               loom_serve_native_artifact_t** artifact,
                               bool* created, bool* ready) {
    iree_string_builder_t key;
    iree_string_builder_initialize(loom_serve_native_domain_allocator(domain),
                                   &key);
    auto status = iree_string_builder_append_string(&key, bytes);
    if (iree_status_is_ok(status)) {
      status = loom_serve_native_artifact_acquire(domain, &key, 1, subscription,
                                                  artifact, created, ready);
    }
    iree_string_builder_deinitialize(&key);
    return status;
  }

  static iree_status_t Image(iree_allocator_t allocator,
                             iree_byte_sequence_t** out_image,
                             uint8_t identity = 1) {
    const uint8_t bytes[] = {identity, 2, 3, 4};
    iree_byte_span_t owned = iree_byte_span_empty();
    IREE_RETURN_IF_ERROR(iree_allocator_clone(
        allocator, iree_make_const_byte_span(bytes, sizeof(bytes)),
        reinterpret_cast<void**>(&owned.data)));
    owned.data_length = sizeof(bytes);
    auto status =
        iree_byte_sequence_create_from_span_move(&owned, allocator, out_image);
    iree_allocator_free(allocator, owned.data);
    return status;
  }

  iree_status_t Populate(loom_serve_native_artifact_t* artifact,
                         uint8_t identity = 1) {
    auto* result = loom_serve_native_artifact_result(artifact);
    loomc_string_view_t root = {};
    IREE_RETURN_IF_ERROR(iree_status_from_loomc(loomc_string_view_clone(
        loomc_make_cstring_view("entry"),
        loomc_allocator_from_iree(storage_.allocator()), &root)));
    result->roots[0] = iree_string_view_from_loomc(root);
    return Image(storage_.allocator(), &result->image, identity);
  }

  static void Ready(void* payload, const loom_serve_native_result_t* result) {
    ++*static_cast<size_t*>(payload);
  }

  // Real immutable compiler profile, independent of the cache under test.
  loomc_target_environment_t* environment_ = nullptr;
  // Complete domain policy using the portable VM profile.
  loom_serve_native_domain_options_t options_ = {};
  // Actual cache allocation ownership and failure injection.
  Storage storage_;
  // Host-owned weak registry.
  loom_serve_native_cache_t* cache_ = nullptr;
  // Catalog-owned exact compilation domain.
  loom_serve_native_domain_t* domain_ = nullptr;
};

TEST_F(NativeArtifactTest, ExactPolicyGroupsIndependentProfiles) {
  auto options = options_;
  IREE_ASSERT_OK(iree_status_from_loomc(loomc_target_profile_select(
      environment_, loomc_make_cstring_view("vm:core"),
      loomc_allocator_system(), &options.profile)));
  loom_serve_native_domain_t* peer = nullptr;
  IREE_ASSERT_OK(loom_serve_native_domain_acquire(cache_, &options, &peer));
  EXPECT_EQ(peer, domain_);
  loom_serve_native_domain_release(peer);
  loomc_target_profile_release(options.profile);
  for (int field = 0; field < 3; ++field) {
    SCOPED_TRACE(field);
    options = options_;
    switch (field) {
      case 0:
        options.control_flow_lowering =
            LOOMC_TARGET_CONTROL_FLOW_LOWERING_STRUCTURED_LOW;
        break;
      case 1:
        options.sanitizer_checks = LOOMC_SANITIZER_CHECK_ACCESS;
        break;
      case 2:
        options.sanitizer_reporting_mode = LOOMC_SANITIZER_REPORTING_MODE_TRAP;
        break;
    }
    IREE_ASSERT_OK(loom_serve_native_domain_acquire(cache_, &options, &peer));
    EXPECT_NE(peer, domain_);
    loom_serve_native_domain_release(peer);
  }
}

TEST_F(NativeArtifactTest, PendingAndReadyHitsRetainOneOutcome) {
  size_t callbacks = 0;
  loom_serve_native_subscription_t subscriptions[] = {
      {Ready, &callbacks, nullptr},
      {Ready, &callbacks, nullptr},
      {Ready, &callbacks, nullptr}};
  loom_serve_native_artifact_t* artifacts[3] = {};
  bool created = false, ready = false;
  for (size_t i = 0; i < 2; ++i) {
    IREE_ASSERT_OK(Acquire(domain_, IREE_SV("exact input"), &subscriptions[i],
                           &artifacts[i], &created, &ready));
    EXPECT_EQ(created, i == 0);
    EXPECT_FALSE(ready);
    EXPECT_EQ(artifacts[0], artifacts[i]);
  }
  auto* result = loom_serve_native_artifact_result(artifacts[0]);
  IREE_ASSERT_OK(Populate(artifacts[0]));
  loom_serve_native_artifact_publish(artifacts[0]);
  EXPECT_EQ(callbacks, 2u);
  const auto retained = storage_.bytes();
  IREE_ASSERT_OK(Acquire(domain_, IREE_SV("exact input"), &subscriptions[2],
                         &artifacts[2], &created, &ready));
  EXPECT_FALSE(created);
  EXPECT_TRUE(ready);
  EXPECT_EQ(artifacts[2], artifacts[0]);
  EXPECT_EQ(callbacks, 2u);
  EXPECT_EQ(storage_.bytes(), retained);
  // Loaded-device consumers can retain the immutable image independently of
  // every catalog and the compilation registry.
  auto* image = result->image;
  iree_byte_sequence_retain(image);
  for (auto* artifact : artifacts) {
    loom_serve_native_artifact_release(artifact);
  }
  loom_serve_native_domain_release(domain_);
  domain_ = nullptr;
  loom_serve_native_cache_free(cache_);
  cache_ = nullptr;
  iree_const_byte_span_t bytes = iree_const_byte_span_empty();
  ASSERT_TRUE(iree_byte_sequence_try_get_contiguous_span(image, &bytes));
  ASSERT_EQ(bytes.data_length, 4u);
  EXPECT_EQ(bytes.data[3], 4);
  iree_byte_sequence_release(image);
  EXPECT_EQ(storage_.bytes(), 0u);
}

TEST_F(NativeArtifactTest,
       DistinctInputsInternSegmentedOutputWithoutAliasingRoots) {
  size_t callbacks = 0;
  loom_serve_native_subscription_t subscriptions[] = {
      {Ready, &callbacks, nullptr}, {Ready, &callbacks, nullptr}};
  loom_serve_native_artifact_t* artifacts[2] = {};
  const char keys[][4] = {{'a', '\0', 'b', 'c'}, {'a', '\0', 'b', 'd'}};
  const char* names[] = {"first", "second"};
  for (size_t i = 0; i < 2; ++i) {
    bool created = false, ready = false;
    IREE_ASSERT_OK(Acquire(domain_, {keys[i], sizeof(keys[i])},
                           &subscriptions[i], &artifacts[i], &created, &ready));
    EXPECT_TRUE(created);
    auto* result = loom_serve_native_artifact_result(artifacts[i]);
    loomc_string_view_t root = {};
    IREE_ASSERT_OK(iree_status_from_loomc(loomc_string_view_clone(
        loomc_make_cstring_view(names[i]),
        loomc_allocator_from_iree(storage_.allocator()), &root)));
    result->roots[0] = iree_string_view_from_loomc(root);
    iree_byte_sequence_t* segment = nullptr;
    IREE_ASSERT_OK(Image(storage_.allocator(), &segment));
    iree_byte_sequence_t* segments[] = {segment, segment};
    IREE_ASSERT_OK(iree_byte_sequence_create_concat(
        2, segments, storage_.allocator(), &result->image));
    iree_byte_sequence_release(segment);
    loom_serve_native_artifact_publish(artifacts[i]);
    IREE_ASSERT_OK(iree_status_clone(result->status));
  }
  EXPECT_NE(artifacts[0], artifacts[1]);
  const auto* first = loom_serve_native_artifact_result(artifacts[0]);
  const auto* second = loom_serve_native_artifact_result(artifacts[1]);
  EXPECT_EQ(first->image, second->image);
  EXPECT_TRUE(iree_string_view_equal(first->roots[0], IREE_SV("first")));
  EXPECT_TRUE(iree_string_view_equal(second->roots[0], IREE_SV("second")));
  iree_const_byte_span_t bytes = iree_const_byte_span_empty();
  ASSERT_TRUE(iree_byte_sequence_try_get_contiguous_span(first->image, &bytes));
  EXPECT_EQ(bytes.data_length, 8u);
  loom_serve_native_artifact_release(artifacts[0]);
  EXPECT_EQ(bytes.data[7], 4);
  loom_serve_native_artifact_release(artifacts[1]);
  EXPECT_EQ(callbacks, 2u);
}

TEST_F(NativeArtifactTest, FailureUnlinksBeforeReentrantRetryAndRetirement) {
  struct Consumer {
    // Exact domain retained until callback completion.
    loom_serve_native_domain_t* domain;
    // Consumer reference released by its own completion.
    loom_serve_native_artifact_t* artifact = nullptr;
    // New producer acquired while the failed producer is still publishing.
    loom_serve_native_artifact_t* retry = nullptr;
    // Retry subscription, outliving the initial callback.
    loom_serve_native_subscription_t retry_subscription = {};
  } consumer{domain_};
  size_t retry_callbacks = 0;
  consumer.retry_subscription = {Ready, &retry_callbacks, nullptr};
  loom_serve_native_subscription_t subscription = {
      [](void* payload, const loom_serve_native_result_t* result) {
        auto* consumer = static_cast<Consumer*>(payload);
        EXPECT_EQ(iree_status_code(result->status),
                  IREE_STATUS_RESOURCE_EXHAUSTED);
        bool created = false, ready = false;
        IREE_ASSERT_OK(Acquire(consumer->domain, IREE_SV("retry"),
                               &consumer->retry_subscription, &consumer->retry,
                               &created, &ready));
        EXPECT_TRUE(created);
        EXPECT_FALSE(ready);
        EXPECT_NE(consumer->retry, consumer->artifact);
        loom_serve_native_artifact_release(consumer->artifact);
        consumer->artifact = nullptr;
      },
      &consumer, nullptr};
  bool created = false, ready = false;
  IREE_ASSERT_OK(Acquire(domain_, IREE_SV("retry"), &subscription,
                         &consumer.artifact, &created, &ready));
  loom_serve_native_artifact_result(consumer.artifact)->status =
      iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED);
  loom_serve_native_artifact_publish(consumer.artifact);
  EXPECT_EQ(consumer.artifact, nullptr);
  IREE_ASSERT_OK(Populate(consumer.retry));
  loom_serve_native_artifact_publish(consumer.retry);
  EXPECT_EQ(retry_callbacks, 1u);
  loom_serve_native_artifact_release(consumer.retry);
}

TEST_F(NativeArtifactTest, AllocationFailuresRetireEveryAcceptedProducer) {
  const size_t baseline = storage_.bytes();
  bool succeeded = false;
  for (size_t failure = 0; !succeeded; ++failure) {
    SCOPED_TRACE(failure);
    storage_.FailAfter(failure);
    loom_serve_native_cache_t* cache = nullptr;
    loom_serve_native_domain_t* domain = nullptr;
    loom_serve_native_artifact_t* artifact = nullptr;
    size_t callbacks = 0;
    loom_serve_native_subscription_t subscription = {Ready, &callbacks,
                                                     nullptr};
    auto status =
        loom_serve_native_cache_allocate(storage_.allocator(), &cache);
    if (iree_status_is_ok(status)) {
      status = loom_serve_native_domain_acquire(cache, &options_, &domain);
    }
    bool created = false, ready = false;
    if (iree_status_is_ok(status)) {
      status = Acquire(domain, IREE_SV("allocation failure"), &subscription,
                       &artifact, &created, &ready);
    }
    if (iree_status_is_ok(status)) {
      status = Populate(artifact);
    }
    if (artifact) {
      auto* result = loom_serve_native_artifact_result(artifact);
      result->status = status;
      loom_serve_native_artifact_publish(artifact);
      status = iree_status_clone(result->status);
      EXPECT_EQ(callbacks, 1u);
    }
    succeeded = iree_status_is_ok(status);
    if (!succeeded) {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
    }
    storage_.FailAfter(SIZE_MAX);
    loom_serve_native_artifact_release(artifact);
    loom_serve_native_domain_release(domain);
    loom_serve_native_cache_free(cache);
    EXPECT_EQ(storage_.bytes(), baseline);
  }
}

TEST_F(NativeArtifactTest, IndexGrowthAndRetirementBoundRetainedStorage) {
  const size_t baseline = storage_.bytes();
  constexpr size_t kCount = 256;
  std::vector<loom_serve_native_artifact_t*> artifacts(kCount);
  size_t callbacks = 0;
  loom_serve_native_subscription_t subscription = {Ready, &callbacks, nullptr};
  for (size_t i = 0; i < kCount; ++i) {
    const std::string key = "native " + std::to_string(i);
    bool created = false, ready = false;
    IREE_ASSERT_OK(Acquire(domain_, iree_make_cstring_view(key.c_str()),
                           &subscription, &artifacts[i], &created, &ready));
    EXPECT_TRUE(created);
    IREE_ASSERT_OK(Populate(artifacts[i], static_cast<uint8_t>(i)));
    loom_serve_native_artifact_publish(artifacts[i]);
  }
  RecordProperty("live_256_inputs_and_images_bytes",
                 std::to_string(storage_.bytes() - baseline));
  for (size_t i = 0; i < kCount; ++i) {
    const std::string key = "native " + std::to_string(i);
    loom_serve_native_artifact_t* hit = nullptr;
    bool created = false, ready = false;
    IREE_ASSERT_OK(Acquire(domain_, iree_make_cstring_view(key.c_str()),
                           &subscription, &hit, &created, &ready));
    EXPECT_FALSE(created);
    EXPECT_TRUE(ready);
    EXPECT_EQ(hit, artifacts[i]);
    iree_const_byte_span_t bytes = iree_const_byte_span_empty();
    ASSERT_TRUE(iree_byte_sequence_try_get_contiguous_span(
        loom_serve_native_artifact_result(hit)->image, &bytes));
    EXPECT_EQ(bytes.data[0], i);
    loom_serve_native_artifact_release(hit);
  }
  EXPECT_EQ(callbacks, kCount);
  for (auto* artifact : artifacts) {
    loom_serve_native_artifact_release(artifact);
  }
  const size_t idle = storage_.bytes();
  RecordProperty("retained_index_capacity_bytes",
                 std::to_string(idle - baseline));
  // Distinct later inputs do not accumulate source keys, image records, or
  // diagnostics after their last owner. Only the peak-live index capacity
  // stays.
  for (size_t i = 0; i < 2 * kCount; ++i) {
    const std::string key = "later " + std::to_string(i);
    loom_serve_native_artifact_t* artifact = nullptr;
    bool created = false, ready = false;
    IREE_ASSERT_OK(Acquire(domain_, iree_make_cstring_view(key.c_str()),
                           &subscription, &artifact, &created, &ready));
    EXPECT_TRUE(created);
    IREE_ASSERT_OK(Populate(artifact));
    loom_serve_native_artifact_publish(artifact);
    loom_serve_native_artifact_release(artifact);
    EXPECT_EQ(storage_.bytes(), idle);
  }
  loom_serve_native_domain_release(domain_);
  domain_ = nullptr;
  loom_serve_native_cache_free(cache_);
  cache_ = nullptr;
  EXPECT_EQ(storage_.bytes(), 0u);
}

}  // namespace
