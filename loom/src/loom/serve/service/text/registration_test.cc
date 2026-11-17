// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/registration.h"

#include <cstdlib>
#include <fstream>
#include <memory>
#include <utility>
#include <vector>

#include "iree/base/threading/notification.h"
#include "iree/base/tooling/flags.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"
#include "loom/runtime/driver.h"
#include "loom/runtime/testing/application.h"
#include "loom/serve/media/image_codec.h"
#include "loom/serve/service/text/source_test_oracle.h"

IREE_FLAG(int32_t, native_text_workers, 1,
          "Shared worker count for native text qualification.");

namespace {
using loom_serve::testing::text::Oracle;
using NativeStream = std::unique_ptr<loom_serve_text_stream_t,
                                     decltype(&loom_serve_text_stream_close)>;
using NativeControl =
    std::unique_ptr<loom_serve_text_checkpoint_control_t,
                    decltype(&loom_serve_text_checkpoint_control_close)>;
std::string String(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

// An embedder owns resources and one serialized driver. Model preparation,
// credited output, and retirement need no server, registry, or socket.
class TextRegistrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(Initialize());
    ASSERT_NO_FATAL_FAILURE(Allocate());
    IREE_ASSERT_OK(Prepare());
  }

  void Initialize() {
    iree_notification_initialize(&notification_);
    auto options = loom_runtime_test_application_options_default();
    options.worker_count = FLAG_native_text_workers;
    if (options.backing == LOOM_RUNTIME_POOL_BACKING_ELASTIC) {
      options.memory_limit = 64 * 1024 * 1024;
    }
    loom_runtime_test_application_t* application = nullptr;
    IREE_ASSERT_OK(loom_runtime_test_application_acquire(
        &options, iree_allocator_system(), &application));
    group_ = loom_runtime_test_application_device_group(application);
    loom_runtime_device_group_set_observer(group_, {Wake, this});

    const char* source = std::getenv("LOOM_SERVE_TEST_MODEL_SOURCE");
    ASSERT_NE(source, nullptr);
    const std::string path = source;
    source_directory_ = path.substr(0, path.find_last_of('/'));
    parameters_path_ =
        iree::testing::MakeTempFilePath("native_text", ".safetensors");
    std::string header =
        R"({"bias":{"dtype":"I32","shape":[1],"data_offsets":[0,4]}})";
    header.append((8 - header.size() % 8) % 8, ' ');
    std::ofstream file(parameters_path_, std::ios::binary);
    uint8_t bytes[8];
    iree_unaligned_store_le_u64(bytes, header.size());
    file.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
    file.write(header.data(), header.size());
    iree_unaligned_store_le_u32(bytes, 2);
    file.write(reinterpret_cast<const char*>(bytes), 4);
    file.close();
    ASSERT_TRUE(file.good());

    device_ = loom_runtime_test_application_device_at(application, 0);
  }

  void Allocate(loom_serve_text_mode_t mode = LOOM_SERVE_TEXT_MODE_GENERATE,
                const loom_service_finite_model_options_t* image = nullptr,
                iree_string_view_t prepare = IREE_SV("prepare.loom")) {
    static const iree_string_view_t providers[] = {IREE_SV("advance.loom"),
                                                   IREE_SV("sequence.loom"),
                                                   IREE_SV("storage.loom")};
    const loom_serve_packing_shape_t shapes[] = {{4, 1}, {16, 4}, {32, 4}};
    const loom_serve_text_options_t model_options = {
        .mode = mode,
        .sources =
            {
                .base_directory =
                    iree_make_cstring_view(source_directory_.c_str()),
                .prepare = prepare,
                .control = IREE_SV("control.loom"),
                .count = IREE_ARRAYSIZE(providers),
                .values = providers,
            },
        .prefill_capacity = 32,
        .context_capacity = 96,
        .pool_capacity = 256,
        .epoch_count = IREE_ARRAYSIZE(shapes),
        .epoch_shapes = shapes,
        .kernel_sanitizer =
            {
                .type = LOOMC_STRUCTURE_TYPE_SANITIZER_OPTIONS,
                .structure_size = sizeof(loomc_sanitizer_options_t),
            },
        .weights_path = iree_make_cstring_view(parameters_path_.c_str()),
        .tokenizer_path = mode == LOOM_SERVE_TEXT_MODE_GENERATE
                              ? IREE_SV("tokenizer.json")
                              : iree_string_view_empty(),
        .row_count = 4,
        .checkpoint_capacity = 2,
    };
    const loom_serve_text_registration_options_t service_options = {
        .name = IREE_SV("fixture-text"),
        .chunk_size = 3,
        .default_max_tokens = 16,
        .pending_capacity = 2,
        .continuation_epochs = 1,
    };
    IREE_ASSERT_OK(loom_serve_text_registration_allocate(
        device_, loom_runtime_device_group_pool_at(group_, 0), &model_options,
        image, &service_options, {Wake, this}, iree_allocator_system(),
        &registration_));
  }

  iree_status_t Prepare(loom_serve_text_load_flags_t flags = 0) {
    bool complete = false;
    iree_status_t status = iree_ok_status();
    loom_runtime_compilation_report_t report = {};
    while (!complete) {
      const auto ticket = iree_notification_prepare_wait(&notification_);
      status =
          iree_status_join(status, loom_runtime_device_group_advance(group_));
      status =
          iree_status_join(status, loom_serve_text_registration_prepare(
                                       registration_,
                                       iree_status_is_ok(status)
                                           ? flags
                                           : LOOM_SERVE_TEXT_LOAD_FLAG_CANCEL,
                                       &complete, &service_, &report));
      if (complete) {
        iree_notification_cancel_wait(&notification_);
      } else {
        iree_notification_commit_wait(&notification_, ticket,
                                      IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      }
    }
    if (iree_status_is_ok(status)) {
      runtime_ = loom_serve_text_service_interface(service_).runtime;
      loom_runtime_driver_initialize(1, &runtime_, &driver_);
      loom_runtime_driver_set_observer(&driver_, {Wake, this});
    }
    return status;
  }

  void TearDown() override {
    ASSERT_NO_FATAL_FAILURE(Retire());
    if (group_) {
      loom_runtime_device_group_set_observer(group_, {});
    }
    iree_notification_deinitialize(&notification_);
    iree_status_free(driver_.status);
  }

  static void Wake(void* self) {
    iree_notification_post(
        &static_cast<TextRegistrationTest*>(self)->notification_,
        IREE_ALL_WAITERS);
  }

  template <typename Predicate>
  void DriveUntil(Predicate done) {
    while (true) {
      const auto ticket = iree_notification_prepare_wait(&notification_);
      driver_.status = iree_status_join(
          driver_.status, loom_runtime_device_group_advance(group_));
      bool progress = false;
      loom_runtime_driver_advance(&driver_, &progress);
      loom_runtime_driver_collect(&driver_, &progress);
      const bool complete = done();
      if (complete || progress || !iree_status_is_ok(driver_.status)) {
        iree_notification_cancel_wait(&notification_);
      } else {
        iree_notification_commit_wait(&notification_, ticket,
                                      IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      }
      IREE_ASSERT_OK(iree_status_clone(driver_.status));
      if (complete) {
        break;
      }
    }
  }

  void Retire() {
    if (service_) {
      driver_.stopping = true;
      ASSERT_NO_FATAL_FAILURE(DriveUntil(
          [&] { return !runtime_.vtable->is_pending(runtime_.self); }));
      loom_runtime_driver_set_observer(&driver_, {});
      service_ = nullptr;
      driver_.services = nullptr;
      driver_.capacity = 0;
    }
    if (registration_) {
      while (true) {
        const auto ticket = iree_notification_prepare_wait(&notification_);
        auto status = loom_runtime_device_group_advance(group_);
        bool complete = false;
        status = iree_status_join(status, loom_serve_text_registration_retire(
                                              registration_, &complete));
        if (complete || !iree_status_is_ok(status)) {
          iree_notification_cancel_wait(&notification_);
        } else {
          iree_notification_commit_wait(&notification_, ticket,
                                        IREE_DURATION_ZERO,
                                        IREE_TIME_INFINITE_FUTURE);
        }
        IREE_ASSERT_OK(status);
        if (complete) {
          break;
        }
      }
      loom_serve_text_registration_free(std::exchange(registration_, nullptr));
    }
  }

  void Open(const char* session, NativeStream* stream,
            iree_string_view_t body = kRequest) {
    loom_serve_text_request_t request = {
        .session = iree_make_cstring_view(session),
    };
    iree_string_builder_initialize(iree_allocator_system(), &request.body);
    auto status = iree_string_builder_append_string(&request.body, body);
    loom_serve_text_stream_t* value = nullptr;
    if (iree_status_is_ok(status)) {
      status = loom_serve_text_service_try_open(service_, &request, &value);
    }
    iree_string_builder_deinitialize(&request.body);
    stream->reset(value);
    IREE_ASSERT_OK(status);
    ASSERT_NE(*stream, nullptr);
  }

  void AwaitBlock(const NativeStream& stream) {
    ASSERT_NO_FATAL_FAILURE(DriveUntil([&] {
      return loom_serve_text_stream_peek(stream.get()) ||
             loom_serve_text_stream_is_complete(stream.get());
    }));
    IREE_ASSERT_OK(loom_serve_text_stream_status(stream.get()));
  }

  void ReadFinal(const NativeStream& stream,
                 const std::string& expected = Oracle::Complete("abxa", 4),
                 size_t input_tokens = 4, size_t output_tokens = 4) {
    std::string content;
    while (!loom_serve_text_stream_is_complete(stream.get())) {
      ASSERT_NO_FATAL_FAILURE(AwaitBlock(stream));
      if (const auto* block = loom_serve_text_stream_peek(stream.get())) {
        for (iree_host_size_t i = 0;
             i < loom_serve_text_stream_block_count(block); ++i) {
          const auto span = loom_serve_text_stream_block_at(block, i);
          EXPECT_EQ(span.kind, LOOM_SERVE_TEXT_CHAT_OUTPUT_CONTENT);
          content.append(span.text.data, span.text.size);
        }
        loom_serve_text_stream_consume(stream.get());
      }
    }
    IREE_ASSERT_OK(loom_serve_text_stream_status(stream.get()));
    const auto* result = loom_serve_text_stream_result(stream.get());
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(String(result->content), content);
    EXPECT_EQ(content, expected);
    EXPECT_EQ(String(result->response.model_name), "fixture-text");
    EXPECT_EQ(result->response.usage.input_tokens, input_tokens);
    EXPECT_EQ(result->response.usage.output_tokens, output_tokens);
  }

  static constexpr iree_string_view_t kRequest = IREE_SVL(
      R"({"model":"fixture-text","messages":[{"role":"user","content":"abxa"}],"max_tokens":4,"stream":true})");
  // Outer owner wait; worker callbacks only post to this notification.
  iree_notification_t notification_;
  // Process-owned resources shared with other service instances.
  loom_runtime_device_group_t* group_ = nullptr;
  // Borrowed device record used by every registration in this fixture.
  loom_runtime_device_t* device_ = nullptr;
  // Source root borrowed by preparation through its terminal outcome.
  std::string source_directory_;
  // Parameter path borrowed by preparation, backed by the outer test scratch.
  std::string parameters_path_;
  // Owned preparation, ready service, optional image, and final model
  // retirement.
  loom_serve_text_registration_t* registration_ = nullptr;
  // Borrowed admission/session state, alive until registration retirement.
  loom_serve_text_service_t* service_ = nullptr;
  // Stable borrowed service slot, detached after final driver collection.
  loom_runtime_service_t runtime_ = {};
  // Allocation-free progression shared with every other native service.
  loom_runtime_driver_t driver_ = {};
};

TEST_F(TextRegistrationTest, PressureDoesNotConstructDelivery) {
  NativeStream first(nullptr, loom_serve_text_stream_close);
  NativeStream second(nullptr, loom_serve_text_stream_close);
  ASSERT_NO_FATAL_FAILURE(Open("first", &first));
  ASSERT_NO_FATAL_FAILURE(Open("second", &second));
  struct Input {
    // Owning input, moved only by successful receipt construction.
    loom_serve_text_request_t request = {};
    // Detached delivery handle returned by the real stream constructor.
    loom_serve_text_stream_t* stream = nullptr;
    // Number of times native admission permits receipt construction.
    int constructions = 0;
  } input;
  iree_string_builder_initialize(iree_allocator_system(), &input.request.body);
  IREE_ASSERT_OK(
      iree_string_builder_append_string(&input.request.body, kRequest));
  const auto body = iree_string_builder_view(&input.request.body);
  auto create = [](void* self, iree_allocator_t allocator,
                   loom_serve_text_consumer_t* out_consumer) {
    auto* input = static_cast<Input*>(self);
    ++input->constructions;
    return loom_serve_text_stream_allocate(IREE_SV("fixture-text"),
                                           &input->request, allocator,
                                           &input->stream, out_consumer);
  };
  bool accepted = false;
  IREE_ASSERT_OK(loom_serve_text_service_try_accept(
      service_, body, {}, {}, iree_time_now(), create, &input, &accepted));
  EXPECT_FALSE(accepted);
  EXPECT_EQ(input.constructions, 0);
  EXPECT_EQ(iree_string_builder_buffer(&input.request.body), body.data);
  EXPECT_EQ(iree_string_builder_size(&input.request.body), body.size);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        loom_serve_text_service_try_accept(
                            service_, body, {}, IREE_SV("missing"),
                            iree_time_now(), create, &input, &accepted));
  EXPECT_FALSE(accepted);
  EXPECT_EQ(input.constructions, 0);
  EXPECT_EQ(
      loom_serve_text_service_interface(service_).telemetry.counters->requests,
      2u);

  first.reset();
  second.reset();
  ASSERT_NO_FATAL_FAILURE(DriveUntil(
      [&] { return !runtime_.vtable->has_requests(runtime_.self); }));
  IREE_ASSERT_OK(loom_serve_text_service_try_accept(
      service_, body, {}, {}, iree_time_now(), create, &input, &accepted));
  NativeStream stream(input.stream, loom_serve_text_stream_close);
  iree_string_builder_deinitialize(&input.request.body);
  ASSERT_TRUE(accepted);
  EXPECT_EQ(input.constructions, 1);
  ASSERT_NE(stream, nullptr);
  ASSERT_NO_FATAL_FAILURE(ReadFinal(stream));
}

TEST_F(TextRegistrationTest, StalledPeerAndDetachedResults) {
  using NativeCount =
      std::unique_ptr<loom_serve_text_input_count_call_t,
                      decltype(&loom_serve_text_input_count_call_close)>;
  iree_string_builder_t body;
  iree_string_builder_initialize(iree_allocator_system(), &body);
  IREE_ASSERT_OK(iree_string_builder_append_string(&body, kRequest));
  loom_serve_text_input_count_call_t* count_value = nullptr;
  auto status = loom_serve_text_service_try_open_input_count(service_, &body,
                                                             &count_value);
  iree_string_builder_deinitialize(&body);
  NativeCount count(count_value, loom_serve_text_input_count_call_close);
  IREE_ASSERT_OK(status);
  ASSERT_NE(count, nullptr);
  ASSERT_NO_FATAL_FAILURE(DriveUntil([&] {
    return loom_serve_text_input_count_call_is_complete(count.get());
  }));
  IREE_ASSERT_OK(loom_serve_text_input_count_call_status(count.get()));
  EXPECT_EQ(loom_serve_text_input_count_call_result(count.get()), 4u);
  const auto parameters =
      loom_serve_text_registration_parameters(registration_);
  EXPECT_EQ(parameters.activity.activations, 0u);
  EXPECT_EQ(parameters.activity.read_bytes, 0u);

  NativeStream held(nullptr, loom_serve_text_stream_close);
  ASSERT_NO_FATAL_FAILURE(Open("held", &held));
  ASSERT_NO_FATAL_FAILURE(AwaitBlock(held));
  const auto* block = loom_serve_text_stream_peek(held.get());
  ASSERT_NE(block, nullptr);
  ASSERT_GT(loom_serve_text_stream_block_count(block), 0u);
  const auto span = loom_serve_text_stream_block_at(block, 0);
  const std::string held_text = String(span.text);
  NativeStream peer(nullptr, loom_serve_text_stream_close);
  ASSERT_NO_FATAL_FAILURE(Open("peer", &peer));
  ASSERT_NO_FATAL_FAILURE(ReadFinal(peer));
  EXPECT_EQ(loom_serve_text_stream_peek(held.get()), block);
  EXPECT_EQ(String(span.text), held_text);
  EXPECT_EQ(loom_serve_text_stream_result(held.get()), nullptr);

  const loom_serve_text_checkpoint_request_t pin = {
      .action = LOOM_SERVE_TEXT_CHECKPOINT_ACTION_PIN,
      .name = IREE_SV("saved"),
      .session = IREE_SV("peer"),
  };
  loom_serve_text_checkpoint_control_t* control_value = nullptr;
  status = loom_serve_text_service_try_open_checkpoint(service_, &pin,
                                                       &control_value);
  NativeControl control(control_value,
                        loom_serve_text_checkpoint_control_close);
  IREE_ASSERT_OK(status);
  ASSERT_NE(control, nullptr);
  ASSERT_NO_FATAL_FAILURE(DriveUntil([&] {
    return loom_serve_text_checkpoint_control_is_complete(control.get());
  }));
  IREE_ASSERT_OK(loom_serve_text_checkpoint_control_status(control.get()));
  ASSERT_NO_FATAL_FAILURE(ReadFinal(held));
  ASSERT_NO_FATAL_FAILURE(Retire());
  EXPECT_EQ(String(loom_serve_text_stream_result(held.get())->content),
            Oracle::Complete("abxa", 4));
  EXPECT_EQ(String(loom_serve_text_stream_result(peer.get())->content),
            Oracle::Complete("abxa", 4));
  EXPECT_EQ(loom_serve_text_input_count_call_result(count.get()), 4u);
  const auto* checkpoint =
      loom_serve_text_checkpoint_control_result(control.get());
  ASSERT_NE(checkpoint, nullptr);
  EXPECT_GT(checkpoint->position, 0u);
}

TEST_F(TextRegistrationTest, CloseQueuedInputAndStopWithBorrowedOutput) {
  NativeStream closed(nullptr, loom_serve_text_stream_close);
  ASSERT_NO_FATAL_FAILURE(Open("closed", &closed));
  closed.reset();
  NativeStream held(nullptr, loom_serve_text_stream_close);
  ASSERT_NO_FATAL_FAILURE(Open("held", &held));
  ASSERT_NO_FATAL_FAILURE(AwaitBlock(held));
  const auto* block = loom_serve_text_stream_peek(held.get());
  ASSERT_NE(block, nullptr);
  ASSERT_GT(loom_serve_text_stream_block_count(block), 0u);
  const auto span = loom_serve_text_stream_block_at(block, 0);
  const std::string held_text = String(span.text);
  ASSERT_NO_FATAL_FAILURE(Retire());
  EXPECT_TRUE(loom_serve_text_stream_is_complete(held.get()));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                        loom_serve_text_stream_status(held.get()));
  EXPECT_EQ(loom_serve_text_stream_result(held.get()), nullptr);
  EXPECT_EQ(String(span.text), held_text);
}

class TextPreparationTest : public TextRegistrationTest {
 protected:
  void SetUp() override { ASSERT_NO_FATAL_FAILURE(Initialize()); }
};

TEST_F(TextPreparationTest, CancellationJoinsAcceptedSourcePreparation) {
  ASSERT_NO_FATAL_FAILURE(Allocate());
  bool complete = false;
  loom_runtime_compilation_report_t report = {};
  IREE_ASSERT_OK(loom_serve_text_registration_prepare(
      registration_, 0, &complete, &service_, &report));
  ASSERT_FALSE(complete);
  ASSERT_EQ(service_, nullptr);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                        Prepare(LOOM_SERVE_TEXT_LOAD_FLAG_CANCEL));
  EXPECT_EQ(service_, nullptr);
  ASSERT_NO_FATAL_FAILURE(Retire());
}

TEST_F(TextPreparationTest, FailedSourceRetiresWithoutPublishingService) {
  ASSERT_NO_FATAL_FAILURE(Allocate(LOOM_SERVE_TEXT_MODE_GENERATE, nullptr,
                                   IREE_SV("missing.loom")));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND, Prepare());
  EXPECT_EQ(service_, nullptr);
  ASSERT_NO_FATAL_FAILURE(Retire());
}

class TextSequenceTest : public TextPreparationTest {
 protected:
  struct Receipt {
    // Independently retained logical sequence identity.
    std::string handle;
    // Exact source-consumed IDs, copied during each finite CPU output loan.
    std::vector<int32_t> tokens;
    // Last source output, retained after owner retirement.
    std::vector<uint8_t> bytes;
    // CPU publication awaiting its owner-side collection.
    bool pending = false;
    // The source marked its last emitted chunk final.
    bool final = false;
    // Owner has relinquished the receipt and every input/output borrower.
    bool complete = false;
    // Terminal request status, published with completion.
    iree_status_code_t code = IREE_STATUS_OK;
  };

  // Callback state outlives TearDown, including an early test assertion.
  Receipt receipt_;
};

TEST_F(TextSequenceTest, NativeSequencePreservesConsumedIdsAndOutput) {
  ASSERT_NO_FATAL_FAILURE(Allocate(LOOM_SERVE_TEXT_MODE_ENCODE));
  IREE_ASSERT_OK(Prepare());
  Receipt& receipt = receipt_;
  static const loom_serve_text_consumer_vtable_t vtable = {
      .is_open = [](void*) { return true; },
      .is_blocked = [](void*) { return false; },
      .begin_sequence =
          [](void* self, iree_string_view_t handle, iree_host_size_t position) {
            auto* receipt = static_cast<Receipt*>(self);
            receipt->handle = String(handle);
            EXPECT_EQ(position, 0u);
            return iree_ok_status();
          },
      .append_sequence =
          [](void* self, iree_string_view_t handle, iree_host_size_t position,
             iree_host_size_t count, const int32_t* tokens,
             iree_const_byte_span_t bytes, bool final) {
            auto* receipt = static_cast<Receipt*>(self);
            EXPECT_EQ(String(handle), receipt->handle);
            EXPECT_EQ(position, receipt->tokens.size());
            receipt->tokens.insert(receipt->tokens.end(), tokens,
                                   tokens + count);
            receipt->bytes.assign(bytes.data, bytes.data + bytes.data_length);
            receipt->pending = true;
            receipt->final = final;
            return iree_ok_status();
          },
      .publish =
          [](void* self, bool* ready, bool* progress) {
            auto* receipt = static_cast<Receipt*>(self);
            *ready = true;
            *progress = std::exchange(receipt->pending, false);
            return iree_ok_status();
          },
      .reject =
          [](void* self, iree_status_t status, loom_serve_text_telemetry_t*,
             const loom_serve_text_receipt_t*) {
            auto* receipt = static_cast<Receipt*>(self);
            receipt->code = iree_status_code(status);
            iree_status_free(status);
            receipt->complete = true;
            return iree_ok_status();
          },
      .end =
          [](void* self, bool, iree_status_code_t code) {
            auto* receipt = static_cast<Receipt*>(self);
            receipt->code = code;
            receipt->complete = true;
          },
  };
  bool accepted = false;
  IREE_ASSERT_OK(loom_serve_text_service_try_accept(
      service_, IREE_SV(R"({"position":0,"token_ids":[0,1,2,0]})"), {}, {},
      iree_time_now(),
      [](void* self, iree_allocator_t, loom_serve_text_consumer_t* consumer) {
        *consumer = {self, &vtable};
        return iree_ok_status();
      },
      &receipt, &accepted));
  ASSERT_TRUE(accepted);
  EXPECT_FALSE(receipt.complete);
  EXPECT_TRUE(receipt.tokens.empty());
  ASSERT_NO_FATAL_FAILURE(DriveUntil([&] { return receipt.complete; }));
  EXPECT_EQ(receipt.code, IREE_STATUS_OK);
  EXPECT_TRUE(receipt.final);
  ASSERT_NO_FATAL_FAILURE(Retire());
  EXPECT_EQ(receipt.tokens, (std::vector<int32_t>{0, 1, 2, 0}));
  uint32_t hidden = 0;
  for (int32_t token : receipt.tokens) {
    hidden = (5 * hidden + token + 2) % 97;
  }
  ASSERT_EQ(receipt.bytes.size(), 4u);
  EXPECT_EQ(iree_unaligned_load_le_u16(receipt.bytes.data()), hidden >> 16);
  EXPECT_EQ(iree_unaligned_load_le_u16(receipt.bytes.data() + 2),
            hidden & 0xFFFF);
}

class TextImageRegistrationTest : public TextRegistrationTest {
 protected:
  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(Initialize());
    ASSERT_NO_FATAL_FAILURE(AllocateImage());
    IREE_ASSERT_OK(Prepare());
  }

  void AllocateImage(
      iree_string_view_t options = IREE_SV("{\"batch_units\":2}")) {
    const char* source = std::getenv("LOOM_SERVE_TEST_PROJECTOR_SOURCE");
    ASSERT_NE(source, nullptr);
    const std::string path = source;
    image_directory_ = path.substr(0, path.find_last_of('/'));
    static const iree_string_view_t providers[] = {IREE_SV("project.loom")};
    const loom_service_finite_model_options_t image = {
        .sources =
            {
                .base_directory =
                    iree_make_cstring_view(image_directory_.c_str()),
                .prepare = IREE_SV("prepare.loom"),
                .control = IREE_SV("control.loom"),
                .count = IREE_ARRAYSIZE(providers),
                .values = providers,
            },
        .asset_directory = IREE_SV("."),
        .options = options,
        .specialization_cache = {2, 1024 * 1024},
    };
    ASSERT_NO_FATAL_FAILURE(Allocate(LOOM_SERVE_TEXT_MODE_GENERATE, &image));
  }

  static constexpr iree_string_view_t kImageRequest = IREE_SVL(
      R"({"model":"fixture-text","messages":[{"role":"user","content":[{"type":"text","text":"a"},{"type":"image_url","image_url":{"url":"data:image/x-portable-arbitrarymap;base64,UDcKV0lEVEggMgpIRUlHSFQgMgpERVBUSCAzCk1BWFZBTCAyNTUKVFVQTFRZUEUgUkdCCkVOREhEUgr/AAAA/wAAAP////8=","detail":"low"}},{"type":"text","text":"b"}]},{"role":"user","content":[{"type":"text","text":"x"},{"type":"image_url","image_url":{"url":"data:image/x-portable-arbitrarymap;base64,UDcKV0lEVEggMgpIRUlHSFQgMgpERVBUSCA0Ck1BWFZBTCAyNTUKVFVQTFRZUEUgUkdCX0FMUEhBCkVOREhEUgoKFB4AKDI8gEZQWv9kbnjI","detail":"low"}},{"type":"text","text":"a"}]},{"role":"user","content":[{"type":"text","text":"b"},{"type":"image_url","image_url":{"url":"data:image/x-portable-arbitrarymap;base64,UDcKV0lEVEggMgpIRUlHSFQgMgpERVBUSCAzCk1BWFZBTCAyNTUKVFVQTFRZUEUgUkdCCkVOREhEUgr/AAAA/wAAAP////8=","detail":"low"}},{"type":"text","text":"x"}]}],"max_tokens":4,"stream":true})");
  static constexpr iree_string_view_t kImagePeerRequest = IREE_SVL(
      R"({"model":"fixture-text","messages":[{"role":"user","content":[{"type":"text","text":"a"},{"type":"image_url","image_url":{"url":"data:image/x-portable-arbitrarymap;base64,UDcKV0lEVEggMgpIRUlHSFQgMgpERVBUSCAzCk1BWFZBTCAyNTUKVFVQTFRZUEUgUkdCCkVOREhEUgr/AAAA/wAAAP////8=","detail":"low"}},{"type":"text","text":"b"}]}],"max_tokens":4,"stream":true})");
  // Optional image source views survive its delayed preparation and retirement.
  std::string image_directory_;
};

TEST_F(TextImageRegistrationTest,
       ImageCohortsRetainNativeOutputWhileLatePeerCompletes) {
  if (!loom_serve_media_image_codec(LOOM_SERVE_MEDIA_IMAGE_FORMAT_PAM)) {
    GTEST_SKIP() << "PAM decoder is not enabled";
  }
  const auto* image = loom_serve_text_registration_image(registration_);
  ASSERT_NE(image, nullptr);
  EXPECT_EQ(loom_service_finite_component_phase(image),
            LOOM_SERVICE_FINITE_COMPONENT_DECLARED);
  NativeStream peer(nullptr, loom_serve_text_stream_close);
  ASSERT_NO_FATAL_FAILURE(Open("plain", &peer));
  ASSERT_NO_FATAL_FAILURE(ReadFinal(peer));
  EXPECT_EQ(loom_service_finite_component_phase(image),
            LOOM_SERVICE_FINITE_COMPONENT_DECLARED);

  NativeStream image_peer(nullptr, loom_serve_text_stream_close);
  ASSERT_NO_FATAL_FAILURE(Open("image-peer", &image_peer, kImagePeerRequest));
  NativeStream vision(nullptr, loom_serve_text_stream_close);
  ASSERT_NO_FATAL_FAILURE(Open("vision", &vision, kImageRequest));
  // Acceptance queues only the request, including its optional image demand.
  EXPECT_EQ(loom_service_finite_component_phase(image),
            LOOM_SERVICE_FINITE_COMPONENT_DECLARED);
  auto feature = [](float value) {
    int32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
  };
  Oracle oracle;
  // The peer and first projection share a two-row result. The remaining
  // projections form a private two-row cohort. The source includes the cohort
  // width in each feature, so serial projection yields different final text.
  for (int32_t token :
       {0, feature(512), 1, 2, feature(12), 0, 1, feature(512), 2}) {
    oracle.Consume(token);
  }
  std::string expected;
  size_t output_tokens = 0;
  for (size_t i = 0; i < 4; ++i) {
    const int32_t token = oracle.Predict();
    ++output_tokens;  // EOS counts as an emitted token without decoded content.
    if (token == 3) {
      break;
    }
    expected.push_back("abx"[token]);
    oracle.Consume(token);
  }
  ASSERT_NO_FATAL_FAILURE(AwaitBlock(image_peer));
  ASSERT_NE(loom_serve_text_stream_peek(image_peer.get()), nullptr);
  image_peer.reset();
  ASSERT_NO_FATAL_FAILURE(AwaitBlock(vision));
  const auto* held = loom_serve_text_stream_peek(vision.get());
  ASSERT_NE(held, nullptr);
  NativeStream late(nullptr, loom_serve_text_stream_close);
  ASSERT_NO_FATAL_FAILURE(Open("late", &late));
  NativeStream cancelled(nullptr, loom_serve_text_stream_close);
  ASSERT_NO_FATAL_FAILURE(Open("cancelled", &cancelled));
  cancelled.reset();
  ASSERT_NO_FATAL_FAILURE(ReadFinal(late));
  EXPECT_EQ(loom_serve_text_stream_peek(vision.get()), held);
  ASSERT_NO_FATAL_FAILURE(ReadFinal(vision, expected, 9, output_tokens));
  EXPECT_EQ(loom_service_finite_component_phase(image),
            LOOM_SERVICE_FINITE_COMPONENT_READY);
  ASSERT_NO_FATAL_FAILURE(Retire());
  EXPECT_EQ(String(loom_serve_text_stream_result(vision.get())->content),
            expected);
}

class FailedTextImageRegistrationTest : public TextImageRegistrationTest {
 protected:
  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(Initialize());
    ASSERT_NO_FATAL_FAILURE(AllocateImage(IREE_SV("{\"batch_units\":0}")));
    IREE_ASSERT_OK(Prepare());
  }
};

TEST_F(FailedTextImageRegistrationTest, ImageSourceFailureLeavesTextUsable) {
  if (!loom_serve_media_image_codec(LOOM_SERVE_MEDIA_IMAGE_FORMAT_PAM)) {
    GTEST_SKIP() << "PAM decoder is not enabled";
  }
  NativeStream vision(nullptr, loom_serve_text_stream_close);
  ASSERT_NO_FATAL_FAILURE(Open("vision", &vision, kImageRequest));
  ASSERT_NO_FATAL_FAILURE(DriveUntil(
      [&] { return loom_serve_text_stream_is_complete(vision.get()); }));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        loom_serve_text_stream_status(vision.get()));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_status_clone(loom_service_finite_component_status(
          loom_serve_text_registration_image(registration_))));
  EXPECT_EQ(loom_service_finite_component_phase(
                loom_serve_text_registration_image(registration_)),
            LOOM_SERVICE_FINITE_COMPONENT_FAILED);
  NativeStream peer(nullptr, loom_serve_text_stream_close);
  ASSERT_NO_FATAL_FAILURE(Open("plain", &peer));
  ASSERT_NO_FATAL_FAILURE(ReadFinal(peer));
}

}  // namespace
