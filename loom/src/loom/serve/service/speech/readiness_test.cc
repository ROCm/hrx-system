// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "iree/base/threading/notification.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/runtime/device/device.h"
#include "loom/serve/runtime/execution/execution.h"
#include "loom/serve/server/registry.h"
#include "loom/serve/server/router.h"
#include "loom/serve/service/speech/service.h"
#include "loom/serve/service/speech/service_test.h"
#include "loom/serve/service/speech/stream.h"
#include "loom/serve/telemetry/latency.h"
#include "loom/serve/testing/http_harness.h"
#include "loom/serve/testing/stalled_http_client.h"
#include "loom/serve/testing/worker_gate.h"
#include "loom/serve/transport/http/listener.h"
#include "loom/serve/transport/http/server.h"

namespace {

using loom_serve::testing::AcquireApplication;
using loom_serve::testing::ObservationCount;
using loom_serve::testing::OpenSpeech;
using loom_serve::testing::SpeechStream;
using loom_serve::testing::SpeechWorkerCount;

// The numerical dependency is replaced; accepted HAL transfers, proactor
// waits, CPU tasks, TCP credit and service retirement remain production paths.
class SpeechReadinessTest : public ::testing::Test {
 protected:
  struct loom_serve_test_speech_invocation_t {
    // Stable row retained independently of its ordinal in the current cohort.
    size_t row = 0;
    // Source cursor advances only after retired feedback collection.
    uint64_t block = 0;
    // Text-derived output length, independent of neighboring requests.
    uint64_t blocks = 0;
    // Immutable voice waveform borrowed by real upload operations.
    std::vector<uint8_t> input;
    // Per-request feedback borrowed through accepted queue retirement.
    std::vector<uint8_t> output;
    // Actual queue allocation, retained through failure retirement.
    iree_hal_buffer_t* buffer = nullptr;
  };

  void Initialize(size_t frames = 3, uint64_t blocks = 4,
                  uint64_t held_block = 1, size_t capacity = 1,
                  size_t held_members = 1, bool start = true,
                  loom_serve_speech_format_t format = {24000, 1}) {
    frames_ = frames;
    format_ = format;
    blocks_ = blocks;
    held_block_ = held_block;
    held_members_ = held_members;
    batch_.reserve(capacity);
    IREE_ASSERT_OK(AcquireApplication(allocator_, &host_, &group_));
    auto* device = loom_serve_device_group_at(group_, 0);
    IREE_ASSERT_OK(loom_serve_device_create_execution(device, &execution_));
    IREE_ASSERT_OK(loom_serve_execution_reserve_workspace(
        execution_, frames * format.channels * sizeof(float) * capacity, 16));
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        loom_serve_device_handle(device), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &gate_));
    const auto transport_options = loom_serve_http_server_options_default();
    IREE_ASSERT_OK(loom_serve_http_server_allocate(
        loom_serve_host_proactor_pool(host_), &transport_options, allocator_,
        &server_));
    auto listener_options = loom_serve_http_listener_options_default();
    listener_options.address = IREE_SV("127.0.0.1:0");
    IREE_ASSERT_OK(loom_serve_http_listener_allocate(server_, &listener_options,
                                                     allocator_, &listener_));
    IREE_ASSERT_OK(http_.Initialize(server_, listener_));
    const loom_serve_speech_generator_t generator = {
        .self = this,
        .format = format,
        .block_capacity = frames_,
        .request_capacity = capacity,
        .batch_units = capacity,
        .prepare =
            [](void* self, iree_host_size_t row,
               const loom_serve_speech_request_t* request, const void* voice,
               void** out) {
              auto* test = static_cast<SpeechReadinessTest*>(self);
              auto invocation =
                  std::make_unique<loom_serve_test_speech_invocation_t>();
              invocation->row = row;
              invocation->blocks = iree_string_builder_size(&request->input);
              invocation->input.resize(test->frames_ * test->format_.channels *
                                       4);
              invocation->output.resize(invocation->input.size());
              const float sample =
                  iree_string_view_equal(
                      iree_string_builder_view(&request->voice),
                      IREE_SV("negative"))
                      ? -0.5f
                      : 0.5f;
              for (size_t i = 0; i < test->frames_ * test->format_.channels;
                   ++i) {
                iree_unaligned_store_le_f32(invocation->input.data() + i * 4,
                                            sample);
              }
              *out = invocation.get();
              test->invocations_[row] = std::move(invocation);
              test->row_submitted_[row].store(0);
              return iree_ok_status();
            },
        .discard =
            [](void* self, void* invocation) {
              static_cast<SpeechReadinessTest*>(self)
                  ->invocations_
                      [static_cast<loom_serve_test_speech_invocation_t*>(
                           invocation)
                           ->row]
                  .reset();
            },
        .try_admit =
            [](void* self, void*, loom_serve_residency_admission_t* out,
               loom_serve_execution_frontier_t* frontier) {
              auto* test = static_cast<SpeechReadinessTest*>(self);
              ++test->admission_attempts_;
              *out = test->admission_;
              *frontier = loom_serve_execution_frontier(test->execution_);
              return iree_ok_status();
            },
        .complete_admit =
            [](void* self, void*) {
              auto* test = static_cast<SpeechReadinessTest*>(self);
              ++test->admission_completions_;
            },
        .prepare_work =
            [](void*, void*, loom_serve_speech_work_t* out) {
              *out = {.units = 1, .tag = 0};
              return iree_ok_status();
            },
        .prepare_batch =
            [](void* self, iree_host_size_t count, void* const* invocations) {
              auto* test = static_cast<SpeechReadinessTest*>(self);
              test->batch_.clear();
              for (size_t i = 0; i < count; ++i) {
                test->batch_.push_back(
                    static_cast<loom_serve_test_speech_invocation_t*>(
                        invocations[i]));
              }
              return iree_ok_status();
            },
        .submit =
            [](void* self, loom_serve_execution_frontier_t* out) {
              return static_cast<SpeechReadinessTest*>(self)->Submit(out);
            },
        .collect =
            [](void* self, iree_host_size_t ordinal,
               loom_serve_speech_block_t* out) {
              auto* test = static_cast<SpeechReadinessTest*>(self);
              auto& invocation = *test->batch_[ordinal];
              ++test->row_collected_[invocation.row];
              if (invocation.row == 0 &&
                  invocation.block == test->output_failure_block_) {
                if (test->fail_collection_) {
                  return iree_make_status(IREE_STATUS_DATA_LOSS,
                                          "speech source completion failed");
                }
                if (test->nonfinite_output_) {
                  iree_unaligned_store_le_f32(
                      invocation.output.data(),
                      std::numeric_limits<float>::quiet_NaN());
                }
              }
              const size_t frames = test->empty_output_ ? 0 : test->frames_;
              *out = {
                  .position = invocation.block * frames,
                  .frame_count = frames,
                  .samples = iree_make_const_byte_span(
                      invocation.output.data(),
                      frames * test->format_.channels * sizeof(float)),
                  .complete = invocation.block + 1 == invocation.blocks,
              };
              ++invocation.block;
              return iree_ok_status();
            },
        .release_batch =
            [](void* self) {
              auto* test = static_cast<SpeechReadinessTest*>(self);
              for (auto* invocation : test->batch_) {
                iree_hal_buffer_release(invocation->buffer);
                invocation->buffer = nullptr;
              }
              test->batch_.clear();
            },
        .release =
            [](void* self, void* invocation, bool* out_complete,
               loom_serve_execution_frontier_t* out_frontier) {
              auto* test = static_cast<SpeechReadinessTest*>(self);
              test->invocations_
                  [static_cast<loom_serve_test_speech_invocation_t*>(invocation)
                       ->row]
                      .reset();
              ++test->released_;
              test->http_.Notify();
              *out_complete = true;
              *out_frontier = {};
              return iree_ok_status();
            },
    };
    const loom_serve_speech_service_options_t service_options = {
        .name = IREE_SV("speech"),
        .pending_capacity = 1,
        .heartbeat_interval = 0,
    };
    IREE_ASSERT_OK(loom_serve_speech_service_allocate(
        host_, generator, &service_options, nullptr, allocator_, &service_));
    actual_ = loom_serve_speech_service_interface(service_);
    // Observe completed owner turns without replacing any service behavior.
    wrapped_ = actual_;
    auto& wrapped = wrapped_;
    wrapped.self = this;
    wrapped.set_observer = [](void* self, loom_serve_wake_callback_t wake) {
      auto* test = static_cast<SpeechReadinessTest*>(self);
      test->actual_.set_observer(test->actual_.self, wake);
    };
    wrapped.is_pending = [](void* self) {
      auto* test = static_cast<SpeechReadinessTest*>(self);
      return test->actual_.is_pending(test->actual_.self);
    };
    wrapped.is_active = [](void* self) {
      auto* test = static_cast<SpeechReadinessTest*>(self);
      return test->actual_.is_active(test->actual_.self);
    };
    wrapped.accept = [](void* self, loom_serve_http_connection_t* connection,
                        const loom_serve_http_request_t* request,
                        loom_serve_service_flags_t flags) {
      auto* test = static_cast<SpeechReadinessTest*>(self);
      return test->actual_.accept(test->actual_.self, connection, request,
                                  flags);
    };
    wrapped.advance = [](void* self, loom_serve_service_flags_t flags,
                         bool* progress) {
      auto* test = static_cast<SpeechReadinessTest*>(self);
      iree_status_t status =
          test->actual_.advance(test->actual_.self, flags, progress);
      if (iree_status_is_ok(status) &&
          std::string(test->actual_.snapshot(test->actual_.self).phase) ==
              "delivering") {
        test->credit_wait_.store(true, std::memory_order_release);
        test->http_.Notify();
      }
      return status;
    };
    wrapped.snapshot = [](void* self) {
      auto* test = static_cast<SpeechReadinessTest*>(self);
      return test->actual_.snapshot(test->actual_.self);
    };
    wrapped.inspection.request_at =
        [](void* self, iree_host_size_t index,
           loom_serve_http_request_snapshot_t* out) {
          auto* test = static_cast<SpeechReadinessTest*>(self);
          return test->actual_.inspection.request_at(test->actual_.self, index,
                                                     out);
        };
    if (start) {
      StartRouter();
    } else {
      actual_.set_observer(actual_.self,
                           {[](void* self) {
                              iree_notification_post(
                                  static_cast<iree_notification_t*>(self),
                                  IREE_ALL_WAITERS);
                            },
                            loom_serve_http_server_notification(server_)});
    }
  }

  template <typename Predicate>
  void AdvanceUntil(Predicate ready, loom_serve_service_flags_t flags = 0) {
    auto* notification = loom_serve_http_server_notification(server_);
    while (!ready()) {
      const auto ticket = iree_notification_prepare_wait(notification);
      bool progress = false;
      auto status = actual_.advance(actual_.self, flags, &progress);
      if (ready() || progress || !iree_status_is_ok(status)) {
        iree_notification_cancel_wait(notification);
      } else {
        iree_notification_commit_wait(notification, ticket, IREE_DURATION_ZERO,
                                      IREE_TIME_INFINITE_FUTURE);
      }
      IREE_ASSERT_OK(status);
    }
  }

  void StartRouter() {
    http_.Start([this]() -> iree_status_t {
      loom_serve_registry_t* registry = nullptr;
      IREE_RETURN_IF_ERROR(loom_serve_registry_allocate(1, &wrapped_, 0, {},
                                                        allocator_, &registry));
      iree_status_t status = loom_serve_router_run(
          loom_serve_host_executor(host_), server_, registry,
          {
              .request_quota = 4,
              .holding_capacity = 4,
              .resources = {.device_group = group_},
          },
          allocator_);
      return iree_status_join(status, loom_serve_registry_free(registry));
    });
  }

  iree_status_t Submit(loom_serve_execution_frontier_t* out) {
    iree_status_t status = iree_ok_status();
    bool held = false;
    std::array<iree_hal_transfer_operation_t, 4> uploads = {};
    std::array<iree_hal_transfer_operation_t, 4> downloads = {};
    for (size_t i = 0; i < batch_.size() && iree_status_is_ok(status); ++i) {
      auto& invocation = *batch_[i];
      const auto reservation =
          iree_hal_make_pool_reservation_request(invocation.input.size(), 16);
      status = loom_serve_execution_alloca(execution_, 1, &reservation,
                                           &invocation.buffer);
      held |= invocation.block == held_block_;
      uploads[i] = {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
          .upload =
              {
                  .source = invocation.input.data(),
                  .target_buffer = invocation.buffer,
                  .length = invocation.input.size(),
              },
      };
      downloads[i] = {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
          .download =
              {
                  .source_buffer = invocation.buffer,
                  .target = invocation.output.data(),
                  .length = invocation.output.size(),
              },
      };
    }
    held &= batch_.size() >= held_members_;
    if (iree_status_is_ok(status) && held) {
      uint64_t value = 1;
      status = loom_serve_execution_barrier(execution_, {1, &gate_, &value});
    }
    uint64_t value = 0;
    if (iree_status_is_ok(status)) {
      status = loom_serve_execution_transfer(execution_, batch_.size(),
                                             uploads.data(), &value);
    }
    if (iree_status_is_ok(status)) {
      status = loom_serve_execution_feedback(execution_, batch_.size(),
                                             downloads.data(), &value);
    }
    if (iree_status_is_ok(status)) {
      auto frontier = loom_serve_execution_frontier(execution_);
      status = loom_serve_execution_barrier(
          execution_, {1, &frontier.semaphores[1], &frontier.values[1]});
    }
    for (size_t i = 0; i < batch_.size() && iree_status_is_ok(status); ++i) {
      status =
          loom_serve_execution_dealloca(execution_, batch_[i]->buffer, &value);
    }
    if (iree_status_is_ok(status)) {
      *out = loom_serve_execution_frontier(execution_);
      for (auto* invocation : batch_) {
        row_submitted_[invocation->row].fetch_add(1);
      }
      largest_batch_.store(std::max(largest_batch_.load(), batch_.size()));
      ++submitted_;
      if (held) {
        held_.store(true, std::memory_order_release);
      }
      http_.Notify();
    }
    return status;
  }

  void ReleaseDependency(bool fail = false) {
    if (!gate_ || gate_released_) {
      return;
    }
    gate_released_ = true;
    failed_ = fail;
    if (fail) {
      iree_hal_semaphore_fail(gate_, iree_make_status(IREE_STATUS_ABORTED,
                                                      "speech device witness"));
    } else {
      IREE_EXPECT_OK(iree_hal_semaphore_signal(gate_, 1, nullptr));
    }
  }

  void CheckStatus(iree_status_t status) {
    if (failed_) {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, status);
    } else {
      IREE_EXPECT_OK(status);
    }
  }

  void TearDown() override {
    if (server_) {
      loom_serve_http_server_request_stop(server_);
    }
    ReleaseDependency();
    if (http_.joinable()) {
      CheckStatus(http_.Join());
    } else if (service_ && !failed_) {
      AdvanceUntil([&] { return !actual_.is_pending(actual_.self); },
                   LOOM_SERVE_SERVICE_FLAG_STOPPING);
    }
    if (service_) {
      actual_.set_observer(actual_.self, {});
    }
    loom_serve_speech_service_free(service_);
    // Failed readiness transfers request ownership to the model. Release its
    // views, join actual queue retirement, then release borrowed host bytes.
    for (auto& invocation : invocations_) {
      if (invocation) {
        iree_hal_buffer_release(invocation->buffer);
        invocation->buffer = nullptr;
      }
    }
    if (execution_) {
      CheckStatus(loom_serve_execution_drain(execution_));
      loom_serve_execution_release(execution_);
    }
    for (auto& invocation : invocations_) {
      invocation.reset();
    }
    IREE_EXPECT_OK(loom_serve_http_listener_free(listener_));
    loom_serve_http_server_free(server_);
    iree_hal_semaphore_release(gate_);
  }

  std::string Request(uint64_t blocks = 0,
                      const char* voice = "positive") const {
    return std::string(R"({"model":"speech","input":")") +
           std::string(blocks ? blocks : blocks_, 'a') + R"(","voice":")" +
           voice + R"(","response_format":"pcm"})";
  }

  iree_status_t Send(loom_serve::testing::HttpClient* client,
                     uint64_t blocks = 0, const char* voice = "positive") {
    const std::string body = Request(blocks, voice);
    return http_.Send(
        "POST /v1/audio/speech HTTP/1.1\r\nHost: local\r\n"
        "Content-Type: application/json\r\nContent-Length: " +
            std::to_string(body.size()) + "\r\n\r\n" + body,
        client->put());
  }

  void CheckHealth() {
    loom_serve::testing::HttpClient health;
    IREE_ASSERT_OK(http_.Send("GET /healthz HTTP/1.1\r\nHost: local\r\n\r\n",
                              health.put()));
    std::string response;
    IREE_ASSERT_OK(http_.Read(health.release(), false, &response));
    EXPECT_EQ(response.find("HTTP/1.1 200"), 0u);
  }

  static void CheckPcm(const std::string& response, size_t frames,
                       int sign = 1) {
    ASSERT_EQ(response.find("HTTP/1.1 200"), 0u);
    const auto begin = response.find("\r\n\r\n");
    ASSERT_NE(begin, std::string::npos);
    ASSERT_EQ(response.size() - begin - 4, frames * 2);
    for (size_t i = begin + 4; i < response.size(); i += 2) {
      ASSERT_EQ(iree_unaligned_load_le_u16(response.data() + i),
                static_cast<uint16_t>(sign * 16384));
    }
  }

  // Model and service use the same application resource owners.
  iree_allocator_t allocator_ = iree_allocator_system();
  // Borrowed process task/proactor owner; one worker detects blocking waits.
  loom_serve_host_t* host_ = nullptr;
  // Borrowed process device group enclosing all execution allocations.
  loom_serve_device_group_t* group_ = nullptr;
  // Actual queue dependencies and tracked request views.
  loom_serve_execution_t* execution_ = nullptr;
  // Explicit device dependency held by the test's outer client thread.
  iree_hal_semaphore_t* gate_ = nullptr;
  // Outer-thread gate signal/failure has been published once.
  bool gate_released_ = false;
  // Expected terminal device failure, read only by the outer test owner.
  bool failed_ = false;
  // Frames in every deterministic feedback block.
  size_t frames_ = 0;
  // Native frame geometry, independent of the HTTP PCM wire format.
  loom_serve_speech_format_t format_ = {};
  // Source completion produces no waveform despite its retired device work.
  bool empty_output_ = false;
  // Inject a source error after successful retirement of one selected row.
  bool fail_collection_ = false;
  // Deliver a nonfinite source sample to the production output encoder.
  bool nonfinite_output_ = false;
  // Output block at which the first resident row encounters either error.
  uint64_t output_failure_block_ = 1;
  // Terminal block count for each request.
  uint64_t blocks_ = 0;
  // Per-request block index that imports gate_ before accepting transfers.
  uint64_t held_block_ = 0;
  // Minimum cohort size importing the held dependency.
  size_t held_members_ = 1;
  // Dependency result changed only between manually driven owner turns.
  loom_serve_residency_admission_t admission_ =
      LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED;
  // Owner-only admission attempts, including deferred capacity decisions.
  size_t admission_attempts_ = 0;
  // Owner-only completion count, including unchanged readiness frontiers.
  size_t admission_completions_ = 0;
  // Model-owned rows; terminal failures leave them here for actual teardown.
  std::array<std::unique_ptr<loom_serve_test_speech_invocation_t>, 4>
      invocations_;
  // Original selected ordinals survive cancellation and feedback collection.
  std::vector<loom_serve_test_speech_invocation_t*> batch_;
  // Accepted work per resident row, published without exposing source cursors.
  std::array<std::atomic<uint64_t>, 4> row_submitted_ = {};
  // Collection attempts include failed and cancelled original batch members.
  std::array<std::atomic<uint64_t>, 4> row_collected_ = {};
  // Largest accepted cohort, independent of request completion order.
  std::atomic<size_t> largest_batch_{0};
  // A complete numerical cohort has accepted the held dependency.
  std::atomic<bool> held_{false};
  // Accepted generation count, published after both queues accept work.
  std::atomic<uint64_t> submitted_{0};
  // Successful release count, published by the serialized service owner.
  std::atomic<uint64_t> released_{0};
  // An owner turn observed exhausted carrier send credit.
  std::atomic<bool> credit_wait_{false};
  // HTTP owner outliving service, listener and client operations.
  loom_serve_http_server_t* server_ = nullptr;
  // Loopback listener borrowing the shared host network resources.
  loom_serve_http_listener_t* listener_ = nullptr;
  // Production service under test, retained until router retirement.
  loom_serve_speech_service_t* service_ = nullptr;
  // Unmodified callbacks, forwarded by the observation wrapper.
  loom_serve_service_t actual_ = {};
  // Production operations with observational forwarding on owner turns.
  loom_serve_service_t wrapped_ = {};
  // Client readiness and the outer router thread.
  loom_serve::testing::HttpHarness http_;
};

class SpeechOutputFailureTest : public SpeechReadinessTest,
                                public ::testing::WithParamInterface<bool> {};

TEST_P(SpeechOutputFailureTest, RetiredFailurePreservesPeersAndRetirement) {
  ASSERT_NO_FATAL_FAILURE(Initialize(3, 4, 1, 4, 4, false));
  std::array<loom_serve::testing::HttpClient, 4> clients;
  std::array<loom_serve_http_connection_t*, 4> connections = {};
  for (size_t i = 0; i < clients.size(); ++i) {
    IREE_ASSERT_OK(Send(&clients[i], 4 + i));
    const loom_serve_http_request_t* request = nullptr;
    IREE_ASSERT_OK(
        http_.TakeRequest(clients[i].get(), &connections[i], &request));
    IREE_ASSERT_OK(actual_.accept(actual_.self, connections[i], request, 0));
  }
  ASSERT_NO_FATAL_FAILURE(
      AdvanceUntil([&] { return held_.load(std::memory_order_acquire); }));
  EXPECT_EQ(largest_batch_.load(), 4u);
  // Admission permits earlier rows to advance before the full cohort forms.
  // Arm the fault only at this accepted device cut, before collection can run.
  ASSERT_NE(invocations_[0], nullptr);
  output_failure_block_ = invocations_[0]->block;
  const size_t prefix_frames = output_failure_block_ * frames_;
  fail_collection_ = GetParam();
  nonfinite_output_ = !GetParam();
  const linger reset = {1, 0};
  ASSERT_EQ(setsockopt(clients[1].get(), SOL_SOCKET, SO_LINGER, &reset,
                       sizeof(reset)),
            0);
  ASSERT_EQ(close(clients[1].release()), 0);
  iree_notification_await(
      loom_serve_http_server_notification(server_),
      [](void* self) {
        return loom_serve_http_connection_failed(
            static_cast<loom_serve_http_connection_t*>(self));
      },
      connections[1], iree_infinite_timeout());
  bool progress = false;
  IREE_ASSERT_OK(actual_.advance(actual_.self, 0, &progress));
  EXPECT_EQ(released_.load(), 0u);
  ReleaseDependency();
  ASSERT_NO_FATAL_FAILURE(
      AdvanceUntil([&] { return !actual_.is_pending(actual_.self); }));
  EXPECT_EQ(actual_.telemetry.counters->failed, 1u);
  EXPECT_EQ(actual_.telemetry.counters->cancelled, 1u);
  EXPECT_EQ(actual_.telemetry.counters->completed, 2u);
  EXPECT_EQ(released_.load(), 4u);
  for (size_t i = 0; i < clients.size(); ++i) {
    EXPECT_EQ(row_collected_[i].load(), row_submitted_[i].load());
  }
  std::string response;
  IREE_ASSERT_OK(http_.Read(clients[0].release(), true, &response));
  CheckPcm(response, prefix_frames);
  for (size_t i : {2u, 3u}) {
    IREE_ASSERT_OK(http_.Read(clients[i].release(), false, &response));
    CheckPcm(response, 3 * (4 + i));
  }

  // Reuse the failed row through the actual router, after the whole cohort
  // has retired. Neither failure publishes a partial block or poisons serving.
  fail_collection_ = false;
  nonfinite_output_ = false;
  StartRouter();
  ASSERT_NO_FATAL_FAILURE(CheckHealth());
  loom_serve::testing::HttpClient replacement;
  IREE_ASSERT_OK(Send(&replacement, 2));
  IREE_ASSERT_OK(http_.Read(replacement.release(), false, &response));
  CheckPcm(response, 6);
}

TEST_P(SpeechOutputFailureTest, FailureBeforeFirstAudioReturnsJson) {
  fail_collection_ = GetParam();
  nonfinite_output_ = !GetParam();
  output_failure_block_ = 0;
  ASSERT_NO_FATAL_FAILURE(Initialize(3, 2, UINT64_MAX, 2));
  loom_serve::testing::HttpClient failed;
  IREE_ASSERT_OK(Send(&failed));
  std::string response;
  IREE_ASSERT_OK(http_.Read(failed.release(), false, &response));
  EXPECT_EQ(response.find("HTTP/1.1 500 Internal Server Error\r\n"), 0u);
  EXPECT_NE(response.find("Content-Type: application/json\r\n"),
            std::string::npos);
  EXPECT_NE(response.find(GetParam() ? "speech source completion failed"
                                     : "nonfinite waveform sample"),
            std::string::npos);
  ASSERT_NO_FATAL_FAILURE(CheckHealth());
}

INSTANTIATE_TEST_SUITE_P(SourceAndEncoding, SpeechOutputFailureTest,
                         ::testing::Bool());

TEST_F(SpeechReadinessTest,
       DeferredAdmissionRetainsRequestUntilCapacityReturns) {
  ASSERT_NO_FATAL_FAILURE(Initialize(3, 1, UINT64_MAX, 1, 1, false));
  admission_ = LOOM_SERVE_RESIDENCY_ADMISSION_DEFERRED;
  loom_serve::testing::HttpClient client;
  IREE_ASSERT_OK(Send(&client));
  loom_serve_http_connection_t* connection = nullptr;
  const loom_serve_http_request_t* request = nullptr;
  IREE_ASSERT_OK(http_.TakeRequest(client.get(), &connection, &request));
  IREE_ASSERT_OK(actual_.accept(actual_.self, connection, request, 0));

  // Deferred admission retains preparation without accepting any activation
  // or waiting on a frontier. A retry waits for the capacity owner's wake.
  ASSERT_NO_FATAL_FAILURE(
      AdvanceUntil([&] { return admission_attempts_ != 0; }));
  bool progress = false;
  IREE_ASSERT_OK(actual_.advance(actual_.self, 0, &progress));
  EXPECT_FALSE(progress);
  EXPECT_EQ(admission_completions_, 0u);
  ASSERT_EQ(actual_.telemetry.counters->rejected, 0u);
  EXPECT_EQ(actual_.telemetry.counters->completed, 0u);
  EXPECT_EQ(submitted_.load(), 0u);

  admission_ = LOOM_SERVE_RESIDENCY_ADMISSION_ACQUIRED;
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] { return released_.load() == 1; }));
  std::string response;
  IREE_ASSERT_OK(http_.Read(client.release(), false, &response));
  CheckPcm(response, 3);
  EXPECT_EQ(actual_.telemetry.counters->completed, 1u);
  EXPECT_EQ(actual_.telemetry.counters->rejected, 0u);
  EXPECT_EQ(admission_completions_, 1u);
}

TEST_F(SpeechReadinessTest, NativePressureMovesOnlyAcceptedInput) {
  ASSERT_NO_FATAL_FAILURE(Initialize(3, 1, 0, 1, 1, false));
  std::array<loom_serve_speech_request_t, 3> requests = {};
  std::vector<SpeechStream> streams;
  for (auto& request : requests) {
    IREE_ASSERT_OK(loom_serve_speech_request_initialize_text(
        IREE_SV("a"), IREE_SV("positive"), {}, 17, 1.0f, allocator_, &request));
    loom_serve_speech_stream_t* stream = nullptr;
    IREE_ASSERT_OK(
        loom_serve_speech_service_try_open(service_, &request, &stream));
    streams.emplace_back(stream, loom_serve_speech_stream_close);
  }
  ASSERT_NE(streams[0], nullptr);
  ASSERT_NE(streams[1], nullptr);
  EXPECT_EQ(streams[2], nullptr);
  EXPECT_EQ(iree_string_builder_size(&requests[0].input), 0u);
  EXPECT_EQ(iree_string_builder_size(&requests[1].input), 0u);
  EXPECT_EQ(std::string(iree_string_builder_buffer(&requests[2].input)), "a");
  EXPECT_EQ(requests[2].seed, 17u);
  streams[1].reset();
  bool progress = false;
  IREE_ASSERT_OK(actual_.advance(actual_.self, 0, &progress));
  loom_serve_speech_stream_t* stream = nullptr;
  IREE_ASSERT_OK(
      loom_serve_speech_service_try_open(service_, &requests[2], &stream));
  streams[2].reset(stream);
  EXPECT_NE(streams[2], nullptr);
  EXPECT_EQ(iree_string_builder_size(&requests[2].input), 0u);
  for (auto& request : requests) {
    loom_serve_speech_request_deinitialize(&request);
  }
}

TEST_F(SpeechReadinessTest, NativeCancellationRetainsOriginalDeviceCohort) {
  ASSERT_NO_FATAL_FAILURE(Initialize(3, 10, 1, 2, 2, false));
  SpeechStream first(nullptr, loom_serve_speech_stream_close);
  SpeechStream second(nullptr, loom_serve_speech_stream_close);
  IREE_ASSERT_OK(OpenSpeech(service_, IREE_SV("abcdefghij"),
                            IREE_SV("positive"), 0, &first));
  IREE_ASSERT_OK(OpenSpeech(service_, IREE_SV("abcdefghijkl"),
                            IREE_SV("negative"), 0, &second));
  size_t frames = 0;
  while (!held_.load()) {
    ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] {
      return held_.load() || loom_serve_speech_stream_peek(first.get()) ||
             loom_serve_speech_stream_peek(second.get());
    }));
    loom_serve_speech_stream_consume(first.get());
    if (const auto* block = loom_serve_speech_stream_peek(second.get())) {
      frames += block->frame_count;
      loom_serve_speech_stream_consume(second.get());
    }
  }
  EXPECT_EQ(largest_batch_.load(), 2u);
  first.reset();
  bool progress = false;
  IREE_ASSERT_OK(actual_.advance(actual_.self, 0, &progress));
  EXPECT_TRUE(progress);
  EXPECT_EQ(released_.load(), 0u);
  EXPECT_EQ(actual_.snapshot(actual_.self).active_requests, 2u);
  IREE_ASSERT_OK(actual_.advance(actual_.self, 0, &progress));
  EXPECT_FALSE(progress);
  ReleaseDependency();
  while (!loom_serve_speech_stream_is_complete(second.get())) {
    ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] {
      return loom_serve_speech_stream_peek(second.get()) ||
             loom_serve_speech_stream_is_complete(second.get());
    }));
    if (const auto* block = loom_serve_speech_stream_peek(second.get())) {
      EXPECT_EQ(block->position, frames);
      for (size_t i = 0; i < block->frame_count; ++i) {
        EXPECT_EQ(iree_unaligned_load_le_f32(block->samples.data + i * 4),
                  -0.5f);
      }
      frames += block->frame_count;
      loom_serve_speech_stream_consume(second.get());
    }
  }
  EXPECT_EQ(frames, 36u);
  EXPECT_EQ(released_.load(), 2u);
  EXPECT_EQ(actual_.telemetry.counters->cancelled, 1u);
  EXPECT_EQ(actual_.telemetry.counters->completed, 1u);
}

TEST_F(SpeechReadinessTest, ClosedNativeHandleRetainsAcceptedEncodingStorage) {
  ASSERT_NO_FATAL_FAILURE(Initialize(3, 4, 0, 1, 1, false));
  SpeechStream stream(nullptr, loom_serve_speech_stream_close);
  IREE_ASSERT_OK(
      OpenSpeech(service_, IREE_SV("abcd"), IREE_SV("positive"), 0, &stream));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] { return held_.load(); }));
  std::vector<std::unique_ptr<loom_serve::testing::WorkerGate>> workers;
  ReleaseDependency();
  // CPU-device transfers share this executor too. Retire those borrowers on
  // the outer test thread before holding the workers for the encoding cut.
  const auto frontier = loom_serve_execution_frontier(execution_);
  for (size_t i = 0; i < 2; ++i) {
    IREE_ASSERT_OK(loom_serve_workspace_wait(
        loom_serve_device_workspace(loom_serve_device_group_at(group_, 0)),
        frontier.semaphores[i], frontier.values[i]));
  }
  for (int i = 0; i < SpeechWorkerCount(); ++i) {
    workers.push_back(std::make_unique<loom_serve::testing::WorkerGate>(
        loom_serve_host_executor(host_)));
  }
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] {
    return std::string(actual_.snapshot(actual_.self).phase) == "encoding";
  }));
  stream.reset();
  bool progress = false;
  IREE_ASSERT_OK(actual_.advance(actual_.self, LOOM_SERVE_SERVICE_FLAG_STOPPING,
                                 &progress));
  EXPECT_TRUE(actual_.is_pending(actual_.self));
  EXPECT_EQ(released_.load(), 0u);
  workers.clear();
  ASSERT_NO_FATAL_FAILURE(
      AdvanceUntil([&] { return !actual_.is_pending(actual_.self); },
                   LOOM_SERVE_SERVICE_FLAG_STOPPING));
  EXPECT_EQ(submitted_.load(), 1u);
  EXPECT_EQ(released_.load(), 1u);
  EXPECT_EQ(actual_.telemetry.counters->cancelled, 1u);
}

TEST_F(SpeechReadinessTest,
       NativeFailureAfterShutdownSurvivesServiceDestruction) {
  ASSERT_NO_FATAL_FAILURE(Initialize(3, 4, 0, 1, 1, false));
  SpeechStream stream(nullptr, loom_serve_speech_stream_close);
  IREE_ASSERT_OK(
      OpenSpeech(service_, IREE_SV("abcd"), IREE_SV("positive"), 0, &stream));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] { return held_.load(); }));
  bool progress = false;
  IREE_ASSERT_OK(actual_.advance(actual_.self, LOOM_SERVE_SERVICE_FLAG_STOPPING,
                                 &progress));
  EXPECT_FALSE(loom_serve_speech_stream_is_complete(stream.get()));
  ReleaseDependency(true);
  iree_status_t failure = iree_ok_status();
  auto* notification = loom_serve_http_server_notification(server_);
  while (actual_.is_pending(actual_.self)) {
    const auto ticket = iree_notification_prepare_wait(notification);
    auto status = actual_.advance(actual_.self,
                                  LOOM_SERVE_SERVICE_FLAG_STOPPING, &progress);
    if (progress || !iree_status_is_ok(status) ||
        !actual_.is_pending(actual_.self)) {
      iree_notification_cancel_wait(notification);
    } else {
      iree_notification_commit_wait(notification, ticket, IREE_DURATION_ZERO,
                                    IREE_TIME_INFINITE_FUTURE);
    }
    failure = iree_status_join(failure, status);
  }
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, failure);
  EXPECT_EQ(released_.load(), 0u);
  EXPECT_NE(invocations_[0], nullptr);
  EXPECT_EQ(actual_.telemetry.counters->failed, 1u);
  actual_.set_observer(actual_.self, {});
  loom_serve_speech_service_free(service_);
  service_ = nullptr;
  EXPECT_TRUE(loom_serve_speech_stream_is_complete(stream.get()));
  EXPECT_EQ(loom_serve_speech_stream_peek(stream.get()), nullptr);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        loom_serve_speech_stream_status(stream.get()));
}

TEST_F(SpeechReadinessTest, NativeGeometryIsIndependentOfHttpPcm) {
  ASSERT_NO_FATAL_FAILURE(
      Initialize(3, 1, UINT64_MAX, 1, 1, false, {48000, 2}));
  SpeechStream stream(nullptr, loom_serve_speech_stream_close);
  IREE_ASSERT_OK(
      OpenSpeech(service_, IREE_SV("a"), IREE_SV("negative"), 0, &stream));
  ASSERT_NO_FATAL_FAILURE(AdvanceUntil([&] { return released_.load() == 1; }));
  EXPECT_EQ(loom_serve_speech_stream_format(stream.get()).sample_rate, 48000u);
  EXPECT_EQ(loom_serve_speech_stream_format(stream.get()).channels, 2u);
  const auto* block = loom_serve_speech_stream_peek(stream.get());
  ASSERT_NE(block, nullptr);
  EXPECT_EQ(block->frame_count, 3u);
  ASSERT_EQ(block->samples.data_length, 6 * sizeof(float));
  for (size_t i = 0; i < 6; ++i) {
    EXPECT_EQ(iree_unaligned_load_le_f32(block->samples.data + i * 4), -0.5f);
  }
  loom_serve::testing::HttpClient client;
  IREE_ASSERT_OK(Send(&client));
  loom_serve_http_connection_t* connection = nullptr;
  const loom_serve_http_request_t* request = nullptr;
  IREE_ASSERT_OK(http_.TakeRequest(client.get(), &connection, &request));
  IREE_ASSERT_OK(actual_.accept(actual_.self, connection, request, 0));
  std::string response;
  IREE_ASSERT_OK(http_.Read(client.release(), false, &response));
  EXPECT_EQ(response.find("HTTP/1.1 400"), 0u);
  EXPECT_NE(response.find("24000 Hz mono"), std::string::npos);
  EXPECT_EQ(submitted_.load(), 1u);
}

TEST_F(SpeechReadinessTest,
       HeldDeviceKeepsHealthLiveAndResetRetiresBeforeReuse) {
  ASSERT_NO_FATAL_FAILURE(Initialize());
  loom_serve::testing::HttpClient client;
  IREE_ASSERT_OK(Send(&client));
  IREE_ASSERT_OK(http_.Await(-1, [&] { return submitted_.load() == 2; }));
  IREE_ASSERT_OK(http_.AwaitResponse(client.get()));
  ASSERT_NO_FATAL_FAILURE(CheckHealth());
  EXPECT_EQ(released_.load(), 0u);
  const linger reset = {1, 0};
  ASSERT_EQ(
      setsockopt(client.get(), SOL_SOCKET, SO_LINGER, &reset, sizeof(reset)),
      0);
  ASSERT_EQ(close(client.release()), 0);
  ReleaseDependency();
  IREE_ASSERT_OK(http_.Await(-1, [&] { return released_.load() == 1; }));
  IREE_ASSERT_OK(Send(&client));
  std::string response;
  IREE_ASSERT_OK(http_.Read(client.release(), false, &response));
  CheckPcm(response, 12);
  EXPECT_EQ(released_.load(), 2u);
}

TEST_F(SpeechReadinessTest, ShutdownJoinsAcceptedWorkWithoutFurtherSynthesis) {
  ASSERT_NO_FATAL_FAILURE(Initialize());
  loom_serve::testing::HttpClient client;
  IREE_ASSERT_OK(Send(&client));
  IREE_ASSERT_OK(http_.Await(-1, [&] { return submitted_.load() == 2; }));
  loom_serve_http_server_request_stop(server_);
  EXPECT_EQ(released_.load(), 0u);
  ReleaseDependency();
  IREE_ASSERT_OK(http_.Join());
  EXPECT_EQ(submitted_.load(), 2u);
  EXPECT_EQ(released_.load(), 1u);
  ASSERT_NE(actual_.telemetry.latency, nullptr);
  EXPECT_EQ(ObservationCount(actual_.telemetry.latency->first_output), 1u);
  EXPECT_EQ(
      ObservationCount(
          actual_.telemetry.latency->outcomes[LOOM_SERVE_TELEMETRY_CANCELLED]),
      1u);
}

TEST_F(SpeechReadinessTest, FailedReadinessKeepsRequestForModelRetirement) {
  ASSERT_NO_FATAL_FAILURE(Initialize());
  loom_serve::testing::HttpClient client;
  IREE_ASSERT_OK(Send(&client));
  IREE_ASSERT_OK(http_.Await(-1, [&] { return submitted_.load() == 2; }));
  ASSERT_NO_FATAL_FAILURE(CheckHealth());
  ReleaseDependency(true);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, http_.Join());
  EXPECT_EQ(submitted_.load(), 2u);
  EXPECT_EQ(released_.load(), 0u);
  EXPECT_NE(invocations_[0], nullptr);
  ASSERT_NE(actual_.telemetry.latency, nullptr);
  EXPECT_EQ(ObservationCount(actual_.telemetry.latency->first_output), 1u);
  EXPECT_EQ(
      ObservationCount(
          actual_.telemetry.latency->outcomes[LOOM_SERVE_TELEMETRY_FAILED]),
      1u);
}

TEST_F(SpeechReadinessTest, EmptyFinalResponseHasNoFirstAudioMilestone) {
  empty_output_ = true;
  ASSERT_NO_FATAL_FAILURE(Initialize(3, 1, UINT64_MAX));
  loom_serve::testing::HttpClient client;
  IREE_ASSERT_OK(Send(&client));
  std::string response;
  IREE_ASSERT_OK(http_.Read(client.release(), false, &response));
  CheckPcm(response, 0);
  loom_serve_http_server_request_stop(server_);
  IREE_ASSERT_OK(http_.Join());
  ASSERT_NE(actual_.telemetry.latency, nullptr);
  EXPECT_EQ(ObservationCount(actual_.telemetry.latency->first_output), 0u);
  EXPECT_EQ(
      ObservationCount(
          actual_.telemetry.latency->outcomes[LOOM_SERVE_TELEMETRY_COMPLETED]),
      1u);
}

TEST_F(SpeechReadinessTest, SlowReaderResumesFromBoundedSendCredit) {
  ASSERT_NO_FATAL_FAILURE(Initialize(3, 4, UINT64_MAX));
  loom_serve::testing::StalledHttpClient stalled;
  ASSERT_NO_FATAL_FAILURE(stalled.Send(server_, "/v1/audio/speech", Request()));
  stalled.WaitUntilSendCreditExhausted();
  IREE_ASSERT_OK(http_.Await(
      -1, [&] { return credit_wait_.load(std::memory_order_acquire); }));
  ASSERT_NO_FATAL_FAILURE(CheckHealth());
  EXPECT_EQ(submitted_.load(), 1u);
  CheckPcm(stalled.Read(), 12);
  EXPECT_EQ(submitted_.load(), 4u);
  EXPECT_EQ(released_.load(), 1u);
}

TEST_F(SpeechReadinessTest, CancelledBatchMemberRetiresWithOriginalCohort) {
  ASSERT_NO_FATAL_FAILURE(Initialize(3, 16, 1, 4, 4, false));
  std::array<loom_serve::testing::HttpClient, 4> clients;
  std::array<loom_serve_http_connection_t*, 4> connections = {};
  for (size_t i = 0; i < clients.size(); ++i) {
    IREE_ASSERT_OK(Send(&clients[i], 16 + i, i % 2 ? "negative" : "positive"));
    const loom_serve_http_request_t* request = nullptr;
    IREE_ASSERT_OK(
        http_.TakeRequest(clients[i].get(), &connections[i], &request));
    IREE_ASSERT_OK(actual_.accept(actual_.self, connections[i], request, 0));
  }
  ASSERT_NO_FATAL_FAILURE(
      AdvanceUntil([&] { return held_.load(std::memory_order_acquire); }));
  EXPECT_EQ(largest_batch_.load(), 4u);
  const auto submitted = submitted_.load();
  const linger reset = {1, 0};
  ASSERT_EQ(setsockopt(clients[1].get(), SOL_SOCKET, SO_LINGER, &reset,
                       sizeof(reset)),
            0);
  ASSERT_EQ(close(clients[1].release()), 0);
  iree_notification_await(
      loom_serve_http_server_notification(server_),
      [](void* self) {
        return loom_serve_http_connection_failed(
            static_cast<loom_serve_http_connection_t*>(self));
      },
      connections[1], iree_infinite_timeout());
  bool progress = false;
  IREE_ASSERT_OK(actual_.advance(actual_.self, 0, &progress));
  EXPECT_TRUE(progress);
  EXPECT_EQ(released_.load(), 0u);
  EXPECT_EQ(actual_.snapshot(actual_.self).active_requests, 4u);
  EXPECT_EQ(submitted_.load(), submitted);
  ReleaseDependency();
  ASSERT_NO_FATAL_FAILURE(
      AdvanceUntil([&] { return actual_.telemetry.counters->cancelled == 1; }));
  EXPECT_EQ(released_.load(), 1u);
  ASSERT_NO_FATAL_FAILURE(
      AdvanceUntil([&] { return actual_.telemetry.counters->completed == 3; }));
  for (size_t i : {0u, 2u, 3u}) {
    std::string response;
    IREE_ASSERT_OK(http_.Read(clients[i].release(), false, &response));
    CheckPcm(response, 3 * (16 + i), i % 2 ? -1 : 1);
  }
  EXPECT_EQ(released_.load(), 4u);
  EXPECT_FALSE(actual_.is_pending(actual_.self));
}

TEST_F(SpeechReadinessTest, StalledReaderDoesNotBlockOtherRequests) {
  ASSERT_NO_FATAL_FAILURE(Initialize(3, 4, UINT64_MAX, 4));
  loom_serve::testing::StalledHttpClient stalled;
  ASSERT_NO_FATAL_FAILURE(stalled.Send(server_, "/v1/audio/speech", Request()));
  stalled.WaitUntilSendCreditExhausted();
  IREE_ASSERT_OK(http_.Await(
      -1, [&] { return credit_wait_.load(std::memory_order_acquire); }));
  std::array<loom_serve::testing::HttpClient, 3> peers;
  for (size_t i = 0; i < peers.size(); ++i) {
    IREE_ASSERT_OK(Send(&peers[i], i + 1, i % 2 ? "positive" : "negative"));
  }
  for (size_t i = 0; i < peers.size(); ++i) {
    std::string response;
    IREE_ASSERT_OK(http_.Read(peers[i].release(), false, &response));
    CheckPcm(response, 3 * (i + 1), i % 2 ? 1 : -1);
  }
  EXPECT_EQ(released_.load(), 3u);
  EXPECT_EQ(row_submitted_[0].load(), 1u);
  ASSERT_NO_FATAL_FAILURE(CheckHealth());
  CheckPcm(stalled.Read(), 12);
  EXPECT_EQ(row_submitted_[0].load(), 4u);
  EXPECT_EQ(released_.load(), 4u);
}

}  // namespace
