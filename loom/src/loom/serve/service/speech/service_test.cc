// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/speech/service_test.h"

#include "iree/base/tooling/flags.h"

IREE_FLAG(int32_t, test_workers, 1, "Workers in the speech test application.");
IREE_FLAG(bool, test_elastic, false, "Use elastic backing for source speech.");

namespace loom_serve::testing {

int SpeechWorkerCount() { return FLAG_test_workers; }
bool SpeechElasticBacking() { return FLAG_test_elastic; }

iree_status_t OpenSpeech(loom_serve_speech_service_t* service,
                         iree_string_view_t input, iree_string_view_t voice,
                         uint64_t seed, SpeechStream* out_stream) {
  loom_serve_speech_request_t request;
  IREE_RETURN_IF_ERROR(loom_serve_speech_request_initialize_text(
      input, voice, {}, seed, 1.0f, iree_allocator_system(), &request));
  loom_serve_speech_stream_t* stream = nullptr;
  iree_status_t status =
      loom_serve_speech_service_try_open(service, &request, &stream);
  out_stream->reset(stream);
  loom_serve_speech_request_deinitialize(&request);
  return status;
}

uint64_t ObservationCount(
    const loom_serve_telemetry_latency_histogram_t& histogram) {
  uint64_t count = 0;
  for (uint64_t bucket : histogram.buckets) {
    count += bucket;
  }
  return count;
}

// The target leaf injects the compiler/driver pair. Each test process retains
// one application; separate worker profiles preserve concurrent qualification.
iree_status_t AcquireApplication(iree_allocator_t allocator,
                                 loom_serve_host_t** out_host,
                                 loom_serve_device_group_t** out_group) {
  auto options = loom_serve_test_application_options_default();
  options.worker_count = SpeechWorkerCount();
  options.backing = SpeechElasticBacking() ? LOOM_SERVE_DEVICE_BACKING_ELASTIC
                                           : LOOM_SERVE_DEVICE_BACKING_FIXED;
  options.slab_size = SpeechElasticBacking() ? kSpeechSlabSize : 0;
  options.memory_limit = SpeechElasticBacking() ? 64 * 1024 : 0;
  loom_serve_test_application_t* application = nullptr;
  IREE_RETURN_IF_ERROR(
      loom_serve_test_application_acquire(&options, allocator, &application));
  *out_host = loom_serve_test_application_host(application);
  *out_group = loom_serve_test_application_device_group(application);
  return iree_ok_status();
}

}  // namespace loom_serve::testing
