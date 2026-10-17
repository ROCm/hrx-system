// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_SPEECH_SERVICE_TEST_H_
#define LOOM_SERVE_SERVICE_SPEECH_SERVICE_TEST_H_

#include <memory>

#include "loom/serve/service/speech/service.h"
#include "loom/serve/service/speech/stream.h"
#include "loom/serve/telemetry/latency.h"
#include "loom/serve/testing/service_application.h"

namespace loom_serve::testing {

using SpeechStream = std::unique_ptr<loom_serve_speech_stream_t,
                                     decltype(&loom_serve_speech_stream_close)>;

int SpeechWorkerCount();
bool SpeechElasticBacking();
iree_status_t AcquireApplication(iree_allocator_t allocator,
                                 loom_serve_host_t** out_host,
                                 loom_serve_device_group_t** out_group);
iree_status_t OpenSpeech(loom_serve_speech_service_t* service,
                         iree_string_view_t input, iree_string_view_t voice,
                         uint64_t seed, SpeechStream* out_stream);
uint64_t ObservationCount(
    const loom_serve_telemetry_latency_histogram_t& histogram);

}  // namespace loom_serve::testing

#endif  // LOOM_SERVE_SERVICE_SPEECH_SERVICE_TEST_H_
