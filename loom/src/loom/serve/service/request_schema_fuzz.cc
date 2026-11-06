// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstdint>
#include <string>

#include "iree/base/api.h"
#include "iree/base/string_builder.h"
#include "loom/serve/service/image/request.h"
#include "loom/serve/service/recognition/transcription.h"
#include "loom/serve/service/speech/request.h"
#include "loom/serve/service/speech/voice.h"
#include "loom/serve/service/speech/voice_http.h"
#include "loom/serve/transport/http/request.h"
#include "loom/service/finite/decision/protocol.h"
#include "loom/service/finite/embedding/protocol.h"

namespace {

constexpr size_t kMaximumInputLength = 64 * 1024;
constexpr iree_host_size_t kGeneratedBodyOverhead = 4096;

void AppendMultipartField(std::string* body, const char* name,
                          iree_string_view_t value, const char* filename,
                          const char* content_type) {
  *body += "--loom-fuzz-boundary\r\nContent-Disposition: form-data; name=\"";
  *body += name;
  *body += "\"";
  if (filename) {
    *body += "; filename=\"";
    *body += filename;
    *body += "\"";
  }
  *body += "\r\n";
  if (content_type) {
    *body += "Content-Type: ";
    *body += content_type;
    *body += "\r\n";
  }
  *body += "\r\n";
  body->append(value.data, value.size);
  *body += "\r\n";
}

std::string JsonRequest(const char* target, iree_string_view_t body) {
  std::string request = "POST ";
  request += target;
  request +=
      " HTTP/1.1\r\nHost: local\r\nContent-Type: application/json\r\n"
      "Content-Length: ";
  request += std::to_string(body.size);
  request += "\r\n\r\n";
  request.append(body.data, body.size);
  return request;
}

std::string MultipartRequest(char family, iree_string_view_t input) {
  std::string body;
  const bool voice = family == 'v';
  AppendMultipartField(&body, "model",
                       voice ? IREE_SV("speech") : IREE_SV("recognition"),
                       nullptr, nullptr);
  if (voice) {
    AppendMultipartField(&body, "id", IREE_SV("fuzz-voice"), nullptr, nullptr);
  }
  AppendMultipartField(&body, "file", input,
                       voice ? "voice.pcm" : "transcription.pcm",
                       "audio/pcm;rate=16000;channels=1");
  body += "--loom-fuzz-boundary--\r\n";

  std::string request = "POST /v1/audio/";
  request += voice ? "voices" : "transcriptions";
  request +=
      " HTTP/1.1\r\nHost: local\r\nContent-Type: multipart/form-data; "
      "boundary=loom-fuzz-boundary\r\nContent-Length: ";
  request += std::to_string(body.size());
  request += "\r\n\r\n";
  request += body;
  return request;
}

void ExerciseFinite(const loom_service_finite_protocol_t* protocol,
                    iree_string_view_t body) {
  loom_service_finite_request_t* request = nullptr;
  iree_status_t status = iree_allocator_malloc(
      iree_allocator_system(), protocol->request_size, (void**)&request);
  if (iree_status_is_ok(status)) {
    status = protocol->initialize(body, iree_allocator_system(), request);
    protocol->deinitialize(request);
    iree_allocator_free(iree_allocator_system(), request);
  }
  iree_status_free(status);
}

void ExerciseRequest(char family, const loom_serve_http_request_t* request) {
  iree_status_t status = iree_ok_status();
  switch (family) {
    case 'd':
      ExerciseFinite(loom_service_decision_protocol(), request->body);
      return;
    case 'e':
      ExerciseFinite(loom_service_embedding_protocol(), request->body);
      return;
    case 'i': {
      loom_serve_image_request_t image_request;
      status = loom_serve_image_request_initialize(
          request->body, IREE_SV("image"), 1024, 1024, iree_allocator_system(),
          &image_request);
      loom_serve_image_request_deinitialize(&image_request);
      break;
    }
    case 's': {
      loom_serve_speech_request_t speech_request;
      status = loom_serve_speech_request_initialize(
          request->body, IREE_SV("speech"), iree_allocator_system(),
          &speech_request);
      loom_serve_speech_request_deinitialize(&speech_request);
      break;
    }
    case 'r': {
      iree_string_builder_t id;
      iree_string_builder_initialize(iree_allocator_system(), &id);
      status = loom_serve_speech_voice_http_remove_parse(
          request, IREE_SV("speech"), &id);
      iree_string_builder_deinitialize(&id);
      break;
    }
    case 'v': {
      loom_serve_speech_voice_request_t voice_request;
      status = loom_serve_speech_voice_http_parse(
          request, IREE_SV("speech"), iree_allocator_system(), &voice_request);
      loom_serve_speech_voice_request_deinitialize(&voice_request);
      break;
    }
    case 't': {
      loom_serve_transcription_t transcription;
      status = loom_serve_transcription_parse(
          loom_serve_http_multipart_parts(request->multipart),
          IREE_SV("recognition"),
          loom_serve_recognition_format_t{.sample_rate = 16000, .channels = 1},
          &transcription);
      break;
    }
    default:
      return;
  }
  iree_status_free(status);
}

void Exercise(char family, const std::string& wire_request) {
  const loom_serve_http_request_limits_t limits = {
      .header_byte_capacity = 8192,
      .body_byte_capacity = kMaximumInputLength + kGeneratedBodyOverhead,
      .header_capacity = 64,
      .multipart_header_byte_capacity = 8192,
      .multipart_part_capacity = 64,
  };
  loom_serve_http_request_parser_t* parser = nullptr;
  iree_status_t status = loom_serve_http_request_parser_allocate(
      limits, iree_allocator_system(), &parser);
  const loom_serve_http_request_t* request = nullptr;
  if (iree_status_is_ok(status)) {
    status = loom_serve_http_request_parser_feed(
        parser,
        iree_make_const_byte_span(wire_request.data(), wire_request.size()),
        &request);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_http_request_parser_finalize(parser, &request);
  }
  if (iree_status_is_ok(status)) {
    ExerciseRequest(family, request);
  }
  iree_status_free(status);
  loom_serve_http_request_parser_free(parser);
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (!size || size > kMaximumInputLength) {
    return 0;
  }
  const char family = static_cast<char>(data[0]);
  const iree_string_view_t input =
      iree_make_string_view(reinterpret_cast<const char*>(data + 1), size - 1);
  switch (family) {
    case 'd':
      Exercise(family, JsonRequest("/v1/systemone", input));
      break;
    case 'e':
      Exercise(family, JsonRequest("/v1/embeddings", input));
      break;
    case 'i':
      Exercise(family, JsonRequest("/v1/images/generations", input));
      break;
    case 's':
      Exercise(family, JsonRequest("/v1/audio/speech", input));
      break;
    case 'r':
      Exercise(family, JsonRequest("/v1/audio/voices/remove", input));
      break;
    case 'v':
    case 't':
      Exercise(family, MultipartRequest(family, input));
      break;
    default:
      break;
  }
  return 0;
}
