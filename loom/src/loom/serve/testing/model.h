// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_TESTING_MODEL_H_
#define LOOM_SERVE_TESTING_MODEL_H_

#include <string>

#include "iree/base/internal/json.h"
#include "iree/io/file_contents.h"
#include "loom/serve/server/configuration.h"

namespace loom_serve::testing {

// Reads the authored definition for HTTP lifecycle tests, leaving registration
// name/admission/base to the test's deployment. The production parser remains
// responsible for validating the request sent to the actual server.
inline std::string ReadDefinition(const std::string& directory) {
  const std::string path = directory + "/model.json";
  iree_io_file_contents_t* contents = nullptr;
  IREE_CHECK_OK(iree_io_file_contents_read(iree_make_cstring_view(path.c_str()),
                                           iree_allocator_system(), &contents));
  const std::string json(
      reinterpret_cast<const char*>(contents->const_buffer.data),
      contents->const_buffer.data_length);
  iree_io_file_contents_free(contents);
  iree_string_view_t definition;
  IREE_CHECK_OK(iree_json_lookup_object_value(
      iree_make_string_view(json.data(), json.size()), IREE_SV("definition"),
      &definition));
  return std::string(definition.data, definition.size);
}

// Owns the same decoded document used by CLI and HTTP registration. Native
// modality tests borrow it through completion of asynchronous preparation.
class ModelDocument {
 public:
  ModelDocument() = default;
  ModelDocument(const ModelDocument&) = delete;
  ModelDocument& operator=(const ModelDocument&) = delete;
  ~ModelDocument() { loom_serve_model_configuration_deinitialize(&value_); }

  iree_status_t Load(const std::string& path) {
    loom_serve_model_configuration_deinitialize(&value_);
    return loom_serve_model_configuration_load(
        iree_make_cstring_view(path.c_str()), &value_, iree_allocator_system());
  }

  const loom_serve_model_configuration_t& value() const { return value_; }

 private:
  // Owned parser result, zero-initialized before loading and after failure.
  loom_serve_model_configuration_t value_ = {};
};

}  // namespace loom_serve::testing

#endif  // LOOM_SERVE_TESTING_MODEL_H_
