// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/http/package.h"

#include "loom/serve/transport/http/multipart.h"

iree_status_t loom_serve_http_package_parse(
    const loom_serve_http_request_t* request, iree_allocator_t host_allocator,
    iree_string_view_t* out_json, loom_runtime_package_t** out_package) {
  *out_json = iree_string_view_empty();
  *out_package = NULL;
  iree_string_view_t content_type, type;
  loom_serve_http_request_lookup_header(request, IREE_SV("Content-Type"),
                                        &content_type);
  iree_string_view_split(content_type, ';', &type, NULL);
  type = iree_string_view_trim(type);
  if (!content_type.data ||
      iree_string_view_equal_case(type, IREE_SV("application/json"))) {
    *out_json = request->body;
    return iree_ok_status();
  }
  if (!iree_string_view_equal_case(type, IREE_SV("multipart/form-data"))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source package requires JSON or multipart/form-data");
  }
  const loom_serve_http_part_list_t parts =
      loom_serve_http_multipart_parts(request->multipart);
  iree_status_t status = iree_ok_status();
  loom_runtime_package_file_t* files = NULL;
  iree_host_size_t count = 0, capacity = 0;
  iree_string_view_t json = iree_string_view_empty();
  iree_string_view_t document = iree_string_view_empty();
  for (iree_host_size_t i = 0; i < parts.count && iree_status_is_ok(status);
       ++i) {
    const loom_serve_http_part_t part = parts.values[i];
    if (iree_string_view_equal(part.name, IREE_SV("request"))) {
      if (json.data) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "source package requires one request part");
      } else {
        json = part.contents;
        document = part.filename.data ? part.filename : IREE_SV("request.json");
      }
    } else if (iree_string_view_equal(part.name, IREE_SV("file")) &&
               part.filename.size) {
      if (count == capacity) {
        status =
            iree_allocator_grow_array(host_allocator, count + 1, sizeof(*files),
                                      &capacity, (void**)&files);
      }
      if (iree_status_is_ok(status)) {
        files[count++] = (loom_runtime_package_file_t){
            .path = part.filename,
            .contents = iree_make_const_byte_span(part.contents.data,
                                                  part.contents.size),
        };
      }
    } else {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "source package accepts request and named file parts");
    }
  }
  if (iree_status_is_ok(status) && !json.size) {
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "source package requires a nonempty request part");
  }
  if (iree_status_is_ok(status)) {
    status = loom_runtime_package_create(document, count, files, host_allocator,
                                         out_package);
  }
  iree_allocator_free(host_allocator, files);
  if (iree_status_is_ok(status)) {
    *out_json = json;
  }
  return status;
}
