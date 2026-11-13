// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/http/variants.h"

#include <inttypes.h>
#include <string.h>

#include "iree/base/internal/json.h"
#include "iree/base/internal/unicode.h"
#include "loom/serve/server/http/package.h"
#include "loom/serve/server/http/response.h"
#include "loom/serve/service/inspection.h"
#include "loom/util/json.h"

typedef enum loom_serve_http_variant_action_e {
  LOOM_SERVE_HTTP_VARIANT_STAGE,
  LOOM_SERVE_HTTP_VARIANT_SELECT,
  LOOM_SERVE_HTTP_VARIANT_REMOVE,
} loom_serve_http_variant_action_t;

typedef struct loom_serve_http_variant_wait_t {
  // Owner-thread link; callback addresses never move.
  struct loom_serve_http_variant_wait_t* next;
  // Retained claim, or NULL after a disconnected selection requester leaves.
  loom_serve_http_connection_t* connection;
  // Stable registry slot, checked with incarnation before any model access.
  iree_host_size_t model_index;
  // Registration identity; zero is a borrowed service that cannot unregister.
  uint64_t incarnation;
  // Nonrecycled candidate identity; no entry or compiled payload is borrowed.
  loom_runtime_variant_id_t id;
  // Distinguishes a passive compilation observer from an accepted callback.
  loom_serve_http_variant_action_t action;
  // Accepted selection's terminal owner-thread publication.
  struct {
    // The service has relinquished callback storage.
    bool complete;
    // Owned result, consumed even when the peer disconnected.
    iree_status_t status;
    // Actual preference after this specific selection attempt.
    loom_runtime_variant_id_t selected;
  } selection;
} loom_serve_http_variant_wait_t;

struct loom_serve_http_variants_t {
  // Allocator owning this handler and asynchronous observation records.
  iree_allocator_t host_allocator;
  // Borrowed transport, alive through all claims.
  loom_serve_http_server_t* server;
  // Borrowed registration table, mutated on the same serialized owner.
  loom_serve_registry_t* registry;
  // Pending observations and accepted callback storage.
  loom_serve_http_variant_wait_t* waits;
  // Retained connection claims, sharing the router's control-credit budget.
  iree_host_size_t claims;
};

typedef struct loom_serve_http_variant_input_t {
  // Allocator for decoded model/provider locators.
  iree_allocator_t host_allocator;
  // Owned decoded deployment selector.
  iree_string_view_t model;
  // Owned native-provider locators, copied by the service before return.
  loom_runtime_sources_t sources;
  // Base for staging or candidate for selection/removal.
  loom_runtime_variant_id_t id;
  // Expected current preference for selection.
  loom_runtime_variant_id_t expected;
  // Whether staging retains the HTTP claim until compilation is terminal.
  bool wait;
} loom_serve_http_variant_input_t;

typedef struct loom_serve_http_variant_field_t {
  // Literal protocol field name.
  const char* name;
  // Accepted JSON types as bit positions.
  uint32_t types;
  // Raw borrowed value; NULL data means absent.
  iree_string_view_t value;
} loom_serve_http_variant_field_t;

static iree_status_t loom_serve_http_variant_field(void* user_data,
                                                   iree_string_view_t key,
                                                   iree_json_value_type_t type,
                                                   iree_string_view_t value) {
  loom_serve_http_variant_field_t* fields = user_data;
  for (iree_host_size_t i = 0; fields[i].name; ++i) {
    if (!iree_string_view_equal(key, iree_make_cstring_view(fields[i].name))) {
      continue;
    }
    if (fields[i].value.data || !(fields[i].types & (1u << type))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "duplicate or wrongly typed variant field '%s'",
                              fields[i].name);
    }
    fields[i].value = value;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown variant field '%.*s'", (int)key.size,
                          key.data);
}

static iree_status_t loom_serve_http_variant_string(
    iree_string_view_t raw, iree_allocator_t host_allocator,
    iree_string_view_t* out_value) {
  if (!raw.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "variant strings must be nonempty");
  }
  char* data = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, raw.size + 1, (void**)&data));
  iree_host_size_t length = 0;
  iree_status_t status =
      iree_json_unescape_string(raw, raw.size, data, &length);
  const iree_string_view_t value = iree_make_string_view(data, length);
  if (iree_status_is_ok(status) && (!length || memchr(data, 0, length) ||
                                    !iree_unicode_utf8_validate(value))) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "variant strings require UTF-8 without NUL");
  }
  if (iree_status_is_ok(status)) {
    data[length] = 0;
    *out_value = value;
  } else {
    iree_allocator_free(host_allocator, data);
  }
  return status;
}

static iree_status_t loom_serve_http_variant_source_count(
    void* user_data, iree_host_size_t index, iree_json_value_type_t type,
    iree_string_view_t value) {
  if (type != IREE_JSON_VALUE_TYPE_STRING || !value.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "sources require nonempty provider paths");
  }
  *(iree_host_size_t*)user_data = index + 1;
  return iree_ok_status();
}

static iree_status_t loom_serve_http_variant_source(void* user_data,
                                                    iree_host_size_t index,
                                                    iree_json_value_type_t type,
                                                    iree_string_view_t value) {
  loom_serve_http_variant_input_t* input = user_data;
  return loom_serve_http_variant_string(
      value, input->host_allocator,
      &((iree_string_view_t*)input->sources.values)[index]);
}

static void loom_serve_http_variant_input_deinitialize(
    loom_serve_http_variant_input_t* input) {
  iree_allocator_free(input->host_allocator, (void*)input->model.data);
  iree_allocator_free(input->host_allocator,
                      (void*)input->sources.base_directory.data);
  for (iree_host_size_t i = 0;
       input->sources.values && i < input->sources.count; ++i) {
    iree_allocator_free(input->host_allocator,
                        (void*)input->sources.values[i].data);
  }
  iree_allocator_free(input->host_allocator, (void*)input->sources.values);
  loom_runtime_package_release(input->sources.package);
}

static iree_status_t loom_serve_http_variant_parse(
    iree_string_view_t body, loom_serve_http_variant_action_t action,
    loom_serve_http_variant_input_t* input) {
  const bool stage = action == LOOM_SERVE_HTTP_VARIANT_STAGE;
  const uint32_t string_type = 1u << IREE_JSON_VALUE_TYPE_STRING;
  const uint32_t number_type = 1u << IREE_JSON_VALUE_TYPE_NUMBER;
  loom_serve_http_variant_field_t fields[] = {
      {"model", string_type},
      {stage ? "base" : "variant", number_type},
      {"expected", action == LOOM_SERVE_HTTP_VARIANT_SELECT ? number_type : 0},
      {"sources", stage ? 1u << IREE_JSON_VALUE_TYPE_ARRAY : 0},
      {"base_directory", stage ? string_type : 0},
      {"wait", stage ? (1u << IREE_JSON_VALUE_TYPE_TRUE) |
                           (1u << IREE_JSON_VALUE_TYPE_FALSE)
                     : 0},
      {0}};
  if (!iree_unicode_utf8_validate(body)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "variant JSON requires UTF-8");
  }
  iree_string_view_t cursor = body, object;
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&cursor));
  IREE_RETURN_IF_ERROR(iree_json_consume_object(&cursor, &object));
  IREE_RETURN_IF_ERROR(iree_json_consume_insignificant(&cursor));
  if (cursor.size) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "trailing variant JSON");
  }
  IREE_RETURN_IF_ERROR(iree_json_enumerate_object_typed(
      object, loom_serve_http_variant_field, fields));
  IREE_RETURN_IF_ERROR(loom_serve_http_variant_string(
      fields[0].value, input->host_allocator, &input->model));
  IREE_RETURN_IF_ERROR(iree_json_parse_uint64(fields[1].value, &input->id));
  if (!input->id) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "variant identities must be positive");
  }
  if (action == LOOM_SERVE_HTTP_VARIANT_SELECT) {
    IREE_RETURN_IF_ERROR(
        iree_json_parse_uint64(fields[2].value, &input->expected));
    if (!input->expected) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "expected identity must be positive");
    }
  }
  if (!stage) {
    return iree_ok_status();
  }
  input->wait = true;
  if (fields[5].value.data) {
    IREE_RETURN_IF_ERROR(iree_json_parse_bool(fields[5].value, &input->wait));
  }
  if (fields[4].value.data) {
    IREE_RETURN_IF_ERROR(
        loom_serve_http_variant_string(fields[4].value, input->host_allocator,
                                       &input->sources.base_directory));
    if (!loom_runtime_path_is_absolute(input->sources.base_directory)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "base_directory must be absolute");
    }
  }
  IREE_RETURN_IF_ERROR(iree_json_enumerate_array_typed(
      fields[3].value, loom_serve_http_variant_source_count,
      &input->sources.count));
  if (!input->sources.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "at least one native provider is required");
  }
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array(
      input->host_allocator, input->sources.count,
      sizeof(*input->sources.values), (void**)&input->sources.values));
  return iree_json_enumerate_array_typed(fields[3].value,
                                         loom_serve_http_variant_source, input);
}

static iree_status_t loom_serve_http_variant_model(
    loom_serve_http_variants_t* variants, iree_string_view_t name,
    iree_host_size_t* out_index) {
  const iree_host_size_t capacity =
      loom_serve_registry_capacity(variants->registry);
  for (iree_host_size_t i = 0; i < capacity; ++i) {
    const loom_serve_registered_model_t* model =
        loom_serve_registry_at(variants->registry, i);
    if (!name.size || !iree_string_view_equal(name, model->service.name)) {
      continue;
    }
    if (model->removal || model->phase != LOOM_SERVE_REGISTRATION_READY ||
        !iree_status_is_ok(model->status)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "model has no healthy live service");
    }
    if (!model->service.reconfiguration.pool) {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "model has no native variant capability");
    }
    *out_index = i;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND, "model is not registered");
}

static iree_status_t loom_serve_http_variant_write(
    const loom_runtime_variant_snapshot_t* snapshot, bool detailed,
    iree_string_builder_t* output) {
  static const char* const phases[] = {"compiling", "ready", "failed"};
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      output,
      "{\"object\":\"variant\",\"id\":%" PRIu64 ",\"base\":%" PRIu64
      ",\"phase\":\"%s\",\"users\":%zu",
      snapshot->id, snapshot->base, phases[snapshot->phase], snapshot->users));
  loom_output_stream_t stream;
  loom_output_stream_for_builder(output, &stream);
  if (snapshot->code != IREE_STATUS_OK) {
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        output, ",\"error\":{\"code\":%d,\"message\":", snapshot->code));
    IREE_RETURN_IF_ERROR(
        loom_json_write_escaped_string(&stream, snapshot->message));
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(output, "}"));
  }
  const loom_runtime_compilation_report_t* report = snapshot->compilation;
  if (detailed && report &&
      (report->commands || report->kernels || report->reused_kernels ||
       report->severities[0] || report->severities[1] ||
       report->severities[2])) {
    IREE_RETURN_IF_ERROR(
        iree_string_builder_append_cstring(output, ",\"compilation\":"));
    IREE_RETURN_IF_ERROR(loom_runtime_compilation_write(report, &stream));
  }
  return iree_string_builder_append_cstring(output, "}");
}

static iree_status_t loom_serve_http_variant_response(
    loom_serve_http_variants_t* variants,
    loom_serve_http_connection_t* connection,
    const loom_service_reconfiguration_t* configuration,
    loom_runtime_variant_id_t id, int code, iree_string_builder_t* scratch) {
  loom_runtime_variant_snapshot_t snapshot;
  if (!loom_runtime_variant_pool_snapshot(configuration->pool, id, &snapshot)) {
    return loom_serve_http_response_reject(
        variants->server, connection, 404, "Not Found",
        IREE_SV("variant is no longer retained"), scratch);
  }
  iree_string_builder_reset(scratch);
  iree_status_t status = iree_string_builder_append_format(
      scratch,
      "HTTP/1.1 %d %s\r\nContent-Type: application/json\r\n"
      "Cache-Control: no-store\r\nConnection: close\r\n\r\n"
      "{\"selected\":%" PRIu64 ",\"variant\":",
      code,
      code == 202   ? "Accepted"
      : code == 422 ? "Unprocessable Content"
                    : "OK",
      configuration->selected(configuration->self));
  if (iree_status_is_ok(status)) {
    status = loom_serve_http_variant_write(&snapshot, true, scratch);
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(scratch, "}");
  }
  if (iree_status_is_ok(status)) {
    loom_serve_http_response_send(variants->server, connection,
                                  iree_string_builder_view(scratch));
  } else {
    loom_serve_http_connection_abort(connection);
  }
  return status;
}

static iree_status_t loom_serve_http_variant_inventory(
    loom_serve_http_variants_t* variants,
    loom_serve_http_connection_t* connection,
    const loom_serve_http_request_t* request, iree_string_builder_t* scratch) {
  iree_string_view_t name;
  if (!loom_serve_http_request_lookup_header(request, IREE_SV("X-Loom-Model"),
                                             &name) ||
      !name.size || request->body.size) {
    return loom_serve_http_response_reject(
        variants->server, connection, 400, "Bad Request",
        IREE_SV("variant inventory requires X-Loom-Model and no body"),
        scratch);
  }
  iree_host_size_t index = 0;
  iree_status_t status = loom_serve_http_variant_model(variants, name, &index);
  iree_string_view_t identity = iree_string_view_empty();
  loom_serve_http_request_lookup_header(request, IREE_SV("X-Loom-Variant"),
                                        &identity);
  iree_string_view_t offset_value = iree_string_view_empty();
  loom_serve_http_request_lookup_header(request, IREE_SV("X-Loom-Offset"),
                                        &offset_value);
  iree_host_size_t offset = 0;
  uint64_t id = 0;
  if (iree_status_is_ok(status) && identity.data) {
    status = iree_json_parse_uint64(identity, &id);
    if (iree_status_is_ok(status) && (!id || offset_value.data)) {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "variant identity must be positive with no offset");
    }
  }
  if (iree_status_is_ok(status) && offset_value.data &&
      !loom_serve_http_inspection_offset(offset_value, &offset)) {
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                         "X-Loom-Offset must be an unsigned decimal index");
  }
  if (!iree_status_is_ok(status)) {
    return loom_serve_http_response_error(variants->server, connection, status,
                                          scratch);
  }
  const loom_serve_registered_model_t* model =
      loom_serve_registry_at(variants->registry, index);
  const loom_service_reconfiguration_t* configuration =
      &model->service.reconfiguration;
  if (id) {
    return loom_serve_http_variant_response(variants, connection, configuration,
                                            id, 200, scratch);
  }
  iree_string_builder_reset(scratch);
  status = iree_string_builder_append_cstring(
      scratch,
      "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
      "Cache-Control: no-store\r\nConnection: close\r\n\r\n"
      "{\"object\":\"list\",\"model\":");
  if (iree_status_is_ok(status)) {
    loom_output_stream_t stream;
    loom_output_stream_for_builder(scratch, &stream);
    status = loom_json_write_escaped_string(&stream, name);
  }
  const iree_host_size_t count =
      loom_runtime_variant_pool_count(configuration->pool);
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_format(
        scratch,
        ",\"selected\":%" PRIu64 ",\"count\":%zu,\"offset\":%zu,\"data\":[",
        configuration->selected(configuration->self), count, offset);
  }
  const iree_host_size_t length =
      offset < count ? iree_min(64, count - offset) : 0;
  for (iree_host_size_t i = 0; i < length && iree_status_is_ok(status); ++i) {
    loom_runtime_variant_snapshot_t snapshot;
    loom_runtime_variant_pool_at(configuration->pool, offset + i, &snapshot);
    if (i) {
      status = iree_string_builder_append_cstring(scratch, ",");
    }
    if (iree_status_is_ok(status)) {
      status = loom_serve_http_variant_write(&snapshot, false, scratch);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_string_builder_append_cstring(scratch, "],\"next_offset\":");
  }
  if (iree_status_is_ok(status)) {
    status = offset < count && length < count - offset
                 ? iree_string_builder_append_format(scratch, "%zu}",
                                                     offset + length)
                 : iree_string_builder_append_cstring(scratch, "null}");
  }
  if (iree_status_is_ok(status)) {
    loom_serve_http_response_send(variants->server, connection,
                                  iree_string_builder_view(scratch));
  } else {
    loom_serve_http_connection_abort(connection);
  }
  return status;
}

static void loom_serve_http_variant_selected(
    void* user_data, iree_status_t status, loom_runtime_variant_id_t selected) {
  loom_serve_http_variant_wait_t* wait = user_data;
  wait->selection.status = status;
  wait->selection.selected = selected;
  wait->selection.complete = true;
}

iree_status_t loom_serve_http_variants_allocate(
    loom_serve_http_server_t* server, loom_serve_registry_t* registry,
    iree_allocator_t host_allocator,
    loom_serve_http_variants_t** out_variants) {
  *out_variants = NULL;
  loom_serve_http_variants_t* variants = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*variants),
                                             (void**)&variants));
  variants->host_allocator = host_allocator;
  variants->server = server;
  variants->registry = registry;
  *out_variants = variants;
  return iree_ok_status();
}

void loom_serve_http_variants_free(loom_serve_http_variants_t* variants) {
  if (!variants) {
    return;
  }
  IREE_ASSERT(!variants->waits);
  iree_allocator_free(variants->host_allocator, variants);
}

bool loom_serve_http_variants_pending(
    const loom_serve_http_variants_t* variants) {
  return variants->waits != NULL;
}

iree_host_size_t loom_serve_http_variants_claims(
    const loom_serve_http_variants_t* variants) {
  return variants->claims;
}

bool loom_serve_http_variants_handles(iree_string_view_t target) {
  return iree_string_view_equal(target, IREE_SV("/v1/models/variants")) ||
         iree_string_view_equal(target,
                                IREE_SV("/v1/models/variants/select")) ||
         iree_string_view_equal(target, IREE_SV("/v1/models/variants/remove"));
}

iree_status_t loom_serve_http_variants_accept(
    loom_serve_http_variants_t* variants,
    loom_serve_http_connection_t* connection,
    const loom_serve_http_request_t* request, bool allow_wait,
    iree_string_builder_t* scratch) {
  const bool inventory =
      iree_string_view_equal(request->target, IREE_SV("/v1/models/variants"));
  if (inventory && iree_string_view_equal(request->method, IREE_SV("GET"))) {
    return loom_serve_http_variant_inventory(variants, connection, request,
                                             scratch);
  }
  if (!iree_string_view_equal(request->method, IREE_SV("POST"))) {
    return loom_serve_http_response_reject(
        variants->server, connection, 405, "Method Not Allowed",
        IREE_SV("variant mutations require POST"), scratch);
  }
  const loom_serve_http_variant_action_t action =
      inventory ? LOOM_SERVE_HTTP_VARIANT_STAGE
      : iree_string_view_equal(request->target,
                               IREE_SV("/v1/models/variants/select"))
          ? LOOM_SERVE_HTTP_VARIANT_SELECT
          : LOOM_SERVE_HTTP_VARIANT_REMOVE;
  loom_serve_http_variant_input_t input = {.host_allocator =
                                               variants->host_allocator};
  iree_string_view_t json = request->body;
  iree_status_t status = iree_ok_status();
  if (action == LOOM_SERVE_HTTP_VARIANT_STAGE) {
    status = loom_serve_http_package_parse(request, variants->host_allocator,
                                           &json, &input.sources.package);
  }
  if (iree_status_is_ok(status)) {
    status = loom_serve_http_variant_parse(json, action, &input);
  }
  iree_host_size_t index = 0;
  if (iree_status_is_ok(status)) {
    status = loom_serve_http_variant_model(variants, input.model, &index);
  }
  const bool observe = action == LOOM_SERVE_HTTP_VARIANT_SELECT || input.wait;
  if (iree_status_is_ok(status) && observe && !allow_wait) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "management observation capacity is full");
  }
  loom_serve_http_variant_wait_t* wait = NULL;
  if (iree_status_is_ok(status) && observe) {
    status = iree_allocator_malloc(variants->host_allocator, sizeof(*wait),
                                   (void**)&wait);
  }
  loom_service_reconfiguration_t configuration = {0};
  loom_runtime_variant_id_t id = input.id;
  if (iree_status_is_ok(status)) {
    const loom_serve_registered_model_t* model =
        loom_serve_registry_at(variants->registry, index);
    configuration = model->service.reconfiguration;
    if (wait) {
      wait->connection = connection;
      wait->model_index = index;
      wait->incarnation = model->incarnation;
      wait->action = action;
      wait->id = input.id;
    }
    if (action == LOOM_SERVE_HTTP_VARIANT_STAGE) {
      status = configuration.stage_native(configuration.self, input.id,
                                          &input.sources, &id);
      if (wait) {
        wait->id = id;
      }
    } else if (action == LOOM_SERVE_HTTP_VARIANT_SELECT) {
      status =
          configuration.select(configuration.self, input.id, input.expected,
                               loom_serve_http_variant_selected, wait);
    } else {
      status = loom_runtime_variant_pool_remove(configuration.pool, input.id);
    }
  }
  loom_serve_http_variant_input_deinitialize(&input);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(variants->host_allocator, wait);
    return loom_serve_http_response_error(variants->server, connection, status,
                                          scratch);
  }
  if (wait) {
    wait->next = variants->waits;
    variants->waits = wait;
    ++variants->claims;
    return iree_ok_status();
  }
  if (action == LOOM_SERVE_HTTP_VARIANT_STAGE) {
    return loom_serve_http_variant_response(variants, connection,
                                            &configuration, id, 202, scratch);
  }
  iree_string_builder_reset(scratch);
  status = iree_string_builder_append_format(
      scratch,
      "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
      "Connection: close\r\n\r\n{\"removed\":%" PRIu64 "}",
      id);
  if (iree_status_is_ok(status)) {
    loom_serve_http_response_send(variants->server, connection,
                                  iree_string_builder_view(scratch));
  } else {
    loom_serve_http_connection_abort(connection);
  }
  return status;
}

iree_status_t loom_serve_http_variants_advance(
    loom_serve_http_variants_t* variants, iree_host_size_t model_index,
    bool stopping, iree_string_builder_t* scratch, bool* out_progress) {
  *out_progress = false;
  iree_status_t status = iree_ok_status();
  loom_serve_http_variant_wait_t** link = &variants->waits;
  while (*link && iree_status_is_ok(status)) {
    loom_serve_http_variant_wait_t* wait = *link;
    if (wait->model_index != model_index) {
      link = &wait->next;
      continue;
    }
    const loom_serve_registered_model_t* model =
        loom_serve_registry_at(variants->registry, model_index);
    if (wait->connection &&
        (stopping || loom_serve_http_connection_failed(wait->connection))) {
      loom_serve_http_connection_abort(wait->connection);
      wait->connection = NULL;
      --variants->claims;
      *out_progress = true;
    }
    if (wait->action == LOOM_SERVE_HTTP_VARIANT_SELECT) {
      if (!wait->selection.complete) {
        link = &wait->next;
        continue;
      }
      if (wait->connection && !iree_status_is_ok(wait->selection.status)) {
        status =
            loom_serve_http_response_error(variants->server, wait->connection,
                                           wait->selection.status, scratch);
        wait->selection.status = iree_ok_status();
      } else if (wait->connection) {
        iree_string_builder_reset(scratch);
        status = iree_string_builder_append_format(
            scratch,
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            "Connection: close\r\n\r\n{\"selected\":%" PRIu64 "}",
            wait->selection.selected);
        if (iree_status_is_ok(status)) {
          loom_serve_http_response_send(variants->server, wait->connection,
                                        iree_string_builder_view(scratch));
        } else {
          loom_serve_http_connection_abort(wait->connection);
        }
      } else if (!iree_status_is_ok(wait->selection.status)) {
        loom_serve_http_server_record_error(variants->server,
                                            wait->selection.status);
      }
      iree_status_free(wait->selection.status);
    } else if (wait->connection) {
      const loom_service_reconfiguration_t* configuration =
          &model->service.reconfiguration;
      loom_runtime_variant_snapshot_t snapshot;
      if (model->incarnation != wait->incarnation || model->removal ||
          model->phase != LOOM_SERVE_REGISTRATION_READY) {
        status = loom_serve_http_response_reject(
            variants->server, wait->connection, 410, "Gone",
            IREE_SV("model registration is retiring"), scratch);
      } else if (!loom_runtime_variant_pool_snapshot(configuration->pool,
                                                     wait->id, &snapshot)) {
        status = loom_serve_http_response_reject(
            variants->server, wait->connection, 404, "Not Found",
            IREE_SV("variant is no longer retained"), scratch);
      } else if (snapshot.phase == LOOM_RUNTIME_VARIANT_COMPILING) {
        link = &wait->next;
        continue;
      } else {
        status = loom_serve_http_variant_response(
            variants, wait->connection, configuration, wait->id,
            snapshot.phase == LOOM_RUNTIME_VARIANT_FAILED ? 422 : 200, scratch);
      }
    }
    if (wait->connection) {
      --variants->claims;
    }
    *link = wait->next;
    iree_allocator_free(variants->host_allocator, wait);
    *out_progress = true;
  }
  return status;
}
