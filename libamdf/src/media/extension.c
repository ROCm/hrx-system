// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/extension.h"

#include "libamdf/src/media/provider.h"
#include "libamdf/src/structure.h"

static amdf_status_t amdf_media_validate_device(const amdf_device_t* device) {
  if (device == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  return amdf_device_is_extension(device, AMDF_EXTENSION_MEDIA)
             ? AMDF_STATUS_OK
             : amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}

static amdf_status_t amdf_media_validate_queue(
    const amdf_kernel_queue_t* queue) {
  return amdf_media_validate_device(queue != NULL ? queue->device : NULL);
}

static amdf_status_t AMDF_CALL
amdf_media_device_create(amdf_endpoint_t* endpoint,
                         const amdf_media_device_create_info_t* create_info,
                         amdf_device_t** out_device) {
  if (endpoint == NULL || out_device == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const amdf_status_t status = amdf_structure_validate_input(
      create_info, AMDF_STRUCTURE_TYPE_MEDIA_DEVICE_CREATE_INFO,
      sizeof(*create_info));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  return amdf_media_provider_device_create(endpoint, create_info, out_device);
}

static amdf_status_t AMDF_CALL amdf_media_device_enumerate_codecs(
    amdf_device_t* device, uint32_t queue_family_ordinal, uint32_t capacity,
    amdf_media_codec_info_t* codecs, uint32_t* count) {
  const amdf_status_t status = amdf_media_validate_device(device);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (count == NULL || (capacity != 0 && codecs == NULL)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (queue_family_ordinal >= device->queue_family_count) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  return amdf_media_device_operations(device)->device_enumerate_codecs(
      device, queue_family_ordinal, capacity, codecs, count);
}

static amdf_status_t AMDF_CALL amdf_media_device_query_configuration(
    amdf_device_t* device, const amdf_media_configuration_query_t* query,
    amdf_media_configuration_info_t* out_info) {
  amdf_status_t status = amdf_media_validate_device(device);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  status = amdf_structure_validate_header(
      query, AMDF_STRUCTURE_TYPE_MEDIA_CONFIGURATION_QUERY, sizeof(*query));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  status = amdf_structure_validate_output(
      out_info, AMDF_STRUCTURE_TYPE_MEDIA_CONFIGURATION_INFO,
      sizeof(*out_info));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (query->queue_family_ordinal >= device->queue_family_count) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  return amdf_media_device_operations(device)->device_query_configuration(
      device, query, out_info);
}

static amdf_status_t AMDF_CALL amdf_media_device_query_image_requirements(
    amdf_device_t* device, const amdf_media_image_description_t* description,
    amdf_media_image_requirements_t* out_requirements) {
  const amdf_status_t status = amdf_media_validate_device(device);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (description == NULL || out_requirements == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  return amdf_media_device_operations(device)->device_query_image_requirements(
      device, description, out_requirements);
}

static amdf_status_t AMDF_CALL amdf_media_kernel_queue_create(
    amdf_device_t* device,
    const amdf_media_kernel_queue_create_info_t* create_info,
    amdf_kernel_queue_t** out_queue) {
  amdf_status_t status = amdf_media_validate_device(device);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  status = amdf_structure_validate_input(
      create_info, AMDF_STRUCTURE_TYPE_MEDIA_KERNEL_QUEUE_CREATE_INFO,
      sizeof(*create_info));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (out_queue == NULL || create_info->maximum_pending_submission_count == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (create_info->queue_family_ordinal >= device->queue_family_count) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  return amdf_media_device_operations(device)->kernel_queue_create(
      device, create_info, out_queue);
}

static amdf_status_t AMDF_CALL amdf_media_image_create(
    amdf_device_t* device, const amdf_media_image_create_info_t* create_info,
    amdf_media_image_t** out_image) {
  amdf_status_t status = amdf_media_validate_device(device);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  status = amdf_structure_validate_input(
      create_info, AMDF_STRUCTURE_TYPE_MEDIA_IMAGE_CREATE_INFO,
      sizeof(*create_info));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (out_image == NULL || create_info->memory == NULL ||
      (create_info->layout_access_count != 0 &&
       create_info->layout_access_ordinals == NULL)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  return amdf_media_device_operations(device)->image_create(device, create_info,
                                                            out_image);
}

static amdf_status_t AMDF_CALL amdf_media_image_query_info(
    amdf_media_image_t* image, uint32_t access_ordinal,
    const amdf_media_image_query_t* query, amdf_media_image_info_t* out_info) {
  if (image == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  amdf_status_t status = amdf_structure_validate_input(
      query, AMDF_STRUCTURE_TYPE_MEDIA_IMAGE_QUERY, sizeof(*query));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  status = amdf_structure_validate_header(
      out_info, AMDF_STRUCTURE_TYPE_MEDIA_IMAGE_INFO, sizeof(*out_info));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  return image->device->operations->image_query_info(image, access_ordinal,
                                                     query, out_info);
}

static void AMDF_CALL amdf_media_image_destroy(amdf_media_image_t* image) {
  if (image != NULL) {
    image->device->operations->image_destroy(image);
  }
}

static amdf_status_t AMDF_CALL
amdf_media_session_create(amdf_kernel_queue_t* queue,
                          const amdf_media_session_create_info_t* create_info,
                          amdf_media_session_t** out_session) {
  amdf_status_t status = amdf_media_validate_queue(queue);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  status = amdf_structure_validate_header(
      create_info, AMDF_STRUCTURE_TYPE_MEDIA_SESSION_CREATE_INFO,
      sizeof(*create_info));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (out_session == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  return amdf_media_device_operations(queue->device)
      ->session_create(queue, create_info, out_session);
}

static amdf_status_t AMDF_CALL amdf_media_session_query_info(
    amdf_media_session_t* session, amdf_media_session_info_t* out_info) {
  if (session == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const amdf_status_t status = amdf_structure_validate_output(
      out_info, AMDF_STRUCTURE_TYPE_MEDIA_SESSION_INFO, sizeof(*out_info));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  return amdf_media_device_operations(session->queue->device)
      ->session_query_info(session, out_info);
}

static amdf_status_t AMDF_CALL
amdf_media_command_create(amdf_media_session_t* session,
                          const amdf_media_command_create_info_t* create_info,
                          amdf_media_command_t** out_command) {
  amdf_status_t status =
      amdf_media_validate_queue(session != NULL ? session->queue : NULL);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  status = amdf_structure_validate_input(
      create_info, AMDF_STRUCTURE_TYPE_MEDIA_COMMAND_CREATE_INFO,
      sizeof(*create_info));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (out_command == NULL || create_info->maximum_slice_count == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  return amdf_media_device_operations(session->queue->device)
      ->command_create(session, create_info, out_command);
}

static amdf_status_t AMDF_CALL amdf_media_command_prepare(
    amdf_media_command_t* command,
    const amdf_media_command_prepare_info_t* prepare_info) {
  if (command == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const amdf_status_t status = amdf_structure_validate_input(
      prepare_info, AMDF_STRUCTURE_TYPE_MEDIA_COMMAND_PREPARE_INFO,
      sizeof(*prepare_info));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (prepare_info->activation_count != 0 &&
      prepare_info->activations == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (prepare_info->decode != NULL && prepare_info->encode != NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  return amdf_media_device_operations(command->session->queue->device)
      ->command_prepare(command, prepare_info);
}

static amdf_status_t AMDF_CALL amdf_media_kernel_queue_submit(
    amdf_kernel_queue_t* queue,
    const amdf_media_kernel_queue_submission_info_t* submission_info,
    uint64_t* accepted_submission) {
  if (accepted_submission == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  *accepted_submission = 0;
  amdf_status_t status = amdf_media_validate_queue(queue);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  status = amdf_structure_validate_input(
      submission_info, AMDF_STRUCTURE_TYPE_MEDIA_KERNEL_QUEUE_SUBMISSION_INFO,
      sizeof(*submission_info));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (submission_info->command == NULL ||
      submission_info->command->session->queue != queue) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (submission_info->wait_count > queue->info.maximum_wait_count) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  status = amdf_kernel_queue_dependency_validate_waits(
      queue, submission_info->wait_count, submission_info->waits);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  return amdf_media_device_operations(queue->device)
      ->kernel_queue_submit(queue, submission_info, accepted_submission);
}

static amdf_status_t AMDF_CALL amdf_media_command_query_result(
    amdf_media_command_t* command, amdf_media_command_result_info_t* out_info) {
  if (command == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const amdf_status_t status = amdf_structure_validate_header(
      out_info, AMDF_STRUCTURE_TYPE_MEDIA_COMMAND_RESULT_INFO,
      sizeof(*out_info));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  return amdf_media_device_operations(command->session->queue->device)
      ->command_query_result(command, out_info);
}

static amdf_status_t AMDF_CALL
amdf_media_command_destroy(amdf_media_command_t* command) {
  if (command == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  return amdf_media_device_operations(command->session->queue->device)
      ->command_destroy(command);
}

static amdf_status_t AMDF_CALL
amdf_media_session_destroy(amdf_media_session_t* session) {
  if (session == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  return amdf_media_device_operations(session->queue->device)
      ->session_destroy(session);
}

static const amdf_media_api_t amdf_media_api = {
    .structure_size = sizeof(amdf_media_api_t),
    .extension_version = AMDF_MEDIA_EXTENSION_VERSION_1,
    .device_create = amdf_media_device_create,
    .device_enumerate_codecs = amdf_media_device_enumerate_codecs,
    .device_query_configuration = amdf_media_device_query_configuration,
    .device_query_image_requirements =
        amdf_media_device_query_image_requirements,
    .kernel_queue_create = amdf_media_kernel_queue_create,
    .image_create = amdf_media_image_create,
    .image_query_info = amdf_media_image_query_info,
    .image_destroy = amdf_media_image_destroy,
    .session_create = amdf_media_session_create,
    .session_query_info = amdf_media_session_query_info,
    .command_create = amdf_media_command_create,
    .command_prepare = amdf_media_command_prepare,
    .kernel_queue_submit = amdf_media_kernel_queue_submit,
    .command_query_result = amdf_media_command_query_result,
    .command_destroy = amdf_media_command_destroy,
    .session_destroy = amdf_media_session_destroy,
};

amdf_status_t amdf_media_extension_query(uint32_t minimum_version,
                                         uint32_t maximum_version,
                                         const void** out_extension_api) {
  if (minimum_version > AMDF_MEDIA_EXTENSION_VERSION_1 ||
      maximum_version < AMDF_MEDIA_EXTENSION_VERSION_1) {
    return amdf_make_api_status(AMDF_STATUS_CODE_VERSION_MISMATCH);
  }
  *out_extension_api = &amdf_media_api;
  return AMDF_STATUS_OK;
}
