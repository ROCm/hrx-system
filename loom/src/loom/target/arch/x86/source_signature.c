// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/source_signature.h"

#include <string.h>

#include "loom/ops/func/ops.h"

typedef struct loom_x86_source_signature_classifier_t {
  uint16_t maximum_vector_bytes;
  bool unsupported;
  bool memory;
  loom_x86_source_signature_class_t
      eightbytes[LOOM_X86_SOURCE_SIGNATURE_MAX_EIGHTBYTE_COUNT];
} loom_x86_source_signature_classifier_t;

static loom_x86_source_signature_class_t loom_x86_source_signature_merge_class(
    loom_x86_source_signature_class_t lhs,
    loom_x86_source_signature_class_t rhs) {
  if (lhs == rhs) {
    return lhs;
  }
  if (lhs == LOOM_X86_SOURCE_SIGNATURE_CLASS_NO_CLASS) {
    return rhs;
  }
  if (rhs == LOOM_X86_SOURCE_SIGNATURE_CLASS_NO_CLASS) {
    return lhs;
  }
  if (lhs == LOOM_X86_SOURCE_SIGNATURE_CLASS_MEMORY ||
      rhs == LOOM_X86_SOURCE_SIGNATURE_CLASS_MEMORY) {
    return LOOM_X86_SOURCE_SIGNATURE_CLASS_MEMORY;
  }
  if (lhs == LOOM_X86_SOURCE_SIGNATURE_CLASS_INTEGER ||
      rhs == LOOM_X86_SOURCE_SIGNATURE_CLASS_INTEGER) {
    return LOOM_X86_SOURCE_SIGNATURE_CLASS_INTEGER;
  }
  return LOOM_X86_SOURCE_SIGNATURE_CLASS_SSE;
}

static void loom_x86_source_signature_mark(
    loom_x86_source_signature_classifier_t* classifier, uint64_t byte_offset,
    uint64_t byte_length, loom_x86_source_signature_class_t first_class,
    loom_x86_source_signature_class_t remaining_class) {
  if (classifier->memory || byte_length == 0) {
    return;
  }
  const uint64_t first = byte_offset / 8u;
  const uint64_t last = (byte_offset + byte_length - 1u) / 8u;
  if (last >= LOOM_X86_SOURCE_SIGNATURE_MAX_EIGHTBYTE_COUNT) {
    classifier->memory = true;
    return;
  }
  for (uint64_t i = first; i <= last; ++i) {
    const loom_x86_source_signature_class_t value =
        i == first ? first_class : remaining_class;
    classifier->eightbytes[i] =
        loom_x86_source_signature_merge_class(classifier->eightbytes[i], value);
  }
}

static void loom_x86_source_signature_classify_leaf(
    const loom_c_abi_signature_layout_t* layout,
    loom_x86_source_signature_classifier_t* classifier,
    const loom_c_abi_leaf_t* leaf) {
  if (classifier->unsupported || classifier->memory) {
    return;
  }
  if (!(leaf->flags & LOOM_C_ABI_LEAF_FLAG_NATURALLY_ALIGNED)) {
    classifier->memory = true;
    return;
  }
  IREE_ASSERT_LT(leaf->type_index, layout->signature.types.count);
  const loom_attribute_t type =
      layout->signature.types.values[leaf->type_index];
  switch (loom_func_abi_c_type_attr_kind(type)) {
    case LOOM_FUNC_ABI_C_TYPE_KIND_BOOLEAN:
    case LOOM_FUNC_ABI_C_TYPE_KIND_SIGNED_INTEGER:
    case LOOM_FUNC_ABI_C_TYPE_KIND_UNSIGNED_INTEGER:
    case LOOM_FUNC_ABI_C_TYPE_KIND_POINTER:
      loom_x86_source_signature_mark(classifier, leaf->byte_offset,
                                     leaf->byte_length,
                                     LOOM_X86_SOURCE_SIGNATURE_CLASS_INTEGER,
                                     LOOM_X86_SOURCE_SIGNATURE_CLASS_INTEGER);
      return;
    case LOOM_FUNC_ABI_C_TYPE_KIND_FLOATING: {
      const loom_func_abi_c_type_format_t format =
          loom_func_abi_c_type_attr_format(type);
      const loom_x86_source_signature_class_t abi_class =
          format == LOOM_FUNC_ABI_C_TYPE_FORMAT_F8E4M3FN ||
                  format == LOOM_FUNC_ABI_C_TYPE_FORMAT_F8E5M2
              ? LOOM_X86_SOURCE_SIGNATURE_CLASS_INTEGER
              : LOOM_X86_SOURCE_SIGNATURE_CLASS_SSE;
      loom_x86_source_signature_mark(classifier, leaf->byte_offset,
                                     leaf->byte_length, abi_class, abi_class);
      return;
    }
    case LOOM_FUNC_ABI_C_TYPE_KIND_VECTOR: {
      const uint64_t carrier_byte_length =
          leaf->byte_length <= 16u ? 16u : leaf->byte_length;
      if ((carrier_byte_length != 16u && carrier_byte_length != 32u &&
           carrier_byte_length != 64u) ||
          carrier_byte_length > classifier->maximum_vector_bytes) {
        classifier->memory = true;
        return;
      }
      loom_x86_source_signature_mark(classifier, leaf->byte_offset,
                                     leaf->byte_length,
                                     LOOM_X86_SOURCE_SIGNATURE_CLASS_SSE,
                                     LOOM_X86_SOURCE_SIGNATURE_CLASS_SSEUP);
      return;
    }
    case LOOM_FUNC_ABI_C_TYPE_KIND_ARRAY:
    case LOOM_FUNC_ABI_C_TYPE_KIND_RECORD:
    default:
      classifier->unsupported = true;
      return;
  }
}

bool loom_x86_source_signature_classify(
    const loom_c_abi_signature_layout_t* layout,
    const loom_c_abi_object_t* object, const loom_c_abi_leaf_t* object_leaves,
    uint16_t maximum_vector_bytes,
    loom_x86_source_signature_classification_t* out_classification) {
  IREE_ASSERT_ARGUMENT(layout);
  IREE_ASSERT_ARGUMENT(object);
  IREE_ASSERT(object->leaf_count == 0 || object_leaves != NULL);
  IREE_ASSERT_ARGUMENT(out_classification);
  *out_classification = (loom_x86_source_signature_classification_t){0};
  if (object->byte_length > UINT32_MAX || object->byte_alignment > UINT32_MAX ||
      object->logical_count == 0 ||
      (maximum_vector_bytes != 0 && maximum_vector_bytes != 16 &&
       maximum_vector_bytes != 32 && maximum_vector_bytes != 64)) {
    return false;
  }

  loom_x86_source_signature_classifier_t classifier = {
      .maximum_vector_bytes = maximum_vector_bytes,
      .memory = object->byte_length > 64,
  };
  for (uint16_t i = 0; i < object->leaf_count; ++i) {
    loom_x86_source_signature_classify_leaf(layout, &classifier,
                                            &object_leaves[i]);
  }
  if (classifier.unsupported) {
    return false;
  }

  const uint8_t eightbyte_count =
      classifier.memory ? 0 : (uint8_t)((object->byte_length + 7) / 8);
  for (uint8_t i = 0; i < eightbyte_count; ++i) {
    if (classifier.eightbytes[i] == LOOM_X86_SOURCE_SIGNATURE_CLASS_SSEUP &&
        (i == 0 ||
         (classifier.eightbytes[i - 1] != LOOM_X86_SOURCE_SIGNATURE_CLASS_SSE &&
          classifier.eightbytes[i - 1] !=
              LOOM_X86_SOURCE_SIGNATURE_CLASS_SSEUP))) {
      classifier.eightbytes[i] = LOOM_X86_SOURCE_SIGNATURE_CLASS_SSE;
    }
  }
  if (eightbyte_count > 2) {
    if (classifier.eightbytes[0] != LOOM_X86_SOURCE_SIGNATURE_CLASS_SSE) {
      classifier.memory = true;
    }
    for (uint8_t i = 1; i < eightbyte_count; ++i) {
      if (classifier.eightbytes[i] != LOOM_X86_SOURCE_SIGNATURE_CLASS_SSEUP) {
        classifier.memory = true;
      }
    }
  }
  *out_classification = (loom_x86_source_signature_classification_t){
      .byte_length = (uint32_t)object->byte_length,
      .byte_alignment = (uint32_t)object->byte_alignment,
      .component_count = object->logical_count,
      .eightbyte_count = classifier.memory ? 0 : eightbyte_count,
      .is_memory = classifier.memory,
  };
  if (!classifier.memory) {
    memcpy(out_classification->eightbytes, classifier.eightbytes,
           sizeof(classifier.eightbytes));
  }
  return true;
}
