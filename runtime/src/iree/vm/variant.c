// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/variant.h"

static inline uint64_t iree_vm_variant_scalar_metadata(
    iree_vm_scalar_type_t scalar_type) {
  return ((uint64_t)scalar_type << 2) | IREE_VM_VARIANT_TAG_SCALAR;
}

static bool iree_vm_scalar_type_bit_width(iree_vm_scalar_type_t scalar_type,
                                          uint32_t* out_bit_width) {
  switch (scalar_type) {
    case IREE_VM_SCALAR_TYPE_I8:
    case IREE_VM_SCALAR_TYPE_F8E4M3FN:
    case IREE_VM_SCALAR_TYPE_F8E5M2:
      *out_bit_width = 8;
      return true;
    case IREE_VM_SCALAR_TYPE_I16:
    case IREE_VM_SCALAR_TYPE_F16:
    case IREE_VM_SCALAR_TYPE_BF16:
      *out_bit_width = 16;
      return true;
    case IREE_VM_SCALAR_TYPE_I32:
    case IREE_VM_SCALAR_TYPE_F32:
      *out_bit_width = 32;
      return true;
    case IREE_VM_SCALAR_TYPE_I64:
    case IREE_VM_SCALAR_TYPE_F64:
      *out_bit_width = 64;
      return true;
    default:
      return false;
  }
}

static inline bool iree_vm_variant_ref_is_borrowed(iree_vm_variant_t variant) {
  return (variant.metadata & IREE_VM_VARIANT_TAG_MASK) ==
         IREE_VM_VARIANT_TAG_BORROWED_REF;
}

static inline iree_vm_variant_t iree_vm_variant_make_ref(
    void* object, iree_vm_ref_type_t type, uint64_t tag) {
  if (!object) return iree_vm_variant_null();
  iree_vm_variant_t variant = {
      (uint64_t)(uintptr_t)object,
      (uint64_t)(uintptr_t)type | tag,
  };
  return variant;
}

static iree_status_t iree_vm_variant_check_ref_type(
    iree_vm_variant_t variant, iree_vm_ref_type_t expected_type) {
  if (!iree_vm_variant_is_ref(variant)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "variant is not a ref");
  }
  if (!expected_type) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected ref type is null");
  }
  if (variant.payload && iree_vm_variant_ref_type(variant) != expected_type) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "variant ref type mismatch");
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_function_ref_from_variant(
    iree_vm_variant_t variant, iree_vm_function_ref_t* out_function_ref) {
  if (!out_function_ref) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_function_ref is required");
  }
  if (!iree_vm_variant_is_function_ref(variant)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "variant is not a function ref");
  }
  iree_vm_function_ref_t function_ref = {
      variant.payload,
      variant.metadata & ~(uint64_t)IREE_VM_VARIANT_TAG_MASK,
  };
  *out_function_ref = function_ref;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_variant_from_scalar_bits(
    iree_vm_scalar_type_t scalar_type, uint64_t bits,
    iree_vm_variant_t* out_variant) {
  if (!out_variant) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_variant is required");
  }
  uint32_t bit_width = 0;
  if (!iree_vm_scalar_type_bit_width(scalar_type, &bit_width)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unrecognized scalar type %u",
                            (unsigned)scalar_type);
  }
  if (bit_width < 64) bits &= (UINT64_C(1) << bit_width) - 1;
  iree_vm_variant_t variant = {
      bits,
      iree_vm_variant_scalar_metadata(scalar_type),
  };
  *out_variant = variant;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_scalar_bits_from_variant(
    iree_vm_variant_t variant, iree_vm_scalar_type_t expected_type,
    uint64_t* out_bits) {
  if (!out_bits) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_bits is required");
  }
  uint32_t bit_width = 0;
  if (!iree_vm_scalar_type_bit_width(expected_type, &bit_width)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unrecognized scalar type %u",
                            (unsigned)expected_type);
  }
  if (iree_vm_variant_scalar_type(variant) != expected_type) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "variant scalar type mismatch");
  }
  *out_bits = bit_width < 64
                  ? variant.payload & ((UINT64_C(1) << bit_width) - 1)
                  : variant.payload;
  return iree_ok_status();
}

#define IREE_VM_DEFINE_INTEGER_EXTRACTOR(name, value_type,                 \
                                         unsigned_value_type, scalar_type) \
  IREE_API_EXPORT iree_status_t iree_vm_##name##_from_variant(             \
      iree_vm_variant_t variant, value_type* out_value) {                  \
    if (!out_value) {                                                      \
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,                \
                              "out_value is required");                    \
    }                                                                      \
    uint64_t bits = 0;                                                     \
    IREE_RETURN_IF_ERROR(                                                  \
        iree_vm_scalar_bits_from_variant(variant, scalar_type, &bits));    \
    unsigned_value_type narrow_bits = (unsigned_value_type)bits;           \
    memcpy(out_value, &narrow_bits, sizeof(narrow_bits));                  \
    return iree_ok_status();                                               \
  }

IREE_VM_DEFINE_INTEGER_EXTRACTOR(i8, int8_t, uint8_t, IREE_VM_SCALAR_TYPE_I8)
IREE_VM_DEFINE_INTEGER_EXTRACTOR(i16, int16_t, uint16_t,
                                 IREE_VM_SCALAR_TYPE_I16)
IREE_VM_DEFINE_INTEGER_EXTRACTOR(i32, int32_t, uint32_t,
                                 IREE_VM_SCALAR_TYPE_I32)

#undef IREE_VM_DEFINE_INTEGER_EXTRACTOR

IREE_API_EXPORT iree_status_t
iree_vm_i64_from_variant(iree_vm_variant_t variant, int64_t* out_value) {
  if (!out_value) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_value is required");
  }
  uint64_t bits = 0;
  IREE_RETURN_IF_ERROR(iree_vm_scalar_bits_from_variant(
      variant, IREE_VM_SCALAR_TYPE_I64, &bits));
  memcpy(out_value, &bits, sizeof(bits));
  return iree_ok_status();
}

#define IREE_VM_DEFINE_BITS_EXTRACTOR(name, value_type, scalar_type)    \
  IREE_API_EXPORT iree_status_t iree_vm_##name##_bits_from_variant(     \
      iree_vm_variant_t variant, value_type* out_bits) {                \
    if (!out_bits) {                                                    \
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,             \
                              "out_bits is required");                  \
    }                                                                   \
    uint64_t bits = 0;                                                  \
    IREE_RETURN_IF_ERROR(                                               \
        iree_vm_scalar_bits_from_variant(variant, scalar_type, &bits)); \
    *out_bits = (value_type)bits;                                       \
    return iree_ok_status();                                            \
  }

IREE_VM_DEFINE_BITS_EXTRACTOR(f8e4m3fn, uint8_t, IREE_VM_SCALAR_TYPE_F8E4M3FN)
IREE_VM_DEFINE_BITS_EXTRACTOR(f8e5m2, uint8_t, IREE_VM_SCALAR_TYPE_F8E5M2)
IREE_VM_DEFINE_BITS_EXTRACTOR(f16, uint16_t, IREE_VM_SCALAR_TYPE_F16)
IREE_VM_DEFINE_BITS_EXTRACTOR(bf16, uint16_t, IREE_VM_SCALAR_TYPE_BF16)

#undef IREE_VM_DEFINE_BITS_EXTRACTOR

IREE_API_EXPORT iree_status_t
iree_vm_f32_from_variant(iree_vm_variant_t variant, float* out_value) {
  if (!out_value) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_value is required");
  }
  uint64_t bits = 0;
  IREE_RETURN_IF_ERROR(iree_vm_scalar_bits_from_variant(
      variant, IREE_VM_SCALAR_TYPE_F32, &bits));
  uint32_t narrow_bits = (uint32_t)bits;
  memcpy(out_value, &narrow_bits, sizeof(narrow_bits));
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_vm_f64_from_variant(iree_vm_variant_t variant, double* out_value) {
  if (!out_value) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_value is required");
  }
  uint64_t bits = 0;
  IREE_RETURN_IF_ERROR(iree_vm_scalar_bits_from_variant(
      variant, IREE_VM_SCALAR_TYPE_F64, &bits));
  memcpy(out_value, &bits, sizeof(bits));
  return iree_ok_status();
}

IREE_API_EXPORT iree_vm_variant_t
iree_vm_variant_from_ptr_borrowed(void* ptr, iree_vm_ref_type_t type) {
  return iree_vm_variant_make_ref(ptr, type, IREE_VM_VARIANT_TAG_BORROWED_REF);
}

IREE_API_EXPORT iree_vm_variant_t
iree_vm_variant_from_ptr_retained(void* ptr, iree_vm_ref_type_t type) {
  if (ptr) iree_vm_ref_object_retain(ptr);
  return iree_vm_variant_make_ref(ptr, type, IREE_VM_VARIANT_TAG_OWNED_REF);
}

IREE_API_EXPORT iree_vm_variant_t
iree_vm_variant_from_ptr_move(void** inout_ptr, iree_vm_ref_type_t type) {
  void* ptr = *inout_ptr;
  *inout_ptr = NULL;
  return iree_vm_variant_make_ref(ptr, type, IREE_VM_VARIANT_TAG_OWNED_REF);
}

IREE_API_EXPORT iree_vm_variant_t
iree_vm_variant_from_ref_borrowed(iree_vm_ref_t ref) {
  return iree_vm_variant_make_ref(ref.object, iree_vm_ref_type(ref),
                                  IREE_VM_VARIANT_TAG_BORROWED_REF);
}

IREE_API_EXPORT iree_vm_variant_t
iree_vm_variant_from_ref_retained(iree_vm_ref_t ref) {
  if (ref.object) iree_vm_ref_object_retain(ref.object);
  return iree_vm_variant_make_ref(ref.object, iree_vm_ref_type(ref),
                                  IREE_VM_VARIANT_TAG_OWNED_REF);
}

IREE_API_EXPORT iree_vm_variant_t
iree_vm_variant_from_ref_move(iree_vm_ref_t* ref) {
  iree_vm_ref_t moved_ref = iree_vm_ref_move(ref);
  return iree_vm_variant_make_ref(moved_ref.object, iree_vm_ref_type(moved_ref),
                                  IREE_VM_VARIANT_TAG_OWNED_REF);
}

IREE_API_EXPORT iree_status_t iree_vm_ref_from_variant_borrowed(
    iree_vm_variant_t variant, iree_vm_ref_t* out_ref) {
  if (!out_ref) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_ref is required");
  }
  if (!iree_vm_variant_is_ref(variant)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "variant is not a ref");
  }
  *out_ref = iree_vm_ref_from_ptr_borrowed((void*)(uintptr_t)variant.payload,
                                           iree_vm_variant_ref_type(variant));
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_ref_from_variant_retained(
    iree_vm_variant_t variant, iree_vm_ref_t* out_ref) {
  if (!out_ref) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_ref is required");
  }
  if (!iree_vm_variant_is_ref(variant)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "variant is not a ref");
  }
  *out_ref = iree_vm_ref_from_ptr_retained((void*)(uintptr_t)variant.payload,
                                           iree_vm_variant_ref_type(variant));
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_ref_from_variant_move(
    iree_vm_variant_t* variant, iree_vm_ref_t* out_ref) {
  if (!variant || !out_ref) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "variant and out_ref are required");
  }
  if (!iree_vm_variant_is_ref(*variant)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "variant is not a ref");
  }
  iree_vm_variant_t old_variant = *variant;
  void* object = (void*)(uintptr_t)old_variant.payload;
  if (object && iree_vm_variant_ref_is_borrowed(old_variant)) {
    iree_vm_ref_object_retain(object);
  }
  iree_vm_ref_t ref = {
      object,
      object ? (uintptr_t)iree_vm_variant_ref_type(old_variant) : 0,
  };
  *variant = iree_vm_variant_empty();
  *out_ref = ref;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_ptr_from_variant_borrowed(
    iree_vm_variant_t variant, iree_vm_ref_type_t expected_type,
    void** out_ptr) {
  if (!out_ptr) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_ptr is required");
  }
  IREE_RETURN_IF_ERROR(iree_vm_variant_check_ref_type(variant, expected_type));
  *out_ptr = (void*)(uintptr_t)variant.payload;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_ptr_from_variant_retained(
    iree_vm_variant_t variant, iree_vm_ref_type_t expected_type,
    void** out_ptr) {
  if (!out_ptr) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_ptr is required");
  }
  IREE_RETURN_IF_ERROR(iree_vm_variant_check_ref_type(variant, expected_type));
  void* object = (void*)(uintptr_t)variant.payload;
  if (object) iree_vm_ref_object_retain(object);
  *out_ptr = object;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_vm_ptr_from_variant_move(
    iree_vm_variant_t* variant, iree_vm_ref_type_t expected_type,
    void** out_ptr) {
  if (!variant || !out_ptr) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "variant and out_ptr are required");
  }
  IREE_RETURN_IF_ERROR(iree_vm_variant_check_ref_type(*variant, expected_type));
  iree_vm_variant_t old_variant = *variant;
  void* object = (void*)(uintptr_t)old_variant.payload;
  if (object && iree_vm_variant_ref_is_borrowed(old_variant)) {
    iree_vm_ref_object_retain(object);
  }
  *variant = iree_vm_variant_empty();
  *out_ptr = object;
  return iree_ok_status();
}

IREE_API_EXPORT bool iree_vm_variant_ref_isa(iree_vm_variant_t variant,
                                             iree_vm_ref_type_t type) {
  return variant.payload && iree_vm_variant_is_ref(variant) &&
         iree_vm_variant_ref_type(variant) == type;
}

IREE_API_EXPORT iree_vm_variant_t
iree_vm_variant_retain(iree_vm_variant_t variant) {
  if (variant.payload && iree_vm_variant_is_ref(variant)) {
    iree_vm_ref_object_retain((void*)(uintptr_t)variant.payload);
    variant.metadata &= ~(uint64_t)IREE_VM_VARIANT_TAG_MASK;
  }
  return variant;
}

IREE_API_EXPORT iree_vm_variant_t
iree_vm_variant_move(iree_vm_variant_t* variant) {
  iree_vm_variant_t result = *variant;
  if (result.payload && iree_vm_variant_ref_is_borrowed(result)) {
    iree_vm_ref_object_retain((void*)(uintptr_t)result.payload);
    result.metadata &= ~(uint64_t)IREE_VM_VARIANT_TAG_MASK;
  }
  *variant = iree_vm_variant_empty();
  return result;
}

IREE_API_EXPORT void iree_vm_variant_reset(iree_vm_variant_t* variant) {
  iree_vm_variant_t old_variant = *variant;
  *variant = iree_vm_variant_empty();
  if (old_variant.payload &&
      (old_variant.metadata & IREE_VM_VARIANT_TAG_MASK) ==
          IREE_VM_VARIANT_TAG_OWNED_REF) {
    iree_vm_ref_object_release((void*)(uintptr_t)old_variant.payload,
                               iree_vm_variant_ref_type(old_variant));
  }
}

IREE_API_EXPORT void iree_vm_variant_span_reset(iree_vm_variant_span_t span) {
  for (iree_host_size_t i = 0; i < span.count; ++i) {
    iree_vm_variant_reset(&span.data[i]);
  }
}
