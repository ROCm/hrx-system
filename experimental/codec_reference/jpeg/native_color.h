#ifndef CODEC_EXPERIMENT_NATIVE_COLOR_H_
#define CODEC_EXPERIMENT_NATIVE_COLOR_H_

#include "native_header.h"

#ifdef __cplusplus
extern "C" {
#endif

// Resolves the source interpretation and sampling accepted by the qualified
// color leaf. JFIF takes precedence over Adobe; absent markers use RGB
// component identifiers or the conventional YCbCr interpretation. Unsupported
// transforms and sampling fail before a packet can select an undefined device
// branch. Success yields the existing color-kind values one through eight.
iree_status_t jpeg_device_color_kind(const jpeg_header_t* header,
                                     uint32_t* out_kind);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // CODEC_EXPERIMENT_NATIVE_COLOR_H_
