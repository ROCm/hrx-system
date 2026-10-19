#include "native_color.h"

iree_status_t jpeg_device_color_kind(const jpeg_header_t* header,
                                     uint32_t* out_kind) {
  *out_kind = 0;
  if (header->component_count == 1) {
    *out_kind = 1;
    return iree_ok_status();
  }
  bool rgb = false;
  if (!header->jfif) {
    if (header->adobe_transform >= 0) {
      if (header->adobe_transform > 1) {
        return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "JPEG Adobe transform %d is outside RGB/YCbCr",
                                header->adobe_transform);
      }
      rgb = header->adobe_transform == 0;
    } else {
      rgb = header->components[0].identifier == 'R' &&
            header->components[1].identifier == 'G' &&
            header->components[2].identifier == 'B';
    }
  }
  unsigned horizontal = header->components[0].horizontal;
  unsigned vertical = header->components[0].vertical;
  unsigned reduced_horizontal = header->components[1].horizontal;
  unsigned reduced_vertical = header->components[1].vertical;
  if (reduced_horizontal != header->components[2].horizontal ||
      reduced_vertical != header->components[2].vertical ||
      horizontal % reduced_horizontal || vertical % reduced_vertical) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "JPEG sampling needs a different color leaf");
  }
  unsigned width_ratio = horizontal / reduced_horizontal;
  unsigned height_ratio = vertical / reduced_vertical;
  if (rgb) {
    if (width_ratio == 1 && height_ratio == 1) {
      *out_kind = 2;
    }
  } else {
    // Six observed YCbCr layouts: 444, 422, 420, 440, 411, and 410.
    if (width_ratio == 1 && height_ratio == 1) {
      *out_kind = 3;
    }
    if (width_ratio == 2 && height_ratio == 1) {
      *out_kind = 4;
    }
    if (width_ratio == 2 && height_ratio == 2) {
      *out_kind = 5;
    }
    if (width_ratio == 1 && height_ratio == 2) {
      *out_kind = 6;
    }
    if (width_ratio == 4 && height_ratio == 1) {
      *out_kind = 7;
    }
    if (width_ratio == 4 && height_ratio == 2) {
      *out_kind = 8;
    }
  }
  if (!*out_kind) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "JPEG sampling needs a different color leaf");
  }
  return iree_ok_status();
}
