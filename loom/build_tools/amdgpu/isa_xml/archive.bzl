# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""External data archive used by Loom AMDGPU target metadata generation."""

AMDGPU_ISA_XML_ARCHIVE = {
    "build_file": "//loom/build_tools/amdgpu/isa_xml:BUILD.overlay",
    "repo_name": "amdgpu_isa_xml",
    "sha256": "1f17587abc9d0d355cf6f65bdba7496838b4e8313922769199689615c434965c",
    "urls": ["https://gpuopen.com/download/AMD_GPU_MR_ISA_XML_2026_03_05.zip"],
}
