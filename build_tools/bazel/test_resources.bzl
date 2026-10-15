# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Physical-resource serialization shared by test rules and policies."""

# APIs sharing the assigned GPU serialize within one test invocation. Native
# GPU/XDNA interop also coordinates XDNA users through this coarse group.
# Independent invocations still require exclusive device assignment by the job.
GPU_DEVICE_RESOURCE_GROUP = "gpu-device"

def with_resource_group_tags(tags, resource_group):
    """Adds Bazel execution and metadata tags for a local resource group.

    Args:
      tags: Existing test tags to preserve.
      resource_group: Local resource name, or None when no resource is shared.

    Returns:
      The original tags followed by the resource-group enforcement tags.
    """
    if tags == None:
        tags = []
    if resource_group == None or resource_group == "":
        return tags
    return tags + [
        "exclusive-if-local",
        "resource_group:" + resource_group,
    ]
