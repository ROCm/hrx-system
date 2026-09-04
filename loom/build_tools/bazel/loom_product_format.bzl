# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Public rules describing Loom product artifact formats."""

LoomProductFormatInfo = provider(
    doc = "One public Loom product format and its persisted output shape.",
    fields = {
        "artifact_directory_extension": "Artifact directory suffix, or empty for a single-file format.",
        "format": "Public loom-compile --format value.",
        "output_extension": "Primary output file suffix including its leading period.",
        "output_kind": "Persistence shape: file or artifact_set.",
        "product": "Semantic product emitted by this format.",
    },
)

def _loom_product_format_impl(ctx):
    return [
        LoomProductFormatInfo(
            artifact_directory_extension = ctx.attr.artifact_directory_extension,
            format = ctx.attr.format,
            output_extension = ctx.attr.output_extension,
            output_kind = ctx.attr.output_kind,
            product = ctx.attr.product,
        ),
    ]

_loom_product_format = rule(
    implementation = _loom_product_format_impl,
    attrs = {
        "artifact_directory_extension": attr.string(),
        "format": attr.string(mandatory = True),
        "output_extension": attr.string(mandatory = True),
        "output_kind": attr.string(
            mandatory = True,
            values = ["artifact_set", "file"],
        ),
        "product": attr.string(mandatory = True),
    },
    doc = "Declares one public product format and persisted output shape.",
)

def _validate_product_format(product, format, output_extension):
    if not product:
        fail("Loom product format product must not be empty")
    if not format:
        fail("Loom product format name must not be empty")
    if not output_extension.startswith("."):
        fail("Loom product format output extension must begin with '.'")
    if "/" in output_extension or "\\" in output_extension:
        fail("Loom product format output extension must not contain a path")

def loom_file_product_format(
        name,
        product,
        format,
        output_extension,
        **kwargs):
    """Declares a product format persisted as one file.

    Args:
      name: Bazel target name.
      product: Semantic product accepted by loom-compile.
      format: Public loom-compile --format value.
      output_extension: Output filename suffix including its leading period.
      **kwargs: Common rule attributes forwarded to the format target.
    """
    _validate_product_format(product, format, output_extension)
    _loom_product_format(
        name = name,
        format = format,
        output_extension = output_extension,
        output_kind = "file",
        product = product,
        **kwargs
    )

def loom_artifact_set_product_format(
        name,
        product,
        format,
        manifest_extension,
        artifact_directory_extension,
        **kwargs):
    """Declares a product format persisted as a manifest and artifact tree.

    Args:
      name: Bazel target name.
      product: Semantic product accepted by loom-compile.
      format: Public loom-compile --format value.
      manifest_extension: Manifest filename suffix including its leading period.
      artifact_directory_extension: Artifact directory suffix including its leading period.
      **kwargs: Common rule attributes forwarded to the format target.
    """
    _validate_product_format(product, format, manifest_extension)
    if not artifact_directory_extension.startswith("."):
        fail("Loom artifact directory extension must begin with '.'")
    if "/" in artifact_directory_extension or "\\" in artifact_directory_extension:
        fail("Loom artifact directory extension must not contain a path")
    _loom_product_format(
        name = name,
        artifact_directory_extension = artifact_directory_extension,
        format = format,
        output_extension = manifest_extension,
        output_kind = "artifact_set",
        product = product,
        **kwargs
    )
