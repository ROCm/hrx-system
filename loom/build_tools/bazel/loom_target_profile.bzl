# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Public rules for immutable Loom target profiles."""

load(":loom_product_format.bzl", "LoomProductFormatInfo")

LoomTargetProfileInfo = provider(
    doc = "Immutable Loom target identity shared by all target families.",
    fields = {
        "family": "Target fact family understood by the compiler.",
        "selector": "Family-owned exact, generic, or overlay target selector.",
    },
)

LoomTargetFormatSupportInfo = provider(
    doc = "Product formats supported by one immutable target profile.",
    fields = {
        "canonical_formats": "Product-name keyed canonical format targets.",
        "formats": "Ordered compatible product format targets.",
    },
)

def _loom_target_profile_impl(ctx):
    canonical_formats = {}
    for format_target in ctx.attr.canonical_formats:
        format_info = format_target[LoomProductFormatInfo]
        if format_info.product in canonical_formats:
            fail(
                "%s declares multiple canonical formats for product %r" %
                (ctx.label, format_info.product),
            )
        canonical_formats[format_info.product] = format_target

    formats = ctx.attr.canonical_formats + ctx.attr.formats
    seen_formats = {}
    for format_target in formats:
        format_info = format_target[LoomProductFormatInfo]
        key = "%s\n%s" % (format_info.product, format_info.format)
        if key in seen_formats:
            fail(
                "%s declares product format %s/%s more than once via %s and %s" % (
                    ctx.label,
                    format_info.product,
                    format_info.format,
                    seen_formats[key].label,
                    format_target.label,
                ),
            )
        seen_formats[key] = format_target

    return [
        LoomTargetProfileInfo(
            family = ctx.attr.family,
            selector = ctx.attr.selector,
        ),
        LoomTargetFormatSupportInfo(
            canonical_formats = canonical_formats,
            formats = formats,
        ),
    ]

_loom_target_profile = rule(
    implementation = _loom_target_profile_impl,
    attrs = {
        "canonical_formats": attr.label_list(
            providers = [LoomProductFormatInfo],
            doc = "Canonical product formats, at most one for each product.",
        ),
        "family": attr.string(
            mandatory = True,
            doc = "Target fact family accepted by the compiler.",
        ),
        "selector": attr.string(
            mandatory = True,
            doc = "Family-owned exact, generic, or overlay selector.",
        ),
        "formats": attr.label_list(
            providers = [LoomProductFormatInfo],
            doc = "Additional noncanonical formats supported by this profile.",
        ),
    },
    doc = "Declares one immutable Loom target identity profile.",
)

def loom_target_profile(
        name,
        family,
        selector,
        canonical_formats = [],
        formats = [],
        **kwargs):
    """Declares an immutable target profile understood by the compiler.

    Args:
      name: Bazel target name.
      family: Target fact family accepted by the compiler.
      selector: Family-owned exact, generic, or overlay selector.
      canonical_formats: Canonical format labels, at most one for each product.
      formats: Additional noncanonical format labels supported by this profile.
      **kwargs: Common rule attributes forwarded to the profile target.
    """
    if not family:
        fail("Loom target profile family must not be empty")
    if ":" in family:
        fail("Loom target profile family must not contain ':'")
    if not selector:
        fail("Loom target profile selector must not be empty")
    _loom_target_profile(
        name = name,
        canonical_formats = canonical_formats,
        family = family,
        formats = formats,
        selector = selector,
        **kwargs
    )
