// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/builtin_registry.h"

#include <stdint.h>

#include "loom/codegen/low/pipeline/pass_environment.h"
#include "loom/codegen/low/pipeline/pass_requirements.h"
#include "loom/codegen/low/transforms/allocation.h"
#include "loom/codegen/low/transforms/dce.h"
#include "loom/codegen/low/transforms/operand_forms.h"
#include "loom/codegen/low/transforms/pipeline/source_to_low.h"
#include "loom/codegen/low/transforms/pipeline/target_legalize.h"
#include "loom/transforms/cfg/branch_fusion.h"
#include "loom/transforms/cfg/branch_sink.h"
#include "loom/transforms/cfg/cfg_simplify.h"
#include "loom/transforms/cleanup/canonicalize.h"
#include "loom/transforms/cleanup/cse.h"
#include "loom/transforms/cleanup/dce.h"
#include "loom/transforms/cleanup/strip_hints.h"
#include "loom/transforms/kernel/kernel_async_legality.h"
#include "loom/transforms/kernel/kernel_resources.h"
#include "loom/transforms/kernel/promote_private_fragments.h"
#include "loom/transforms/loop/licm.h"
#include "loom/transforms/loop/loop_fusion.h"
#include "loom/transforms/math/legalize.h"
#include "loom/transforms/ownership/ownership_lifetime.h"
#include "loom/transforms/scf/scf_to_cfg.h"
#include "loom/transforms/scf/scf_unroll.h"
#include "loom/transforms/symbol/inline_callables.h"
#include "loom/transforms/symbol/refine_boundaries.h"
#include "loom/transforms/symbol/symbol_dce.h"
#include "loom/transforms/symbol/template_selection.h"
#include "loom/transforms/vector/memory_footprint.h"
#include "loom/transforms/vector/to_scalar.h"
#include "loom/transforms/view/linearize_view_accesses.h"

static const loom_pass_option_schema_t kCanonicalizeOptionSchema[] = {
    {
        .name = IREE_SVL("max-iterations"),
        .kind = LOOM_PASS_OPTION_SCHEMA_UINT32,
        .minimum_uint32 = 1,
        .maximum_uint32 = UINT32_MAX,
    },
};

static const loom_pass_option_enum_value_t
    kLowMaterializeAllocationDiagnosticsValues[] = {
        {.value = IREE_SVL("none")},
        {.value = IREE_SVL("spills")},
};

static const loom_pass_option_schema_t kLowMaterializeAllocationOptionSchema[] =
    {
        {
            .name = IREE_SVL("budgets"),
            .kind = LOOM_PASS_OPTION_SCHEMA_STRING,
        },
        {
            .name = IREE_SVL("diagnostics"),
            .kind = LOOM_PASS_OPTION_SCHEMA_ENUM,
            .enum_values = kLowMaterializeAllocationDiagnosticsValues,
            .enum_value_count =
                IREE_ARRAYSIZE(kLowMaterializeAllocationDiagnosticsValues),
        },
        {
            .name = IREE_SVL("spill-storage-spaces"),
            .kind = LOOM_PASS_OPTION_SCHEMA_STRING,
        },
};

static const loom_pass_requirement_def_t
    kLowMaterializeAllocationRequirements[] = {
        {
            .capability_type = &loom_low_pass_capability_type,
            .key = IREE_SVL(
                LOOM_LOW_PASS_REQUIREMENT_TARGET_LOW_DESCRIPTOR_REGISTRY),
            .description =
                IREE_SVL("Requires a pass environment target-low descriptor "
                         "registry."),
        },
};

static const loom_pass_requirement_def_t kLowDceRequirements[] = {
    {
        .capability_type = &loom_low_pass_capability_type,
        .key =
            IREE_SVL(LOOM_LOW_PASS_REQUIREMENT_TARGET_LOW_DESCRIPTOR_REGISTRY),
        .description =
            IREE_SVL("Requires a pass environment target-low descriptor "
                     "registry."),
    },
};

static const loom_pass_requirement_def_t kLowSelectOperandFormsRequirements[] =
    {
        {
            .capability_type = &loom_low_pass_capability_type,
            .key = IREE_SVL(
                LOOM_LOW_PASS_REQUIREMENT_TARGET_LOW_DESCRIPTOR_REGISTRY),
            .description =
                IREE_SVL("Requires a pass environment target-low descriptor "
                         "registry."),
        },
};

static const loom_pass_option_enum_value_t
    kLowSelectOperandFormsDiagnosticsValues[] = {
        {.value = IREE_SVL("none")},
        {.value = IREE_SVL("operand-forms")},
};

static const loom_pass_option_schema_t kLowSelectOperandFormsOptionSchema[] = {
    {
        .name = IREE_SVL("diagnostics"),
        .kind = LOOM_PASS_OPTION_SCHEMA_ENUM,
        .enum_values = kLowSelectOperandFormsDiagnosticsValues,
        .enum_value_count =
            IREE_ARRAYSIZE(kLowSelectOperandFormsDiagnosticsValues),
    },
};

static const loom_pass_option_enum_value_t kLowSourceToLowDiagnosticsValues[] =
    {
        {.value = IREE_SVL("all")},
        {.value = IREE_SVL("memory")},
        {.value = IREE_SVL("none")},
};

static const loom_pass_option_enum_value_t kLowSourceToLowControlFlowValues[] =
    {
        {.value = IREE_SVL("cfg")},
        {.value = IREE_SVL("structured-low")},
};

static const loom_pass_option_schema_t kLowSourceToLowOptionSchema[] = {
    {
        .name = IREE_SVL("control-flow"),
        .kind = LOOM_PASS_OPTION_SCHEMA_ENUM,
        .enum_values = kLowSourceToLowControlFlowValues,
        .enum_value_count = IREE_ARRAYSIZE(kLowSourceToLowControlFlowValues),
    },
    {
        .name = IREE_SVL("diagnostics"),
        .kind = LOOM_PASS_OPTION_SCHEMA_ENUM,
        .enum_values = kLowSourceToLowDiagnosticsValues,
        .enum_value_count = IREE_ARRAYSIZE(kLowSourceToLowDiagnosticsValues),
    },
    {
        .name = IREE_SVL("max-errors"),
        .kind = LOOM_PASS_OPTION_SCHEMA_UINT32,
        .minimum_uint32 = 0,
        .maximum_uint32 = UINT32_MAX,
    },
};

static const loom_pass_requirement_def_t kLowSourceToLowRequirements[] = {
    {
        .capability_type = &loom_low_pass_capability_type,
        .key =
            IREE_SVL(LOOM_LOW_PASS_REQUIREMENT_TARGET_LOW_DESCRIPTOR_REGISTRY),
        .description =
            IREE_SVL("Requires a pass environment target-low descriptor "
                     "registry."),
    },
    {
        .capability_type = &loom_low_pass_capability_type,
        .key = IREE_SVL(
            LOOM_LOW_PASS_REQUIREMENT_TARGET_LOW_LOWER_POLICY_REGISTRY),
        .description =
            IREE_SVL("Requires a pass environment source-to-target-low "
                     "lowering policy registry."),
    },
};

static const loom_pass_option_enum_value_t kLowTargetLegalizeModeValues[] = {
    {.value = IREE_SVL("eager")},
    {.value = IREE_SVL("final")},
};

static const loom_pass_option_enum_value_t kLowTargetLegalizePolicyValues[] = {
    {.value = IREE_SVL("prefer-native")},
    {.value = IREE_SVL("reference-only")},
    {.value = IREE_SVL("require-native")},
};

static const loom_pass_option_schema_t kLowTargetLegalizeOptionSchema[] = {
    {
        .name = IREE_SVL("max-errors"),
        .kind = LOOM_PASS_OPTION_SCHEMA_UINT32,
        .minimum_uint32 = 0,
        .maximum_uint32 = UINT32_MAX,
    },
    {
        .name = IREE_SVL("max-iterations"),
        .kind = LOOM_PASS_OPTION_SCHEMA_UINT32,
        .minimum_uint32 = 1,
        .maximum_uint32 = UINT32_MAX,
    },
    {
        .name = IREE_SVL("mode"),
        .kind = LOOM_PASS_OPTION_SCHEMA_ENUM,
        .enum_values = kLowTargetLegalizeModeValues,
        .enum_value_count = IREE_ARRAYSIZE(kLowTargetLegalizeModeValues),
    },
    {
        .name = IREE_SVL("policy"),
        .kind = LOOM_PASS_OPTION_SCHEMA_ENUM,
        .enum_values = kLowTargetLegalizePolicyValues,
        .enum_value_count = IREE_ARRAYSIZE(kLowTargetLegalizePolicyValues),
    },
};

static const loom_pass_requirement_def_t kLowTargetLegalizeRequirements[] = {
    {
        .capability_type = &loom_low_pass_capability_type,
        .key =
            IREE_SVL(LOOM_LOW_PASS_REQUIREMENT_TARGET_LOW_DESCRIPTOR_REGISTRY),
        .description =
            IREE_SVL("Requires a pass environment target-low descriptor "
                     "registry."),
    },
    {
        .capability_type = &loom_low_pass_capability_type,
        .key = IREE_SVL(
            LOOM_LOW_PASS_REQUIREMENT_TARGET_LOW_LOWER_POLICY_REGISTRY),
        .description =
            IREE_SVL("Requires a pass environment source-to-target-low "
                     "lowering policy registry."),
    },
};

static const loom_pass_option_schema_t kRefineBoundariesOptionSchema[] = {
    {
        .name = IREE_SVL("max-iterations"),
        .kind = LOOM_PASS_OPTION_SCHEMA_UINT32,
        .minimum_uint32 = 1,
        .maximum_uint32 = UINT32_MAX,
    },
};

static const loom_pass_option_enum_value_t kTemplateSelectionModeValues[] = {
    {.value = IREE_SVL("early")},
    {.value = IREE_SVL("final")},
};

static const loom_pass_option_schema_t kTemplateSelectionOptionSchema[] = {
    {
        .name = IREE_SVL("mode"),
        .kind = LOOM_PASS_OPTION_SCHEMA_ENUM,
        .enum_values = kTemplateSelectionModeValues,
        .enum_value_count = IREE_ARRAYSIZE(kTemplateSelectionModeValues),
    },
};

static const loom_pass_descriptor_t kBuiltinPassDescriptors[] = {
    {
        .key = IREE_SVL("branch-fusion"),
        .info = loom_branch_fusion_pass_info,
        .function_run = loom_branch_fusion_run,
    },
    {
        .key = IREE_SVL("branch-sink"),
        .info = loom_branch_sink_pass_info,
        .function_run = loom_branch_sink_run,
    },
    {
        .key = IREE_SVL("canonicalize"),
        .info = loom_canonicalize_pass_info,
        .function_run = loom_canonicalize_run,
        .create = loom_canonicalize_create,
        .option_schema = kCanonicalizeOptionSchema,
        .option_schema_count = IREE_ARRAYSIZE(kCanonicalizeOptionSchema),
    },
    {
        .key = IREE_SVL("cfg-simplify"),
        .info = loom_cfg_simplify_pass_info,
        .function_run = loom_cfg_simplify_run,
    },
    {
        .key = IREE_SVL("cse"),
        .info = loom_cse_pass_info,
        .function_run = loom_cse_run,
    },
    {
        .key = IREE_SVL("dce"),
        .info = loom_dce_pass_info,
        .function_run = loom_dce_run,
    },
    {
        .key = IREE_SVL("inline-callables"),
        .info = loom_inline_callables_pass_info,
        .module_run = loom_inline_callables_run,
    },
    {
        .key = IREE_SVL("kernel-async-legality"),
        .info = loom_kernel_async_legality_pass_info,
        .function_run = loom_kernel_async_legality_run,
    },
    {
        .key = IREE_SVL("legalize-math"),
        .info = loom_math_legalize_pass_info,
        .function_run = loom_math_legalize_run,
        .create = loom_math_legalize_create,
        .option_schema = kCanonicalizeOptionSchema,
        .option_schema_count = IREE_ARRAYSIZE(kCanonicalizeOptionSchema),
    },
    {
        .key = IREE_SVL("licm"),
        .info = loom_licm_pass_info,
        .function_run = loom_licm_run,
    },
    {
        .key = IREE_SVL("linearize-view-accesses"),
        .info = loom_linearize_view_accesses_pass_info,
        .function_run = loom_linearize_view_accesses_run,
    },
    {
        .key = IREE_SVL("loop-fusion"),
        .info = loom_loop_fusion_pass_info,
        .function_run = loom_loop_fusion_run,
    },
    {
        .key = IREE_SVL("low-dce"),
        .info = loom_low_dce_pass_info,
        .function_run = loom_low_dce_run,
        .requirement_defs = kLowDceRequirements,
        .requirement_count = IREE_ARRAYSIZE(kLowDceRequirements),
    },
    {
        .key = IREE_SVL("low-materialize-allocation"),
        .info = loom_low_materialize_allocation_pass_info,
        .function_run = loom_low_materialize_allocation_run,
        .create = loom_low_materialize_allocation_create,
        .option_schema = kLowMaterializeAllocationOptionSchema,
        .option_schema_count =
            IREE_ARRAYSIZE(kLowMaterializeAllocationOptionSchema),
        .requirement_defs = kLowMaterializeAllocationRequirements,
        .requirement_count =
            IREE_ARRAYSIZE(kLowMaterializeAllocationRequirements),
    },
    {
        .key = IREE_SVL("low-select-operand-forms"),
        .info = loom_low_select_operand_forms_pass_info,
        .function_run = loom_low_select_operand_forms_run,
        .create = loom_low_select_operand_forms_create,
        .option_schema = kLowSelectOperandFormsOptionSchema,
        .option_schema_count =
            IREE_ARRAYSIZE(kLowSelectOperandFormsOptionSchema),
        .requirement_defs = kLowSelectOperandFormsRequirements,
        .requirement_count = IREE_ARRAYSIZE(kLowSelectOperandFormsRequirements),
    },
    {
        .key = IREE_SVL("normalize-kernel-resources"),
        .info = loom_normalize_kernel_resources_pass_info,
        .function_run = loom_normalize_kernel_resources_run,
    },
    {
        .key = IREE_SVL("ownership-lifetime"),
        .info = loom_ownership_lifetime_pass_info,
        .module_run = loom_ownership_lifetime_run,
    },
    {
        .key = IREE_SVL("promote-private-fragments"),
        .info = loom_promote_private_fragments_pass_info,
        .function_run = loom_promote_private_fragments_run,
    },
    {
        .key = IREE_SVL("refine-boundaries"),
        .info = loom_refine_boundaries_pass_info,
        .module_run = loom_refine_boundaries_run,
        .create = loom_refine_boundaries_create,
        .option_schema = kRefineBoundariesOptionSchema,
        .option_schema_count = IREE_ARRAYSIZE(kRefineBoundariesOptionSchema),
    },
    {
        .key = IREE_SVL("scf-to-cfg"),
        .info = loom_scf_to_cfg_pass_info,
        .function_run = loom_scf_to_cfg_run,
    },
    {
        .key = IREE_SVL("select-templates"),
        .info = loom_template_selection_pass_info,
        .module_run = loom_template_selection_run,
        .create = loom_template_selection_create,
        .option_schema = kTemplateSelectionOptionSchema,
        .option_schema_count = IREE_ARRAYSIZE(kTemplateSelectionOptionSchema),
    },
    {
        .key = IREE_SVL("source-to-low"),
        .info = loom_low_source_to_low_pass_info,
        .module_run = loom_low_source_to_low_run,
        .create = loom_low_source_to_low_create,
        .option_schema = kLowSourceToLowOptionSchema,
        .option_schema_count = IREE_ARRAYSIZE(kLowSourceToLowOptionSchema),
        .requirement_defs = kLowSourceToLowRequirements,
        .requirement_count = IREE_ARRAYSIZE(kLowSourceToLowRequirements),
    },
    {
        .key = IREE_SVL("strip-hints"),
        .info = loom_strip_hints_pass_info,
        .function_run = loom_strip_hints_run,
    },
    {
        .key = IREE_SVL("symbol-dce"),
        .info = loom_symbol_dce_pass_info,
        .module_run = loom_symbol_dce_run,
    },
    {
        .key = IREE_SVL("target-legalize"),
        .info = loom_low_target_legalize_pass_info,
        .module_run = loom_low_target_legalize_run,
        .create = loom_low_target_legalize_create,
        .option_schema = kLowTargetLegalizeOptionSchema,
        .option_schema_count = IREE_ARRAYSIZE(kLowTargetLegalizeOptionSchema),
        .requirement_defs = kLowTargetLegalizeRequirements,
        .requirement_count = IREE_ARRAYSIZE(kLowTargetLegalizeRequirements),
    },
    {
        .key = IREE_SVL("unroll-scf-for"),
        .info = loom_scf_unroll_pass_info,
        .function_run = loom_scf_unroll_run,
        .create = loom_scf_unroll_create,
    },
    {
        .key = IREE_SVL("vector-gather-to-scalar"),
        .info = loom_vector_gather_to_scalar_pass_info,
        .function_run = loom_vector_gather_to_scalar_run,
    },
    {
        .key = IREE_SVL("vector-memory-footprint"),
        .info = loom_vector_memory_footprint_pass_info,
        .function_run = loom_vector_memory_footprint_run,
    },
    {
        .key = IREE_SVL("vector-reduce-axes-to-scalar"),
        .info = loom_vector_reduce_axes_to_scalar_pass_info,
        .function_run = loom_vector_reduce_axes_to_scalar_run,
    },
    {
        .key = IREE_SVL("vector-to-scalar"),
        .info = loom_vector_to_scalar_pass_info,
        .function_run = loom_vector_to_scalar_run,
    },
};

static const loom_pass_registry_t kBuiltinPassRegistry = {
    .descriptors = kBuiltinPassDescriptors,
    .descriptor_count = IREE_ARRAYSIZE(kBuiltinPassDescriptors),
};

const loom_pass_registry_t* loom_pass_builtin_registry(void) {
  return &kBuiltinPassRegistry;
}
