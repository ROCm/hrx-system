# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""BACKEND domain — target codegen feedback and emission."""

from loom.errors import ErrorDef, ErrorDomain, ErrorParam, ParamKind, Severity

# ERR_BACKEND_003: Register pressure peak was observed.
ERR_BACKEND_003 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=3,
    severity=Severity.REMARK,
    summary="Register pressure peak observed.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "'@{function_name}' peak {value_class} pressure is {peak} "
        "unit(s) against budget {budget} at block '{block_name}' "
        "near '{operation_name}'; contributors: {contributors}"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("value_class", ParamKind.STRING),
        ErrorParam("budget", ParamKind.U32),
        ErrorParam("peak", ParamKind.U32),
        ErrorParam("block_name", ParamKind.STRING),
        ErrorParam("operation_name", ParamKind.STRING),
        ErrorParam("contributors", ParamKind.STRING_LIST),
    ),
)

# ERR_BACKEND_005: Register allocation failed.
ERR_BACKEND_005 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=5,
    severity=Severity.ERROR,
    summary="Register allocation failed.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "failed to allocate {value_class} registers for '@{function_name}' "
        "with budget {budget}, peak {peak}, and failure code '{failure_code}'"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("value_class", ParamKind.STRING),
        ErrorParam("budget", ParamKind.U32),
        ErrorParam("peak", ParamKind.U32),
        ErrorParam("failure_code", ParamKind.STRING),
    ),
    fix_hint=(
        "Lower pressure before allocation or allow spill codegen for "
        "{value_class} values"
    ),
)

# ERR_BACKEND_006: Register coalescing decision was recorded.
ERR_BACKEND_006 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=6,
    severity=Severity.REMARK,
    summary="Register coalescing decision recorded.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "{decision} coalescing '{value_name}' with '{partner_value_name}' "
        "in '@{function_name}' with constraint '{coalescing_constraint}'"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("partner_value_name", ParamKind.STRING),
        ErrorParam("decision", ParamKind.STRING),
        ErrorParam("coalescing_constraint", ParamKind.STRING),
    ),
)

# ERR_BACKEND_008: Spill traffic was predicted before allocation.
ERR_BACKEND_008 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=8,
    severity=Severity.WARNING,
    summary="Spill traffic predicted.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "predicts spilling {value_class} value '{value_name}' in "
        "'@{function_name}' for {spill_bytes} byte(s), {store_count} "
        "store(s), {reload_count} reload(s), and cause '{spill_cause}'"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("value_class", ParamKind.STRING),
        ErrorParam("spill_bytes", ParamKind.U32),
        ErrorParam("store_count", ParamKind.U32),
        ErrorParam("reload_count", ParamKind.U32),
        ErrorParam("spill_cause", ParamKind.STRING),
    ),
    fix_hint=(
        "Use the pressure contributors for '{value_name}' to shorten the "
        "live range or choose a lower-pressure configuration"
    ),
)

# ERR_BACKEND_009: Spill/reload IR was inserted.
ERR_BACKEND_009 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=9,
    severity=Severity.WARNING,
    summary="Spill and reload operations inserted.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "inserted spill storage '{storage_name}' for {value_class} value "
        "'{value_name}' in '@{function_name}' using {spill_bytes} byte(s), "
        "{store_count} store(s), and {reload_count} reload(s)"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("value_class", ParamKind.STRING),
        ErrorParam("storage_name", ParamKind.STRING),
        ErrorParam("spill_bytes", ParamKind.U32),
        ErrorParam("store_count", ParamKind.U32),
        ErrorParam("reload_count", ParamKind.U32),
    ),
)

# ERR_BACKEND_010: Occupancy or resource estimate was recorded.
ERR_BACKEND_010 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=10,
    severity=Severity.REMARK,
    summary="Occupancy/resource estimate recorded.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "'@{function_name}' uses {used} of {budget} {resource_name} unit(s) "
        "with estimated occupancy {occupancy_percent}% limited by "
        "'{limiting_resource}'"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("resource_name", ParamKind.STRING),
        ErrorParam("budget", ParamKind.U32),
        ErrorParam("used", ParamKind.U32),
        ErrorParam("occupancy_percent", ParamKind.U32),
        ErrorParam("limiting_resource", ParamKind.STRING),
    ),
)

# ERR_BACKEND_013: Schedule resource bottleneck estimate was recorded.
ERR_BACKEND_013 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=13,
    severity=Severity.REMARK,
    summary="Schedule resource bottleneck estimate recorded.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "'@{function_name}' resource '{resource_name}' is a schedule "
        "bottleneck: {estimated_min_cycles} estimated cycle(s) from "
        "{total_unit_cycles} unit-cycle(s), {use_count} use(s), capacity "
        "{capacity_per_cycle}/cycle, peak {peak_units_per_cycle}/cycle, "
        "contention group {contention_group}"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("resource_name", ParamKind.STRING),
        ErrorParam("capacity_per_cycle", ParamKind.U32),
        ErrorParam("contention_group", ParamKind.U32),
        ErrorParam("use_count", ParamKind.U32),
        ErrorParam("total_unit_cycles", ParamKind.U64),
        ErrorParam("estimated_min_cycles", ParamKind.U64),
        ErrorParam("peak_units_per_cycle", ParamKind.U32),
    ),
)

# ERR_BACKEND_014: Schedule hazard gap requires downstream materialization.
ERR_BACKEND_014 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=14,
    severity=Severity.REMARK,
    summary="Schedule hazard gap requires materialization.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "'@{function_name}' descriptor '{descriptor_key}' has unresolved "
        "{hazard_kind} hazard on {reference_kind} '{reference_name}': "
        "requires distance {required_distance}, schedule distance "
        "{actual_distance}, and delay/wait {required_delay} cycle(s) between "
        "packet {producer_packet} and packet {consumer_packet}"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("descriptor_key", ParamKind.STRING),
        ErrorParam("hazard_kind", ParamKind.STRING),
        ErrorParam("reference_kind", ParamKind.STRING),
        ErrorParam("reference_name", ParamKind.STRING),
        ErrorParam("required_distance", ParamKind.U32),
        ErrorParam("actual_distance", ParamKind.U32),
        ErrorParam("required_delay", ParamKind.U32),
        ErrorParam("producer_packet", ParamKind.U32),
        ErrorParam("consumer_packet", ParamKind.U32),
    ),
)

# ERR_BACKEND_015: Schedule candidate selection was recorded.
ERR_BACKEND_015 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=15,
    severity=Severity.REMARK,
    summary="Schedule candidate selection recorded.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "'@{function_name}' scheduler chose '{chosen_packet}' over "
        "'{rejected_packet}' at block '{block_name}' ordinal "
        "{scheduled_ordinal}: {candidate_count} ready candidate(s), chosen "
        "dep-latency/latency/projected/killed/produced "
        "{chosen_dependency_latency_cycles}/{chosen_latency_cycles}/"
        "{chosen_projected_live_units}/"
        "{chosen_killed_live_units}/{chosen_produced_live_units}, chosen "
        "cliff class/units/penalty/next "
        "{chosen_pressure_cliff_reg_class_id}/{chosen_pressure_cliff_units}/"
        "{chosen_pressure_cliff_penalty}/"
        "{chosen_units_until_pressure_cliff}, rejected "
        "dep-latency/latency/projected/killed/produced "
        "{rejected_dependency_latency_cycles}/{rejected_latency_cycles}/"
        "{rejected_projected_live_units}/"
        "{rejected_killed_live_units}/{rejected_produced_live_units}, rejected "
        "cliff class/units/penalty/next "
        "{rejected_pressure_cliff_reg_class_id}/"
        "{rejected_pressure_cliff_units}/{rejected_pressure_cliff_penalty}/"
        "{rejected_units_until_pressure_cliff}"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("block_name", ParamKind.STRING),
        ErrorParam("scheduled_ordinal", ParamKind.U32),
        ErrorParam("candidate_count", ParamKind.U32),
        ErrorParam("chosen_packet", ParamKind.STRING),
        ErrorParam("rejected_packet", ParamKind.STRING),
        ErrorParam("chosen_dependency_latency_cycles", ParamKind.U32),
        ErrorParam("chosen_latency_cycles", ParamKind.U32),
        ErrorParam("chosen_projected_live_units", ParamKind.U64),
        ErrorParam("chosen_killed_live_units", ParamKind.U64),
        ErrorParam("chosen_produced_live_units", ParamKind.U64),
        ErrorParam("chosen_pressure_cliff_reg_class_id", ParamKind.U32),
        ErrorParam("chosen_pressure_cliff_units", ParamKind.U32),
        ErrorParam("chosen_pressure_cliff_penalty", ParamKind.U32),
        ErrorParam("chosen_units_until_pressure_cliff", ParamKind.U32),
        ErrorParam("rejected_dependency_latency_cycles", ParamKind.U32),
        ErrorParam("rejected_latency_cycles", ParamKind.U32),
        ErrorParam("rejected_projected_live_units", ParamKind.U64),
        ErrorParam("rejected_killed_live_units", ParamKind.U64),
        ErrorParam("rejected_produced_live_units", ParamKind.U64),
        ErrorParam("rejected_pressure_cliff_reg_class_id", ParamKind.U32),
        ErrorParam("rejected_pressure_cliff_units", ParamKind.U32),
        ErrorParam("rejected_pressure_cliff_penalty", ParamKind.U32),
        ErrorParam("rejected_units_until_pressure_cliff", ParamKind.U32),
    ),
)

# ERR_BACKEND_016: Schedule model quality was recorded.
ERR_BACKEND_016 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=16,
    severity=Severity.REMARK,
    summary="Schedule model quality recorded.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "'@{function_name}' schedule class '{schedule_class}' uses "
        "{model_quality} model quality with {latency_kind} latency "
        "{latency_cycles} cycle(s), {issue_use_count} resource row(s), "
        "{hazard_count} hazard row(s), and {use_count} scheduled use(s)"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("schedule_class", ParamKind.STRING),
        ErrorParam("model_quality", ParamKind.STRING),
        ErrorParam("latency_kind", ParamKind.STRING),
        ErrorParam("latency_cycles", ParamKind.U32),
        ErrorParam("issue_use_count", ParamKind.U32),
        ErrorParam("hazard_count", ParamKind.U32),
        ErrorParam("use_count", ParamKind.U32),
    ),
)

# ERR_BACKEND_017: Target memory access selection was recorded.
ERR_BACKEND_017 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=17,
    severity=Severity.REMARK,
    summary="Target memory access decision recorded.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "{decision} {operation_kind} memory packet '{packet_key}' for "
        "'@{function_name}' in {memory_space}: element {element_bytes} "
        "byte(s), vector lanes {vector_lanes}, dynamic stride "
        "{dynamic_stride_bytes} byte(s), vector lane stride "
        "{vector_lane_stride_bytes} byte(s), bank stride {bank_stride_words} "
        "word(s), conflict degree {bank_conflict_degree}, bank conflict "
        "'{bank_conflict_kind}'"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("memory_space", ParamKind.STRING),
        ErrorParam("operation_kind", ParamKind.STRING),
        ErrorParam("packet_key", ParamKind.STRING),
        ErrorParam("decision", ParamKind.STRING),
        ErrorParam("element_bytes", ParamKind.U32),
        ErrorParam("vector_lanes", ParamKind.U32),
        ErrorParam("dynamic_stride_bytes", ParamKind.U32),
        ErrorParam("vector_lane_stride_bytes", ParamKind.U32),
        ErrorParam("bank_stride_words", ParamKind.U32),
        ErrorParam("bank_conflict_degree", ParamKind.U32),
        ErrorParam("bank_conflict_kind", ParamKind.STRING),
    ),
)

# ERR_BACKEND_018: Target-low packet selection was recorded.
ERR_BACKEND_018 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=18,
    severity=Severity.REMARK,
    summary="Target-low packet decision recorded.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "{decision} {packet_category} low packet '{packet_key}' for "
        "'@{function_name}': operands {operand_count}, results "
        "{result_count}"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("packet_category", ParamKind.STRING),
        ErrorParam("packet_key", ParamKind.STRING),
        ErrorParam("decision", ParamKind.STRING),
        ErrorParam("operand_count", ParamKind.U32),
        ErrorParam("result_count", ParamKind.U32),
    ),
)

# ERR_BACKEND_019: Spill storage is unsupported by the selected target.
ERR_BACKEND_019 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=19,
    severity=Severity.ERROR,
    summary="Spill storage unsupported.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "cannot materialize {value_class} value '{value_name}' in "
        "'@{function_name}': spill slot space '{spill_slot_space}' maps to "
        "storage space '{storage_space}', but supported storage spaces are "
        "{supported_storage_spaces}"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("value_class", ParamKind.STRING),
        ErrorParam("spill_slot_space", ParamKind.STRING),
        ErrorParam("storage_space", ParamKind.STRING),
        ErrorParam("supported_storage_spaces", ParamKind.STRING_LIST),
    ),
    fix_hint=(
        "Choose a spill slot space supported by target lowering or implement "
        "target storage lowering for '{storage_space}'"
    ),
)

# ERR_BACKEND_020: Register assignment is outside an operand address map.
ERR_BACKEND_020 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=20,
    severity=Severity.ERROR,
    summary="Register assignment is not operand-addressable.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "cannot encode {value_class} value '{value_name}' for "
        "'@{function_name}' packet {packet_index} descriptor '{packet_key}' "
        "operand '{operand_field}': assigned register base {assigned_base} "
        "with {assigned_units} unit(s) is outside the encodable "
        "{address_map_kind} address map with {addressable_units} addressable "
        "unit(s)"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("packet_key", ParamKind.STRING),
        ErrorParam("packet_index", ParamKind.U32),
        ErrorParam("operand_field", ParamKind.STRING),
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("value_class", ParamKind.STRING),
        ErrorParam("assigned_base", ParamKind.U32),
        ErrorParam("assigned_units", ParamKind.U32),
        ErrorParam("address_map_kind", ParamKind.STRING),
        ErrorParam("addressable_units", ParamKind.U32),
    ),
    fix_hint=(
        "Constrain allocation for '{operand_field}' to an addressable subset "
        "or materialize the target address state before final emission"
    ),
)

# ERR_BACKEND_021: Final emission-frame preparation failed.
ERR_BACKEND_021 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=21,
    severity=Severity.ERROR,
    summary="Final emission frame is not target-ready.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "could not prepare a final emission frame for '@{function_name}': "
        "{failure_code}; iteration {iteration_count} of {iteration_limit}, "
        "{spill_plan_count} pending spill plan(s), "
        "{spill_assignment_count} spill-slot assignment(s), and "
        "{scheduled_packet_count} scheduled packet(s)"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("failure_code", ParamKind.STRING),
        ErrorParam("iteration_count", ParamKind.U64),
        ErrorParam("iteration_limit", ParamKind.U64),
        ErrorParam("spill_plan_count", ParamKind.U64),
        ErrorParam("spill_assignment_count", ParamKind.U64),
        ErrorParam("scheduled_packet_count", ParamKind.U64),
    ),
    fix_hint=(
        "Implement target lowering for the remaining spill or address-state "
        "traffic, or constrain allocation so final emission is target-ready"
    ),
)

# ERR_BACKEND_022: Register location exceeds the allocation capacity.
ERR_BACKEND_022 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=22,
    severity=Severity.ERROR,
    summary="Register location exceeds allocation capacity.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "cannot place {request_kind} for '@{function_name}' in "
        "{register_class}: location range [{location_base}, {location_end}) "
        "with {location_count} unit(s) exceeds allocation capacity "
        "{allocation_capacity}"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("request_kind", ParamKind.STRING),
        ErrorParam("register_class", ParamKind.STRING),
        ErrorParam("location_base", ParamKind.U32),
        ErrorParam("location_count", ParamKind.U32),
        ErrorParam("location_end", ParamKind.U64),
        ErrorParam("allocation_capacity", ParamKind.U32),
    ),
    fix_hint=(
        "Constrain the requested fixed or reserved location to the target "
        "register class capacity, or select a target descriptor set with a "
        "larger allocation space"
    ),
)

# ERR_BACKEND_023: Allocation budget references an unknown register class.
ERR_BACKEND_023 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=23,
    severity=Severity.ERROR,
    summary="Allocation budget references an unknown register class.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "cannot apply allocation budget for '@{function_name}': register "
        "class '{register_class}' is not defined by descriptor set "
        "'{descriptor_set_key}'"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("register_class", ParamKind.STRING),
        ErrorParam("descriptor_set_key", ParamKind.STRING),
    ),
    fix_hint=(
        "Use a register class defined by the selected target descriptor set, "
        "or select a descriptor set that provides the requested class"
    ),
)

# ERR_BACKEND_024: Allocation budget duplicates register-class storage.
ERR_BACKEND_024 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=24,
    severity=Severity.ERROR,
    summary="Allocation budget duplicates register-class storage.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "cannot apply allocation budget for '@{function_name}': register "
        "class '{register_class}' aliases storage already budgeted by "
        "'{existing_register_class}'"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("register_class", ParamKind.STRING),
        ErrorParam("existing_register_class", ParamKind.STRING),
    ),
    fix_hint=("Specify one budget for each aliasing register-storage class group"),
)

# ERR_BACKEND_025: Reserved range is missing a register class.
ERR_BACKEND_025 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=25,
    severity=Severity.ERROR,
    summary="Reserved range is missing a register class.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "cannot apply reserved range {reserved_range_index} for "
        "'@{function_name}': register_class is empty"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("reserved_range_index", ParamKind.U64),
    ),
    fix_hint="Provide the descriptor-set register class owned by the range",
)

# ERR_BACKEND_026: Reserved range references an unknown register class.
ERR_BACKEND_026 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=26,
    severity=Severity.ERROR,
    summary="Reserved range references an unknown register class.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "cannot apply reserved range {reserved_range_index} for "
        "'@{function_name}': register class '{register_class}' is not "
        "defined by descriptor set '{descriptor_set_key}'"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("reserved_range_index", ParamKind.U64),
        ErrorParam("register_class", ParamKind.STRING),
        ErrorParam("descriptor_set_key", ParamKind.STRING),
    ),
    fix_hint=(
        "Reserve storage from a register class defined by the selected "
        "target descriptor set"
    ),
)

# ERR_BACKEND_027: Allocation location kind is unsupported for the request.
ERR_BACKEND_027 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=27,
    severity=Severity.ERROR,
    summary="Allocation location kind is unsupported for the request.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "cannot place {request_kind} for '@{function_name}': location kind "
        "'{location_kind}' is not target-visible register storage"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("request_kind", ParamKind.STRING),
        ErrorParam("location_kind", ParamKind.STRING),
    ),
    fix_hint="Use physical_register or target_id for fixed and reserved ranges",
)

# ERR_BACKEND_028: Allocation location range is empty.
ERR_BACKEND_028 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=28,
    severity=Severity.ERROR,
    summary="Allocation location range is empty.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "cannot place {request_kind} for '@{function_name}': location range "
        "at {location_base} in '{location_kind}' has zero units"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("request_kind", ParamKind.STRING),
        ErrorParam("location_kind", ParamKind.STRING),
        ErrorParam("location_base", ParamKind.U32),
    ),
    fix_hint="Request at least one storage unit",
)

# ERR_BACKEND_029: Allocation location range overflows uint32 storage.
ERR_BACKEND_029 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=29,
    severity=Severity.ERROR,
    summary="Allocation location range overflows uint32 storage.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "cannot place {request_kind} for '@{function_name}': location range "
        "[{location_base}, {location_end}) in '{location_kind}' with "
        "{location_count} unit(s) exceeds uint32 storage"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("request_kind", ParamKind.STRING),
        ErrorParam("location_kind", ParamKind.STRING),
        ErrorParam("location_base", ParamKind.U32),
        ErrorParam("location_count", ParamKind.U32),
        ErrorParam("location_end", ParamKind.U64),
    ),
    fix_hint="Reduce the requested base or unit count",
)

# ERR_BACKEND_030: Allocation location kind does not match the register class.
ERR_BACKEND_030 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=30,
    severity=Severity.ERROR,
    summary="Allocation location kind does not match the register class.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "cannot place {request_kind} for '@{function_name}' in "
        "{register_class}: requested location kind '{location_kind}' but "
        "the class uses '{expected_location_kind}'"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("request_kind", ParamKind.STRING),
        ErrorParam("register_class", ParamKind.STRING),
        ErrorParam("location_kind", ParamKind.STRING),
        ErrorParam("expected_location_kind", ParamKind.STRING),
    ),
    fix_hint="Use the location kind owned by the selected register class",
)

# ERR_BACKEND_031: Reserved ranges overlap.
ERR_BACKEND_031 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=31,
    severity=Severity.ERROR,
    summary="Reserved ranges overlap.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "cannot apply reserved range {reserved_range_index} for "
        "'@{function_name}' in {register_class}: range "
        "[{location_base}, {location_end}) overlaps existing range "
        "{existing_reserved_range_index} [{existing_location_base}, "
        "{existing_location_end})"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("reserved_range_index", ParamKind.U64),
        ErrorParam("existing_reserved_range_index", ParamKind.U64),
        ErrorParam("register_class", ParamKind.STRING),
        ErrorParam("location_base", ParamKind.U32),
        ErrorParam("location_count", ParamKind.U32),
        ErrorParam("location_end", ParamKind.U64),
        ErrorParam("existing_location_base", ParamKind.U32),
        ErrorParam("existing_location_count", ParamKind.U32),
        ErrorParam("existing_location_end", ParamKind.U64),
    ),
    fix_hint="Make target-owned reserved ranges disjoint",
)

# ERR_BACKEND_032: Fixed allocation references an invalid SSA value ID.
ERR_BACKEND_032 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=32,
    severity=Severity.ERROR,
    summary="Fixed allocation references an invalid SSA value ID.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "cannot apply fixed value {fixed_value_index} for "
        "'@{function_name}': value id {value_id} is outside the module value "
        "table with {value_count} entries"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("fixed_value_index", ParamKind.U64),
        ErrorParam("value_id", ParamKind.U32),
        ErrorParam("value_count", ParamKind.U64),
    ),
    fix_hint="Use an SSA value from the allocated function",
)

# ERR_BACKEND_033: Fixed allocation value has no allocatable interval.
ERR_BACKEND_033 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=33,
    severity=Severity.ERROR,
    summary="Fixed allocation value has no allocatable interval.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "cannot place fixed value '%{value_name}' for '@{function_name}': "
        "value id {value_id} has no allocatable live interval"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("value_id", ParamKind.U32),
    ),
    fix_hint="Only fix values that participate in low allocation",
)

# ERR_BACKEND_034: Fixed allocation unit count does not match liveness.
ERR_BACKEND_034 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=34,
    severity=Severity.ERROR,
    summary="Fixed allocation unit count does not match liveness.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "cannot place fixed value '%{value_name}' for '@{function_name}': "
        "value id {value_id} requires {required_unit_count} unit(s), but "
        "the fixed location has {location_count}"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("value_id", ParamKind.U32),
        ErrorParam("required_unit_count", ParamKind.U32),
        ErrorParam("location_count", ParamKind.U32),
    ),
    fix_hint="Match the fixed location size to the value allocation unit count",
)

# ERR_BACKEND_035: Fixed allocation base is misaligned.
ERR_BACKEND_035 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=35,
    severity=Severity.ERROR,
    summary="Fixed allocation base is misaligned.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "cannot place fixed value '%{value_name}' for '@{function_name}': "
        "value id {value_id} location base {location_base} is not aligned "
        "to {required_alignment}"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("value_id", ParamKind.U32),
        ErrorParam("location_base", ParamKind.U32),
        ErrorParam("required_alignment", ParamKind.U32),
    ),
    fix_hint="Choose a fixed location base aligned for the value",
)

# ERR_BACKEND_036: Fixed allocation duplicates an SSA value.
ERR_BACKEND_036 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=36,
    severity=Severity.ERROR,
    summary="Fixed allocation duplicates an SSA value.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "cannot apply fixed value {fixed_value_index} for "
        "'@{function_name}': value '%{value_name}' with id {value_id} was "
        "already fixed by request {existing_fixed_value_index}"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("fixed_value_index", ParamKind.U64),
        ErrorParam("existing_fixed_value_index", ParamKind.U64),
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("value_id", ParamKind.U32),
    ),
    fix_hint="Provide at most one fixed allocation request per SSA value",
)

# ERR_BACKEND_037: Fixed allocation selector is unresolved.
ERR_BACKEND_037 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=37,
    severity=Severity.ERROR,
    summary="Fixed allocation selector is unresolved.",
    message=(
        "cannot apply fixed allocation request for '@{function_name}': "
        "selector '%{value_name}' does not name a value in the low function"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("value_name", ParamKind.STRING),
    ),
    fix_hint="Use an SSA value name or numeric value id from the selected low function",
)

# ERR_BACKEND_038: Fixed allocation selector is ambiguous.
ERR_BACKEND_038 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=38,
    severity=Severity.ERROR,
    summary="Fixed allocation selector is ambiguous.",
    message=(
        "cannot apply fixed allocation request for '@{function_name}': "
        "selector '%{value_name}' matches value ids {first_value_id} and "
        "{second_value_id}"
    ),
    params=(
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("value_name", ParamKind.STRING),
        ErrorParam("first_value_id", ParamKind.U32),
        ErrorParam("second_value_id", ParamKind.U32),
    ),
    fix_hint="Use a numeric value id or give the intended SSA value a unique name",
)

# ERR_BACKEND_039: Schedule pressure cliff references an unknown register class.
ERR_BACKEND_039 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=39,
    severity=Severity.ERROR,
    summary="Schedule pressure cliff references an unknown register class.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "cannot apply pressure cliff for '@{function_name}': register class "
        "'{register_class}' is not defined by descriptor set "
        "'{descriptor_set_key}'"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("register_class", ParamKind.STRING),
        ErrorParam("descriptor_set_key", ParamKind.STRING),
    ),
    fix_hint=(
        "Use a pressure cliff register class defined by the selected target "
        "descriptor set"
    ),
)

# ERR_BACKEND_040: Source memory cache-policy decision was recorded.
ERR_BACKEND_040 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=40,
    severity=Severity.REMARK,
    summary="Source memory cache-policy decision recorded.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "{decision} cache policy '{cache_scope}/{cache_temporal}' for "
        "'@{function_name}' op '{op_name}' {operation_kind} in "
        "{memory_space}: decision '{decision_key}', encoding "
        "'{encoding_key}', scope_attr present {scope_attr_present} value "
        "{scope_attr}, th_attr present {th_attr_present} value {th_attr}, "
        "nt_attr present {nt_attr_present} value {nt_attr}"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("memory_space", ParamKind.STRING),
        ErrorParam("operation_kind", ParamKind.STRING),
        ErrorParam("cache_scope", ParamKind.STRING),
        ErrorParam("cache_temporal", ParamKind.STRING),
        ErrorParam("decision_key", ParamKind.STRING),
        ErrorParam("decision", ParamKind.STRING),
        ErrorParam("encoding_key", ParamKind.STRING),
        ErrorParam("scope_attr_present", ParamKind.BOOL),
        ErrorParam("scope_attr", ParamKind.I64),
        ErrorParam("th_attr_present", ParamKind.BOOL),
        ErrorParam("th_attr", ParamKind.I64),
        ErrorParam("nt_attr_present", ParamKind.BOOL),
        ErrorParam("nt_attr", ParamKind.I64),
    ),
)

# ERR_BACKEND_041: Source memory prefetch decision was recorded.
ERR_BACKEND_041 = ErrorDef(
    domain=ErrorDomain.BACKEND,
    code=41,
    severity=Severity.REMARK,
    summary="Source memory prefetch decision recorded.",
    message=(
        "target '{target_key}' export '{export_name}' config '{config_key}' "
        "{decision} prefetch '{prefetch_intent}/{prefetch_locality}' for "
        "'@{function_name}' op '{op_name}' in {memory_space}: decision "
        "'{decision_key}', packet '{packet_key}', immediate offset "
        "{immediate_offset}, scalar byte offset {scalar_byte_offset}, dynamic "
        "index '{dynamic_index_kind}', count {count}"
    ),
    params=(
        ErrorParam("target_key", ParamKind.STRING),
        ErrorParam("export_name", ParamKind.STRING),
        ErrorParam("config_key", ParamKind.STRING),
        ErrorParam("function_name", ParamKind.STRING),
        ErrorParam("op_name", ParamKind.STRING),
        ErrorParam("memory_space", ParamKind.STRING),
        ErrorParam("prefetch_intent", ParamKind.STRING),
        ErrorParam("prefetch_locality", ParamKind.STRING),
        ErrorParam("decision_key", ParamKind.STRING),
        ErrorParam("decision", ParamKind.STRING),
        ErrorParam("packet_key", ParamKind.STRING),
        ErrorParam("immediate_offset", ParamKind.I64),
        ErrorParam("scalar_byte_offset", ParamKind.U32),
        ErrorParam("dynamic_index_kind", ParamKind.STRING),
        ErrorParam("count", ParamKind.U32),
    ),
)

ALL_BACKEND_ERRORS: tuple[ErrorDef, ...] = (
    ERR_BACKEND_003,
    ERR_BACKEND_005,
    ERR_BACKEND_006,
    ERR_BACKEND_008,
    ERR_BACKEND_009,
    ERR_BACKEND_010,
    ERR_BACKEND_013,
    ERR_BACKEND_014,
    ERR_BACKEND_015,
    ERR_BACKEND_016,
    ERR_BACKEND_017,
    ERR_BACKEND_018,
    ERR_BACKEND_019,
    ERR_BACKEND_020,
    ERR_BACKEND_021,
    ERR_BACKEND_022,
    ERR_BACKEND_023,
    ERR_BACKEND_024,
    ERR_BACKEND_025,
    ERR_BACKEND_026,
    ERR_BACKEND_027,
    ERR_BACKEND_028,
    ERR_BACKEND_029,
    ERR_BACKEND_030,
    ERR_BACKEND_031,
    ERR_BACKEND_032,
    ERR_BACKEND_033,
    ERR_BACKEND_034,
    ERR_BACKEND_035,
    ERR_BACKEND_036,
    ERR_BACKEND_037,
    ERR_BACKEND_038,
    ERR_BACKEND_039,
    ERR_BACKEND_040,
    ERR_BACKEND_041,
)
