// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/packet_hazard_plan_json.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/util/stream.h"

namespace loom {
namespace {

TEST(LowPacketHazardPlanJsonTest, MissingDataOmitsActionAndProgress) {
  const loom_low_packet_hazard_plan_record_t record = {
      .kind = LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_MISSING_TARGET_DATA,
      .action_id = LOOM_LOW_PACKET_HAZARD_PLAN_ACTION_NONE,
      .reason_id = 4,
      .reason_name = IREE_SV("target.missing-data"),
      .producer_node_index = LOOM_LOW_SCHEDULE_NODE_NONE,
      .producer_scheduled_ordinal = LOOM_LOW_PACKET_HAZARD_PLAN_ORDINAL_NONE,
      .producer_packet_index = LOOM_LOW_PACKET_HAZARD_PLAN_PACKET_NONE,
      .progress_class_id = LOOM_LOW_PACKET_PROGRESS_CLASS_NONE,
  };
  const loom_low_packet_hazard_plan_t plan = {
      .records = &record,
      .record_count = 1,
  };
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_EXPECT_OK(loom_low_packet_hazard_plan_write_json_array(&plan, &stream));
  const iree_string_view_t output = iree_string_builder_view(&builder);
  EXPECT_EQ(std::string(output.data, output.size),
            "[{\"index\":0,\"kind\":\"missing_target_data\",\"action_id\":null,"
            "\"action_name\":null,\"reason_id\":4,"
            "\"reason_name\":\"target.missing-data\",\"producer_node\":null,"
            "\"producer_packet\":null,\"producer_scheduled_ordinal\":null,"
            "\"consumer_node\":0,\"insertion_packet\":0,\"block\":0,"
            "\"scheduled_ordinal\":0,\"progress_class_id\":null,"
            "\"progress_class_name\":null,\"required\":null,\"observed\":null,"
            "\"residual\":null}]");
  iree_string_builder_deinitialize(&builder);
}

}  // namespace
}  // namespace loom
