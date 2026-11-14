#pragma once
#include <cstddef>

namespace loom_hybrid {
enum class Stage { prefix, ffn, residual };

// Emitters retain each bank until its residual completes and propagate device failures.
template <typename Emit>
bool schedule_pair(std::size_t layers, Emit&& emit) {
  if (!layers) return false;
  if (!emit(Stage::prefix, 0, 0) || !emit(Stage::ffn, 0, 0)) return false;
  for (std::size_t layer = 0; layer < layers; ++layer) {
    if (!emit(Stage::prefix, 1, layer) || !emit(Stage::ffn, 1, layer) ||
        !emit(Stage::residual, 0, layer)) return false;
    if (layer + 1 < layers &&
        (!emit(Stage::prefix, 0, layer + 1) ||
         !emit(Stage::ffn, 0, layer + 1))) return false;
    if (!emit(Stage::residual, 1, layer)) return false;
  }
  return true;
}
}  // namespace loom_hybrid
