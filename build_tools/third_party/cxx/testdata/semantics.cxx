template <class T>
concept arithmetic = requires(T a, T b) {
  a + b;
  a * b;
};

template <arithmetic T>
constexpr T transform(T value) {
  return value * value + T{1};
}

constexpr int transform(int value, int bias) { return value + bias; }

struct accumulator {
  int value;
  constexpr int operator()(int input) const { return value + input; }
};

static_assert(transform(4) == 17);
static_assert(transform(4, 3) == 7);
static_assert(accumulator{5}(6) == 11);
