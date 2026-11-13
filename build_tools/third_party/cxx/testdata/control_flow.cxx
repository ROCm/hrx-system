constexpr int classify(int value) {
  int result = 0;
  for (int index = 0; index < value; ++index) {
    if ((index & 1) == 0) {
      result += index;
    } else {
      result -= index;
    }
  }
  switch (value) {
    case 0:
      return result;
    case 1:
      return result + 1;
    default:
      return result + 2;
  }
}

static_assert(classify(4) == 0);
