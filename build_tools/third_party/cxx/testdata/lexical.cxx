constexpr auto binary = 0b1010'0110u;
constexpr auto hexadecimal = 0x1.fp+3;
constexpr auto message = u8"frontend \u03bb";
constexpr auto raw = R"delimiter(raw\ntext)delimiter";

static_assert(binary == 166u);
static_assert(hexadecimal == 15.5);
