#include <fuzz/macros.h>
#include <fuzz/nested.h>

#if __has_include(<fuzz/core.h>) && !__has_include(<fuzz/missing.h>)
constexpr auto FUZZ_CONCAT(fuzz_, value) = FUZZ_IDENTITY(fuzz::answer);
#endif

static_assert(fuzz::make_box(fuzz_value).get() == 42);
