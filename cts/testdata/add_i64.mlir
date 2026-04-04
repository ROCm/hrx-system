module {
  util.func public @add_i64(%lhs: i64, %rhs: i64) -> i64 {
    %sum = arith.addi %lhs, %rhs : i64
    util.return %sum : i64
  }
}
