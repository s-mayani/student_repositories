// Cholesky decomposition for semi-positive definite matrices
// Avoids sqrt of negative numbers
// Computation is inplace
MatrixD_t LDLtCholesky3x3(const MatrixD_t &M) {
  MatrixD_t Q;
  VectorD_t row_factors;

  // Compute first row multiplicators
  row_factors[0] = M[1][0] / M[0][0];
  row_factors[1] = M[2][0] / M[0][0];

  // Eliminate value at [1,0]
  M[1] = M[1] - row_factors[0] * M[0];

  // Eliminate value at [2,0]
  M[2] = M[2] - row_factors[1] * M[0];

  // Eliminate value at [2,1]
  row_factors[2] = M[2][1] / M[1][1];
  M[2] = M[2] - row_factors[2] * M[1];

  // Check that the resulting matrix is semi-positive definite
  VectorD_t D = {M[0][0], M[1][1], M[2][2]};
  PAssert_GE(M[0][0], 0.0);
  PAssert_GE(M[1][1], 0.0);
  PAssert_GE(M[2][2], 0.0);

  // Compute Q = sqrt(D) * L^T
  // Where D is diag(M) and `row-factors` are the lower triangular values of L^T
  // Loop is unrolled as we only ever do this for 3x3 Matrices
  Q[0][0] = Kokkos::sqrt(M[0][0]);
  Q[1][0] = row_factors[0] * Kokkos::sqrt(M[1][1]);
  Q[1][1] = Kokkos::sqrt(M[1][1]);
  Q[2][0] = row_factors[1] * Kokkos::sqrt(M[2][2]);
  Q[2][1] = row_factors[2] * Kokkos::sqrt(M[2][2]);
  Q[2][2] = Kokkos::sqrt(M[2][2]);

  return Q;
}
