enum DiffType { Centered, Forward, Backward, CenteredDeriv2 };
enum OpDim { X, Y, Z };

template <OpDim D, typename T, class Callable>
inline T centered_stencil(const T &hInv, const Callable &F, size_type i,
                          size_type j, size_type k) {
  return 0.5 * hInv *
         (-shiftedIdxApply<D>(F, -1, i, j, k) +
          shiftedIdxApply<D>(F, 1, i, j, k));
}
