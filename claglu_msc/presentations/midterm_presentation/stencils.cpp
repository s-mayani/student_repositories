template<Dim D, typename T, class Callable>
inline T centered_stencil(const T &hInv, const Callable &F, size_type i, j, k){
    return 0.5 * hInv * (- shiftedIdxApply<D>(F,-1,i,j,k) + shiftedIdxApply<D>(F,1,i,j,k));
}
