template <OpDim D, unsigned Dim, typename T, DiffType Diff, class Callable>
class DiffOpChain : public BaseDiffOp<D, Dim, T, Diff, Callable> {
public:
  typedef T value_type;
  typedef typename Field_t<Dim>::view_type FView_t;

  DiffOpChain(const FView_t &view, Vector_t<Dim> hInvVector)
      : BaseDiffOp<D, Dim, T, Diff, Callable>(view, hInvVector),
        leftOp_m(view, this->hInvVector_m) {}

  // Specialization to call the stencil operator on the left operator
  inline T operator()(size_type i, size_type j, size_type k) const {
    return this->template stencilOp(leftOp_m, i, j, k);
  }

private:
  // Need additional callable which might contain other operators
  const Callable leftOp_m;
};
