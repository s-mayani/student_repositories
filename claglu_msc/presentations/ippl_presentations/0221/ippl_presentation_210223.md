---
marp: true
paginate: true
# auto-scaling true
---

## Chainable Stencil Operators
<!-- <br/> -->
Current interface:

1. Define general stencil along 1 dimension
<br/>
2. Instantiate wrapper class for a specific stencil and dimension
Defines `T operator()(Index idx){}` on a field or another stencil instance
<br/>
3. Potentially concatenate $N$ of these stenils
<br/>

Possible use case:
Generate Hessian with different numerical stencils along each dimension

---
## Appendix: Code Snippets

Stencil Definition:
```C++
template<Dim D, typename T, class Callable>
inline T forward_stencil(const Index &idx, const T &hInv, const Callable &F){
    return 0.5 * hInv * (-3.0*F(idx) + 4.0*F(idx.get_shifted<D>(1)) - F(idx.get_shifted<D>(2)));
}
```
<br/>

Chainable operator along dimension `D` with stencil `Diff`:
```C++
template<Dim D, typename T, DiffType Diff, class C>
class DiffOpChain{
    // [...]
    const inline T operator()(Index idx) const;
};
```