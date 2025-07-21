---
marp: true
paginate: true
---
<!-- theme: gaia -->
## Defining Subfield from Field
<br/>

Used already in `HaloCells.hpp`

Interface in Field class (`field.hpp`):
```C++
// Creates new field containing subview of initial view data
template<class T, unsigned Dim, class M, class C>                                            
template <typename ...Args>                                                                  
Field<T,Dim,M,C> Field<T,Dim,M,C>::subField(Mesh_t &m, Layout_t &l, int nghost, Args... args);
```

Is there anything similarly already implemented?

---
## Hessian with forward & backward Difference


```C++
template <typename IdxOp=std::binary_function<size_t,size_t,size_t>, typename T, unsigned Dim, class M, class C>
auto forwardHess(Field<T, Dim, M, C>& u){
    return onesidedHess<std::plus<size_t>>(u);
}

template <typename IdxOp=std::binary_function<size_t,size_t,size_t>, typename T, unsigned Dim, class M, class C>
auto backwardHess(Field<T, Dim, M, C>& u){
    return onesidedHess<std::minus<size_t>>(u);
}        
```