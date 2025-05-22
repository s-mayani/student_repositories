---
marp: true
paginate: true
---
<!-- theme: gaia -->
## Design Question
<br/>

Create new `onesided_hess()` function:

* one-sided differencing on domain boundary
* regular (centered) difference for the other gridpoints

This only makes sense for non-periodic domains (i.e. open B.C.).

<br/>
What should be done when the user calls `onesided_hess()`  on a periodic field?

---
## Potential approaches
<br/>

* Throw a runtime error and abort (stating the illegal combination of BC's and this operator)
* Fall back to the previous implementation employing centered difference only (`hess()`) -> outcome might come as a surprise for the user
* \[...\]


---
## Question on TMP
<br/>
Where is the loop actually executed?


`hess()` creates struct access via `operator()(size_t i, size_t j, size_t k)`



```
[...]
result = {0.0, 0.0, 0.0};
result = hess(field);    
                         
result = result - exact;

// Actual index access during reduction
Kokkos::parallel_reduce(...)
```