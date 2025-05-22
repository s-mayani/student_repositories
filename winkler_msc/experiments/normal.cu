#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <xoshiro.hpp>
#include <iostream>
#include <cmath>
#include <curand.h>
#include "common.cu"
double ncdf(double x){
    #ifndef __CUDA_ARCH__
    using std::erf;
    #endif
    return 0.5 * (1.0 + erf(x * 0.70710678118654752440));
}
template<typename scalar>
double erfinv_(scalar y){
    #ifndef __CUDA_ARCH__
    using std::sqrt;
    using std::erf;
    using std::exp;
    using std::log;    
    using std::abs;    
    #endif
    constexpr scalar CENTRAL_RANGE = 0.7;
    scalar x, z, num, dem; /*working variables */
    /* coefficients in rational expansion */
    scalar a[4] = {0.886226899, -1.645349621, 0.914624893, -0.140543331};
    scalar b[4] = {-2.118377725, 1.442710462, -0.329097515, 0.012229801};
    scalar c[4] = {-1.970840454, -1.624906493, 3.429567803, 1.641345311};
    scalar d[2] = {3.543889200, 1.637067800};
    if (fabs(y) > 1.0) return (NAN); /* This needs IEEE constant*/
    if (fabs(y) == 1.0) return ((copysign(1.0, y)) * 1e308);
    if (fabs(y) <= CENTRAL_RANGE) {
        z = y * y;
        num = (((a[3] * z + a[2]) * z + a[1]) * z + a[0]);
        dem = ((((b[3] * z + b[2]) * z + b[1]) * z + b[0]) * z + 1.0);
        x = y * num / dem;
    } else if ((fabs(y) > CENTRAL_RANGE) && (fabs(y) < 1.0)) {
        z = sqrt(-log((1.0 - fabs(y)) / 2.0));
        num = ((c[3] * z + c[2]) * z + c[1]) * z + c[0];
        dem = (d[1] * z + d[0]) * z + 1.0;
        x = (copysign(1.0, y)) * num / dem;
    }
    /* Two steps of Newton-Raphson correction */
    x = x - (erf(x) - y) / ((2.0 / sqrt(M_PI)) * exp(-x * x));
    x = x - (erf(x) - y) / ((2.0 / sqrt(M_PI)) * exp(-x * x));

    return x;
}
double convertToNormal(double sample, double sigma) {
    double z = 1.4142135623730950488 * erfinv_(2 * sample - 1);
    return z * sigma; // z has mean 0 and standard deviation 1, so scale by sigma
}

constexpr size_t GB = 1073741824ull;


int main() {
    for(double x = -5;x <= 5;x += 1.0 / 128){
        std::cout << x << " " << convertToNormal(ncdf(x), 1.0) << "\n";
    }
    /*
    float* data = cuda_malloc_helper<float>(GB);
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10);
    curandSetPseudoRandomGeneratorSeed(gen, 10574837847384787111ull);
    curandSetGeneratorOffset(gen, 1000);
    auto t1 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    for(int i = 0;i < 100;i++){
        curandGenerateNormal(gen, data, GB, 0.0f, 1.0f);
        thrust::device_ptr<float> dptr(data);
        thrust::device_ptr<float> maxptr = thrust::max_element(dptr, dptr + GB);
        std::cout << *maxptr << std::endl;
    }
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::cout << (double(GB * 100)) / (double(t2 - t1) / 1e9) << " time\n";*/
}
