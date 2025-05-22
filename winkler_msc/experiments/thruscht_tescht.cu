#include "common.cu"
#include <thread>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/iterator/counting_iterator.h>
int main(){
    float* pos = cuda_malloc_helper<float>(6);
    float* vel = cuda_malloc_helper<float>(6);
    float* hpos = cuda_malloc_host_helper<float>(6);
    float* hvel = cuda_malloc_host_helper<float>(6);

    hpos[0] = 100.0f;
    hpos[1] = 2.0f;
    hpos[2] = 3.0f;
    hpos[3] = 6.0f;
    hpos[4] = 10.0f;
    hpos[5] = 1.0f;
    
    hvel[0] = -100.0f;
    hvel[1] = -2.0f;
    hvel[2] = -3.0f;
    hvel[3] = -6.0f;
    hvel[4] = -10.0f;
    hvel[5] = -1.0f;
    cudaMemcpy(pos, hpos, 6 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vel, hvel, 6 * sizeof(float), cudaMemcpyHostToDevice);
    thrust::device_ptr<float> pptr(pos);
    thrust::device_ptr<float> vptr(vel);
    thrust::sort(
        thrust::device,
        thrust::make_zip_iterator(pptr, vptr),
        thrust::make_zip_iterator(pptr + 6, vptr + 6),  
        []__host__ __device__(thrust::tuple<float, float> x, thrust::tuple<float, float> y){
            return thrust::get<0>(x) < thrust::get<0>(y);
    });
    cudaDeviceSynchronize();
    for(size_t i = 0;i < 6;i++){
        std::cout << hpos[i] << ", " << hvel[i] << "\n";
    }
    
}
#ifdef ass
int maian(){
    float* dptr = cuda_malloc_managed_helper<float>(5);
    dptr[0] = 1.0f;
    dptr[1] = 2.0f;
    dptr[2] = 3.0f;
    dptr[3] = 4.0f;
    dptr[4] = 5.0f;
    dptr[5] = 6.0f;
    thrust::device_ptr<float> tptr(dptr);
    auto red = thrust::reduce(
        thrust::make_zip_iterator(tptr, thrust::make_counting_iterator(1)),
        thrust::make_zip_iterator(tptr + 6, thrust::make_counting_iterator(7)),
        thrust::tuple<float, int>{0.0f, 1},  []__host__ __device__(thrust::tuple<float, int> x, thrust::tuple<float, int> y){
        return thrust::tuple<float, int>{thrust::get<0>(x) + thrust::get<0>(y), thrust::get<1>(x) * thrust::get<1>(y)};
        printf("%d\n", thrust::get<1>(x));
    });
    cudaDeviceSynchronize();
    printf("%d\n", thrust::get<1>(red));
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
}
#endif
