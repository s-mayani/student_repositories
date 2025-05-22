cmake_minimum_required(VERSION 3.19)
include(FetchContent)

FetchContent_Declare(
    eglutils_fc
    GIT_REPOSITORY https://github.com/manuel5975p/frast.git
    GIT_TAG        egl_branch
)
FetchContent_MakeAvailable(eglutils_fc)
project(lolfinally LANGUAGES CXX CUDA)
set(CMAKE_CXX_FLAGS_RELEASE  "-O3 -DNDEBUG -I/home/manuel/Documents/winkler_msc/experiments/build/_deps/eglutils_fc-src -I/home/manuel/Documents/winkler_msc/experiments/build/_deps/glad-build/include")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG -I/home/manuel/Documents/winkler_msc/experiments/build/_deps/eglutils_fc-src -I/home/manuel/Documents/winkler_msc/experiments/build/_deps/glad-build/include")
#add_executable(thrust_omp_test "thrust_omp_test.cpp")
#target_compile_options(thrust_omp_test PUBLIC "-fopenmp" "-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP")
#target_link_options(thrust_omp_test PUBLIC "-fopenmp")
#target_include_directories(thrust_omp_test PUBLIC "/opt/cuda/include")
add_executable(gridtest "gridtest.cu" "bunch_impl.cu" "cfg.cpp")
target_compile_features(gridtest PUBLIC cxx_std_20)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda")
set_target_properties(gridtest PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(gridtest eglutils cufft)
include_directories("${eglutils_fc_SOURCE_DIR}")