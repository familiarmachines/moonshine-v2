cmake_minimum_required(VERSION 3.22.1)

project("moonshine-cuda" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -O3)
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
  add_compile_options(-mcpu=native)
else()
  add_compile_options(-march=native)
endif()

add_executable(moonshine-ref
  main-ref.cpp
  model.cpp
  safetensors.cpp
  tokenizer.cpp
  wav.cpp
)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  # Must be set before enable_language(CUDA), which otherwise pins the
  # toolkit default (compute_52) in the cache - that both blocks wmma/mma.h
  # (arch-guarded >= 700) and forces driver JIT from generic PTX at runtime.
  set(CMAKE_CUDA_ARCHITECTURES 87)  # Jetson Orin
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_executable(moonshine-cuda
    main-cuda.cpp
    model_cuda.cu
    model.cpp
    safetensors.cpp
    tokenizer.cpp
    wav.cpp
  )
  target_link_libraries(moonshine-cuda PRIVATE CUDA::cublas CUDA::cudart)

  add_executable(moonshine-bench
    main-bench.cpp
    model_cuda.cu
    model.cpp
    safetensors.cpp
    tokenizer.cpp
    wav.cpp
  )
  target_link_libraries(moonshine-bench PRIVATE CUDA::cublas CUDA::cudart)

  # Encoder profiling harness: steady-state streaming ticks, encode only.
  add_executable(enc-only
    enc-only.cpp
    model_cuda.cu
    model.cpp
    safetensors.cpp
    tokenizer.cpp
    wav.cpp
  )
  target_link_libraries(enc-only PRIVATE CUDA::cublas CUDA::cudart)

  # Live microphone transcriber. ALSA is optional: with it, mic capture is
  # available; without it the target still builds as a WAV-driven tester.
  add_executable(moonshine-mic
    main-mic.cpp
    model_cuda.cu
    model.cpp
    safetensors.cpp
    tokenizer.cpp
    wav.cpp
  )
  target_link_libraries(moonshine-mic PRIVATE CUDA::cublas CUDA::cudart)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(ALSA alsa)
  endif()
  if(ALSA_FOUND)
    target_compile_definitions(moonshine-mic PRIVATE MOONSHINE_HAVE_ALSA)
    target_include_directories(moonshine-mic PRIVATE ${ALSA_INCLUDE_DIRS})
    target_link_libraries(moonshine-mic PRIVATE ${ALSA_LIBRARIES})
  else()
    message(STATUS "ALSA not found - moonshine-mic builds WAV-only (no mic)")
  endif()

  # Cooperative-groups grid sync in the fused encoder needs relocatable
  # device code.
  set_target_properties(moonshine-cuda moonshine-bench enc-only moonshine-mic
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON
  )
else()
  message(STATUS "No CUDA compiler found - building CPU reference only")
endif()
