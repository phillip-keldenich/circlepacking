#MIT License
#
#Copyright (c) 2018 TU Braunschweig, Algorithms Group
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

if(NOT COMMAND util_setup_target)
	message(FATAL_ERROR "Before including UseCUDA.cmake, please do include(\"path/to/algcmake/util.cmake\" NO_POLICY_SCOPE)!")
endif()

if(NOT CUDA_FOUND)
	set(CUDA_PROPAGATE_HOST_FLAGS On)
	set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE Off)
	set(CUDA_HOST_COMPILATION_CPP On)
	set(CUDA_SEPARABLE_COMPILATION On)

	# theoretically, we could use the built-in CUDA support, but it is broken beyond repair (produces invalid command lines to nvcc on all systems I tested it on)
	# and we would have to use find_package(CUDA) anyways to link non-CUDA code like algcuda to CUDA (and similarly for include dirs).
	find_package(CUDA REQUIRED)
endif()

if(CUDA_FOUND AND NOT _ALGCUDA_FLAGS)
	set(UTIL_CUDA_SUPPORTED_ARCH "Auto" CACHE STRING "A set of architectures to support (Auto | Common | All | LIST(ARCH_AND_PTX ...) (last option is a list such as \"3.0 3.5+PTX 5.2(5.0) Maxwell\")")
	cuda_select_nvcc_arch_flags(_CUDA_NVCC_ARCH_FLAGS Auto)

	set(_ALGCUDA_FLAGS "--relocatable-device-code=true;--std=c++${CMAKE_CXX_STANDARD};${_CUDA_NVCC_ARCH_FLAGS};-I${CMAKE_CURRENT_LIST_DIR}/cuda-utils/cuh/include;-I${CMAKE_CURRENT_LIST_DIR}/cuda-utils/cpp/include")
	set(_ALGCUDA_FLAGS_R "")
	set(_ALGCUDA_FLAGS_D "-g")
	set(_ALGCUDA_FLAGS_LD "")

	if(NOT WIN32)
		set(_ALGCUDA_FLAGS "${_ALGCUDA_FLAGS};--compiler-options=-fvisibility=hidden;--compiler-options=-fvisibility-inlines-hidden")
		set(_ALGCUDA_FLAGS_LD "--compiler-options=-fPIC")
	endif()

	if(MSVC)
		set(_ALGCUDA_FLAGS "${_ALGCUDA_FLAGS};--compiler-options=/W4;--compiler-options=/wd4127;--compiler-options=/wd4244;--compiler-options=/wd4701")
	elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
		set(_ALGCUDA_FLAGS "${_ALGCUDA_FLAGS};--compiler-options=-Wall;--compiler-options=-Wextra;--compiler-options=-Wno-long-long;--compiler-options=-Wno-missing-braces")
		if(UTIL_ENABLE_HOST_SPECIFIC_OPTIMIZATIONS)
			set(_ALGCUDA_FLAGS_R "${_ALGCUDA_FLAGS_R};--compiler-options=-march=native;--compiler-options=-mtune=native")
		endif()
	endif()
endif()

if(CUDA_FOUND AND NOT TARGET algcuda)
	set(_ACSD "${CMAKE_CURRENT_LIST_DIR}/cuda-cpp-wrapper/src")
	set(_ALGCUDA_SOURCES "${_ACSD}/error.cpp" "${_ACSD}/memory.cpp" "${_ACSD}/properties.cpp" "${_ACSD}/device.cpp")

	add_library(algcuda EXCLUDE_FROM_ALL STATIC ${_ALGCUDA_SOURCES})
	util_setup_target(algcuda LIBRARIES "${CUDA_cudadevrt_LIBRARY};${CUDA_LIBRARIES}" INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/cuda-cpp-wrapper/include;${CMAKE_CURRENT_LIST_DIR}/cuda-utils/cpp/include;${CUDA_INCLUDE_DIRS}")
	set_target_properties(algcuda PROPERTIES POSITION_INDEPENDENT_CODE On)
endif()

if(TARGET algcuda AND NOT COMMAND util_add_cuda_sources)
	function(util_add_cuda_sources TARGETNAME CUDA_SOURCES)
		cuda_wrap_srcs("${TARGETNAME}" OBJ _CUDA_COMPILED_SRCS "${CUDA_SOURCES}" SHARED
			OPTIONS "${_ALGCUDA_FLAGS}"
			DEBUG "${_ALGCUDA_FLAGS_D}"
			RELEASE "${_ALGCUDA_FLAGS_R}"
			RELWITHDEBINFO "${_ALGCUDA_FLAGS_D};${_ALGCUDA_FLAGS_R}"
			MINSIZEREL "${_ALGCUDA_FLAGS_R}"
		)
		cuda_compute_separable_compilation_object_file_name(_CUDA_LINK_FILE "${TARGETNAME}" "${_CUDA_COMPILED_SRCS}")

		cuda_link_separable_compilation_objects("${_CUDA_LINK_FILE}" "${TARGETNAME}" "${_ALGCUDA_FLAGS};${_ALGCUDA_FLAGS_LD}" "${_CUDA_COMPILED_SRCS}")
		target_sources("${TARGETNAME}" PRIVATE "${CUDA_SOURCES};${_CUDA_COMPILED_SRCS};${_CUDA_LINK_FILE}")
		target_link_libraries("${TARGETNAME}" PRIVATE algcuda)

		set(UTIL_CUDA_CUBLAS_LIBRARIES "")
		foreach(LIBVAR ${CUDA_CUBLAS_LIBRARIES})
			if(LIBVAR)
				set(UTIL_CUDA_CUBLAS_LIBRARIES "${UTIL_CUDA_CUBLAS_LIBRARIES};${LIBVAR}")
			endif()
		endforeach()

		set(UTIL_CUDA_CUFFT_LIBRARIES "")
		foreach(LIBVAR ${CUDA_CUFFT_LIBRARIES})
			if(LIBVAR)
				set(UTIL_CUDA_CUFFT_LIBRARIES "${UTIL_CUDA_CUFFT_LIBRARIES};${LIBVAR}")
			endif()
		endforeach()

		target_link_libraries("${TARGETNAME}" PUBLIC ${UTIL_CUDA_CUBLAS_LIBRARIES})
		target_link_libraries("${TARGETNAME}" PUBLIC ${UTIL_CUDA_CUFFT_LIBRARIES})
	endfunction(util_add_cuda_sources)
endif()

