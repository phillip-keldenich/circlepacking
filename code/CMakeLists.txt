cmake_minimum_required(VERSION 3.5)
project(circlepacking_prover CXX)

set(UTIL_BOOST_LIBRARIES program_options)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/util.cmake" NO_POLICY_SCOPE)
include("${CMAKE_CURRENT_SOURCE_DIR}/algcuda/UseCUDA.cmake")

set(CIRCLEPACKING_CUDA_SOURCES
	lane_packing/lane_packing.cu
	lane_packing/middle_configuration.cu
	lane_packing/start_configuration.cu
	lane_packing/end_configuration.cu
	lane_packing/collision_angles.cu
	lane_packing/lower_bound_radius_single_placement.cu
)

add_executable(circlepacking_prover prover.cpp)
util_setup_target(circlepacking_prover)
util_add_cuda_sources(circlepacking_prover "${CIRCLEPACKING_CUDA_SOURCES}")

