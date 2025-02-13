cmake_minimum_required(VERSION 3.10)
project(Optimal-Constraint-Thruster-Allocation)

set(CMAKE_CXX_STANDARD 17)  # or the C++ standard you require
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Find Eigen3
find_package(Eigen3 REQUIRED NO_DEFAULT_PATH PATHS /home/sigurdvb/anaconda3/envs/thrust_alloc_cpp_env/share/eigen3/cmake)
if(TARGET Eigen3::Eigen)
    message(STATUS "Eigen3 found successfully!")
else()
    message(FATAL_ERROR "Eigen3 not found!")
endif()

# Include directories
include_directories(/home/sigurdvb/anaconda3/envs/thrust_alloc_cpp_env/include/eigen3)

#find_package(Boost REQUIRED)
#include_directories(${Boost_INCLUDE_DIRS})


# Source files
set(SOURCES
    src/main.cpp
    src/ran.cpp
    src/ALOSpsi.cpp
    src/hermite_spline.cpp
    src/crosstrack_hermite.cpp
    src/control_method.cpp
    src/los_observer.cpp
    src/ref_model.cpp
    src/utilities.cpp
)

add_executable(Optimal-Constraint-Thruster-Allocation ${SOURCES})

