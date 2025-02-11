cmake_minimum_required(version 3.10)
project(Optimal-Constraint-Thruster-Allocation)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
Include_directories(${~}/include)

# Find Eigen
find_package(Eigen3 REQUIRED)

#Source files
set{SOURCES
    src/main.cpp
    src/ran.cpp
    src/guidance.cpp
    src/ALOSpsi.cpp
    src/hermite_spline.cpp
    src/crosstrack_hermite.cpp
    src/control_method_selector.cpp
    src/integrators.cpp
    src/utilities.cpp
}

add_executable(Optimal-Constraint-Thruster-Allocation ${SOURCES})