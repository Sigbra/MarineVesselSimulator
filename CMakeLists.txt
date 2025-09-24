cmake_minimum_required(VERSION 3.10)
project(MarineVesselSimulator)

# -----------------------------------
# C++ standard
# -----------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------------
# Use Conda environment prefix if available
# -----------------------------------
if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_PREFIX $ENV{CONDA_PREFIX})
    message(STATUS "Using conda environment at: ${CONDA_PREFIX}")
endif()

# -----------------------------------
# Include project headers
# -----------------------------------
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/external/matplotlib-cpp)

# -----------------------------------
# External Dependencies
# -----------------------------------

# 1. Eigen3 (header-only, use target)
find_package(Eigen3 REQUIRED NO_DEFAULT_PATH 
    PATHS ${CONDA_PREFIX}/share/eigen3/cmake
)
if(TARGET Eigen3::Eigen)
    message(STATUS "Eigen3 found successfully!")
else()
    message(FATAL_ERROR "Eigen3 not found!")
endif()

# 2. FFTW (link directory)
link_directories(${CONDA_PREFIX}/lib)

# 3. Gnuplot
find_program(GNUPLOT_EXECUTABLE gnuplot PATHS ${CONDA_PREFIX}/bin)
if(GNUPLOT_EXECUTABLE)
    message(STATUS "Found Gnuplot at ${GNUPLOT_EXECUTABLE}")
else()
    message(FATAL_ERROR "Gnuplot not found!")
endif()

# 4. Python + NumPy
find_package(Python3 COMPONENTS Development NumPy REQUIRED)
message(STATUS "Python3 include dirs: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3 NumPy include dirs: ${Python3_NumPy_INCLUDE_DIRS}")

# 5. yaml-cpp
find_package(yaml-cpp REQUIRED)

# 6. GSL
find_package(GSL REQUIRED)
message(STATUS "GSL found: ${GSL_INCLUDE_DIRS}")

# 7. CasADi
find_path(CASADI_INCLUDE_DIR casadi/casadi.hpp HINTS ${CONDA_PREFIX}/include)
find_library(CASADI_LIBRARY casadi HINTS ${CONDA_PREFIX}/lib)
if(NOT CASADI_INCLUDE_DIR OR NOT CASADI_LIBRARY)
    message(FATAL_ERROR "CasADi include/library not found!")
else()
    message(STATUS "CasADi include: ${CASADI_INCLUDE_DIR}")
    message(STATUS "CasADi library: ${CASADI_LIBRARY}")
endif()

# -----------------------------------
# Source Files
# -----------------------------------
set(SOURCES
    src/main.cpp

    src/Control/control_alloc_selector.cpp
    src/Control/pseudo_inverse_allocation.cpp
    src/Control/MPC_control_alloc.cpp
    src/Control/non_lin_constrained_control_alloc.cpp
    src/Control/PID_heading_motion_control.cpp
    src/Control/PID_MIMO_motion_control.cpp
    src/Control/MPC_control_system.cpp

    src/Guidance/LOS.cpp
    src/Guidance/ALOS.cpp
    src/Guidance/LOS_observer.cpp
    src/Guidance/dynamic_positioning.cpp
    src/Guidance/guidance_selector.cpp

    src/Models/ran.cpp
    src/Models/ref_model.cpp
    src/Models/model_utilities.cpp

    src/Observers/kalmanObserver.cpp

    src/Planning/fermat_spiral_planning.cpp
    src/Planning/straight_line_planning.cpp
    src/Planning/plan_selector.cpp

    src/Sensors/GNSS.cpp
    src/Sensors/IMU.cpp

    src/Utilities/calculations.cpp
    src/Utilities/plotting.cpp
)

# -----------------------------------
# Create Executable
# -----------------------------------
add_executable(${PROJECT_NAME} ${SOURCES})

# -----------------------------------
# Link Libraries and Include Directories
# -----------------------------------

# Eigen (header-only target)
target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)

# FFTW
target_link_libraries(${PROJECT_NAME} PRIVATE fftw3)

# Python + NumPy
target_include_directories(${PROJECT_NAME} PRIVATE 
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME} PRIVATE ${Python3_LIBRARIES})

# CasADi
target_include_directories(${PROJECT_NAME} PRIVATE ${CASADI_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE ${CASADI_LIBRARY})
set_target_properties(${PROJECT_NAME} PROPERTIES
    BUILD_RPATH "${CONDA_PREFIX}/lib"
)

# yaml-cpp
target_link_libraries(${PROJECT_NAME} PRIVATE yaml-cpp)

# GSL
target_include_directories(${PROJECT_NAME} PRIVATE ${GSL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${GSL_LIBRARIES} m)

# matplotlib-cpp
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/external/matplotlib-cpp)
