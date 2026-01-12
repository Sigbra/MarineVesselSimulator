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

    # Make CMake search the Conda env first and hint FindGSL
    set(CMAKE_PREFIX_PATH "${CONDA_PREFIX};${CMAKE_PREFIX_PATH}")
    set(GSL_ROOT_DIR "${CONDA_PREFIX}")
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
message(STATUS "GSL include dirs: ${GSL_INCLUDE_DIRS}")
message(STATUS "GSL libraries: ${GSL_LIBRARIES}")

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

    src/Observers/EKF12.cpp
    src/Observers/EKF13.cpp
    src/Observers/EKF15.cpp
    src/Observers/EKF18.cpp
    src/Observers/nn_ekf_v11.cpp
    src/Observers/nn_ekf_v12.cpp
    src/Observers/quatObserver.cpp
    src/Observers/observer_selector.cpp

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

# -------- LibTorch (minimal additions) --------
# Prefer a user-specified Torch_DIR, else try common locations.
if(NOT Torch_DIR)
    # 1) your previous path (kept)
    if(EXISTS "${CMAKE_SOURCE_DIR}/external/libtorch/share/cmake/Torch/TorchConfig.cmake")
        set(Torch_DIR "${CMAKE_SOURCE_DIR}/external/libtorch/share/cmake/Torch")
    # 2) common vendor path
    elseif(EXISTS "${CMAKE_SOURCE_DIR}/third_party/libtorch/share/cmake/Torch/TorchConfig.cmake")
        set(Torch_DIR "${CMAKE_SOURCE_DIR}/third_party/libtorch/share/cmake/Torch")
    # 3) your home install from earlier guidance
    elseif(EXISTS "$ENV{HOME}/libs/libtorch-cu128-2.9.0/share/cmake/Torch/TorchConfig.cmake")
        set(Torch_DIR "$ENV{HOME}/libs/libtorch-cu128-2.9.0/share/cmake/Torch")
    endif()
endif()

find_package(Torch REQUIRED)
message(STATUS "LibTorch found. TORCH_LIBRARIES = ${TORCH_LIBRARIES}")
target_link_libraries(${PROJECT_NAME} PRIVATE "${TORCH_LIBRARIES}")

# Add LibTorch's lib dir to RPATH so runtime finds its .so's without env vars
# Torch sets TORCH_INSTALL_PREFIX; use it if available, else derive from Torch_DIR.
if(DEFINED TORCH_INSTALL_PREFIX)
    set(_TORCH_LIBDIR "${TORCH_INSTALL_PREFIX}/lib")
elseif(Torch_DIR)
    get_filename_component(_TORCH_ROOT "${Torch_DIR}" DIRECTORY)     # .../share/cmake
    get_filename_component(_TORCH_ROOT "${_TORCH_ROOT}" DIRECTORY)   # .../share
    get_filename_component(_TORCH_ROOT "${_TORCH_ROOT}" DIRECTORY)   # .../
    set(_TORCH_LIBDIR "${_TORCH_ROOT}/lib")
endif()

if(_TORCH_LIBDIR)
    set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY BUILD_RPATH "${_TORCH_LIBDIR}")
    set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY INSTALL_RPATH "${_TORCH_LIBDIR}")
endif()

# Provide project root as a compile definition for absolute asset paths
target_compile_definitions(${PROJECT_NAME} PRIVATE
    MVS_PROJECT_ROOT="${CMAKE_SOURCE_DIR}"
)

# GCC<9 needs stdc++fs for <filesystem>
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(${PROJECT_NAME} PRIVATE stdc++fs)
endif()