cmake_minimum_required(VERSION 3.10)
project(MarineVesselSimulator)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CMAKE_PREFIX_PATH to conda environment directory if available
if(DEFINED ENV{CONDA_PREFIX})
    set(CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
    message(STATUS "Using conda environment at: $ENV{CONDA_PREFIX}")
endif()

# -----------------------------------
# Directories
# -----------------------------------
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(/home/sigurdvb/anaconda3/envs/thrust_alloc_cpp_env/include/eigen3)
include_directories(${CMAKE_SOURCE_DIR}/external/matplotlib-cpp)

# -----------------------------------
# External Dependencies
# -----------------------------------

# 1. Find Eigen3
find_package(Eigen3 REQUIRED NO_DEFAULT_PATH 
    PATHS /home/sigurdvb/anaconda3/envs/thrust_alloc_cpp_env/share/eigen3/cmake)
if(TARGET Eigen3::Eigen)
    message(STATUS "Eigen3 found successfully!")
else()
    message(FATAL_ERROR "Eigen3 not found!")
endif()

# 2. FFTW (adjust library name if necessary)
link_directories(/home/sigurdvb/anaconda3/envs/thrust_alloc_cpp_env/lib)

# 3. Gnuplot
set(GNUPlot_EXECUTABLE "/home/sigurdvb/anaconda3/envs/thrust_alloc_cpp_env/bin/gnuplot")
include(FindGnuplot)
if(GNUPLOT_EXECUTABLE)
    message(STATUS "Found Gnuplot at ${GNUPLOT_EXECUTABLE}")
else()
    message(FATAL_ERROR "Gnuplot not found!")
endif()

# 4. Python and NumPy for matplotlib-cpp
find_package(Python3 COMPONENTS Development NumPy REQUIRED)
message(STATUS "Python3 include dirs: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3 NumPy include dirs: ${Python3_NumPy_INCLUDE_DIRS}")

# 5. yaml-cpp setup
find_package(yaml-cpp REQUIRED)

# 6. GSL Setup
find_package(GSL REQUIRED)
message(STATUS "GSL found: ${GSL_INCLUDE_DIRS}")

# 7. CasADi Setup (Manually Finding CasADi)
find_path(CASADI_INCLUDE_DIR casadi/casadi.hpp
    HINTS /home/sigurdvb/anaconda3/envs/thrust_alloc_cpp_env/include
)
if(NOT CASADI_INCLUDE_DIR)
    message(FATAL_ERROR "CasADi include directory not found!")
else()
    message(STATUS "CasADi include directory: ${CASADI_INCLUDE_DIR}")
endif()

find_library(CASADI_LIBRARY casadi
    HINTS /home/sigurdvb/anaconda3/envs/thrust_alloc_cpp_env/lib
)
if(NOT CASADI_LIBRARY)
    message(FATAL_ERROR "CasADi library not found!")
else()
    message(STATUS "CasADi library: ${CASADI_LIBRARY}")
endif()

# Add CasADi include directory and library path
include_directories(${CASADI_INCLUDE_DIR})
link_directories(/home/sigurdvb/anaconda3/envs/thrust_alloc_cpp_env/lib)

# -----------------------------------
# Source Files
# -----------------------------------
set(SOURCES
    src/main.cpp

    src/Control/control_alloc_selector.cpp
    src/Control/MPC_control_alloc.cpp
    src/Control/non_lin_constrained_control_alloc.cpp
    src/Control/PID_heading_motion_control.cpp
    src/Control/PID_MIMO_motion_control.cpp
    src/Control/MPC_motion_control.cpp

    src/Guidance/LOS.cpp
    src/Guidance/ALOS.cpp
    src/Guidance/MPC_guidance.cpp
    src/Guidance/LOS_observer.cpp
    src/Guidance/dynamic_positioning.cpp
    src/Guidance/guidance_selector.cpp

    src/Models/ran.cpp
    src/Models/ref_model.cpp
    src/Models/model_utilities.cpp

    src/Planning/fermat_spiral_planning.cpp
    src/Planning/straight_line_planning.cpp
    src/Planning/plan_selector.cpp

    src/Utilities/calculations.cpp
    src/Utilities/plotting.cpp
)

# -----------------------------------
# Create Executable
# -----------------------------------
add_executable(${PROJECT_NAME} ${SOURCES})

# -----------------------------------
# Link Libraries
# -----------------------------------
# FFTW
target_link_libraries(${PROJECT_NAME} PUBLIC fftw3)

# Python and NumPy (for matplotlib-cpp)
target_include_directories(${PROJECT_NAME} PRIVATE 
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME} PRIVATE ${Python3_LIBRARIES})

# CasADi
target_link_libraries(${PROJECT_NAME} PRIVATE ${CASADI_LIBRARY})

# Ensure CasADi is in the runtime path
set_target_properties(${PROJECT_NAME} PROPERTIES
    BUILD_RPATH "/home/sigurdvb/anaconda3/envs/thrust_alloc_cpp_env/lib"
)

# yaml-cpp
target_link_libraries(${PROJECT_NAME} PRIVATE ${YAML_CPP_LIBRARIES})

# GSL (add its include directory and link the libraries along with the math library)
target_include_directories(${PROJECT_NAME} PRIVATE ${GSL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${GSL_LIBRARIES} m)

# Target the external matplotlib-cpp

target_include_directories(MarineVesselSimulator PRIVATE
    ${CMAKE_SOURCE_DIR}/external/matplotlib-cpp
)