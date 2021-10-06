cmake_minimum_required(VERSION 3.11)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

project(deformxfer)

set(CMAKE_CXX_STANDARD 11)

include(FetchContent)

# eigen 3.4
FetchContent_Declare(Eigen3
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
  GIT_SHALLOW TRUE
)
set(EIGEN_BUILD_DOC OFF)
set(BUILD_TESTING OFF)
set(EIGEN_BUILD_PKGCONFIG OFF)
FetchContent_MakeAvailable(Eigen3)

# libigl
FetchContent_Declare(libigl
  GIT_REPOSITORY https://github.com/libigl/libigl.git
  GIT_TAG v2.3.0
)
set(LIBIGL_USE_STATIC_LIBRARY OFF)
set(LIBIGL_WITH_OPENGL ON)
set(LIBIGL_WITH_OPENGL_GLFW ON)
FetchContent_GetProperties(libigl)
if(NOT libigl_POPULATED)
    FetchContent_Populate(libigl)
endif()
list(PREPEND CMAKE_MODULE_PATH "${libigl_SOURCE_DIR}/cmake")
include(${libigl_SOURCE_DIR}/cmake/libigl.cmake)

# BlendshapeMaker, for demo's example data
FetchContent_Declare(BlendshapeMaker
  GIT_REPOSITORY
    https://github.com/diegothomas/Avatar-generation-3DRW2019-.git
)
FetchContent_MakeAvailable(BlendshapeMaker)
FetchContent_GetProperties(BlendshapeMaker
  SOURCE_DIR BM_DIR
)
add_definitions(-DBLENDSHAPE_DIR=\"${BM_DIR}/BlendshapeMaker/data/blendshapes/\")
add_definitions(-DINPUTSHAPE_DIR=\"${BM_DIR}/BlendshapeMaker/Inputs/\")
add_definitions(-DLANDMARKS_DIR=\"${BM_DIR}/BlendshapeMaker/data/landmarks/\")

# Add demo executable
include_directories(dx demo)
add_executable(dx_demo demo/main.cpp)
target_link_libraries(dx_demo
  Eigen3::Eigen
  igl::opengl
  igl::opengl_glfw
)
