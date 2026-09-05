cmake_minimum_required(VERSION 3.10)
project(ArrowWrap)

# ---------------------------------------------------------------------
# Required inputs (must be provided by the caller)
# ---------------------------------------------------------------------
if (NOT DEFINED ARROW_INSTALL_DIR)
    message(FATAL_ERROR "ARROW_INSTALL_DIR not provided. Pass -DARROW_INSTALL_DIR=...")
endif()

if (NOT DEFINED JLCXX_INSTALL_DIR)
    message(FATAL_ERROR "JLCXX_INSTALL_DIR not provided. Pass -DJLCXX_INSTALL_DIR=...")
endif()

# ---------------------------------------------------------------------
# Find CxxWrap (JlCxx)
# ---------------------------------------------------------------------
set(CMAKE_PREFIX_PATH "${JLCXX_INSTALL_DIR}" ${CMAKE_PREFIX_PATH})
find_package(JlCxx REQUIRED)

# ---------------------------------------------------------------------
# Build shared library
# ---------------------------------------------------------------------
add_library(arrow_wrap SHARED arrow_wrap.cpp)

# ---------------------------------------------------------------------
# Find Arrow and Parquet libraries
# ---------------------------------------------------------------------
find_library(ARROW_LIB arrow
    PATHS "${ARROW_INSTALL_DIR}"
    PATH_SUFFIXES lib lib64
    NO_DEFAULT_PATH
)

find_library(PARQUET_LIB parquet
    PATHS "${ARROW_INSTALL_DIR}"
    PATH_SUFFIXES lib lib64
    NO_DEFAULT_PATH
)

if (NOT ARROW_LIB OR NOT PARQUET_LIB)
    message(FATAL_ERROR
        "Could not find libarrow or libparquet in ${ARROW_INSTALL_DIR}"
    )
endif()

# ---------------------------------------------------------------------
# Include directories
# ---------------------------------------------------------------------
target_include_directories(arrow_wrap PRIVATE
    "${ARROW_INSTALL_DIR}/include"
    "${JLCXX_INSTALL_DIR}/include"
)

# ---------------------------------------------------------------------
# Link libraries
# ---------------------------------------------------------------------
target_link_libraries(arrow_wrap PRIVATE
    JlCxx::cxxwrap_julia
    ${ARROW_LIB}
    ${PARQUET_LIB}
)

# ---------------------------------------------------------------------
# Runtime path (RPATH) for shared libraries
# ---------------------------------------------------------------------
set_target_properties(arrow_wrap PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${ARROW_INSTALL_DIR}/lib;${ARROW_INSTALL_DIR}/lib64"
)

# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------
message(STATUS "ArrowWrap configuration:")
message(STATUS "  ARROW_INSTALL_DIR = ${ARROW_INSTALL_DIR}")
message(STATUS "  JLCXX_INSTALL_DIR = ${JLCXX_INSTALL_DIR}")
message(STATUS "  ARROW_LIB        = ${ARROW_LIB}")
message(STATUS "  PARQUET_LIB      = ${PARQUET_LIB}")
