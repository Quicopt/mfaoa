# libmfaoa, the mean-field AOA solver in C, built against the system's BLAS.
#
#   cmake -S . -B build && cmake --build build && ctest --test-dir build
#   cmake --install build --prefix <dir>
#
# CMake's FindBLAS picks the BLAS: Accelerate on macOS, otherwise the first
# one it finds. The library needs its C interface, CBLAS, declared in
# cblas.h (OpenBLAS, BLIS, FlexiBLAS, the reference CBLAS) or in Accelerate's
# own header. Choose a BLAS with -DBLA_VENDOR=OpenBLAS (or FLAME for BLIS,
# Apple, ...), and point CMake to an installation outside the default paths
# with -DCMAKE_PREFIX_PATH.
#
# -DMFAOA_BLAS=OFF builds the library without a BLAS. Callers must then pass
# a dgemv in every problem with dense couplings (see mfaoa_problem.dgemv);
# the Python package does that with SciPy's.

cmake_minimum_required(VERSION 3.21)

# The version is written once, in include/mfaoa.h.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/include/mfaoa.h" _mfaoa_version_lines
     REGEX "^#define MFAOA_VERSION_(MAJOR|MINOR|PATCH) ")
foreach(_part MAJOR MINOR PATCH)
  string(REGEX MATCH "MFAOA_VERSION_${_part} ([0-9]+)" _match "${_mfaoa_version_lines}")
  set(_mfaoa_${_part} "${CMAKE_MATCH_1}")
endforeach()
project(mfaoa VERSION ${_mfaoa_MAJOR}.${_mfaoa_MINOR}.${_mfaoa_PATCH} LANGUAGES C)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
option(BUILD_SHARED_LIBS "Build a shared library rather than a static one" ON)
option(MFAOA_BUILD_TESTS "Build the tests and the example" ${PROJECT_IS_TOP_LEVEL})
option(MFAOA_BLAS "Link a BLAS for dense couplings" ON)

if(MFAOA_BLAS)
  find_package(BLAS REQUIRED)
endif()

add_library(mfaoa src/mfaoa.c)
add_library(mfaoa::mfaoa ALIAS mfaoa)
target_include_directories(mfaoa PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(mfaoa PRIVATE c_std_11)
target_compile_definitions(mfaoa PRIVATE MFAOA_BUILD)
if(NOT BUILD_SHARED_LIBS)
  target_compile_definitions(mfaoa PUBLIC MFAOA_STATIC)
endif()
# While the major version is 0, every minor version may change the
# interface, so it is part of the shared library's name (libmfaoa.so.0.1).
if(PROJECT_VERSION_MAJOR EQUAL 0)
  set(_mfaoa_soversion ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR})
else()
  set(_mfaoa_soversion ${PROJECT_VERSION_MAJOR})
endif()
set_target_properties(mfaoa PROPERTIES C_VISIBILITY_PRESET hidden)
# The Python package loads the library by its plain name, and wheels cannot
# hold the symbolic links between the versioned names.
if(NOT SKBUILD)
  set_target_properties(mfaoa PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${_mfaoa_soversion})
endif()
# No fused multiply-adds beyond the ones src/mfaoa.c asks for, so that GCC
# and Clang, whose defaults differ, compute the same results.
if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU" AND NOT MSVC)
  target_compile_options(mfaoa PRIVATE -ffp-contract=off)
endif()

if(UNIX AND NOT APPLE)
  target_link_libraries(mfaoa PRIVATE m)
endif()
# FindBLAS finds the library but not its C header. Accelerate has its own;
# every other BLAS ships cblas.h, usually in the include folder next to it,
# which is searched first. Frameworks are never searched: on macOS the one
# cblas.h there is Accelerate's.
if(NOT MFAOA_BLAS)
  target_compile_definitions(mfaoa PRIVATE MFAOA_NO_BLAS)
elseif(BLAS_LIBRARIES MATCHES "Accelerate")
  target_link_libraries(mfaoa PRIVATE BLAS::BLAS)
  target_compile_definitions(mfaoa PRIVATE MFAOA_ACCELERATE)
else()
  target_link_libraries(mfaoa PRIVATE BLAS::BLAS)
  list(GET BLAS_LIBRARIES 0 _mfaoa_blas_library)
  get_filename_component(_mfaoa_blas_dir "${_mfaoa_blas_library}" DIRECTORY)
  set(_mfaoa_find_framework "${CMAKE_FIND_FRAMEWORK}")
  set(CMAKE_FIND_FRAMEWORK NEVER)
  find_path(CBLAS_INCLUDE_DIR cblas.h
    PATHS "${_mfaoa_blas_dir}/../include"
    PATH_SUFFIXES openblas blis mkl
    NO_DEFAULT_PATH)
  find_path(CBLAS_INCLUDE_DIR cblas.h PATH_SUFFIXES openblas blis mkl)
  set(CMAKE_FIND_FRAMEWORK "${_mfaoa_find_framework}")
  if(NOT CBLAS_INCLUDE_DIR)
    message(FATAL_ERROR "cblas.h not found for ${_mfaoa_blas_library}; set CBLAS_INCLUDE_DIR")
  endif()
  target_include_directories(mfaoa PRIVATE "${CBLAS_INCLUDE_DIR}")
endif()
if(MFAOA_BLAS)
  message(STATUS "mfaoa links BLAS: ${BLAS_LIBRARIES}")
else()
  message(STATUS "mfaoa links no BLAS: dense couplings need the caller's dgemv")
endif()

# Built for the Python package (by scikit-build-core, see pyproject.toml),
# the library goes into the package folder and nothing else is installed.
if(SKBUILD)
  install(TARGETS mfaoa LIBRARY DESTINATION mfaoa RUNTIME DESTINATION mfaoa)
  return()
endif()

# Installation: the library, the header, and the files with which other
# builds find them: find_package(mfaoa) for CMake, mfaoa.pc for pkg-config.
set(MFAOA_CMAKE_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/mfaoa)
install(TARGETS mfaoa EXPORT mfaoaTargets)
install(FILES include/mfaoa.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT mfaoaTargets NAMESPACE mfaoa:: DESTINATION ${MFAOA_CMAKE_DIR})

# Programs that link a static library with a BLAS must link the BLAS too.
if(BUILD_SHARED_LIBS OR NOT MFAOA_BLAS)
  set(MFAOA_NEEDS_BLAS OFF)
else()
  set(MFAOA_NEEDS_BLAS ON)
endif()
configure_package_config_file(cmake/mfaoaConfig.cmake.in
  ${PROJECT_BINARY_DIR}/mfaoaConfig.cmake INSTALL_DESTINATION ${MFAOA_CMAKE_DIR})
write_basic_package_version_file(${PROJECT_BINARY_DIR}/mfaoaConfigVersion.cmake
  COMPATIBILITY SameMinorVersion)
install(FILES ${PROJECT_BINARY_DIR}/mfaoaConfig.cmake ${PROJECT_BINARY_DIR}/mfaoaConfigVersion.cmake
  DESTINATION ${MFAOA_CMAKE_DIR})

# mfaoa.pc locates the installation relative to itself, so that it stays
# right when the install prefix is chosen at install time.
if(IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}" OR IS_ABSOLUTE "${CMAKE_INSTALL_INCLUDEDIR}")
  set(MFAOA_PC_PREFIX "${CMAKE_INSTALL_PREFIX}")
  set(MFAOA_PC_LIBDIR "${CMAKE_INSTALL_FULL_LIBDIR}")
  set(MFAOA_PC_INCLUDEDIR "${CMAKE_INSTALL_FULL_INCLUDEDIR}")
else()
  file(RELATIVE_PATH _mfaoa_up "/root/${CMAKE_INSTALL_LIBDIR}/pkgconfig" "/root")
  string(REGEX REPLACE "/$" "" _mfaoa_up "${_mfaoa_up}")
  set(MFAOA_PC_PREFIX "\${pcfiledir}/${_mfaoa_up}")
  set(MFAOA_PC_LIBDIR "\${prefix}/${CMAKE_INSTALL_LIBDIR}")
  set(MFAOA_PC_INCLUDEDIR "\${prefix}/${CMAKE_INSTALL_INCLUDEDIR}")
endif()
set(MFAOA_PC_CFLAGS "")
if(NOT BUILD_SHARED_LIBS)
  set(MFAOA_PC_CFLAGS " -DMFAOA_STATIC")
endif()
# The BLAS as linker flags, for programs that link the static library.
set(_mfaoa_pc_libs "")
foreach(_lib IN LISTS BLAS_LIBRARIES)
  if(_lib MATCHES "/([^/]+)\\.framework$")
    list(APPEND _mfaoa_pc_libs "-framework ${CMAKE_MATCH_1}")
  elseif(IS_ABSOLUTE "${_lib}")
    get_filename_component(_dir "${_lib}" DIRECTORY)
    get_filename_component(_name "${_lib}" NAME_WE)
    string(REGEX REPLACE "^lib" "" _name "${_name}")
    list(APPEND _mfaoa_pc_libs "-L${_dir} -l${_name}")
  else()
    list(APPEND _mfaoa_pc_libs "${_lib}")
  endif()
endforeach()
if(UNIX AND NOT APPLE)
  list(APPEND _mfaoa_pc_libs "-lm")
endif()
list(JOIN _mfaoa_pc_libs " " MFAOA_PC_LIBS_PRIVATE)
configure_file(cmake/mfaoa.pc.in ${PROJECT_BINARY_DIR}/mfaoa.pc @ONLY)
install(FILES ${PROJECT_BINARY_DIR}/mfaoa.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

if(MFAOA_BUILD_TESTS)
  enable_testing()
  add_executable(mfaoa_check test/check.c)
  target_link_libraries(mfaoa_check PRIVATE mfaoa)
  if(UNIX AND NOT APPLE)
    target_link_libraries(mfaoa_check PRIVATE m)
  endif()
  if(NOT MFAOA_BLAS)
    target_compile_definitions(mfaoa_check PRIVATE CHECK_NO_BLAS)
  endif()
  add_test(NAME reference_runs COMMAND mfaoa_check ${PROJECT_SOURCE_DIR}/test/golden.txt)

  # The example solves a dense problem with the library's own BLAS.
  if(MFAOA_BLAS)
    add_executable(mfaoa_example examples/solve.c)
    target_link_libraries(mfaoa_example PRIVATE mfaoa)
    add_test(NAME example COMMAND mfaoa_example)
    set_tests_properties(example PROPERTIES
      PASS_REGULAR_EXPRESSION "sigma = \\(-1, 1, 1, 1\\), E = -4\\.25")
  endif()
endif()
