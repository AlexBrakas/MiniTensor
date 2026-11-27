# Set the minimum required version of CMake and project name
cmake_minimum_required(VERSION 3.12)
project(MiniTensor)

# Set the C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable Hot Reload for MSVC compilers if supported
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

find_package(OpenMP REQUIRED)

# AVX-512 kernels for local builds; leave OFF in CI
option(MINITENSOR_AVX512 "Build with AVX-512 kernels" OFF)

# --- Create the Library ---
add_library(MiniTensor STATIC 
    src/Tensor.cpp
#    src/Layer.cpp
#    src/Loss.cpp
#    src/Optimizer.cpp
)

# Make sure the library's public headers are accessible
target_include_directories(MiniTensor PUBLIC include)
target_link_libraries(MiniTensor PUBLIC OpenMP::OpenMP_CXX)

if(MINITENSOR_AVX512)
    if(MSVC)
        target_compile_options(MiniTensor PRIVATE /arch:AVX512)
    else()
        target_compile_options(MiniTensor PRIVATE -mavx512f)
    endif()
endif()

# --- Create the Main Application Executable ---
#add_executable(MiniTensorExample app/main.cpp)

# --- Create and Register Tests ---
enable_testing()

# Create an executable for our tensor tests
add_executable(RunTensorTests
    tests/test_tensor.cpp
    src_baseline/TensorBaseline.cpp
)

# Link the test executable against your MiniTensor library
target_link_libraries(RunTensorTests PRIVATE MiniTensor)

# Register the executable with CTest
add_test(NAME TensorModuleTest COMMAND RunTensorTests)

# --- INSTALL STEP ---
# This creates the final, distributable package.
install(TARGETS MiniTensor 
    # Copy the library to the 'lib' folder
    ARCHIVE DESTINATION lib 
)

install(DIRECTORY include/
    # Copy the public headers to the 'include' folder
    DESTINATION include
)