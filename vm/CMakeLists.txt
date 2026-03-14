cmake_minimum_required(VERSION 3.20)
project(adam-vm C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)  # Allow GNU extensions (computed goto)

# Compiler flags
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    # -Wno-pedantic: computed goto (&&label) is a GCC/Clang extension we use intentionally
    add_compile_options(-Wall -Wextra -Wno-unused-parameter)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O2)
    else()
        add_compile_options(-g -O0)
        add_compile_definitions(DEBUG_TRACE_EXECUTION DEBUG_STRESS_GC)
    endif()

    # Enable computed goto (GCC/Clang extension)
    add_compile_definitions(ADAM_COMPUTED_GOTO)
elseif(MSVC)
    add_compile_options(/W4)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(/O2)
    endif()
endif()

# VM library (shared between CLI and tests)
add_library(adam-vm-lib STATIC
    src/value.c
    src/chunk.c
    src/object.c
    src/table.c
    src/gc.c
    src/vm.c
    src/native.c
    src/debug.c
)
target_include_directories(adam-vm-lib PUBLIC include)

# CLI executable
add_executable(adam-vm src/main.c)
target_link_libraries(adam-vm PRIVATE adam-vm-lib)
target_link_libraries(adam-vm PRIVATE m)

# Tests
enable_testing()
add_executable(test-vm src/test_vm.c)
target_link_libraries(test-vm PRIVATE adam-vm-lib m)
add_test(NAME vm-tests COMMAND test-vm)
