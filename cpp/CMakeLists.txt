cmake_minimum_required(VERSION 3.10)
project(NanoDeepSeek)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# search for openmp library
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Source files
set(SOURCES
    src/main.cpp
)

# Include directories
include_directories(src)

# Create executable
add_executable(nanodeepseek ${SOURCES})

target_link_libraries(nanodeepseek PRIVATE OpenMP::OpenMP_CXX)

# Compiler options
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -pedantic)
endif()

