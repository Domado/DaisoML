cmake_minimum_required(VERSION 3.10)
project(DaisoML CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Explicitly list all source files to avoid picking up unwanted files
set(SOURCES
    main.cpp
    utils.cpp
    tensor.cpp
    tokenizer.cpp
    sampler.cpp
    model.cpp
    layers/embedding.cpp
    layers/rmsnorm.cpp
    layers/attention.cpp
    layers/feed_forward.cpp
)

# Add the main executable
add_executable(daiso_run ${SOURCES})

target_include_directories(daiso_run PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Add the utility to create a dummy model file
add_executable(create_dummy_model
    create_dummy_model.cpp
    tensor.cpp
    utils.cpp
)
target_include_directories(create_dummy_model PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)


# Optional: Add compiler flags for optimization and debugging
# target_compile_options(daiso_run PRIVATE -O3 -DNDEBUG)
# target_compile_options(daiso_run PRIVATE -g)

message(STATUS "DaisoML project configured. Build with 'make'.")
