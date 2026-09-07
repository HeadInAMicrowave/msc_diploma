cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(toxicity LANGUAGES CXX)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(M1_DIR   ${CMAKE_CURRENT_SOURCE_DIR}/first_model)    # first_model.cpp, textcnn.hpp, textcnn_config.hpp
set(M2_DIR   ${CMAKE_CURRENT_SOURCE_DIR}/second_model)   # wordpiece.hpp, span_markup.hpp, teacher_client.cpp, parity_test.cpp, markup_test.cpp
set(ORCH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/orchestrator)   # orchestrator.cpp, detox_loop.hpp, interfaces.hpp, cleanup.hpp, adapters, loop_test.cpp

# nlohmann/json, header-only,
include(FetchContent)
FetchContent_Declare(nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(nlohmann_json)

# Helper: one LibTorch executable per model, with the Windows DLL copy applied.
function(add_torch_model target src)
    add_executable(${target} ${src})
    target_link_libraries(${target} "${TORCH_LIBRARIES}")
    set_property(TARGET ${target} PROPERTY CXX_STANDARD 20)
    set_property(TARGET ${target} PROPERTY CXX_STANDARD_REQUIRED ON)

    if (MSVC)
        file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
        add_custom_command(TARGET ${target} POST_BUILD
                           COMMAND ${CMAKE_COMMAND} -E copy_if_different
                           ${TORCH_DLLS} $<TARGET_FILE_DIR:${target}>)
    endif()
endfunction()

# Helper: a plain C++20 executable with no LibTorch
function(add_plain_tool target src)
    add_executable(${target} ${src})
    set_property(TARGET ${target} PROPERTY CXX_STANDARD 20)
    set_property(TARGET ${target} PROPERTY CXX_STANDARD_REQUIRED ON)
endfunction()

# Model 1: the TextCNN trainer.
add_torch_model(first_model ${M1_DIR}/first_model.cpp)
target_include_directories(first_model PRIVATE ${M1_DIR})
target_compile_definitions(first_model PRIVATE
    DATASET_DIR="${M1_DIR}/dataset")

# Model 2 tools (build-time).
add_plain_tool(parity_test ${M2_DIR}/parity_test.cpp)
target_include_directories(parity_test PRIVATE ${M2_DIR})
target_link_libraries(parity_test PRIVATE nlohmann_json::nlohmann_json)

add_plain_tool(markup_test ${M2_DIR}/markup_test.cpp)
target_include_directories(markup_test PRIVATE ${M2_DIR})

# basic_dump prints the runtime tokenizer's output per line; augment_obfuscation.py
# uses it (--basic-dump) to prove generated rows tokenize identically at inference.
add_plain_tool(basic_dump ${M2_DIR}/basic_dump.cpp)
target_include_directories(basic_dump PRIVATE ${M2_DIR})

find_package(CURL)
if (CURL_FOUND)
    add_plain_tool(teacher_client ${M2_DIR}/teacher_client.cpp)
    target_include_directories(teacher_client PRIVATE ${M2_DIR})
    target_link_libraries(teacher_client PRIVATE nlohmann_json::nlohmann_json CURL::libcurl)
else()
    message(STATUS "libcurl not found: teacher_client target skipped")
endif()

# Runtime: the orchestrator (both models + the greedy loop) and its offline test.
# loop_test needs neither LibTorch nor the models
add_plain_tool(loop_test ${ORCH_DIR}/loop_test.cpp)
target_include_directories(loop_test PRIVATE ${ORCH_DIR})

add_torch_model(orchestrator ${ORCH_DIR}/orchestrator.cpp)
target_include_directories(orchestrator PRIVATE ${ORCH_DIR} ${M1_DIR} ${M2_DIR})
target_link_libraries(orchestrator nlohmann_json::nlohmann_json)
