if (NOT GGML_KDNN_ROOT)
    message(FATAL_ERROR "GGML_KDNN_ROOT must point to a KDNN installation")
endif()

find_path(KDNN_INCLUDE_DIR kdnn.hpp HINTS "${GGML_KDNN_ROOT}/include" REQUIRED NO_DEFAULT_PATH)
find_library(KDNN_LIBRARY kdnn HINTS "${GGML_KDNN_ROOT}/lib" REQUIRED NO_DEFAULT_PATH)

ggml_add_backend_library(ggml-kdnn ggml-kdnn.cpp)
target_include_directories(ggml-kdnn SYSTEM PRIVATE "${KDNN_INCLUDE_DIR}")
target_link_libraries(ggml-kdnn PRIVATE "${KDNN_LIBRARY}")

message(STATUS "KDNN found: ${KDNN_LIBRARY}")
