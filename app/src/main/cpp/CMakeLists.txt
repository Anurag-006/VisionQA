cmake_minimum_required(VERSION 3.22.1)
project(visionqa_jni)

# Path to your llama.cpp source
set(LLAMA_CPP_DIR "C:/Users/anura/llama.cpp")

# Define our JNI library
add_library(visionqa_jni SHARED llama_jni.cpp)

# Updated include directories to find clip.h and llama headers
target_include_directories(visionqa_jni PRIVATE
        ${LLAMA_CPP_DIR}/include
        ${LLAMA_CPP_DIR}/ggml/include
        ${LLAMA_CPP_DIR}/tools/mtmd  # Refactored location for multimodal headers
)

# Find the Android logging library
find_library(log-lib log)

# Link against the new compiled .so files in your jniLibs folder
target_link_libraries(visionqa_jni
        ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libggml-base.so
        ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libggml-cpu.so
        ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libggml-vulkan.so
        ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libggml.so
        ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libllama.so
        ${CMAKE_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libmtmd.so
        ${log-lib}
        android
)