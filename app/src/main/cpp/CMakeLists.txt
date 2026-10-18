cmake_minimum_required(VERSION 3.22.1)

project(llama_jni CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# 薄 JNI 层，链接官方「预编译」的 llama.cpp
#   llama-b11696-bin-android-arm64-snapdragon（lib/ + include/）
#
# 为什么用预编译而不是拿源码编：
#   官方包已经带了完整的 include/llama.h 和 libllama.so，而这一版 llama.cpp 的
#   C API 足够完整（模型加载 / tokenize / decode / 采样 / KV 管理都有），所以只需
#   写一层薄 JNI，把**策略逻辑全部留在 Kotlin 里**（EOS 救回、净化、流式推送、
#   上下文策略都在 Kotlin 侧，可单测、可复用 NLLB provider 的写法）。
#   从源码编译要拉整个仓库再交叉编译 ggml 全部后端，风险和耗时都大得多。
#
# ⚠️ libllama.so 的 NEEDED 里**写死**了 libggml-rpc.so / libggml-opencl.so /
#    libggml-hexagon.so，少任何一个都会 dlopen 失败：
#        dlopen failed: library "libggml-opencl.so" not found
#    所以这三个必须一起进 APK，即使当前只用 CPU。
#    （libggml-htp-v79.so 是 libggml-hexagon.so 运行时按需 dlopen 的骨架，
#      不是硬依赖，只在选 NPU 时才需要。）
#
# ⚠️ 全部使用静态 STL（c++_static）：预编译的 libllama.so 自己也是静态链的
#    （它的 NEEDED 里没有 libc++_shared.so），用 c++_static 可以少塞一个 so。
#    两边传给对方的只有 C 类型和不透明指针，不存在 C++ 对象跨界，所以安全。
# ---------------------------------------------------------------------------

set(LLAMA_LIBS ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI})

add_library(llama SHARED IMPORTED)
set_target_properties(llama PROPERTIES
        IMPORTED_LOCATION ${LLAMA_LIBS}/libllama.so)

add_library(ggml SHARED IMPORTED)
set_target_properties(ggml PROPERTIES
        IMPORTED_LOCATION ${LLAMA_LIBS}/libggml.so)

# ⚠️ 后端注册表那几个函数**分散在两个库里**，两个都得链：
#      libggml.so       ggml_backend_dev_by_name / _dev_count / _dev_get / load_all
#                       （注册表本体：设备枚举、动态加载 libggml-*.so）
#      libggml-base.so  ggml_backend_dev_name 等基础访问器
#    只链其一都会在链接期报 undefined symbol，别被名字骗了 —— 名字看着像一家的。
add_library(ggml-base SHARED IMPORTED)
set_target_properties(ggml-base PROPERTIES
        IMPORTED_LOCATION ${LLAMA_LIBS}/libggml-base.so)

add_library(llama_jni SHARED llama_jni.cpp)

target_include_directories(llama_jni PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(llama_jni
        llama
        ggml
        ggml-base
        log)

# Android 15+ 要求 16KB 页对齐。AGP 一般会自动加，但这里写死不依赖 AGP 版本。
target_link_options(llama_jni PRIVATE "-Wl,-z,max-page-size=16384")
