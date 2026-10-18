import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    // AGP 9.x 内置 Kotlin 支持，**不能再应用 org.jetbrains.kotlin.android**
    // （应用会直接构建失败：The 'org.jetbrains.kotlin.android' plugin is no longer required for Kotlin support since AGP 9.0）
    id("com.android.application")
    // Compose 编译器仍需独立插件，版本与内置 Kotlin 版本保持一致
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "com.screenast.app"
    // compileSdk 必须 >= 37：androidx.core 1.19.1 / activity-compose 1.13.0 等
    // 声明的 API 需求为 37（AGP 会把“请用 37”升级为构建错误）。
    // targetSdk 保持 36，与真机实际 API 36 对应，避免提前启用 37 的运行时行为变更。
    compileSdk = 37
    buildToolsVersion = "37.0.0"

    // llama.cpp（2B GGUF 端侧翻译）的薄 JNI 层需要 NDK 交叉编译。
    // 版本挑 r28 系：与 llama-b11696 官方预编译包的构建环境（Clang 21）同代，
    // 避免出现「我们编的 JNI 与预编译 .so 用了不同 STL / ABI」这类问题。
    ndkVersion = "28.2.13676358"

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    defaultConfig {
        applicationId = "com.screenast.app"
        minSdk = 29          // AudioPlaybackCapture 起始版本（Android 10）
        targetSdk = 36       // Android 16 / 澎湃 OS 3
        versionCode = 6
        versionName = "0.7.3-beta"
        vectorDrawables { useSupportLibrary = true }

        // ⚠️ 只出 arm64-v8a。三条理由：
        //   1) 真机 ro.product.cpu.abi 与 abilist 里都只有 arm64-v8a；
        //   2) 新引入的 llama.cpp 预编译包只有 arm64 版，别的 ABI 根本补不齐；
        //   3) 原来四个 ABI 全带时 APK 308MB，其中 arm64 只占 66MB —— 砍掉另外
        //      三个能省约 200MB，正好抵消新加的 76MB llama.cpp 库。
        // 以后要跑模拟器就删掉这段（并自行补上对应 ABI 的 llama.cpp 库）。
        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    buildTypes {
        debug {
            applicationIdSuffix = ".debug"
        }
        release {
            isMinifyEnabled = false
        }
    }

    buildFeatures {
        compose = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    lint {
        abortOnError = false
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
        jniLibs {
            // 让 .so 被**解压**到 applicationInfo.nativeLibraryDir，而不是留在 APK 内。
            //
            // 原因：Hexagon NPU 的 skel（libggml-htp-v79.so）要由 FastRPC 按路径加载，
            // 而 ggml-hexagon.cpp:3206 拼出来的 URI 是裸文件名
            // （file:///libggml-htp-v79.so），只能按 ADSP_LIBRARY_PATH 找。
            // 默认（false）时 nativeLibraryDir 这个目录根本不存在，HTP0 会话打不开：
            //     HTP0 failed to open session : uri file:///libggml-htp-v79.so ... error 0x80000406
            // 代价：APK 里的 .so 由「不压缩存储」变成压缩存储，安装后多占约 76MB。
            useLegacyPackaging = true

            // 这里**故意什么都不排除**。
            //
            // 打包不再有 onnxruntime 同名冲突，因为冲突已在 AAR 源头解决：
            // sherpa-onnx 那份的 soname 已被改名为 libonnxruntime_sherpa.so，
            // 见 tools/patch_sherpa_soname.py（打了补丁的 AAR 就地替换了原件，
            // 原始文件备份为 third_party/sherpa-onnx/sherpa-onnx-1.13.8.orig.aar）。
            //
            // ⚠️ 走过的两条弯路，别重蹈：
            // 1) pickFirsts += "**/libonnxruntime.so"
            //    选中的是 sherpa 里较老的那份，官方 JNI 需要新符号，启动即崩：
            //      UnsatisfiedLinkError: dlopen failed: cannot locate symbol "OrtGetApiBase"
            //      referenced by "...!/lib/x86_64/libonnxruntime4j_jni.so"
            // 2) excludes += "**/libonnxruntime.so"
            //    这个是**无差别匹配**，把两份都删了 —— 连官方那份也没进 APK，
            //    于是连 Microsoft 自己的 JNI 都找不到依赖，崩在同一个位置。
            //
            // 两份必须并存且必须都进 APK：符号版本标签互不兼容
            // （sherpa 要 OrtGetApiBase@VERS_1.28.2，官方要 ...@VERS_1.31.0）。
        }
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
    }
}

// ---- 临时诊断开关：摘掉 sherpa-onnx，隔离两份 onnxruntime ----
//
// 起因：真机上 NLLB 的 `OrtSession.run` 稳定 SIGSEGV，崩溃点
// `Java_ai_onnxruntime_OrtSession_run+552` 反汇编出来是
//     ldr x8, [x21, #0x48] ; blr x8
// 即 JNI 接口表指针本身是垃圾。而**同一版本 onnxruntime 1.31.0、同一批模型文件，
// 在 PC 上用 Python 跑得干干净净**（`tools/repro_crash_on_pc.py`：with-past 增量
// 解码 9 步跑通，cache 48 条含 encoder 侧，译文「您好,欢迎回到频道.」完全正确）。
//
// 于是最大嫌疑落在 APK 里并存的两份 onnxruntime：
//   libonnxruntime.so          官方 1.31.0（libonnxruntime4j_jni.so 依赖它）
//   libonnxruntime_sherpa.so   sherpa 的 1.28.2（soname 已改，见 patch_sherpa_soname.py）
// soname 不同才让它们能共存，但 **JNI 导出符号是全局的**：`libonnxruntime4j_jni.so`
// 解析 `OrtGetApiBase` 时可能落到 sherpa 那份，于是 Java 侧按 1.31 的布局持有
// `OrtSession*`、native 侧跑的却是 1.28 的代码 —— 与"JNIEnv 是垃圾"高度吻合。
//
// 用法：`bash tools/toggle_noasr.sh on|off`（内部就是挪走 SenseVoiceAsr.kt +
// 把下面 dependencies 里那行 sherpa AAR 注释掉，跑完再 `off` 还原）。
// 若摘掉 sherpa 后不再崩溃，即坐实是两份 native 库互相踩。
//
// 注意：`exclude` 不能写在 `kotlin { }` 块里（那里只有 compilerOptions，报
// "receiver type mismatch"），也不能用 `kotlin.sourceSets["main"]`（AGP 9 内置 Kotlin
// 没注册名为 "main" 的 KotlinSourceSet），`android.sourceSets["main"].kotlin` 同样
// 不可用（报 receiver type mismatch）。
//
// 所以这个开关改用 tools/toggle_noasr.sh 实现：直接挪走 SenseVoiceAsr.kt、
// 注释掉 sherpa 依赖，跑完自动还原。脚本笨但绝对可靠，不跟构建 DSL 较劲。

dependencies {
    val composeBom = platform("androidx.compose:compose-bom:2026.09.00")
    implementation(composeBom)

    implementation("androidx.core:core-ktx:1.19.1")
    implementation("androidx.activity:activity-compose:1.13.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.9.4")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.9.4")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.material3:material3")

    // 设置持久化。AGP 9 的 R8/校验对缺失依赖不会提前报错，所以漏了这行
    // 只在 compileDebugKotlin 阶段暴露成一片 Unresolved reference 'datastore'。
    implementation("androidx.datastore:datastore-preferences:1.2.0")

    // 端侧翻译（NLLB）的推理引擎。
    // 版本与本机 Python 侧验证用的 onnxruntime 1.31.0 对齐：模型是同一个文件，
    // 算子实现一致才能保证 Kotlin 端和 Python 端产出相同 token 序列。
    // 它自带的 libonnxruntime.so 与 sherpa-onnx 内嵌的同名库的冲突，
    // 已在 sherpa AAR 侧改 soname 解决（见上面 packaging.jniLibs 的注释）。
    // HTTP 客户端：云端翻译 provider（OpenAI 兼容协议，含代理支持）用。
    // OkHttp 原生读 `Proxy` 对象，比自己包 HttpURLConnection 干净得多 ——
    // 需求里明确要求「代理环境下能正常翻译」，这是那条需求的落点。
    implementation("com.squareup.okhttp3:okhttp:4.12.0")

    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.31.0")

    // 端侧语音识别（sherpa-onnx，SenseVoice 模型）。
    // 不用 JitPack：`com.github.k2-fsa.sherpa-onnx` 在本机实测返回 HTTP 500，
    // 国内基本不可用。改用 GitHub Release 的 AAR 作本地文件依赖 —— 已经是磁盘上的
    // 字节，构建过程零网络风险。
    //
    // 排查两份 onnxruntime 冲突时，用 tools/toggle_noasr.sh 临时摘掉这一行（见 kotlin{} 上方注释）。
    implementation(files("../third_party/sherpa-onnx/sherpa-onnx-1.13.8.aar"))
}
