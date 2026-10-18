// ---------------------------------------------------------------------------
// llama.cpp 的薄 JNI 层（2B GGUF 端侧翻译用）
//
// 设计原则：**C++ 侧只做「哑」操作，所有策略留在 Kotlin。**
//   这里只暴露：加载/卸载模型、tokenize、detokenize、解码一批 token、
//   采样一个 token、清 KV。EOS 判据、提前收尾救回、输出净化、流式推送、
//   上下文滑动窗口等全部在 Kotlin 侧实现 —— 那里能单测、能复用现有
//   NllbTranslationProvider 的写法，改一行不用重新交叉编译。
//
// 单调序列（append-only）的使用方式：
//   decode(prompt 的全部 token) → 循环 { sample → decode(单个 token) }
//   llama_batch_get_one 会把位置交给 llama.cpp 自己按 KV 里已有的最大值递增，
//   所以只要不删 KV 中途插入，前缀就是天然复用的（第二次 decode 只算新增部分）。
//   要丢弃全部上文时调用 nativeMemoryClear。
//
// 对应 Kotlin 侧：com.screenast.app.translate.llama.LlamaNative
// ---------------------------------------------------------------------------

#include <jni.h>
#include <android/log.h>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "llama.h"
#include "ggml-backend.h"

#define LOG_TAG "LlamaJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

// 把 llama.cpp / ggml 自己的日志接到 logcat。
//
// **默认回调是 fprintf(stderr)，而 Android 上 stderr 是丢掉的。** 后果是后端加载失败、
// 设备注册不上这类关键信息完全看不到 —— 界面上只表现为「设备列表里少了一个」，
// 无从查起。接了之后 hexagon 为什么没注册、skel 找不到在哪一步，全都能看见。
void log_callback(ggml_log_level level, const char * text, void * /*user_data*/) {
    int prio = ANDROID_LOG_DEBUG;
    switch (level) {
        case GGML_LOG_LEVEL_ERROR: prio = ANDROID_LOG_ERROR; break;
        case GGML_LOG_LEVEL_WARN:  prio = ANDROID_LOG_WARN;  break;
        case GGML_LOG_LEVEL_INFO:  prio = ANDROID_LOG_INFO;  break;
        default: break;
    }
    // text 自带结尾换行，__android_log_print 还会再加一个，去掉免得日志里全是空行。
    std::string s(text);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    if (!s.empty()) __android_log_print(prio, LOG_TAG, "%s", s.c_str());
}

// 最近一次失败的描述。Kotlin 侧在拿到 0 / 负数返回后调用 nativeLastError() 取。
std::mutex  g_err_mutex;
std::string g_last_error;

void set_error(const std::string & msg) {
    std::lock_guard<std::mutex> lock(g_err_mutex);
    g_last_error = msg;
    LOGE("%s", msg.c_str());
}

void clear_error() {
    std::lock_guard<std::mutex> lock(g_err_mutex);
    g_last_error.clear();
}

// jstring → 真正的 UTF-8 字节。
//
// **不要用 GetStringUTFChars**：它给的是 JNI 定义的 *Modified UTF-8*，U+10000
// 以上（emoji、部分生僻字）会被编成 6 字节的代理对形式，直接喂给 llama_tokenize
// 会拆错。这里按 UTF-16 重编一遍真 UTF-8。
std::string to_utf8(JNIEnv * env, jstring js) {
    if (js == nullptr) return {};
    const jsize n = env->GetStringLength(js);
    if (n == 0) return {};
    const jchar * u = env->GetStringChars(js, nullptr);

    std::string out;
    out.reserve((size_t) n * 3);
    for (jsize i = 0; i < n; i++) {
        uint32_t cp = u[i];
        if (cp >= 0xD800 && cp <= 0xDBFF && i + 1 < n) {
            const uint32_t lo = u[i + 1];
            if (lo >= 0xDC00 && lo <= 0xDFFF) {
                cp = 0x10000u + ((cp - 0xD800u) << 10) + (lo - 0xDC00u);
                i++;
            }
        }
        if (cp < 0x80) {
            out.push_back((char) cp);
        } else if (cp < 0x800) {
            out.push_back((char) (0xC0 | (cp >> 6)));
            out.push_back((char) (0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back((char) (0xE0 | (cp >> 12)));
            out.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char) (0x80 | (cp & 0x3F)));
        } else {
            out.push_back((char) (0xF0 | (cp >> 18)));
            out.push_back((char) (0x80 | ((cp >> 12) & 0x3F)));
            out.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char) (0x80 | (cp & 0x3F)));
        }
    }
    env->ReleaseStringChars(js, u);
    return out;
}

// 模型 + 上下文 + 词表 + 已进 KV 的 token 数。
//
// n_past 自己记账而不是查询 KV：本层只支持 append-only 用法，自己记更省事也更准。
// 一旦引入「删掉中间某段再续写」就需要改成查 llama_memory_seq_pos_max + 1。
struct Handle {
    llama_model       * model = nullptr;
    llama_context     * ctx   = nullptr;
    const llama_vocab * vocab = nullptr;
    uint32_t            n_ctx = 0;
    // 单次 llama_decode 能吃的 token 上限。llama_context::decode 里是
    // GGML_ASSERT(n_tokens_all <= cparams.n_batch) —— 超了不是返回错误，
    // 而是直接 ggml_abort() → SIGABRT（整进程闪退，logcat 里只有 abort 栈）。
    // 所以 nativeDecode 必须自己按这个值切片。
    uint32_t            n_batch = 0;
    int32_t             n_past = 0;
};

inline Handle * as_handle(jlong h) { return reinterpret_cast<Handle *>(h); }

// 设备选择用的数组必须是静态存储期 —— llama_model_load_from_file 期间会被引用，
// 传一个临时 vector 的 data() 是悬垂指针。加载本身用锁串行化。
std::mutex                g_load_mutex;
std::vector<ggml_backend_dev_t> g_device_list;

} // namespace

extern "C" {

JNIEXPORT jstring JNICALL
Java_com_screenast_app_translate_llama_LlamaNative_nativeVersion(JNIEnv * env, jclass) {
    return env->NewStringUTF(llama_version());
}

JNIEXPORT void JNICALL
Java_com_screenast_app_translate_llama_LlamaNative_nativeInit(JNIEnv * env, jclass, jstring dsp_lib_dirs) {
    // ★ HTP skel 的查找路径 —— 不设这个变量，NPU 会话一定打不开。
    //
    // ggml-hexagon.cpp:3206 把 skel 的 URI 拼成**裸文件名**：
    //     "file:///libggml-htp-v%u.so?htp_iface_skel_handle_invoke&_modver=1.0"
    // 目录部分永远是空的，所以 FastRPC 只能按 ADSP_LIBRARY_PATH 去找这个文件。
    // ⚠️ 这个变量只认**一个**目录，不是 PATH 那样的冒号列表 —— 传 "a:b:c" 的话
    //    FastRPC 会把整串当成一个目录名去拼路径（日志里能看到 "a:b:c/./libggml-htp-v79.so"），
    //    于是永远找不到。App 里 .so 本来都在 APK 内（默认 extractNativeLibs=false 时
    //    连 applicationInfo.nativeLibraryDir 这个目录都不存在），于是：
    //     HTP0 failed to open session : uri file:///libggml-htp-v79.so ... error 0x80000406
    // 必须在 ggml_backend_load_all() 之前设好。
    if (dsp_lib_dirs != nullptr) {
        const char * dirs = env->GetStringUTFChars(dsp_lib_dirs, nullptr);
        if (dirs != nullptr && dirs[0] != '\0') {
            setenv("ADSP_LIBRARY_PATH", dirs, 1);
            LOGI("ADSP_LIBRARY_PATH=%s", dirs);
        }
        if (dirs != nullptr) env->ReleaseStringUTFChars(dsp_lib_dirs, dirs);
    }

    static std::once_flag once;
    std::call_once(once, [] {
        llama_log_set(log_callback, nullptr);
        llama_backend_init();
        // 扫描 APK 的 native 库目录，把 libggml-cpu / -opencl / -hexagon 注册进来。
        // llama_backend_init() 内部通常已经调过，这里显式再来一次是幂等的。
        ggml_backend_load_all();

        const size_t n = ggml_backend_dev_count();
        LOGI("ggml 后端设备共 %zu 个：", n);
        for (size_t i = 0; i < n; i++) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            LOGI("  [%zu] name=%s", i, ggml_backend_dev_name(d));
        }
        LOGI("llama.cpp 版本：%s", llama_version());
    });
}

JNIEXPORT jstring JNICALL
Java_com_screenast_app_translate_llama_LlamaNative_nativeLastError(JNIEnv * env, jclass) {
    std::lock_guard<std::mutex> lock(g_err_mutex);
    return env->NewStringUTF(g_last_error.c_str());
}

// 加载模型并建上下文。成功返回句柄，失败返回 0（错误详情见 nativeLastError）。
//
// device 传空串 = 不限制设备列表（此时靠 nGpuLayers=0 把权重留在 CPU）；
// 传 "GPUOpenCL" / "HTP0" 则只把该设备交给 llama.cpp（配合 nGpuLayers=-1 全量卸载）。
JNIEXPORT jlong JNICALL
Java_com_screenast_app_translate_llama_LlamaNative_nativeLoad(
        JNIEnv * env, jclass,
        jstring jpath, jint n_ctx, jint n_threads, jint n_batch,
        jstring jdevice, jint n_gpu_layers) {

    clear_error();

    if (jpath == nullptr) { set_error("模型路径为空"); return 0; }
    const std::string path   = to_utf8(env, jpath);
    const std::string device = to_utf8(env, jdevice);

    std::lock_guard<std::mutex> lock(g_load_mutex);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    // 想中途取消加载时由这层返回 false；这里暂时不需要，保持默认。
    mparams.progress_callback = nullptr;

    g_device_list.clear();
    if (!device.empty()) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(device.c_str());
        if (dev == nullptr) {
            std::string msg = "找不到设备 \"" + device + "\"，可用设备：";
            const size_t n = ggml_backend_dev_count();
            for (size_t i = 0; i < n; i++) {
                msg += ggml_backend_dev_name(ggml_backend_dev_get(i));
                if (i + 1 < n) msg += ", ";
            }
            set_error(msg);
            return 0;
        }
        g_device_list.push_back(dev);
        g_device_list.push_back(nullptr);   // 必须以 NULL 结尾
        mparams.devices = g_device_list.data();
        LOGI("限定设备：%s", device.c_str());
    }

    LOGI("开始加载模型：%s（n_ctx=%d threads=%d batch=%d ngl=%d）",
         path.c_str(), n_ctx, n_threads, n_batch, n_gpu_layers);

    llama_model * model = llama_model_load_from_file(path.c_str(), mparams);
    if (model == nullptr) {
        set_error("llama_model_load_from_file 返回空，模型可能损坏或不是 GGUF");
        return 0;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = (uint32_t) n_ctx;
    cparams.n_batch         = (uint32_t) n_batch;
    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads;
    // 翻译是短序列，不需要投机/嵌入，显式关掉省内存。
    cparams.embeddings      = false;
    cparams.no_perf         = false;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        llama_model_free(model);
        set_error("llama_init_from_model 返回空，n_ctx 可能超出模型支持范围");
        return 0;
    }

    auto * h = new Handle();
    h->model = model;
    h->ctx   = ctx;
    h->vocab = llama_model_get_vocab(model);
    h->n_ctx = llama_n_ctx(ctx);
    h->n_batch = (uint32_t) llama_n_batch(ctx);
    h->n_past = 0;

    LOGI("加载完成：实际 n_ctx=%u n_batch=%u", h->n_ctx, h->n_batch);
    return reinterpret_cast<jlong>(h);
}

JNIEXPORT void JNICALL
Java_com_screenast_app_translate_llama_LlamaNative_nativeFree(JNIEnv *, jclass, jlong jh) {
    Handle * h = as_handle(jh);
    if (h == nullptr) return;
    if (h->ctx)   llama_free(h->ctx);
    if (h->model) llama_model_free(h->model);
    delete h;
}

JNIEXPORT jint JNICALL
Java_com_screenast_app_translate_llama_LlamaNative_nativeContextSize(JNIEnv *, jclass, jlong jh) {
    Handle * h = as_handle(jh);
    return h ? (jint) h->n_ctx : 0;
}

// 已经进 KV 的 token 数。单调序列下等于「上次解码结束时的位置」。
JNIEXPORT jint JNICALL
Java_com_screenast_app_translate_llama_LlamaNative_nativeKVUsed(JNIEnv *, jclass, jlong jh) {
    Handle * h = as_handle(jh);
    return h ? (jint) h->n_past : 0;
}

// 清空 KV（丢弃全部上文）。之后可以重新从零开始一条新序列。
JNIEXPORT void JNICALL
Java_com_screenast_app_translate_llama_LlamaNative_nativeMemoryClear(JNIEnv *, jclass, jlong jh) {
    Handle * h = as_handle(jh);
    if (h == nullptr) return;
    llama_memory_clear(llama_get_memory(h->ctx), true);
    h->n_past = 0;
}

// 文本 → token 数组。
//
// add_special 传 false：这个模型的 add_bos_token = false，而且生产提示词是逐字节
// 手拼的（<|im_start|> 等由 parse_special 识别成单个 token），多塞一个 BOS 会破坏前缀。
JNIEXPORT jintArray JNICALL
Java_com_screenast_app_translate_llama_LlamaNative_nativeTokenize(
        JNIEnv * env, jclass, jlong jh, jstring jtext, jboolean add_special, jboolean parse_special) {

    Handle * h = as_handle(jh);
    if (h == nullptr) { set_error("tokenize：句柄为空"); return nullptr; }
    if (jtext == nullptr) return env->NewIntArray(0);

    const std::string text = to_utf8(env, jtext);
    const int32_t text_len = (int32_t) text.size();

    // 先按「UTF-8 字节数 + 富余」估一个容量；不够时 llama_tokenize 会返回所需的负数。
    int32_t cap = text_len + 16;
    std::vector<llama_token> tokens(cap);

    int32_t n = llama_tokenize(h->vocab, text.c_str(), text_len,
                               tokens.data(), cap, add_special, parse_special);
    if (n < 0) {
        cap = -n;
        tokens.resize(cap);
        n = llama_tokenize(h->vocab, text.c_str(), text_len,
                           tokens.data(), cap, add_special, parse_special);
    }

    if (n < 0) { set_error("tokenize 二次调用仍然失败"); return nullptr; }

    jintArray out = env->NewIntArray(n);
    if (out != nullptr && n > 0) {
        env->SetIntArrayRegion(out, 0, n, reinterpret_cast<const jint *>(tokens.data()));
    }
    return out;
}

// 单个 token → **原始字节**。lstrip=0、special=true（需要看到 <|im_end|> 之类才能判停）。
//
// ⚠️ 这里刻意返回 jbyteArray 而不是 jstring。
//    词表是字节级 BPE，一个汉字（3 字节）经常被切成两个 token，前一个 token 的
//    piece 就是**半截 UTF-8**（例如「酩」的前两字节 0xE9 0x85）。而 JNI 的
//    NewStringUTF 要求的是 Modified UTF-8，碰到这种字节会**直接 SIGABRT**：
//        JNI DETECTED ERROR IN APPLICATION: input is not valid Modified UTF-8:
//        illegal continuation byte 0 / in call to NewStringUTF
//    实测就崩在这里（子进程 abort，日志里只有这一行线索）。
//    把字节原样交给 Kotlin，由 Utf8Stream 跨 token 攒齐一个完整字符再解码。
JNIEXPORT jbyteArray JNICALL
Java_com_screenast_app_translate_llama_LlamaNative_nativeTokenToPiece(
        JNIEnv * env, jclass, jlong jh, jint token) {

    Handle * h = as_handle(jh);
    if (h == nullptr) return env->NewByteArray(0);

    char buf[256];
    int32_t n = llama_token_to_piece(h->vocab, token, buf, sizeof(buf), 0, true);

    std::vector<char> big;
    const char * src = buf;
    if (n < 0) {
        // 缓冲区不够时返回的是「所需的负长度」，不是失败。原实现直接当成失败返回空串，
        // 会**静默吞掉**超长 piece。
        big.resize((size_t) (-n) + 16);
        n = llama_token_to_piece(h->vocab, token, big.data(), (int32_t) big.size(), 0, true);
        if (n < 0) return env->NewByteArray(0);
        src = big.data();
    }
    if (n == 0) return env->NewByteArray(0);

    jbyteArray out = env->NewByteArray(n);
    if (out != nullptr) {
        env->SetByteArrayRegion(out, 0, n, reinterpret_cast<const jbyte *>(src));
    }
    return out;
}

JNIEXPORT jboolean JNICALL
Java_com_screenast_app_translate_llama_LlamaNative_nativeIsEog(JNIEnv *, jclass, jlong jh, jint token) {
    Handle * h = as_handle(jh);
    if (h == nullptr) return JNI_TRUE;
    return llama_vocab_is_eog(h->vocab, token) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jint JNICALL
Java_com_screenast_app_translate_llama_LlamaNative_nativeEosToken(JNIEnv *, jclass, jlong jh) {
    Handle * h = as_handle(jh);
    return h ? (jint) llama_vocab_eos(h->vocab) : -1;
}

// 把一批 token 追加进 KV 并前向一次。返回 0 成功，1 = 超出上下文，负数 = 其它失败。
//
// ⚠️ 只支持 append：位置由 llama.cpp 按 KV 里已有的最大位置递增（llama_batch_get_one
//    把 pos 留空）。想「改写历史」必须先 nativeMemoryClear 整条重来。
JNIEXPORT jint JNICALL
Java_com_screenast_app_translate_llama_LlamaNative_nativeDecode(
        JNIEnv * env, jclass, jlong jh, jintArray jtokens) {

    Handle * h = as_handle(jh);
    if (h == nullptr) { set_error("decode：句柄为空"); return -1; }
    if (jtokens == nullptr) return 0;

    const jsize n = env->GetArrayLength(jtokens);
    if (n == 0) return 0;

    if ((int64_t) h->n_past + n > (int64_t) h->n_ctx) {
        set_error("上下文已满：" + std::to_string(h->n_past) + " + " +
                  std::to_string(n) + " > " + std::to_string(h->n_ctx));
        return 1;
    }

    std::vector<llama_token> tokens(n);
    env->GetIntArrayRegion(jtokens, 0, n, reinterpret_cast<jint *>(tokens.data()));

    // ★ 必须切片：llama_context::decode() 开头是
    //     GGML_ASSERT(n_tokens_all <= cparams.n_batch);
    // 而 prompt 带着「上下文档数」的原文+译文一起塞，句子一多就超过 n_batch
    // （默认 256/512），此时不是返回错误码，而是 ggml_abort() → SIGABRT 闪退。
    // 现象就是「翻到一半突然闪退」，且崩溃栈只有 abort + llama_context::decode。
    //
    // llama_batch_get_one 不填 pos，llama.cpp 会按 KV 里已有最大位置自动递增，
    // 所以分片前向和一次性前向等价。
    const int32_t chunk_max = (h->n_batch > 0) ? (int32_t) h->n_batch : (int32_t) n;
    for (int32_t done = 0; done < (int32_t) n; ) {
        const int32_t chunk = std::min<int32_t>((int32_t) n - done, chunk_max);
        llama_batch batch = llama_batch_get_one(tokens.data() + done, chunk);
        const int32_t rc = llama_decode(h->ctx, batch);
        if (rc != 0) {
            set_error("llama_decode 失败，返回 " + std::to_string(rc) +
                      "（已解码 " + std::to_string(done) + "/" + std::to_string(n) + "）");
            return rc < 0 ? rc : -2;
        }
        done += chunk;
    }
    h->n_past += (int32_t) n;
    return 0;
}

// 贪心采样链。目前只要确定性输出，所以链里只有 greedy；
// 以后要加重复惩罚 / logit bias 在这里 llama_sampler_chain_add 即可。
JNIEXPORT jlong JNICALL
Java_com_screenast_app_translate_llama_LlamaNative_nativeSamplerNew(JNIEnv *, jclass) {
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    sparams.no_perf = true;
    llama_sampler * chain = llama_sampler_chain_init(sparams);
    if (chain == nullptr) { set_error("采样链创建失败"); return 0; }
    llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    return reinterpret_cast<jlong>(chain);
}

JNIEXPORT void JNICALL
Java_com_screenast_app_translate_llama_LlamaNative_nativeSamplerFree(JNIEnv *, jclass, jlong js) {
    auto * s = reinterpret_cast<llama_sampler *>(js);
    if (s != nullptr) llama_sampler_free(s);
}

// 从「最后一次 decode 的最后一个位置」采一个 token。idx 传 -1 表示最后一行。
JNIEXPORT jint JNICALL
Java_com_screenast_app_translate_llama_LlamaNative_nativeSample(
        JNIEnv *, jclass, jlong jh, jlong js) {

    Handle * h = as_handle(jh);
    auto * s = reinterpret_cast<llama_sampler *>(js);
    if (h == nullptr || s == nullptr) { set_error("sample：句柄或采样器为空"); return -1; }
    return (jint) llama_sampler_sample(s, h->ctx, -1);
}

} // extern "C"
