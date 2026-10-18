package com.screenast.app.translate.llama

/**
 * llama.cpp 的 JNI 门面（2B GGUF 端侧翻译）。
 *
 * 这一层**故意做得很薄**：只暴露加载/卸载、tokenize、解码、采样、清 KV。
 * EOS 判据、提前收尾救回、输出净化、流式推送、上下文策略全部在 Kotlin 侧实现
 * （见 `LlamaTranslationProvider`）—— 那里能单测、改一行不用重新交叉编译 C++。
 *
 * ⚠️ 所有 `native*` 方法都必须在 `load()` 成功返回非 0 句柄之后才能调用。
 * 句柄用 `Long` 传递（C++ 侧的裸指针），**用完必须 `free()`**，否则模型
 * 那 1.2GB 权重会一直挂着。
 *
 * ⚠️ 解码是 **append-only**：`decode()` 只会把 token 追加到 KV 末尾，位置由
 * llama.cpp 按 KV 里已有的最大值自动递增。所以「改历史」的唯一办法是
 * `memoryClear()` 整条重来 —— 这也是前缀缓存能生效的原因（第二次 decode
 * 只算新增的那部分）。
 */
internal object LlamaNative {

    init {
        // 对应 app/src/main/cpp/CMakeLists.txt 里的 target 名 llama_jni
        System.loadLibrary("llama_jni")
    }

    /** llama.cpp 的版本串，例如 "0.7.0-dev (build 11696, commit ceefd16da)"。 */
    external fun nativeVersion(): String

    /**
     * 初始化后端并注册设备。进程内只需调一次（C++ 侧有 call_once 保护）。
     *
     * @param dspLibDirs **单个**目录（不是冒号列表），会写进 `ADSP_LIBRARY_PATH`，供 FastRPC 找
     *   HTP 的 skel（`libggml-htp-v79.so`）。传空串则不设。**必须在加载模型前调用**。
     */
    external fun nativeInit(dspLibDirs: String)

    /** 最近一次失败的文字描述。返回 0 / 负数之后调它拿原因。 */
    external fun nativeLastError(): String

    /**
     * 加载 GGUF 并建上下文。失败返回 0。
     *
     * @param nCtx        上下文长度。翻译场景 512~2048 足够；**要按需给，不要照抄模型的
     *                    262144** —— KV 按 n_ctx 预分配，给大了直接吃掉几个 GB。
     * @param nBatch      单次 decode 的最大 token 数，prefill 长提示词时要够大。
     * @param device      空串 = 不限定设备（权重留在 CPU）；"GPUOpenCL" / "HTP0" 指定后端。
     * @param nGpuLayers  卸载到设备的层数；0 = 全 CPU，负数 = 全部卸载。
     */
    external fun nativeLoad(
        path: String,
        nCtx: Int,
        nThreads: Int,
        nBatch: Int,
        device: String,
        nGpuLayers: Int,
    ): Long

    external fun nativeFree(handle: Long)

    /** 实际生效的上下文长度（可能被模型上限砍过，未必等于请求值）。 */
    external fun nativeContextSize(handle: Long): Int

    /** 已经进 KV 的 token 数。单调序列下就是当前序列长度。 */
    external fun nativeKVUsed(handle: Long): Int

    /** 清空 KV，丢弃全部上文。 */
    external fun nativeMemoryClear(handle: Long)

    /** 文本 → token。失败返回 null。 */
    external fun nativeTokenize(
        handle: Long,
        text: String,
        addSpecial: Boolean,
        parseSpecial: Boolean,
    ): IntArray?

    /**
     * 单个 token → **原始字节**（不是字符串，见下）。
     *
     * ⚠️ 这里返回 `ByteArray` 而不是 `String` 是必须的，不是风格问题：一个汉字占 3 字节，
     * 经常被切成两个 token，单个 piece 就是半截 UTF-8。把它交给 JNI 的 `NewStringUTF`
     * 会**直接 SIGABRT 掉进程**（实测崩在 `nativeTokenToPiece`）。请交给 [Utf8Stream] 攒齐。
     */
    external fun nativeTokenToPiece(handle: Long, token: Int): ByteArray?

    /** 这个 token 是否属于「结束」类（eos / eot / EOG）。 */
    external fun nativeIsEog(handle: Long, token: Int): Boolean

    /** 模型的 eos token id（本模型 `<|im_end|>` = 248046）。 */
    external fun nativeEosToken(handle: Long): Int

    /**
     * 把一批 token 追加进 KV 并前向一次。
     * @return 0 成功；1 = 超出上下文；负数 = 其它失败。
     */
    external fun nativeDecode(handle: Long, tokens: IntArray): Int

    /** 建贪心采样链（temperature 0，确定性）。 */
    external fun nativeSamplerNew(): Long

    external fun nativeSamplerFree(sampler: Long)

    /** 从最后一次 decode 的最后一个位置采一个 token。 */
    external fun nativeSample(handle: Long, sampler: Long): Int
}
