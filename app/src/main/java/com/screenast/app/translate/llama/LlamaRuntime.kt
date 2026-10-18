package com.screenast.app.translate.llama

/**
 * llama.cpp 会话的进程级单例。
 *
 * 为什么必须是单例：模型 1.27GB。调试台和正式 provider 各持一份句柄的话，
 * 一次「调试台看一眼 + 回主界面翻译」就会同时挂两份，直接 2.5GB —— 而手机
 * 上本来就还躺着 NLLB 的 3.5GB native。**一个进程只允许有一个 llama 会话。**
 *
 * 线程模型：所有 native 调用都在 [lock] 里串行。llama.cpp 的 context 不是
 * 线程安全的，而翻译在后台线程、调试台在另一个线程，不加锁会偶发段错误
 * （症状是 native crash，日志里什么都看不到，极难排查）。
 */
internal object LlamaRuntime {

    private val lock = Any()

    @Volatile
    private var handle: Long = 0L

    private var sampler: Long = 0L

    /** 当前加载用的**配置指纹**；和请求不同就重新加载。 */
    @Volatile
    private var loadedKey: String? = null

    /** 最近一次加载耗时（ms），用于区分「首句慢」是加载还是推理。 */
    @Volatile
    var lastLoadMs: Long = 0L
        private set

    /**
     * **单个**目录（不是冒号列表），写进 `ADSP_LIBRARY_PATH`，供 FastRPC 找 HTP 的 skel。
     *
     * **必须在第一次 [acquire] 之前设好** —— C++ 侧是 `call_once`，之后再改无效。
     * 取值的正解见 [LlamaModels.dspLibDirs]。
     */
    @Volatile
    var dspLibDirs: String = ""

    val isLoaded: Boolean get() = handle != 0L

    /** 实际生效的上下文长度（模型上限会砍请求值）。 */
    val contextSize: Int get() = if (handle == 0L) 0 else LlamaNative.nativeContextSize(handle)

    /** 已进 KV 的 token 数。 */
    val kvUsed: Int get() = if (handle == 0L) 0 else LlamaNative.nativeKVUsed(handle)

    /**
     * 取到可用的句柄；没加载过就加载。已加载且**配置完全一致**则直接复用（不重新读盘）。
     *
     * ⚠️ 缓存键必须包含 device / ngl / nCtx 这些参数，不能只用路径。
     *    只用路径的后果：先按 CPU 加载、再请求 HTP0，会被判定为「已加载」直接返回
     *    那个 CPU 句柄 —— 设备静默不生效，测出来的数还是 CPU 的数，且毫无提示。
     *
     * @return 句柄；0 表示失败，原因见 [LlamaNative.nativeLastError]。
     */
    fun acquire(
        path: String,
        nCtx: Int,
        nThreads: Int,
        nBatch: Int,
        device: String,
        nGpuLayers: Int,
    ): Long = synchronized(lock) {
        val key = "$path|$nCtx|$nThreads|$nBatch|$device|$nGpuLayers"
        if (handle != 0L && loadedKey == key) return handle

        // 配置变了：先把旧的放掉，否则两份权重同时在内存里。
        if (handle != 0L) releaseLocked()

        LlamaNative.nativeInit(dspLibDirs)
        val t0 = System.currentTimeMillis()
        val h = LlamaNative.nativeLoad(path, nCtx, nThreads, nBatch, device, nGpuLayers)
        lastLoadMs = System.currentTimeMillis() - t0
        if (h == 0L) return 0L

        handle = h
        loadedKey = key
        sampler = LlamaNative.nativeSamplerNew()
        h
    }

    fun release() = synchronized(lock) { releaseLocked() }

    private fun releaseLocked() {
        if (sampler != 0L) {
            LlamaNative.nativeSamplerFree(sampler)
            sampler = 0L
        }
        if (handle != 0L) {
            LlamaNative.nativeFree(handle)
            handle = 0L
        }
        loadedKey = null
    }

    /** 清空 KV。换句、换语言、放弃上下文时都要调。 */
    fun memoryClear() = synchronized(lock) {
        if (handle != 0L) LlamaNative.nativeMemoryClear(handle)
    }

    /**
     * 把一段提示词喂进 KV 并前向一次，但**不采样**。
     * 用于「只想把上文塞进去」的场合。
     */
    fun prefill(prompt: String): Int = synchronized(lock) {
        if (handle == 0L) return -1
        val toks = LlamaNative.nativeTokenize(handle, prompt, false, true) ?: return -1
        LlamaNative.nativeDecode(handle, toks)
    }

    /**
     * 纯贪心生成循环。
     *
     * 这里**不做任何文本后处理**（EOS 救回、`</think>` 截断、标点清洗都在 Kotlin 侧
     * 的 provider 里）—— 这一层的职责只有一个：把 token 一个个吐出来。
     *
     * @param prompt  完整提示词（含特殊 token 字面量，靠 parseSpecial 解析）。
     * @param maxNew  最多生成多少 token，防止退化时无限循环。
     * @param onPiece 每产出一个**完整字符**的回调，用于流式字幕。**在锁内调用**，
     *                实现里不要做耗时操作，否则会把别的线程堵在 [lock] 上。
     * @return 生成的原始文本（未经清洗）。
     */
    fun generate(
        prompt: String,
        maxNew: Int,
        onPiece: ((String) -> Unit)? = null,
    ): String = synchronized(lock) {
        if (handle == 0L) return ""

        val toks = LlamaNative.nativeTokenize(handle, prompt, false, true) ?: return ""
        val rc = LlamaNative.nativeDecode(handle, toks)
        if (rc != 0) return ""

        // 跨 token 攒 UTF-8。**不要**把 piece 直接当字符串 —— 一个汉字常被切成两个
        // token，半截字节交给 JNI 会 SIGABRT，交给 String() 会变替换符。
        val utf8 = Utf8Stream()
        val sb = StringBuilder()
        var produced = 0
        while (produced < maxNew) {
            val t = LlamaNative.nativeSample(handle, sampler)
            if (t < 0 || LlamaNative.nativeIsEog(handle, t)) break
            val piece = utf8.feed(LlamaNative.nativeTokenToPiece(handle, t) ?: EMPTY_BYTES)
            if (piece.isNotEmpty()) {
                sb.append(piece)
                onPiece?.invoke(piece)
            }
            // 单个 token 也要 decode 回去，否则下一个 token 的 KV 位置不对。
            if (LlamaNative.nativeDecode(handle, intArrayOf(t)) != 0) break
            produced++
        }

        val tail = utf8.flush()
        if (tail.isNotEmpty()) {
            sb.append(tail)
            onPiece?.invoke(tail)
        }
        sb.toString()
    }

    private val EMPTY_BYTES = ByteArray(0)
}
