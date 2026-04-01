#include <jni.h>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <cstring>
#include <algorithm>
#include <android/log.h>

#include "llama.h"
#include "mtmd.h"

#define TAG "LlamaJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static struct llama_model*   g_model = nullptr;
static struct llama_context* g_ctx   = nullptr;
static struct mtmd_context*  g_mtmd  = nullptr;
static std::mutex            g_mutex;
static std::atomic<bool>     g_abort{false};

static void free_all() {
    if (g_mtmd)  { mtmd_free(g_mtmd);        g_mtmd  = nullptr; }
    if (g_ctx)   { llama_free(g_ctx);         g_ctx   = nullptr; }
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }
    LOGI("All resources freed.");
}

static jstring get_safe_utf8_string(JNIEnv* env, const char* data, size_t length) {
    jbyteArray jBytes = env->NewByteArray((jsize)length);
    env->SetByteArrayRegion(jBytes, 0, (jsize)length, reinterpret_cast<const jbyte*>(data));
    jclass     strClass  = env->FindClass("java/lang/String");
    jmethodID  strCtor   = env->GetMethodID(strClass, "<init>", "([BLjava/lang/String;)V");
    jstring    enc       = env->NewStringUTF("UTF-8");
    jstring    result    = (jstring)env->NewObject(strClass, strCtor, jBytes, enc);
    env->DeleteLocalRef(jBytes);
    env->DeleteLocalRef(enc);
    env->DeleteLocalRef(strClass);
    return result;
}

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM*, void*) {
    llama_backend_init();
    return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT void JNICALL JNI_OnUnload(JavaVM*, void*) {
    llama_backend_free();
}

// ── loadModel ──────────────────────────────────────────────────────────────────
extern "C" JNIEXPORT jboolean JNICALL
Java_com_anurag_visionqa_ai_LlamaJNI_loadModel(
        JNIEnv* env, jobject,
        jstring modelPath, jstring mmprojPath,
        jint nThreads, jint nCtx)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    free_all();

    const char* m_path = env->GetStringUTFChars(modelPath,  nullptr);
    const char* v_path = env->GetStringUTFChars(mmprojPath, nullptr);

    // 1. Load model
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    g_model = llama_model_load_from_file(m_path, mp);
    if (!g_model) { LOGE("Model load failed."); goto fail; }

    // 2. Create context — force 4096, 2048 is too small for vision
    {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = 4096;
        cp.n_batch         = 512;
        cp.n_threads       = (uint32_t)nThreads;
        cp.n_threads_batch = (uint32_t)nThreads;
        g_ctx = llama_init_from_model(g_model, cp);
    }
    if (!g_ctx) { LOGE("Context init failed."); goto fail; }

    // 3. Load vision encoder
    {
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.n_threads = (int32_t)nThreads;
        mparams.use_gpu   = false;
        g_mtmd = mtmd_init_from_file(v_path, g_model, mparams);
    }
    if (!g_mtmd) { LOGE("MTMD init failed."); goto fail; }

    {
        const llama_vocab* vocab = llama_model_get_vocab(g_model);
        LOGI("✅ Model ready | n_vocab=%d n_embd=%d n_embd_inp=%d ctx=4096 BOS=%d EOS=%d",
             llama_vocab_n_tokens(vocab),
             llama_model_n_embd(g_model),
             llama_model_n_embd_inp(g_model),
             llama_vocab_bos(vocab),
             llama_vocab_eos(vocab));
    }

    env->ReleaseStringUTFChars(modelPath,  m_path);
    env->ReleaseStringUTFChars(mmprojPath, v_path);
    return JNI_TRUE;

    fail:
    free_all();
    env->ReleaseStringUTFChars(modelPath,  m_path);
    env->ReleaseStringUTFChars(mmprojPath, v_path);
    return JNI_FALSE;
}

// ── decode helpers ─────────────────────────────────────────────────────────────
static bool decode_text_batch(
        const llama_token* toks, size_t n_toks,
        llama_pos& n_past, bool need_logits)
{
    const size_t BATCH = 512;
    for (size_t off = 0; off < n_toks; off += BATCH) {
        size_t chunk     = std::min(BATCH, n_toks - off);
        bool   last      = (off + chunk >= n_toks);

        llama_batch b = llama_batch_init((int32_t)chunk, 0, 1);
        b.n_tokens = (int32_t)chunk;
        for (size_t j = 0; j < chunk; j++) {
            b.token[j]     = toks[off + j];
            b.pos[j]       = n_past + (llama_pos)j;
            b.n_seq_id[j]  = 1;
            b.seq_id[j][0] = 0;
            b.logits[j]    = (last && j == chunk - 1 && need_logits) ? 1 : 0;
        }
        int ret = llama_decode(g_ctx, b);
        llama_batch_free(b);
        if (ret != 0) { LOGE("decode_text_batch failed: ret=%d", ret); return false; }
        n_past += (llama_pos)chunk;
    }
    return true;
}

static bool decode_image_batch(
        float* embd, size_t n_toks, int32_t n_embd,
        llama_pos& n_past, bool need_logits)
{
    const size_t BATCH = 512;
    for (size_t off = 0; off < n_toks; off += BATCH) {
        size_t chunk = std::min(BATCH, n_toks - off);
        bool   last  = (off + chunk >= n_toks);

        llama_batch b = llama_batch_init((int32_t)chunk, n_embd, 1);
        b.n_tokens = (int32_t)chunk;
        std::memcpy(b.embd, embd + off * n_embd, chunk * n_embd * sizeof(float));
        for (size_t j = 0; j < chunk; j++) {
            b.pos[j]       = n_past + (llama_pos)j;
            b.n_seq_id[j]  = 1;
            b.seq_id[j][0] = 0;
            b.logits[j]    = (last && j == chunk - 1 && need_logits) ? 1 : 0;
        }
        int ret = llama_decode(g_ctx, b);
        llama_batch_free(b);
        if (ret != 0) { LOGE("decode_image_batch failed: ret=%d", ret); return false; }
        n_past += (llama_pos)chunk;
    }
    return true;
}

// ── generate ──────────────────────────────────────────────────────────────────
extern "C" JNIEXPORT jstring JNICALL
Java_com_anurag_visionqa_ai_LlamaJNI_generate(
        JNIEnv* env, jobject,
        jintArray pixelData,
        jint imageWidth, jint imageHeight,
        jstring prompt,
        jint maxTokens,
        jobject tokenCallback)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_model || !g_ctx || !g_mtmd)
        return env->NewStringUTF("Error: Engine not ready.");

    g_abort.store(false);

    jclass    cb_class  = env->GetObjectClass(tokenCallback);
    jmethodID cb_invoke = env->GetMethodID(cb_class, "invoke", "(Ljava/lang/Object;)Ljava/lang/Object;");
    env->DeleteLocalRef(cb_class);

    // 1. ARGB → RGB
    jint*  argb = env->GetIntArrayElements(pixelData, nullptr);
    jsize  npx  = env->GetArrayLength(pixelData);
    std::vector<uint8_t> rgb((size_t)npx * 3);
    for (jsize i = 0; i < npx; i++) {
        uint32_t px = (uint32_t)argb[i];
        rgb[i*3+0]  = (px >> 16) & 0xFF;
        rgb[i*3+1]  = (px >>  8) & 0xFF;
        rgb[i*3+2]  =  px        & 0xFF;
    }
    env->ReleaseIntArrayElements(pixelData, argb, JNI_ABORT);

    mtmd_bitmap* bitmap = mtmd_bitmap_init((uint32_t)imageWidth, (uint32_t)imageHeight, rgb.data());
    if (!bitmap) return env->NewStringUTF("Error: bitmap init failed.");

    // 2. Build prompt
    // FIX: add_bos=true, add_eos=false — never double-add EOS before generation
    const char* marker = mtmd_default_marker();
    const char* raw_q  = env->GetStringUTFChars(prompt, nullptr);

    std::string full_text =
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            + std::string(marker) + "\n"
            + std::string(raw_q ? raw_q : "") + "\n"
            + "<|im_end|>\n"
              "<|im_start|>assistant\n";

    env->ReleaseStringUTFChars(prompt, raw_q);

    LOGI("Prompt (first 120 chars): %.120s", full_text.c_str());

    // 3. Tokenize
    // FIX: add_special=true, parse_special=true — but BOS is handled by ChatML
    //      so we set add_special=false to avoid double BOS
    mtmd_input_text itext  = { full_text.c_str(), /*add_special=*/false, /*parse_special=*/true };
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    const mtmd_bitmap* bitmaps_list[1] = { bitmap };

    if (mtmd_tokenize(g_mtmd, chunks, &itext, bitmaps_list, 1) != 0) {
        LOGE("mtmd_tokenize failed.");
        mtmd_bitmap_free(bitmap);
        mtmd_input_chunks_free(chunks);
        return env->NewStringUTF("Error: Tokenization failed.");
    }

    size_t n_chunks = mtmd_input_chunks_size(chunks);
    LOGI("=== %zu chunks after tokenize ===", n_chunks);
    for (size_t i = 0; i < n_chunks; i++) {
        const mtmd_input_chunk* ch = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            LOGI("  [%zu] IMAGE: %zu tokens", i, mtmd_input_chunk_get_n_tokens(ch));
        } else {
            size_t nt = 0; mtmd_input_chunk_get_tokens_text(ch, &nt);
            LOGI("  [%zu] TEXT:  %zu tokens", i, nt);
        }
    }

    // 4. Prefill
    llama_memory_clear(llama_get_memory(g_ctx), true);
    llama_pos n_past    = 0;
    bool      prefill_ok = true;

    for (size_t i = 0; i < n_chunks && prefill_ok; i++) {
        const mtmd_input_chunk* ch = mtmd_input_chunks_get(chunks, i);
        bool is_last = (i == n_chunks - 1);

        if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            LOGI("Encoding image chunk %zu (n_past=%d)...", i, (int)n_past);

            if (mtmd_encode_chunk(g_mtmd, ch) != 0) {
                LOGE("mtmd_encode_chunk failed.");
                prefill_ok = false; break;
            }

            float*  embd   = mtmd_get_output_embd(g_mtmd);
            size_t  n_toks = mtmd_input_chunk_get_n_tokens(ch);
            int32_t n_embd = llama_model_n_embd_inp(g_model);

            LOGI("  Image: n_toks=%zu n_embd=%d n_past_before=%d", n_toks, n_embd, (int)n_past);

            if (n_past + (llama_pos)n_toks > (llama_pos)llama_n_ctx(g_ctx)) {
                LOGE("Context overflow: need %zu slots, have %d",
                     n_toks, (int)(llama_n_ctx(g_ctx) - n_past));
                prefill_ok = false; break;
            }

            prefill_ok = decode_image_batch(embd, n_toks, n_embd, n_past, is_last);
            LOGI("  Image prefill done. n_past=%d", (int)n_past);

        } else {
            size_t n_toks = 0;
            const llama_token* toks = mtmd_input_chunk_get_tokens_text(ch, &n_toks);
            if (n_toks == 0) continue;

            LOGI("Text chunk %zu: %zu tokens (n_past=%d)", i, n_toks, (int)n_past);

            if (n_past + (llama_pos)n_toks > (llama_pos)llama_n_ctx(g_ctx)) {
                LOGE("Context overflow on text chunk.");
                prefill_ok = false; break;
            }

            prefill_ok = decode_text_batch(toks, n_toks, n_past, is_last);
        }
    }

    mtmd_bitmap_free(bitmap);
    mtmd_input_chunks_free(chunks);

    if (!prefill_ok) return env->NewStringUTF("Error: Prefill failed.");
    LOGI("=== Prefill complete. n_past=%d. Generating... ===", (int)n_past);

    // 5. Generation — use greedy for now to eliminate randomness as a variable
    const llama_vocab* vocab = llama_model_get_vocab(g_model);

    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(sparams);
    // Greedy: deterministic, no hallucination from temperature
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    std::string result;
    result.reserve(512);

    for (int step = 0; step < (int)maxTokens; step++) {
        if (g_abort.load() || n_past >= (llama_pos)llama_n_ctx(g_ctx)) break;

        llama_token id = llama_sampler_sample(smpl, g_ctx, -1);
        llama_sampler_accept(smpl, id);

        if (llama_vocab_is_eog(vocab, id)) {
            LOGI("EOS at step %d", step);
            break;
        }

        char piece[256] = {};
        int  plen = llama_token_to_piece(vocab, id, piece, sizeof(piece), 0, true);
        if (plen <= 0) { LOGE("Bad piece at step %d (id=%d plen=%d)", step, id, plen); break; }

        LOGI("step=%d id=%d piece='%s'", step, id, piece);

        result.append(piece, (size_t)plen);

        // Stream token to Kotlin
        jbyteArray jb = env->NewByteArray(plen);
        env->SetByteArrayRegion(jb, 0, plen, (const jbyte*)piece);
        env->CallObjectMethod(tokenCallback, cb_invoke, jb);
        env->DeleteLocalRef(jb);
        if (env->ExceptionCheck()) env->ExceptionClear();

        // Decode the new token
        llama_batch nb    = llama_batch_init(1, 0, 1);
        nb.n_tokens       = 1;
        nb.token[0]       = id;
        nb.pos[0]         = n_past++;
        nb.n_seq_id[0]    = 1;
        nb.seq_id[0][0]   = 0;
        nb.logits[0]      = 1;
        int ret = llama_decode(g_ctx, nb);
        llama_batch_free(nb);
        if (ret != 0) { LOGE("llama_decode failed at step %d", step); break; }
    }

    llama_sampler_free(smpl);
    LOGI("Generation done. result='%s'", result.c_str());
    return get_safe_utf8_string(env, result.c_str(), result.length());
}

extern "C" JNIEXPORT void JNICALL
Java_com_anurag_visionqa_ai_LlamaJNI_abortGeneration(JNIEnv*, jobject) {
    g_abort.store(true);
}

extern "C" JNIEXPORT void JNICALL
Java_com_anurag_visionqa_ai_LlamaJNI_freeModel(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> lock(g_mutex);
    free_all();
}