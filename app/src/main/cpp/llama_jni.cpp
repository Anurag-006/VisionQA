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

static struct llama_model* g_model = nullptr;
static struct llama_context* g_ctx   = nullptr;
static struct mtmd_context* g_mtmd  = nullptr;
static std::mutex            g_mutex;
static std::atomic<bool>     g_abort{false};

// ── Persistent KV-cache state ─────────────────────────────────────────────────
// After the first call we save n_past at the point right after the image was
// fully decoded into the KV cache. On follow-up turns we restore to this
// position and only feed the new text history — skipping the expensive image
// re-encoding entirely.
static llama_pos g_n_past_after_image = -1;  // -1 = cache is cold

static void free_all() {
    if (g_mtmd)  { mtmd_free(g_mtmd);         g_mtmd  = nullptr; }
    if (g_ctx)   { llama_free(g_ctx);          g_ctx   = nullptr; }
    if (g_model) { llama_model_free(g_model);  g_model = nullptr; }
    g_n_past_after_image = -1;
    LOGI("All resources freed.");
}

static jstring make_error(JNIEnv* env, const char* fmt, ...) {
    char buf[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    LOGE("%s", buf);
    return env->NewStringUTF(buf);
}

static jstring get_safe_utf8_string(JNIEnv* env, const char* data, size_t length) {
    jbyteArray jBytes  = env->NewByteArray((jsize)length);
    env->SetByteArrayRegion(jBytes, 0, (jsize)length, reinterpret_cast<const jbyte*>(data));
    jclass    strClass = env->FindClass("java/lang/String");
    jmethodID strCtor  = env->GetMethodID(strClass, "<init>", "([BLjava/lang/String;)V");
    jstring   enc      = env->NewStringUTF("UTF-8");
    jstring   result   = (jstring)env->NewObject(strClass, strCtor, jBytes, enc);
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

    LOGI("Attempting to load Brain from: %s", m_path);
    LOGI("Attempting to load Eyes from: %s", v_path);

    try {
        // 1. Load model
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        mp.use_mmap = true;  // Keep this false for stability on Android
        mp.use_mlock = false;

        g_model = llama_model_load_from_file(m_path, mp);
        if (!g_model) { LOGE("❌ Model load failed (returned nullptr)."); goto fail; }

        // 2. Create context
        {
            llama_context_params cp = llama_context_default_params();
            cp.n_ctx           = (uint32_t)nCtx;
            cp.n_batch         = 512;
            cp.n_threads       = (uint32_t)nThreads;
            cp.n_threads_batch = (uint32_t)nThreads;
            g_ctx = llama_init_from_model(g_model, cp);
        }
        if (!g_ctx) { LOGE("❌ Context init failed."); goto fail; }

        // 3. Load vision encoder
        {
            mtmd_context_params mparams = mtmd_context_params_default();
            mparams.n_threads = (int32_t)nThreads;
            mparams.use_gpu   = false;
            g_mtmd = mtmd_init_from_file(v_path, g_model, mparams);
        }
        if (!g_mtmd) { LOGE("❌ MTMD init failed (returned nullptr)."); goto fail; }

        const llama_vocab* vocab = llama_model_get_vocab(g_model);
        LOGI("✅ Model ready | n_vocab=%d n_embd=%d ctx=%d",
             llama_vocab_n_tokens(vocab),
             llama_model_n_embd(g_model),
             nCtx);

    } catch (const std::exception& e) {
        LOGE("💥 CRITICAL C++ EXCEPTION: %s", e.what());
        goto fail;
    } catch (...) {
        LOGE("💥 CRITICAL UNKNOWN C++ EXCEPTION");
        goto fail;
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

// ── resetCache — call this when the user picks a NEW image ────────────────────
extern "C" JNIEXPORT void JNICALL
Java_com_anurag_visionqa_ai_LlamaJNI_resetCache(JNIEnv*, jobject) {
    std::lock_guard<std::mutex> lock(g_mutex);
    // FIX: Replaced with the new Memory API
    if (g_ctx) llama_memory_clear(llama_get_memory(g_ctx), true);
    g_n_past_after_image = -1;
    LOGI("KV cache reset. Next generate() will re-encode the image.");
}

// ── decode helpers ─────────────────────────────────────────────────────────────
static bool decode_text_batch(
        const llama_token* toks, size_t n_toks,
        llama_pos& n_past, bool need_logits)
{
    const size_t BATCH = 512;
    for (size_t off = 0; off < n_toks; off += BATCH) {
        size_t chunk = std::min(BATCH, n_toks - off);
        bool   last  = (off + chunk >= n_toks);

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
        jstring history,
        jint maxTokens,
        jobject tokenCallback)
{
    std::lock_guard<std::mutex> lock(g_mutex);

    if (!g_model || !g_ctx || !g_mtmd)
        return make_error(env, "❌ Engine not ready. Please restart the app.");
    if (imageWidth <= 0 || imageHeight <= 0)
        return make_error(env, "❌ Invalid image dimensions: %dx%d", imageWidth, imageHeight);

    const char* marker = mtmd_default_marker();
    if (!marker || strlen(marker) == 0)
        return make_error(env, "❌ Vision bridge marker empty. Incompatible library version.");

    g_abort.store(false);

    jclass    cb_class  = env->GetObjectClass(tokenCallback);
    jmethodID cb_invoke = env->GetMethodID(cb_class, "invoke",
                                           "(Ljava/lang/Object;)Ljava/lang/Object;");
    env->DeleteLocalRef(cb_class);

    const char* raw_q   = env->GetStringUTFChars(prompt,  nullptr);
    const char* raw_h   = env->GetStringUTFChars(history, nullptr);
    std::string question = (raw_q && strlen(raw_q) > 0) ? raw_q : "Describe this image in detail.";
    std::string hist_str = raw_h ? raw_h : "";
    env->ReleaseStringUTFChars(prompt,  raw_q);
    env->ReleaseStringUTFChars(history, raw_h);

    bool      is_first = (g_n_past_after_image < 0);
    llama_pos n_past   = 0;

    // ════════════════════════════════════════════════════════════════════════
    // PATH A — First turn: encode system prompt + image + first user question
    // ════════════════════════════════════════════════════════════════════════
    if (is_first) {
        LOGI("PATH A — first turn, encoding image %dx%d", (int)imageWidth, (int)imageHeight);

        jint* argb = env->GetIntArrayElements(pixelData, nullptr);
        jsize npx  = env->GetArrayLength(pixelData);
        std::vector<uint8_t> rgb((size_t)npx * 3);
        for (jsize i = 0; i < npx; i++) {
            uint32_t px = (uint32_t)argb[i];
            rgb[i*3+0]  = (px >> 16) & 0xFF;
            rgb[i*3+1]  = (px >>  8) & 0xFF;
            rgb[i*3+2]  =  px        & 0xFF;
        }
        env->ReleaseIntArrayElements(pixelData, argb, JNI_ABORT);

        mtmd_bitmap* bitmap = mtmd_bitmap_init(
                (uint32_t)imageWidth, (uint32_t)imageHeight, rgb.data());
        if (!bitmap)
            return make_error(env, "❌ Image processing failed (%dx%d).", imageWidth, imageHeight);

        std::string full_text =
                "<|im_start|>system\n"
                "You are an expert visual assistant. When shown an image, describe it "
                "in full detail: identify every visible object, its color, size, position, "
                "texture, and any text or branding. For follow-up questions, give complete "
                "and specific answers directly referencing what you see in the image. "
                "Never give one-word answers.<|im_end|>\n"
                "<|im_start|>user\n"
                + std::string(marker) + "\n"
                + question + "<|im_end|>\n"
                             "<|im_start|>assistant\n";

        mtmd_input_text itext = {
                full_text.c_str(), /*add_special=*/true, /*parse_special=*/true
        };
        mtmd_input_chunks* chunks          = mtmd_input_chunks_init();
        const mtmd_bitmap* bitmaps_list[1]    = { bitmap };

        int tok_ret = mtmd_tokenize(g_mtmd, chunks, &itext, bitmaps_list, 1);
        if (tok_ret != 0) {
            mtmd_bitmap_free(bitmap);
            mtmd_input_chunks_free(chunks);
            return make_error(env,
                              "❌ Tokenization failed (err=%d). ctx=%d img=%dx%d prompt_len=%zu.",
                              tok_ret, (int)llama_n_ctx(g_ctx),
                              (int)imageWidth, (int)imageHeight, full_text.size());
        }

        // FIX: Replaced with the new Memory API
        llama_memory_clear(llama_get_memory(g_ctx), true);
        n_past = 0;

        bool        ok  = true;
        std::string err;
        size_t      nc  = mtmd_input_chunks_size(chunks);

        for (size_t i = 0; i < nc && ok; i++) {
            const mtmd_input_chunk* ch      = mtmd_input_chunks_get(chunks, i);
            bool                    is_last = (i == nc - 1);

            if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                size_t n_img = mtmd_input_chunk_get_n_tokens(ch);
                if (n_past + (llama_pos)n_img > (llama_pos)llama_n_ctx(g_ctx)) {
                    char tmp[200];
                    snprintf(tmp, sizeof(tmp),
                             "❌ Context too small: need %zu image tokens, only %d free. ctx=%d",
                             n_img, (int)(llama_n_ctx(g_ctx)-n_past), (int)llama_n_ctx(g_ctx));
                    err = tmp; ok = false; break;
                }
                if (mtmd_encode_chunk(g_mtmd, ch) != 0) {
                    err = "❌ Vision encoding failed. Try a different image."; ok = false; break;
                }
                float* embd   = mtmd_get_output_embd(g_mtmd);
                int32_t n_embd = llama_model_n_embd_inp(g_model);
                ok = decode_image_batch(embd, n_img, n_embd, n_past, is_last);
                if (!ok) err = "❌ Image prefill decode failed.";
            } else {
                size_t             nt   = 0;
                const llama_token* toks = mtmd_input_chunk_get_tokens_text(ch, &nt);
                if (nt == 0) continue;
                if (n_past + (llama_pos)nt > (llama_pos)llama_n_ctx(g_ctx)) {
                    err = "❌ Context overflow on text. Try a shorter question."; ok = false; break;
                }
                ok = decode_text_batch(toks, nt, n_past, is_last);
                if (!ok) err = "❌ Text prefill decode failed.";
            }
        }

        mtmd_bitmap_free(bitmap);
        mtmd_input_chunks_free(chunks);
        if (!ok) return get_safe_utf8_string(env, err.c_str(), err.size());

        g_n_past_after_image = n_past;
        LOGI("PATH A prefill done. n_past=%d saved as image anchor.", (int)n_past);

        // ════════════════════════════════════════════════════════════════════════
        // PATH B — Follow-up turn: rewind KV cache, feed text history only
        // ════════════════════════════════════════════════════════════════════════
    } else {
        LOGI("PATH B — follow-up, reusing cache anchor at n_past=%d", (int)g_n_past_after_image);

        n_past = g_n_past_after_image;

        if (!llama_memory_seq_rm(llama_get_memory(g_ctx), 0, n_past, -1)) {
            LOGE("llama_memory_seq_rm failed — falling back to full clear");
            g_n_past_after_image = -1;
            return make_error(env, "❌ Cache rewind failed. Please send your question again.");
        }

        std::string continuation =
                hist_str
                + "<|im_start|>user\n"
                + question + "<|im_end|>\n"
                             "<|im_start|>assistant\n";

        LOGI("PATH B continuation len=%zu, n_past_start=%d",
             continuation.size(), (int)n_past);

        const llama_vocab* vocab_b = llama_model_get_vocab(g_model);
        std::vector<llama_token> toks(continuation.size() + 64);
        int ntok = llama_tokenize(
                vocab_b,
                continuation.c_str(), (int32_t)continuation.size(),
                toks.data(), (int32_t)toks.size(),
                /*add_special=*/false,  // BOS already baked in from PATH A
                /*parse_special=*/true
        );
        if (ntok < 0) {
            toks.resize(-ntok + 16);
            ntok = llama_tokenize(vocab_b,
                                  continuation.c_str(), (int32_t)continuation.size(),
                                  toks.data(), (int32_t)toks.size(), false, true);
        }
        if (ntok <= 0)
            return make_error(env, "❌ Follow-up tokenization failed (n=%d).", ntok);

        if (n_past + (llama_pos)ntok > (llama_pos)llama_n_ctx(g_ctx))
            return make_error(env,
                              "❌ Context full (%d/%d tokens used). Please start a new conversation.",
                              (int)(n_past + ntok), (int)llama_n_ctx(g_ctx));

        if (!decode_text_batch(toks.data(), (size_t)ntok, n_past, true))
            return make_error(env, "❌ Follow-up prefill failed.");

        LOGI("PATH B prefill done. n_past=%d", (int)n_past);
    }

    // ── Generation loop (identical for both paths) ────────────────────────────
    const llama_vocab* vocab = llama_model_get_vocab(g_model);

    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(sparams);

    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(64, 1.05f, 0.0f, 0.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist((uint32_t)-1));

    std::string result;
    result.reserve(1024);

    for (int step = 0; step < (int)maxTokens; step++) {
        if (g_abort.load() || n_past >= (llama_pos)llama_n_ctx(g_ctx)) break;

        llama_token id = llama_sampler_sample(smpl, g_ctx, -1);
        llama_sampler_accept(smpl, id);

        if (llama_vocab_is_eog(vocab, id)) { LOGI("EOS at step %d", step); break; }

        char piece[256] = {};
        int  plen = llama_token_to_piece(vocab, id, piece, sizeof(piece), 0, true);
        if (plen <= 0) { LOGE("Bad piece at step %d id=%d", step, id); break; }

        result.append(piece, (size_t)plen);

        jbyteArray jb = env->NewByteArray(plen);
        env->SetByteArrayRegion(jb, 0, plen, (const jbyte*)piece);
        env->CallObjectMethod(tokenCallback, cb_invoke, jb);
        env->DeleteLocalRef(jb);
        if (env->ExceptionCheck()) env->ExceptionClear();

        llama_batch nb  = llama_batch_init(1, 0, 1);
        nb.n_tokens     = 1;
        nb.token[0]     = id;
        nb.pos[0]       = n_past++;
        nb.n_seq_id[0]  = 1;
        nb.seq_id[0][0] = 0;
        nb.logits[0]    = 1;
        int ret = llama_decode(g_ctx, nb);
        llama_batch_free(nb);
        if (ret != 0) { LOGE("llama_decode failed at step %d ret=%d", step, ret); break; }
    }

    llama_sampler_free(smpl);
    LOGI("Generation done. %zu chars. final n_past=%d", result.size(), (int)n_past);
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