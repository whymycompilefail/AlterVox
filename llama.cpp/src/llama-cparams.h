#pragma once

#include "llama.h"

#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>

#define LLAMA_MAX_SEQ 256

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_ctx_seq;       // context for a single sequence
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_rs_seq;        // number of recurrent-state snapshots per seq for rollback
    uint32_t n_outputs_max;   // max outputs supported by the context
    uint32_t n_outputs_max_per_seq;
    int32_t  n_threads;       // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

    int32_t  nextn_layer_offset = 0;

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;

    bool embeddings;
    bool embeddings_nextn;        // also extract the hidden state before the final output norm
    bool embeddings_nextn_masked; // extract for only rows where batch.logits != 0
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool auto_fa;

    // [CPU-SPEC] adaptive flash attention: choose the attention path per decode step
    //   based on the current KV length instead of once at context creation.
    //   motivation: on CPU, the flash attention kernel degrades fast with context
    //   length, while the legacy path does not. see specs/fa-auto-adaptive.md
    bool     fa_adaptive      = false;
    int32_t  fa_kv_threshold  = 1024; // use FA while n_kv <= this, else legacy
    int32_t  fa_max_tokens    = 1;    // only switch away from FA when n_tokens <= this (prefill stays on FA)
    bool fused_gdn_ar;       // use fused gated delta net (autoregressive)
    bool fused_gdn_ch;       // use fused gated delta net (chunked)
    bool auto_fgdn;
    bool fused_lid;          // use fused lightning indexer
    bool auto_flid;
    bool fused_dsv4_hc_pre;
    bool fused_dsv4_hc_comb;
    bool fused_dsv4_hc_post;
    bool auto_fhc;
    bool no_perf;
    bool warmup;             // TODO: remove [TAG_LLAMA_GRAPH_NO_WARMUP]
    bool op_offload;
    bool kv_unified;
    bool pipeline_parallel;

    std::vector<bool> embeddings_layer_inp; // [n_layer()] extract input embeddings for layer

    enum llama_context_type ctx_type;
    enum llama_rope_scaling_type rope_scaling_type;
    enum llama_pooling_type pooling_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;

    llama_context * ctx_other;
};

// [CPU-SPEC] single source of truth for "should this graph use flash attention?".
//   n_kv     - number of KV entries participating in attention (context length)
//   n_tokens - number of query tokens in the current ubatch (1 == pure decode)
// keep in sync with the mask type chosen in llama-graph.cpp, otherwise the graph
// would be built with an f16 mask but a legacy (f32 mask) attention path, or vice versa.
static inline bool llama_fa_enabled(const llama_cparams & cparams, int64_t n_kv, int64_t n_tokens) {
    if (!cparams.flash_attn) {
        return false; // explicitly disabled
    }

    if (!cparams.fa_adaptive) {
        return true; // explicitly enabled, or not in adaptive mode
    }

    // never switch to the legacy path during prefill: it materializes the full
    // [n_kv, n_tokens] KQ matrix in f32, which blows up the compute buffer
    if (n_tokens > cparams.fa_max_tokens) {
        return true;
    }

    return n_kv <= cparams.fa_kv_threshold;
}

// [CPU-SPEC] Spec 03 §6: single source of truth for "may the legacy attention path consume a
//   quantized V cache through the MUL_MAT_TQ op?".
//
//   Upstream forbids a quantized V cache without flash attention, and §15.2 confirms the reason
//   is real: the legacy layout exposes V with the token index innermost, and a block-quantized V
//   fed to mul_mat that way produces garbage. Spec 03 removes the reason by storing V in the
//   flash-attn layout and transposing the loop inside a new op instead.
//
//   Default is OFF. The op is correct (§15.1) but it does not pay for itself: at ctx 8192 it
//   reaches 17.2 t/s against 25.5 t/s for the plain "quantized K + f16 V" configuration it was
//   meant to beat, so Spec 03's own gate G3 fails and the switch stays off until the kernel is
//   reworked. LLAMA_QV_OP=1 turns it on; all three call sites must read this one helper, or the
//   graph and the cache layout will disagree.
static inline bool llama_qv_op_enabled() {
    static const bool enabled = []() {
        const char * env = getenv("LLAMA_QV_OP");
        return env && atoi(env) != 0;
    }();
    return enabled;
}

// [CPU-SPEC] debugging aids. All off by default; they exist so the two findings in §15 can be
//   re-derived from the checked-out tree instead of taken on faith.
//
//   LLAMA_QV_LEGACY_LAYOUT=1: keep the upstream (token-innermost) V layout even when V is
//     quantized, so the legacy graph consumes it through plain ggml_mul_mat. Measured at
//     32.8 t/s - exactly the roofline the op was chasing - but the output is garbage (§15.2).
//     This is the number that makes the temptation concrete, so it is worth keeping reproducible.
//   LLAMA_QV_FORCE=1: route an f16 V through the TQ op as well. Nothing is dequantized then, so
//     TQ and upstream mul_mat move the same bytes and compute the same value - the gap that
//     remains is the kernel's own structure (§15.3: it is ~2x slower than mul_mat).
static inline bool llama_qv_legacy_layout() {
    static const bool enabled = []() {
        const char * env = getenv("LLAMA_QV_LEGACY_LAYOUT");
        return env && atoi(env) != 0;
    }();
    return enabled;
}

static inline bool llama_qv_force_layout() {
    static const bool enabled = []() {
        const char * env = getenv("LLAMA_QV_FORCE");
        return env && atoi(env) != 0;
    }();
    return enabled;
}
