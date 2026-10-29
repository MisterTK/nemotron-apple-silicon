// Profiling aid used for the Bolt decode breakdown (not part of the optimization patch).
// Paste into ggml_metal_op_encode_impl() in ggml/src/ggml-metal/ggml-metal-ops.cpp, right after the
// GGML_TENSOR_FLAG_COMPUTE check. GGML_METAL_SKIP=MUL_MAT_ID,MUL_MAT:iq2_xxs,... then skips encoding the
// matching nodes (outputs become garbage); the tok/s delta in llama-bench is that op's real in-graph cost.

    // TMP profiling: GGML_METAL_SKIP=OP[:src0type],... skips encoding matching nodes
    {
        static const char * skip = getenv("GGML_METAL_SKIP");
        if (skip) {
            char key[64];
            snprintf(key, sizeof(key), "%s:%s", ggml_op_name(node->op), node->src[0] ? ggml_type_name(node->src[0]->type) : "");
            const size_t nop = strlen(ggml_op_name(node->op));
            for (const char * p = skip; *p; ) {
                const char * e = strchr(p, ',');
                const size_t n = e ? (size_t)(e - p) : strlen(p);
                if ((n == nop || n == strlen(key)) && strncmp(p, key, n) == 0) {
                    return 1;
                }
                p += n + (e ? 1 : 0);
            }
        }
    }

