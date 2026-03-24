#include "models.h"

ggml_cgraph * clip_graph_qwen3a::build() {
    ggml_tensor * inp = build_inp_raw(1);

    int64_t chunk_size = 100;
    int64_t last_chunk_size_before_cnn = inp->ne[0] % chunk_size;
    int64_t last_chunk_size_after_cnn = (((last_chunk_size_before_cnn + 1) / 2 + 1) / 2 + 1) / 2;
    int64_t last_chunk_padding_before_cnn = (chunk_size - last_chunk_size_before_cnn) % chunk_size;

    int64_t n_full_chunks = inp->ne[0] / chunk_size;
    int64_t n_chunks = n_full_chunks + !!last_chunk_size_before_cnn;
    int64_t n_samples_post_cnn = n_full_chunks * 13 + last_chunk_size_after_cnn;

    // Here we pad and split the samples into chunks, along the time dimension.
    // For example, a tensor of 420 samples, has shape [time=420, frames=128, channels=1] before the split,
    // and [time=100, frames=128, channels=1, chunks=5] after the split.
    inp = ggml_pad(ctx0, inp, last_chunk_padding_before_cnn, 0, 0, 0);
    inp = ggml_view_4d(ctx0, inp,
        chunk_size,
        inp->ne[1],
        1,
        n_chunks,
        inp->nb[1],
        inp->nb[2],
        inp->nb[0] * chunk_size,
        0
    );
    inp = ggml_cont(ctx0, inp);

    inp = ggml_conv_2d(ctx0, model.conv2d_1_w, inp, 2, 2, 1, 1, 1, 1);
    inp = ggml_add(ctx0, inp, model.conv2d_1_b);
    inp = ggml_gelu_erf(ctx0, inp);

    inp = ggml_conv_2d(ctx0, model.conv2d_2_w, inp, 2, 2, 1, 1, 1, 1);
    inp = ggml_add(ctx0, inp, model.conv2d_2_b);
    inp = ggml_gelu_erf(ctx0, inp);

    inp = ggml_conv_2d(ctx0, model.conv2d_3_w, inp, 2, 2, 1, 1, 1, 1);
    inp = ggml_add(ctx0, inp, model.conv2d_3_b);
    inp = ggml_gelu_erf(ctx0, inp);

    // Shape at this point: [time, frames, channels, chunks]
    inp = ggml_permute(ctx0, inp, 2, 0, 1, 3);
    // Shape at this point: [frames, channels, time, chunks]
    inp = ggml_cont(ctx0, inp);
    inp = ggml_reshape_3d(ctx0, inp, inp->ne[0] * inp->ne[1], inp->ne[2], inp->ne[3]);
    // Shape at this point: [frames * channels, time, chunks]

    inp = ggml_mul_mat(ctx0, model.conv_out_w, inp);
    if (model.conv_out_b) {
        inp = ggml_add(ctx0, inp, model.conv_out_b);
    }
    // Shape at this point: [features, time, chunks]
    // Concat the chunks back into a single long array and trim the padding from the last chunk, if there's any.
    inp = ggml_reshape_2d(ctx0, inp, inp->ne[0], inp->ne[1] * inp->ne[2]);
    inp = ggml_view_2d(ctx0, inp,
        inp->ne[0],
        n_samples_post_cnn,
        inp->nb[1],
        0);
    // Shape at this point: [features, time]

    auto n_pos = inp->ne[1];

    ggml_tensor * pos_embd_selected = ggml_view_2d(
        ctx0, model.position_embeddings,
        model.position_embeddings->ne[0], n_pos,
        model.position_embeddings->nb[1], 0
    );
    ggml_tensor * cur = build_vit(
                            inp, n_pos,
                            NORM_TYPE_NORMAL,
                            hparams.ffn_op,
                            pos_embd_selected,
                            nullptr);

    cur = build_ffn(cur,
        model.mm_1_w, model.mm_1_b,
        nullptr, nullptr,
        model.mm_2_w, model.mm_2_b,
        FFN_GELU_ERF,
        -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}
