/*
 * JVID video encoder foundation
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * JVID video encoder foundation
 */

#include <limits.h>
#include <stdint.h>
#include <string.h>

#include "libavutil/imgutils.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/timestamp.h"
#include "libavutil/hwcontext.h"

#include "avcodec.h"
#include "codec_internal.h"
#include "encode.h"
#include "zlib_wrapper.h"

#define JVID_HEADER_SIZE         16
#define JVID_VERSION_1            1
#define JVID_FRAME_RAW_I420       1
#define JVID_FRAME_BLOCK_I420     2
#define JVID_FLAG_DEFLATE         1

#define JVID_BLOCK_SKIP           0
#define JVID_BLOCK_RAW            1
#define JVID_BLOCK_RLE            2

typedef struct JVIDEncContext {
    const AVClass *class;
    FFZStream zstream;
    uint8_t *payload;
    size_t payload_alloc;
    uint8_t *prev_frame;
    size_t prev_frame_size;
    int has_prev_frame;
    int block_size;
    int quant_shift;
    int change_threshold;
} JVIDEncContext;

static uint8_t jvid_quantize_sample(uint8_t sample, int shift)
{
    if (!shift)
        return sample;

    return (uint8_t)((sample >> shift) << shift);
}

static int jvid_ensure_buffer(uint8_t **buf, size_t *alloc, size_t size)
{
    uint8_t *tmp;

    if (size <= *alloc)
        return 0;

    tmp = av_realloc(*buf, size);
    if (!tmp)
        return AVERROR(ENOMEM);

    *buf = tmp;
    *alloc = size;
    return 0;
}

static int jvid_write_byte(uint8_t **dst, uint8_t *end, uint8_t value)
{
    if (*dst >= end)
        return AVERROR(ENOSPC);
    *(*dst)++ = value;
    return 0;
}

static int jvid_emit_raw_block(const uint8_t *src, ptrdiff_t linesize,
                               uint8_t *scratch, int bw, int bh, int shift,
                               uint8_t **dst, uint8_t *end, uint8_t *recon)
{
    int pos = 0;
    int ret;

    ret = jvid_write_byte(dst, end, JVID_BLOCK_RAW);
    if (ret < 0)
        return ret;

    for (int y = 0; y < bh; y++) {
        for (int x = 0; x < bw; x++) {
            uint8_t q = jvid_quantize_sample(src[y * linesize + x], shift);
            scratch[pos] = q;
            recon[pos++] = q;
        }
    }

    if (end - *dst < pos)
        return AVERROR(ENOSPC);

    memcpy(*dst, scratch, pos);
    *dst += pos;
    return 0;
}

static int jvid_emit_rle_block(const uint8_t *src, ptrdiff_t linesize,
                               uint8_t *scratch, int bw, int bh, int shift,
                               uint8_t **dst, uint8_t *end, uint8_t *recon)
{
    int pixels = bw * bh;
    int scratch_len = 0;
    int pos = 0;
    int ret;

    for (int y = 0; y < bh; y++) {
        for (int x = 0; x < bw; x++) {
            uint8_t q = jvid_quantize_sample(src[y * linesize + x], shift);
            recon[pos] = q;

            if (!pos || q != scratch[scratch_len - 1] || scratch[scratch_len - 2] == 255) {
                scratch[scratch_len++] = 1;
                scratch[scratch_len++] = q;
            } else {
                scratch[scratch_len - 2]++;
            }
            pos++;
        }
    }

    if (scratch_len >= pixels)
        return jvid_emit_raw_block(src, linesize, scratch, bw, bh, shift, dst, end, recon);

    ret = jvid_write_byte(dst, end, JVID_BLOCK_RLE);
    if (ret < 0)
        return ret;
    if (end - *dst < scratch_len)
        return AVERROR(ENOSPC);

    memcpy(*dst, scratch, scratch_len);
    *dst += scratch_len;
    return 0;
}

static int jvid_plane_has_change(const uint8_t *src, ptrdiff_t src_linesize,
                                 const uint8_t *prev, int plane_offset,
                                 int plane_stride, int x0, int y0,
                                 int bw, int bh, int threshold)
{
    for (int y = 0; y < bh; y++) {
        const uint8_t *row = src + (y0 + y) * src_linesize + x0;
        const uint8_t *prev_row = prev + plane_offset + (y0 + y) * plane_stride + x0;

        for (int x = 0; x < bw; x++) {
            int diff = row[x] - prev_row[x];
            if (diff < 0)
                diff = -diff;
            if (diff > threshold)
                return 1;
        }
    }

    return 0;
}

static void jvid_store_prev_plane(uint8_t *prev, int plane_offset, int plane_stride,
                                  int x0, int y0, int bw, int bh,
                                  const uint8_t *recon)
{
    for (int y = 0; y < bh; y++)
        memcpy(prev + plane_offset + (y0 + y) * plane_stride + x0,
               recon + y * bw, bw);
}

static int jvid_encode_plane(JVIDEncContext *s, const uint8_t *src,
                             ptrdiff_t src_linesize, int plane_offset,
                             int plane_w, int plane_h, int plane_stride,
                             uint8_t **dst, uint8_t *end, int have_prev,
                             uint8_t *scratch, uint8_t *recon, int *used_skip)
{
    int block = s->block_size;

    for (int y0 = 0; y0 < plane_h; y0 += block) {
        int bh = FFMIN(block, plane_h - y0);

        for (int x0 = 0; x0 < plane_w; x0 += block) {
            int bw = FFMIN(block, plane_w - x0);
            int changed = !have_prev ||
                          jvid_plane_has_change(src, src_linesize, s->prev_frame,
                                                plane_offset, plane_stride,
                                                x0, y0, bw, bh, s->change_threshold);
            int ret;

            if (!changed) {
                ret = jvid_write_byte(dst, end, JVID_BLOCK_SKIP);
                if (ret < 0)
                    return ret;
                *used_skip = 1;
                continue;
            }

            ret = jvid_emit_rle_block(src + y0 * src_linesize + x0, src_linesize,
                                      scratch, bw, bh, s->quant_shift, dst, end, recon);
            if (ret < 0)
                return ret;

            jvid_store_prev_plane(s->prev_frame, plane_offset, plane_stride,
                                  x0, y0, bw, bh, recon);
        }
    }

    return 0;
}

static av_cold int jvid_encode_init(AVCodecContext *avctx)
{
    JVIDEncContext *s = avctx->priv_data;
    int ret;

    if (!avctx->width || !avctx->height) {
        av_log(avctx, AV_LOG_ERROR, "Invalid JVID dimensions %dx%d.\n",
               avctx->width, avctx->height);
        return AVERROR(EINVAL);
    }

    if (avctx->width > UINT16_MAX || avctx->height > UINT16_MAX) {
        av_log(avctx, AV_LOG_ERROR,
               "JVID version 1 supports dimensions up to 65535x65535.\n");
        return AVERROR(EINVAL);
    }

    ret = av_image_check_size2(avctx->width, avctx->height,
                               avctx->max_pixels ? avctx->max_pixels : INT64_MAX,
                               avctx->sw_pix_fmt != AV_PIX_FMT_NONE ? avctx->sw_pix_fmt : avctx->pix_fmt,
                               0, avctx);
    if (ret < 0)
        return ret;

    if (avctx->pix_fmt == AV_PIX_FMT_CUDA) {
        AVHWFramesContext *frames_ctx;

        if (!avctx->hw_frames_ctx) {
            av_log(avctx, AV_LOG_ERROR,
                   "JVID CUDA encode requires hw_frames_ctx.\n");
            return AVERROR(EINVAL);
        }

        frames_ctx = (AVHWFramesContext *)avctx->hw_frames_ctx->data;
        if (frames_ctx->format != AV_PIX_FMT_CUDA ||
            frames_ctx->sw_format != AV_PIX_FMT_YUV420P) {
            av_log(avctx, AV_LOG_ERROR,
                   "JVID CUDA encode supports CUDA frames backed by YUV420P only.\n");
            return AVERROR(EINVAL);
        }

        avctx->sw_pix_fmt = AV_PIX_FMT_YUV420P;
    } else if (avctx->pix_fmt != AV_PIX_FMT_YUV420P) {
        av_log(avctx, AV_LOG_ERROR,
               "JVID encodes YUV420P or CUDA(YUV420P) video only.\n");
        return AVERROR(EINVAL);
    }

    if (s->block_size < 1 || s->block_size > 255 ||
        s->quant_shift < 0 || s->quant_shift > 7 ||
        s->change_threshold < 0 || s->change_threshold > 255)
        return AVERROR(EINVAL);

    avctx->bits_per_raw_sample = 8;
    ret = ff_deflate_init(&s->zstream, avctx->compression_level > 0 ?
                                      avctx->compression_level : 6, avctx);
    if (ret < 0)
        return ret;

    av_log(avctx, AV_LOG_DEBUG,
           "JVID init %dx%d pix_fmt=%d block=%d quant_shift=%d threshold=%d\n",
           avctx->width, avctx->height, avctx->pix_fmt,
           s->block_size, s->quant_shift, s->change_threshold);
    return 0;
}

static av_cold int jvid_encode_close(AVCodecContext *avctx)
{
    JVIDEncContext *s = avctx->priv_data;

    ff_deflate_end(&s->zstream);
    av_freep(&s->payload);
    av_freep(&s->prev_frame);
    s->payload_alloc = 0;
    s->prev_frame_size = 0;
    s->has_prev_frame = 0;
    return 0;
}

static int jvid_encode_frame(AVCodecContext *avctx, AVPacket *pkt,
                             const AVFrame *frame, int *got_packet)
{
    JVIDEncContext *s = avctx->priv_data;
    z_stream *zstream = &s->zstream.zstream;
    const AVFrame *src_frame = frame;
    int y_size, uv_w, uv_h, uv_size, payload_size, packet_size;
    int y_stride, uv_stride, have_prev;
    uint8_t *payload, *payload_end;
    uint8_t *recon, *scratch;
    int used_skip = 0;
    int zret, ret;
    AVFrame *sw_frame = NULL;

    if (!frame) {
        av_log(avctx, AV_LOG_DEBUG, "JVID flush frame\n");
        *got_packet = 0;
        return 0;
    }

    if (frame->format == AV_PIX_FMT_CUDA) {
        sw_frame = av_frame_alloc();
        if (!sw_frame)
            return AVERROR(ENOMEM);

        sw_frame->format = AV_PIX_FMT_YUV420P;
        sw_frame->width  = frame->width;
        sw_frame->height = frame->height;
        sw_frame->pts    = frame->pts;

        ret = av_frame_get_buffer(sw_frame, 0);
        if (ret < 0)
            goto fail;

        ret = av_hwframe_transfer_data(sw_frame, frame, 0);
        if (ret < 0) {
            av_log(avctx, AV_LOG_ERROR, "Failed to transfer CUDA frame to system memory.\n");
            goto fail;
        }

        src_frame = sw_frame;
    }

    av_log(avctx, AV_LOG_DEBUG,
           "JVID encode frame %dx%d format=%d linesize=%d pts=%s\n",
           src_frame->width, src_frame->height, src_frame->format, src_frame->linesize[0],
           av_ts2str(src_frame->pts));

    if (src_frame->format != AV_PIX_FMT_YUV420P) {
        av_log(avctx, AV_LOG_ERROR,
               "JVID encodes color video as YUV420P only.\n");
        ret = AVERROR(EINVAL);
        goto fail;
    }
    if (src_frame->width != avctx->width || src_frame->height != avctx->height) {
        ret = AVERROR(EINVAL);
        goto fail;
    }
    if (!src_frame->data[0] || !src_frame->data[1] || !src_frame->data[2] ||
        src_frame->linesize[0] <= 0 || src_frame->linesize[1] <= 0 || src_frame->linesize[2] <= 0) {
        ret = AVERROR(EINVAL);
        goto fail;
    }

    y_size = avctx->width * avctx->height;
    uv_w = AV_CEIL_RSHIFT(avctx->width, 1);
    uv_h = AV_CEIL_RSHIFT(avctx->height, 1);
    uv_size = uv_w * uv_h;
    payload_size = y_size + 2 * uv_size;
    y_stride = avctx->width;
    uv_stride = uv_w;

    ret = jvid_ensure_buffer(&s->prev_frame, &s->prev_frame_size, payload_size);
    if (ret < 0)
        return ret;

    ret = jvid_ensure_buffer(&s->payload, &s->payload_alloc,
                             payload_size * 3 + 1024);
    if (ret < 0)
        return ret;

    have_prev = s->has_prev_frame;
    payload = s->payload;
    payload_end = s->payload + s->payload_alloc;
    scratch = payload_end - (s->block_size * s->block_size * 2);
    recon = scratch - (s->block_size * s->block_size);

    if (recon < payload)
        return AVERROR(ENOMEM);

    payload[0] = s->block_size;
    payload[1] = s->quant_shift;
    payload[2] = s->change_threshold;
    payload[3] = 0;
    payload += 4;

    ret = jvid_encode_plane(s, src_frame->data[0], src_frame->linesize[0], 0,
                            avctx->width, avctx->height, y_stride,
                            &payload, payload_end - (s->block_size * s->block_size * 3),
                            have_prev, scratch, recon, &used_skip);
    if (ret < 0)
        goto fail;
    ret = jvid_encode_plane(s, src_frame->data[1], src_frame->linesize[1], y_size,
                            uv_w, uv_h, uv_stride,
                            &payload, payload_end - (s->block_size * s->block_size * 3),
                            have_prev, scratch, recon, &used_skip);
    if (ret < 0)
        goto fail;
    ret = jvid_encode_plane(s, src_frame->data[2], src_frame->linesize[2], y_size + uv_size,
                            uv_w, uv_h, uv_stride,
                            &payload, payload_end - (s->block_size * s->block_size * 3),
                            have_prev, scratch, recon, &used_skip);
    if (ret < 0)
        goto fail;

    payload_size = payload - s->payload;
    packet_size = deflateBound(zstream, payload_size);
    if (packet_size < 0 || packet_size > INT_MAX - JVID_HEADER_SIZE)
        return AVERROR(EINVAL);

    ret = ff_get_encode_buffer(avctx, pkt, JVID_HEADER_SIZE + packet_size, 0);
    if (ret < 0)
        return ret;

    pkt->data[0]  = 'J';
    pkt->data[1]  = 'V';
    pkt->data[2]  = 'I';
    pkt->data[3]  = 'D';
    pkt->data[4]  = JVID_VERSION_1;
    pkt->data[5]  = JVID_FRAME_BLOCK_I420;
    pkt->data[6]  = JVID_FLAG_DEFLATE;
    pkt->data[7]  = JVID_HEADER_SIZE;
    pkt->data[8]  = avctx->width & 0xFF;
    pkt->data[9]  = avctx->width >> 8;
    pkt->data[10] = avctx->height & 0xFF;
    pkt->data[11] = avctx->height >> 8;
    pkt->data[12] = s->block_size;
    pkt->data[13] = s->quant_shift;
    pkt->data[14] = s->change_threshold;
    pkt->data[15] = 0;

    zret = deflateReset(zstream);
    if (zret != Z_OK)
        return AVERROR_EXTERNAL;

    zstream->next_in   = s->payload;
    zstream->avail_in  = payload_size;
    zstream->next_out  = pkt->data + JVID_HEADER_SIZE;
    zstream->avail_out = packet_size;
    zret = deflate(zstream, Z_FINISH);
    if (zret != Z_STREAM_END)
        return AVERROR_EXTERNAL;

    pkt->size = JVID_HEADER_SIZE + zstream->total_out;
    pkt->flags = used_skip ? 0 : AV_PKT_FLAG_KEY;
    s->has_prev_frame = 1;
    *got_packet = 1;

    av_log(avctx, AV_LOG_DEBUG, "JVID produced packet size=%d payload=%d\n",
           pkt->size, payload_size);
    av_frame_free(&sw_frame);
    return 0;

fail:
    av_frame_free(&sw_frame);
    return ret;
}

#define OFFSET(x) offsetof(JVIDEncContext, x)
#define VE AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_ENCODING_PARAM

static const AVOption jvid_options[] = {
    { "block_size", "Changed-region block size", OFFSET(block_size), AV_OPT_TYPE_INT, { .i64 = 16 }, 4, 64, VE },
    { "quant_shift", "Lossy quantization right shift", OFFSET(quant_shift), AV_OPT_TYPE_INT, { .i64 = 2 }, 0, 7, VE },
    { "change_threshold", "Per-sample delta threshold for block skip", OFFSET(change_threshold), AV_OPT_TYPE_INT, { .i64 = 4 }, 0, 255, VE },
    { NULL },
};

static const AVClass jvid_encoder_class = {
    .class_name = "jvid encoder",
    .item_name  = av_default_item_name,
    .option     = jvid_options,
    .version    = LIBAVUTIL_VERSION_INT,
};

const FFCodec ff_jvid_encoder = {
    .p.name         = "jvid",
    CODEC_LONG_NAME("JVID video"),
    .p.type         = AVMEDIA_TYPE_VIDEO,
    .p.id           = AV_CODEC_ID_JVID,
    .priv_data_size = sizeof(JVIDEncContext),
    .p.priv_class   = &jvid_encoder_class,
    .p.capabilities = AV_CODEC_CAP_ENCODER_REORDERED_OPAQUE,
    .init           = jvid_encode_init,
    .close          = jvid_encode_close,
    FF_CODEC_ENCODE_CB(jvid_encode_frame),
    CODEC_PIXFMTS(AV_PIX_FMT_YUV420P, AV_PIX_FMT_CUDA),
    .caps_internal  = FF_CODEC_CAP_INIT_CLEANUP,
};
