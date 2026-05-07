/*
 * JVID video decoder foundation
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
 * JVID video decoder foundation
 */

#include <stdint.h>
#include <string.h>

#include "libavutil/frame.h"
#include "libavutil/hwcontext.h"
#include "libavutil/imgutils.h"
#include "libavutil/intreadwrite.h"
#include "libavutil/mem.h"
#include "avcodec.h"
#include "codec_internal.h"
#include "decode.h"
#include "hwconfig.h"
#include "zlib_wrapper.h"

#define JVID_MIN_HEADER_SIZE    12
#define JVID_HEADER_SIZE        16
#define JVID_VERSION_1           1
#define JVID_FLAG_DEFLATE        1

#define JVID_BLOCK_SKIP          0
#define JVID_BLOCK_RAW           1
#define JVID_BLOCK_RLE           2

enum JVIDFrameType {
    JVID_FRAME_RAW_I = 0,
    JVID_FRAME_RAW_I420 = 1,
    JVID_FRAME_BLOCK_I420 = 2,
};

typedef struct JVIDContext {
    unsigned version;
    int width;
    int height;
    FFZStream zstream;
    uint8_t *inflated;
    size_t inflated_alloc;
    uint8_t *prev_frame;
    size_t prev_frame_alloc;
    int has_prev_frame;
} JVIDContext;

typedef struct JVIDHeader {
    unsigned version;
    unsigned frame_type;
    unsigned flags;
    unsigned header_size;
    int width;
    int height;
    unsigned block_size;
    unsigned quant_shift;
    unsigned change_threshold;
} JVIDHeader;

static int jvid_ensure_buffer(uint8_t **buf, size_t *alloc, size_t size)
{
    uint8_t *tmp;

    if (size <= *alloc)
        return 0;

    tmp = av_realloc(*buf, size + AV_INPUT_BUFFER_PADDING_SIZE);
    if (!tmp)
        return AVERROR(ENOMEM);

    *buf = tmp;
    *alloc = size;
    return 0;
}

static int jvid_decode_init(AVCodecContext *avctx)
{
    JVIDContext *s = avctx->priv_data;

    return ff_inflate_init(&s->zstream, avctx);
}

static av_cold int jvid_decode_close(AVCodecContext *avctx)
{
    JVIDContext *s = avctx->priv_data;

    ff_inflate_end(&s->zstream);
    av_freep(&s->inflated);
    av_freep(&s->prev_frame);
    s->inflated_alloc = 0;
    s->prev_frame_alloc = 0;
    s->has_prev_frame = 0;
    return 0;
}

static int jvid_parse_header(AVCodecContext *avctx, JVIDHeader *hdr,
                             const uint8_t *buf, int buf_size)
{
    if (buf_size < JVID_MIN_HEADER_SIZE) {
        av_log(avctx, AV_LOG_ERROR, "Packet too small for JVID header.\n");
        return AVERROR_INVALIDDATA;
    }

    if (buf[0] != 'J' || buf[1] != 'V' || buf[2] != 'I' || buf[3] != 'D') {
        av_log(avctx, AV_LOG_ERROR, "Invalid JVID packet magic.\n");
        return AVERROR_INVALIDDATA;
    }

    hdr->version          = buf[4];
    hdr->frame_type       = buf[5];
    hdr->flags            = buf[6];
    hdr->header_size      = buf[7];
    hdr->width            = AV_RL16(buf + 8);
    hdr->height           = AV_RL16(buf + 10);
    hdr->block_size       = hdr->header_size >= JVID_HEADER_SIZE ? buf[12] : 0;
    hdr->quant_shift      = hdr->header_size >= JVID_HEADER_SIZE ? buf[13] : 0;
    hdr->change_threshold = hdr->header_size >= JVID_HEADER_SIZE ? buf[14] : 0;

    if (hdr->version != JVID_VERSION_1) {
        av_log(avctx, AV_LOG_ERROR, "Unsupported JVID version %u.\n",
               hdr->version);
        return AVERROR_PATCHWELCOME;
    }

    if (hdr->header_size < JVID_MIN_HEADER_SIZE || hdr->header_size > (unsigned)buf_size) {
        av_log(avctx, AV_LOG_ERROR, "Invalid JVID header size %u.\n",
               hdr->header_size);
        return AVERROR_INVALIDDATA;
    }

    if (hdr->frame_type == JVID_FRAME_BLOCK_I420 &&
        (hdr->header_size < JVID_HEADER_SIZE || !hdr->block_size || hdr->quant_shift > 7)) {
        av_log(avctx, AV_LOG_ERROR, "Invalid block-coded JVID header.\n");
        return AVERROR_INVALIDDATA;
    }

    return 0;
}

static int jvid_copy_prev_block(uint8_t *dst, ptrdiff_t dst_linesize,
                                const uint8_t *prev, int plane_offset,
                                int plane_stride, int x0, int y0,
                                int bw, int bh)
{
    if (!prev)
        return AVERROR_INVALIDDATA;

    for (int y = 0; y < bh; y++)
        memcpy(dst + (y0 + y) * dst_linesize + x0,
               prev + plane_offset + (y0 + y) * plane_stride + x0, bw);

    return 0;
}

static void jvid_store_prev_block(uint8_t *prev, int plane_offset, int plane_stride,
                                  const uint8_t *src, ptrdiff_t src_linesize,
                                  int x0, int y0, int bw, int bh)
{
    for (int y = 0; y < bh; y++)
        memcpy(prev + plane_offset + (y0 + y) * plane_stride + x0,
               src + (y0 + y) * src_linesize + x0, bw);
}

static int jvid_decode_plane_blocks(AVCodecContext *avctx, JVIDContext *s,
                                    uint8_t **psrc, const uint8_t *src_end,
                                    uint8_t *dst, ptrdiff_t dst_linesize,
                                    int plane_offset, int plane_w, int plane_h,
                                    int plane_stride, int block_size, int have_prev)
{
    uint8_t *src = *psrc;

    for (int y0 = 0; y0 < plane_h; y0 += block_size) {
        int bh = FFMIN(block_size, plane_h - y0);

        for (int x0 = 0; x0 < plane_w; x0 += block_size) {
            int bw = FFMIN(block_size, plane_w - x0);
            int pixels = bw * bh;
            int tag;

            if (src >= src_end)
                return AVERROR_INVALIDDATA;

            tag = *src++;
            switch (tag) {
            case JVID_BLOCK_SKIP: {
                int ret;

                if (!have_prev) {
                    av_log(avctx, AV_LOG_ERROR, "JVID skip block without reference frame.\n");
                    return AVERROR_INVALIDDATA;
                }

                ret = jvid_copy_prev_block(dst, dst_linesize, s->prev_frame,
                                           plane_offset, plane_stride,
                                           x0, y0, bw, bh);
                if (ret < 0)
                    return ret;
                break;
            }
            case JVID_BLOCK_RAW:
                if (src_end - src < pixels)
                    return AVERROR_INVALIDDATA;
                for (int y = 0; y < bh; y++)
                    memcpy(dst + (y0 + y) * dst_linesize + x0,
                           src + y * bw, bw);
                src += pixels;
                break;
            case JVID_BLOCK_RLE: {
                int written = 0;

                while (written < pixels) {
                    uint8_t run, value;
                    int row, col;

                    if (src_end - src < 2)
                        return AVERROR_INVALIDDATA;

                    run = *src++;
                    value = *src++;
                    if (!run || written + run > pixels)
                        return AVERROR_INVALIDDATA;

                    while (run--) {
                        row = written / bw;
                        col = written % bw;
                        dst[(y0 + row) * dst_linesize + x0 + col] = value;
                        written++;
                    }
                }
                break;
            }
            default:
                av_log(avctx, AV_LOG_ERROR, "Unsupported JVID block tag %d.\n", tag);
                return AVERROR_INVALIDDATA;
            }
        }
    }

    *psrc = src;
    return 0;
}

static int jvid_decode_block_frame(AVCodecContext *avctx, JVIDContext *s,
                                   AVFrame *frame, const JVIDHeader *hdr,
                                   const uint8_t *src, int payload_size)
{
    int y_size = hdr->width * hdr->height;
    int uv_w = AV_CEIL_RSHIFT(hdr->width, 1);
    int uv_h = AV_CEIL_RSHIFT(hdr->height, 1);
    int uv_size = uv_w * uv_h;
    int frame_size = y_size + 2 * uv_size;
    int have_prev = s->has_prev_frame &&
                    s->width == hdr->width && s->height == hdr->height &&
                    s->prev_frame;
    uint8_t *mutable_src = (uint8_t *)src;
    const uint8_t *src_end = src + payload_size;
    int ret;

    if (payload_size < 4)
        return AVERROR_INVALIDDATA;

    ret = jvid_ensure_buffer(&s->prev_frame, &s->prev_frame_alloc, frame_size);
    if (ret < 0)
        return ret;

    mutable_src += 4;
    ret = jvid_decode_plane_blocks(avctx, s, &mutable_src, src_end,
                                   frame->data[0], frame->linesize[0],
                                   0, hdr->width, hdr->height,
                                   hdr->width, hdr->block_size, have_prev);
    if (ret < 0)
        return ret;
    ret = jvid_decode_plane_blocks(avctx, s, &mutable_src, src_end,
                                   frame->data[1], frame->linesize[1],
                                   y_size, uv_w, uv_h,
                                   uv_w, hdr->block_size, have_prev);
    if (ret < 0)
        return ret;
    ret = jvid_decode_plane_blocks(avctx, s, &mutable_src, src_end,
                                   frame->data[2], frame->linesize[2],
                                   y_size + uv_size, uv_w, uv_h,
                                   uv_w, hdr->block_size, have_prev);
    if (ret < 0)
        return ret;

    jvid_store_prev_block(s->prev_frame, 0, hdr->width,
                          frame->data[0], frame->linesize[0],
                          0, 0, hdr->width, hdr->height);
    jvid_store_prev_block(s->prev_frame, y_size, uv_w,
                          frame->data[1], frame->linesize[1],
                          0, 0, uv_w, uv_h);
    jvid_store_prev_block(s->prev_frame, y_size + uv_size, uv_w,
                          frame->data[2], frame->linesize[2],
                          0, 0, uv_w, uv_h);
    s->has_prev_frame = 1;

    if (mutable_src != src_end) {
        av_log(avctx, AV_LOG_DEBUG, "JVID trailing payload bytes: %td\n", src_end - mutable_src);
    }

    return 0;
}

static int jvid_decode_frame(AVCodecContext *avctx, AVFrame *frame,
                             int *got_frame, AVPacket *avpkt)
{
    JVIDContext *s = avctx->priv_data;
    z_stream *zstream = &s->zstream.zstream;
    JVIDHeader hdr;
    AVHWFramesContext *frames_ctx = NULL;
    AVFrame *sw_frame = NULL;
    AVFrame *dst_frame = frame;
    enum AVPixelFormat pix_fmt;
    uint8_t *src_data[4] = { 0 };
    int src_linesize[4] = { 0 };
    int payload_size;
    int compressed_size;
    int zret;
    const uint8_t *src;
    int ret;

    ret = jvid_parse_header(avctx, &hdr, avpkt->data, avpkt->size);
    if (ret < 0)
        return ret;

    switch (hdr.frame_type) {
    case JVID_FRAME_RAW_I:
        pix_fmt = AV_PIX_FMT_GRAY8;
        break;
    case JVID_FRAME_RAW_I420:
    case JVID_FRAME_BLOCK_I420:
        pix_fmt = AV_PIX_FMT_YUV420P;
        break;
    default:
        av_log(avctx, AV_LOG_ERROR,
               "Unsupported JVID frame type %u.\n", hdr.frame_type);
        return AVERROR_PATCHWELCOME;
    }

    ret = ff_set_dimensions(avctx, hdr.width, hdr.height);
    if (ret < 0)
        return ret;

    if (avctx->width != hdr.width || avctx->height != hdr.height) {
        av_log(avctx, AV_LOG_ERROR, "JVID dimension mismatch.\n");
        return AVERROR_INVALIDDATA;
    }

    payload_size = av_image_get_buffer_size(pix_fmt, hdr.width, hdr.height, 1);
    if (payload_size < 0)
        return payload_size;
    compressed_size = avpkt->size - hdr.header_size;
    if (compressed_size < 0)
        return AVERROR_INVALIDDATA;

    if (avctx->hw_frames_ctx) {
        frames_ctx = (AVHWFramesContext *)avctx->hw_frames_ctx->data;
        if ((frames_ctx->format != AV_PIX_FMT_CUDA &&
             frames_ctx->format != AV_PIX_FMT_VIDEOTOOLBOX) ||
            frames_ctx->sw_format != pix_fmt) {
            av_log(avctx, AV_LOG_ERROR,
                   "JVID hardware decode requires a CUDA or VideoToolbox hw_frames_ctx matching sw_format %s.\n",
                   av_get_pix_fmt_name(pix_fmt));
            return AVERROR(EINVAL);
        }

        sw_frame = av_frame_alloc();
        if (!sw_frame)
            return AVERROR(ENOMEM);
        sw_frame->format = pix_fmt;
        sw_frame->width  = hdr.width;
        sw_frame->height = hdr.height;

        ret = av_frame_get_buffer(sw_frame, 0);
        if (ret < 0)
            goto fail;

        avctx->sw_pix_fmt = pix_fmt;
        avctx->pix_fmt    = frames_ctx->format;
        dst_frame         = sw_frame;
    } else {
        avctx->pix_fmt = pix_fmt;
        ret = ff_get_buffer(avctx, frame, 0);
        if (ret < 0)
            return ret;
    }

    src = avpkt->data + hdr.header_size;
    if (hdr.flags & JVID_FLAG_DEFLATE) {
        int inflated_size = hdr.frame_type == JVID_FRAME_BLOCK_I420 ? payload_size * 3 + 1024 : payload_size;

        ret = jvid_ensure_buffer(&s->inflated, &s->inflated_alloc, inflated_size);
        if (ret < 0)
            return ret;

        zret = inflateReset(zstream);
        if (zret != Z_OK)
            return AVERROR_EXTERNAL;
        zstream->next_in   = (Bytef *)src;
        zstream->avail_in  = compressed_size;
        zstream->next_out  = s->inflated;
        zstream->avail_out = inflated_size;
        zret = inflate(zstream, Z_FINISH);
        if (zret != Z_STREAM_END) {
            av_log(avctx, AV_LOG_ERROR, "Invalid compressed JVID payload.\n");
            return AVERROR_INVALIDDATA;
        }
        src = s->inflated;
        compressed_size = zstream->total_out;
    }

    if (hdr.frame_type == JVID_FRAME_BLOCK_I420) {
        ret = jvid_decode_block_frame(avctx, s, dst_frame, &hdr, src, compressed_size);
        if (ret < 0)
            goto fail;
    } else {
        if (!(hdr.flags & JVID_FLAG_DEFLATE) && compressed_size < payload_size)
            goto fail_invalid;

        ret = av_image_fill_linesizes(src_linesize, pix_fmt, hdr.width);
        if (ret < 0)
            goto fail;
        ret = av_image_fill_pointers(src_data, pix_fmt, hdr.height, src, src_linesize);
        if (ret < 0)
            goto fail;
        av_image_copy(dst_frame->data, dst_frame->linesize,
                      (const uint8_t * const *)src_data, src_linesize,
                      pix_fmt, hdr.width, hdr.height);

        if (hdr.frame_type == JVID_FRAME_RAW_I420) {
            int y_size = hdr.width * hdr.height;
            int uv_w = AV_CEIL_RSHIFT(hdr.width, 1);
            int uv_h = AV_CEIL_RSHIFT(hdr.height, 1);
            int uv_size = uv_w * uv_h;

            ret = jvid_ensure_buffer(&s->prev_frame, &s->prev_frame_alloc,
                                     y_size + 2 * uv_size);
            if (ret < 0)
                goto fail;

            jvid_store_prev_block(s->prev_frame, 0, hdr.width,
                                  dst_frame->data[0], dst_frame->linesize[0],
                                  0, 0, hdr.width, hdr.height);
            jvid_store_prev_block(s->prev_frame, y_size, uv_w,
                                  dst_frame->data[1], dst_frame->linesize[1],
                                  0, 0, uv_w, uv_h);
            jvid_store_prev_block(s->prev_frame, y_size + uv_size, uv_w,
                                  dst_frame->data[2], dst_frame->linesize[2],
                                  0, 0, uv_w, uv_h);
            s->has_prev_frame = 1;
        }
    }

    if (sw_frame) {
        ret = av_hwframe_get_buffer(avctx->hw_frames_ctx, frame, 0);
        if (ret < 0) {
            av_log(avctx, AV_LOG_ERROR, "Failed to allocate hardware output frame.\n");
            goto fail;
        }

        ret = av_hwframe_transfer_data(frame, sw_frame, 0);
        if (ret < 0) {
            av_log(avctx, AV_LOG_ERROR, "Failed to transfer decoded frame to hardware memory.\n");
            goto fail;
        }

        frame->width  = sw_frame->width;
        frame->height = sw_frame->height;
        frame->pts    = avpkt->pts;
    }

    frame->pict_type = (avpkt->flags & AV_PKT_FLAG_KEY) ? AV_PICTURE_TYPE_I
                                                        : AV_PICTURE_TYPE_P;
    if (avpkt->flags & AV_PKT_FLAG_KEY)
        frame->flags |= AV_FRAME_FLAG_KEY;

    s->version = hdr.version;
    s->width   = hdr.width;
    s->height  = hdr.height;

    *got_frame = 1;
    av_frame_free(&sw_frame);
    return avpkt->size;

fail_invalid:
    ret = AVERROR_INVALIDDATA;
fail:
    av_frame_free(&sw_frame);
    return ret;
}

const FFCodec ff_jvid_decoder = {
    .p.name         = "jvid",
    CODEC_LONG_NAME("JVID video"),
    .p.type         = AVMEDIA_TYPE_VIDEO,
    .p.id           = AV_CODEC_ID_JVID,
    .priv_data_size = sizeof(JVIDContext),
    .init           = jvid_decode_init,
    .close          = jvid_decode_close,
    FF_CODEC_DECODE_CB(jvid_decode_frame),
    .p.capabilities = AV_CODEC_CAP_DR1,
    .hw_configs     = (const AVCodecHWConfigInternal *const []) {
        HW_CONFIG_ENCODER_FRAMES(CUDA, CUDA),
        HW_CONFIG_ENCODER_FRAMES(VIDEOTOOLBOX, VIDEOTOOLBOX),
        NULL
    },
    .caps_internal  = FF_CODEC_CAP_INIT_CLEANUP,
};
