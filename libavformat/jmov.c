/*
 * JMOV container format foundation
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
 * JMOV container format foundation
 *
 * Version 1 stores a lightweight chunked file structure:
 * - file header: "JMOV" + version
 * - header chunks: STRM, META, DATA
 * - packet chunks: PKT0 ... DONE
 *
 * The layout is intentionally simple so it can already carry audio, video,
 * subtitles, attachments, and generic data streams while leaving plenty of
 * room for indexing, side data, richer metadata, and codec-specific boxes.
 *
 * Stream descriptors also preserve generic audio fields like frame_size,
 * initial_padding, trailing_padding, and seek_preroll so common codecs such as
 * AAC, MP3, Opus, Vorbis, FLAC, AC-3, E-AC-3, and ALAC can round-trip with
 * fewer container-level surprises.
 */

#include <limits.h>
#include <stdint.h>
#include <string.h>

#include "libavutil/channel_layout.h"
#include "libavutil/intreadwrite.h"
#include "libavutil/mathematics.h"
#include "libavutil/mem.h"
#include "libavutil/timestamp.h"

#include "avformat.h"
#include "avio.h"
#include "demux.h"
#include "internal.h"
#include "mux.h"

#define JMOV_MAGIC              MKTAG('J', 'M', 'O', 'V')
#define JMOV_VERSION            1

#define JMOV_TAG_STRM           MKTAG('S', 'T', 'R', 'M')
#define JMOV_TAG_SMTA           MKTAG('S', 'M', 'T', 'A')
#define JMOV_TAG_META           MKTAG('M', 'E', 'T', 'A')
#define JMOV_TAG_DATA           MKTAG('D', 'A', 'T', 'A')
#define JMOV_TAG_PKT0           MKTAG('P', 'K', 'T', '0')
#define JMOV_TAG_DONE           MKTAG('D', 'O', 'N', 'E')

#define JMOV_STREAM_DESC_CORE_SIZE 64U
#define JMOV_STREAM_DESC_SIZE      80U
#define JMOV_PACKET_DESC_SIZE   28U

enum JMOVMediaType {
    JMOV_MEDIA_VIDEO      = 'V',
    JMOV_MEDIA_AUDIO      = 'A',
    JMOV_MEDIA_SUBTITLE   = 'S',
    JMOV_MEDIA_DATA       = 'D',
    JMOV_MEDIA_ATTACHMENT = 'T',
};

enum JMOVPacketFlags {
    JMOV_PKT_FLAG_KEY     = 1 << 0,
    JMOV_PKT_FLAG_CORRUPT = 1 << 1,
    JMOV_PKT_FLAG_DISCARD = 1 << 2,
};

typedef struct JMOVMuxContext {
    int wrote_data_chunk;
} JMOVMuxContext;

typedef struct JMOVPacketStats {
    int seen;
    int64_t first_ts;
    int64_t end_ts;
    int64_t payload_size;
    int64_t packets;
} JMOVPacketStats;

static int jmov_media_type_from_codecpar(const AVCodecParameters *par)
{
    switch (par->codec_type) {
    case AVMEDIA_TYPE_VIDEO:      return JMOV_MEDIA_VIDEO;
    case AVMEDIA_TYPE_AUDIO:      return JMOV_MEDIA_AUDIO;
    case AVMEDIA_TYPE_SUBTITLE:   return JMOV_MEDIA_SUBTITLE;
    case AVMEDIA_TYPE_DATA:       return JMOV_MEDIA_DATA;
    case AVMEDIA_TYPE_ATTACHMENT: return JMOV_MEDIA_ATTACHMENT;
    default:                      return AVERROR(EINVAL);
    }
}

static enum AVMediaType jmov_media_type_to_av(unsigned type)
{
    switch (type) {
    case JMOV_MEDIA_VIDEO:      return AVMEDIA_TYPE_VIDEO;
    case JMOV_MEDIA_AUDIO:      return AVMEDIA_TYPE_AUDIO;
    case JMOV_MEDIA_SUBTITLE:   return AVMEDIA_TYPE_SUBTITLE;
    case JMOV_MEDIA_DATA:       return AVMEDIA_TYPE_DATA;
    case JMOV_MEDIA_ATTACHMENT: return AVMEDIA_TYPE_ATTACHMENT;
    default:                    return AVMEDIA_TYPE_UNKNOWN;
    }
}

static void jmov_write_chunk_header(AVIOContext *pb, uint32_t tag, uint32_t size)
{
    avio_wl32(pb, tag);
    avio_wl32(pb, size);
}

static int jmov_write_meta_chunk(AVFormatContext *s,
                                 const AVDictionaryEntry *tag)
{
    AVIOContext *pb = s->pb;
    size_t key_len = strlen(tag->key);
    size_t value_len = strlen(tag->value);

    if (key_len > UINT16_MAX || value_len > UINT16_MAX)
        return AVERROR(EINVAL);

    jmov_write_chunk_header(pb, JMOV_TAG_META,
                            4 + (uint32_t)key_len + (uint32_t)value_len);
    avio_wl16(pb, key_len);
    avio_wl16(pb, value_len);
    avio_write(pb, tag->key, key_len);
    avio_write(pb, tag->value, value_len);

    return 0;
}

static int jmov_write_stream_meta_chunk(AVFormatContext *s, AVStream *st,
                                        const AVDictionaryEntry *tag)
{
    AVIOContext *pb = s->pb;
    size_t key_len = strlen(tag->key);
    size_t value_len = strlen(tag->value);

    if (st->index > UINT8_MAX || key_len > UINT16_MAX || value_len > UINT16_MAX)
        return AVERROR(EINVAL);

    jmov_write_chunk_header(pb, JMOV_TAG_SMTA,
                            5 + (uint32_t)key_len + (uint32_t)value_len);
    avio_w8(pb, st->index);
    avio_wl16(pb, key_len);
    avio_wl16(pb, value_len);
    avio_write(pb, tag->key, key_len);
    avio_write(pb, tag->value, value_len);

    return 0;
}

static int jmov_write_stream_chunk(AVFormatContext *s, AVStream *st)
{
    AVIOContext *pb = s->pb;
    const AVCodecParameters *par = st->codecpar;
    int media_type = jmov_media_type_from_codecpar(par);
    AVRational tb = st->time_base;
    uint64_t mask = 0;

    if (media_type < 0) {
        av_log(s, AV_LOG_ERROR, "Unsupported JMOV stream type %d.\n",
               par->codec_type);
        return media_type;
    }

    if (st->index > UINT8_MAX || par->extradata_size < 0) {
        av_log(s, AV_LOG_ERROR, "JMOV stream index or extradata invalid.\n");
        return AVERROR(EINVAL);
    }

    if (!tb.num || !tb.den) {
        switch (par->codec_type) {
        case AVMEDIA_TYPE_AUDIO:
            tb = (AVRational){ 1, par->sample_rate > 0 ? par->sample_rate : 48000 };
            break;
        case AVMEDIA_TYPE_SUBTITLE:
        case AVMEDIA_TYPE_DATA:
        case AVMEDIA_TYPE_ATTACHMENT:
            tb = (AVRational){ 1, 1000 };
            break;
        default:
            tb = (AVRational){ 1, 25 };
            break;
        }
    }

    if (par->extradata_size > INT_MAX - (int)JMOV_STREAM_DESC_SIZE)
        return AVERROR(EINVAL);

    if (par->ch_layout.order == AV_CHANNEL_ORDER_NATIVE)
        mask = par->ch_layout.u.mask;

    jmov_write_chunk_header(pb, JMOV_TAG_STRM,
                            JMOV_STREAM_DESC_SIZE + par->extradata_size);
    avio_w8(pb, st->index);
    avio_w8(pb, media_type);
    avio_w8(pb, 0);
    avio_w8(pb, 0);
    avio_wl32(pb, par->codec_id);
    avio_wl32(pb, par->codec_tag);
    avio_wl32(pb, tb.num);
    avio_wl32(pb, tb.den);
    avio_wl32(pb, par->extradata_size);
    avio_wl32(pb, par->width);
    avio_wl32(pb, par->height);
    avio_wl32(pb, par->sample_rate);
    avio_wl32(pb, par->bits_per_coded_sample);
    avio_wl32(pb, par->block_align);
    avio_wl32(pb, par->format);
    avio_wl32(pb, par->ch_layout.nb_channels);
    avio_wl64(pb, mask);
    avio_wl32(pb, st->disposition);
    avio_wl32(pb, par->frame_size);
    avio_wl32(pb, par->initial_padding);
    avio_wl32(pb, par->trailing_padding);
    avio_wl32(pb, par->seek_preroll);

    if (par->extradata_size)
        avio_write(pb, par->extradata, par->extradata_size);

    return 0;
}

static int jmov_write_header(AVFormatContext *s)
{
    JMOVMuxContext *jmov = s->priv_data;
    const AVDictionaryEntry *tag = NULL;
    int ret;

    if (s->nb_streams > UINT8_MAX) {
        av_log(s, AV_LOG_ERROR, "JMOV supports at most 256 streams.\n");
        return AVERROR(EINVAL);
    }

    avio_wl32(s->pb, JMOV_MAGIC);
    avio_w8(s->pb, JMOV_VERSION);
    avio_w8(s->pb, 0);
    avio_wl16(s->pb, 0);
    av_log(s, AV_LOG_DEBUG, "JMOV write header streams=%u\n", s->nb_streams);

    ff_standardize_creation_time(s);
    while ((tag = av_dict_iterate(s->metadata, tag))) {
        ret = jmov_write_meta_chunk(s, tag);
        if (ret < 0)
            return ret;
    }

    for (unsigned i = 0; i < s->nb_streams; i++) {
        const AVDictionaryEntry *stream_tag = NULL;

        ret = jmov_write_stream_chunk(s, s->streams[i]);
        if (ret < 0)
            return ret;

        while ((stream_tag = av_dict_iterate(s->streams[i]->metadata, stream_tag))) {
            ret = jmov_write_stream_meta_chunk(s, s->streams[i], stream_tag);
            if (ret < 0)
                return ret;
        }
    }

    jmov_write_chunk_header(s->pb, JMOV_TAG_DATA, 0);
    jmov->wrote_data_chunk = 1;

    return 0;
}

static int jmov_write_packet(AVFormatContext *s, AVPacket *pkt)
{
    JMOVMuxContext *jmov = s->priv_data;
    uint8_t flags = 0;

    if (!jmov->wrote_data_chunk)
        return AVERROR_BUG;
    if (pkt->stream_index < 0 || pkt->stream_index > UINT8_MAX)
        return AVERROR(EINVAL);

    if (pkt->flags & AV_PKT_FLAG_KEY)
        flags |= JMOV_PKT_FLAG_KEY;
    if (pkt->flags & AV_PKT_FLAG_CORRUPT)
        flags |= JMOV_PKT_FLAG_CORRUPT;
    if (pkt->flags & AV_PKT_FLAG_DISCARD)
        flags |= JMOV_PKT_FLAG_DISCARD;

    av_log(s, AV_LOG_DEBUG, "JMOV write packet stream=%d size=%d pts=%s dts=%s\n",
           pkt->stream_index, pkt->size, av_ts2str(pkt->pts), av_ts2str(pkt->dts));

    jmov_write_chunk_header(s->pb, JMOV_TAG_PKT0, JMOV_PACKET_DESC_SIZE + pkt->size);
    avio_w8(s->pb, pkt->stream_index);
    avio_w8(s->pb, flags);
    avio_wl16(s->pb, 0);
    avio_wl64(s->pb, (uint64_t)pkt->pts);
    avio_wl64(s->pb, (uint64_t)pkt->dts);
    avio_wl32(s->pb, pkt->duration);
    avio_wl32(s->pb, pkt->size);
    avio_write(s->pb, pkt->data, pkt->size);

    return 0;
}

static int jmov_write_trailer(AVFormatContext *s)
{
    jmov_write_chunk_header(s->pb, JMOV_TAG_DONE, 0);
    return 0;
}

static int jmov_probe(const AVProbeData *p)
{
    if (p->buf_size >= 8 && AV_RL32(p->buf) == JMOV_MAGIC && p->buf[4] == JMOV_VERSION)
        return AVPROBE_SCORE_MAX;
    return 0;
}

static int jmov_parse_meta(AVFormatContext *s, uint32_t size)
{
    char *key = NULL, *value = NULL;
    unsigned key_len, value_len;
    int ret = AVERROR_INVALIDDATA;

    if (size < 4)
        return AVERROR_INVALIDDATA;

    key_len = avio_rl16(s->pb);
    value_len = avio_rl16(s->pb);
    if ((uint64_t)key_len + value_len != size - 4)
        return AVERROR_INVALIDDATA;

    key = av_mallocz(key_len + 1);
    value = av_mallocz(value_len + 1);
    if (!key || !value) {
        ret = AVERROR(ENOMEM);
        goto end;
    }

    if (avio_read(s->pb, key, key_len) != key_len ||
        avio_read(s->pb, value, value_len) != value_len)
        goto end;

    ret = av_dict_set(&s->metadata, key, value,
                      AV_DICT_DONT_STRDUP_KEY | AV_DICT_DONT_STRDUP_VAL);
    key = value = NULL;

end:
    av_free(key);
    av_free(value);
    return ret;
}

static int jmov_parse_stream(AVFormatContext *s, uint32_t size)
{
    AVStream *st;
    AVCodecParameters *par;
    enum AVMediaType media_type;
    uint32_t desc_size;
    uint32_t extradata_size;
    uint32_t tb_num, tb_den;
    uint32_t stream_index;
    uint32_t channels;
    uint64_t ch_mask;
    int ret;

    if (size < JMOV_STREAM_DESC_CORE_SIZE)
        return AVERROR_INVALIDDATA;

    stream_index = avio_r8(s->pb);
    media_type   = jmov_media_type_to_av(avio_r8(s->pb));
    avio_skip(s->pb, 2);
    if (stream_index > UINT8_MAX || media_type == AVMEDIA_TYPE_UNKNOWN)
        return AVERROR_INVALIDDATA;

    st = avformat_new_stream(s, NULL);
    if (!st)
        return AVERROR(ENOMEM);
    if (stream_index != (uint32_t)st->index)
        return AVERROR_INVALIDDATA;

    par = st->codecpar;
    par->codec_type    = media_type;
    par->codec_id      = avio_rl32(s->pb);
    par->codec_tag     = avio_rl32(s->pb);
    tb_num             = avio_rl32(s->pb);
    tb_den             = avio_rl32(s->pb);
    extradata_size     = avio_rl32(s->pb);
    par->width         = avio_rl32(s->pb);
    par->height        = avio_rl32(s->pb);
    par->sample_rate   = avio_rl32(s->pb);
    par->bits_per_coded_sample = avio_rl32(s->pb);
    par->block_align   = avio_rl32(s->pb);
    par->format        = avio_rl32(s->pb);
    channels           = avio_rl32(s->pb);
    ch_mask            = avio_rl64(s->pb);
    st->disposition    = avio_rl32(s->pb);
    if (size >= JMOV_STREAM_DESC_SIZE) {
        par->frame_size      = avio_rl32(s->pb);
        par->initial_padding = avio_rl32(s->pb);
        par->trailing_padding = avio_rl32(s->pb);
        par->seek_preroll    = avio_rl32(s->pb);
    }

    desc_size = size >= JMOV_STREAM_DESC_SIZE ? JMOV_STREAM_DESC_SIZE :
                                                JMOV_STREAM_DESC_CORE_SIZE;

    if ((uint64_t)extradata_size > size - desc_size)
        return AVERROR_INVALIDDATA;

    if (channels) {
        if (ch_mask) {
            ret = av_channel_layout_from_mask(&par->ch_layout, ch_mask);
            if (ret < 0)
                return ret;
        } else {
            par->ch_layout.order = AV_CHANNEL_ORDER_UNSPEC;
            par->ch_layout.nb_channels = channels;
        }
    }

    if (tb_num && tb_den)
        avpriv_set_pts_info(st, 64, tb_num, tb_den);
    else
        avpriv_set_pts_info(st, 64, 1, 1000);

    if (extradata_size) {
        ret = ff_alloc_extradata(par, extradata_size);
        if (ret < 0)
            return ret;
        if (avio_read(s->pb, par->extradata, extradata_size) != (int)extradata_size)
            return AVERROR_INVALIDDATA;
    }

    if (size > desc_size + extradata_size)
        avio_skip(s->pb, size - desc_size - extradata_size);

    return 0;
}

static int jmov_parse_stream_meta(AVFormatContext *s, uint32_t size)
{
    AVStream *st;
    char *key = NULL, *value = NULL;
    uint32_t stream_index;
    unsigned key_len, value_len;
    int ret = AVERROR_INVALIDDATA;

    if (size < 5)
        return AVERROR_INVALIDDATA;

    stream_index = avio_r8(s->pb);
    key_len      = avio_rl16(s->pb);
    value_len    = avio_rl16(s->pb);

    if (stream_index >= s->nb_streams ||
        (uint64_t)key_len + value_len != size - 5)
        return AVERROR_INVALIDDATA;

    st = s->streams[stream_index];

    key = av_mallocz(key_len + 1);
    value = av_mallocz(value_len + 1);
    if (!key || !value) {
        ret = AVERROR(ENOMEM);
        goto end;
    }

    if (avio_read(s->pb, key, key_len) != key_len ||
        avio_read(s->pb, value, value_len) != value_len)
        goto end;

    ret = av_dict_set(&st->metadata, key, value,
                      AV_DICT_DONT_STRDUP_KEY | AV_DICT_DONT_STRDUP_VAL);
    key = value = NULL;

end:
    av_free(key);
    av_free(value);
    return ret;
}

static int jmov_read_header(AVFormatContext *s)
{
    AVIOContext *pb = s->pb;
    int64_t data_offset;

    if (avio_rl32(pb) != JMOV_MAGIC)
        return AVERROR_INVALIDDATA;
    if (avio_r8(pb) != JMOV_VERSION)
        return AVERROR_PATCHWELCOME;

    avio_skip(pb, 3);

    while (!avio_feof(pb)) {
        uint32_t tag  = avio_rl32(pb);
        uint32_t size = avio_rl32(pb);

        switch (tag) {
        case JMOV_TAG_META: {
            int ret = jmov_parse_meta(s, size);
            if (ret < 0)
                return ret;
            break;
        }
        case JMOV_TAG_STRM: {
            int ret = jmov_parse_stream(s, size);
            if (ret < 0)
                return ret;
            break;
        }
        case JMOV_TAG_SMTA: {
            int ret = jmov_parse_stream_meta(s, size);
            if (ret < 0)
                return ret;
            break;
        }
        case JMOV_TAG_DATA:
            avio_skip(pb, size);
            data_offset = avio_tell(pb);
            ffformatcontext(s)->data_offset = data_offset;
            if (s->nb_streams && (pb->seekable & AVIO_SEEKABLE_NORMAL)) {
                JMOVPacketStats *stats = av_calloc(s->nb_streams, sizeof(*stats));
                int64_t max_duration = AV_NOPTS_VALUE;
                int64_t file_size;

                if (!stats)
                    return AVERROR(ENOMEM);

                while (!avio_feof(pb)) {
                    uint32_t pkt_tag  = avio_rl32(pb);
                    uint32_t pkt_size = avio_rl32(pb);

                    if (avio_feof(pb))
                        break;

                    if (pkt_tag == JMOV_TAG_PKT0) {
                        uint32_t payload_size;
                        uint32_t duration;
                        uint32_t stream_index;
                        int64_t pts, dts, ts, end_ts;

                        if (pkt_size < JMOV_PACKET_DESC_SIZE) {
                            av_free(stats);
                            return AVERROR_INVALIDDATA;
                        }

                        stream_index = avio_r8(pb);
                        avio_skip(pb, 3);
                        pts          = (int64_t)avio_rl64(pb);
                        dts          = (int64_t)avio_rl64(pb);
                        duration     = avio_rl32(pb);
                        payload_size = avio_rl32(pb);

                        if (payload_size != pkt_size - JMOV_PACKET_DESC_SIZE ||
                            stream_index >= s->nb_streams) {
                            av_free(stats);
                            return AVERROR_INVALIDDATA;
                        }

                        ts = pts != AV_NOPTS_VALUE ? pts : dts;
                        if (ts != AV_NOPTS_VALUE) {
                            JMOVPacketStats *st_stats = &stats[stream_index];

                            end_ts = duration ? ts + duration : ts;
                            if (!st_stats->seen || ts < st_stats->first_ts)
                                st_stats->first_ts = ts;
                            if (!st_stats->seen || end_ts > st_stats->end_ts)
                                st_stats->end_ts = end_ts;
                            st_stats->seen = 1;
                        }
                        stats[stream_index].payload_size += payload_size;
                        stats[stream_index].packets++;
                        avio_skip(pb, payload_size);
                    } else if (pkt_tag == JMOV_TAG_DONE) {
                        avio_skip(pb, pkt_size);
                        break;
                    } else {
                        avio_skip(pb, pkt_size);
                    }
                }

                for (unsigned i = 0; i < s->nb_streams; i++) {
                    AVStream *st = s->streams[i];
                    JMOVPacketStats *st_stats = &stats[i];

                    if (st_stats->seen && st_stats->end_ts >= st_stats->first_ts) {
                        int64_t duration = st_stats->end_ts - st_stats->first_ts;

                        st->start_time = st_stats->first_ts;
                        st->duration   = duration;
                        if (duration > 0 && st_stats->payload_size > 0 &&
                            st->codecpar->bit_rate <= 0)
                            st->codecpar->bit_rate = av_rescale_q(st_stats->payload_size,
                                                                  (AVRational){ 8 * st->time_base.den,
                                                                                st->time_base.num },
                                                                  (AVRational){ duration, 1 });
                        if (st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
                            st->nb_frames = st_stats->packets;

                        duration = av_rescale_q(duration, st->time_base, AV_TIME_BASE_Q);
                        if (max_duration == AV_NOPTS_VALUE || duration > max_duration)
                            max_duration = duration;
                    }
                }

                if (max_duration != AV_NOPTS_VALUE)
                    s->duration = max_duration;
                file_size = avio_size(pb);
                if (s->duration > 0 && file_size > 0)
                    s->bit_rate = av_rescale(file_size, 8 * AV_TIME_BASE, s->duration);

                av_free(stats);
                if (avio_seek(pb, data_offset, SEEK_SET) < 0)
                    return AVERROR(EIO);
            }
            return 0;
        default:
            avio_skip(pb, size);
            break;
        }
    }

    return AVERROR_INVALIDDATA;
}

static int jmov_read_packet(AVFormatContext *s, AVPacket *pkt)
{
    AVIOContext *pb = s->pb;

    while (!avio_feof(pb)) {
        uint32_t tag  = avio_rl32(pb);
        uint32_t size = avio_rl32(pb);
        int64_t pos = avio_tell(pb) - 8;

        switch (tag) {
        case JMOV_TAG_PKT0: {
            uint32_t payload_size;
            uint32_t duration;
            uint32_t stream_index;
            uint8_t flags;
            int ret;

            if (size < JMOV_PACKET_DESC_SIZE)
                return AVERROR_INVALIDDATA;

            stream_index = avio_r8(pb);
            flags        = avio_r8(pb);
            avio_skip(pb, 2);
            pkt->pts     = (int64_t)avio_rl64(pb);
            pkt->dts     = (int64_t)avio_rl64(pb);
            duration     = avio_rl32(pb);
            payload_size = avio_rl32(pb);

            if (payload_size != size - JMOV_PACKET_DESC_SIZE ||
                stream_index >= s->nb_streams)
                return AVERROR_INVALIDDATA;

            ret = av_get_packet(pb, pkt, payload_size);
            if (ret < 0)
                return ret;
            pkt->stream_index = stream_index;
            pkt->duration     = duration;
            pkt->pos          = pos;

            if (flags & JMOV_PKT_FLAG_KEY)
                pkt->flags |= AV_PKT_FLAG_KEY;
            if (flags & JMOV_PKT_FLAG_CORRUPT)
                pkt->flags |= AV_PKT_FLAG_CORRUPT;
            if (flags & JMOV_PKT_FLAG_DISCARD)
                pkt->flags |= AV_PKT_FLAG_DISCARD;

            return 0;
        }
        case JMOV_TAG_DONE:
            avio_skip(pb, size);
            return AVERROR_EOF;
        default:
            avio_skip(pb, size);
            break;
        }
    }

    return AVERROR_EOF;
}

const FFInputFormat ff_jmov_demuxer = {
    .p.name         = "jmov",
    .p.long_name    = NULL_IF_CONFIG_SMALL("JMOV container"),
    .p.extensions   = "jmov",
    .p.flags        = AVFMT_GENERIC_INDEX,
    .read_probe     = jmov_probe,
    .read_header    = jmov_read_header,
    .read_packet    = jmov_read_packet,
};

const FFOutputFormat ff_jmov_muxer = {
    .p.name           = "jmov",
    .p.long_name      = NULL_IF_CONFIG_SMALL("JMOV container"),
    .p.extensions     = "jmov",
    .priv_data_size   = sizeof(JMOVMuxContext),
    .p.audio_codec    = AV_CODEC_ID_AAC,
    .p.video_codec    = AV_CODEC_ID_JVID,
    .p.subtitle_codec = AV_CODEC_ID_SUBRIP,
    .write_header     = jmov_write_header,
    .write_packet     = jmov_write_packet,
    .write_trailer    = jmov_write_trailer,
    .p.flags          = AVFMT_GLOBALHEADER | AVFMT_TS_NONSTRICT,
};
