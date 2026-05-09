#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  tools/rekey_jmov_in_place.sh <file-or-dir> [<file-or-dir> ...]

Re-encodes the video streams in .jmov files to jvid with regular keyframes,
while copying non-video streams. Each file is rewritten in place by first
encoding to a temporary file in the same directory and then renaming it over
the original.
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ffmpeg_bin="${FFMPEG_BIN:-$repo_root/ffmpeg}"
ffprobe_bin="${FFPROBE_BIN:-$repo_root/ffprobe}"
job_count="${JOBS:-}"
ffmpeg_threads="${FFMPEG_THREADS:-0}"
ffmpeg_loglevel="${FFMPEG_LOGLEVEL:-info}"

if [[ ! -x "$ffmpeg_bin" ]]; then
    ffmpeg_bin="${FFMPEG_BIN:-ffmpeg}"
fi

if [[ ! -x "$ffprobe_bin" ]]; then
    ffprobe_bin="${FFPROBE_BIN:-ffprobe}"
fi

need_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        printf 'error: required command not found: %s\n' "$1" >&2
        exit 1
    fi
}

need_cmd "$ffmpeg_bin"
need_cmd "$ffprobe_bin"

if [[ -z "$job_count" ]]; then
    if command -v sysctl >/dev/null 2>&1; then
        job_count="$(sysctl -n hw.ncpu 2>/dev/null || true)"
    fi
    if [[ -z "$job_count" ]] && command -v getconf >/dev/null 2>&1; then
        job_count="$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)"
    fi
    job_count="${job_count:-1}"
fi

videotoolbox_args=()
if [[ "$(uname -s)" == "Darwin" ]] && "$ffmpeg_bin" -hide_banner -hwaccels 2>/dev/null | grep -q 'videotoolbox'; then
    videotoolbox_args=(-hwaccel videotoolbox -hwaccel_output_format videotoolbox)
fi

fps_for_file() {
    local input=$1
    local rate

    rate="$("$ffprobe_bin" -v error \
        -select_streams v:0 \
        -show_entries stream=avg_frame_rate,r_frame_rate \
        -of default=nokey=1:noprint_wrappers=1 \
        "$input" | awk '
            function fps_from_ratio(r,    a) {
                split(r, a, "/")
                if (length(a) != 2 || a[2] == 0)
                    return 0
                return a[1] / a[2]
            }
            {
                fps = fps_from_ratio($0)
                if (fps > 0) {
                    print fps
                    exit
                }
            }
        ')"

    if [[ -z "$rate" ]]; then
        rate="30"
    fi

    printf '%s\n' "$rate"
}

gop_for_fps() {
    local fps=$1

    awk -v fps="$fps" 'BEGIN {
        g = int(fps * 2)
        if (g < 1)
            g = 1
        print g
    }'
}

rekey_file() {
    local input=$1
    local dir base tmp fps gop

    dir="$(cd "$(dirname "$input")" && pwd)"
    base="$(basename "$input")"

    fps="$(fps_for_file "$input")"
    gop="$(gop_for_fps "$fps")"
    tmp="$dir/.${base}.rekey.$$.$RANDOM.tmp.jmov"

    printf 'Re-encoding %s (fps=%s, gop=%s)\n' "$input" "$fps" "$gop"
    printf 'Using ffmpeg loglevel=%s threads=%s hwaccel=%s\n' \
        "$ffmpeg_loglevel" \
        "$ffmpeg_threads" \
        "$(if [[ ${#videotoolbox_args[@]} -gt 0 ]]; then printf 'videotoolbox'; else printf 'none'; fi)"

    rm -f "$tmp"
    if ! "$ffmpeg_bin" -nostdin -hide_banner -y \
        -loglevel "$ffmpeg_loglevel" \
        -stats \
        "${videotoolbox_args[@]}" \
        -threads "$ffmpeg_threads" \
        -i "$input" \
        -map 0 \
        -c copy \
        -c:v jvid \
        -g "$gop" \
        "$tmp"; then
        rm -f "$tmp"
        printf 'error: failed to re-encode %s\n' "$input" >&2
        return 1
    fi

    mv -f "$tmp" "$input"
}

wait_for_slot() {
    while [[ "$(jobs -pr | wc -l | tr -d ' ')" -ge "$job_count" ]]; do
        wait -n
    done
}

queue_file() {
    local file=$1

    wait_for_slot
    rekey_file "$file" &
}

process_path() {
    local path=$1

    if [[ -d "$path" ]]; then
        while IFS= read -r file; do
            queue_file "$file"
        done < <(find "$path" -type f -name '*.jmov' | sort)
    elif [[ -f "$path" ]]; then
        queue_file "$path"
    else
        printf 'warning: skipping missing path: %s\n' "$path" >&2
    fi
}

for path in "$@"; do
    process_path "$path"
done

wait
