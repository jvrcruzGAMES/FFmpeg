#!/usr/bin/env bash
#
# Build FFmpeg static libraries for iOS device and iOS simulator.
#
# Outputs are installed separately under:
#   build/ios/iphoneos-arm64
#   build/ios/iphonesimulator-arm64
#
# Common overrides:
#   MIN_IOS=13.0 ./tools/build-ios.sh
#   DEVICE_ARCHS="arm64" SIM_ARCHS="arm64 x86_64" ./tools/build-ios.sh
#   BUILD_ROOT="$PWD/out/ios" ./tools/build-ios.sh
#   EXTRA_CONFIGURE_FLAGS="--disable-everything --enable-decoder=h264 --enable-hwaccel=h264_videotoolbox" ./tools/build-ios.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${BUILD_ROOT:-$ROOT_DIR/build/ios}"
MIN_IOS="${MIN_IOS:-12.0}"
DEVICE_ARCHS="${DEVICE_ARCHS:-arm64}"
SIM_ARCHS="${SIM_ARCHS:-arm64}"
EXTRA_CONFIGURE_FLAGS="${EXTRA_CONFIGURE_FLAGS:-}"

COMMON_CONFIGURE_FLAGS=(
    --target-os=darwin
    --enable-cross-compile
    --disable-programs
    --disable-doc
    --disable-debug
    --disable-shared
    --enable-static
    --enable-pic
    --enable-videotoolbox
)

usage() {
    cat <<EOF
Usage: $(basename "$0") [--device-only] [--simulator-only] [--clean]

Environment overrides:
  MIN_IOS                 Minimum iOS version. Default: $MIN_IOS
  DEVICE_ARCHS            Device architectures. Default: $DEVICE_ARCHS
  SIM_ARCHS               Simulator architectures. Default: $SIM_ARCHS
  BUILD_ROOT              Install root. Default: $BUILD_ROOT
  EXTRA_CONFIGURE_FLAGS   Extra flags passed to FFmpeg configure.

This script enables Apple VideoToolbox hardware acceleration and installs static
libraries and headers only. It does not create an XCFramework.
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

require_toolchain() {
    command -v xcrun >/dev/null 2>&1 || die "xcrun was not found. Install Xcode or Xcode Command Line Tools."
    xcrun --sdk iphoneos --show-sdk-path >/dev/null
    xcrun --sdk iphonesimulator --show-sdk-path >/dev/null
}

min_version_flag() {
    case "$1" in
        iphoneos) echo "-miphoneos-version-min=$MIN_IOS" ;;
        iphonesimulator) echo "-mios-simulator-version-min=$MIN_IOS" ;;
        *) die "unknown SDK: $1" ;;
    esac
}

target_triple() {
    local sdk="$1"
    local arch="$2"

    case "$sdk" in
        iphoneos) echo "$arch-apple-ios$MIN_IOS" ;;
        iphonesimulator) echo "$arch-apple-ios$MIN_IOS-simulator" ;;
        *) die "unknown SDK: $sdk" ;;
    esac
}

build_one() {
    local sdk="$1"
    local arch="$2"
    local sdk_path
    local min_flag
    local target
    local prefix
    local cc

    sdk_path="$(xcrun --sdk "$sdk" --show-sdk-path)"
    min_flag="$(min_version_flag "$sdk")"
    target="$(target_triple "$sdk" "$arch")"
    prefix="$BUILD_ROOT/$sdk-$arch"
    cc="$(xcrun --sdk "$sdk" -f clang) -target $target"

    echo
    echo "==> Building FFmpeg for $sdk $arch"
    echo "    Install prefix: $prefix"

    cd "$ROOT_DIR"
    make distclean >/dev/null 2>&1 || true

    ./configure \
        --prefix="$prefix" \
        --arch="$arch" \
        --cc="$cc" \
        --ar="$(xcrun --sdk "$sdk" -f ar)" \
        --ranlib="$(xcrun --sdk "$sdk" -f ranlib)" \
        --strip="$(xcrun --sdk "$sdk" -f strip)" \
        --sysroot="$sdk_path" \
        --extra-cflags="-target $target $min_flag -isysroot $sdk_path" \
        --extra-ldflags="-target $target $min_flag -isysroot $sdk_path" \
        "${COMMON_CONFIGURE_FLAGS[@]}" \
        $EXTRA_CONFIGURE_FLAGS

    make -j"$(sysctl -n hw.ncpu)"
    make install
}

build_archs() {
    local sdk="$1"
    local archs="$2"
    local arch

    for arch in $archs; do
        build_one "$sdk" "$arch"
    done
}

main() {
    local build_device=1
    local build_simulator=1

    while [ "$#" -gt 0 ]; do
        case "$1" in
            --device-only)
                build_simulator=0
                ;;
            --simulator-only)
                build_device=0
                ;;
            --clean)
                rm -rf "$BUILD_ROOT"
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                usage >&2
                exit 2
                ;;
        esac
        shift
    done

    require_toolchain
    mkdir -p "$BUILD_ROOT"

    if [ "$build_device" -eq 1 ]; then
        build_archs iphoneos "$DEVICE_ARCHS"
    fi

    if [ "$build_simulator" -eq 1 ]; then
        build_archs iphonesimulator "$SIM_ARCHS"
    fi

    cd "$ROOT_DIR"
    make distclean >/dev/null 2>&1 || true

    echo
    echo "Done. Static libraries and headers were installed under:"
    echo "  $BUILD_ROOT"
    echo
    echo "No XCFramework was generated."
}

main "$@"
