#!/usr/bin/env bash
# Build OpenOne.app and a .dmg on macOS (Apple Silicon; Intel builds work but have no local runtime, remote only).
#
#   packaging/macos/build.sh [--offline] [--python python3.12]
#
#   --offline   bundle the MPS inference runtime inside the .app (works without internet at onboarding, ~+900 MB)
#
# Signing (optional): set MACOS_SIGN_IDENTITY="Developer ID Application: ..." to sign with the hardened runtime, and
# NOTARY_PROFILE=<notarytool keychain profile> to notarize and staple. Without them the app is ad-hoc signed only:
# first launch needs right-click > Open (or: xattr -dr com.apple.quarantine /Applications/OpenOne.app).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
VERSION=0.2.0
PY=python3.12
OFFLINE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --offline) OFFLINE=1 ;;
    --python) PY=$2; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
  shift
done
ARCH=$(uname -m)                         # arm64 | x86_64
TAG=$([ "$ARCH" = arm64 ] && echo arm64 || echo x64)

# 1. build venv (UI process only: pywebview + jsonschema + PyInstaller; no torch)
[ -x .build-venv/bin/python ] || "$PY" -m venv .build-venv
.build-venv/bin/python -m pip install --disable-pip-version-check -q -r packaging/requirements-build.txt

# 2. icon
if [ ! -f packaging/assets/icon.icns ]; then
  IS=$(mktemp -d)/icon.iconset; mkdir -p "$IS"
  for s in 16 32 64 128 256 512; do
    sips -z $s $s packaging/assets/icon-1024.png --out "$IS/icon_${s}x${s}.png" >/dev/null
    d=$((s * 2)); sips -z $d $d packaging/assets/icon-1024.png --out "$IS/icon_${s}x${s}@2x.png" >/dev/null
  done
  iconutil -c icns "$IS" -o packaging/assets/icon.icns
fi

# 3. app bundle
rm -rf build dist/OpenOne dist/OpenOne.app
.build-venv/bin/python -m PyInstaller --noconfirm --clean --distpath dist --workpath build packaging/openone.spec
APP=dist/OpenOne.app

# 4. smoke test: the frozen app starts headless and answers /api/status
TMPD=$(mktemp -d)
OPENONE_DATA_DIR="$TMPD" "$APP/Contents/MacOS/OpenOne" --serve 0 & PID=$!
for _ in $(seq 1 60); do [ -s "$TMPD/serve-url.txt" ] && break; sleep 0.5; done
URL=$(cat "$TMPD/serve-url.txt"); TOK=${URL#*t=}; BASE=${URL%%/\?*}
curl -fsS -H "X-OpenOne-Token: $TOK" "$BASE/api/status" | grep -q '"app_version": "'$VERSION'"'
kill $PID; rm -rf "$TMPD"
echo "smoke test ok"

# 5. optional bundled runtime (Apple Silicon: torch with MPS)
SUFFIX=""
if [ "$OFFLINE" = 1 ]; then
  [ "$ARCH" = arm64 ] || { echo "--offline needs Apple Silicon (no Intel-Mac torch builds)" >&2; exit 1; }
  .build-venv/bin/python packaging/bundle_runtime.py mps "$APP/Contents/Resources/runtime" mac-arm64
  SUFFIX="-offline"
fi

# 6. signing
if [ -n "${MACOS_SIGN_IDENTITY:-}" ]; then
  codesign --force --deep --options runtime --timestamp --entitlements packaging/macos/entitlements.plist \
    --sign "$MACOS_SIGN_IDENTITY" "$APP"
else
  codesign --force --deep --sign - "$APP"
fi
codesign --verify --deep --strict "$APP"

# 7. dmg (app + /Applications link) and a zip of the .app
mkdir -p release
DMG=release/OpenOne-$VERSION-macos-$TAG$SUFFIX.dmg
STAGE=$(mktemp -d); cp -R "$APP" "$STAGE/"; ln -s /Applications "$STAGE/Applications"
rm -f "$DMG"
hdiutil create -volname "OpenOne $VERSION" -srcfolder "$STAGE" -ov -format UDZO "$DMG" >/dev/null
rm -rf "$STAGE"
ditto -c -k --keepParent "$APP" "release/OpenOne-$VERSION-macos-$TAG$SUFFIX.app.zip"
if [ -n "${MACOS_SIGN_IDENTITY:-}" ]; then codesign --sign "$MACOS_SIGN_IDENTITY" --timestamp "$DMG"; fi
if [ -n "${NOTARY_PROFILE:-}" ]; then
  xcrun notarytool submit "$DMG" --keychain-profile "$NOTARY_PROFILE" --wait
  xcrun stapler staple "$DMG"
fi
(cd release && shasum -a 256 OpenOne-$VERSION-macos-* > SHA256SUMS-macos-$TAG$SUFFIX.txt)
ls -lh release
