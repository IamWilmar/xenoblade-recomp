#!/usr/bin/env bash
# Compiles the APK (start screen in Java + NativeActivity with libnier.so and libnnsdk.so) into build/xenobladeport.apk.
# Needs: Android SDK with NDK 28.2.13676358, build-tools 35.0.0 and platform android-33; a JDK (11+); cmake; python3;
# zip. Run tools/setup.sh once first (third-party sources).
#
# Signing: by default the Android debug keystore (~/.android/debug.keystore, created if missing). For your own release
# key: KEYSTORE=<file> KS_PASS=<password> [KEY_ALIAS=<alias>] tools/build_apk.sh
# Compila la APK. Firma con la clave de depuración de Android salvo que se pase KEYSTORE.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SDK="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$HOME/Library/Android/sdk}}"
NDK="${NDK_DIR:-$SDK/ndk/28.2.13676358}"  # this exact NDK (ANDROID_NDK_HOME often points to another)
BT="$SDK/build-tools/35.0.0"
JAR="$SDK/platforms/android-33/android.jar"
BUILD="$ROOT/build/android"
STAGE="$ROOT/build/apk"
APK="$ROOT/build/xenobladeport.apk"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2> /dev/null || echo 4)}"

for need in "$NDK/build/cmake/android.toolchain.cmake" "$BT/aapt2" "$JAR"; do
  [ -e "$need" ] || { echo "missing: $need (see README: Building)" >&2; exit 1; }
done
[ -d "$ROOT/third_party/eden/src/shader_recompiler" ] || { echo "run tools/setup.sh first" >&2; exit 1; }

mkdir -p "$BUILD"
cmake -S "$ROOT/runtime" -B "$BUILD" -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" -DANDROID_NDK="$NDK" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-30 -DANDROID_STL=c++_static > /dev/null
if ! cmake --build "$BUILD" -j"$JOBS" > "$BUILD/build.log" 2>&1; then
  grep -E "error" "$BUILD/build.log" >&2 || tail -40 "$BUILD/build.log" >&2
  echo "build failed (full log: $BUILD/build.log)" >&2
  exit 1
fi

rm -rf "$STAGE"
mkdir -p "$STAGE/lib/arm64-v8a" "$STAGE/classes"
# Start screen (SetupActivity: find the user's game files, then start): javac + d8 into classes.dex.
javac -Xlint:-options --release 11 -classpath "$JAR" -d "$STAGE/classes" $(find "$ROOT/android/java" -name '*.java')
"$BT/d8" --release --min-api 30 --lib "$JAR" --output "$STAGE" $(find "$STAGE/classes" -name '*.class')
cp "$BUILD/libnier.so" "$BUILD/libnnsdk.so" "$STAGE/lib/arm64-v8a/"
"$BT/aapt2" compile --dir "$ROOT/android/res" -o "$STAGE/res.zip"
"$BT/aapt2" link --manifest "$ROOT/android/AndroidManifest.xml" -I "$JAR" -R "$STAGE/res.zip" --auto-add-overlay \
  --debug-mode -o "$STAGE/base.apk"
(cd "$STAGE" && zip -q -r base.apk lib classes.dex)
"$BT/zipalign" -f -p 4 "$STAGE/base.apk" "$STAGE/aligned.apk"

if [ -n "${KEYSTORE:-}" ]; then
  "$BT/apksigner" sign --ks "$KEYSTORE" --ks-pass env:KS_PASS ${KEY_ALIAS:+--ks-key-alias "$KEY_ALIAS"} \
    --out "$APK" "$STAGE/aligned.apk"
else
  DEBUG_KS="$HOME/.android/debug.keystore"
  if [ ! -f "$DEBUG_KS" ]; then
    mkdir -p "$(dirname "$DEBUG_KS")"
    keytool -genkeypair -keystore "$DEBUG_KS" -storepass android -keypass android -alias androiddebugkey \
      -keyalg RSA -keysize 2048 -validity 10000 -dname "CN=Android Debug,O=Android,C=US" > /dev/null
  fi
  "$BT/apksigner" sign --ks "$DEBUG_KS" --ks-pass pass:android --key-pass pass:android --out "$APK" "$STAGE/aligned.apk"
fi
echo "$APK"
