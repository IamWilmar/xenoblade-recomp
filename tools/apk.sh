#!/usr/bin/env bash
# Development loop: builds the APK (tools/build_apk.sh), installs it on the device connected by adb, starts the game and
# shows its log as soon as it ends, crashes or hangs. The game folder must have been chosen once in the app's start
# screen (or pass NIER_DATA=<folder on the device>).
# Usage: tools/apk.sh            SECS=<n> max wait (default 60); TAIL=<n> log lines to show;
#                               INSTALL_ONLY=1 builds and installs without starting.
# Bucle de desarrollo: compila, instala por adb, arranca el juego y enseña su log.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APK="$("$ROOT/tools/build_apk.sh")"

PKG=io.github.iamwilmar.xenobladeport
adb shell am force-stop "$PKG" 2> /dev/null || true
adb install -r "$APK" > /dev/null
echo "installed: $(adb shell dumpsys package "$PKG" | grep -m1 lastUpdateTime | tr -d ' ')"
adb shell appops set "$PKG" MANAGE_EXTERNAL_STORAGE allow
source "$ROOT/tools/datos.sh"
# As if the folder had been chosen in the start screen: the NativeActivity reads it from there.
adb shell "run-as $PKG sh -c 'mkdir -p files && echo $DATA > files/carpeta.txt'"
if [ -n "${INSTALL_ONLY:-}" ]; then exit 0; fi
# Without the screen on, Android gives the app no window and the game does not start.
adb shell input keyevent KEYCODE_WAKEUP
adb shell rm -f "$DATA/log.txt"
adb logcat -c
adb shell am start -n "$PKG/android.app.NativeActivity" > /dev/null && echo "started"

# Wait until the game ends, crashes or the watchdog fires.
for _ in $(seq "$(( ${SECS:-60} / 2 ))"); do
  sleep 2
  adb shell "grep -qE 'CRASH|nnMain volvió|^error:' $DATA/log.txt 2> /dev/null" && break
  adb shell pidof "$PKG" > /dev/null || break
done
sleep 1

mkdir -p "$ROOT/work/runs"
LOG="$ROOT/work/runs/apk-$(date +%Y%m%d-%H%M%S).log"
adb shell cat "$DATA/log.txt" > "$LOG" 2> /dev/null || true
echo "--- logcat:" >> "$LOG"
adb logcat -d -s NieR:V DEBUG:V libc:E AndroidRuntime:E >> "$LOG" 2> /dev/null || true
tail -n "${TAIL:-80}" "$LOG"
echo "full log: $LOG"
