#!/bin/sh
# Boot the alphaclash Android emulator.
#   scripts/emulator.sh            windowed (setup, watching)
#   scripts/emulator.sh -no-window headless (unattended runs)
export JAVA_HOME=/opt/homebrew/opt/openjdk@17
export ANDROID_HOME=/opt/homebrew/share/android-commandlinetools
export ANDROID_SDK_ROOT=$ANDROID_HOME
exec "$ANDROID_SDK_ROOT/emulator/emulator" -avd alphaclash -no-boot-anim -no-audio "$@"
