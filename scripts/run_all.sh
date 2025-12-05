#!/bin/bash

set -e  # exit if any command fails

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Project root: $ROOT_DIR"
cd "$ROOT_DIR"

########################################
# C++ IMPLEMENTATION
########################################
echo
echo "==============================="
echo " Running C++ implementation "
echo "==============================="
echo

cd "$ROOT_DIR/Project"

echo "[C++] Building with make..."
make

echo "[C++] Running C++ program (interactive)..."
./project

echo
echo "==============================="
echo " C++ run finished "
echo "==============================="
echo

########################################
# JAVA IMPLEMENTATION
########################################
echo
echo "==============================="
echo " Running Java implementation "
echo "==============================="
echo

cd "$ROOT_DIR/oop-java/src"

echo "[Java] Compiling..."
javac */*.java *.java

echo "[Java] Running Main (interactive)..."
java Main

echo
echo "==============================="
echo " Java run finished "
echo "==============================="
echo

########################################
# LISP IMPLEMENTATION
########################################
echo
echo "==============================="
echo " Running Lisp implementation "
echo "==============================="
echo

cd "$ROOT_DIR"

echo "[Lisp] Running ml_demo.lisp with SBCL..."
sbcl --load ml_demo.lisp

echo
echo "==============================="
echo " Lisp run finished "
echo "==============================="
echo
