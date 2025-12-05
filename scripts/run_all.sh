#!/bin/bash

set -e  # exit if any command fails

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"


run_cpp() {
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

  cd "$ROOT_DIR"
}

run_java() {
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

  cd "$ROOT_DIR"
}

run_lisp() {
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
}

# --- main menu loop ---------------------------------------------------------

while true; do
  echo
  echo "==============================================="
  echo "   AI/ML Library Implementation Comparison"
  echo "==============================================="
  echo "Please select an implementation to run:"
  echo "  (1) Procedural (C/C++)"
  echo "  (2) Object-Oriented (Java)"
  echo "  (3) Functional (Lisp)"
  echo "  (4) Quit"
  echo
  read -rp "Enter option: " choice

  case "$choice" in
    1) run_cpp ;;
    2) run_java ;;
    3) run_lisp ;;
    4)
      echo "Exiting unified runner. Goodbye!"
      exit 0
      ;;
    *)
      echo "Invalid option. Please enter 1-4."
      ;;
  esac
done
