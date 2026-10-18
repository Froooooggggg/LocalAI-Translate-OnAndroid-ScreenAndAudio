#!/usr/bin/env bash
set -u
RAW=https://raw.githubusercontent.com/ggml-org/llama.cpp/master

echo "### raw docs/android.md (status) ###"
curl -s -o /tmp/android.md -w "http=%{http_code} bytes=%{size_download}\n" --max-time 40 "$RAW/docs/android.md"
head -80 /tmp/android.md

echo
echo "### llama.h state/memory seq API ###"
curl -s --max-time 60 "$RAW/include/llama.h" -o /tmp/llama.h
wc -l /tmp/llama.h
grep -n "state_seq\|memory_seq\|_seq_\|recurrent" /tmp/llama.h | head -60

echo
echo "### qwen35.cpp: first 45 lines ###"
curl -s --max-time 60 "$RAW/src/models/qwen35.cpp" -o /tmp/qwen35.cpp
wc -l /tmp/qwen35.cpp
head -45 /tmp/qwen35.cpp

echo
echo "### qwen35.cpp: recurrent/state/cache/fa mentions ###"
grep -n "recurrent\|n_embd_state\|state\|cache\|flash\|SWA\|shift" /tmp/qwen35.cpp | head -40
