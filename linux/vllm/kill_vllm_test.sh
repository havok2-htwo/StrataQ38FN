#!/bin/bash
# stop the test vLLM (port 8003) and its workers
for p in $(pgrep -f "api_server --model /media/raid0/Models/models--VnimanieAI"); do kill $p; done
sleep 8
for p in $(pgrep -f "VLLM::"); do kill -9 $p 2>/dev/null; done
for p in $(pgrep -f "api_server --model /media/raid0/Models/models--VnimanieAI"); do kill -9 $p; done
true
