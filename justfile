# https://x.com/nb4ld/status/2093621334601236483
# https://huggingface.co/maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal

start:
  docker run -d \
    --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=${HF_TOKEN}" \
    --ipc=host \
    --name=qwen38-27b-sglang \
    lmsysorg/sglang:nightly-dev-cu13-20260830-a1fe4e30 \
    python3 -m sglang.launch_server \
    --trust-remote-code \
    --model-path gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090 \
    --served-model-name qwen3.8-27b \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.90 \
    --attention-backend flashinfer \
    --max-running-requests 1 --cuda-graph-max-bs 1 \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
    --mamba-full-memory-ratio 5.61 \
    --mamba-radix-cache-strategy extra_buffer_lazy --mamba-ssm-dtype bfloat16 \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal \
    --speculative-draft-model-quantization modelopt_fp4 \
    --speculative-num-draft-tokens 8 \
    --chunked-prefill-size 1024 \
    --cuda-graph-bs-prefill 64 128 256 512 1024 \
    --enable-memory-saver \
    --enable-hierarchical-cache --hicache-ratio 2 --hicache-write-policy write_through \
    --host 0.0.0.0 --port 30000

stop:
  docker stop qwen38-27b-sglang
  docker rm qwen38-27b-sglang
