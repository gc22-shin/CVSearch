#!/bin/bash

# ==============================================================================
# CVSearch Evaluation Script with Command-Line Arguments
# ==============================================================================

# 1. Default configurations
ROOT_PATH="/data1/workspaces/jgshin22/GraduationProject/IndustryVLM"
MODEL_PATH="models/Qwen3-VL-2B-Instruct"
BENCHMARK="vstar"

ANNOTATION_PATH="datasets/hr_data"
SAM_MODEL_PATH="models/sam3/sam3.pt"
NLP_MODEL_PATH="models/en_core_web_sm-3.8.0"

NUM_CHUNKS=1
CHUNK_IDX=0
SPLIT_NUM=4
GPU_ID=0
SAM_GPU_ID=1

# 2. Function to display help menu
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --model_path <path>    Path to the MLLM model (default: ${MODEL_PATH})"
    echo "  --benchmark <name>     Benchmark dataset to evaluate (default: ${BENCHMARK})"
    echo "                         Choices: vstar, hr-bench_4k, hr-bench_8k, mme-realworld-lite, etc."
    echo "  --gpu_id <id>          Physical GPU ID for the VLM (default: ${GPU_ID})"
    echo "  --sam_gpu_id <id>      Physical GPU ID for SAM3 (default: ${SAM_GPU_ID})"
    echo "  --num_chunks <int>     Total number of chunks for parallel execution (default: ${NUM_CHUNKS})"
    echo "  --chunk_idx <int>      Index of the current chunk (default: ${CHUNK_IDX})"
    echo "  --help                 Show this help message"
    echo ""
    echo "Example:"
    echo "  bash $0 --model_path models/Qwen3-VL-8B --benchmark hr-bench_8k --gpu_id 1"
    exit 1
}

# 3. Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_path) MODEL_PATH="$2"; shift ;;
        --benchmark) BENCHMARK="$2"; shift ;;
        --gpu_id) GPU_ID="$2"; shift ;;
        --sam_gpu_id) SAM_GPU_ID="$2"; shift ;;
        --num_chunks) NUM_CHUNKS="$2"; shift ;;
        --chunk_idx) CHUNK_IDX="$2"; shift ;;
        --help) usage ;;
        *) echo "❌ Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# 4. Set the GPU device
# VLM -> cuda:0, SAM3 -> cuda:1 (logical indices after remapping)
if [ "${GPU_ID}" = "${SAM_GPU_ID}" ]; then
    export CUDA_VISIBLE_DEVICES=${GPU_ID}
    SAM_DEVICE="cuda:0"
else
    export CUDA_VISIBLE_DEVICES=${GPU_ID},${SAM_GPU_ID}
    SAM_DEVICE="cuda:1"
fi

# ==============================================================================

echo "========================================================================"
echo "🚀 Starting CVSearch Evaluation"
echo "========================================================================"
echo "Model     : ${MODEL_PATH}"
echo "Benchmark : ${BENCHMARK}"
echo "GPU IDs    : ${CUDA_VISIBLE_DEVICES} (VLM=cuda:0, SAM3=${SAM_DEVICE})"
echo "Chunks    : ${CHUNK_IDX} / ${NUM_CHUNKS}"
echo "========================================================================"

# Execute the Python inference script for CVSearch
python perform_CVSearch.py \
    --root-path "${ROOT_PATH}" \
    --model-path "${MODEL_PATH}" \
    --benchmark "${BENCHMARK}" \
    --annotation_path "${ANNOTATION_PATH}" \
    --sam-model-path "${SAM_MODEL_PATH}" \
    --vlm-device "cuda:0" \
    --sam-device "${SAM_DEVICE}" \
    --nlp-model-path "${NLP_MODEL_PATH}" \
    --num-chunks ${NUM_CHUNKS} \
    --chunk-idx ${CHUNK_IDX} \
    --split-num ${SPLIT_NUM}

# ------------------------------------------------------------------------------
# Baseline Comparison
# If you want to run the Direct Answer (Vanilla Backbone) baseline,
# uncomment the following lines and comment out the Python command above:
# ------------------------------------------------------------------------------
# python perform_CVSearch.py \
#     --root-path "${ROOT_PATH}" \
#     --model-path "${MODEL_PATH}" \
#     --benchmark "${BENCHMARK}" \
#     --annotation_path "${ANNOTATION_PATH}" \
#     --direct-answer