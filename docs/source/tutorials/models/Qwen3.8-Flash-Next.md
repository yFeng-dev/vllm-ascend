# Qwen3.8-Flash-Next (Experimental)

## 1 Introduction

!!! note

    The "PD Separation Deployment" mentioned in this document is provided only as an example/reference design to demonstrate the basic process of deploying Prefill and Decode in separated mode. It is not recommended for use in the production environment.

    **For the production environment on the customer side, [MindIE Motor](https://www.hiascend.com/document/detail/en/mindie/latest/quickstart/docs/en/user_guide/quick_start.md) + vLLM-Ascend is recommended.**

Qwen3.8-Flash-Next is a multimodal Mixture-of-Experts (MoE) model and an experimental preview of the architecture that will underpin Qwen4. Its language model combines Gated DeltaNet (GDN) and Qwen Sparse Attention (QSA), gated residual connections, Position Learning Enhancement (PLE), and a native Multi-Token Prediction (MTP) head.

This document is validated and written based on `vllm-ascend:v0.26.0rc`. The current model (Qwen3.8-Flash-Next) is first supported in this version.

!!! warning

    Automatic Prefix Caching is experimental for Qwen3.8-Flash-Next. Enable it
    with `--enable-prefix-caching --mamba-cache-mode align` and revalidate NPU
    memory capacity for the target context length and concurrency.

## 2 Supported Features

Refer to [supported features](../../user_guide/support_matrix/supported_features.md) to get the model's supported feature matrix.

Refer to [Feature Guide](../../user_guide/feature_guide/index.md) to get feature configuration details.

## 3 Prerequisites

### 3.1 Model Weight

The following model weights are available:

- `Qwen3.8-Flash-Next-w8a8-mtp` (Quantized version): requires 1 Atlas 800 A3 (64GB × 8) node. [https://www.modelscope.cn/models/Eco-Tech/Qwen3.8-Flash-Next-w8a8-mtp](https://www.modelscope.cn/models/Eco-Tech/Qwen3.8-Flash-Next-w8a8-mtp)

It is recommended to download the model weight to the shared directory of multiple nodes, such as /root/.cache/.

## 4 Installation

### 4.1 Docker Image Installation

The pre-built images listed below have been validated for this tutorial.

A pre-built Qwen3.8 A3 image is available in the [vllm-ascend repository](https://quay.io/repository/ascend/vllm-ascend?tab=tags&tag=latest). Select the image that matches the host operating system:

- openEuler: `quay.io/ascend/vllm-ascend:qwen3.8-next-a3-openeuler`
- Ubuntu: `quay.io/ascend/vllm-ascend:qwen3.8-next-a3`

The following example uses the AArch64 image and exposes 8 devices on an Atlas 800 A3 node. The service command in [Section 5.1](#51-single-node-online-deployment) uses the first eight devices.

=== "A3 series"

    Start the docker image on each node.

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:qwen3.8-next-a3
    export NAME=vllm-ascend-qwen38-flash-next

    docker pull "$IMAGE"

    docker run --rm \
        --name "$NAME" \
        --shm-size=16g \
        --net=host \
        --device /dev/davinci0 \
        --device /dev/davinci1 \
        --device /dev/davinci2 \
        --device /dev/davinci3 \
        --device /dev/davinci4 \
        --device /dev/davinci5 \
        --device /dev/davinci6 \
        --device /dev/davinci7 \
        --device /dev/davinci8 \
        --device /dev/davinci9 \
        --device /dev/davinci10 \
        --device /dev/davinci11 \
        --device /dev/davinci12 \
        --device /dev/davinci13 \
        --device /dev/davinci14 \
        --device /dev/davinci15 \
        --device /dev/davinci_manager \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /path/to/models:/models \
        -it "$IMAGE" bash
    ```

After entering the container, verify that vLLM and vLLM-Ascend can be imported:

```shell
python -c "import vllm, vllm_ascend; print('vllm and vllm-ascend are ready')"
```

### 4.2 Source Code Installation

If you don't want to use the docker image as above, you can also build all from source:

- Install `vllm-ascend` from source, refer to [installation](../../getting_started/installation.md#installation-existing-cann-install).

If you want to deploy a multi-node environment, you need to set up the environment on each node.

## 5 Online Service Deployment {: #5-online-service-deployment }

### 5.1 Single-Node Online Deployment

For single-node online deployment on Atlas 800 A3, the following DP1 × TP8 command was validated with an Ascend-compatible W8A8 checkpoint. It enables QSA Lightning Indexer, QSA Expand E3, MTP speculative decoding, Function Calling, and reasoning parsing.

Before starting the service:

- Replace `MODEL_PATH` with the directory recorded when downloading the model weight.
- Revalidate memory capacity before changing the context length, concurrency, or memory utilization.

=== "A3 series"

    The following example is for Atlas 800 A3.

    ```bash
    unset CPLUS_INCLUDE_PATH CPATH C_INCLUDE_PATH
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh

    export MODEL_PATH=/models/Qwen3.8-Flash-Next-w8a8-mtp
    vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port 8088 \
        --served-model-name qwen3.8-flash-next \
        --trust-remote-code \
        --quantization ascend \
        --tensor-parallel-size 8 \
        --data-parallel-size 1 \
        --data-parallel-size-local 1 \
        --data-parallel-start-rank 0 \
        --enable-expert-parallel \
        --max-model-len 135168 \
        --max-num-seqs 8 \
        --max-num-batched-tokens 4096 \
        --gpu-memory-utilization 0.93 \
        --enable-prefix-caching \
        --mamba-cache-mode align \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_xml \
        --reasoning-parser qwen3 \
        --compilation-config '{"cudagraph_capture_sizes":[4,8,12,16,20,24,28,32],"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}' \
        --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"fuse_norm_quant":false}}'
    ```

    Key parameter descriptions:

    - `--quantization ascend` loads the Ascend-compatible W8A8 checkpoint.
    - `--tensor-parallel-size 8` and `--data-parallel-size 1` configure the A3 single-DP TP8 topology.
    - `--enable-expert-parallel` enables expert parallelism for the MoE layers.
    - `--enable-prefix-caching --mamba-cache-mode align` enables Prefix Caching with aligned GDN and PLE state checkpoints.
    - The validated command does not set `--language-model-only`, so the vision encoder remains enabled and the service accepts both text and multimodal requests. For a text-only deployment, you may add `--language-model-only` to skip loading the vision encoder.
    - `--enable-auto-tool-choice --tool-call-parser qwen3_xml` enables automatic Function Calling with the Qwen3 XML parser.
    - `--reasoning-parser qwen3` separates reasoning from the final answer in the OpenAI-compatible response.
    - `--speculative-config` uses the model's MTP head to draft three tokens. `enforce_eager=true` keeps the MTP proposer in eager mode.
    - `--compilation-config` enables `FULL_DECODE_ONLY` ACLGraph for decode.
    - `fuse_norm_quant=false` selects the validated norm and quantization path for this W8A8 deployment.

### 5.2 Mooncake V2 Prefill/Decode Separation

For the MRV2 QSA cache layout, use `MooncakeConnectorV2` on both sides. It
preserves physical cache strides, transfers PLE state as a complete TP replica,
and maps sharded GDN state per layer. This requires the replicated-state and
single-head NHD fixes from [#19974](https://github.com/vllm-project/vllm-ascend/pull/19974)
and [#19824](https://github.com/vllm-project/vllm-ascend/pull/19824), together with
the Qwen model implementation. The legacy layerwise connector's global TP ratio
restriction does not apply to Mooncake V2.

The following configuration uses sixteen A3 devices: eight for Prefill and eight
for Decode. Set `P_TP` and `D_TP` to `8,8`, `8,4`, or `4,8`; a TP4 side uses DP2
with expert parallelism. Both sides must advertise the same topology. Retain the
model's shared cache allocation and standardized memory accounting; do not
restore the older page-strided allocation gates from earlier model revisions.

Set the model path, reachable host address, and network interface in each shell:

```bash
export MODEL_PATH=/models/Qwen3.8-Flash-Next-w8a8
export VLLM_HOST_IP="<host-ip>"
export HCCL_SOCKET_IFNAME="<network-interface>"
export GLOO_SOCKET_IFNAME=$HCCL_SOCKET_IFNAME
export TP_SOCKET_IFNAME=$HCCL_SOCKET_IFNAME
export VLLM_USE_V2_MODEL_RUNNER=1
export HCCL_BUFFSIZE=512
export OMP_NUM_THREADS=8
export P_TP=8 D_TP=8
```

For Prefill, set:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SIDE=prefill ROLE=kv_producer TP=$P_TP
export PORT=18100 KV_PORT=25100 DP_PORT=29410
```

For Decode, use a separate shell and set:

```bash
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export SIDE=decode ROLE=kv_consumer TP=$D_TP
export PORT=18200 KV_PORT=25200 DP_PORT=29420
```

Run the same command in each shell after setting that side's variables:

```bash
vllm serve "$MODEL_PATH" \
    --host "$VLLM_HOST_IP" --port "$PORT" \
    --served-model-name qwen38-pd --trust-remote-code --quantization ascend \
    --tensor-parallel-size "$TP" --data-parallel-size "$((8 / TP))" \
    --data-parallel-rpc-port "$DP_PORT" --enable-expert-parallel \
    --max-model-len 8192 --max-num-seqs 64 --max-num-batched-tokens 2048 \
    --block-size 1536 --gpu-memory-utilization 0.85 \
    --no-enable-prefix-caching --enforce-eager \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --additional-config '{"enable_cpu_binding":false,"ascend_compilation_config":{"fuse_norm_quant":false}}' \
    --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3,"enforce_eager":true}' \
    --kv-transfer-config "{\"kv_connector\":\"MooncakeConnectorV2\",\"kv_role\":\"$ROLE\",\"engine_id\":\"qwen38-$SIDE\",\"kv_port\":$KV_PORT,\"kv_connector_extra_config\":{\"prefill\":{\"tp_size\":$P_TP,\"dp_size\":$((8 / P_TP))},\"decode\":{\"tp_size\":$D_TP,\"dp_size\":$((8 / D_TP))}}}"
```

After both `/health` endpoints return HTTP 200, start the proxy from the
vLLM-Ascend repository root:

```bash
python examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
    --host 127.0.0.1 --port 18000 \
    --prefiller-hosts "$VLLM_HOST_IP" --prefiller-ports 18100 \
    --decoder-hosts "$VLLM_HOST_IP" --decoder-ports 18200
```

Send requests to the proxy's `/v1/chat/completions` endpoint with
`model="qwen38-pd"`. Validate long prompts exceeding one prefill chunk as well
as concurrent short requests. Check actual KV transfer metadata and backend
errors in addition to HTTP status and answer correctness. Recheck memory
capacity before increasing context length or concurrency.

## 6 Functional Verification

After the service is started, the model can be invoked by sending a prompt. The `chat/completions` API is supported. Use the `--served-model-name` you configured (`qwen3.8-flash-next` for `Qwen3.8-Flash-Next`).

### 6.1 Basic Chat Completion

```bash
curl http://127.0.0.1:8088/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.8-flash-next",
        "messages": [
            {"role": "user", "content": "Who are you?"}
        ],
        "max_tokens": 1024,
        "temperature": 0
    }'
```

Expected result: the service returns HTTP 200 OK.

### 6.2 Multimodal Chat Completion

The validated deployment accepts image-and-text input. Replace `<IMAGE_URL>` with an image URL accessible from the serving environment:

```bash
curl http://127.0.0.1:8088/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.8-flash-next",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "<IMAGE_URL>"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Describe this image."
                    }
                ]
            }
        ],
        "max_tokens": 1024,
        "temperature": 0
    }'
```

Expected result: the service returns HTTP 200 OK and describes the supplied image in `choices[0].message.content`.

### 6.3 Function Calling

Function Calling requires the following startup options, which are already included in [Section 5](#5-online-service-deployment):

```text
--enable-auto-tool-choice --tool-call-parser qwen3_xml
```

Send a request containing the available tools and set `tool_choice` to `auto`:

```bash
curl http://127.0.0.1:8088/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.8-flash-next",
        "messages": [
            {
                "role": "user",
                "content": "What is the weather in Beijing?"
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name"
                            }
                        },
                        "required": ["city"]
                    }
                }
            }
        ],
        "tool_choice": "auto",
        "max_tokens": 1024,
        "temperature": 0
    }'
```

When the model chooses the function, the parsed result appears in `choices[0].message.tool_calls`. The caller must execute the function and send its result back to the model in a follow-up message.

### 6.4 Reasoning Parser and Thinking Control

Reasoning parsing requires the following startup option, which is already included in [Section 5](#5-online-service-deployment):

```text
--reasoning-parser qwen3
```

Qwen3.8-Flash-Next uses thinking mode by default. To request reasoning explicitly, set `enable_thinking` to `true` (or omit it). The parser returns the reasoning separately from the final answer:

```bash
curl http://127.0.0.1:8088/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.8-flash-next",
        "messages": [
            {
                "role": "user",
                "content": "Who are you?"
            }
        ],
        "max_tokens": 1024,
        "temperature": 0,
        "top_p": 1,
        "chat_template_kwargs": {
            "enable_thinking": true
        }
    }'
```

The final answer is returned in `choices[0].message.content`, while the parsed reasoning is returned separately in the `reasoning_content` field of the response.

To disable thinking and request a direct answer, use the validated non-thinking request:

```bash
curl http://127.0.0.1:8088/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.8-flash-next",
        "messages": [
            {
                "role": "user",
                "content": "Who are you?"
            }
        ],
        "max_tokens": 1024,
        "temperature": 0,
        "top_p": 1,
        "chat_template_kwargs": {
            "enable_thinking": false
        }
    }'
```

## 7 Accuracy Evaluation

### 7.1 Using AISBench

1. Refer to [Using AISBench](../../developer_guide/evaluation/using_ais_bench.md) for evaluation setup and usage.

2. After execution, you can get the result. Here are the results of `Qwen3.8-Flash-Next` in `vllm-ascend:v0.26.0rc` for reference only.

The A3 W8A8 deployment described in [Section 5.1](#51-single-node-online-deployment) was validated on GPQA Diamond with the following result.

| Hardware | Dataset | Metric | Score |
| --- | --- | --- | --- |
| Atlas 800 A3 | GPQA Diamond | Accuracy | 90.4 |

## 8 Performance Evaluation

### 8.1 Using AISBench

Refer to [Using AISBench for performance evaluation](../../developer_guide/evaluation/using_ais_bench.md#execute-performance-evaluation) for details.

### 8.2 Using vLLM Benchmark

Refer to [vllm benchmark](https://docs.vllm.ai/en/latest/benchmarking/) for more details.

## 9 Performance Tuning

### 9.1 Recommended Configurations

> For complete startup commands and parameter descriptions, please refer to the deployment examples in [Section 5](#5-online-service-deployment).

### 9.2 Tuning Guidelines

#### 9.2.1 General Tuning Reference

Please refer to the [Public Performance Tuning Documentation](../../developer_guide/performance_and_debug/optimization_and_tuning.md) for tuning methods.

Please refer to the [Feature Matrix](../../user_guide/support_matrix/feature_matrix.md) for detailed feature descriptions.

## 10 FAQ

For common environment, installation, and parameter issues, refer to the [Public FAQs](../../faqs.md).
