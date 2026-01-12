from transformers import AutoTokenizer
from argparse import ArgumentParser
from guardbench import benchmark
from vllm import LLM, SamplingParams
import math


def moderate(
    conversations: list[list[dict[str, str]]],  # MANDATORY!
    llm: LLM,
    sampling_params: SamplingParams,
    tokenizer: AutoTokenizer,
    unsafe_token_id: int,
) -> list[float]:
    # Llama Guard does not support conversation starting with the assistant
    # Therefore, we drop the first utterance if it is from the assistant
    for i, x in enumerate(conversations):
        if x[0]["role"] == "assistant":
            conversations[i] = x[1:]

    resp = llm.chat(
        messages=conversations,
        sampling_params=sampling_params,
    )

    assert len(resp) == len(conversations), "We should get a response for each prompt"
    assert len(resp[0].outputs) == 1, "We request only 1 sequence per prompt"
    assert len(resp[0].outputs[0].logprobs) == 1, "We request only 1 token per prompt"
    assert len(resp[0].outputs[0].logprobs[0]) == tokenizer.vocab_size, "We request logprobs for all tokens"

    return [math.exp(item.outputs[0].logprobs[0][unsafe_token_id].logprob) for item in resp]

def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--safe_keyword", type=str, default="safe")
    parser.add_argument("--unsafe_keyword", type=str, default="unsafe")
    args = parser.parse_args()
    #     [
    #         "--model", "meta-llama/LlamaGuard-7b",
    #         "--datasets", "advbench_behaviors", 
    #         "--output_dir", "outputs_eldar_sweep",
    #     ]
    # )

    model = args.model
    datasets = args.datasets
    if len(datasets) == 1 and datasets[0] == "all":
        datasets = "all"
    batch_size = args.batch_size
    output_dir = args.output_dir
    tp = args.tp
    safe_keyword = args.safe_keyword
    unsafe_keyword = args.unsafe_keyword

    tokenizer = AutoTokenizer.from_pretrained(model)
    safe_token_id = tokenizer.encode(safe_keyword, add_special_tokens=False)
    assert len(safe_token_id) == 1, f"tokenizer.encode({safe_keyword}) should return a list of length 1 but got {safe_token_id}"
    safe_token_id = safe_token_id[0]
    unsafe_token_id = tokenizer.encode(unsafe_keyword, add_special_tokens=False)
    assert len(unsafe_token_id) == 1, f"tokenizer.encode({unsafe_keyword}) should return a list of length 1 but got {unsafe_token_id}"
    unsafe_token_id = unsafe_token_id[0]

    model = LLM(
        model=model,
        tensor_parallel_size=tp,
        gpu_memory_utilization=0.9,
        max_logprobs=tokenizer.vocab_size,
    )

    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        logprobs=tokenizer.vocab_size,
    )

    benchmark(
        moderate=moderate,
        model_name=model,
        batch_size=batch_size,
        datasets=datasets,
        out_dir=output_dir,
        # Moderate kwargs - the following arguments are given as input to `moderate`
        llm=model,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        unsafe_token_id=unsafe_token_id,
    )

if __name__ == "__main__":
    main()