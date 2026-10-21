"""vLLM wrapper and prompt templates."""
import numpy as np

ANSWER_SYSTEM = (
    "You answer maintenance and troubleshooting questions using ONLY the numbered passages provided.\n"
    "For every recommendation, cite the passage id in brackets, e.g. [P3].\n"
    "Label each statement as DOCUMENTED (stated in a passage) or INFERENCE (your reasoning).\n"
    "Do not add any maintenance action that is not stated in a passage.\n"
    "If the passages do not support an answer for the specified equipment model and revision, reply exactly: "
    "INSUFFICIENT_EVIDENCE and list what is missing."
)
NO_RAG_SYSTEM = "You answer maintenance and troubleshooting questions concisely and accurately."
SUFF_PROMPT = (
    "Question: {q}\n\nPassages:\n{ctx}\n\n"
    "Do the passages contain enough information to give a complete, supported answer to the question for the "
    "specified equipment? Answer with one word: Yes or No."
)
JUDGE_PROMPT = (
    "Question: {q}\nReference answer: {ref}\nCandidate answer: {ans}\n\n"
    "Is the candidate answer correct, i.e. consistent with the reference answer and not adding contradicting or "
    "unsupported actions? Answer with one word: Yes or No."
)
REWRITE_PROMPT = (
    "Question: {q}\n\nEvidence retrieved so far did not fully answer it. Unresolved aspects: {gaps}.\n"
    "Write ONE short search query (under 20 words) likely to find a manual passage that resolves them. "
    "Output only the query."
)
COMPLEXITY_PROMPT = (
    "Classify the question. A = answerable without documents; B = needs one document lookup; "
    "C = needs several documents or multi-step reasoning.\nQuestion: {q}\nAnswer with one letter."
)


def format_context(passages):
    lines = []
    for i, p in enumerate(passages, 1):
        meta = f"doc: {p.get('doc_id', '?')}"
        if p.get("revision"):
            meta += f", rev: {p['revision']}"
        if p.get("equipment_model"):
            meta += f", model: {p['equipment_model']}"
        lines.append(f"[P{i}] ({meta}) {p['text']}")
    return "\n\n".join(lines)


def answer_messages(question, equipment_model, passages):
    q = question if not equipment_model else f"Equipment: {equipment_model}\nQuestion: {question}"
    if passages is None:
        return [{"role": "system", "content": NO_RAG_SYSTEM}, {"role": "user", "content": q}]
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": f"Passages:\n{format_context(passages)}\n\n{q}"},
    ]


class LLM:
    def __init__(self, name, gpu_mem_util=0.45, max_model_len=16384, seed=0, tp=1):
        from vllm import LLM as VLLM

        self.llm = VLLM(
            model=name,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=max_model_len,
            seed=seed,
            tensor_parallel_size=tp,
            enable_prefix_caching=True,
        )
        self.tok = self.llm.get_tokenizer()

    def _fmt(self, messages):
        try:
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate(self, batch_messages, max_tokens=400, temperature=0.0):
        from vllm import SamplingParams

        if not batch_messages:
            return []
        prompts = [self._fmt(m) for m in batch_messages]
        sp = SamplingParams(max_tokens=max_tokens, temperature=temperature, logprobs=1)
        outs = self.llm.generate(prompts, sp, use_tqdm=False)
        res = []
        for o in outs:
            c = o.outputs[0]
            n = max(1, len(c.token_ids))
            res.append(
                {
                    "text": c.text.strip(),
                    "n_prompt": len(o.prompt_token_ids),
                    "n_gen": len(c.token_ids),
                    "mean_logprob": float(c.cumulative_logprob / n) if c.cumulative_logprob is not None else 0.0,
                }
            )
        return res

    def yes_prob(self, batch_messages):
        """P(Yes) / (P(Yes)+P(No)) from first-token log-probabilities."""
        from vllm import SamplingParams

        if not batch_messages:
            return []
        prompts = [self._fmt(m) for m in batch_messages]
        outs = self.llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0, logprobs=20), use_tqdm=False)
        res = []
        for o in outs:
            lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
            py = pn = 0.0
            for tid, l in lp.items():
                t = (l.decoded_token or "").strip().lower()
                if t == "yes":
                    py += float(np.exp(l.logprob))
                elif t == "no":
                    pn += float(np.exp(l.logprob))
            res.append(py / (py + pn) if (py + pn) > 0 else 0.5)
        return res

    def count_tokens(self, text):
        return len(self.tok.encode(text))
