"""Construct the component bundle (Ctx) from a config. Heavy imports happen here only."""
import os

from .features import FeatureExtractor
from .systems import Ctx


def load_ctx(cfg, index_dir=None, corpus_path=None, estimator=None, controller=None, conf_model=None,
             with_llm=True, with_verifier=True, device="cuda"):
    from .llm import LLM
    from .models import NLI, Reranker
    from .retrieval import Retriever
    from .verifier import Verifier

    m = cfg["models"]
    reranker = Reranker(m["reranker"], device)
    nli = NLI(m["nli"], device)
    llm = LLM(m["generator"], m["vllm_gpu_mem_util"], m["max_model_len"], cfg["seed"]) if with_llm else None
    retriever = Retriever(cfg, corpus_path, index_dir, reranker, device=device) if index_dir else None
    fx = FeatureExtractor(reranker, nli, llm, cfg["retrieval"]["n_ctx"], cfg["estimator"]["use_llm_sufficiency_feature"])
    ver = Verifier(nli, cfg["eval"]["support_threshold"]) if with_verifier else None
    from .controller import Controller
    from .estimator import Estimator
    from .systems import ConfModel

    est = Estimator.load(estimator, device="cpu") if isinstance(estimator, str) else estimator
    ctl = Controller.load(controller) if isinstance(controller, str) else controller
    cm = ConfModel.load(conf_model) if isinstance(conf_model, str) and os.path.exists(conf_model) else None
    return Ctx(cfg, retriever, llm, fx, est, ver, ctl, cm)
