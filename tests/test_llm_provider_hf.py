"""The local flan-t5 fallback must not depend on a removed pipeline task."""

from unittest.mock import MagicMock, patch

import torch


def test_seq2seq_generator_returns_pipeline_shape():
    from src.llm_provider import _Seq2SeqGenerator

    tokenizer = MagicMock()
    tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
    tokenizer.decode.return_value = "  an answer  "
    model = MagicMock()
    model.generate.return_value = torch.tensor([[0, 5, 6, 1]])

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
         patch("transformers.AutoModelForSeq2SeqLM.from_pretrained", return_value=model):
        gen = _Seq2SeqGenerator("google/flan-t5-small", max_new_tokens=7)
        out = gen("question?")

    assert out == [{"generated_text": "  an answer  "}]
    assert model.generate.call_args.kwargs["max_new_tokens"] == 7
    tokenizer.decode.assert_called_once()


def test_call_huggingface_strips_output():
    import src.llm_provider as lp

    fake = MagicMock(return_value=[{"generated_text": "  hi  "}])
    with patch.object(lp, "_get_hf_pipeline", return_value=fake):
        assert lp._call_huggingface("p", "s") == "hi"
