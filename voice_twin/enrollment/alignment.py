from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voice_twin.audio.resample import resample
from voice_twin.frontend.phonemizer import phonemize_text


@dataclass(slots=True)
class TokenAlignment:
    token: str
    start_s: float
    end_s: float
    word: str = ""

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


class ProportionalAligner:
    def align(self, audio: np.ndarray, sample_rate: int, text: str, language: str) -> list[TokenAlignment]:
        seq = phonemize_text(text, language)
        duration = len(audio) / sample_rate
        step = duration / max(1, len(seq.phones))
        return [
            TokenAlignment(phone, i * step, min(duration, (i + 1) * step))
            for i, phone in enumerate(seq.phones)
        ]


def _ctc_path(log_probs: np.ndarray, targets: list[int], blank: int) -> list[tuple[int, int]]:
    """Viterbi CTC alignment; returns inclusive frame spans per target token."""
    if not targets:
        return []
    ext: list[int] = [blank]
    for token in targets:
        ext.extend([token, blank])
    t_len, _ = log_probs.shape
    s_len = len(ext)
    neg = -1e30
    score = np.full((t_len, s_len), neg, dtype=np.float64)
    back = np.full((t_len, s_len), -1, dtype=np.int16)
    score[0, 0] = log_probs[0, blank]
    if s_len > 1:
        score[0, 1] = log_probs[0, ext[1]]
    for t in range(1, t_len):
        for s in range(s_len):
            candidates = [(score[t - 1, s], s)]
            if s > 0:
                candidates.append((score[t - 1, s - 1], s - 1))
            if s > 1 and ext[s] != blank and ext[s] != ext[s - 2]:
                candidates.append((score[t - 1, s - 2], s - 2))
            best_score, prev = max(candidates, key=lambda x: x[0])
            score[t, s] = best_score + log_probs[t, ext[s]]
            back[t, s] = prev
    state = s_len - 1 if score[-1, -1] >= score[-1, -2] else s_len - 2
    states = [state]
    for t in range(t_len - 1, 0, -1):
        state = int(back[t, state])
        states.append(state)
    states.reverse()
    spans: list[tuple[int, int]] = []
    for i in range(len(targets)):
        target_state = 2 * i + 1
        frames = [t for t, s in enumerate(states) if s == target_state]
        if not frames:
            spans.append((0, 0))
        else:
            spans.append((frames[0], frames[-1] + 1))
    return spans


class CTCForcedAligner:
    """Word-anchored CTC alignment followed by phoneme subdivision inside each word."""

    def __init__(self, model_id: str = "facebook/wav2vec2-base-960h", device: str = "cpu"):
        self.model_id = model_id
        self.device = device
        self._processor = None
        self._model = None

    def _load(self):
        if self._model is None:
            from transformers import AutoModelForCTC, AutoProcessor
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForCTC.from_pretrained(self.model_id).to(self.device).eval()
        return self._processor, self._model

    def align(self, audio: np.ndarray, sample_rate: int, text: str, language: str) -> list[TokenAlignment]:
        import torch

        processor, model = self._load()
        x = resample(np.asarray(audio, dtype=np.float32), sample_rate, 16000)
        normalized = " ".join(text.upper().strip().split())
        encoded = processor.tokenizer(normalized, add_special_tokens=False)
        target_ids = list(encoded.input_ids)
        if not target_ids:
            return ProportionalAligner().align(audio, sample_rate, text, language)
        inputs = processor(x, sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            logits = model(inputs.input_values.to(self.device)).logits[0]
            log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()
        blank = (
            processor.tokenizer.pad_token_id
            if processor.tokenizer.pad_token_id is not None
            else 0
        )
        spans = _ctc_path(log_probs, target_ids, blank)
        frame_s = (len(x) / 16000) / max(1, log_probs.shape[0])
        token_strings = processor.tokenizer.convert_ids_to_tokens(target_ids)

        word_spans: list[tuple[str, float, float]] = []
        current_tokens: list[str] = []
        start_frame: int | None = None
        end_frame = 0
        delimiter = getattr(processor.tokenizer, "word_delimiter_token", "|")
        for tok, (s, e) in zip(token_strings, spans):
            if tok == delimiter:
                if current_tokens and start_frame is not None:
                    word_spans.append(("".join(current_tokens), start_frame * frame_s, end_frame * frame_s))
                current_tokens, start_frame = [], None
                continue
            clean = tok.replace("▁", "")
            if clean:
                if start_frame is None:
                    start_frame = s
                current_tokens.append(clean)
                end_frame = e
        if current_tokens and start_frame is not None:
            word_spans.append(("".join(current_tokens), start_frame * frame_s, end_frame * frame_s))

        words = [w for w in text.split() if any(c.isalnum() for c in w)]
        if len(word_spans) != len(words):
            return ProportionalAligner().align(audio, sample_rate, text, language)

        aligned: list[TokenAlignment] = []
        for raw_word, (_, start, end) in zip(words, word_spans):
            seq = phonemize_text(raw_word, language)
            n = max(1, len(seq.phones))
            step = max(0.001, (end - start) / n)
            for i, phone in enumerate(seq.phones):
                aligned.append(
                    TokenAlignment(
                        token=phone,
                        start_s=start + i * step,
                        end_s=end if i == n - 1 else start + (i + 1) * step,
                        word=raw_word,
                    )
                )
        return aligned
