from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from voice_twin.profiles.voice_dna import VoiceDNA
from voice_twin.schemas import SynthesisRequest


@dataclass(slots=True)
class DigitalTwinPlan:
    plan_id: str
    provider: str
    provider_voice_id: str | None
    provider_binding_kind: str | None
    style_instruction: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _style_instruction(profile: VoiceDNA, request: SynthesisRequest) -> str:
    style = profile.style_profile
    fragments = [
        "Preserve the enrolled speaker identity.",
        "Match the speaker's habitual regional pronunciation, rhythm, pause placement, "
        "pitch range, breath behavior, and conversational timing.",
    ]
    if request.emotion:
        fragments.append(f"Emotion target: {request.emotion}.")
    if style.get("description"):
        fragments.append(str(style["description"]))
    if style.get("pace"):
        fragments.append(f"Preferred pace: {style['pace']}.")
    if style.get("energy"):
        fragments.append(f"Preferred energy: {style['energy']}.")
    fragments.append(f"Accent conditioning strength: {request.accent_strength:.2f}.")
    fragments.append(f"Prosody conditioning strength: {request.prosody_strength:.2f}.")
    return " ".join(fragments)


def build_digital_twin_plan(
    profile: VoiceDNA,
    request: SynthesisRequest,
    provider: str,
) -> DigitalTwinPlan:
    binding = profile.provider_binding(provider) or {}
    payload = f"{profile.speaker_id}|{provider}|{request.text}|{request.seed}".encode()
    plan_id = hashlib.sha256(payload).hexdigest()[:16]
    return DigitalTwinPlan(
        plan_id=plan_id,
        provider=provider,
        provider_voice_id=binding.get("voice_id") or binding.get("voice_key"),
        provider_binding_kind=binding.get("kind"),
        style_instruction=_style_instruction(profile, request),
        metadata={
            "binding": binding,
            "mode": request.mode.value,
            "accent_cells": len(profile.accent_atlas),
            "prosody_memories": len(profile.prosody_memory),
            "event_signature": bool(profile.event_signature),
        },
    )
