from __future__ import annotations

import argparse
import os

from voice_twin.acoustic.gemini_voice import GeminiVoiceManager
from voice_twin.profiles.store import VoiceProfileStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice-id", required=True, help="Local VoiceDNA profile id")
    parser.add_argument("--source-audio", required=True)
    parser.add_argument("--consent-audio", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--stateless", action="store_true")
    args = parser.parse_args()

    key = os.getenv("BVT_PROFILE_KEY")
    store = VoiceProfileStore(
        os.getenv("BVT_PROFILE_ROOT", "artifacts/voices"),
        encryption_key=key.encode() if key else None,
    )
    profile = store.load(args.voice_id)
    remote = GeminiVoiceManager().create_replicated_voice(
        source_audio=args.source_audio,
        consent_audio=args.consent_audio,
        display_name=args.display_name,
        store=not args.stateless,
    )
    profile.bind_provider(
        "gemini",
        kind="replicated",
        voice_id=remote if remote.startswith("voice_") else None,
        voice_key=remote if remote.startswith("voicekey_") else None,
        model="gemini-3.8-flash-tts",
    )
    store.save(profile)
    print({"voice_id": args.voice_id, "provider": "gemini", "binding": remote})


if __name__ == "__main__":
    main()
