from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from voice_twin.schemas import AudioFrame


async def paced_frames(frames, realtime: bool = True) -> AsyncIterator[AudioFrame]:
    for frame in frames:
        yield frame
        if realtime:
            await asyncio.sleep(frame.duration_ms / 1000.0)
