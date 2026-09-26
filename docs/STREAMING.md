# Streaming

Target transport is 24 kHz mono PCM16 in roughly 60 ms frames. Keep session state across chunks so pitch, energy and speaker identity do not reset at phrase boundaries.

Measure request receipt, first acoustic token, first PCM, first playback and completion. TTFA and real-time factor are release gates.
