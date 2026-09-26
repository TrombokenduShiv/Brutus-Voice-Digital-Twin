# Training

## Global training

Train reusable HDVR, accent, prosody and event modules on a multi-speaker, multi-style corpus. Never train the reusable system only on one target speaker.

The default compound objective is:

- acoustic 1.00
- speaker 0.30
- pitch 0.20
- duration 0.15
- pause 0.25
- accent 0.15
- event 0.10
- SSL perceptual 0.30
- channel adversarial 0.05

## Speaker adaptation

Freeze the foundation model and HDVR backbone initially. Adapt target VoiceDNA memories and lightweight LoRA/adapters. The supplied default starts at rank 16, alpha 32, learning rate 5e-5 and at most 8,000 steps with early stopping.

Data must be split by session/day/sentence, not random adjacent chunks.
