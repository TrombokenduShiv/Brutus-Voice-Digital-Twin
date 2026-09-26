# Evaluation

A model is not accepted on a single speaker-similarity number.

Required dimensions: independent speaker encoders, WER/CER, F0 correlation and log-F0 RMSE, phoneme/word duration error, pause precision/recall/F1 and timing error, breath-event F1, accent/allophone tests, TTFA, real-time factor, robot over-air measurements, ABX tests and blinded real-vs-synthetic tests.

Identity Retention normalizes clone-to-target similarity by the target speaker's real-take self-similarity.

The thresholds in configs/evaluation/success_gates.yaml are engineering gates, not universal scientific constants. Replace them with empirically measured human self-baselines when enough repeated target recordings are available.
