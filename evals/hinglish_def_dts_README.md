# Def-DTS-style probe on the same Hinglish sample

This is a local adaptation of [Lee et al., 2025](https://aclanthology.org/2025.findings-acl.1066/),
not a reproduction of the paper's reported experiments. It uses the same 42
invented messages and human reference as `hinglish_topic_demo.txt` and
`hinglish_topic_demo_reference.json`. No real chat, database, or Qdrant data is
read or modified.

For each message, `run_hinglish_def_dts_demo.py` gives Qwen3.5-4B up to two
preceding and three following messages. The model returns summaries of both
contexts and one of five Def-DTS intent labels. Code derives a topic boundary
only from `INTRODUCE_TOPIC` or `CHANGE_TOPIC`. Index 0 is excluded from the
comparison because it is the beginning of the conversation. The paper used
different models and prompt examples; this probe uses a locally quantized 4B
model and short custom examples.

| Run | Predicted boundary indices | Correct | Extra | Missed |
| --- | --- | ---: | ---: | ---: |
| Direct block prompt | 2, 13, 21, 30, 34 | 3 | 2 | 1 |
| Def-DTS-style initial | 3, 4, 10, 11, 12, 13, 17, 19, 21, 26, 33, 34, 38 | 4 | 9 | 0 |
| Def-DTS-style clarified | 2, 3, 4, 10, 11, 12, 13, 17, 19, 21, 23, 33, 38 | 3 | 10 | 1 |

The human reference is 4, 13, 21, 34. Both Def-DTS-style runs classified all
42 messages with no XML parse errors. The first took 98.3 seconds of model
generation; the second took 109.8 seconds. Several false boundaries just before
real shifts arise because the model treats a future change as if it happened at
the current message. The clarified prompt did not fix this. Intent labels only
say whether a shift occurs; they cannot identify message 34 as a resumption of
an earlier weather topic.

The two raw result files are `hinglish_def_dts_results_initial.json` and
`hinglish_def_dts_results.json`. They contain every per-message intent and the
model's context summaries. This small, synthetic probe does not estimate
accuracy on real Hinglish conversations.

To rerun from the repository root with the separate MLX environment described
in `hinglish_topic_demo_README.md`:

```sh
/tmp/mirror-topic-demo-venv/bin/python evals/run_hinglish_def_dts_demo.py
```
