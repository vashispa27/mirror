# Local Hinglish topic-boundary probe

The 42 messages in `hinglish_topic_demo.txt` are invented. The script passes
messages 0–29 and then 30–41 to a local, quantized Qwen3.5-4B model. The second
block receives the previous four messages and the first block's topic state.
The block edge at message 30 is not intended as a topic change.

Files:

- `hinglish_topic_demo.txt`: numbered synthetic messages to inspect.
- `hinglish_topic_demo_reference.json`: human-intended boundaries, not sent to the model.
- `hinglish_topic_demo_results_initial.json`: first run with a concrete JSON example in the prompt.
- `hinglish_topic_demo_results.json`: second run with a neutral output description; this is the current script's output.
- `run_hinglish_topic_demo.py`: repeatable local-only probe. It does not touch the real chat, databases, or Qdrant.

To rerun on an Apple Silicon Mac from the repository root:

```sh
.venv/bin/python -m venv /tmp/mirror-topic-demo-venv
/tmp/mirror-topic-demo-venv/bin/python -m pip install mlx-vlm
/tmp/mirror-topic-demo-venv/bin/python evals/run_hinglish_topic_demo.py
```

The script loads
`mlx-community/Qwen3.5-4B-MLX-4bit` from the Hugging Face cache or downloads it
on first use. Inference stays local; only model files are downloaded.

The current result places boundaries at 2, 13, 21, 30, and 34. The intended
boundaries are 4, 13, 21, and 34. It gets three boundary positions exactly,
starts weather two messages early, splits the ongoing project at block edge 30,
and fails to mark 34 as a return to the earlier weather topic. This one invented
example is a probe, not an accuracy estimate.
