## Kernel Chat UI

Simple chat-style frontend for the kernel agent. The UI is isolated under `UI/`.

### Run

From repo root:

```bash
streamlit run UI/app.py
```

### Notes

- Uses the same OpenAI API key loaded from repo `.env`.
- Session keeps a rolling `current_cpp`; each turn refines the last generated kernel.
- Type `reset` in chat (or click **Reset to baseline**) to restart from `task/base_kernel.py`.
