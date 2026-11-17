# Whisper Transcriber

A small Flask service that turns an audio file into a transcript with speaker
labels. Whisper handles the words, pyannote handles who said them.

## Stack

- **faster-whisper**, `large-v3`, on CUDA when a GPU is visible and on CPU otherwise.
- **pyannote.audio** for diarisation — the model is gated, so it needs a Hugging Face token with access to the pipeline.
- Flask plus flask-cors for the HTTP surface, and a single-page front end in `index.html`.

## Run it

```bash
pip install flask flask-cors faster-whisper pyannote.audio
set HF_TOKEN=hf_...        # PowerShell: $env:HF_TOKEN = "hf_..."
python app.py
```

Then open `index.html`, or point it at the ngrok URL if you expose the port.

The token is read from the environment. Do not commit one — `.gitignore` covers
`token.txt` and the ngrok binary for exactly that reason.

## Notes

- CORS is wide open on purpose: ngrok hands out a new hostname each run, so an origin allowlist would break every restart. Put this behind a fixed domain before treating it as a service.
- `large-v3` loads about 3 GB of weights into memory at startup, so the first request after boot is slow even on a warm machine.
