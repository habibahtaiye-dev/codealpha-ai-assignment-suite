# CodeAlpha AI Assignment Suite

This repository contains four beginner-friendly projects:
1. Language Translation Tool (LibreTranslate-compatible API)
2. FAQ Chatbot (TF-IDF + cosine similarity)
3. AI Music Generation (small LSTM demo trained on synthetic note sequences)
4. Object Detection and Tracking (YOLO + OpenCV)

> Educational starter project. The music model is a demonstration, not a high-quality pretrained music generator. The translation feature needs internet access and a compatible translation API. Object detection downloads YOLO weights on first run.

## Requirements
- Python 3.10+
- For webcam use: a working webcam and camera permissions
- Internet access for translation API and initial YOLO weights download

## Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the web interface
```bash
streamlit run app.py
```
Open the local URL printed by Streamlit (usually http://localhost:8501).

## Run object detection separately
```bash
python object_detection.py
```
Press `q` to close the webcam window. To use a video file, set `VIDEO_SOURCE` in `object_detection.py` to a video path.

## Run the music demo separately
```bash
python music_generator.py
```
This trains a tiny LSTM on generated toy sequences and writes `generated_notes.mid` if `music21` is installed. It is intended to demonstrate the workflow; replace the synthetic dataset with licensed MIDI files for a real project.

## Translation API
The default endpoint is the public LibreTranslate demo endpoint. Public instances can have rate limits or require an API key. Set `LIBRETRANSLATE_URL` and optionally `LIBRETRANSLATE_API_KEY` as environment variables to use another instance:
```bash
# Windows PowerShell
$env:LIBRETRANSLATE_URL="https://your-libretranslate-instance/translate"
$env:LIBRETRANSLATE_API_KEY="your-key"
```
Never commit API keys or passwords to GitHub.

## GitHub submission steps
1. Sign in at https://github.com/ and click **+** → **New repository**.
2. Name it `codealpha-ai-assignment-suite` (or the required assignment name).
3. Choose Public or Private according to your instructor's instructions. Do not add a README during creation because this ZIP already contains one.
4. Extract this ZIP on your computer.
5. Open the extracted folder in VS Code, then open Terminal in that folder.
6. Run:
   ```bash
   git init
   git add .
   git commit -m "Add AI assignment projects"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/codealpha-ai-assignment-suite.git
   git push -u origin main
   ```
   Replace `YOUR-USERNAME` with your GitHub username. If Git asks you to sign in, complete the browser authentication prompt.
7. Open the repository page and verify that `README.md`, `app.py`, `requirements.txt`, and the other files appear.
8. Submit the repository URL through your assignment portal. If they ask for a live demo, deploy the Streamlit app using Streamlit Community Cloud and follow its current GitHub connection instructions.

## Troubleshooting
- If `streamlit` is not recognized, activate the virtual environment and reinstall requirements.
- If TensorFlow is difficult to install on your system, the music demo is optional and can be run separately in a compatible Python environment.
- If webcam access fails, check camera permissions or use a video file.
