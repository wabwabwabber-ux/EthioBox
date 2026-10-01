import asyncio
import base64
import logging
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from threading import Lock

import edge_tts
import google.generativeai as genai
from deep_translator import GoogleTranslator
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024
# Platforms such as Render provide PORT and expect the process to listen on
# the container interface rather than its loopback interface.
APP_HOST = os.environ.get("HOST") or (
    "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voca")

GEMINI_MODEL_NAME = "gemini-3.1-flash"
GEMINI_FALLBACK_MODEL_NAMES = [
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-flash-latest",
]
REQUEST_TIMEOUT_SECONDS = 45
TRANSLATION_TIMEOUT_SECONDS = 20
TTS_TIMEOUT_SECONDS = 45
TRANSCRIPTION_TIMEOUT_SECONDS = 60
MAX_AUDIO_UPLOAD_BYTES = 12 * 1024 * 1024
TTS_VOICE = "am-ET-MekdesNeural"

# Google's free translate endpoint (via deep_translator) rate-limits and blocks
# without warning. Default is OFF: translation goes through Gemini instead.
# Set USE_GOOGLE_TRANSLATE=1 in your environment to try Google first again.
USE_GOOGLE_TRANSLATE = os.environ.get("USE_GOOGLE_TRANSLATE") == "1"

SUPPORTED_AUDIO_MIME_TYPES = {
    "audio/webm",
    "audio/mp4",
    "audio/ogg",
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
}
EXTENSION_MIME_TYPES = {
    ".webm": "audio/webm",
    ".mp4": "audio/mp4",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
}

executor = ThreadPoolExecutor(max_workers=8)
# google-generativeai keeps its configured key globally. Serializing just that
# configuration/request pair prevents one browser-provided key being used by
# another concurrent request.
gemini_lock = Lock()


class AppError(Exception):
    def __init__(self, message, status_code=500, stage="Application", detail=None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.stage = stage
        self.detail = detail


def clean_error_detail(error, api_key=None):
    detail = str(error) or error.__class__.__name__
    if api_key:
        detail = detail.replace(api_key, "[hidden]")
    return detail[:900]


def run_with_timeout(label, fn, timeout, stage, api_key=None):
    future = executor.submit(fn)
    try:
        return future.result(timeout=timeout)
    except TimeoutError as exc:
        future.cancel()
        raise AppError(
            f"{label} took too long. Please try again in a moment.",
            status_code=504,
            stage=stage,
            detail=f"Timed out after {timeout} seconds.",
        ) from exc
    except AppError:
        raise
    except Exception as exc:
        detail = clean_error_detail(exc, api_key)
        logger.warning("%s failed: %s", label, detail)
        raise AppError(
            f"{label} failed. Please try again.",
            status_code=502,
            stage=stage,
            detail=detail,
        ) from exc


def get_gemini_model(model_name, api_key, temperature=0.7):
    if not api_key or api_key == "YOUR_KEY_HERE":
        raise AppError(
            "A Gemini API key is required. Add one in Settings to continue.",
            status_code=400,
            stage="Gemini",
        )

    # The client supplies the key it has saved in localStorage with each request.
    # It is never read from the server environment or included in a response.
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name,
        generation_config={
            "temperature": temperature,
            "top_p": 0.95,
            "max_output_tokens": 1024,
        },
    )


def translate_with_gemini(text, source, target, api_key):
    """Translate with Gemini. Used when Google's free endpoint is unavailable."""
    if target == "amharic":
        instruction = (
            "Translate the following English text into natural, simple Amharic "
            "written in Ge'ez script. It will be read aloud by a text-to-speech voice, "
            "so use plain text only: no markdown, no emojis, no English words."
        )
    else:
        instruction = (
            f"Translate the following {source.capitalize()} text into clear English."
        )
    prompt = (
        f"{instruction} Return only the translation, with no notes or commentary.\n\n"
        f"{text}"
    )

    errors = []
    model_names = [GEMINI_MODEL_NAME, *GEMINI_FALLBACK_MODEL_NAMES]
    for model_name in model_names:
        def generate():
            with gemini_lock:
                model = get_gemini_model(model_name, api_key, temperature=0.2)
                return model.generate_content(
                    prompt,
                    request_options={"timeout": REQUEST_TIMEOUT_SECONDS},
                )

        try:
            response = run_with_timeout(
                f"Gemini translation ({model_name})",
                generate,
                REQUEST_TIMEOUT_SECONDS + 5,
                stage=f"Translation: {source} to {target}",
                api_key=api_key,
            )
        except AppError as exc:
            errors.append(f"{model_name}: {exc.detail or exc.message}")
            if exc.status_code == 400:  # e.g. missing key: retrying won't help
                raise
            continue

        result = extract_gemini_text(response)
        if result:
            return result
        errors.append(f"{model_name}: empty response")

    raise AppError(
        "Translation failed. Please try again.",
        status_code=502,
        stage=f"Translation: {source} to {target}",
        detail=" | ".join(errors),
    )


def translate_text(text, source, target, api_key=None):
    if USE_GOOGLE_TRANSLATE:
        def translate():
            return GoogleTranslator(source=source, target=target).translate(text)

        try:
            translated = run_with_timeout(
                "Translation",
                translate,
                TRANSLATION_TIMEOUT_SECONDS,
                stage=f"Translation: {source} to {target}",
            )
            if translated:
                return translated
        except AppError as exc:
            logger.warning("Google Translate failed (%s); using Gemini", exc.detail)

    return translate_with_gemini(text, source, target, api_key)


def ask_gemini(english_prompt, api_key):
    prompt = (
        "You are Voca, a helpful AI assistant for Amharic speakers. "
        "Answer the user's translated English message clearly, warmly, and directly. "
        "Keep the response in English because it will be translated back to Amharic. "
        "Use plain text only (no markdown, bullet symbols, or emojis) and keep it under about 100 words, "
        "because it will be read aloud.\n\n"
        f"User message:\n{english_prompt}"
    )

    errors = []
    model_names = [GEMINI_MODEL_NAME, *GEMINI_FALLBACK_MODEL_NAMES]

    for model_name in model_names:
        def generate():
            with gemini_lock:
                model = get_gemini_model(model_name, api_key)
                return model.generate_content(
                    prompt,
                    request_options={"timeout": REQUEST_TIMEOUT_SECONDS},
                )

        try:
            response = run_with_timeout(
                f"Gemini model {model_name}",
                generate,
                REQUEST_TIMEOUT_SECONDS + 5,
                stage=f"Gemini: {model_name}",
                api_key=api_key,
            )
        except AppError as exc:
            detail = exc.detail or exc.message
            errors.append(f"{model_name}: {detail}")

            can_try_fallback = (
                model_name != model_names[-1]
                and detail
                and (
                    "not found" in detail.lower()
                    or "not supported" in detail.lower()
                    or "404" in detail
                )
            )
            if can_try_fallback:
                continue
            raise

        gemini_text = extract_gemini_text(response)
        if gemini_text:
            return gemini_text, model_name

        errors.append(f"{model_name}: empty response")

    raise AppError(
        "No Gemini Flash model returned a usable response.",
        status_code=502,
        stage="Gemini",
        detail=" | ".join(errors),
    )


def extract_gemini_text(response):
    try:
        if getattr(response, "text", None):
            return response.text.strip()
    except Exception:
        pass

    chunks = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            text = getattr(part, "text", "")
            if text:
                chunks.append(text)

    return "\n".join(chunks).strip()


def synthesize_amharic_audio(text):
    """Generate a compact MP3 response with Microsoft's Amharic neural voice."""
    if not text:
        raise AppError("There is no Amharic text available to speak.", stage="Text to speech")

    async def synthesize():
        communicate = edge_tts.Communicate(text=text, voice=TTS_VOICE)
        audio_chunks = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.extend(chunk["data"])

        if not audio_chunks:
            raise AppError(
                "Text-to-speech returned no audio.",
                status_code=502,
                stage="Text to speech",
            )
        return base64.b64encode(audio_chunks).decode("ascii")

    return run_with_timeout(
        "Amharic text-to-speech",
        lambda: asyncio.run(synthesize()),
        TTS_TIMEOUT_SECONDS,
        stage="Text to speech",
    )


def get_audio_mime_type(audio_file):
    # Werkzeug exposes the bare MIME type in ``mimetype`` and codec parameters
    # separately; strip parameters too for FileStorage-compatible test doubles.
    mime_type = (audio_file.mimetype or "").lower().split(";", 1)[0].strip()
    if mime_type in SUPPORTED_AUDIO_MIME_TYPES:
        return mime_type

    extension = Path(audio_file.filename or "").suffix.lower()
    if extension in EXTENSION_MIME_TYPES:
        return EXTENSION_MIME_TYPES[extension]

    raise AppError(
        "Upload an audio recording in WebM, MP4, OGG, WAV, or MP3 format.",
        status_code=415,
        stage="Audio input",
    )


def transcribe_amharic_audio(audio_file, api_key):
    """Transcribe a small browser recording using Gemini's inline audio input."""
    mime_type = get_audio_mime_type(audio_file)
    audio_bytes = audio_file.stream.read(MAX_AUDIO_UPLOAD_BYTES + 1)
    if not audio_bytes:
        raise AppError("The recording was empty. Please try again.", status_code=400, stage="Audio input")
    if len(audio_bytes) > MAX_AUDIO_UPLOAD_BYTES:
        raise AppError(
            "Please keep voice recordings under 12 MB.",
            status_code=413,
            stage="Audio input",
        )

    prompt = (
        "Transcribe this recording exactly in Amharic. "
        "Return only the spoken Amharic text, with no translation, labels, or commentary."
    )
    audio_part = {"mime_type": mime_type, "data": audio_bytes}
    errors = []
    model_names = [GEMINI_MODEL_NAME, *GEMINI_FALLBACK_MODEL_NAMES]

    for model_name in model_names:
        def transcribe():
            with gemini_lock:
                model = get_gemini_model(model_name, api_key)
                return model.generate_content(
                    [prompt, audio_part],
                    request_options={"timeout": TRANSCRIPTION_TIMEOUT_SECONDS},
                )

        try:
            response = run_with_timeout(
                f"Gemini transcription model {model_name}",
                transcribe,
                TRANSCRIPTION_TIMEOUT_SECONDS + 5,
                stage=f"Audio transcription: {model_name}",
                api_key=api_key,
            )
        except AppError as exc:
            detail = exc.detail or exc.message
            errors.append(f"{model_name}: {detail}")
            can_try_fallback = (
                model_name != model_names[-1]
                and detail
                and (
                    "not found" in detail.lower()
                    or "not supported" in detail.lower()
                    or "404" in detail
                )
            )
            if can_try_fallback:
                continue
            raise

        transcript = extract_gemini_text(response)
        if transcript:
            return transcript.strip()
        errors.append(f"{model_name}: empty transcript")

    raise AppError(
        "No Gemini Flash model returned a usable Amharic transcript.",
        status_code=502,
        stage="Audio transcription",
        detail=" | ".join(errors),
    )


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify(
        {
            "ok": True,
            "primary_model": GEMINI_MODEL_NAME,
            "fallback_models": GEMINI_FALLBACK_MODEL_NAMES,
        }
    )


@app.post("/api/chat")
def chat():
    is_audio_upload = "audio" in request.files
    payload = request.form if is_audio_upload else (request.get_json(silent=True) or {})
    audio_file = request.files.get("audio") if is_audio_upload else None
    amharic_input = (payload.get("message") or "").strip()
    api_key = (payload.get("api_key") or "").strip()

    if not amharic_input and not audio_file:
        return jsonify({"error": "Please enter an Amharic message."}), 400

    if len(amharic_input) > 4000:
        return jsonify({"error": "Please keep your message under 4,000 characters."}), 413

    if not api_key:
        return jsonify(
            {
                "error": "A Gemini API key is required. Add one in Settings to continue.",
                "stage": "Gemini",
            }
        ), 400

    try:
        if audio_file:
            amharic_input = transcribe_amharic_audio(audio_file, api_key)

        if not amharic_input:
            raise AppError(
                "No Amharic speech was detected in that recording. Please try again.",
                status_code=422,
                stage="Audio transcription",
            )

        if len(amharic_input) > 4000:
            raise AppError(
                "Please keep the transcribed message under 4,000 characters.",
                status_code=413,
                stage="Audio transcription",
            )

        t0 = time.perf_counter()
        english_prompt = translate_text(
            amharic_input, source="amharic", target="english", api_key=api_key
        )
        t1 = time.perf_counter()
        english_response, model_used = ask_gemini(english_prompt, api_key)
        t2 = time.perf_counter()
        amharic_response = translate_text(
            english_response,
            source="english",
            target="amharic",
            api_key=api_key,
        )
        t3 = time.perf_counter()
        audio_data = None
        audio_error = None

        try:
            audio_data = synthesize_amharic_audio(amharic_response)
        except AppError as exc:
            # Preserve the completed text response if the external voice service is unavailable.
            logger.warning("Amharic text-to-speech unavailable: %s", exc.detail or exc.message)
            audio_error = exc.message
        t4 = time.perf_counter()

        timings = {
            "translate_in": round(t1 - t0, 2),
            "gemini_answer": round(t2 - t1, 2),
            "translate_out": round(t3 - t2, 2),
            "text_to_speech": round(t4 - t3, 2),
            "total_after_transcription": round(t4 - t0, 2),
        }
        logger.info("Timings (seconds): %s", timings)

        return jsonify(
            {
                "amharic_input": amharic_input,
                "english_prompt": english_prompt,
                "english_response": english_response,
                "amharic_response": amharic_response,
                "model_used": model_used,
                "audio_data": audio_data,
                "audio_error": audio_error,
                "timings": timings,
            }
        )
    except AppError as exc:
        return (
            jsonify(
                {
                    "error": exc.message,
                    "stage": exc.stage,
                    "detail": exc.detail,
                }
            ),
            exc.status_code,
        )
    except Exception:
        logger.exception("Unexpected application error")
        return jsonify({"error": "Something unexpected happened. Please try again."}), 500


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Route not found."}), 404


@app.errorhandler(HTTPException)
def http_error_as_json(error):
    """Keep framework-level request errors machine-readable for fetch clients."""
    return (
        jsonify(
            {
                "error": error.description or "The request could not be processed.",
                "stage": "HTTP request",
            }
        ),
        error.code or 500,
    )


@app.errorhandler(RequestEntityTooLarge)
def audio_upload_too_large(_):
    return jsonify(
        {
            "error": "Please keep audio uploads under 15 MB.",
            "stage": "Audio input",
        }
    ), 413


@app.errorhandler(Exception)
def unexpected_error(error):
    if isinstance(error, HTTPException):
        return http_error_as_json(error)
    logger.exception("Unhandled application error", exc_info=error)
    return jsonify({"error": "Something unexpected happened. Please try again."}), 500


def pick_port(host=APP_HOST, preferred_port=5000):
    for port in range(preferred_port, preferred_port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind((host, port))
                return port
            except OSError:
                continue
    return preferred_port


if __name__ == "__main__":
    configured_port = os.environ.get("PORT")
    port = int(configured_port) if configured_port else pick_port(APP_HOST)
    app.run(host=APP_HOST, port=port, debug=os.environ.get("FLASK_DEBUG") == "1")