import logging
import socket
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Lock

import google.generativeai as genai
from deep_translator import GoogleTranslator
from flask import Flask, jsonify, render_template, request


app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ethiobox")

GEMINI_MODEL_NAME = "gemini-3.1-flash"
GEMINI_FALLBACK_MODEL_NAMES = [
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-flash-latest",
]
REQUEST_TIMEOUT_SECONDS = 45
TRANSLATION_TIMEOUT_SECONDS = 20

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


def get_gemini_model(model_name, api_key):
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
            "temperature": 0.7,
            "top_p": 0.95,
            "max_output_tokens": 1024,
        },
    )


def translate_text(text, source, target):
    def translate():
        return GoogleTranslator(source=source, target=target).translate(text)

    translated = run_with_timeout(
        "Translation",
        translate,
        TRANSLATION_TIMEOUT_SECONDS,
        stage=f"Translation: {source} to {target}",
    )
    if not translated:
        raise AppError(
            "Translation returned an empty response.",
            status_code=502,
            stage=f"Translation: {source} to {target}",
        )
    return translated


def ask_gemini(english_prompt, api_key):
    prompt = (
        "You are EthioBox, a helpful AI bridge for Amharic speakers. "
        "Answer the user's translated English message clearly, warmly, and directly. "
        "Keep the response in English because it will be translated back to Amharic.\n\n"
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
    payload = request.get_json(silent=True) or {}
    amharic_input = (payload.get("message") or "").strip()
    api_key = (payload.get("api_key") or "").strip()

    if not amharic_input:
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
        english_prompt = translate_text(amharic_input, source="amharic", target="english")
        english_response, model_used = ask_gemini(english_prompt, api_key)
        amharic_response = translate_text(
            english_response,
            source="english",
            target="amharic",
        )

        return jsonify(
            {
                "amharic_input": amharic_input,
                "english_prompt": english_prompt,
                "english_response": english_response,
                "amharic_response": amharic_response,
                "model_used": model_used,
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


def pick_port(preferred_port=5000):
    for port in range(preferred_port, preferred_port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return preferred_port


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=pick_port(), debug=True)
