import os
import time
import random
import requests

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "gemini"
).lower()

OLLAMA_URL = "http://localhost:11434/api/generate"

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:4b-instruct-2507-q4_K_M"
)

# Groq configuration
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

# Current provider selected by the Streamlit UI
_current_provider = DEFAULT_PROVIDER


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


# ============================================================
# PROVIDER SELECTION
# ============================================================

def set_llm_provider(provider):
    """
    Set the LLM provider used by call_llm().

    Supported values:
        gemini
        groq
        ollama
    """

    global _current_provider

    provider = provider.lower().strip()

    if provider not in ["gemini", "groq", "ollama"]:
        raise ValueError(
            "Invalid LLM provider. Use 'gemini', 'groq', or 'ollama'."
        )

    _current_provider = provider


def get_llm_provider():
    return _current_provider


# ============================================================
# GEMINI
# ============================================================

def call_gemini(prompt):

    max_attempts = 3

    for attempt in range(max_attempts):

        try:

            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )

            return response.text

        except Exception as e:

            error = str(e)

            # Retry temporary Gemini errors
            if "429" in error or "503" in error:

                if attempt < max_attempts - 1:

                    wait_time = (
                        10 * (2 ** attempt)
                        + random.uniform(0, 2)
                    )

                    print(
                        f"Gemini API temporarily unavailable. "
                        f"Retrying in {wait_time:.1f} seconds..."
                    )

                    time.sleep(wait_time)

                else:

                    print(
                        "Gemini API failed after 3 attempts."
                    )

                    raise

            else:
                raise


# ============================================================
# GROQ
# ============================================================

def call_groq(prompt):

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. "
            "Add your Groq API key to the .env file."
        )

    max_attempts = 3

    for attempt in range(max_attempts):

        try:

            response = requests.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    "response_format": {
                        "type": "json_object"
                    },

                    # Keep agent responses reasonably small
                    "max_completion_tokens": 800,
                },
                timeout=120,
            )

            response.raise_for_status()

            data = response.json()

            return data["choices"][0]["message"]["content"]

        except requests.exceptions.HTTPError as e:

            # Handle Groq rate limit
            if response.status_code == 429:

                retry_after = response.headers.get(
                    "retry-after"
                )

                if retry_after:
                    try:
                        wait_time = float(retry_after)
                    except ValueError:
                        wait_time = 3
                else:
                    wait_time = 3

                if attempt < max_attempts - 1:

                    print(
                        f"Groq rate limit reached. "
                        f"Retrying in {wait_time:.1f} seconds..."
                    )

                    time.sleep(wait_time)
                    continue

            try:

                error_data = response.json()

                error_message = error_data.get(
                    "error",
                    {}
                ).get(
                    "message",
                    str(e)
                )

            except Exception:

                error_message = str(e)

            raise RuntimeError(
                f"Groq API error: {error_message}"
            )

        except requests.exceptions.ConnectionError:

            raise RuntimeError(
                "Could not connect to Groq."
            )

        except requests.exceptions.Timeout:

            raise RuntimeError(
                "Groq request timed out."
            )

        except Exception as e:

            raise RuntimeError(
                f"Groq error: {str(e)}"
            )


# ============================================================
# OLLAMA
# ============================================================

def call_ollama(prompt):

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
            timeout=300,
        )

        response.raise_for_status()

        data = response.json()

        return data["response"]

    except requests.exceptions.ConnectionError:

        raise RuntimeError(
            "Could not connect to Ollama. "
            "Make sure Ollama is running."
        )

    except requests.exceptions.Timeout:

        raise RuntimeError(
            "Ollama request timed out."
        )

    except Exception as e:

        raise RuntimeError(
            f"Ollama error: {str(e)}"
        )


# ============================================================
# COMMON LLM FUNCTION
# ============================================================

def call_llm(prompt):

    if _current_provider == "groq":

        return call_groq(prompt)

    if _current_provider == "ollama":

        return call_ollama(prompt)

    return call_gemini(prompt)

