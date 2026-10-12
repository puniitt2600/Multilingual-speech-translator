import os
from dotenv import load_dotenv
from groq import Groq
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load .env
load_dotenv()

# Get Groq API key
api_key = os.getenv("GROQ_API_KEY")

# Create Groq client
client = Groq(api_key=api_key)

# Create FastAPI app
app = FastAPI(title="Speech Translator API")

# CORS
app.add_middleware(CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class TranslationRequest(BaseModel):
    text: str
    target_language: str

# Language codes -> language names
LANGUAGES = {
     "en":"English",
     "hi": "Hindi",
     "ta": "Tamil",
     "te": "Telugu",
     "bn": "Bengali",
     "mr": "Marathi",
     "gu": "Gujarati",
     "pa": "Punjabi",
     "kn": "Kannada",
     "ml": "Malayalam",
     "fr": "French",}

# Home page
@app.get("/")
def home():
    return { "message": "Speech Translator API is running!"}

# Translation endpoint
@app.post("/translate")
def translate(request: TranslationRequest):
    try:
        # Get language name
        target_language = LANGUAGES.get(
            request.target_language,
            request.target_language
        )
        # Ask Groq model for translation
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                { "role": "system","content": f"""
                    You are a translation only system.
                    Your ONLY task is to translate the user's text into {target_language}.
                    IMPORTANT RULES:
                    1. Return ONLY the translated text.
                    2. Do NOT explain anything.
                    3. Do NOT show reasoning.
                    4. Do NOT show steps.
                    5. Do NOT provide alternative translations.
                    6. Do NOT add labels such as "Translation:".
                    7. Preserve the original meaning.
                    8. Use natural wording.
                    9. Use the correct writing system/script for {target_language}.
                    10. If the target language is Tamil, write ONLY in Tamil script.
                    11. If the target language is Hindi, write ONLY in Devanagari script.
                    12. If the target language is Telugu, write ONLY in Telugu script.
                    13. If the target language is Bengali, write ONLY in Bengali script.
                    14. If the target language is Punjabi, write ONLY in Gurmukhi script.
                    15. Never return English unless English is the requested target language.
                    Return ONLY the final translation."""},
                { "role": "user", "content": request.text}
            ],
             temperature=0  #tells LLM to produce most predictable and consistent translation
        )
        # Get translated text
        translated_text = response.choices[0].message.content.strip()
        # Return result
        return {
            "original_text": request.text,
            "translated_text": translated_text,
            "target_language": target_language
        }
    except Exception as e:
        return {
            "error": str(e)
        }