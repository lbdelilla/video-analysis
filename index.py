import re
import textstat
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound 
from openai import OpenAI, APIError
import nltk
import os
nltk.download('punkt')  
nltk.download('punkt_tab')  
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv

load_dotenv()

# Función para obtener la transcripción de un video de YouTube
def get_transcription(video_id):
    if not re.match(r'^[a-zA-Z0-9_-]{11}$', video_id):
        raise ValueError("El ID del video no tiene un formato válido.")
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['es', 'en'])
    text = " ".join([t['text'] for t in transcript])
    return text

# Función para analizar claridad y pausas
def analyze_clarity(text):
    sentences = sent_tokenize(text)
    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
    filler_words = len(re.findall(r'\b(um|uh|eh|er|mmm|ah|umm|like|you know)\b', text, re.IGNORECASE))
    readability = textstat.flesch_reading_ease(text)
    interpretation = "Fácil de leer" if readability > 60 else "Difícil de leer" if readability < 30 else "Moderado"
    return {
        "avg_sentence_length": avg_sentence_length,
        "filler_words_count": filler_words,
        "readability_score": readability,
        "readability_interpretation": interpretation
    }

# Función para hacer un análisis de IA sobre la calidad
def analyze_with_ai(text):
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        organization=os.getenv("OPENAI_API_ORGANIZATION"), 
        project=os.getenv("OPENAI_API_PROJECT")           
    )
    if not client.api_key:
        raise ValueError("No se encontró la clave API de OpenAI.")
    truncated = False
    if len(text) > 5000:
        text = text[:5000]
        truncated = True
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Eres un experto en educación analizando una clase."},
            {"role": "user", "content": f"Analiza la claridad, la estructura y la interacción en esta clase: {text}"}
        ]
    )
    analysis = response.choices[0].message.content
    return analysis + ("\n\n⚠️ Nota: El texto fue truncado a 5000 caracteres." if truncated else "")

# ID del video de YouTube
video_id = "enffQXn2hKI"

try:
    transcript_text = get_transcription(video_id)
    clarity_metrics = analyze_clarity(transcript_text)
    ai_analysis = analyze_with_ai(transcript_text)

    with open("class_review.txt", "w", encoding="utf-8" ) as file:
        file.write(f"""
               📌 Análisis de claridad: {clarity_metrics}
               🤖 Análisis de IA:    {ai_analysis}""")
    
    print("📌 Análisis de claridad:", clarity_metrics)
    print("🤖 Análisis de IA:", ai_analysis)
except NoTranscriptFound:  # Cambiar a NoTranscriptFound
    print("Error: No se encontró transcripción para este video.")
except APIError as e:
    print("Error en la API de OpenAI:", e)
except Exception as e:
    print("Error inesperado:", e)