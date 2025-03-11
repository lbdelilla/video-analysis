import re
import textstat
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound 
from openai import OpenAI, APIError
import requests
import nltk
import os
import json
from datetime import datetime, timedelta
nltk.download('punkt')  
nltk.download('punkt_tab')  
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv

load_dotenv()

# Configuración del cliente de Notion
NOTION_API_KEY = os.getenv("NOTION_API_KEY")
DATABASE_ID = os.getenv("NOTION_DATABASE_ID") 
NOTION_URL = os.getenv("NOTION_URL")
NOTION_V = os.getenv("NOTION_VERSION") # Asegúrate de agregar esto a tu .env


def get_transcription(video_id):
    if not re.match(r'^[a-zA-Z0-9_-]{11}$', video_id):
        raise ValueError("El ID del video no tiene un formato válido.")
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['es', 'en'])
    text = " ".join([t['text'] for t in transcript])
    return text

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

def analyze_with_ai(text):
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        organization=os.getenv("OPENAI_API_ORGANIZATION"), 
        project=os.getenv("OPENAI_API_PROJECT")           
    )
    if not client.api_key:
        raise ValueError("No se encontró la clave API de OpenAI.")
    truncated = False
    if len(text) > 10000:
        text = text[:10000]
        truncated = True
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Eres un experto en educación en bootcamps de tecnología analizando una clase. Dando respuestas de máximo 2000 caracteres."},
            {"role": "user", "content": f"Analiza la claridad, la estructura y la interacción en esta clase: {text}. Identifica la cantidad de interlocutores que participan (número). Y genera un listado de recomendaciones para entregarle al mentor con mejoras sugeridas. Además en base al analisis selecciona una etiqueta de las siguientes como severidad del mismo ['Excelente', 'Bueno', 'Regular','Crítico']  Dame una respuesta de máximo 2000 caracteres."}
        ]
    )
    analysis = response.choices[0].message.content
    
    return analysis + ("\n\n⚠️ Nota: El texto fue truncado a 10000 caracteres." if truncated else "")


def extract_severity(text):
    match = re.search(r"\b(Excelente|Bueno|Regular|Crítico)\b", text, re.IGNORECASE)
    return match.group(0) if match else "Sin clasificación"


# Función principal para procesar videos de la semana anterior
def process_previous_week_videos():
    # Calcular las fechas de lunes a viernes de la semana anterior
    today = datetime.now()
    last_monday = today - timedelta(days=today.weekday() + 7)
    last_friday = last_monday + timedelta(days=4)
    
    # Formatear fechas en formato Notion (YYYY-MM-DD)
    start_date = last_monday.strftime("%Y-%m-%d")
    end_date = last_friday.strftime("%Y-%m-%d")

    # start_date = "2025-02-16" # For testing purposes
    # end_date = "2025-03-01" # For testing purposes

    url = f"{NOTION_URL}/{DATABASE_ID}/query"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Content-Type": "application/json",
        "Notion-Version": NOTION_V
    }

    # Consulta a la base de datos de Notion
    data = {
        "filter": {
            "and": [
                {
                    "property": "Video Date",
                    "date": {
                        "after": start_date
                    }
                },
                {
                    "property": "Video Date",
                    "date": {
                        "before": end_date
                    }
                }
            ]
        }
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None
    
    # Procesar cada elemento encontrado
def process_videos():
    notion_data = process_previous_week_videos()
    if not notion_data:
        return

    for page in notion_data["results"]:
        try:
            video_id = page["properties"]["Youtube ID"]["rich_text"][0]["text"]["content"]
            if not video_id:
                print(f"⚠️ Advertencia: No se encontró un 'Youtube ID' en la página {page['id']}")
                continue
            
            page_id = page["id"]

            # Análisis del video 
            transcript_text = get_transcription(video_id)
            clarity_metrics = analyze_clarity(transcript_text)
            ai_analysis = analyze_with_ai(transcript_text)
            severity = extract_severity(ai_analysis)


            # Actualizar la página en Notion con los resultados
            update_notion_page(page_id, clarity_metrics, ai_analysis, severity)
            print(f"✅ Procesado video {video_id}")

        except Exception as e:
            print(f"❌ Error procesando video {video_id}: {e}")

def add_comment_to_notion_page(page_id, analysis):
    url = f"https://api.notion.com/v1/blocks/{page_id}/children"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28"
    }

    if not analysis:
        print(f"❌ No se proporcionó análisis para agregar como comentario en la página {page_id}.")
        return

    MAX_LENGTH = 2000

    comments = []
    while len(analysis) > MAX_LENGTH:
        comments.append(analysis[:MAX_LENGTH])
        analysis = analysis[MAX_LENGTH:]
    comments.append(analysis)  

    # Agregar los comentarios uno por uno
    for i, comment in enumerate(comments):
        data = {
            "children": [
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {
                                    "content": comment
                                }
                            }
                        ]
                    }
                }
            ]
        }

        response = requests.patch(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            print(f"✅ Comentario {i + 1} agregado correctamente en la página {page_id}.")
        else:
            print(f"❌ Error {response.status_code} al agregar el comentario {i + 1} en la página {page_id}: {response.text}")


def update_notion_page(page_id, clarity_metrics, ai_analysis, severity):
    url = f"https://api.notion.com/v1/pages/{page_id}"
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28"
    }

    data = {
        "properties": {
            "Clarity Analysis": {
                "rich_text": [{"text": {"content": str(clarity_metrics)}}]
            },
             "AI Severity Analysis": {"select": {"name": severity}
            }
        }
    }

    add_comment_to_notion_page(page_id, ai_analysis)

    response = requests.patch(url, headers=headers, json=data)

    if response.status_code == 200:
        print(f"✅ Página {page_id} actualizada correctamente en Notion.")
    else:
        print(f"❌ Error {response.status_code} al actualizar la página {page_id}: {response.text}")


if __name__ == "__main__":
    process_videos()
