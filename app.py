import re
import textstat
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import GenericProxyConfig
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled
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

proxy_url = os.getenv("PROXY_URL")
if not proxy_url:
    raise ValueError("❌ No se encontró la variable de entorno PROXY_URL.")
proxy_config = GenericProxyConfig(proxy_url=proxy_url)
api = YouTubeTranscriptApi(proxy_config=proxy_config)

def get_transcription(video_id):
    try:
        if not re.match(r'^[a-zA-Z0-9_-]{11}$', video_id):
            raise ValueError("❌ El ID del video no tiene un formato válido.")
        
        transcript = api.get_transcript(video_id, languages=['es', 'en'])
        text = " ".join([t['text'] for t in transcript])
        return text
    
    except NoTranscriptFound:
        raise RuntimeError("❌ No se encontró una transcripción para este video.")
    except TranscriptsDisabled:
        raise RuntimeError("❌ Las transcripciones están deshabilitadas para este video.")
    except Exception as e:
        raise RuntimeError(f"❌ Error al obtener la transcripción: {e}")
    

def analyze_clarity(text):
    try:
        if not text or not isinstance(text, str):
            raise ValueError("❌ El texto proporcionado no es válido o está vacío.")
        
        sentences = sent_tokenize(text)
        if not sentences:
            raise ValueError("❌ No se pudieron dividir las oraciones.")
        
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        filler_words = len(re.findall(r'\b(um|uh|eh|er|mmm|ah|umm|like|you know)\b', text, re.IGNORECASE))
        readability = textstat.flesch_reading_ease(text)
        interpretation = (
            "Fácil de leer" if readability > 60 else 
            "Difícil de leer" if readability < 30 else 
            "Moderado"
        )

        return {
            "avg_sentence_length": avg_sentence_length,
            "filler_words_count": filler_words,
            "readability_score": readability,
            "readability_interpretation": interpretation
        }

    except Exception as e:
        raise RuntimeError(f"❌ Error al analizar la claridad del texto: {e}")

def analyze_with_ai(text):
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        organization = os.getenv("OPENAI_API_ORGANIZATION")
        project = os.getenv("OPENAI_API_PROJECT")

        if not api_key:
            raise ValueError("❌ No se encontró la clave API de OpenAI.")
        if not organization:
            raise ValueError("❌ No se encontró el valor de la organización en las variables de entorno.")
        if not project:
            raise ValueError("❌ No se encontró el valor del proyecto en las variables de entorno.")

        client = OpenAI(
            api_key=api_key,
            organization=organization, 
            project=project
        )

        truncated = False
        if len(text) > 10000:
            text = text[:10000]
            truncated = True
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Eres un experto en educación en bootcamps de tecnología analizando una clase. Dando respuestas de máximo 2000 caracteres."},
                {"role": "user", "content": f"Analiza la claridad, la estructura y la interacción en esta clase: {text}. Identifica la cantidad de interlocutores que participan (número). Y genera un listado de recomendaciones para entregarle al mentor con mejoras sugeridas. Además en base al análisis selecciona una etiqueta de las siguientes como severidad del mismo ['Excelente', 'Bueno', 'Regular','Crítico']  Dame una respuesta de máximo 2000 caracteres."}
            ]
        )

        analysis = response.choices[0].message.content
        return analysis + ("\n\n⚠️ Nota: El texto fue truncado a 10000 caracteres." if truncated else "")

    except Exception as e:
        raise RuntimeError(f"❌ Error al analizar el texto con OpenAI: {e}")


def extract_severity(text):
    try:
        if not text or not isinstance(text, str):
            raise ValueError("❌ El texto proporcionado no es válido o está vacío.")
        
        match = re.search(r"\b(Excelente|Bueno|Regular|Crítico)\b", text, re.IGNORECASE)
        return match.group(0) if match else "Sin clasificación"
    
    except Exception as e:
        raise RuntimeError(f"❌ Error al extraer la severidad del texto: {e}")


# Función principal para procesar videos de la semana anterior
def process_previous_week_videos():
    try:
        # Validar las variables de entorno
        if not NOTION_URL:
            raise ValueError("❌ La variable NOTION_URL no está configurada.")
        if not DATABASE_ID:
            raise ValueError("❌ La variable DATABASE_ID no está configurada.")
        if not NOTION_API_KEY:
            raise ValueError("❌ La clave API de Notion (NOTION_API_KEY) no está configurada.")
        if not NOTION_V:
            raise ValueError("❌ La versión de Notion (NOTION_V) no está configurada.")

        # Calcular las fechas de lunes a viernes de la semana anterior
        today = datetime.now()
        last_monday = today - timedelta(days=today.weekday() + 7)
        last_friday = last_monday + timedelta(days=4)
        
        # Formatear fechas en formato Notion (YYYY-MM-DD)
        start_date = last_monday.strftime("%Y-%m-%d")
        end_date = last_friday.strftime("%Y-%m-%d")

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
            try:
                result = response.json()
                print(f"✅ Extraidos videos de la semana anterior {result}")
                if not result.get("results"):
                    print("⚠️ No se encontraron videos para la semana anterior.")
                return result
            except ValueError as e:
                raise RuntimeError(f"❌ Error al parsear la respuesta de Notion: {e}")

        elif response.status_code == 400:
            raise ValueError(f"❌ Error 400: Solicitud incorrecta. Revisa el formato de la consulta.")
        elif response.status_code == 401:
            raise PermissionError(f"❌ Error 401: No autorizado. Verifica la clave API de Notion.")
        elif response.status_code == 403:
            raise PermissionError(f"❌ Error 403: Prohibido. Verifica los permisos en la base de datos de Notion.")
        elif response.status_code == 404:
            raise FileNotFoundError(f"❌ Error 404: No se encontró la base de datos o el endpoint.")
        elif response.status_code == 429:
            raise RuntimeError(f"❌ Error 429: Límite de solicitudes excedido. Intenta más tarde.")
        elif response.status_code >= 500:
            raise RuntimeError(f"❌ Error {response.status_code}: Problema en el servidor de Notion. Intenta más tarde.")
        else:
            raise RuntimeError(f"❌ Error {response.status_code}: {response.text}")

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"❌ Error de conexión con Notion: {e}")

    except Exception as e:
        raise RuntimeError(f"❌ Error inesperado: {e}")
    
    # Procesar cada elemento encontrado
def process_videos():
    notion_data = process_previous_week_videos()
    if not notion_data or "results" not in notion_data:
        print("⚠️ No se encontraron datos válidos de Notion para procesar.")
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
        "Notion-Version": NOTION_V
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
        "Notion-Version": NOTION_V
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
