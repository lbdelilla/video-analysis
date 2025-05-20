from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize
import re
import textstat
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled
from openai import OpenAI, APIError
import nltk
import os
nltk.download('punkt')
nltk.download('punkt_tab')

load_dotenv()


def get_video_metadata(video_id):
    try:
        # Obtener la transcripción formateada
        transcript = YouTubeTranscriptApi.get_transcript(
            video_id, languages=['es', 'en'])

        # Calcular la duración total del video
        duration = transcript[-1]['start'] + transcript[-1]['duration']

        # Convertir a minutos (con decimales)
        duration_minutes = round(duration / 60, 2)

        return {
            "formatted_transcript": transcript,
            "duration_minutes": duration_minutes,
            "total_segments": len(transcript)
        }
    except Exception as e:
        raise RuntimeError(f"❌ Error al obtener los metadatos del video: {e}")


def get_transcription(video_id):
    try:
        if not re.match(r'^[a-zA-Z0-9_-]{11}$', video_id):
            raise ValueError("El ID del video no tiene un formato válido.")

        # Obtener metadatos del video
        video_metadata = get_video_metadata(video_id)

        # Obtener el texto de la transcripción
        text = " ".join([t['text']
                        for t in video_metadata['formatted_transcript']])

        with open("class_transcript.txt", "w", encoding="utf-8") as file:
            file.write(f""" Transcripción:    {text}""")

        return {
            "text": text,
            "metadata": video_metadata
        }
    except NoTranscriptFound:
        raise RuntimeError(
            "❌ No se encontró una transcripción para este video.")
    except TranscriptsDisabled:
        raise RuntimeError(
            "❌ Las transcripciones están deshabilitadas para este video.")
    except Exception as e:
        print(f"❌ Error al procesar video {video_id}: {e}")
        raise RuntimeError(f"❌ Error al obtener la transcripción: {e}")

# Función para analizar claridad y pausas


def analyze_clarity(text):
    sentences = sent_tokenize(text)
    avg_sentence_length = sum(len(s.split())
                              for s in sentences) / len(sentences)
    filler_words = len(re.findall(
        r'\b(um|uh|eh|er|mmm|ah|umm|like|you know)\b', text, re.IGNORECASE))
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
    if len(text) > 10000:
        text = text[:10000]
        truncated = True
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un experto en educación en bootcamps de tecnología, analizando la transcripción de una clase para detectar posibles problemas o descontentos de los alumnos en función de su participación, dudas, nivel de comprensión y dinámica de la clase. "
                ),
            },
            {
                "role": "user",
                "content": f"""
                    Analiza la claridad, la estructura y la interacción en esta clase: {text}. 
                    
                    Formato de respuesta:

                    Análisis de la clase:
                    - Resumen general de la dinámica.
                    - Nivel de participación de los alumnos.
                    - Validación de conocimientos durante la clase (¿Se realizaron preguntas? ¿Los alumnos respondieron correctamente?).
                    - Principales dificultades detectadas.
                    - Posibles señales de descontento o desmotivación.
                    - Uso de ejemplos para explicar los temas

                    Recomendaciones para mejorar:
                    - Sugerencias concretas para hacer la clase más dinámica y participativa.
                    - Estrategias para mejorar la comprensión de los alumnos.
                    - Métodos para reforzar el contenido más complejo.

                    Severidad del análisis: 
                    - Excelente:  
                        - Alta participación de los alumnos.  
                        - Respuestas correctas en la validación del conocimiento.  
                        - Pocas o ninguna señal de descontento o desmotivación.  
                        - Fluidez en la explicación y estructura clara.  
                        - No se identifican problemas significativos en la dinámica.  

                    - Buena:  
                        - Participación aceptable, aunque algunos alumnos pueden no estar involucrados.  
                        - Validación del conocimiento con respuestas mayormente correctas.  
                        - Algunas dificultades detectadas, pero no críticas.  
                        - Puede haber áreas de mejora en la dinámica.  

                    - Regular:  
                        - Baja participación en general.  
                        - Respuestas incorrectas o dudas frecuentes en validaciones de conocimiento.  
                        - Señales de desmotivación o descontento en algunos alumnos.  
                        - Problemas en la estructura o ritmo de la clase.  

                    - Crítica:  
                        - Muy baja o nula participación de los alumnos.  
                        - Fallos constantes en la validación del conocimiento.  
                        - Múltiples señales de descontento o frustración entre los alumnos.  
                        - Explicaciones confusas o desorganizadas.  
                        - Necesita ajustes urgentes para mejorar la experiencia de aprendizaje.  


                    Contexto adicional: 
                    - Solo se analiza la parte teórica de la clase, por lo que este análisis no representa la experiencia completa de los alumnos.
                    - Es normal que los alumnos no comprendan todo en una sola clase, ya que manejan grandes volúmenes de información en poco tiempo.
                    - Los alumnos y profesores tienen acceso a material escrito antes de la clase, por lo que se debe considerar si el problema es falta de preparación previa.
                    - El objetivo es proporcionar recomendaciones prácticas para el docente y ayudar a priorizar la atención a las clases que presenten más problemas.
                    - La academia ofrece cursos de Full Stack, Data Science y Ciberseguridad. Las clases deben ser dinámicas y participativas, fomentando la validación del conocimiento por parte de los alumnos. Los profesores solo graban la parte teórica, mientras que la parte práctica se desarrolla con proyectos asignados.
                """
            }
        ]
    )

    analysis = response.choices[0].message.content
    return analysis + ("\n\n⚠️ Nota: La transcripción fue truncado a 10000 caracteres." if truncated else "")


# ID del video de YouTube para probar
video_id = "LA9zp7NgKnI"  # El video que estabas intentando analizar

try:
    # Obtener transcripción y metadatos
    transcript_data = get_transcription(video_id)
    transcript_text = transcript_data["text"]
    video_metadata = transcript_data["metadata"]

    # Mostrar los metadatos
    print("\n📊 Metadatos del video:")
    print(f"📹 Duración: {video_metadata['duration_minutes']} minutos")
    print(f"📝 Total de segmentos: {video_metadata['total_segments']}")

    # Análisis de claridad
    clarity_metrics = analyze_clarity(transcript_text)
    print("\n📌 Análisis de claridad:", clarity_metrics)

    # Análisis de IA
    ai_analysis = analyze_with_ai(transcript_text)
    print("\n🤖 Análisis de IA:", ai_analysis)

except NoTranscriptFound:
    print("Error: No se encontró transcripción para este video.")
except APIError as e:
    print("Error en la API de OpenAI:", e)
except Exception as e:
    print("Error inesperado:", e)
