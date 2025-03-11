# Video Analysis with OpenAI and YouTube Transcript API

Este proyecto permite obtener la transcripción de un video de YouTube, analizar su claridad y evaluar su calidad con la API de OpenAI.

## Características

- **Obtención de transcripción:** Usa `youtube_transcript_api` para extraer los subtítulos del video.
- **Análisis de claridad:** Evalúa la longitud promedio de las oraciones, las palabras de relleno y la legibilidad del texto.
- **Análisis con OpenAI:** Realiza una evaluación de la estructura y claridad usando `gpt-4`.

## Requisitos

Asegúrate de tener instalado Python y los siguientes paquetes:

```bash
pip install -r requirements.txt
```

## Configuración

Crea un archivo `.env` en la raíz del proyecto y agrega tu clave de API de OpenAI:

```
OPENAI_API_KEY=tu_clave_aqui
OPENAI_API_PROJECT =tu_clave_aqui
OPENAI_API_ORGANIZATION =tu_clave_aqui
```

## Uso

Ejecuta el script con:

```bash
python index.py 

```

El script analizará un video de YouTube basado en su ID y mostrará los resultados en la consola.

## Estructura del Proyecto

```
.
├── .venv/                   # Entorno virtual (ignorado en .gitignore)
├── index.py                 # Script principal
├── requirements.txt         # Dependencias del proyecto
├── .env                     # Variables de entorno (ignorado en .gitignore)
└── README.md                # Documentación
```

## Notas
- El script trunca el texto a 5000 caracteres si es demasiado largo.
- Si no hay transcripción disponible, se muestra un mensaje de error.
- Se recomienda probar con diferentes videos para evaluar su funcionalidad.
- Genera un txt con el análisis. 

## Contribuciones

Si deseas contribuir, abre un **pull request** o crea un **issue** con tus sugerencias.

---

© 2025 - Proyecto de Análisis de Video

