# Chatbot Institucional - API Base & RAG Ingestion Service

API modular basada en **FastAPI**, **SQLAlchemy (Async)**, **pgvector** y **LangChain**, diseñada para la ingesta de documentos PDF, segmentación semántica (*chunking*), generación de *embeddings* y persistencia vectorial para arquitecturas RAG (*Retrieval-Augmented Generation*).

---

## 🏛️ Arquitectura Modular

```
chatbot/
├── app/
│   ├── main.py                     # Entrypoint FastAPI, lifespan, middlewares
│   ├── core/
│   │   ├── config.py               # Configuración global (.env) con Pydantic Settings
│   │   └── database.py             # Configuración Asyncpg, AsyncSession y soporte pgvector
│   ├── models/
│   │   ├── document.py             # Modelos Document y DocumentChunk (pgvector)
│   │   └── academic.py             # Modelos relacionales: Carrera, Materia, Alumno, Calificación
│   ├── schemas/
│   │   ├── document.py             # Esquemas Pydantic de documentos
│   │   └── chat.py                 # Esquemas Pydantic para /chat y ToolCallRecord
│   ├── services/
│   │   ├── embedding_service.py    # Servicio desacoplado de embeddings (OpenAI / Mock local)
│   │   ├── document_service.py     # Ingestión RAG, chunking y persistencia vectorial
│   │   ├── vector_search_service.py# Búsqueda semántica de chunks con pgvector y filtro por rol
│   │   ├── academic_service.py     # Dominio relacional y validaciones de privacidad RBAC
│   │   ├── academic_tools.py       # Definición de herramientas (Tools) y ContextVar seguro
│   │   └── chat_agent_service.py   # Orquestador del agente conversacional con Tool Calling
│   └── api/
│       └── v1/
│           ├── api.py              # Router central de versión 1
│           └── endpoints/
│               ├── documents.py    # Endpoint POST /api/v1/documents/upload y consultas
│               └── chat.py         # Endpoint POST /api/v1/chat con Tool Calling
├── tests/
│   └── test_conversational_agent.py# Suite de pruebas unitarias y de integración
├── requirements.txt                # Dependencias del proyecto
├── .env.example                    # Plantilla de variables de entorno
└── README.md
```

---

## 🚀 Requisitos Previos

1. **Python 3.10+** (Probado en Python 3.11, 3.12 y 3.13)
2. **PostgreSQL 14+** con la extensión `pgvector` instalada.
   - Si usas Docker:
     ```bash
     docker run -d --name postgres-rag -p 5432:5432 \
       -e POSTGRES_USER=postgres \
       -e POSTGRES_PASSWORD=postgres \
       -e POSTGRES_DB=chatbot_db \
       pgvector/pgvector:pg16
     ```

---

## ⚙️ Instalación y Configuración

### 1. Crear y activar entorno virtual

```bash
python3 -m venv .venv
source .venv/bin/activate  # En Linux / macOS
# o en Windows: .venv\Scripts\activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Configurar variables de entorno

Copia la plantilla `.env.example` a `.env` y configura tus credenciales:

```bash
cp .env.example .env
```

Parámetros clave en `.env`:
- `POSTGRES_SERVER`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`: Datos de conexión a PostgreSQL.
- `EMBEDDING_PROVIDER`:
  - `openai`: Genera embeddings reales usando OpenAI (`OPENAI_API_KEY`).
  - `mock`: Genera embeddings sintéticos deterministas de 1536 dimensiones (ideal para desarrollo/pruebas locales sin costo).
- `CHUNK_SIZE` y `CHUNK_OVERLAP`: Parámetros para el splitter de LangChain (por defecto `1000` y `200`).

---

## 🏃‍♂️ Ejecución del Servidor

Inicia el servidor en modo desarrollo con recarga automática:

```bash
uvicorn app.main:app --reload --port 8000
```

- **Swagger UI interactivo**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## 📄 Uso del Endpoint de Ingesta

### `POST /api/v1/documents/upload`

Permite subir un archivo PDF institucional con metadatos de categoría y rol destinatario.

#### Parámetros del Formulario (`multipart/form-data`):
- `file`: Archivo binario PDF (`UploadFile`).
- `category`: Categoría institucional (ej. `Reglamento`, `Admisión`, `Beneficios`, `Académico`).
- `target_role`: Rol destinatario (ej. `Estudiante`, `Docente`, `Administrativo`, `Público General`).

#### Ejemplo con cURL:

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@/ruta/a/tu_documento.pdf" \
  -F "category=Reglamento" \
  -F "target_role=Estudiante"
```

#### Respuesta de Éxito (`HTTP 201 Created`):

```json
{
  "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "filename": "reglamento_academico_2026.pdf",
  "category": "Reglamento",
  "target_role": "Estudiante",
  "file_size": 245832,
  "total_chunks": 18,
  "created_at": "2026-11-03T15:30:00Z",
  "message": "Documento procesado e indexado exitosamente en la base vectorial"
}
```

---

## 🛡️ Robustez y Limpieza

1. **Gestión de Archivos Temporales**: Los archivos PDF se escriben temporalmente mediante `tempfile.NamedTemporaryFile` y se leen en buffers de 1MB para evitar sobrecarga de memoria. El bloque `try ... finally` garantiza la eliminación inmediata del archivo temporal en el sistema de archivos incluso ante excepciones.
2. **Validaciones**: Se valida extensión `.pdf`, contenido no vacío, y legibilidad del texto con fallback entre `PyPDFLoader` y `pypdf`.
3. **Persistencia Atómica**: Si la generación de embeddings o la inserción falla en cualquier punto, se ejecuta un `rollback` de la sesión de base de datos evitando registros huérfanos.

---

## 🤖 Agente Conversacional con Tool Calling

El endpoint `POST /api/v1/chat` implementa el patrón **Agente con Herramientas**, orquestando de manera autónoma las consultas hacia la base vectorial (`pgvector`) o la base relacional académica.

### Herramientas Disponibles

1. **`tool_buscar_documentos_institucionales(query: str, rol: str)`**:
   - Realiza búsqueda semántica en `document_chunks` (`pgvector`) mediante distancia coseno.
   - Aplica filtrado estricto por rol del usuario solicitante para impedir que un alumno acceda a circulares o documentos privados.
2. **`tool_consultar_calificaciones_alumno(dni: str)`**:
   - Consulta el historial de asignaturas cursadas, notas y estado académico en las tablas relacionales.
   - **Control de Privacidad / RBAC**: Valida obligatoriamente que si `user_role == "alumno"`, el DNI consultado coincida con su identidad de sesión. Rechaza inmediatamente cualquier intento de consultar notas de otros estudiantes.
3. **`tool_consultar_materias_carrera(carrera_id: int)`**:
   - Devuelve el plan de estudios relacional (materias ordenadas por año, cuatrimestre y correlativas).

### Ejemplo de Solicitud `POST /api/v1/chat`:

```json
{
  "message": "¿Cuáles son mis calificaciones?",
  "user_dni": "12345678",
  "user_role": "alumno"
}
```

### Respuesta de Éxito (`HTTP 200 OK`):

```json
{
  "response": "He consultado el sistema académico para obtener tu información.\n\nHistorial Académico de Juan Pérez (DNI: 12345678)\nCarrera: Ingeniería en Informática\nTotal de materias registradas: 5\nDetalle de Calificaciones:\n- Algoritmos y Estructuras de Datos I (INF-101): Estado 'Aprobada', Nota: 9.0, Fecha: No registrada\n- Matemática Discreta (INF-102): Estado 'Aprobada', Nota: 8.5, Fecha: No registrada\n- Algoritmos y Estructuras de Datos II (INF-103): Estado 'Aprobada', Nota: 7.0, Fecha: No registrada\n- Bases de Datos Relacionales y NoSQL (INF-201): Estado 'Aprobada', Nota: 10.0, Fecha: No registrada\n- Programación Orientada a Objetos (INF-202): Estado 'Cursando', Sin nota numérica, Fecha: No registrada\n\nSi necesitas detalles sobre cómo solicitar una revisión o certificado, no dudes en consultarme.",
  "tool_calls": [
    {
      "tool_name": "tool_consultar_calificaciones_alumno",
      "tool_args": {
        "dni": "12345678"
      },
      "tool_output": "Historial Académico de Juan Pérez (DNI: 12345678)\nCarrera: Ingeniería en Informática..."
    }
  ],
  "user_dni": "12345678",
  "user_role": "alumno"
}
```

---

## 🧪 Pruebas Automatizadas

Ejecuta la suite de pruebas unitarias y de integración:

```bash
python3 -m unittest -v tests/test_conversational_agent.py
```

