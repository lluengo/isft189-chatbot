import contextvars
import json
import logging
from typing import Any, Dict, List, Optional
from langchain_core.tools import tool
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.academic_service import academic_service
from app.services.vector_search_service import vector_search_service

logger = logging.getLogger(__name__)

# Contextos asíncronos para inyección segura de sesión de BD y credenciales de usuario
current_db_session: contextvars.ContextVar[Optional[AsyncSession]] = contextvars.ContextVar(
    "current_db_session", default=None
)
current_user_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "current_user_context", default={}
)


class ToolExecutionContext:
    """
    Gestor de contexto asíncrono para vincular de forma segura la sesión de base de datos
    y las credenciales/rol del usuario solicitante durante el ciclo de vida de la consulta.
    """

    def __init__(self, db: AsyncSession, requester_dni: Optional[str], requester_role: str):
        self.db = db
        self.user_data = {
            "requester_dni": str(requester_dni).strip() if requester_dni else None,
            "requester_role": str(requester_role or "alumno").strip().lower(),
        }
        self._tokens = []

    def __enter__(self):
        self._tokens.append(current_db_session.set(self.db))
        self._tokens.append(current_user_context.set(self.user_data))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if len(self._tokens) >= 2:
            current_user_context.reset(self._tokens.pop())
            current_db_session.reset(self._tokens.pop())


@tool
async def tool_buscar_documentos_institucionales(query: str, rol: str) -> str:
    """
    Realiza la búsqueda vectorial en pgvector filtrando por rol y similitud semántica.
    Útil para consultar reglamentos académicos, fechas de calendario, estatutos, resoluciones,
    y documentación teórica o institucional.
    
    Args:
        query: Pregunta o palabras clave a buscar en los documentos institucionales.
        rol: Rol destinatario del usuario (ej. 'alumno', 'docente', 'administrativo', 'publico').
    """
    db = current_db_session.get()
    if not db:
        return "[Error]: Sesión de base de datos no disponible para búsqueda vectorial."

    user_ctx = current_user_context.get()
    user_actual_role = user_ctx.get("requester_role", "alumno")

    # Seguridad: Un alumno no puede forzar la búsqueda de documentos exclusivos de directivos o docentes
    effective_role = user_actual_role if user_actual_role == "alumno" else (rol or user_actual_role)

    logger.info(f"Ejecutando tool_buscar_documentos_institucionales: query='{query}', rol='{effective_role}'")
    results = await vector_search_service.search_similar_documents(
        query=query,
        rol=effective_role,
        db=db,
    )

    if not results:
        return (
            f"No se encontraron documentos institucionales relevantes para '{query}' "
            f"con el rol '{effective_role}'."
        )

    formatted_docs = []
    for doc in results:
        formatted_docs.append(
            f"--- Documento: {doc['filename']} (Categoría: {doc['category']}, Página: {doc['page']}, "
            f"Similitud: {doc['similarity_score']:.2f}) ---\n"
            f"{doc['content']}"
        )

    return "\n\n".join(formatted_docs)


@tool
async def tool_consultar_calificaciones_alumno(dni: str) -> str:
    """
    Consulta de forma segura las notas y estado académico del alumno en las tablas relacionales.
    Aplica controles estrictos de seguridad para prevenir filtración de datos de terceros.
    
    Args:
        dni: Documento Nacional de Identidad del alumno a consultar.
    """
    db = current_db_session.get()
    if not db:
        return "[Error]: Sesión de base de datos no disponible para consulta de calificaciones."

    user_ctx = current_user_context.get()
    requester_dni = user_ctx.get("requester_dni")
    requester_role = user_ctx.get("requester_role", "alumno")

    clean_target_dni = str(dni or "").strip()

    logger.info(
        f"Ejecutando tool_consultar_calificaciones_alumno: target_dni='{clean_target_dni}', "
        f"requester_dni='{requester_dni}', requester_role='{requester_role}'"
    )

    # Limpiar cualquier transacción abortada antes de ejecutar la query relacional
    try:
        await db.execute(text("SELECT 1"))
    except Exception:
        await db.rollback()

    # Invocar servicio con validaciones de privacidad
    result = await academic_service.get_student_grades(
        target_dni=clean_target_dni,
        requester_dni=requester_dni,
        requester_role=requester_role,
        db=db,
    )

    if result.get("security_violation"):
        return f"[VIOLACIÓN DE SEGURIDAD]: {result.get('error')}"

    if result.get("total_registros", 0) == 0 and "mensaje" in result:
        return result["mensaje"]

    # Formatear respuesta amigable para el LLM
    output = [
        f"Historial Académico de {result['alumno']} (DNI: {result['dni']})",
        f"Carrera: {result['carrera']}",
        f"Total de materias registradas: {result['total_materias_registradas']}",
        "Detalle de Calificaciones:",
    ]
    for cal in result.get("calificaciones", []):
        nota_str = f"Nota: {cal['nota']}" if cal['nota'] is not None else "Sin nota numérica"
        output.append(f"- {cal['materia']} ({cal['codigo']}): Estado '{cal['estado']}', {nota_str}, Fecha: {cal['fecha']}")

    return "\n".join(output)


@tool
async def tool_consultar_materias_carrera(carrera_id: int) -> str:
    """
    Devuelve el plan de estudios relacional (lista de materias por año y cuatrimestre)
    para una carrera específica.
    
    Args:
        carrera_id: ID numérico de la carrera universitaria (ej. 1 para Ingeniería en Informática).
    """
    db = current_db_session.get()
    if not db:
        return "[Error]: Sesión de base de datos no disponible para consulta de plan de estudios."

    logger.info(f"Ejecutando tool_consultar_materias_carrera: carrera_id={carrera_id}")
    result = await academic_service.get_career_subjects(carrera_id=carrera_id, db=db)

    if "error" in result:
        disponibles = result.get("carreras_disponibles", [])
        disp_txt = ", ".join([f"{c['id']}: {c['nombre']} ({c['codigo']})" for c in disponibles])
        return f"{result['error']} Carreras disponibles en el sistema: {disp_txt}"

    output = [
        f"Plan de Estudios: {result['carrera_nombre']} ({result['carrera_codigo']})",
        f"Duración: {result['duracion_anios']} años. Total de materias: {result['total_materias']}",
        f"Descripción: {result.get('descripcion', '')}",
        "Materias por año y cuatrimestre:",
    ]
    for mat in result.get("plan_estudios", []):
        output.append(
            f"- Año {mat['anio']}, Cuatrimestre {mat['cuatrimestre']}: {mat['nombre']} ({mat['codigo']}) "
            f"[Correlativas: {mat['correlativas']}]"
        )

    return "\n".join(output)


# Lista de herramientas registradas para el agente conversacional
ACADEMIC_TOOLS = [
    tool_buscar_documentos_institucionales,
    tool_consultar_calificaciones_alumno,
    tool_consultar_materias_carrera,
]
