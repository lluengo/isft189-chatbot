import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.academic_service import academic_service
from app.services.chat_agent_service import chat_agent_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Procesar consulta conversacional con Agente Tool Calling",
    description=(
        "Recibe la consulta del usuario, su DNI/identidad y su rol institucional. "
        "El agente orquestador decide automáticamente si invoca herramientas sobre la base vectorial (pgvector) "
        "o sobre las tablas relacionales académicas (alumnos, calificaciones, materias), "
        "aplicando controles estrictos de seguridad RBAC contra filtración de datos de terceros."
    ),
)
async def chat_endpoint(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    """
    Endpoint principal para interacción conversacional inteligente.
    """
    try:
        # Asegurar que los datos de prueba del alumno existan si aún no fueron sembrados
        await academic_service.seed_sample_data_if_empty(db)

        response = await chat_agent_service.process_chat(
            message=request.message,
            user_dni=request.user_dni,
            user_role=request.user_role,
            db=db,
        )
        return response
    except Exception as e:
        logger.error(f"Error procesando conversación en /chat: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno procesando la consulta conversacional: {str(e)}",
        )
