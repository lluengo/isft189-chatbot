from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    """
    Carga útil de la solicitud para el endpoint conversacional.
    """
    message: str = Field(
        ...,
        min_length=1,
        description="Consulta o pregunta formulada por el usuario al agente",
        examples=["¿Cuáles son mis notas?", "¿Qué materias tiene la carrera de Ingeniería?", "¿Cuál es el reglamento de finales?"],
    )
    user_dni: Optional[str] = Field(
        default=None,
        description="DNI o identificación del usuario que realiza la consulta",
        examples=["12345678"],
    )
    user_role: str = Field(
        default="alumno",
        description="Rol del usuario en la institución: 'alumno', 'docente', 'administrativo', 'publico'",
        examples=["alumno"],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "¿Cuáles son mis calificaciones?",
                "user_dni": "12345678",
                "user_role": "alumno",
            }
        }
    )


class ToolCallRecord(BaseModel):
    """
    Registro de una herramienta ejecutada por el agente conversacional.
    """
    tool_name: str = Field(..., description="Nombre de la herramienta ejecutada")
    tool_args: Dict[str, Any] = Field(default_factory=dict, description="Argumentos pasados a la herramienta")
    tool_output: str = Field(..., description="Salida o resultado retornado por la herramienta")


class ChatResponse(BaseModel):
    """
    Respuesta del agente conversacional inteligente.
    """
    response: str = Field(..., description="Respuesta final generada para el usuario")
    tool_calls: List[ToolCallRecord] = Field(
        default_factory=list,
        description="Lista de herramientas invocadas durante la resolución de la consulta",
    )
    user_dni: Optional[str] = Field(default=None, description="DNI del usuario autenticado")
    user_role: str = Field(..., description="Rol del usuario autenticado")

    model_config = ConfigDict(from_attributes=True)
