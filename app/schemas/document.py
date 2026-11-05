from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field


class DocumentBase(BaseModel):
    filename: str = Field(..., description="Nombre original del archivo PDF")
    category: str = Field(..., description="Categoría o departamento institucional")
    target_role: str = Field(..., description="Rol objetivo del documento (ej. alumno, docente, administrativo)")


class DocumentUploadResponse(DocumentBase):
    """
    Respuesta devuelta tras la subida, chunking e indexación vectorial exitosa de un documento.
    """
    id: UUID = Field(..., description="Identificador único del documento")
    file_size: int = Field(..., description="Tamaño del archivo en bytes")
    total_chunks: int = Field(..., description="Número total de fragmentos generados e insertados en pgvector")
    created_at: datetime = Field(..., description="Fecha y hora de registro")
    message: str = Field(
        default="Documento procesado e indexado exitosamente en la base vectorial",
        description="Mensaje informativo del resultado"
    )

    model_config = ConfigDict(from_attributes=True)


class DocumentChunkResponse(BaseModel):
    """
    Respuesta para un fragmento individual del documento.
    """
    id: UUID
    document_id: UUID
    chunk_index: int
    content: str
    chunk_metadata: Optional[Dict[str, Any]] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentDetailResponse(DocumentUploadResponse):
    """
    Detalle completo del documento incluyendo metadatos de auditoría.
    """
    updated_at: Optional[datetime] = None
