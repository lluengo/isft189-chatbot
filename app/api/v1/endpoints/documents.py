import logging
from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.document import Document
from app.schemas.document import DocumentDetailResponse, DocumentUploadResponse
from app.services.document_service import (
    EmptyDocumentException,
    InvalidPDFException,
    DocumentServiceException,
    document_service,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Subir y procesar un documento PDF para RAG",
    description=(
        "Recibe un archivo PDF junto con sus metadatos (`category`, `target_role`). "
        "Aplica extracción de texto, fragmentación (chunking) semántica mediante RecursiveCharacterTextSplitter, "
        "genera los embeddings correspondientes y los almacena en pgvector (`document_chunks` y `documents`). "
        "Los archivos temporales son purgados automáticamente tras el procesamiento."
    ),
)
async def upload_document(
    file: UploadFile = File(..., description="Archivo PDF a procesar"),
    category: str = Form(..., min_length=2, max_length=100, description="Categoría institucional (ej: Reglamento, Matrícula)"),
    target_role: str = Form(..., min_length=2, max_length=100, description="Rol destinatario (ej: Estudiante, Docente, Funcionario)"),
    db: AsyncSession = Depends(get_db),
) -> DocumentUploadResponse:
    # 1. Validación de Content-Type y extensión
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Formato no permitido: '{filename}'. Únicamente se admiten archivos .pdf.",
        )

    # 2. Procesamiento e ingestión RAG
    try:
        document = await document_service.process_and_store_document(
            file=file,
            category=category,
            target_role=target_role,
            db=db,
        )
        return DocumentUploadResponse.model_validate(document)

    except InvalidPDFException as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except EmptyDocumentException as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except DocumentServiceException as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en el procesamiento del documento: {str(exc)}",
        )
    except Exception as exc:
        logger.error(f"Error imprevisto al procesar subida de archivo: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ocurrió un error interno no controlado al procesar el archivo.",
        )


@router.get(
    "/",
    response_model=List[DocumentDetailResponse],
    summary="Listar documentos institucionales indexados",
    description="Retorna la lista de todos los documentos indexados en el sistema.",
)
async def list_documents(
    skip: int = 0,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
) -> List[DocumentDetailResponse]:
    result = await db.execute(select(Document).offset(skip).limit(limit).order_by(Document.created_at.desc()))
    documents = result.scalars().all()
    return [DocumentDetailResponse.model_validate(doc) for doc in documents]


@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    summary="Consultar detalle de un documento",
    description="Obtiene los metadatos y estado de chunks de un documento por su ID.",
)
async def get_document(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> DocumentDetailResponse:
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Documento con ID {document_id} no encontrado.",
        )
    return DocumentDetailResponse.model_validate(document)
