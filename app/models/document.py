import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, DateTime, Text, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

from app.core.config import settings
from app.core.database import Base


class Document(Base):
    """
    Modelo representativo de un documento institucional cargado en el sistema.
    """
    __tablename__ = "documents"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    filename = Column(String(255), nullable=False, doc="Nombre original del archivo PDF")
    category = Column(String(100), nullable=False, index=True, doc="Categoría o departamento institucional")
    target_role = Column(String(100), nullable=False, index=True, doc="Rol destinatario (ej. alumno, docente)")
    file_size = Column(Integer, nullable=False, doc="Tamaño del archivo en bytes")
    total_chunks = Column(Integer, default=0, nullable=False, doc="Cantidad de chunks generados")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relación uno a muchos con los chunks vectoriales
    chunks = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id} filename={self.filename} category={self.category}>"


class DocumentChunk(Base):
    """
    Fragmento textual (chunk) del documento con su representación vectorial (embedding).
    """
    __tablename__ = "document_chunks"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    document_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    chunk_index = Column(Integer, nullable=False, doc="Posición ordinal del fragmento en el documento")
    content = Column(Text, nullable=False, doc="Texto correspondiente al fragmento")
    chunk_metadata = Column(JSON, nullable=True, default=dict, doc="Metadatos como número de página, offsets, etc.")
    embedding = Column(
        Vector(settings.EMBEDDING_DIMENSION),
        nullable=False,
        doc=f"Vector embedding de dimensión {settings.EMBEDDING_DIMENSION}"
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relación hacia el documento padre
    document = relationship("Document", back_populates="chunks")

    def __repr__(self) -> str:
        return f"<DocumentChunk id={self.id} doc_id={self.document_id} index={self.chunk_index}>"
