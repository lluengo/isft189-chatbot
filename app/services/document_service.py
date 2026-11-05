import logging
import os
import tempfile
from typing import List, Tuple
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.models.document import Document, DocumentChunk
from app.services.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)


class DocumentServiceException(Exception):
    """Excepción base para errores en el servicio de documentos."""
    pass


class InvalidPDFException(DocumentServiceException):
    """Excepción cuando el archivo recibido no es un PDF válido o está corrupto."""
    pass


class EmptyDocumentException(DocumentServiceException):
    """Excepción cuando el archivo no contiene texto legible."""
    pass


class DocumentService:
    """
    Servicio encargado de la ingestión, procesamiento RAG y persistencia de documentos PDF en pgvector.
    """

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    def _extract_text_and_pages(self, file_path: str) -> List[Tuple[str, int]]:
        """
        Extrae el contenido de texto por página del archivo PDF.
        Retorna una lista de tuplas (texto, número_de_página).
        """
        pages_content: List[Tuple[str, int]] = []

        try:
            # Intento primario con PyPDFLoader de LangChain
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            for doc in docs:
                page_num = doc.metadata.get("page", 0) + 1  # 1-indexed
                if doc.page_content and doc.page_content.strip():
                    pages_content.append((doc.page_content.strip(), page_num))
        except Exception as primary_err:
            logger.warning(
                f"Fallo PyPDFLoader ({primary_err}), intentando extracción nativa con pypdf..."
            )
            try:
                import pypdf
                reader = pypdf.PdfReader(file_path)
                if reader.is_encrypted:
                    try:
                        reader.decrypt("")
                    except Exception:
                        raise InvalidPDFException("El PDF está protegido con contraseña o cifrado.")

                for idx, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        pages_content.append((text.strip(), idx + 1))
            except InvalidPDFException:
                raise
            except Exception as e:
                logger.error(f"Error procesando PDF con pypdf: {e}")
                raise InvalidPDFException(f"El archivo PDF está dañado o no pudo ser procesado: {str(e)}")

        return pages_content

    async def process_and_store_document(
        self,
        file: UploadFile,
        category: str,
        target_role: str,
        db: AsyncSession,
    ) -> Document:
        """
        Flujo completo de procesamiento de PDF:
        1. Guarda temporalmente el stream del archivo en disco de forma segura.
        2. Extrae texto por páginas y valida contenido.
        3. Aplica RecursiveCharacterTextSplitter para segmentar en chunks semánticos.
        4. Genera embeddings para los chunks.
        5. Persiste atómicamente el Documento y sus DocumentChunk en PostgreSQL/pgvector.
        6. Garantiza la eliminación del archivo temporal en el bloque finally.
        """
        # Validación de tipo de archivo
        filename = file.filename or "document.pdf"
        if not filename.lower().endswith(".pdf"):
            raise InvalidPDFException("Solo se permiten archivos con formato PDF (.pdf).")

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_file_path = temp_file.name

        try:
            # 1. Escritura por fragmentos en archivo temporal para controlar consumo de RAM
            file_size = 0
            while chunk := await file.read(1024 * 1024):  # 1 MB por iteración
                temp_file.write(chunk)
                file_size += len(chunk)

            temp_file.flush()
            temp_file.close()

            if file_size == 0:
                raise EmptyDocumentException("El archivo subido está vacío (0 bytes).")

            logger.info(
                f"Archivo temporal creado en {temp_file_path} ({file_size} bytes). Iniciando extracción..."
            )

            # 2. Extracción de páginas
            pages = self._extract_text_and_pages(temp_file_path)
            if not pages:
                raise EmptyDocumentException(
                    "No se pudo extraer texto del documento PDF. "
                    "Verifica si es un PDF escaneado sin OCR o si está vacío."
                )

            # 3. Segmentación en chunks con metadatos de página
            all_chunks: List[dict] = []
            for text, page_num in pages:
                split_texts = self.text_splitter.split_text(text)
                for chunk_text in split_texts:
                    if chunk_text.strip():
                        all_chunks.append({
                            "text": chunk_text,
                            "metadata": {
                                "page": page_num,
                                "source": filename,
                                "category": category,
                                "target_role": target_role,
                            }
                        })

            if not all_chunks:
                raise EmptyDocumentException("El contenido del documento no generó fragmentos de texto válidos.")

            logger.info(f"Se generaron {len(all_chunks)} fragmentos de texto. Calculando embeddings...")

            # 4. Generación de embeddings
            embedding_service = get_embedding_service()
            chunk_texts = [c["text"] for c in all_chunks]
            embeddings = embedding_service.embed_documents(chunk_texts)

            # 5. Persistencia en Base de Datos (Transacción atómica)
            doc_record = Document(
                filename=filename,
                category=category.strip(),
                target_role=target_role.strip(),
                file_size=file_size,
                total_chunks=len(all_chunks),
            )
            db.add(doc_record)
            await db.flush()  # Asigna el ID del documento para asociar los chunks

            chunk_records = [
                DocumentChunk(
                    document_id=doc_record.id,
                    chunk_index=i,
                    content=chunk_data["text"],
                    chunk_metadata=chunk_data["metadata"],
                    embedding=embedding_vector,
                )
                for i, (chunk_data, embedding_vector) in enumerate(zip(all_chunks, embeddings))
            ]

            db.add_all(chunk_records)
            await db.commit()
            await db.refresh(doc_record)

            logger.info(
                f"Documento '{filename}' (ID: {doc_record.id}) indexado exitosamente con {len(chunk_records)} chunks."
            )
            return doc_record

        except Exception as e:
            await db.rollback()
            logger.error(f"Error procesando documento '{filename}': {e}", exc_info=True)
            if isinstance(e, DocumentServiceException):
                raise
            raise DocumentServiceException(f"Error interno al procesar el documento: {str(e)}")

        finally:
            # 6. Limpieza garantizada del archivo temporal
            if os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    logger.debug(f"Archivo temporal eliminado: {temp_file_path}")
                except Exception as err:
                    logger.warning(f"No se pudo eliminar el archivo temporal {temp_file_path}: {err}")


# Instancia singleton del servicio
document_service = DocumentService()
