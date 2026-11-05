import logging
from typing import Any, Dict, List, Optional
from sqlalchemy import or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentChunk
from app.services.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)


class VectorSearchService:
    """
    Servicio encargado de ejecutar búsquedas vectoriales semánticas
    sobre chunks indexados en pgvector, con filtrado por rol y similitud.
    """

    async def search_similar_documents(
        self,
        query: str,
        rol: str,
        db: AsyncSession,
        limit: int = 8,
        max_distance: float = 0.85,
    ) -> List[Dict[str, Any]]:
        """
        Ejecuta la búsqueda de fragmentos de documentos usando similitud de coseno en pgvector.
        Usa SAVEPOINT para aislar cualquier error vectorial sin abortar la transacción principal.
        """
        if not query or not query.strip():
            return []

        try:
            # SAVEPOINT aísla el bloque vectorial: si falla, se hace ROLLBACK
            # al punto de guardado y la sesión queda operativa para otras queries.
            await db.execute(text("SAVEPOINT vector_search_sp"))

            embedding_service = get_embedding_service()
            query_vector = embedding_service.embed_query(query.strip())

            # Expresión de distancia de coseno en pgvector
            cosine_dist = DocumentChunk.embedding.cosine_distance(query_vector)

            # Construcción del filtro de roles
            rol_cleaned = (rol or "todos").strip().lower()
            role_variants = [rol_cleaned]
            if "alumno" in rol_cleaned:
                role_variants.extend(["estudiante", "alumno"])
            elif "estudiante" in rol_cleaned:
                role_variants.extend(["estudiante", "alumno"])
            elif "docente" in rol_cleaned:
                role_variants.extend(["docente", "profesor"])

            role_conditions = [
                Document.target_role.ilike(f"%{v}%") for v in role_variants
            ] + [
                Document.target_role.ilike("%todo%"),
                Document.target_role.ilike("%general%"),
                Document.target_role.ilike("%publico%"),
                Document.target_role.ilike("%público%"),
            ]
            role_filter = or_(*role_conditions)

            query_stmt = (
                select(
                    DocumentChunk.id,
                    DocumentChunk.chunk_index,
                    DocumentChunk.content,
                    DocumentChunk.chunk_metadata,
                    Document.filename,
                    Document.category,
                    Document.target_role,
                    cosine_dist.label("distance"),
                )
                .join(Document, DocumentChunk.document_id == Document.id)
                .where(role_filter)
                .where(cosine_dist <= max_distance)
                .order_by(cosine_dist.asc())
                .limit(limit)
            )

            result = await db.execute(query_stmt)
            rows = result.all()

            # Liberar el savepoint en caso de éxito
            await db.execute(text("RELEASE SAVEPOINT vector_search_sp"))

        except Exception as e:
            logger.error(f"Error ejecutando búsqueda vectorial en pgvector: {e}", exc_info=True)
            # Revertir al savepoint para dejar la transacción principal operativa
            try:
                await db.execute(text("ROLLBACK TO SAVEPOINT vector_search_sp"))
            except Exception:
                pass
            return []

        results = []
        for row in rows:
            dist = float(row.distance) if row.distance is not None else 1.0
            similarity = max(0.0, min(1.0, 1.0 - dist))
            meta = row.chunk_metadata or {}
            results.append({
                "chunk_id": str(row.id),
                "filename": row.filename,
                "category": row.category,
                "target_role": row.target_role,
                "page": meta.get("page", 1),
                "similarity_score": round(similarity, 4),
                "distance": round(dist, 4),
                "content": row.content,
            })

        logger.info(
            f"Búsqueda vectorial '{query[:30]}...' para rol '{rol}' retornó {len(results)} chunks."
        )
        return results


vector_search_service = VectorSearchService()
