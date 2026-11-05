import logging
from typing import AsyncGenerator
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from app.core.config import settings

logger = logging.getLogger(__name__)

# Configuración del motor asíncrono de SQLAlchemy
engine = create_async_engine(
    settings.ASYNC_DATABASE_URL,
    echo=settings.DEBUG,
    future=True,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    connect_args={
        "server_settings": {
            "search_path": "academico,public"
        }
    },
)

# Fábrica de sesiones asíncronas
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Clase base declarativa para los modelos SQLAlchemy."""
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Inyector de dependencias para FastAPI que provee una sesión de base de datos asíncrona.
    Cada request recibe una sesión limpia. Garantiza rollback y cierre adecuado.
    """
    session = AsyncSessionLocal()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def init_db() -> None:
    """
    Inicializa la base de datos:
    1. Asegura la extensión 'vector' para pgvector.
    2. Crea las tablas declaradas si no existen.
    3. Siembra datos académicos iniciales de prueba si está vacía.
    """
    try:
        # Importar modelos para registrar sus metadatos en Base
        import app.models  # noqa: F401
        from app.services.academic_service import academic_service

        async with engine.begin() as conn:
            # Crear esquema academico si no existe
            await conn.execute(text("CREATE SCHEMA IF NOT EXISTS academico;"))
            # Habilitar extensión vector en PostgreSQL
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            # Crear tablas si no existen (útil para desarrollo y pruebas)
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Base de datos y esquemas inicializados correctamente.")

        # Sembrar datos de ejemplo para pruebas académicas inmediatas
        async with AsyncSessionLocal() as session:
            await academic_service.seed_sample_data_if_empty(session)
    except Exception as e:
        logger.warning(
            f"No se pudo inicializar la base de datos automáticamente: {e}. "
            "Asegúrate de que PostgreSQL y pgvector estén disponibles."
        )
