import abc
import hashlib
import logging
from typing import List
from app.core.config import settings

logger = logging.getLogger(__name__)


class BaseEmbeddingService(abc.ABC):
    """
    Contrato base abstracto para generadores de embeddings.
    Permite desacoplar el proveedor de embeddings (OpenAI, HuggingFace, Ollama, etc.).
    """

    @abc.abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Genera embeddings para una lista de fragmentos de texto."""
        pass

    @abc.abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Genera el embedding para una consulta de usuario."""
        pass

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """Dimensión del vector generado."""
        pass


class OpenAIEmbeddingService(BaseEmbeddingService):
    """
    Servicio de embeddings utilizando OpenAI a través de LangChain.
    """

    def __init__(self, api_key: str | None = None, model_name: str | None = None):
        self._model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self._api_key = api_key or settings.OPENAI_API_KEY
        self._dimension = settings.EMBEDDING_DIMENSION

        if not self._api_key:
            raise ValueError(
                "OPENAI_API_KEY no está configurada. Por favor establece la variable de entorno "
                "o selecciona EMBEDDING_PROVIDER=mock para entorno de desarrollo sin API key."
            )

        try:
            from langchain_openai import OpenAIEmbeddings
            self._client = OpenAIEmbeddings(
                model=self._model_name,
                api_key=self._api_key,
                check_embedding_ctx_length=True,
            )
        except Exception as e:
            logger.error(f"Error inicializando OpenAIEmbeddings: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            return self._client.embed_documents(texts)
        except Exception as e:
            logger.warning(f"Error al conectar con OpenAI Embeddings ({e}). Activando fallback a Mock determinista.")
            return MockEmbeddingService(dimension=self._dimension).embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        try:
            return self._client.embed_query(text)
        except Exception as e:
            logger.warning(f"Error al conectar con OpenAI Embeddings ({e}). Activando fallback a Mock determinista.")
            return MockEmbeddingService(dimension=self._dimension).embed_query(text)

    @property
    def dimension(self) -> int:
        return self._dimension


class MockEmbeddingService(BaseEmbeddingService):
    """
    Servicio de prueba determinístico para desarrollo y testing sin llamadas a servicios externos.
    Genera un vector pseudo-aleatorio basado en el hash del texto de entrada.
    """

    def __init__(self, dimension: int | None = None):
        self._dimension = dimension or settings.EMBEDDING_DIMENSION

    def _generate_vector(self, text: str) -> List[float]:
        # Generar vector sintético normalizado a partir de SHA-256
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
        vec = []
        for i in range(self._dimension):
            # Valor entre -1.0 y 1.0
            val = (((seed + i * 31) % 2000) - 1000) / 1000.0
            vec.append(val)
        # Normalizar
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [round(x / norm, 6) for x in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        logger.info(f"Generando embeddings simulados (Mock) para {len(texts)} textos.")
        return [self._generate_vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._generate_vector(text)

    @property
    def dimension(self) -> int:
        return self._dimension


class OllamaEmbeddingService(BaseEmbeddingService):
    """
    Servicio de embeddings utilizando modelos locales en Ollama (ej. nomic-embed-text).
    """

    def __init__(self, base_url: str | None = None, model_name: str | None = None):
        self._base_url = base_url or settings.OLLAMA_BASE_URL
        self._model_name = model_name or settings.OLLAMA_EMBEDDING_MODEL
        self._dimension = 768 if "nomic" in self._model_name.lower() else settings.EMBEDDING_DIMENSION

        try:
            from langchain_community.embeddings import OllamaEmbeddings
            self._client = OllamaEmbeddings(
                base_url=self._base_url,
                model=self._model_name,
            )
        except Exception as e:
            logger.error(f"Error inicializando OllamaEmbeddings: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            return self._client.embed_documents(texts)
        except Exception as e:
            logger.warning(f"Error al conectar con Ollama Embeddings ({e}). Activando fallback a Mock determinista.")
            return MockEmbeddingService(dimension=self._dimension).embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        try:
            return self._client.embed_query(text)
        except Exception as e:
            logger.warning(f"Error al conectar con Ollama Embeddings ({e}). Activando fallback a Mock determinista.")
            return MockEmbeddingService(dimension=self._dimension).embed_query(text)

    @property
    def dimension(self) -> int:
        return self._dimension


def get_embedding_service() -> BaseEmbeddingService:
    """
    Fábrica que instancia el servicio de embeddings configurado.
    Detecta automáticamente si la API key es un placeholder y utiliza MockEmbeddingService
    para evitar excepciones 401 en entornos de desarrollo.
    """
    provider = settings.EMBEDDING_PROVIDER.lower()
    if provider == "mock":
        return MockEmbeddingService()
    elif provider == "ollama":
        return OllamaEmbeddingService()
    elif provider == "openai":
        api_key = settings.OPENAI_API_KEY
        if not api_key or api_key.strip() == "" or "tu-api-key" in api_key.lower():
            logger.warning(
                "OPENAI_API_KEY no está configurada o contiene el placeholder por defecto ('%s'). "
                "Utilizando MockEmbeddingService automáticamente.",
                api_key,
            )
            return MockEmbeddingService()

        try:
            return OpenAIEmbeddingService()
        except Exception as e:
            logger.warning(
                f"No se pudo inicializar OpenAIEmbeddingService ({e}). Activando MockEmbeddingService."
            )
            return MockEmbeddingService()
    else:
        logger.warning(
            f"Proveedor desconocido '{provider}'. Utilizando MockEmbeddingService por defecto."
        )
        return MockEmbeddingService()
