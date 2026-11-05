import json
import logging
import re
from typing import Any, Dict, List, Optional
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.schemas.chat import ChatResponse, ToolCallRecord
from app.services.academic_tools import (
    ACADEMIC_TOOLS,
    ToolExecutionContext,
    tool_buscar_documentos_institucionales,
    tool_consultar_calificaciones_alumno,
    tool_consultar_materias_carrera,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Eres el Asistente Inteligente Institucional de la Universidad.
Tu función es responder consultas de estudiantes, docentes y personal de forma cordial, precisa y formal.

Tienes a tu disposición las siguientes herramientas obligatorias para acceder a la información institucional y académica:
1. 'tool_buscar_documentos_institucionales': Úsala cuando te consulten sobre reglamentos, fechas de exámenes o calendarios, normativas, resoluciones, requisitos teóricos o institucionales (RAG con pgvector).
2. 'tool_consultar_calificaciones_alumno': Úsala para consultar notas, materias aprobadas, cursadas o estado académico de un estudiante en la base de datos relacional. Si el estudiante no especifica un DNI en el mensaje, utiliza su DNI de usuario proporcionado en el contexto.
3. 'tool_consultar_materias_carrera': Úsala cuando pregunten por el plan de estudios, correlativas, duración o materias de una carrera (base de datos relacional).

REGLAS CRÍTICAS:
- NUNCA inventes calificaciones, materias ni artículos de reglamentos. Todo dato específico DEBE provenir de las herramientas.
- Si una herramienta devuelve una advertencia o error de seguridad (como intento de acceso a notas de terceros), explícalo al usuario con respeto y apego a las normas de privacidad.
- Responde siempre en español con un tono profesional, claro y empático.
"""


class ChatAgentService:
    """
    Orquestador del agente conversacional que integra Tool Calling para decidir dinámicamente
    entre la base vectorial (pgvector) y la base de datos relacional académica.
    """

    def __init__(self):
        self.tool_map = {t.name: t for t in ACADEMIC_TOOLS}

    def _should_use_real_openai(self) -> bool:
        """Determina si se dispone de una API Key real y configurada de OpenAI."""
        if settings.LLM_PROVIDER.lower() == "mock":
            return False
        key = settings.OPENAI_API_KEY
        if not key or key.strip() == "" or "tu-api-key" in key.lower():
            return False
        return True

    async def _mock_tool_calling_orchestration(
        self,
        message: str,
        user_dni: Optional[str],
        user_role: str,
        db: AsyncSession,
    ) -> ChatResponse:
        """
        Orquestador determinista de desarrollo/testing que clasifica la intención del usuario,
        invoca las herramientas con ToolExecutionContext y sintetiza la respuesta.
        Permite validar el 100% de la arquitectura sin costo ni dependencia de OpenAI.
        """
        query_lower = message.lower().strip()
        executed_tools: List[ToolCallRecord] = []

        # 1. Intención: Búsqueda documental institucional explícita (Reglamentos, finales, mesas, fechas, normativas)
        palabras_doc_prioritarias = [
            "final", "finales", "mesa", "mesas", "examen", "examenes", "exámenes",
            "fecha", "fechas", "cuando rindo", "cuándo rindo", "turno", "turnos",
            "reglamento", "reglamentos", "norma", "normas", "normativa", "estatuto",
            "resolucion", "resolución", "resoluciones", "calendario", "cronograma", "documento", "archivo"
        ]
        if any(w in query_lower for w in palabras_doc_prioritarias):
            tool_args = {"query": message, "rol": user_role}
            tool_output = await tool_buscar_documentos_institucionales.ainvoke(tool_args)
            executed_tools.append(ToolCallRecord(
                tool_name="tool_buscar_documentos_institucionales",
                tool_args=tool_args,
                tool_output=str(tool_output),
            ))

            final_text = (
                f"De acuerdo con la documentación institucional indexada:\n\n"
                f"{tool_output}"
            )
            return ChatResponse(
                response=final_text,
                tool_calls=executed_tools,
                user_dni=user_dni,
                user_role=user_role,
            )

        # 2. Intención: Historial académico, calificaciones y materias pendientes / avance del alumno
        palabras_historial = [
            "nota", "notas", "calificacion", "calificaciones", "promedio",
            "rendí", "aprobé", "mi final", "mis finales", "mi cursada", "mis cursadas",
            "estado academico", "historial academico", "historial académico",
            "me quedan", "me falta", "me faltan", "cuantas materias", "cuántas materias", "materias aprobadas",
        ]
        if any(w in query_lower for w in palabras_historial):
            # Extraer DNI del mensaje si el usuario especificó otro, o usar el DNI autenticado
            dni_match = re.search(r"\b(\d{7,9})\b", message)
            target_dni = dni_match.group(1) if dni_match else (user_dni or "")

            tool_args = {"dni": target_dni}
            tool_output = await tool_consultar_calificaciones_alumno.ainvoke(tool_args)
            executed_tools.append(ToolCallRecord(
                tool_name="tool_consultar_calificaciones_alumno",
                tool_args=tool_args,
                tool_output=str(tool_output),
            ))

            # Si además pregunta sobre materias restantes o plan de estudio, buscar en documentos del plan
            if any(w in query_lower for w in ["quedan", "falta", "faltan", "terminar", "carrera", "plan"]):
                # Buscar en la base vectorial del plan de estudios (malla curricular completa, materias por año, total de materias)
                doc_args = {
                    "query": "Res-6790-19 Analisis de Sistemas estructura curricular asignaturas materias primer año segundo año tercer año total",
                    "rol": user_role
                }
                doc_output = await tool_buscar_documentos_institucionales.ainvoke(doc_args)
                executed_tools.append(ToolCallRecord(
                    tool_name="tool_buscar_documentos_institucionales",
                    tool_args=doc_args,
                    tool_output=str(doc_output),
                ))

            if "[VIOLACIÓN DE SEGURIDAD]" in str(tool_output):
                final_text = str(tool_output)
            else:
                final_text = (
                    f"He consultado el sistema académico y la base institucional:\n\n"
                    f"{tool_output}\n\n"
                )

            return ChatResponse(
                response=final_text,
                tool_calls=executed_tools,
                user_dni=user_dni,
                user_role=user_role,
            )

        # 3. Intención: Plan de estudios / Materias de la carrera
        palabras_materias = ["materia", "materias", "plan de estudio", "plan de estudios", "carrera", "correlativa", "correlativas", "asignatura", "asignaturas"]
        if any(w in query_lower for w in palabras_materias):
            # Buscar primero en la base vectorial los documentos del plan de estudios
            doc_args = {"query": f"materias plan de estudios {message}", "rol": user_role}
            doc_output = await tool_buscar_documentos_institucionales.ainvoke(doc_args)
            executed_tools.append(ToolCallRecord(
                tool_name="tool_buscar_documentos_institucionales",
                tool_args=doc_args,
                tool_output=str(doc_output),
            ))

            # Consultar también el plan relacional
            carrera_id = 12
            tool_args = {"carrera_id": carrera_id}
            tool_output = await tool_consultar_materias_carrera.ainvoke(tool_args)
            executed_tools.append(ToolCallRecord(
                tool_name="tool_consultar_materias_carrera",
                tool_args=tool_args,
                tool_output=str(tool_output),
            ))

            final_text = (
                f"Aquí tienes la información del plan de estudios y documentación:\n\n"
                f"{doc_output}\n\n{tool_output}"
            )
            return ChatResponse(
                response=final_text,
                tool_calls=executed_tools,
                user_dni=user_dni,
                user_role=user_role,
            )

        # 3. Intención: Búsqueda documental institucional (Reglamentos, resoluciones, fechas, becas)
        palabras_doc = ["reglamento", "norma", "estatuto", "documento", "resolucion", "fecha", "calendario", "beca", "inscripcion", "requisito", "que es", "como es", "donde"]
        if any(w in query_lower for w in palabras_doc) or len(query_lower) > 3:
            tool_args = {"query": message, "rol": user_role}
            tool_output = await tool_buscar_documentos_institucionales.ainvoke(tool_args)
            executed_tools.append(ToolCallRecord(
                tool_name="tool_buscar_documentos_institucionales",
                tool_args=tool_args,
                tool_output=str(tool_output),
            ))

            final_text = (
                f"De acuerdo con la documentación institucional indexada:\n\n"
                f"{tool_output}"
            )
            return ChatResponse(
                response=final_text,
                tool_calls=executed_tools,
                user_dni=user_dni,
                user_role=user_role,
            )

        # 4. Saludo / Consulta general
        return ChatResponse(
            response=(
                f"¡Hola! Soy el asistente virtual institucional de la Universidad. "
                f"Puedo ayudarte a consultar tus calificaciones académicas, ver los planes de estudio de las carreras, "
                f"o buscar información en los reglamentos y resoluciones institucionales. ¿En qué puedo asistirte hoy?"
            ),
            tool_calls=[],
            user_dni=user_dni,
            user_role=user_role,
        )

    async def _openai_tool_calling_orchestration(
        self,
        message: str,
        user_dni: Optional[str],
        user_role: str,
        db: AsyncSession,
    ) -> ChatResponse:
        """
        Orquestación mediante LangChain y ChatOpenAI con enlace de herramientas nativas (Function Calling).
        """
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=settings.OPENAI_CHAT_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            api_key=settings.OPENAI_API_KEY,
        )
        llm_with_tools = llm.bind_tools(ACADEMIC_TOOLS)

        user_context_info = f"[Contexto de sesión: DNI del usuario = {user_dni or 'No especificado'}, Rol = {user_role}]"
        messages: List[BaseMessage] = [
            SystemMessage(content=f"{SYSTEM_PROMPT}\n{user_context_info}"),
            HumanMessage(content=message),
        ]

        executed_tools: List[ToolCallRecord] = []

        # Invocación inicial al LLM para razonamiento y selección de herramientas
        response_msg: AIMessage = await llm_with_tools.ainvoke(messages)
        messages.append(response_msg)

        # Si el modelo decidió llamar a una o varias herramientas
        iteration_count = 0
        while response_msg.tool_calls and iteration_count < 3:
            iteration_count += 1
            for tool_call in response_msg.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_call_id = tool_call["id"]

                logger.info(f"LLM solicitó ejecutar herramienta '{tool_name}' con args: {tool_args}")
                tool_instance = self.tool_map.get(tool_name)

                if tool_instance:
                    try:
                        tool_output = await tool_instance.ainvoke(tool_args)
                    except Exception as err:
                        logger.error(f"Error ejecutando herramienta {tool_name}: {err}")
                        try:
                            await db.rollback()
                        except Exception:
                            pass
                        tool_output = f"[Error al ejecutar la herramienta {tool_name}]: {str(err)}"
                else:
                    tool_output = f"[Error]: Herramienta desconocida '{tool_name}'."

                tool_output_str = str(tool_output)
                executed_tools.append(ToolCallRecord(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_output=tool_output_str,
                ))

                messages.append(ToolMessage(
                    content=tool_output_str,
                    tool_call_id=tool_call_id,
                ))

            # Segunda llamada al LLM con las salidas de las herramientas para generar la respuesta final
            response_msg = await llm_with_tools.ainvoke(messages)
            messages.append(response_msg)

        return ChatResponse(
            response=response_msg.content,
            tool_calls=executed_tools,
            user_dni=user_dni,
            user_role=user_role,
        )

    async def _query_ollama_generate(self, prompt: str) -> str:
        """Invoca la API de generación de Ollama usando httpx."""
        import httpx
        url = f"{settings.OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": settings.OLLAMA_CHAT_MODEL,
            "prompt": prompt,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()

    async def _ollama_orchestration(
        self,
        message: str,
        user_dni: Optional[str],
        user_role: str,
        db: AsyncSession,
    ) -> ChatResponse:
        """
        Orquesta la consulta mediante Ollama (ej. llama3:8b).
        Ejecuta las herramientas y sintetiza la respuesta final con el LLM local.
        """
        # Ejecutar la lógica de herramientas y seguridad dentro del contexto
        agent_res = await self._mock_tool_calling_orchestration(
            message=message,
            user_dni=user_dni,
            user_role=user_role,
            db=db,
        )

        # Si se detectó una violación de privacidad, devolverla de inmediato
        if any("[VIOLACIÓN DE SEGURIDAD]" in tc.tool_output for tc in agent_res.tool_calls):
            return agent_res

        # Si se ejecutaron herramientas, redactar la respuesta con Llama 3
        if agent_res.tool_calls:
            tool_data = "\n\n".join([f"=== {tc.tool_name} ===\n{tc.tool_output}" for tc in agent_res.tool_calls])
            synthesis_prompt = f"""Eres el Asistente Inteligente Institucional de la Universidad.
El usuario preguntó: "{message}"
Rol del usuario: {user_role}

Información obtenida del sistema institucional y de los documentos oficiales:
{tool_data}

INSTRUCCIONES CLAVE:
1. Responde a la pregunta del estudiante utilizando principalmente los documentos oficiales del plan de estudios encontrados (por ejemplo Res-6790-19-ASI.pdf o el documento de la carrera).
2. Distingue con precisión:
   - Materias aprobadas o cursadas por el alumno según su historial académico.
   - La lista de materias totales que componen la carrera según el documento del plan de estudios (primer año, segundo año, tercer año).
3. Enumera las materias del plan de estudios e indica cuáles tiene aprobadas y cuáles le quedan pendientes por cursar para recibirse.
4. Si un examen parcial corresponde a una materia (por ejemplo Primer Parcial y Segundo Parcial de Análisis Matemático I), ten en cuenta que corresponden a una única asignatura de la carrera.
5. Sé claro, ordenado y formal en español."""
            try:
                llama_answer = await self._query_ollama_generate(synthesis_prompt)
                if llama_answer:
                    agent_res.response = llama_answer
            except Exception as gen_err:
                logger.warning(f"Error generando síntesis con Ollama ({gen_err}): usando respuesta base.")
        else:
            # Consulta general (saludo, agradecimiento o pregunta no institucional)
            prompt = f"Eres el Asistente Institucional de la Universidad. Responde cordialmente a: {message}"
            try:
                llama_answer = await self._query_ollama_generate(prompt)
                if llama_answer:
                    agent_res.response = llama_answer
            except Exception as gen_err:
                logger.warning(f"Error en consulta general Ollama ({gen_err})")

        return agent_res

    async def process_chat(
        self,
        message: str,
        user_dni: Optional[str],
        user_role: str,
        db: AsyncSession,
    ) -> ChatResponse:
        """
        Punto de entrada principal para procesar una solicitud conversacional.
        Garantiza que todas las herramientas corran dentro del contexto seguro ToolExecutionContext.
        """
        with ToolExecutionContext(db=db, requester_dni=user_dni, requester_role=user_role):
            provider = settings.LLM_PROVIDER.lower()
            if provider == "ollama":
                try:
                    return await self._ollama_orchestration(
                        message=message,
                        user_dni=user_dni,
                        user_role=user_role,
                        db=db,
                    )
                except Exception as e:
                    logger.warning(f"Error en conexión con Ollama ({e}). Utilizando fallback determinista...")
                    return await self._mock_tool_calling_orchestration(
                        message=message,
                        user_dni=user_dni,
                        user_role=user_role,
                        db=db,
                    )
            elif self._should_use_real_openai():
                try:
                    return await self._openai_tool_calling_orchestration(
                        message=message,
                        user_dni=user_dni,
                        user_role=user_role,
                        db=db,
                    )
                except Exception as e:
                    logger.warning(
                        f"Error en llamada a OpenAI ({e}). Utilizando fallback determinista de Tool Calling..."
                    )
                    return await self._mock_tool_calling_orchestration(
                        message=message,
                        user_dni=user_dni,
                        user_role=user_role,
                        db=db,
                    )
            else:
                return await self._mock_tool_calling_orchestration(
                    message=message,
                    user_dni=user_dni,
                    user_role=user_role,
                    db=db,
                )


chat_agent_service = ChatAgentService()
