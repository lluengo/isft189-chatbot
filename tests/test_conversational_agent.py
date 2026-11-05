import asyncio
import unittest
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from starlette.testclient import TestClient

from app.main import app
from app.models.academic import Alumno, Calificacion, Carrera, Materia
from app.services.academic_service import academic_service
from app.services.academic_tools import (
    ToolExecutionContext,
    tool_buscar_documentos_institucionales,
    tool_consultar_calificaciones_alumno,
    tool_consultar_materias_carrera,
)
from app.services.chat_agent_service import chat_agent_service


class TestConversationalAgent(unittest.IsolatedAsyncioTestCase):
    """
    Suite de pruebas unitarias y de integración para la capa de inteligencia conversacional,
    herramientas académicas y controles de seguridad RBAC.
    """

    async def asyncSetUp(self):
        # Configurar datos ficticios para los tests
        from app.models.academic import Persona, Docente, Cursada, InscripcionCursada

        self.mock_carrera = Carrera(
            id=1,
            nombre="Ingeniería en Informática",
            resolucion="Res. 5834/03",
            duracion_anos=5,
        )
        self.mock_materias = [
            Materia(id=1, carrera_id=1, nombre="Algoritmos I", anio_cursada=1, cuatrimestre=1),
            Materia(id=2, carrera_id=1, nombre="Bases de Datos", anio_cursada=2, cuatrimestre=1),
        ]
        self.mock_carrera.materias = self.mock_materias

        self.mock_persona_juan = Persona(
            id=uuid.uuid4(),
            dni="12345678",
            nombre="Juan",
            apellido="Pérez",
            email="juan.perez@test.edu",
            rol="alumno",
        )
        self.mock_alumno_juan = Alumno(
            id=uuid.uuid4(),
            persona_id=self.mock_persona_juan.id,
            legajo="LEG-1234",
            carrera_id=1,
        )
        self.mock_alumno_juan.persona = self.mock_persona_juan
        self.mock_alumno_juan.carrera = self.mock_carrera

        cur1 = Cursada(id=uuid.uuid4(), materia_id=1, docente_id=uuid.uuid4(), ciclo_lectivo=2026, cuatrimestre=1)
        cur1.materia = self.mock_materias[0]
        cur2 = Cursada(id=uuid.uuid4(), materia_id=2, docente_id=uuid.uuid4(), ciclo_lectivo=2026, cuatrimestre=1)
        cur2.materia = self.mock_materias[1]

        insc1 = InscripcionCursada(id=uuid.uuid4(), alumno_id=self.mock_alumno_juan.id, cursada_id=cur1.id, estado="regular")
        insc1.cursada = cur1
        calif1 = Calificacion(
            id=uuid.uuid4(),
            inscripcion_cursada_id=insc1.id,
            instancia="primer_parcial",
            nota=9.5,
            fecha=datetime.now().date(),
        )
        insc1.calificaciones = [calif1]

        insc2 = InscripcionCursada(id=uuid.uuid4(), alumno_id=self.mock_alumno_juan.id, cursada_id=cur2.id, estado="regular")
        insc2.cursada = cur2
        calif2 = Calificacion(
            id=uuid.uuid4(),
            inscripcion_cursada_id=insc2.id,
            instancia="primer_parcial",
            nota=8.0,
            fecha=datetime.now().date(),
        )
        insc2.calificaciones = [calif2]

        self.mock_alumno_juan.inscripciones = [insc1, insc2]

    async def test_security_student_cannot_access_other_student_grades(self):
        """
        SEGURIDAD: Un alumno autenticado con DNI 12345678 intenta consultar notas del DNI 87654321.
        Debe ser bloqueado por política de privacidad RBAC.
        """
        mock_db = AsyncMock()
        with ToolExecutionContext(db=mock_db, requester_dni="12345678", requester_role="alumno"):
            result_str = await tool_consultar_calificaciones_alumno.ainvoke({"dni": "87654321"})

        self.assertIn("[VIOLACIÓN DE SEGURIDAD]", result_str)
        self.assertIn("Acceso denegado por política de privacidad", result_str)
        # La base de datos no debe haber sido consultada
        mock_db.execute.assert_not_called()

    async def test_security_student_can_access_own_grades(self):
        """
        SEGURIDAD: Un alumno autenticado con DNI 12345678 consulta sus propias notas.
        Debe permitirse y retornar su historial de calificaciones.
        """
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = self.mock_alumno_juan
        mock_db.execute.return_value = mock_result

        with ToolExecutionContext(db=mock_db, requester_dni="12345678", requester_role="alumno"):
            result_str = await tool_consultar_calificaciones_alumno.ainvoke({"dni": "12345678"})

        self.assertNotIn("[VIOLACIÓN DE SEGURIDAD]", result_str)
        self.assertIn("Juan Pérez", result_str)
        self.assertIn("Algoritmos I", result_str)
        self.assertIn("Nota: 9.5", result_str)
        self.assertIn("Bases de Datos", result_str)

    async def test_security_docente_can_access_any_student_grades(self):
        """
        SEGURIDAD: Un docente o administrativo puede consultar notas de cualquier estudiante.
        """
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = self.mock_alumno_juan
        mock_db.execute.return_value = mock_result

        with ToolExecutionContext(db=mock_db, requester_dni="99999999", requester_role="docente"):
            result_str = await tool_consultar_calificaciones_alumno.ainvoke({"dni": "12345678"})

        self.assertNotIn("[VIOLACIÓN DE SEGURIDAD]", result_str)
        self.assertIn("Historial Académico de Juan Pérez", result_str)

    async def test_tool_consultar_materias_carrera(self):
        """
        RELACIONAL: Consulta del plan de estudios de una carrera existente.
        """
        mock_db = AsyncMock()

        # Mock primer execute (select Carrera)
        mock_carrera_res = MagicMock()
        mock_carrera_res.scalar_one_or_none.return_value = self.mock_carrera

        # Mock segundo execute (select Materia)
        mock_materias_res = MagicMock()
        mock_materias_res.scalars.return_value.all.return_value = self.mock_materias

        mock_db.execute.side_effect = [mock_carrera_res, mock_materias_res]

        with ToolExecutionContext(db=mock_db, requester_dni="12345678", requester_role="alumno"):
            result_str = await tool_consultar_materias_carrera.ainvoke({"carrera_id": 1})

        self.assertIn("Ingeniería en Informática", result_str)
        self.assertIn("Algoritmos I", result_str)
        self.assertIn("Bases de Datos", result_str)

    async def test_tool_buscar_documentos_institucionales(self):
        """
        VECTORIAL: Búsqueda en pgvector con filtro de rol.
        """
        mock_db = AsyncMock()
        with patch(
            "app.services.academic_tools.vector_search_service.search_similar_documents",
            new_callable=AsyncMock,
        ) as mock_search:
            mock_search.return_value = [
                {
                    "chunk_id": "chunk-1",
                    "filename": "reglamento_academico.pdf",
                    "category": "Reglamento",
                    "page": 3,
                    "similarity_score": 0.92,
                    "content": "El régimen de correlatividades exige tener aprobadas las materias previas.",
                }
            ]

            with ToolExecutionContext(db=mock_db, requester_dni="12345678", requester_role="alumno"):
                result_str = await tool_buscar_documentos_institucionales.ainvoke(
                    {"query": "reglamento de correlativas", "rol": "alumno"}
                )

            self.assertIn("reglamento_academico.pdf", result_str)
            self.assertIn("régimen de correlatividades", result_str)

    async def test_agent_orchestration_routes_to_grades_tool(self):
        """
        ORQUESTADOR: Mensaje solicitando notas propias invoca tool_consultar_calificaciones_alumno.
        """
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = self.mock_alumno_juan
        mock_db.execute.return_value = mock_result

        response = await chat_agent_service.process_chat(
            message="Hola, ¿cuáles son mis notas?",
            user_dni="12345678",
            user_role="alumno",
            db=mock_db,
        )

        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(response.tool_calls[0].tool_name, "tool_consultar_calificaciones_alumno")
        self.assertIn("Juan Pérez", response.response)

    async def test_agent_orchestration_blocks_idor_attack(self):
        """
        ORQUESTADOR Y SEGURIDAD: Alumno intenta consultar las notas de otro estudiante (DNI 87654321).
        El agente captura el error de seguridad y rechaza la divulgación.
        """
        mock_db = AsyncMock()

        response = await chat_agent_service.process_chat(
            message="Quiero ver las notas del alumno con DNI 87654321",
            user_dni="12345678",
            user_role="alumno",
            db=mock_db,
        )

        self.assertEqual(len(response.tool_calls), 1)
        self.assertIn("[VIOLACIÓN DE SEGURIDAD]", response.response)
        self.assertIn("Acceso denegado", response.response)

    def test_fastapi_chat_endpoint_http(self):
        """
        ENDPOINT HTTP: Verificación integral de POST /api/v1/chat con TestClient de FastAPI.
        """
        client = TestClient(app)
        with patch.object(
            chat_agent_service,
            "process_chat",
            new_callable=AsyncMock,
        ) as mock_process:
            from app.schemas.chat import ChatResponse, ToolCallRecord

            mock_process.return_value = ChatResponse(
                response="Tus notas son excelentes.",
                tool_calls=[
                    ToolCallRecord(
                        tool_name="tool_consultar_calificaciones_alumno",
                        tool_args={"dni": "12345678"},
                        tool_output="Nota 9.5 en Algoritmos",
                    )
                ],
                user_dni="12345678",
                user_role="alumno",
            )

            payload = {
                "message": "¿Cuáles son mis calificaciones?",
                "user_dni": "12345678",
                "user_role": "alumno",
            }
            res = client.post("/api/v1/chat", json=payload)

            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["response"], "Tus notas son excelentes.")
            self.assertEqual(len(data["tool_calls"]), 1)
            self.assertEqual(data["tool_calls"][0]["tool_name"], "tool_consultar_calificaciones_alumno")
            self.assertEqual(data["user_dni"], "12345678")
            self.assertEqual(data["user_role"], "alumno")


if __name__ == "__main__":
    unittest.main()
