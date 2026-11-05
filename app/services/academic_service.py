import logging
from typing import Any, Dict, List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.academic import (
    Persona,
    Alumno,
    Calificacion,
    Carrera,
    Materia,
    Cursada,
    InscripcionCursada,
)

logger = logging.getLogger(__name__)


class SecurityAuthorizationError(Exception):
    """Excepción lanzada ante intentos no autorizados de acceso a datos sensibles de terceros."""
    pass


class AcademicService:
    """
    Servicio de acceso al dominio académico relacional con controles de seguridad y privacidad RBAC.
    """

    async def get_student_grades(
        self,
        target_dni: str,
        requester_dni: Optional[str],
        requester_role: str,
        db: AsyncSession,
    ) -> Dict[str, Any]:
        """
        Consulta las calificaciones y estado académico de un alumno en la base de datos relacional.
        Aplica validación estricta de seguridad:
        - Si el rol es 'alumno', el DNI solicitado debe coincidir exactamente con su identidad.
        - Roles privilegiados ('docente', 'administrativo', 'admin') pueden consultar cualquier DNI.
        """
        clean_target_dni = str(target_dni or "").strip()
        clean_requester_dni = str(requester_dni or "").strip() if requester_dni else None
        clean_role = str(requester_role or "alumno").strip().lower()

        logger.info(
            f"Consulta de calificaciones - Solicitante: [DNI={clean_requester_dni}, Rol={clean_role}], Target DNI={clean_target_dni}"
        )

        # 1. Validación de Seguridad y Privacidad
        if clean_role == "alumno":
            if not clean_requester_dni:
                logger.warning("Intento de acceso a calificaciones sin DNI de identidad del solicitante.")
                return {
                    "error": "Acceso denegado: Debes identificarte con un DNI válido para consultar tus calificaciones.",
                    "security_violation": True,
                }

            if clean_target_dni != clean_requester_dni:
                logger.warning(
                    f"Violación de seguridad detectada: Alumno con DNI {clean_requester_dni} intentó acceder a calificaciones del DNI {clean_target_dni}"
                )
                return {
                    "error": (
                        f"Acceso denegado por política de privacidad: Como alumno (DNI: {clean_requester_dni}) "
                        f"únicamente tienes autorización para consultar tu propio historial académico, "
                        f"no el de otros estudiantes (DNI solicitado: {clean_target_dni})."
                    ),
                    "security_violation": True,
                }

        # 2. Búsqueda del Alumno en BD mediante Persona
        stmt = (
            select(Alumno)
            .join(Persona, Alumno.persona_id == Persona.id)
            .options(
                selectinload(Alumno.persona),
                selectinload(Alumno.carrera),
                selectinload(Alumno.inscripciones)
                    .selectinload(InscripcionCursada.cursada)
                    .selectinload(Cursada.materia),
                selectinload(Alumno.inscripciones)
                    .selectinload(InscripcionCursada.calificaciones),
            )
            .where(Persona.dni == clean_target_dni)
        )
        result = await db.execute(stmt)
        alumno = result.scalar_one_or_none()

        if not alumno:
            return {
                "mensaje": f"No se encontró ningún estudiante registrado con el DNI {clean_target_dni} en el sistema académico.",
                "total_registros": 0,
                "calificaciones": [],
            }

        # 3. Formateo de Calificaciones desde Inscripciones y Calificaciones
        records = []
        for insc in alumno.inscripciones:
            materia_nombre = (
                insc.cursada.materia.nombre
                if (insc.cursada and insc.cursada.materia)
                else "Materia Desconocida"
            )
            if insc.calificaciones:
                for cal in insc.calificaciones:
                    records.append({
                        "materia": materia_nombre,
                        "codigo": cal.instancia.replace("_", " ").title(),
                        "nota": cal.nota,
                        "estado": insc.estado.capitalize(),
                        "fecha": cal.fecha.strftime("%Y-%m-%d") if hasattr(cal.fecha, "strftime") else str(cal.fecha),
                    })
            else:
                records.append({
                    "materia": materia_nombre,
                    "codigo": "Sin evaluaciones",
                    "nota": None,
                    "estado": insc.estado.capitalize(),
                    "fecha": "En curso",
                })

        carrera_nombre = alumno.carrera.nombre if alumno.carrera else "Sin carrera asignada"

        return {
            "alumno": f"{alumno.nombre} {alumno.apellido}",
            "dni": alumno.dni,
            "carrera": carrera_nombre,
            "total_materias_registradas": len(records),
            "calificaciones": records,
        }

    async def get_career_subjects(
        self,
        carrera_id: int,
        db: AsyncSession,
    ) -> Dict[str, Any]:
        """
        Consulta y devuelve el plan de estudios relacional para una carrera dada.
        """
        # 1. Buscar la carrera por ID o si carrera_id es inválido, buscar por defecto la primera disponible
        carrera_stmt = select(Carrera).where(Carrera.id == carrera_id)
        carrera_res = await db.execute(carrera_stmt)
        carrera = carrera_res.scalar_one_or_none()

        if not carrera:
            # Si no existe por ID exacto, buscar si hay alguna carrera cargada
            todas_carreras_res = await db.execute(select(Carrera))
            carreras = todas_carreras_res.scalars().all()
            if len(carreras) == 1:
                # Si solo hay una carrera registrada, usar esa directamente
                carrera = carreras[0]
            elif carreras:
                opciones = [{"id": c.id, "nombre": c.nombre, "codigo": c.resolucion} for c in carreras]
                return {
                    "error": f"No se encontró la carrera con ID {carrera_id}.",
                    "carreras_disponibles": opciones,
                }
            else:
                return {
                    "error": "No hay carreras registradas en el sistema.",
                    "carreras_disponibles": [],
                }

        # 2. Buscar materias ordenadas por año y cuatrimestre
        materias_stmt = (
            select(Materia)
            .where(Materia.carrera_id == carrera_id)
            .order_by(Materia.anio_cursada.asc(), Materia.cuatrimestre.asc(), Materia.nombre.asc())
        )
        materias_res = await db.execute(materias_stmt)
        materias = materias_res.scalars().all()

        plan = []
        for mat in materias:
            plan.append({
                "id": mat.id,
                "codigo": mat.codigo,
                "nombre": mat.nombre,
                "anio": mat.anio_cursada,
                "cuatrimestre": mat.cuatrimestre,
                "correlativas": mat.correlativas or "Ninguna",
            })

        return {
            "carrera_id": carrera.id,
            "carrera_nombre": carrera.nombre,
            "carrera_codigo": carrera.resolucion,
            "duracion_anios": carrera.duracion_anos,
            "descripcion": f"Carrera {carrera.nombre} ({carrera.resolucion})",
            "total_materias": len(plan),
            "plan_estudios": plan,
        }

    async def seed_sample_data_if_empty(self, db: AsyncSession) -> None:
        """
        Puebla la base de datos con carreras, materias, alumnos y calificaciones de ejemplo
        si no existe el alumno de prueba o las carreras.
        """
        # Verificar si ya existe el alumno de prueba
        res_alu = await db.execute(select(Persona).where(Persona.dni == "45200300"))
        if res_alu.scalar_one_or_none() is not None:
            return  # Ya existe el alumno

        # Obtener o crear carrera
        res_carr = await db.execute(select(Carrera).limit(1))
        carrera = res_carr.scalar_one_or_none()

        if not carrera:
            logger.info("Inicializando datos académicos de prueba (Seeding)...")
            carrera = Carrera(
                nombre="Análisis de Sistemas",
                resolucion="Res. 5834/03",
                duracion_anos=3,
            )
            db.add(carrera)
            await db.flush()

        # Verificar materias
        res_m1 = await db.execute(select(Materia).where(Materia.carrera_id == carrera.id).limit(1))
        m1 = res_m1.scalar_one_or_none()
        if not m1:
            m1 = Materia(carrera_id=carrera.id, nombre="Álgebra", anio_cursada=1, cuatrimestre=0)
            m2 = Materia(carrera_id=carrera.id, nombre="Análisis Matemático I", anio_cursada=1, cuatrimestre=0)
            m3 = Materia(carrera_id=carrera.id, nombre="Algoritmo y estructura de datos I", anio_cursada=1, cuatrimestre=0)
            db.add_all([m1, m2, m3])
            await db.flush()
        else:
            m2 = m1

        # Crear o buscar Docente
        res_doc = await db.execute(select(Persona).where(Persona.dni == "25111222"))
        per_doc = res_doc.scalar_one_or_none()
        if not per_doc:
            per_doc = Persona(
                dni="25111222",
                nombre="Laura",
                apellido="Rossi",
                email="laura.rossi@instituto.edu.ar",
                rol="docente",
            )
            db.add(per_doc)
            await db.flush()

        from app.models.academic import Docente, Cursada
        res_docente = await db.execute(select(Docente).where(Docente.persona_id == per_doc.id))
        docente = res_docente.scalar_one_or_none()
        if not docente:
            docente = Docente(persona_id=per_doc.id, legajo_docente="DOC-0101", especialidad="Ciencias Exactas")
            db.add(docente)
            await db.flush()

        # Crear o buscar Cursada
        res_cur = await db.execute(select(Cursada).where(Cursada.materia_id == m2.id).limit(1))
        cur2 = res_cur.scalar_one_or_none()
        if not cur2:
            cur2 = Cursada(materia_id=m2.id, docente_id=docente.id, ciclo_lectivo=2026, cuatrimestre=0)
            db.add(cur2)
            await db.flush()

        # Crear Alumno
        per_alu = Persona(
            dni="45200300",
            nombre="Sofía",
            apellido="Rodríguez",
            email="sofia.rodriguez@alumnos.instituto.edu.ar",
            rol="alumno",
        )
        db.add(per_alu)
        await db.flush()

        alumno = Alumno(persona_id=per_alu.id, legajo="AS-2026-002", carrera_id=carrera.id)
        db.add(alumno)
        await db.flush()

        # Inscripciones y Calificaciones
        insc1 = InscripcionCursada(alumno_id=alumno.id, cursada_id=cur2.id, estado="promocionado")
        db.add(insc1)
        await db.flush()

        from datetime import date
        cal1 = Calificacion(
            inscripcion_cursada_id=insc1.id,
            instancia="primer_parcial",
            nota=9.5,
            fecha=date(2026, 5, 15),
        )
        cal2 = Calificacion(
            inscripcion_cursada_id=insc1.id,
            instancia="segundo_parcial",
            nota=10.0,
            fecha=date(2026, 6, 25),
        )
        db.add_all([cal1, cal2])

        await db.commit()
        logger.info("Datos académicos de prueba sembrados exitosamente.")


academic_service = AcademicService()
