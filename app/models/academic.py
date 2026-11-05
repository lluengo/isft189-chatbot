import uuid
from typing import Optional
from sqlalchemy import (
    Column,
    Date,
    Float,
    ForeignKey,
    Integer,
    SmallInteger,
    String,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import relationship

from app.core.database import Base

# Prefijo de esquema utilizado en todas las tablas académicas
_SCHEMA = "academico"


class Persona(Base):
    """
    Entidad base de identificación para alumnos, docentes y administrativos.
    """
    __tablename__ = "personas"
    __table_args__ = {"schema": _SCHEMA}

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    dni = Column(String(20), unique=True, nullable=False, index=True)
    nombre = Column(String(100), nullable=False)
    apellido = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    rol = Column(String(50), nullable=False)

    # Relaciones
    alumno = relationship("Alumno", back_populates="persona", uselist=False, cascade="all, delete-orphan")
    docente = relationship("Docente", back_populates="persona", uselist=False, cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Persona id={self.id} dni={self.dni} nombre={self.apellido}, {self.nombre}>"


class Carrera(Base):
    """
    Carrera o programa académico terciario.
    """
    __tablename__ = "carreras"
    __table_args__ = {"schema": _SCHEMA}

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    nombre = Column(String(150), unique=True, nullable=False)
    resolucion = Column(String(100), nullable=False)
    duracion_anos = Column(SmallInteger, nullable=False)

    # Relaciones
    materias = relationship("Materia", back_populates="carrera", cascade="all, delete-orphan")
    alumnos = relationship("Alumno", back_populates="carrera")

    # Propiedades de compatibilidad para el servicio
    @property
    def codigo(self) -> str:
        return self.resolucion

    @property
    def duracion_anios(self) -> int:
        return self.duracion_anos

    @property
    def descripcion(self) -> str:
        return f"Carrera {self.nombre} ({self.resolucion})"

    def __repr__(self) -> str:
        return f"<Carrera id={self.id} nombre={self.nombre}>"


class Materia(Base):
    """
    Asignatura perteneciente al plan de estudios de una carrera.
    """
    __tablename__ = "materias"
    __table_args__ = {"schema": _SCHEMA}

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    carrera_id = Column(
        Integer,
        ForeignKey(f"{_SCHEMA}.carreras.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    nombre = Column(String(150), nullable=False)
    anio_cursada = Column(SmallInteger, nullable=False)
    cuatrimestre = Column(SmallInteger, nullable=False, default=0)

    # Relaciones
    carrera = relationship("Carrera", back_populates="materias")
    cursadas = relationship("Cursada", back_populates="materia", cascade="all, delete-orphan")

    # Propiedades de compatibilidad
    @property
    def anio(self) -> int:
        return self.anio_cursada

    @property
    def codigo(self) -> str:
        return f"MAT-{self.id}"

    @property
    def correlativas(self) -> str:
        return "Régimen regular según plan"

    def __repr__(self) -> str:
        return f"<Materia id={self.id} nombre={self.nombre} anio={self.anio_cursada}>"


class Docente(Base):
    """
    Perfil docente vinculado a una persona.
    """
    __tablename__ = "docentes"
    __table_args__ = {"schema": _SCHEMA}

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    persona_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey(f"{_SCHEMA}.personas.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    legajo_docente = Column(String(50), unique=True, nullable=False)
    especialidad = Column(String(150), nullable=True)

    # Relaciones
    persona = relationship("Persona", back_populates="docente")
    cursadas = relationship("Cursada", back_populates="docente")

    def __repr__(self) -> str:
        return f"<Docente id={self.id} legajo={self.legajo_docente}>"


class Alumno(Base):
    """
    Perfil de alumno matriculado en una carrera, vinculado a una persona.
    """
    __tablename__ = "alumnos"
    __table_args__ = {"schema": _SCHEMA}

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    persona_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey(f"{_SCHEMA}.personas.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    legajo = Column(String(50), unique=True, nullable=False)
    carrera_id = Column(
        Integer,
        ForeignKey(f"{_SCHEMA}.carreras.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )

    # Relaciones
    persona = relationship("Persona", back_populates="alumno")
    carrera = relationship("Carrera", back_populates="alumnos")
    inscripciones = relationship("InscripcionCursada", back_populates="alumno", cascade="all, delete-orphan")

    # Propiedades de conveniencia delegadas en persona
    @property
    def dni(self) -> Optional[str]:
        return self.persona.dni if self.persona else None

    @property
    def nombre(self) -> str:
        return self.persona.nombre if self.persona else ""

    @property
    def apellido(self) -> str:
        return self.persona.apellido if self.persona else ""

    @property
    def email(self) -> str:
        return self.persona.email if self.persona else ""

    @property
    def activo(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"<Alumno id={self.id} legajo={self.legajo}>"


class Cursada(Base):
    """
    Comisión u oferta académica dictada en un ciclo lectivo.
    """
    __tablename__ = "cursadas"
    __table_args__ = {"schema": _SCHEMA}

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    materia_id = Column(
        Integer,
        ForeignKey(f"{_SCHEMA}.materias.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    docente_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey(f"{_SCHEMA}.docentes.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    ciclo_lectivo = Column(SmallInteger, nullable=False)
    cuatrimestre = Column(SmallInteger, nullable=False, default=0)

    # Relaciones
    materia = relationship("Materia", back_populates="cursadas")
    docente = relationship("Docente", back_populates="cursadas")
    inscripciones = relationship("InscripcionCursada", back_populates="cursada")

    def __repr__(self) -> str:
        return f"<Cursada id={self.id} materia_id={self.materia_id} ciclo={self.ciclo_lectivo}>"


class InscripcionCursada(Base):
    """
    Inscripción de un alumno a una cursada específica y su estado (regular, promocionado, libre).
    """
    __tablename__ = "inscripciones_cursadas"
    __table_args__ = {"schema": _SCHEMA}

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    alumno_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey(f"{_SCHEMA}.alumnos.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    cursada_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey(f"{_SCHEMA}.cursadas.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    estado = Column(String(50), nullable=False, default="inscripto")

    # Relaciones
    alumno = relationship("Alumno", back_populates="inscripciones")
    cursada = relationship("Cursada", back_populates="inscripciones")
    calificaciones = relationship("Calificacion", back_populates="inscripcion", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<InscripcionCursada id={self.id} alumno_id={self.alumno_id} estado={self.estado}>"


class Calificacion(Base):
    """
    Evaluación individual (parcial, recuperatorio, final) obtenida por el alumno en una cursada.
    """
    __tablename__ = "calificaciones"
    __table_args__ = {"schema": _SCHEMA}

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    inscripcion_cursada_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey(f"{_SCHEMA}.inscripciones_cursadas.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    instancia = Column(String(50), nullable=False)
    nota = Column(Float, nullable=False)
    fecha = Column(Date, nullable=False)

    # Relaciones
    inscripcion = relationship("InscripcionCursada", back_populates="calificaciones")

    def __repr__(self) -> str:
        return f"<Calificacion id={self.id} instancia={self.instancia} nota={self.nota}>"
