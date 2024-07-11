import os
from sqlalchemy import Column, Integer, String, ForeignKey, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy import create_engine

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Room(Base):
    __tablename__ = 'rooms'

    room_id = Column(Integer, primary_key=True, index=True)
    room_name = Column(String(255), unique=True, index=True, nullable=False)
    description = Column(String(255))
    coordinates = Column(String(255))
    picture_path = Column(String(255))
    additional_info = Column(String(255))

class Measurement(Base):
    __tablename__ = 'measurements'

    measurement_id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(TIMESTAMP, nullable=False)
    device_id = Column(String(255), nullable=False)
    room_id = Column(Integer, ForeignKey('rooms.room_id'), nullable=False)

    room = relationship("Room")

class Router(Base):
    __tablename__ = 'routers'

    router_id = Column(Integer, primary_key=True, index=True)
    ssid = Column(String(255), nullable=False)
    bssid = Column(String(255), unique=True, nullable=False)

class MeasurementRouter(Base):
    __tablename__ = 'measurement_router'

    measurement_id = Column(Integer, ForeignKey('measurements.measurement_id'), primary_key=True)
    router_id = Column(Integer, ForeignKey('routers.router_id'), primary_key=True)
    signal_strength = Column(Integer)

    measurement = relationship("Measurement")
    router = relationship("Router")
