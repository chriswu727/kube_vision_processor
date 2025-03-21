from sqlalchemy import Column, Integer, String, DateTime, create_engine, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

Base = declarative_base()

class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, index=True)
    cache_key = Column(String)
    format = Column(String)
    size = Column(String)
    image_type = Column(String)  # 'face' or 'object'
    status = Column(String, default="uploaded")  # uploaded, processing, completed, failed
    task_id = Column(String, nullable=True)
    result = Column(JSON, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.datetime.utcnow)

# Database connection
DATABASE_URL = "postgresql://admin:admin123@postgres:5432/kubevision"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
def init_db():
    Base.metadata.create_all(bind=engine) 