from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, TIMESTAMP, Float
from sqlalchemy.orm import sessionmaker, declarative_base
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.orm import Session
from fastapi import Depends
from datetime import datetime
import os
import shutil

DATABASE_URL = f"postgresql://postgres:postgres@pg:5432/lddb"

# --- Подключение к базе данных ---
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

# --- Модель таблицы ---
class Detail(Base):
    __tablename__ = "details"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(TIMESTAMP, default=datetime.utcnow)
    type_detail = Column(String, nullable=True)
    status = Column(String, nullable=True)
    score = Column(Float, nullable=True)
    percent = Column(Float, nullable=True)
    image = Column(String, nullable=False)

# --- Создаём таблицу, если её нет ---
Base.metadata.create_all(bind=engine)

# --- FastAPI ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # можно указать URL фронтенда в продакшене
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "img"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_files(
    files: list[UploadFile] = File(...),
    type_detail: str = Form(None),
    status: str = Form(None),
    score: float = Form(None),
    percent: float = Form(None)
):
    saved_files = []
    db = SessionLocal()
    try:
        for file in files:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            detail = Detail(
                type_detail=type_detail,
                status=status,
                score=score,
                percent=percent,
                image=file.filename
            )
            db.add(detail)
            saved_files.append(file.filename)

        db.commit()
    except Exception as e:
        db.rollback()
        return {"error": str(e)}
    finally:
        db.close()



    return {"uploaded_files": saved_files}



def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/details")
def get_details(db: Session = Depends(get_db)):
    # Сортировка по дате по возрастанию
    details = db.query(Detail).order_by(Detail.timestamp.asc()).all()

    result = [
        {
            "ID": d.id,
            "Дата-Время": d.timestamp.strftime("%Y-%m-%d %H:%M"),
            "Тип детали": d.type_detail,
            "Статус": d.status,
            "Вид брака": d.status,  # или другая колонка, если есть
            "imageUrl": f"/img/{d.image}"
        }
        for d in details
    ]
    return JSONResponse(content=result)