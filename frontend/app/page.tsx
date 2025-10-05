"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

interface UploadFile {
  file: File;
}

export default function Home() {
  const router = useRouter();
  const [files, setFiles] = useState<UploadFile[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [progress, setProgress] = useState(0); // общий прогресс
  const [isUploading, setIsUploading] = useState(false);

  const [detailData] = useState({
    number: "D-001",
    status: "Активна",
  });

  const [generalData] = useState({
    time: new Date().toLocaleTimeString(),
    totalDetails: 100,
    defective: 5,
    completed: 95,
  });

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFiles = Array.from(e.dataTransfer.files).map(f => ({ file: f }));
    setFiles(prev => [...prev, ...droppedFiles]);
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => e.preventDefault();
  const handleDragEnter = () => setIsDragging(true);
  const handleDragLeave = () => setIsDragging(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const selectedFiles = Array.from(e.target.files).map(f => ({ file: f }));
      setFiles(prev => [...prev, ...selectedFiles]);
    }
  };

  const removeFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const uploadFiles = () => {
    if (files.length === 0) return;
    setIsUploading(true);
    setProgress(0);

    const formData = new FormData();
    files.forEach(f => formData.append("files", f.file));

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "http://localhost:8001/upload");

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        const percent = Math.round((event.loaded / event.total) * 100);
        setProgress(percent);
      }
    };

    xhr.onload = () => {
      if (xhr.status === 200) {
        setFiles([]); // очистить массив после успешной загрузки
        setProgress(100);
        setIsUploading(false);
      } else {
        alert("Ошибка загрузки файлов");
        setIsUploading(false);
      }
    };

    xhr.onerror = () => {
      alert("Ошибка соединения с сервером");
      setIsUploading(false);
    };

    xhr.send(formData);
  };

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-6">Домашняя страница</h1>

      {/* Drag & Drop зона */}
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        className={`
          border-4 rounded-xl p-8 mb-6 transition-all duration-300
          flex flex-col items-center justify-center
          ${isDragging ? "border-blue-400 bg-blue-50" : "border-gray-300 bg-gray-50"}
          cursor-pointer hover:border-blue-500
        `}
      >
        {files.length === 0 ? (
          <>
            <p className="mb-4 text-gray-600 text-lg">
              Перетащите сюда фотографии или нажмите кнопку ниже
            </p>
            <label className="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 cursor-pointer transition">
              Прикрепить фото
              <input type="file" multiple accept="image/*" onChange={handleFileChange} className="hidden" />
            </label>
          </>
        ) : (
          <>
            <div className="flex flex-wrap gap-4 justify-center mb-4">
              {files.map((uploadFile, idx) => (
                <div key={idx} className="w-32 h-40 border rounded-lg overflow-hidden relative flex flex-col">
                  <img
                    src={URL.createObjectURL(uploadFile.file)}
                    alt={uploadFile.file.name}
                    className="w-full h-28 object-cover"
                  />
                  <button
                    onClick={() => removeFile(idx)}
                    className="absolute top-1 right-1 bg-red-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm hover:bg-red-600"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>

            <div className="w-full bg-gray-200 h-4 rounded mb-4">
              <div
                className="bg-green-500 h-4 rounded"
                style={{ width: `${progress}%` }}
              ></div>
            </div>
            <span className="text-center block mb-4">{progress}%</span>

            <button
              onClick={uploadFiles}
              disabled={isUploading}
              className="px-6 py-3 bg-green-500 text-white rounded-lg hover:bg-green-600 transition"
            >
              {isUploading ? "Загрузка..." : "Загрузить фото"}
            </button>
          </>
        )}
      </div>

      {/* Два квадрата вертикально */}
      <div className="flex flex-col gap-6">
        {/* Квадрат 1: данные детали */}
        <div className="border rounded p-4 w-[650px] h-[200px] bg-gray-100 flex flex-col justify-between">
          <h2 className="text-lg font-bold mb-2">Данные детали</h2>
          <p><strong>Номер детали:</strong> {detailData.number}</p>
          <p><strong>Статус детали:</strong> {detailData.status}</p>
          <button
            onClick={() => router.push("/defects")}
            className="mt-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 self-start"
          >
            Подробнее
          </button>
        </div>

        {/* Квадрат 2: общая информация */}
        <div className="border rounded p-4 w-[650px] h-[200px] bg-gray-100 flex flex-col justify-between">
          <h2 className="text-lg font-bold mb-2">Общая информация</h2>
          <p><strong>Время:</strong> {generalData.time}</p>
          <p><strong>Общее количество деталей:</strong> {generalData.totalDetails}</p>
          <p><strong>Бракованные детали:</strong> {generalData.defective}</p>
          <p><strong>Совершенные детали:</strong> {generalData.completed}</p>
          <button
            onClick={() => router.push("/defects")}
            className="mt-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 self-start"
          >
            Подробнее
          </button>
        </div>
      </div>
    </div>
  );
}
