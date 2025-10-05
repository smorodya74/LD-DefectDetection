'use client';
import React, { useState, useEffect } from "react";
import DefectTable from "../components/DefectTable";
import { Range } from "react-date-range";
import DefectFilters from "../components/DefectFilters";

interface RawData {
  date: string;
  time: string;
  part: string;
  produced: number;
  defective: number;
}

interface Data {
  ID: number;
  "Дата-Время": string;
  "Тип детали": string;
  Статус: string;
  "Вид брака": string;
  imageUrl: string;
}

export default function DefectsPage() {
  const [rows, setRows] = useState<Data[]>([]);
  const [selectedPart, setSelectedPart] = useState<string>("Все детали");

  // диапазон дат: вчера — сегодня
  const today = new Date();
  const yesterday = new Date();
  yesterday.setDate(today.getDate() - 1);

  const [dateRange, setDateRange] = useState<Range[]>([
    {
      startDate: yesterday,
      endDate: today,
      key: "selection",
    },
  ]);

  // === Загрузка и преобразование данных ===
  useEffect(() => {
    fetch("/data/production_data_sep1_oct4_2025.json")
      .then((res) => res.json())
      .then((data: RawData[]) => {
        const transformed: Data[] = data.map((item, index) => ({
          ID: index + 1,
          "Дата-Время": `${item.date} ${item.time}`,
          "Тип детали": item.part,
          Статус: item.defective > 0 ? "Брак" : "ОК",
          "Вид брака": "-", // можно заменить, если появятся данные о дефекте
          imageUrl: "/images/noimage.jpg", // пока заглушка
        }));
        setRows(transformed);
      })
      .catch((err) => console.error("Ошибка загрузки JSON:", err));
  }, []);

  return (
    <div
      className="p-6 bg-white flex flex-col"
      style={{ height: "calc(100vh - 48px)" }} // учитываем отступы
    >
      {/* Шапка */}
      <div className="mb-4">
        <h1 className="text-2xl font-bold mb-4">Дефекты</h1>

        <DefectFilters
          selectedPart={selectedPart}
          setSelectedPart={setSelectedPart}
          dateRange={dateRange}
          setDateRange={setDateRange}
        />
      </div>

      {/* Рабочая область */}
      <div className="flex-1 overflow-auto">
        <DefectTable
          rows={rows}
          selectedPart={selectedPart}
          dateRange={dateRange}
        />
      </div>
    </div>
  );
}
