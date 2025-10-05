'use client';

import { useEffect, useState, useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  PieChart,
  Pie,
  Cell,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { DateRange, Range } from "react-date-range";
import { ru } from "date-fns/locale";
import "react-date-range/dist/styles.css";
import "react-date-range/dist/theme/default.css";

interface Detail {
  ID: number;
  "Дата-Время": string;
  "Тип детали": string;
  Статус: string;
  "Вид брака": string;
  imageUrl: string;
}

export default function GraphicsPage() {
  const [details, setDetails] = useState<Detail[]>([]);
  const [showCalendar, setShowCalendar] = useState(false);
  const [selectedPart, setSelectedPart] = useState<string>("Все детали");

  const [dateRange, setDateRange] = useState<Range[]>([
    { startDate: new Date("2025-09-01"), endDate: new Date("2025-10-04"), key: "selection" },
  ]);

  const COLORS = ["#0088FE", "#FF8042", "#00C49F", "#FFBB28"];

  // === Загрузка данных из FastAPI ===
  useEffect(() => {
    fetch("http://localhost:8001/details")
      .then(res => res.json())
      .then(data => setDetails(data))
      .catch(err => console.error("Ошибка при загрузке данных:", err));
  }, []);

  // Динамические типы деталей
  const partTypes = ["Все детали", ...Array.from(new Set(details.map(d => d["Тип детали"])))];

  // Фильтрация по дате и детали
  const filtered = useMemo(() => {
    const start = dateRange[0].startDate!;
    const end = dateRange[0].endDate!;
    return details.filter(d => {
      const date = new Date(d["Дата-Время"]);
      const matchesDate = date >= start && date <= end;
      const matchesPart = selectedPart === "Все детали" || d["Тип детали"] === selectedPart;
      return matchesDate && matchesPart;
    });
  }, [details, dateRange, selectedPart]);

  // === Подготовка данных для графиков ===
  const chartData = useMemo(() => {
    if (!filtered.length) return [];

    const start = dateRange[0].startDate!;
    const end = dateRange[0].endDate!;
    const isSingleDay = start.toDateString() === end.toDateString();
    if (isSingleDay) end.setHours(23, 59, 59, 999);

    if (isSingleDay) {
      // Динамика по часам
      const grouped: Record<string, { produced: number; defective: number }> = {};
      filtered.forEach(d => {
        const hour = new Date(d["Дата-Время"]).getHours().toString().padStart(2, "0");
        if (!grouped[hour]) grouped[hour] = { produced: 0, defective: 0 };
        grouped[hour].produced += 1;
        grouped[hour].defective += d.Статус !== "Без дефектов" ? 1 : 0;
      });

      return Object.entries(grouped)
        .sort(([a], [b]) => parseInt(a) - parseInt(b))
        .map(([hour, stats]) => ({
          time: `${hour}:00`,
          produced: stats.produced,
          defective: stats.defective,
        }));
    } else {
      // Динамика по дням
      const grouped: Record<string, { produced: number; defective: number }> = {};
      filtered.forEach(d => {
        const dateLabel = new Date(d["Дата-Время"]).toLocaleDateString();
        if (!grouped[dateLabel]) grouped[dateLabel] = { produced: 0, defective: 0 };
        grouped[dateLabel].produced += 1;
        grouped[dateLabel].defective += d.Статус !== "Без дефектов" ? 1 : 0;
      });

      return Object.entries(grouped)
        .sort(([a], [b]) => new Date(a).getTime() - new Date(b).getTime())
        .map(([dateLabel, stats]) => ({
          dateLabel,
          produced: stats.produced,
          defective: stats.defective,
        }));
    }
  }, [filtered, dateRange]);

  const defectsData = useMemo(() => {
    const grouped: Record<string, number> = {};
    filtered.forEach(d => {
      if (d.Статус === "Без дефектов") return;
      const type = d["Вид брака"];
      grouped[type] = (grouped[type] || 0) + 1;
    });
    return Object.entries(grouped).map(([name, value]) => ({ name, value }));
  }, [filtered]);

  // === Таблица процент брака ===
  const defectStats = useMemo(() => {
    const grouped: Record<string, { produced: number; defective: number }> = {};
    filtered.forEach(d => {
      const part = d["Тип детали"];
      if (!grouped[part]) grouped[part] = { produced: 0, defective: 0 };
      grouped[part].produced += 1;
      grouped[part].defective += d.Статус !== "Без дефектов" ? 1 : 0;
    });
    return Object.entries(grouped).map(([part, stats]) => ({
      part,
      produced: stats.produced,
      defective: stats.defective,
      percent: stats.produced ? ((stats.defective / stats.produced) * 100).toFixed(2) : "0.00",
    }));
  }, [filtered]);

  const defectTableTitle = useMemo(() => {
    const start = dateRange[0].startDate!;
    const end = dateRange[0].endDate!;
    return `Процент брака по типам деталей за ${start.toLocaleDateString()} — ${end.toLocaleDateString()}`;
  }, [dateRange]);

  const xAxisKey = useMemo(() => {
    const start = dateRange[0].startDate!;
    const end = dateRange[0].endDate!;
    const oneDay = 24 * 60 * 60 * 1000;
    return (end.getTime() - start.getTime()) / oneDay <= 1 ? "time" : "dateLabel";
  }, [dateRange]);

  return (
    <div className="p-6 space-y-10 bg-white min-h-screen">
      {/* Панель выбора */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <h1 className="text-2xl font-bold">Аналитика производства</h1>
        <div className="flex gap-4 items-center">
          <select
            value={selectedPart}
            onChange={(e) => setSelectedPart(e.target.value)}
            className="border border-gray-300 rounded-xl px-4 py-2 bg-white shadow-sm"
          >
            {partTypes.map(pt => <option key={pt}>{pt}</option>)}
          </select>
          <div className="relative">
            <button
              onClick={() => setShowCalendar(!showCalendar)}
              className="border border-gray-300 rounded-xl px-4 py-2 bg-white shadow-sm hover:bg-gray-50"
            >
              {dateRange[0].startDate?.toLocaleDateString()} — {dateRange[0].endDate?.toLocaleDateString()}
            </button>
            {showCalendar && (
              <div className="absolute right-0 mt-2 z-50 bg-white shadow-lg rounded-xl border p-3">
                <DateRange
                  editableDateInputs
                  onChange={(item) => setDateRange([item.selection])}
                  moveRangeOnFirstSelection={false}
                  ranges={dateRange}
                  locale={ru}
                  maxDate={new Date("2025-10-04")}
                />
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="grid gap-8 md:grid-cols-2">
        {/* Выпуск деталей */}
        {/* <div className="bg-white rounded-2xl shadow p-6">
          <h2 className="text-xl font-semibold mb-2">Выпуск деталей</h2>
          <div className="w-full h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey={xAxisKey} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="produced" stroke="#0088FE" name="Произведено" />
                <Line type="monotone" dataKey="defective" stroke="#FF0000" name="Брак" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div> */}

        {/* Виды брака */}
        {/* <div className="bg-white rounded-2xl shadow p-4">
          <h2 className="text-xl font-semibold mb-2">Виды брака</h2>
          <div className="w-full h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={defectsData} dataKey="value" cx="50%" cy="50%" outerRadius={100} label>
                  {defectsData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div> */}


        <div className="bg-white rounded-2xl shadow p-4 md:col-span-2">
          <h2 className="text-xl font-semibold mb-2">Выпуск деталей</h2>
          <div className="w-full h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey={xAxisKey} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="produced" stroke="#0088FE" name="Произведено" />
                <Line type="monotone" dataKey="defective" stroke="#FF0000" strokeWidth={2} name="Количество брака" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Динамика брака */}
        <div className="bg-white rounded-2xl shadow p-4 md:col-span-2">
          <h2 className="text-xl font-semibold mb-2">Динамика брака</h2>
          <div className="w-full h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey={xAxisKey} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="defective" stroke="#FF0000" strokeWidth={2} name="Количество брака" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Таблица */}
      <div className="bg-white rounded-2xl shadow p-6">
        <h2 className="text-xl font-semibold mb-4">{defectTableTitle}</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full border border-gray-200 text-sm">
            <thead className="bg-gray-100">
              <tr>
                <th className="px-4 py-2 border">Тип детали</th>
                <th className="px-4 py-2 border">Произведено</th>
                <th className="px-4 py-2 border">Брак</th>
                <th className="px-4 py-2 border">% Брака</th>
              </tr>
            </thead>
            <tbody>
              {defectStats.length ? defectStats.map(row => (
                <tr key={row.part} className="text-center hover:bg-gray-50">
                  <td className="px-4 py-2 border font-medium">{row.part}</td>
                  <td className="px-4 py-2 border">{row.produced}</td>
                  <td className="px-4 py-2 border text-red-600">{row.defective}</td>
                  <td className={`px-4 py-2 border font-semibold ${parseFloat(row.percent) > 5 ? "text-red-500" : "text-green-600"}`}>
                    {row.percent}%
                  </td>
                </tr>
              )) : (
                <tr>
                  <td colSpan={4} className="text-center py-4 text-gray-500">Нет данных за выбранный период</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
