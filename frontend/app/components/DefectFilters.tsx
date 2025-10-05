'use client'
import React from "react";
import Button from "@mui/material/Button";
import TextField from "@mui/material/TextField";
import MenuItem from "@mui/material/MenuItem";
import { DateRange, Range } from "react-date-range";
import { ru } from "date-fns/locale";
import { format, subHours, subDays, subWeeks } from "date-fns";

interface Props {
  selectedPart: string;
  setSelectedPart: (value: string) => void;
  dateRange: Range[];
  setDateRange: (value: Range[]) => void;
}

const partTypes = ["Все детали", "Кран (1/2)", "Кран (3/4)"];

export default function DefectFilters({
  selectedPart,
  setSelectedPart,
  dateRange,
  setDateRange,
}: Props) {
  const [showCalendar, setShowCalendar] = React.useState(false);

  const quickFilters = {
    hour: () =>
      setDateRange([
        { startDate: subHours(new Date(), 1), endDate: new Date(), key: "selection" },
      ]),
    day: () =>
      setDateRange([
        { startDate: subDays(new Date(), 1), endDate: new Date(), key: "selection" },
      ]),
    week: () =>
      setDateRange([
        { startDate: subWeeks(new Date(), 1), endDate: new Date(), key: "selection" },
      ]),
  };

  return (
    <div className="flex flex-wrap items-center gap-4 mb-4">
      <TextField
        select
        label="Фильтр по детали"
        value={selectedPart}
        onChange={(e) => setSelectedPart(e.target.value)}
        size="small"
      >
        {partTypes.map((part) => (
          <MenuItem key={part} value={part}>
            {part}
          </MenuItem>
        ))}
      </TextField>

      <div className="flex items-center gap-2">
        <Button variant="outlined" size="small" onClick={quickFilters.hour}>
          За 1 час
        </Button>
        <Button variant="outlined" size="small" onClick={quickFilters.day}>
          За 1 день
        </Button>
        <Button variant="outlined" size="small" onClick={quickFilters.week}>
          За 1 неделю
        </Button>
      </div>

      <div className="relative">
        <button
          onClick={() => setShowCalendar(!showCalendar)}
          className="border border-gray-300 rounded-xl px-4 py-2 bg-white shadow-sm hover:bg-gray-50"
        >
          {dateRange[0].startDate
            ? format(dateRange[0].startDate, "yyyy-MM-dd")
            : "Выберите дату"}{" "}
          —{" "}
          {dateRange[0].endDate
            ? format(dateRange[0].endDate, "yyyy-MM-dd")
            : "Выберите дату"}
        </button>

        {showCalendar && (
          <div className="absolute z-50 mt-2 bg-white shadow-lg rounded-xl border p-3">
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
  );
}
