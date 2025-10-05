'use client'
import React from "react";
import Paper from "@mui/material/Paper";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TablePagination from "@mui/material/TablePagination";
import TableRow from "@mui/material/TableRow";
import Button from "@mui/material/Button";
import { Range } from "react-date-range";

interface Column {
  id: "ID" | "Дата-Время" | "Тип детали" | "Статус" | "Вид брака" | "Действие";
  label: string;
  minWidth?: number;
  align?: "center" | "left";
}

const columns: readonly Column[] = [
  { id: "ID", label: "ID", minWidth: 50, align: "center" },
  { id: "Дата-Время", label: "Дата-Время", minWidth: 150, align: "center" },
  { id: "Тип детали", label: "Тип детали", minWidth: 120, align: "center" },
  { id: "Статус", label: "Статус", minWidth: 100, align: "center" },
  { id: "Вид брака", label: "Вид брака", minWidth: 120, align: "center" },
  { id: "Действие", label: "Действие", minWidth: 100, align: "center" },
];

interface Data {
  ID: number;
  "Дата-Время": string;
  "Тип детали": string;
  Статус: string;
  "Вид брака": string;
  imageUrl: string;
}

interface Props {
  rows: Data[];
  selectedPart: string;
  dateRange: Range[];
}

export default function DefectTable({ rows, selectedPart, dateRange }: Props) {
  const [page, setPage] = React.useState(0);
  const [rowsPerPage, setRowsPerPage] = React.useState(10);

  const handleChangePage = (event: unknown, newPage: number) => setPage(newPage);
  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(+event.target.value);
    setPage(0);
  };
  const openImage = (url: string) => window.open(url, "_blank");

  const filteredRows = rows.filter((row) => {
    const rowDate = new Date(row["Дата-Время"]);
    const matchesPart = selectedPart === "Все детали" || row["Тип детали"] === selectedPart;
    const start = dateRange[0].startDate;
    const end = dateRange[0].endDate;
    const matchesDate = (!start || rowDate >= start) && (!end || rowDate <= end);
    return matchesPart && matchesDate;
  });

  return (
    <Paper sx={{ width: "100%", overflow: "hidden" }}>
      <TableContainer sx={{ maxHeight: 440 }}>
        <Table stickyHeader aria-label="sticky table">
          <TableHead>
            <TableRow>
              {columns.map((column) => (
                <TableCell key={column.id} align={column.align} style={{ minWidth: column.minWidth }}>
                  {column.label}
                </TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {filteredRows.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage).map((row) => (
              <TableRow hover key={row.ID}>
                {columns.map((column) =>
                  column.id === "Действие" ? (
                    <TableCell key={column.id} align={column.align}>
                      <Button variant="contained" size="small" onClick={() => openImage(row.imageUrl)}>
                        Посмотреть
                      </Button>
                    </TableCell>
                  ) : (
                    <TableCell key={column.id} align={column.align}>
                      {row[column.id]}
                    </TableCell>
                  )
                )}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
      <TablePagination
        rowsPerPageOptions={[10, 25, 100]}
        component="div"
        count={filteredRows.length}
        rowsPerPage={rowsPerPage}
        page={page}
        onPageChange={handleChangePage}
        onRowsPerPageChange={handleChangeRowsPerPage}
      />
    </Paper>
  );
}
