"use client";
import { ReactNode, useState } from "react";
import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";
import Button from "@mui/material/Button";
import ArrowForwardIosOutlinedIcon from "@mui/icons-material/ArrowForwardIosOutlined";
import ArrowBackIosOutlinedIcon from "@mui/icons-material/ArrowBackIosOutlined";
import DeleteSweepIcon from '@mui/icons-material/DeleteSweep';
import AreaChartIcon from "@mui/icons-material/AreaChart";
import HomeIcon from "@mui/icons-material/Home";
import SettingsIcon from "@mui/icons-material/Settings";

type NavItem = {
  label: string;
  href: string;
  icon: ReactNode;
};

const navItems: NavItem[] = [
  { label: "Главная", href: "/", icon: <HomeIcon /> },
  { label: "Аналитика", href: "/graphics", icon: <AreaChartIcon /> },
  { label: "База дефектов", href: "/defects", icon: <DeleteSweepIcon />}
];

export default function Sidebar() {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const pathname = usePathname();

  return (
    <>
      {/* Боковое меню */}
      <aside
        className={`h-screen text-white flex flex-col transition-all duration-300 fixed left-0 top-0 z-50`}
        style={{
          backgroundColor: "#1F1E1C",
          width: isCollapsed ? "5rem" : "16rem",
        }}
      >
        {/* Логотип */}
        <div
          className="p-4 border-b flex items-center gap-3 relative h-16"
          style={{ borderColor: "#2E2C29" }}
        >
          <Image src="/logo.png" alt="logo" width={48} height={48} />

          <span
            className={`text-xl font-bold text-white transform transition-all duration-300 absolute left-[72px] top-1/2 -translate-y-1/2 ${isCollapsed
                ? "opacity-0 -translate-x-2"
                : "opacity-100 translate-x-0 delay-200"
              }`}
          >
            Defect Detection
          </span>
        </div>
        
        {/* Навигация */}
        <nav className="flex-1 p-3 space-y-3">
          {navItems.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center rounded-xl transition-all duration-300 ${isActive ? "bg-[#35332F]" : "hover:bg-[#2A2926]"
                  } ${isCollapsed ? "justify-center p-4" : "px-6 py-4"}`}
              >
                <span className="text-2xl">{item.icon}</span>

                {!isCollapsed && (
                  <span
                    className="ml-3 text-lg font-medium transition-all duration-300 whitespace-nowrap"
                  >
                    {item.label}
                  </span>
                )}
              </Link>

            );
          })}
        </nav>

        {/* Кнопка сворачивания */}
        <Button
          variant="outlined"
          className="flex items-center justify-center h-14 p-0 mx-2 mb-3 rounded-xl text-white"
          style={{
            borderColor: "#3C3A37",
            backgroundColor: "#2A2926",
          }}
          onClick={() => setIsCollapsed(!isCollapsed)}
        >
          {isCollapsed ? (
            <ArrowForwardIosOutlinedIcon />
          ) : (
            <ArrowBackIosOutlinedIcon />
          )}
        </Button>
      </aside>

      {/* Отступ для основного контента */}
      <div
        className="transition-all duration-300"
        style={{
          marginLeft: isCollapsed ? "5rem" : "16rem",
        }}
      >
        {/* Здесь будет ваш основной контент */}
      </div>
    </>
  );
}