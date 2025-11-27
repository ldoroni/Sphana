import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Moon, Sun, Github } from 'lucide-react';
import { SearchBar } from '../search/SearchBar';
import { useTheme } from '../../hooks/useTheme';

interface HeaderProps {
  onOpenSearch: () => void;
}

export function Header({ onOpenSearch }: HeaderProps) {
  const { theme, toggleTheme } = useTheme();

  return (
    <header className="sticky top-0 z-40 w-full border-b border-gray-200 dark:border-gray-800 bg-white/80 dark:bg-gray-900/80 backdrop-blur-xl">
      <div className="container mx-auto px-6 h-16 flex items-center justify-between gap-4">
        {/* Logo */}
        <Link to="/" className="flex items-center gap-3 hover:opacity-80 transition-opacity">
          <div className="w-8 h-8 bg-gradient-to-br from-primary-500 to-accent-500 rounded-lg flex items-center justify-center text-white font-bold text-sm">
            S
          </div>
          <div className="hidden sm:block">
            <div className="text-lg font-bold text-gray-900 dark:text-gray-100">Sphana</div>
            <div className="text-xs text-gray-500 dark:text-gray-400">Neural RAG Database</div>
          </div>
        </Link>

        {/* Search Bar */}
        <div className="flex-1 max-w-md">
          <SearchBar onOpenSearch={onOpenSearch} />
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2">
          <button
            onClick={toggleTheme}
            className="p-2 rounded-lg text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
            aria-label="Toggle theme"
          >
            {theme === 'light' ? <Moon size={20} /> : <Sun size={20} />}
          </button>
          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="p-2 rounded-lg text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
            aria-label="GitHub"
          >
            <Github size={20} />
          </a>
        </div>
      </div>
    </header>
  );
}

