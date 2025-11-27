import React, { useState, useEffect, useRef } from 'react';
import { Search } from 'lucide-react';

interface SearchBarProps {
  onOpenSearch: () => void;
}

export function SearchBar({ onOpenSearch }: SearchBarProps) {
  const [isMac, setIsMac] = useState(false);

  useEffect(() => {
    setIsMac(navigator.platform.toUpperCase().indexOf('MAC') >= 0);
  }, []);

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === 'k' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        onOpenSearch();
      }
    };

    document.addEventListener('keydown', down);
    return () => document.removeEventListener('keydown', down);
  }, [onOpenSearch]);

  return (
    <button
      onClick={onOpenSearch}
      className="flex items-center gap-3 w-full max-w-md px-4 py-2 text-sm text-gray-600 dark:text-gray-400 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-lg hover:border-primary-500 dark:hover:border-primary-500 transition-colors"
    >
      <Search size={18} />
      <span className="flex-1 text-left">Search documentation...</span>
      <kbd className="hidden sm:inline-block px-2 py-0.5 text-xs font-mono bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded">
        {isMac ? '⌘K' : 'Ctrl+K'}
      </kbd>
    </button>
  );
}

