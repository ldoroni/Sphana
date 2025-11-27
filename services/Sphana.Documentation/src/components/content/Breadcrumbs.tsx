import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { ChevronRight, Home } from 'lucide-react';
import type { NavigationPage } from '../../types/navigation';

interface BreadcrumbsProps {
  pages: NavigationPage[];
}

export function Breadcrumbs({ pages }: BreadcrumbsProps) {
  const location = useLocation();
  const currentPage = pages.find(p => p.route === location.pathname);

  if (!currentPage) return null;

  return (
    <nav className="flex items-center space-x-2 text-sm text-gray-600 dark:text-gray-400 mb-8">
      <Link
        to="/"
        className="flex items-center hover:text-primary-600 dark:hover:text-primary-400 transition-colors"
      >
        <Home size={16} />
      </Link>
      <ChevronRight size={16} className="text-gray-400" />
      <span className="text-gray-500 dark:text-gray-500">{currentPage.category}</span>
      <ChevronRight size={16} className="text-gray-400" />
      <span className="text-gray-900 dark:text-gray-100 font-medium">{currentPage.title}</span>
    </nav>
  );
}

