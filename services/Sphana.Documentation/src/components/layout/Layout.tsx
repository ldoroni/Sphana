import React, { useState } from 'react';
import { Outlet } from 'react-router-dom';
import { Header } from './Header';
import { Sidebar } from './Sidebar';
import { Footer } from './Footer';
import { SearchModal } from '../search/SearchModal';
import { useNavigation } from '../../hooks/useNavigation';

export function Layout() {
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const { navigation, loading, error } = useNavigation();

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto mb-4"></div>
          <p className="text-gray-600 dark:text-gray-400">Loading documentation...</p>
        </div>
      </div>
    );
  }

  if (error || !navigation) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center text-red-600 dark:text-red-400">
          <p className="text-xl font-semibold mb-2">Failed to load documentation</p>
          <p className="text-sm">{error?.message || 'Unknown error'}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950">
      <Header onOpenSearch={() => setIsSearchOpen(true)} />
      
      <div className="flex">
        <Sidebar navigation={navigation} />
        
        <main className="flex-1 min-w-0">
          <div className="container mx-auto px-6 lg:px-12 py-8 max-w-5xl">
            <Outlet />
          </div>
          <Footer />
        </main>
      </div>

      <SearchModal
        isOpen={isSearchOpen}
        onClose={() => setIsSearchOpen(false)}
        pages={navigation.pages}
      />
    </div>
  );
}

