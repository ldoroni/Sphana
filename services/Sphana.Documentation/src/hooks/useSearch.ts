import { useState, useEffect, useCallback } from 'react';
import Fuse from 'fuse.js';
import { buildSearchIndex, type SearchItem } from '../utils/searchIndex';
import type { NavigationPage } from '../types/navigation';

export function useSearch(pages: NavigationPage[]) {
  const [searchIndex, setSearchIndex] = useState<Fuse<SearchItem> | null>(null);
  const [isIndexing, setIsIndexing] = useState(true);

  useEffect(() => {
    if (pages.length === 0) return;

    // Load all markdown content and build search index
    const contentMap = new Map<string, string>();
    const promises = pages.map(page =>
      fetch(`/content/${page.file}`)
        .then(res => res.text())
        .then(content => {
          contentMap.set(page.id, content);
        })
        .catch(err => {
          console.error(`Failed to load ${page.file} for search indexing:`, err);
        })
    );

    Promise.all(promises).then(() => {
      const index = buildSearchIndex(pages, contentMap);
      setSearchIndex(index);
      setIsIndexing(false);
    });
  }, [pages]);

  const search = useCallback(
    (query: string) => {
      if (!searchIndex || !query.trim()) return [];
      return searchIndex.search(query);
    },
    [searchIndex]
  );

  return { search, isIndexing };
}

