import React from 'react';
import { useNavigate } from 'react-router-dom';
import { FileText, ChevronRight } from 'lucide-react';
import type { SearchItem } from '../../utils/searchIndex';

interface SearchResultsProps {
  results: Fuse.FuseResult<SearchItem>[];
  selectedIndex: number;
  onClose: () => void;
}

export function SearchResults({ results, selectedIndex, onClose }: SearchResultsProps) {
  const navigate = useNavigate();

  const handleSelect = (route: string) => {
    navigate(route);
    onClose();
  };

  if (results.length === 0) {
    return (
      <div className="px-4 py-12 text-center text-gray-500 dark:text-gray-400">
        <p>No results found.</p>
        <p className="text-sm mt-2">Try searching for something else.</p>
      </div>
    );
  }

  return (
    <div className="py-2">
      {results.map((result, index) => (
        <button
          key={result.item.id}
          onClick={() => handleSelect(result.item.route)}
          className={`w-full flex items-start gap-3 px-4 py-3 text-left transition-colors ${
            index === selectedIndex
              ? 'bg-primary-50 dark:bg-primary-900/20'
              : 'hover:bg-gray-50 dark:hover:bg-gray-800'
          }`}
        >
          <FileText size={18} className="flex-shrink-0 mt-0.5 text-gray-400" />
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs text-gray-500 dark:text-gray-400">{result.item.category}</span>
              <ChevronRight size={12} className="text-gray-400" />
              <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
                {result.item.title}
              </span>
            </div>
            {result.matches && result.matches[0] && (
              <p className="text-sm text-gray-600 dark:text-gray-400 line-clamp-2">
                {getMatchPreview(result.matches[0], result.item)}
              </p>
            )}
          </div>
        </button>
      ))}
    </div>
  );
}

function getMatchPreview(match: Fuse.FuseResultMatch, item: SearchItem): string {
  if (match.key === 'title') {
    return item.content.substring(0, 150) + '...';
  }
  
  if (match.key === 'content' && match.indices && match.indices.length > 0) {
    const [start] = match.indices[0];
    const contextStart = Math.max(0, start - 50);
    const contextEnd = Math.min(item.content.length, start + 100);
    let preview = item.content.substring(contextStart, contextEnd);
    
    if (contextStart > 0) preview = '...' + preview;
    if (contextEnd < item.content.length) preview = preview + '...';
    
    return preview;
  }
  
  return item.content.substring(0, 150) + '...';
}

