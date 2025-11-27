import Fuse from 'fuse.js';
import type { NavigationPage } from '../types/navigation';

export interface SearchItem {
  id: string;
  title: string;
  content: string;
  headings: string[];
  route: string;
  category: string;
}

export function buildSearchIndex(
  pages: NavigationPage[],
  contentMap: Map<string, string>
): Fuse<SearchItem> {
  const searchItems: SearchItem[] = pages.map(page => {
    const content = contentMap.get(page.id) || '';
    const headings = extractHeadingsFromContent(content);
    
    return {
      id: page.id,
      title: page.title,
      content: content.substring(0, 5000), // Limit content length for performance
      headings,
      route: page.route,
      category: page.category,
    };
  });

  return new Fuse(searchItems, {
    keys: [
      { name: 'title', weight: 3 },
      { name: 'headings', weight: 2 },
      { name: 'content', weight: 1 },
      { name: 'category', weight: 1.5 },
    ],
    threshold: 0.3,
    includeMatches: true,
    minMatchCharLength: 2,
  });
}

function extractHeadingsFromContent(content: string): string[] {
  const headingRegex = /^#{1,6}\s+(.+)$/gm;
  const headings: string[] = [];
  let match;

  while ((match = headingRegex.exec(content)) !== null) {
    headings.push(match[1].trim());
  }

  return headings;
}

