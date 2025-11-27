import React from 'react';
import { useParams } from 'react-router-dom';
import { useMarkdownLoader } from '../hooks/useMarkdownLoader';
import { useNavigation } from '../hooks/useNavigation';
import { MarkdownRenderer } from '../components/content/MarkdownRenderer';
import { Breadcrumbs } from '../components/content/Breadcrumbs';
import { TableOfContents } from '../components/content/TableOfContents';
import { AlertCircle } from 'lucide-react';

export function DocPage() {
  const { pageId } = useParams<{ pageId: string }>();
  const { navigation } = useNavigation();
  const page = navigation?.pages.find((p) => p.id === pageId);
  const { content, loading, error } = useMarkdownLoader(page?.file || null);

  if (!page) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center">
        <AlertCircle size={48} className="text-gray-400 mb-4" />
        <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-2">Page Not Found</h2>
        <p className="text-gray-600 dark:text-gray-400">
          The page you're looking for doesn't exist.
        </p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  if (error || !content) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center">
        <AlertCircle size={48} className="text-red-500 mb-4" />
        <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-2">Error Loading Page</h2>
        <p className="text-gray-600 dark:text-gray-400">
          {error?.message || 'Failed to load content'}
        </p>
      </div>
    );
  }

  return (
    <div className="flex gap-12">
      <div className="flex-1 min-w-0">
        {navigation && <Breadcrumbs pages={navigation.pages} />}
        <article>
          <MarkdownRenderer content={content.content} />
        </article>
      </div>
      <TableOfContents content={content.content} />
    </div>
  );
}

