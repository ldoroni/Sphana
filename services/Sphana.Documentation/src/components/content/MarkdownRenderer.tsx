import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { CodeBlock } from './CodeBlock';
import { AlertCircle, CheckCircle, AlertTriangle, Info } from 'lucide-react';

interface MarkdownRendererProps {
  content: string;
}

export function MarkdownRenderer({ content }: MarkdownRendererProps) {
  return (
    <div className="prose prose-lg dark:prose-invert max-w-none">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          code: CodeBlock,
          h1: ({ children }) => (
            <h1 className="text-4xl font-bold mb-6 mt-8 text-gray-900 dark:text-gray-100 border-b-2 border-primary-500 pb-4">
              {children}
            </h1>
          ),
          h2: ({ children }) => {
            const id = String(children).toLowerCase().replace(/[^a-z0-9\s-]/g, '').replace(/\s+/g, '-');
            return (
              <h2 id={id} className="text-3xl font-semibold mb-4 mt-12 text-gray-900 dark:text-gray-100 scroll-mt-20">
                {children}
              </h2>
            );
          },
          h3: ({ children }) => {
            const id = String(children).toLowerCase().replace(/[^a-z0-9\s-]/g, '').replace(/\s+/g, '-');
            return (
              <h3 id={id} className="text-2xl font-semibold mb-3 mt-8 text-gray-900 dark:text-gray-100 scroll-mt-20">
                {children}
              </h3>
            );
          },
          h4: ({ children }) => {
            const id = String(children).toLowerCase().replace(/[^a-z0-9\s-]/g, '').replace(/\s+/g, '-');
            return (
              <h4 id={id} className="text-xl font-semibold mb-2 mt-6 text-gray-900 dark:text-gray-100 scroll-mt-20">
                {children}
              </h4>
            );
          },
          a: ({ href, children }) => (
            <a
              href={href}
              className="text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300 underline decoration-primary-500/30 hover:decoration-primary-500 transition-colors"
              target={href?.startsWith('http') ? '_blank' : undefined}
              rel={href?.startsWith('http') ? 'noopener noreferrer' : undefined}
            >
              {children}
            </a>
          ),
          blockquote: ({ children }) => {
            const content = String(children);
            let icon = <Info size={20} />;
            let colorClasses = 'border-blue-500 bg-blue-50 dark:bg-blue-950/30 text-blue-900 dark:text-blue-100';

            if (content.includes('**Note:**') || content.includes('**note:**')) {
              icon = <Info size={20} />;
              colorClasses = 'border-blue-500 bg-blue-50 dark:bg-blue-950/30 text-blue-900 dark:text-blue-100';
            } else if (content.includes('**Warning:**') || content.includes('**warning:**')) {
              icon = <AlertTriangle size={20} />;
              colorClasses = 'border-yellow-500 bg-yellow-50 dark:bg-yellow-950/30 text-yellow-900 dark:text-yellow-100';
            } else if (content.includes('**Success:**') || content.includes('**success:**')) {
              icon = <CheckCircle size={20} />;
              colorClasses = 'border-green-500 bg-green-50 dark:bg-green-950/30 text-green-900 dark:text-green-100';
            } else if (content.includes('**Error:**') || content.includes('**error:**')) {
              icon = <AlertCircle size={20} />;
              colorClasses = 'border-red-500 bg-red-50 dark:bg-red-950/30 text-red-900 dark:text-red-100';
            }

            return (
              <blockquote className={`border-l-4 pl-4 pr-4 py-3 my-6 rounded-r-lg flex gap-3 ${colorClasses}`}>
                <div className="flex-shrink-0 mt-1">{icon}</div>
                <div className="flex-1">{children}</div>
              </blockquote>
            );
          },
          table: ({ children }) => (
            <div className="overflow-x-auto my-6">
              <table className="min-w-full divide-y divide-gray-300 dark:divide-gray-700">{children}</table>
            </div>
          ),
          thead: ({ children }) => (
            <thead className="bg-gray-50 dark:bg-gray-800">{children}</thead>
          ),
          th: ({ children }) => (
            <th className="px-4 py-3 text-left text-xs font-semibold text-gray-700 dark:text-gray-300 uppercase tracking-wider">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="px-4 py-3 text-sm text-gray-900 dark:text-gray-100 border-b border-gray-200 dark:border-gray-700">
              {children}
            </td>
          ),
          ul: ({ children }) => (
            <ul className="list-disc list-outside ml-6 my-4 space-y-2 text-gray-700 dark:text-gray-300">
              {children}
            </ul>
          ),
          ol: ({ children }) => (
            <ol className="list-decimal list-outside ml-6 my-4 space-y-2 text-gray-700 dark:text-gray-300">
              {children}
            </ol>
          ),
          li: ({ children }) => (
            <li className="text-gray-700 dark:text-gray-300 leading-relaxed">{children}</li>
          ),
          p: ({ children }) => (
            <p className="text-gray-700 dark:text-gray-300 leading-relaxed my-4">{children}</p>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

