import React, { useEffect, useState } from 'react';
import { extractHeadings, generateTableOfContents } from '../../utils/markdownParser';
import type { Heading } from '../../types/content';

interface TableOfContentsProps {
  content: string;
}

export function TableOfContents({ content }: TableOfContentsProps) {
  const [headings, setHeadings] = useState<Heading[]>([]);
  const [activeId, setActiveId] = useState<string>('');

  useEffect(() => {
    const allHeadings = extractHeadings(content);
    const toc = generateTableOfContents(allHeadings);
    setHeadings(toc);
  }, [content]);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setActiveId(entry.target.id);
          }
        });
      },
      { rootMargin: '-100px 0px -66% 0px' }
    );

    headings.forEach((heading) => {
      const element = document.getElementById(heading.id);
      if (element) observer.observe(element);
    });

    return () => observer.disconnect();
  }, [headings]);

  if (headings.length === 0) return null;

  return (
    <nav className="hidden xl:block w-64 flex-shrink-0">
      <div className="sticky top-20 py-8 pr-8">
        <h4 className="text-sm font-semibold text-gray-900 dark:text-gray-100 uppercase tracking-wide mb-4">
          On This Page
        </h4>
        <ul className="space-y-2 text-sm">
          {headings.map((heading) => (
            <li key={heading.id} className={heading.level === 3 ? 'ml-4' : ''}>
              <a
                href={`#${heading.id}`}
                className={`block py-1 transition-colors hover:text-primary-600 dark:hover:text-primary-400 ${
                  activeId === heading.id
                    ? 'text-primary-600 dark:text-primary-400 font-medium'
                    : 'text-gray-600 dark:text-gray-400'
                }`}
                onClick={(e) => {
                  e.preventDefault();
                  document.getElementById(heading.id)?.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start',
                  });
                }}
              >
                {heading.text}
              </a>
            </li>
          ))}
        </ul>
      </div>
    </nav>
  );
}

