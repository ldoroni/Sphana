import { useState, useEffect } from 'react';
import { parseMarkdown } from '../utils/markdownParser';
import type { MarkdownContent } from '../types/content';

export function useMarkdownLoader(filepath: string | null) {
  const [content, setContent] = useState<MarkdownContent | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    if (!filepath) {
      setContent(null);
      return;
    }

    setLoading(true);
    setError(null);

    fetch(`/content/${filepath}`)
      .then(res => {
        if (!res.ok) throw new Error(`Failed to load ${filepath}`);
        return res.text();
      })
      .then(rawContent => {
        const parsed = parseMarkdown(rawContent);
        setContent(parsed);
        setLoading(false);
      })
      .catch(err => {
        setError(err);
        setLoading(false);
      });
  }, [filepath]);

  return { content, loading, error };
}

