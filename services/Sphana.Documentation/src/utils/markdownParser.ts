import type { Heading } from '../types/content';

export interface MarkdownContent {
  content: string;
  frontMatter: Record<string, any>;
}

export function parseMarkdown(rawContent: string): MarkdownContent {
  // Simple frontmatter parser without Buffer dependency
  let content = rawContent;
  let frontMatter: Record<string, any> = {};
  
  // Check for frontmatter (--- at start)
  if (rawContent.startsWith('---')) {
    const endIndex = rawContent.indexOf('---', 3);
    if (endIndex !== -1) {
      const frontmatterText = rawContent.substring(3, endIndex).trim();
      content = rawContent.substring(endIndex + 3).trim();
      
      // Parse simple YAML frontmatter
      frontMatter = parseFrontmatter(frontmatterText);
    }
  }
  
  return {
    content,
    frontMatter,
  };
}

function parseFrontmatter(text: string): Record<string, any> {
  const frontmatter: Record<string, any> = {};
  const lines = text.split('\n');
  
  for (const line of lines) {
    const colonIndex = line.indexOf(':');
    if (colonIndex !== -1) {
      const key = line.substring(0, colonIndex).trim();
      const value = line.substring(colonIndex + 1).trim();
      
      // Store the value (remove quotes if present)
      frontmatter[key] = value.replace(/^["']|["']$/g, '');
    }
  }
  
  return frontmatter;
}

export function extractHeadings(markdown: string): Heading[] {
  const headingRegex = /^(#{1,6})\s+(.+)$/gm;
  const headings: Heading[] = [];
  let match;

  while ((match = headingRegex.exec(markdown)) !== null) {
    const level = match[1].length;
    const text = match[2].trim();
    const id = text
      .toLowerCase()
      .replace(/[^a-z0-9\s-]/g, '')
      .replace(/\s+/g, '-');

    headings.push({ level, text, id });
  }

  return headings;
}

export function generateTableOfContents(headings: Heading[]): Heading[] {
  // Filter to only show h2 and h3 headings in TOC
  return headings.filter(h => h.level >= 2 && h.level <= 3);
}
