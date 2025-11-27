export interface FrontMatter {
  title?: string;
  description?: string;
  [key: string]: any;
}

export interface MarkdownContent {
  content: string;
  frontMatter: FrontMatter;
}

export interface Heading {
  level: number;
  text: string;
  id: string;
}

