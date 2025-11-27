export interface NavigationPage {
  id: string;
  title: string;
  route: string;
  file: string;
  icon: string;
  category: string;
  description?: string;
}

export interface Navigation {
  pages: NavigationPage[];
  categories: string[];
}

