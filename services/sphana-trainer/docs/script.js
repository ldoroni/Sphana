// Sphana Trainer Documentation - Enhanced Interactive Features

(function() {
    'use strict';

    // ====================
    // Multi-Page Navigation
    // ====================
    const PageNavigator = {
        currentPage: null,
        pages: {},
        scrollSpyListener: null,
        pageOrder: [
            'overview',
            'installation',
            'quick-start',
            'architecture',
            'components',
            'workflow',
            'data-preparation',
            'training',
            'export-package',
            'workflows',
            'cli-reference',
            'configuration',
            'api',
            'distributed',
            'mlflow',
            'optimization'
        ],

        init() {
            this.indexPages();
            this.addNavigationButtons();
            this.setupNavigationHandlers();
            this.loadInitialPage();
            window.addEventListener('popstate', () => this.loadPageFromHash());
        },

        indexPages() {
            document.querySelectorAll('.section').forEach(section => {
                const id = section.id;
                if (id) {
                    this.pages[id] = {
                        element: section,
                        title: section.querySelector('h2')?.textContent || id,
                        navLinks: document.querySelectorAll(`.nav-link[href="#${id}"]`)
                    };
                }
            });
        },

        addNavigationButtons() {
            this.pageOrder.forEach((pageId, index) => {
                const page = this.pages[pageId];
                if (!page) return;

                const navContainer = document.createElement('div');
                navContainer.className = 'page-navigation';

                const prevPage = index > 0 ? this.pages[this.pageOrder[index - 1]] : null;
                const nextPage = index < this.pageOrder.length - 1 ? this.pages[this.pageOrder[index + 1]] : null;

                if (prevPage) {
                    const prevBtn = this.createNavButton(this.pageOrder[index - 1], prevPage.title, 'prev');
                    navContainer.appendChild(prevBtn);
                }

                if (nextPage) {
                    const nextBtn = this.createNavButton(this.pageOrder[index + 1], nextPage.title, 'next');
                    navContainer.appendChild(nextBtn);
                }

                if (prevPage || nextPage) {
                    page.element.appendChild(navContainer);
                }
            });
        },

        createNavButton(pageId, title, direction) {
            const button = document.createElement('button');
            button.className = `page-nav-btn ${direction}`;
            button.setAttribute('data-page', pageId);

            const leftArrow = `<svg viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M12.707 5.293a1 1 0 010 1.414L9.414 10l3.293 3.293a1 1 0 01-1.414 1.414l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 0z" clip-rule="evenodd"/></svg>`;
            const rightArrow = `<svg viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clip-rule="evenodd"/></svg>`;

            if (direction === 'prev') {
                button.innerHTML = `
                    ${leftArrow}
                    <div style="text-align: left;">
                        <div class="page-nav-label">Previous</div>
                        <div class="page-nav-title">${title}</div>
                    </div>
                `;
            } else {
                button.innerHTML = `
                    <div style="text-align: right;">
                        <div class="page-nav-label">Next</div>
                        <div class="page-nav-title">${title}</div>
                    </div>
                    ${rightArrow}
                `;
            }

            button.addEventListener('click', () => {
                this.navigateToPage(pageId);
            });

            return button;
        },

        setupNavigationHandlers() {
            document.querySelectorAll('.nav-link').forEach(link => {
                link.addEventListener('click', (e) => {
                    e.preventDefault();
                    const hash = link.getAttribute('href').substring(1);
                    this.navigateToPage(hash);
                });

                // Add double-click to toggle subnav
                link.addEventListener('dblclick', (e) => {
                    e.preventDefault();
                    const subnav = link.nextElementSibling;
                    if (subnav && subnav.classList.contains('subnav')) {
                        link.classList.toggle('expanded');
                        subnav.classList.toggle('expanded');
                    }
                });
            });
        },

        navigateToPage(pageId) {
            if (!this.pages[pageId]) {
                console.warn('Page not found:', pageId);
                return;
            }

            // Hide all sections
            Object.values(this.pages).forEach(page => {
                page.element.classList.remove('active');
                page.navLinks.forEach(link => link.classList.remove('active'));
            });

            // Show current section
            const page = this.pages[pageId];
            page.element.classList.add('active');
            page.navLinks.forEach(link => link.classList.add('active'));

            // Update URL hash
            history.pushState(null, '', `#${pageId}`);

            // Update document title
            document.title = `${page.title} - Sphana Trainer Documentation`;

            // Update nested navigation
            this.updateNestedNav(pageId);

            // Scroll to top
            window.scrollTo({ top: 0, behavior: 'smooth' });

            this.currentPage = pageId;
        },

        updateNestedNav(pageId) {
            // First, collapse all other nav items
            document.querySelectorAll('.nav-link.expanded').forEach(link => {
                link.classList.remove('expanded');
            });
            document.querySelectorAll('.subnav.expanded').forEach(subnav => {
                subnav.classList.remove('expanded');
            });

            // Find the nav link for this page
            const navLink = document.querySelector(`.nav-link[href="#${pageId}"]`);
            if (!navLink) return;

            // Get the section element
            const section = this.pages[pageId]?.element;
            if (!section) return;

            // Check if subnav already exists
            let subnav = navLink.nextElementSibling;
            if (!subnav || !subnav.classList.contains('subnav')) {
                // Create subnav if it doesn't exist
                subnav = document.createElement('div');
                subnav.className = 'subnav';
                navLink.parentNode.insertBefore(subnav, navLink.nextSibling);
            }

            // Clear existing subnav
            subnav.innerHTML = '';

            // Get h3 headings within this section
            const headings = section.querySelectorAll('h3');

            if (headings.length === 0) {
                // Remove subnav indicator if no headings
                navLink.classList.remove('has-subnav');
                return;
            }

            // Add subnav indicator
            navLink.classList.add('has-subnav');

            // Build subnav links
            headings.forEach((heading, index) => {
                const text = heading.textContent;

                // Create unique ID if not exists
                if (!heading.id) {
                    heading.id = `${pageId}-heading-${index}`;
                }

                const link = document.createElement('a');
                link.href = `#${heading.id}`;
                link.textContent = text;
                link.className = 'subnav-link';

                // Smooth scroll on click with proper offset
                link.addEventListener('click', (e) => {
                    e.preventDefault();
                    const headerHeight = 72; // var(--header-height)
                    const offset = 20; // extra space
                    const elementPosition = heading.getBoundingClientRect().top + window.pageYOffset;
                    const offsetPosition = elementPosition - headerHeight - offset;
                    
                    window.scrollTo({
                        top: offsetPosition,
                        behavior: 'smooth'
                    });
                });

                subnav.appendChild(link);
            });

            // Auto-expand the subnav for the active page
            navLink.classList.add('expanded');
            subnav.classList.add('expanded');
            
            // Initialize scroll spy for this page
            this.initScrollSpy(pageId);
        },

        initScrollSpy(pageId) {
            // Remove previous scroll listener if any
            if (this.scrollSpyListener) {
                window.removeEventListener('scroll', this.scrollSpyListener);
            }

            // Create new scroll spy listener
            this.scrollSpyListener = () => {
                const section = this.pages[pageId]?.element;
                if (!section) return;

                const headings = section.querySelectorAll('h3');
                const scrollPosition = window.scrollY + 100; // offset for header

                // Find which heading is currently visible
                let currentHeading = null;
                headings.forEach(heading => {
                    if (heading.offsetTop <= scrollPosition) {
                        currentHeading = heading;
                    }
                });

                // Update active state for subnav links
                const subnavLinks = document.querySelectorAll('.subnav-link');
                subnavLinks.forEach(link => {
                    link.classList.remove('active');
                    if (currentHeading && link.href.includes(`#${currentHeading.id}`)) {
                        link.classList.add('active');
                    }
                });
            };

            // Add scroll listener
            window.addEventListener('scroll', this.scrollSpyListener, { passive: true });
            
            // Run once immediately
            this.scrollSpyListener();
        },

        loadInitialPage() {
            const hash = window.location.hash.substring(1);
            const defaultPage = 'overview';
            this.navigateToPage(hash || defaultPage);
        },

        loadPageFromHash() {
            const hash = window.location.hash.substring(1);
            if (hash && this.pages[hash]) {
                this.navigateToPage(hash);
            }
        }
    };

    // ====================
    // Autocomplete Search
    // ====================
    const AutocompleteSearch = {
        searchInput: null,
        resultsContainer: null,
        searchIndex: [],
        selectedIndex: -1,
        debounceTimeout: null,

        init() {
            this.searchInput = document.getElementById('search');
            this.resultsContainer = document.getElementById('searchResults');

            if (!this.searchInput || !this.resultsContainer) return;

            this.buildSearchIndex();
            this.setupEventListeners();
        },

        buildSearchIndex() {
            // Index all sections, headings, and commands
            document.querySelectorAll('.section').forEach(section => {
                const sectionId = section.id;
                const sectionTitle = section.querySelector('h2')?.textContent || '';
                const category = this.getSectionCategory(sectionId);

                // Index section itself
                this.searchIndex.push({
                    type: 'section',
                    id: sectionId,
                    title: sectionTitle,
                    category: category,
                    path: sectionTitle,
                    keywords: [sectionTitle.toLowerCase()]
                });

                // Index h3 headings
                section.querySelectorAll('h3').forEach(h3 => {
                    const title = h3.textContent;
                    this.searchIndex.push({
                        type: 'heading',
                        id: sectionId,
                        title: title,
                        category: category,
                        path: `${sectionTitle} > ${title}`,
                        keywords: [title.toLowerCase(), sectionTitle.toLowerCase()]
                    });
                });

                // Index commands
                section.querySelectorAll('.command-ref h4 code, .code-label').forEach(code => {
                    const title = code.textContent;
                    if (title.includes('python') || title.includes('sphana')) {
                        this.searchIndex.push({
                            type: 'command',
                            id: sectionId,
                            title: title,
                            category: 'Command',
                            path: `${sectionTitle} > Commands`,
                            keywords: [title.toLowerCase(), 'cli', 'command']
                        });
                    }
                });
            });
        },

        getSectionCategory(sectionId) {
            const categories = {
                'overview': 'Getting Started',
                'installation': 'Getting Started',
                'quick-start': 'Getting Started',
                'architecture': 'Core Concepts',
                'components': 'Core Concepts',
                'workflow': 'Core Concepts',
                'data-preparation': 'User Guide',
                'training': 'User Guide',
                'export-package': 'User Guide',
                'workflows': 'User Guide',
                'cli-reference': 'Reference',
                'configuration': 'Reference',
                'api': 'Reference',
                'distributed': 'Advanced',
                'mlflow': 'Advanced',
                'optimization': 'Advanced'
            };
            return categories[sectionId] || 'Documentation';
        },

        setupEventListeners() {
            this.searchInput.addEventListener('input', (e) => {
                clearTimeout(this.debounceTimeout);
                this.debounceTimeout = setTimeout(() => {
                    this.performSearch(e.target.value);
                }, 200);
            });

            this.searchInput.addEventListener('keydown', (e) => {
                this.handleKeyNavigation(e);
            });

            this.searchInput.addEventListener('focus', () => {
                if (this.searchInput.value) {
                    this.performSearch(this.searchInput.value);
                }
            });

            // Close results when clicking outside
            document.addEventListener('click', (e) => {
                if (!this.searchInput.contains(e.target) && !this.resultsContainer.contains(e.target)) {
                    this.hideResults();
                }
            });

            // Close on Escape
            this.searchInput.addEventListener('keydown', (e) => {
                if (e.key === 'Escape') {
                    this.hideResults();
                    this.searchInput.blur();
                }
            });
        },

        performSearch(query) {
            if (!query || query.length < 2) {
                this.hideResults();
                return;
            }

            const queryLower = query.toLowerCase();
            const results = this.searchIndex
                .filter(item => {
                    return item.keywords.some(keyword => keyword.includes(queryLower)) ||
                           item.title.toLowerCase().includes(queryLower);
                })
                .slice(0, 10); // Limit to 10 results

            this.displayResults(results, query);
        },

        displayResults(results, query) {
            if (results.length === 0) {
                this.resultsContainer.innerHTML = `
                    <div class="search-no-results">
                        <svg style="width: 48px; height: 48px; margin: 0 auto 1rem; opacity: 0.3; display: block;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="11" cy="11" r="8"/>
                            <path d="M21 21l-4.35-4.35"/>
                        </svg>
                        <div>No results found for "${query}"</div>
                    </div>
                `;
                this.resultsContainer.classList.add('visible');
                return;
            }

            const html = results.map((result, index) => {
                const highlightedTitle = this.highlightMatch(result.title, query);
                const icon = this.getTypeIcon(result.type);
                
                return `
                    <div class="search-result-item ${index === 0 ? 'selected' : ''}" 
                         data-index="${index}" 
                         data-page-id="${result.id}">
                        <div class="search-result-category">${icon} ${result.category}</div>
                        <div class="search-result-title">${highlightedTitle}</div>
                        <div class="search-result-path">${result.path}</div>
                    </div>
                `;
            }).join('');

            this.resultsContainer.innerHTML = html;
            this.resultsContainer.classList.add('visible');
            this.selectedIndex = 0;

            // Add click handlers
            this.resultsContainer.querySelectorAll('.search-result-item').forEach(item => {
                item.addEventListener('click', () => {
                    const pageId = item.getAttribute('data-page-id');
                    this.navigateToResult(pageId);
                });
            });
        },

        highlightMatch(text, query) {
            const regex = new RegExp(`(${this.escapeRegex(query)})`, 'gi');
            return text.replace(regex, '<strong style="color: var(--color-primary); font-weight: 700;">$1</strong>');
        },

        escapeRegex(string) {
            return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        },

        getTypeIcon(type) {
            const icons = {
                'section': '📄',
                'heading': '📌',
                'command': '⚡'
            };
            return icons[type] || '📄';
        },

        handleKeyNavigation(e) {
            const items = this.resultsContainer.querySelectorAll('.search-result-item');
            if (items.length === 0) return;

            if (e.key === 'ArrowDown') {
                e.preventDefault();
                this.selectedIndex = Math.min(this.selectedIndex + 1, items.length - 1);
                this.updateSelection(items);
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                this.selectedIndex = Math.max(this.selectedIndex - 1, 0);
                this.updateSelection(items);
            } else if (e.key === 'Enter') {
                e.preventDefault();
                const selectedItem = items[this.selectedIndex];
                if (selectedItem) {
                    const pageId = selectedItem.getAttribute('data-page-id');
                    this.navigateToResult(pageId);
                }
            }
        },

        updateSelection(items) {
            items.forEach((item, index) => {
                if (index === this.selectedIndex) {
                    item.classList.add('selected');
                    item.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
                } else {
                    item.classList.remove('selected');
                }
            });
        },

        navigateToResult(pageId) {
            this.hideResults();
            this.searchInput.value = '';
            this.searchInput.blur();
            PageNavigator.navigateToPage(pageId);
        },

        hideResults() {
            this.resultsContainer.classList.remove('visible');
            this.selectedIndex = -1;
        }
    };

    // ====================
    // Theme Management
    // ====================
    const ThemeManager = {
        STORAGE_KEY: 'sphana-theme',
        DARK: 'dark',
        LIGHT: 'light',

        init() {
            this.setTheme(this.getPreferredTheme());
            this.createToggle();
            this.watchSystemTheme();
        },

        getPreferredTheme() {
            const stored = localStorage.getItem(this.STORAGE_KEY);
            if (stored) return stored;
            
            return window.matchMedia('(prefers-color-scheme: dark)').matches 
                ? this.DARK 
                : this.LIGHT;
        },

        setTheme(theme) {
            document.documentElement.setAttribute('data-theme', theme);
            localStorage.setItem(this.STORAGE_KEY, theme);
            this.updateToggleIcon(theme);
        },

        toggle() {
            const current = document.documentElement.getAttribute('data-theme');
            const next = current === this.DARK ? this.LIGHT : this.DARK;
            this.setTheme(next);
        },

        createToggle() {
            const button = document.createElement('button');
            button.className = 'theme-toggle';
            button.setAttribute('aria-label', 'Toggle dark mode');
            button.setAttribute('title', 'Toggle theme (T)');
            button.innerHTML = '<svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor"><path d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z"/></svg>';
            
            button.addEventListener('click', () => this.toggle());
            
            const headerRight = document.querySelector('.header-right');
            if (headerRight) {
                headerRight.appendChild(button);
            }
            
            this.toggleButton = button;
            this.updateToggleIcon(this.getPreferredTheme());
        },

        updateToggleIcon(theme) {
            if (this.toggleButton) {
                if (theme === this.DARK) {
                    // Sun icon
                    this.toggleButton.innerHTML = '<svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor"><path d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z"/></svg>';
                } else {
                    // Moon icon
                    this.toggleButton.innerHTML = '<svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor"><path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z"/></svg>';
                }
            }
        },

        watchSystemTheme() {
            window.matchMedia('(prefers-color-scheme: dark)')
                .addEventListener('change', (e) => {
                    if (!localStorage.getItem(this.STORAGE_KEY)) {
                        this.setTheme(e.matches ? this.DARK : this.LIGHT);
                    }
                });
        }
    };

    // ====================
    // Copy to Clipboard (Enhanced)
    // ====================
    function initCopyButtons() {
        document.querySelectorAll('.copy-btn').forEach(button => {
            button.addEventListener('click', async function() {
                const targetId = this.getAttribute('data-target');
                const codeElement = document.getElementById(targetId);
                
                if (!codeElement) {
                    console.error('Code element not found:', targetId);
                    return;
                }

                try {
                    const text = codeElement.textContent.trim();
                    await navigator.clipboard.writeText(text);
                    
                    // Enhanced visual feedback
                    const originalText = this.textContent;
                    const originalBg = this.style.background;
                    
                    this.textContent = '✓ Copied!';
                    this.style.background = 'rgba(16, 185, 129, 0.4)';
                    this.style.borderColor = 'rgba(16, 185, 129, 0.6)';
                    
                    // Ripple effect
                    createRipple(this);
                    
                    setTimeout(() => {
                        this.textContent = originalText;
                        this.style.background = originalBg;
                        this.style.borderColor = '';
                    }, 2000);
                } catch (err) {
                    console.error('Failed to copy:', err);
                    this.textContent = '✗ Failed';
                    setTimeout(() => {
                        this.textContent = 'Copy';
                    }, 2000);
                }
            });
        });
    }

    function createRipple(element) {
        const ripple = document.createElement('span');
        const rect = element.getBoundingClientRect();
        const size = Math.max(rect.width, rect.height);
        
        ripple.style.cssText = `
            position: absolute;
            width: ${size}px;
            height: ${size}px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.5);
            transform: scale(0);
            animation: ripple 0.6s ease-out;
            pointer-events: none;
            left: 50%;
            top: 50%;
            margin-left: -${size/2}px;
            margin-top: -${size/2}px;
        `;
        
        element.style.position = 'relative';
        element.style.overflow = 'hidden';
        element.appendChild(ripple);
        
        setTimeout(() => ripple.remove(), 600);
    }

    // ====================
    // Active Navigation (Enhanced)
    // ====================
    function initActiveNav() {
        const sections = document.querySelectorAll('.section[id]');
        const navLinks = document.querySelectorAll('.nav-link');
        
        if (sections.length === 0 || navLinks.length === 0) return;

        // Create map of section IDs to nav links
        const navMap = new Map();
        navLinks.forEach(link => {
            const href = link.getAttribute('href');
            if (href && href.startsWith('#')) {
                navMap.set(href.substring(1), link);
            }
        });

        let currentActive = null;
        let ticking = false;

        // Function to update active nav link
        function updateActiveNav() {
            if (ticking) return;
            
            ticking = true;
            requestAnimationFrame(() => {
                // Find which section is currently most visible
                let currentSection = null;
                let maxVisibility = 0;
                
                sections.forEach(section => {
                    const rect = section.getBoundingClientRect();
                    const viewportHeight = window.innerHeight;
                    
                    // Calculate how much of the section is visible
                    const visibleTop = Math.max(0, rect.top);
                    const visibleBottom = Math.min(viewportHeight, rect.bottom);
                    const visibleHeight = Math.max(0, visibleBottom - visibleTop);
                    const visibility = visibleHeight / viewportHeight;
                    
                    // Section is most visible if it's in the viewport
                    if (rect.top < viewportHeight * 0.3 && rect.bottom > 0) {
                        if (visibility > maxVisibility) {
                            maxVisibility = visibility;
                            currentSection = section;
                        }
                    }
                });
                
                // Update active state
                if (currentSection) {
                    const id = currentSection.getAttribute('id');
                    const link = navMap.get(id);
                    
                    if (link && link !== currentActive) {
                        // Remove active from all
                        navLinks.forEach(l => l.classList.remove('active'));
                        
                        // Add active to current
                        link.classList.add('active');
                        currentActive = link;
                        
                        // Scroll nav link into view smoothly
                        setTimeout(() => {
                            link.scrollIntoView({ 
                                behavior: 'smooth', 
                                block: 'nearest'
                            });
                        }, 100);
                    }
                }
                
                ticking = false;
            });
        }

        // Update on scroll
        let scrollTimeout;
        window.addEventListener('scroll', () => {
            if (scrollTimeout) {
                clearTimeout(scrollTimeout);
            }
            scrollTimeout = setTimeout(updateActiveNav, 50);
        }, { passive: true });

        // Initial update - run immediately and after a short delay
        updateActiveNav();
        setTimeout(updateActiveNav, 100);
        setTimeout(updateActiveNav, 500);

        // Also update on resize
        window.addEventListener('resize', updateActiveNav, { passive: true });
        
        // Update when clicking nav links
        navLinks.forEach(link => {
            link.addEventListener('click', () => {
                setTimeout(updateActiveNav, 300);
            });
        });
    }

    // ====================
    // Smooth Scroll (Enhanced)
    // ====================
    function initSmoothScroll() {
        document.querySelectorAll('a[href^="#"]').forEach(link => {
            link.addEventListener('click', function(e) {
                const href = this.getAttribute('href');
                if (href === '#') return;

                const target = document.querySelector(href);
                if (target) {
                    e.preventDefault();
                    const headerOffset = 90;
                    const elementPosition = target.getBoundingClientRect().top;
                    const offsetPosition = elementPosition + window.pageYOffset - headerOffset;

                    window.scrollTo({
                        top: offsetPosition,
                        behavior: 'smooth'
                    });
                    
                    // Update URL without scrolling
                    history.pushState(null, null, href);
                }
            });
        });
    }

    // ====================
    // Enhanced Search
    // ====================
    function initSearch() {
        const searchInput = document.getElementById('search');
        if (!searchInput) return;

        // Build search index
        const searchIndex = [];
        document.querySelectorAll('.section').forEach(section => {
            const id = section.getAttribute('id');
            const heading = section.querySelector('h2');
            const text = section.textContent.toLowerCase();
            
            if (id && heading) {
                // Extract keywords
                const keywords = text
                    .split(/\s+/)
                    .filter(word => word.length > 3)
                    .slice(0, 100);
                
                searchIndex.push({
                    id,
                    title: heading.textContent.trim(),
                    text,
                    keywords: new Set(keywords),
                    element: section
                });
            }
        });

        let debounceTimer;
        let lastQuery = '';

        searchInput.addEventListener('input', function(e) {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => {
                const query = e.target.value.trim().toLowerCase();
                
                if (query === lastQuery) return;
                lastQuery = query;
                
                if (query.length === 0) {
                    resetSearch(searchIndex);
                    return;
                }

                performSearch(query, searchIndex);
            }, 200);
        });

        // Keyboard shortcuts
        searchInput.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                this.value = '';
                this.dispatchEvent(new Event('input'));
                this.blur();
            }
        });

        // Add search hint
        addSearchHint(searchInput);
    }

    function performSearch(query, searchIndex) {
        const words = query.split(/\s+/).filter(w => w.length > 0);
        let matchCount = 0;
        let firstMatch = null;

        searchIndex.forEach(item => {
            const matches = words.every(word => item.text.includes(word));
            
            if (matches) {
                item.element.style.display = '';
                item.element.classList.add('search-highlight');
                matchCount++;
                if (!firstMatch) firstMatch = item.element;
            } else {
                item.element.style.display = 'none';
                item.element.classList.remove('search-highlight');
            }
        });

        // Show results count
        showSearchResults(matchCount, query);

        // Scroll to first match
        if (firstMatch && matchCount > 0) {
            setTimeout(() => {
                firstMatch.scrollIntoView({ 
                    behavior: 'smooth', 
                    block: 'start' 
                });
            }, 100);
        }
    }

    function resetSearch(searchIndex) {
        searchIndex.forEach(item => {
            item.element.style.display = '';
            item.element.classList.remove('search-highlight');
        });
        showSearchResults(0, '');
    }

    function showSearchResults(count, query) {
        let resultDiv = document.getElementById('search-results-count');
        
        if (!resultDiv) {
            resultDiv = document.createElement('div');
            resultDiv.id = 'search-results-count';
            resultDiv.style.cssText = `
                position: fixed;
                bottom: 2rem;
                left: 50%;
                transform: translateX(-50%) translateY(10px);
                background: var(--color-surface-elevated);
                border: 2px solid var(--color-primary);
                padding: 1rem 2rem;
                border-radius: 12px;
                box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2),
                           0 0 0 1px rgba(59, 130, 246, 0.1);
                font-size: var(--font-size-sm);
                font-weight: 600;
                color: var(--color-text);
                z-index: 1000;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                opacity: 0;
                pointer-events: none;
                backdrop-filter: blur(10px);
                letter-spacing: 0.02em;
            `;
            document.body.appendChild(resultDiv);
        }

        if (count > 0) {
            resultDiv.innerHTML = `
                <svg style="display: inline-block; vertical-align: middle; margin-right: 0.5rem; width: 20px; height: 20px;" viewBox="0 0 20 20" fill="currentColor">
                    <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/>
                </svg>
                <span>Found <strong style="color: var(--color-primary);">${count}</strong> section${count !== 1 ? 's' : ''} matching "<em style="font-style: italic; color: var(--color-primary);">${query}</em>"</span>
            `;
            resultDiv.style.opacity = '1';
            resultDiv.style.transform = 'translateX(-50%) translateY(0)';
        } else if (query) {
            resultDiv.innerHTML = `
                <svg style="display: inline-block; vertical-align: middle; margin-right: 0.5rem; width: 20px; height: 20px;" viewBox="0 0 20 20" fill="currentColor">
                    <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"/>
                </svg>
                <span>No sections match "<em style="font-style: italic; color: var(--color-text-secondary);">${query}</em>"</span>
            `;
            resultDiv.style.opacity = '1';
            resultDiv.style.transform = 'translateX(-50%) translateY(0)';
        } else {
            resultDiv.style.opacity = '0';
            resultDiv.style.transform = 'translateX(-50%) translateY(10px)';
        }
    }

    function addSearchHint(searchInput) {
        // Only add hint on desktop
        if (window.innerWidth < 768) return;
        
        const wrapper = document.createElement('div');
        wrapper.style.cssText = `
            position: relative;
            flex: 1;
        `;
        
        const hint = document.createElement('kbd');
        hint.textContent = 'Ctrl+K';
        hint.style.cssText = `
            position: absolute;
            right: 0.75rem;
            top: 50%;
            transform: translateY(-50%);
            font-size: 0.7rem;
            color: var(--color-text-muted);
            background: var(--color-surface);
            padding: 0.2rem 0.4rem;
            border: 1px solid var(--color-border);
            border-radius: 0.25rem;
            pointer-events: none;
            font-family: var(--font-mono);
            opacity: 0.7;
            transition: opacity 0.2s;
        `;
        
        // Wrap the input
        const parent = searchInput.parentElement;
        parent.insertBefore(wrapper, searchInput);
        wrapper.appendChild(searchInput);
        wrapper.appendChild(hint);
        
        searchInput.addEventListener('focus', () => {
            hint.style.opacity = '0';
        });
        
        searchInput.addEventListener('blur', () => {
            if (!searchInput.value) {
                hint.style.opacity = '0.7';
            }
        });
    }

    // ====================
    // Keyboard Shortcuts
    // ====================
    function initKeyboardShortcuts() {
        const shortcuts = {
            'k': () => {
                const searchInput = document.getElementById('search');
                if (searchInput) {
                    searchInput.focus();
                    searchInput.select();
                }
            },
            't': () => ThemeManager.toggle(),
            'n': () => {
                // Navigate to next page
                const currentIndex = PageNavigator.pageOrder.indexOf(PageNavigator.currentPage);
                if (currentIndex >= 0 && currentIndex < PageNavigator.pageOrder.length - 1) {
                    PageNavigator.navigateToPage(PageNavigator.pageOrder[currentIndex + 1]);
                }
            },
            'p': () => {
                // Navigate to previous page
                const currentIndex = PageNavigator.pageOrder.indexOf(PageNavigator.currentPage);
                if (currentIndex > 0) {
                    PageNavigator.navigateToPage(PageNavigator.pageOrder[currentIndex - 1]);
                }
            },
            'g': {
                'h': () => PageNavigator.navigateToPage('overview'),
                'g': () => window.scrollTo({ top: 0, behavior: 'smooth' })
            },
            'escape': () => {
                // Close search results
                const searchInput = document.getElementById('search');
                if (searchInput) {
                    searchInput.blur();
                }
                AutocompleteSearch.hideResults();
            }
        };

        let lastKey = null;
        let lastKeyTime = 0;

        document.addEventListener('keydown', function(e) {
            // Ignore if typing in input/textarea
            if (e.target.matches('input, textarea')) return;

            const key = e.key.toLowerCase();
            const isModified = e.ctrlKey || e.metaKey || e.altKey;

            if (isModified && key === 'k') {
                e.preventDefault();
                shortcuts['k']();
                return;
            }

            // Sequential shortcuts (like 'gg')
            const now = Date.now();
            if (now - lastKeyTime < 500 && lastKey && shortcuts[lastKey]) {
                const combo = shortcuts[lastKey];
                if (typeof combo === 'object' && combo[key]) {
                    e.preventDefault();
                    combo[key]();
                    lastKey = null;
                    return;
                }
            }

            // Single shortcuts
            if (shortcuts[key] && typeof shortcuts[key] === 'function') {
                e.preventDefault();
                shortcuts[key]();
            } else if (shortcuts[key] && typeof shortcuts[key] === 'object') {
                lastKey = key;
                lastKeyTime = now;
            }
        });
    }

    function scrollToSection(id) {
        const element = document.querySelector(id);
        if (element) {
            element.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }

    // ====================
    // Back to Top Button (Enhanced)
    // ====================
    function initBackToTop() {
        const button = document.createElement('button');
        button.innerHTML = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 15l-6-6-6 6"/></svg>';
        button.className = 'back-to-top';
        button.setAttribute('aria-label', 'Back to top');
        button.setAttribute('title', 'Back to top (GG)');
        button.style.cssText = `
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            width: 56px;
            height: 56px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--color-primary), var(--color-accent));
            color: white;
            border: none;
            cursor: pointer;
            box-shadow: var(--shadow-xl);
            opacity: 0;
            visibility: hidden;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            z-index: 999;
            display: flex;
            align-items: center;
            justify-content: center;
        `;
        document.body.appendChild(button);

        let scrollTimeout;
        let ticking = false;

        window.addEventListener('scroll', function() {
            if (!ticking) {
                window.requestAnimationFrame(() => {
                    const show = window.pageYOffset > 400;
                    button.style.opacity = show ? '1' : '0';
                    button.style.visibility = show ? 'visible' : 'hidden';
                    button.style.transform = show ? 'scale(1)' : 'scale(0.8)';
                    ticking = false;
                });
                ticking = true;
            }
        });

        button.addEventListener('click', function() {
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        });

        button.addEventListener('mouseenter', function() {
            this.style.transform = 'scale(1.1) translateY(-4px)';
            this.style.boxShadow = '0 25px 50px -12px rgba(0, 0, 0, 0.25)';
        });

        button.addEventListener('mouseleave', function() {
            this.style.transform = 'scale(1)';
            this.style.boxShadow = 'var(--shadow-xl)';
        });
    }

    // ====================
    // Code Block Enhancements
    // ====================
    function enhanceCodeBlocks() {
        document.querySelectorAll('pre code').forEach(block => {
            const lines = block.textContent.split('\n');
            
            // Add line count indicator for large blocks
            if (lines.length > 15) {
                const lineCount = document.createElement('div');
                lineCount.textContent = `${lines.length} lines`;
                lineCount.style.cssText = `
                    position: absolute;
                    top: 0.5rem;
                    right: 4rem;
                    font-size: var(--font-size-xs);
                    color: var(--color-text-muted);
                    opacity: 0.5;
                `;
                
                const codeBlock = block.closest('.code-block');
                if (codeBlock) {
                    codeBlock.style.position = 'relative';
                    codeBlock.appendChild(lineCount);
                }
            }
        });
    }

    // ====================
    // Progressive Content Loading
    // ====================
    function initProgressiveLoading() {
        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('visible');
                        observer.unobserve(entry.target);
                    }
                });
            },
            {
                threshold: 0.1,
                rootMargin: '50px'
            }
        );

        document.querySelectorAll('.feature-card, .component-card, .stage, .command-ref')
            .forEach(el => {
                el.style.opacity = '0';
                el.style.transform = 'translateY(20px)';
                el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
                observer.observe(el);
            });

        // Add visible class handler
        const style = document.createElement('style');
        style.textContent = `
            .visible {
                opacity: 1 !important;
                transform: translateY(0) !important;
            }
        `;
        document.head.appendChild(style);
    }

    // ====================
    // Reading Progress Bar
    // ====================
    function initReadingProgress() {
        const progressBar = document.createElement('div');
        progressBar.style.cssText = `
            position: fixed;
            top: var(--header-height);
            left: 0;
            width: 0%;
            height: 3px;
            background: linear-gradient(90deg, var(--color-primary), var(--color-accent));
            transition: width 0.1s ease;
            z-index: 999;
        `;
        document.body.appendChild(progressBar);

        let ticking = false;
        window.addEventListener('scroll', () => {
            if (!ticking) {
                window.requestAnimationFrame(() => {
                    const winScroll = document.documentElement.scrollTop;
                    const height = document.documentElement.scrollHeight - 
                                 document.documentElement.clientHeight;
                    const scrolled = (winScroll / height) * 100;
                    progressBar.style.width = scrolled + '%';
                    ticking = false;
                });
                ticking = true;
            }
        });
    }

    // ====================
    // External Links Handler
    // ====================
    function markExternalLinks() {
        document.querySelectorAll('a[href^="http"]').forEach(link => {
            if (link.hostname === window.location.hostname) return;
            
            link.setAttribute('target', '_blank');
            link.setAttribute('rel', 'noopener noreferrer');
            link.setAttribute('title', 'Opens in new tab');
            
            if (!link.querySelector('.external-icon')) {
                const icon = document.createElement('span');
                icon.className = 'external-icon';
                icon.innerHTML = '<svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor" style="display:inline-block;margin-left:0.25em;opacity:0.6;vertical-align:middle;"><path d="M10.5 1.5h-3a.75.75 0 000 1.5h1.19L5.47 6.22a.75.75 0 101.06 1.06l3.22-3.22v1.19a.75.75 0 001.5 0v-3a.75.75 0 00-.75-.75z"/><path d="M2.5 3a.5.5 0 00-.5.5v6a.5.5 0 00.5.5h6a.5.5 0 00.5-.5V6.75a.75.75 0 011.5 0v2.75A2 2 0 018.5 11.5h-6A2 2 0 01.5 9.5v-6A2 2 0 012.5 1.5h2.75a.75.75 0 010 1.5H2.5z"/></svg>';
                link.appendChild(icon);
            }
        });
    }

    // ====================
    // Mobile Navigation
    // ====================
    function initMobileNav() {
        if (window.innerWidth > 1024) return;

        const sidebar = document.querySelector('.sidebar');
        if (!sidebar) return;

        const toggle = document.createElement('button');
        toggle.innerHTML = '<svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M3 5a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM3 10a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM3 15a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clip-rule="evenodd"/></svg> <span>Table of Contents</span>';
        toggle.style.cssText = `
            width: 100%;
            padding: 1rem;
            background: linear-gradient(135deg, var(--color-primary), var(--color-accent));
            color: white;
            border: none;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            font-size: var(--font-size-base);
            transition: all 0.3s;
        `;

        const nav = sidebar.querySelector('.nav');
        if (nav) {
            nav.style.display = 'none';
            nav.style.maxHeight = '400px';
            nav.style.overflowY = 'auto';
            
            toggle.addEventListener('click', function() {
                const isOpen = nav.style.display === 'block';
                nav.style.display = isOpen ? 'none' : 'block';
                if (isOpen) {
                    this.innerHTML = '<svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M3 5a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM3 10a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM3 15a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clip-rule="evenodd"/></svg> <span>Table of Contents</span>';
                } else {
                    this.innerHTML = '<svg width="20" height="20" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"/></svg> <span>Close</span>';
                }
            });

            sidebar.insertBefore(toggle, nav);
        }
    }

    // ====================
    // Add ripple animation CSS
    // ====================
    function addAnimationStyles() {
        const style = document.createElement('style');
        style.textContent = `
            @keyframes ripple {
                to {
                    transform: scale(4);
                    opacity: 0;
                }
            }
            
            .search-highlight {
                position: relative;
                animation: highlightPulse 0.6s ease-out;
            }
            
            @keyframes highlightPulse {
                0%, 100% {
                    background-color: transparent;
                }
                50% {
                    background-color: rgba(37, 99, 235, 0.1);
                }
            }
            
            .search-highlight::before {
                content: '';
                position: absolute;
                top: -8px;
                left: -16px;
                right: -16px;
                bottom: -8px;
                background: rgba(37, 99, 235, 0.05);
                border-left: 3px solid var(--color-primary);
                border-radius: var(--border-radius-lg);
                pointer-events: none;
                z-index: -1;
            }
        `;
        document.head.appendChild(style);
    }

    // ====================
    // Initialize Everything
    // ====================
    function init() {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', init);
            return;
        }

        console.log('%c🚀 Sphana Trainer Documentation', 
            'font-size: 1.5em; font-weight: bold; color: #2563eb;');
        console.log('%cInitializing enhanced features...', 
            'color: #64748b;');

        try {
            addAnimationStyles();
            PageNavigator.init(); // Initialize multi-page navigation
            AutocompleteSearch.init(); // Initialize autocomplete search
            ThemeManager.init();
            initCopyButtons();
            // initActiveNav(); // Disabled - PageNavigator handles this now
            // initSmoothScroll(); // Disabled - Not needed for multi-page
            // initSearch(); // Disabled - Replaced by AutocompleteSearch
            initKeyboardShortcuts();
            initBackToTop();
            enhanceCodeBlocks();
            initProgressiveLoading();
            initReadingProgress();
            markExternalLinks();
            initMobileNav();

            console.log('%c✓ Documentation ready!', 
                'color: #10b981; font-weight: bold;');
            console.log('%cKeyboard shortcuts: Ctrl+K (search), T (theme), N (next), P (prev), GH (home), GG (top)', 
                'color: #94a3b8; font-size: 0.9em;');
        } catch (error) {
            console.error('Error initializing documentation:', error);
        }
    }

    // Start initialization
    init();

})();
