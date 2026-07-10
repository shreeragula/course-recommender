document.addEventListener('DOMContentLoaded', () => {
    const isLandingPage = !!document.querySelector('.landing-header');

    // 1. Dark/Light Theme Logic (Skip on landing page)
    if (!isLandingPage) {
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'light') {
            document.documentElement.classList.add('light-mode');
        }

        const toggleBtn = document.createElement('button');
        toggleBtn.className = 'theme-toggle-btn';
        toggleBtn.innerHTML = savedTheme === 'light' ? '🌙' : '☀️';
        toggleBtn.title = "Toggle Light/Dark Mode";
        document.body.appendChild(toggleBtn);

        toggleBtn.addEventListener('click', () => {
            const isLight = document.documentElement.classList.toggle('light-mode');
            localStorage.setItem('theme', isLight ? 'light' : 'dark');
            toggleBtn.innerHTML = isLight ? '🌙' : '☀️';
        });
    }

    // 2. Recently Viewed Courses Tracking
    const defaultRecentlyViewed = [
        { name: "Python Basics", url: "https://www.coursera.org/search?query=python%20basics" },
        { name: "SQL Masterclass", url: "https://www.coursera.org/search?query=sql%20masterclass" },
        { name: "React Fundamentals", url: "https://www.coursera.org/search?query=react%20fundamentals" }
    ];

    if (!localStorage.getItem('recentlyViewed')) {
        localStorage.setItem('recentlyViewed', JSON.stringify(defaultRecentlyViewed));
    }

    // Event delegation to capture course link clicks
    document.addEventListener('click', (e) => {
        const anchor = e.target.closest('a');
        if (!anchor) return;

        let isCourseLink = false;
        let courseName = '';

        if (anchor.classList.contains('btn-visit')) {
            isCourseLink = true;
            const card = anchor.closest('.recommendation-card');
            if (card) {
                const titleEl = card.querySelector('.course-title');
                if (titleEl) courseName = titleEl.textContent.trim();
            }
        } else if (anchor.classList.contains('course-card')) {
            isCourseLink = true;
            const titleEl = anchor.querySelector('.course-title');
            if (titleEl) courseName = titleEl.textContent.trim();
        }

        if (isCourseLink && courseName) {
            let recentlyViewed = JSON.parse(localStorage.getItem('recentlyViewed') || '[]');
            recentlyViewed = recentlyViewed.filter(item => item.name !== courseName);
            recentlyViewed.unshift({ name: courseName, url: anchor.href });
            recentlyViewed = recentlyViewed.slice(0, 5);
            localStorage.setItem('recentlyViewed', JSON.stringify(recentlyViewed));
            renderRecentlyViewed();
        }
    });

    function renderRecentlyViewed() {
        const listContainer = document.getElementById('recently-viewed-list');
        if (!listContainer) return;

        const recentlyViewed = JSON.parse(localStorage.getItem('recentlyViewed') || '[]');
        listContainer.innerHTML = '';

        if (recentlyViewed.length === 0) {
            listContainer.innerHTML = '<li style="color: var(--text-muted); font-size: 0.8rem; font-style: italic;">No courses viewed recently</li>';
            return;
        }

        recentlyViewed.forEach(item => {
            const li = document.createElement('li');
            li.style.overflow = 'hidden';
            li.style.textOverflow = 'ellipsis';
            li.style.whiteSpace = 'nowrap';

            const a = document.createElement('a');
            a.href = item.url;
            a.target = '_blank';
            a.textContent = item.name;
            a.style.color = 'var(--primary)';
            a.style.textDecoration = 'none';
            a.style.fontSize = '0.85rem';
            a.style.transition = 'color 0.2s';

            a.addEventListener('mouseenter', () => a.style.color = 'var(--primary-hover)');
            a.addEventListener('mouseleave', () => a.style.color = 'var(--primary)');

            a.addEventListener('click', () => {
                let rv = JSON.parse(localStorage.getItem('recentlyViewed') || '[]');
                rv = rv.filter(i => i.name !== item.name);
                rv.unshift(item);
                localStorage.setItem('recentlyViewed', JSON.stringify(rv));
                renderRecentlyViewed();
            });

            li.appendChild(a);
            listContainer.appendChild(li);
        });
    }

    renderRecentlyViewed();
});
