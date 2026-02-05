/**
 * Telegram Analytics Dashboard - JavaScript
 *
 * Handles all interactivity:
 * - Data fetching from API
 * - Chart rendering with Chart.js
 * - Real-time updates
 * - User interactions
 * - Export functionality
 */

// ==========================================
// MOBILE MENU
// ==========================================

function toggleMobileMenu() {
    const sidebar = document.querySelector('.sidebar');
    const overlay = document.querySelector('.sidebar-overlay');
    sidebar.classList.toggle('open');
    if (overlay) overlay.classList.toggle('active');
}

// Close mobile menu when clicking a nav link
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.nav-link').forEach(function(link) {
        link.addEventListener('click', function() {
            if (window.innerWidth <= 768) {
                const sidebar = document.querySelector('.sidebar');
                const overlay = document.querySelector('.sidebar-overlay');
                sidebar.classList.remove('open');
                if (overlay) overlay.classList.remove('active');
            }
        });
    });
});

// ==========================================
// GLOBAL STATE
// ==========================================

const state = {
    timeframe: 'month',
    charts: {},
    autoRefresh: null,
    currentPage: 1,
    usersPerPage: 20
};

// Chart.js default configuration
Chart.defaults.color = '#a0aec0';
Chart.defaults.borderColor = '#2d3748';
Chart.defaults.font.family = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';

// ==========================================
// UTILITY FUNCTIONS
// ==========================================

function formatNumber(num) {
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num.toLocaleString();
}

function formatDate(timestamp) {
    if (!timestamp) return '-';
    return new Date(timestamp * 1000).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
}

function getTimeframe() {
    const select = document.getElementById('timeframe');
    return select ? select.value : state.timeframe;
}

async function fetchAPI(endpoint) {
    try {
        const timeframe = getTimeframe();
        const separator = endpoint.includes('?') ? '&' : '?';
        const response = await fetch(`${endpoint}${separator}timeframe=${timeframe}`);
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        return null;
    }
}

function showLoading(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    }
}

function showEmpty(elementId, message = 'No data available') {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">📭</div>
                <p>${message}</p>
            </div>
        `;
    }
}

// ==========================================
// DATA LOADING
// ==========================================

async function loadAllData() {
    state.timeframe = getTimeframe();

    // Load all data in parallel
    await Promise.all([
        loadOverviewStats(),
        loadMessagesChart(),
        loadUsersChart(),
        loadHourlyChart(),
        loadDailyChart(),
        loadHeatmap(),
        loadTopUsers(),
        loadTopWords(),
        loadTopDomains()
    ]);
}

async function loadOverviewStats() {
    const data = await fetchAPI('/api/overview');
    if (!data) return;

    // Update stat cards
    document.getElementById('total-messages').textContent = formatNumber(data.total_messages);
    document.getElementById('active-users').textContent = formatNumber(data.active_users);
    document.getElementById('messages-per-day').textContent = formatNumber(data.messages_per_day);
    document.getElementById('links-count').textContent = formatNumber(data.links_count);
    document.getElementById('media-count').textContent = formatNumber(data.media_count);
    document.getElementById('replies-count').textContent = formatNumber(data.replies_count);
}

// ==========================================
// CHARTS
// ==========================================

async function loadMessagesChart() {
    const granularitySelect = document.getElementById('messages-granularity');
    const granularity = granularitySelect ? granularitySelect.value : 'day';

    const data = await fetchAPI(`/api/chart/messages?granularity=${granularity}`);
    if (!data || data.length === 0) return;

    const ctx = document.getElementById('messages-chart');
    if (!ctx) return;

    // Destroy existing chart
    if (state.charts.messages) {
        state.charts.messages.destroy();
    }

    state.charts.messages = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.map(d => d.label),
            datasets: [{
                label: 'Messages',
                data: data.map(d => d.value),
                borderColor: '#0088cc',
                backgroundColor: 'rgba(0, 136, 204, 0.1)',
                fill: true,
                tension: 0.4,
                pointRadius: 2,
                pointHoverRadius: 5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { maxTicksLimit: 10 }
                },
                y: {
                    beginAtZero: true,
                    grid: { color: '#2d3748' }
                }
            },
            interaction: {
                intersect: false,
                mode: 'index'
            }
        }
    });
}

async function loadUsersChart() {
    const data = await fetchAPI('/api/chart/users?granularity=day');
    if (!data || data.length === 0) return;

    const ctx = document.getElementById('users-chart');
    if (!ctx) return;

    if (state.charts.users) {
        state.charts.users.destroy();
    }

    state.charts.users = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.map(d => d.label),
            datasets: [{
                label: 'Active Users',
                data: data.map(d => d.value),
                borderColor: '#28a745',
                backgroundColor: 'rgba(40, 167, 69, 0.1)',
                fill: true,
                tension: 0.4,
                pointRadius: 2,
                pointHoverRadius: 5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { maxTicksLimit: 10 }
                },
                y: {
                    beginAtZero: true,
                    grid: { color: '#2d3748' }
                }
            }
        }
    });
}

async function loadHourlyChart() {
    const data = await fetchAPI('/api/chart/hourly');
    if (!data || data.length === 0) return;

    const ctx = document.getElementById('hourly-chart');
    if (!ctx) return;

    if (state.charts.hourly) {
        state.charts.hourly.destroy();
    }

    state.charts.hourly = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.map(d => d.label),
            datasets: [{
                label: 'Messages',
                data: data.map(d => d.value),
                backgroundColor: '#0088cc',
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { maxTicksLimit: 12 }
                },
                y: {
                    beginAtZero: true,
                    grid: { color: '#2d3748' }
                }
            }
        }
    });
}

async function loadDailyChart() {
    const data = await fetchAPI('/api/chart/daily');
    if (!data || data.length === 0) return;

    const ctx = document.getElementById('daily-chart');
    if (!ctx) return;

    if (state.charts.daily) {
        state.charts.daily.destroy();
    }

    const colors = [
        '#dc3545', // Sunday - red
        '#ffc107', // Monday - yellow
        '#28a745', // Tuesday - green
        '#17a2b8', // Wednesday - cyan
        '#0088cc', // Thursday - blue
        '#6f42c1', // Friday - purple
        '#fd7e14'  // Saturday - orange
    ];

    state.charts.daily = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.map(d => d.label.substring(0, 3)),
            datasets: [{
                label: 'Messages',
                data: data.map(d => d.value),
                backgroundColor: colors,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { display: false }
                },
                y: {
                    beginAtZero: true,
                    grid: { color: '#2d3748' }
                }
            }
        }
    });
}

async function loadHeatmap() {
    const data = await fetchAPI('/api/chart/heatmap');
    if (!data || !data.data) return;

    const container = document.getElementById('heatmap');
    if (!container) return;

    // Find max value for color scaling
    const maxValue = Math.max(...data.data.flat());

    // Generate color based on intensity
    function getColor(value) {
        if (value === 0) return 'rgba(0, 136, 204, 0.1)';
        const intensity = value / maxValue;
        return `rgba(0, 136, 204, ${0.2 + intensity * 0.8})`;
    }

    let html = '<table class="heatmap-table"><thead><tr><th></th>';

    // Hour headers
    for (let h = 0; h < 24; h++) {
        html += `<th>${h}</th>`;
    }
    html += '</tr></thead><tbody>';

    // Day rows
    data.days.forEach((day, dayIndex) => {
        html += `<tr><td class="day-label">${day.substring(0, 3)}</td>`;
        for (let h = 0; h < 24; h++) {
            const value = data.data[dayIndex][h];
            const color = getColor(value);
            html += `<td><div class="heatmap-cell" style="background: ${color}" title="${day} ${h}:00 - ${value} messages"></div></td>`;
        }
        html += '</tr>';
    });

    html += '</tbody></table>';
    container.innerHTML = html;
}

// ==========================================
// TOP LISTS
// ==========================================

async function loadTopUsers() {
    const listElement = document.getElementById('top-users-list');
    if (!listElement) return;

    showLoading('top-users-list');

    const data = await fetchAPI('/api/users?limit=10');
    if (!data || !data.users || data.users.length === 0) {
        showEmpty('top-users-list');
        return;
    }

    let html = '';
    data.users.forEach((user, index) => {
        const rankClass = index === 0 ? 'gold' : index === 1 ? 'silver' : index === 2 ? 'bronze' : '';
        const initial = user.name.charAt(0).toUpperCase();

        html += `
            <div class="list-item" onclick="window.location.href='/user/${user.user_id}'" style="cursor: pointer">
                <div class="list-rank ${rankClass}">#${user.rank}</div>
                <div class="user-avatar">${initial}</div>
                <div class="list-info">
                    <div class="list-name">${escapeHtml(user.name)}</div>
                    <div class="list-subtitle">${user.percentage}% of total</div>
                </div>
                <div class="list-value">${formatNumber(user.messages)}</div>
            </div>
        `;
    });

    listElement.innerHTML = html;
}

async function loadTopWords() {
    const listElement = document.getElementById('top-words-list');
    if (!listElement) return;

    showLoading('top-words-list');

    const data = await fetchAPI('/api/top/words?limit=10');
    if (!data || data.length === 0) {
        showEmpty('top-words-list');
        return;
    }

    const maxCount = data[0].count;
    let html = '';

    data.forEach((item, index) => {
        const percentage = (item.count / maxCount * 100).toFixed(0);
        html += `
            <div class="list-item">
                <div class="list-rank">#${index + 1}</div>
                <div class="list-info">
                    <div class="list-name">${escapeHtml(item.word)}</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${percentage}%"></div>
                    </div>
                </div>
                <div class="list-value">${formatNumber(item.count)}</div>
            </div>
        `;
    });

    listElement.innerHTML = html;
}

async function loadTopDomains() {
    const listElement = document.getElementById('top-domains-list');
    if (!listElement) return;

    showLoading('top-domains-list');

    const data = await fetchAPI('/api/top/domains?limit=10');
    if (!data || data.length === 0) {
        showEmpty('top-domains-list');
        return;
    }

    const maxCount = data[0].count;
    let html = '';

    data.forEach((item, index) => {
        const percentage = (item.count / maxCount * 100).toFixed(0);
        html += `
            <div class="list-item">
                <div class="list-rank">#${index + 1}</div>
                <div class="list-info">
                    <div class="list-name">${escapeHtml(item.domain)}</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${percentage}%"></div>
                    </div>
                </div>
                <div class="list-value">${formatNumber(item.count)}</div>
            </div>
        `;
    });

    listElement.innerHTML = html;
}

// ==========================================
// USER MODAL
// ==========================================

async function openUserModal(userId) {
    // Create modal if it doesn't exist
    let modal = document.getElementById('user-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'user-modal';
        modal.className = 'modal-overlay';
        modal.innerHTML = `
            <div class="modal">
                <div class="modal-header">
                    <h2>User Details</h2>
                    <button class="modal-close" onclick="closeUserModal()">&times;</button>
                </div>
                <div class="modal-body" id="user-modal-content">
                    <div class="loading"><div class="spinner"></div></div>
                </div>
            </div>
        `;
        document.body.appendChild(modal);

        // Close on backdrop click
        modal.addEventListener('click', (e) => {
            if (e.target === modal) closeUserModal();
        });
    }

    modal.classList.add('active');
    document.getElementById('user-modal-content').innerHTML = '<div class="loading"><div class="spinner"></div></div>';

    const data = await fetchAPI(`/api/user/${userId}`);
    if (!data || data.error) {
        document.getElementById('user-modal-content').innerHTML = '<div class="empty-state"><p>User not found</p></div>';
        return;
    }

    const initial = data.name.charAt(0).toUpperCase();

    document.getElementById('user-modal-content').innerHTML = `
        <div class="user-profile">
            <div class="user-profile-avatar">${initial}</div>
            <div class="user-profile-info">
                <h3>${escapeHtml(data.name)}</h3>
                <p>Rank #${data.rank} • Member since ${formatDate(data.first_seen)}</p>
            </div>
        </div>

        <div class="user-stats-grid">
            <div class="user-stat">
                <div class="user-stat-value">${formatNumber(data.messages)}</div>
                <div class="user-stat-label">Messages</div>
            </div>
            <div class="user-stat">
                <div class="user-stat-value">${formatNumber(data.characters)}</div>
                <div class="user-stat-label">Characters</div>
            </div>
            <div class="user-stat">
                <div class="user-stat-value">${data.daily_average}</div>
                <div class="user-stat-label">Daily Avg</div>
            </div>
            <div class="user-stat">
                <div class="user-stat-value">${formatNumber(data.links)}</div>
                <div class="user-stat-label">Links</div>
            </div>
            <div class="user-stat">
                <div class="user-stat-value">${formatNumber(data.media)}</div>
                <div class="user-stat-label">Media</div>
            </div>
            <div class="user-stat">
                <div class="user-stat-value">${data.active_days}</div>
                <div class="user-stat-label">Active Days</div>
            </div>
        </div>

        <h4 style="margin-bottom: 1rem;">Activity by Hour</h4>
        <canvas id="user-hourly-chart" height="150"></canvas>
    `;

    // Render user's hourly chart
    const ctx = document.getElementById('user-hourly-chart');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Array.from({length: 24}, (_, i) => `${i}:00`),
            datasets: [{
                data: data.hourly_activity,
                backgroundColor: '#0088cc',
                borderRadius: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { display: false }, ticks: { maxTicksLimit: 12 } },
                y: { beginAtZero: true, grid: { color: '#2d3748' } }
            }
        }
    });
}

function closeUserModal() {
    const modal = document.getElementById('user-modal');
    if (modal) modal.classList.remove('active');
}

// ==========================================
// EXPORT FUNCTIONS
// ==========================================

function exportUsers() {
    const timeframe = getTimeframe();
    window.location.href = `/api/export/users?timeframe=${timeframe}`;
}

function exportMessages() {
    const timeframe = getTimeframe();
    window.location.href = `/api/export/messages?timeframe=${timeframe}`;
}

// ==========================================
// AUTO REFRESH
// ==========================================

function toggleAutoRefresh() {
    if (state.autoRefresh) {
        clearInterval(state.autoRefresh);
        state.autoRefresh = null;
        console.log('Auto-refresh disabled');
    } else {
        state.autoRefresh = setInterval(loadAllData, 60000); // Refresh every minute
        console.log('Auto-refresh enabled (60s)');
    }
}

// ==========================================
// UTILITY
// ==========================================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Escape to close modal
    if (e.key === 'Escape') {
        closeUserModal();
    }
    // R to refresh
    if (e.key === 'r' && !e.ctrlKey && !e.metaKey) {
        const activeElement = document.activeElement;
        if (activeElement.tagName !== 'INPUT' && activeElement.tagName !== 'TEXTAREA') {
            loadAllData();
        }
    }
});
