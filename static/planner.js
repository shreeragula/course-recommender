// Study Planner Javascript
let plannerTasks = JSON.parse(localStorage.getItem('plannerTasks')) || [];
let currentMonth = new Date().getMonth();
let currentYear = new Date().getFullYear();
let notificationPermission = false;

// Request notification permission on load
if ("Notification" in window) {
    Notification.requestPermission().then(permission => {
        notificationPermission = permission === "granted";
    });
}

function planner_saveTasksToStorage() {
    localStorage.setItem('plannerTasks', JSON.stringify(plannerTasks));
    planner_updateDashboardWidget();
}

function planner_openAddTaskModal(taskId = null) {
    document.getElementById('planner-task-modal').style.display = 'flex';
    document.getElementById('task-id').value = taskId || '';
    
    if (taskId) {
        document.getElementById('planner-modal-title').innerText = "Edit Task";
        let task = plannerTasks.find(t => t.id === taskId);
        if (task) {
            document.getElementById('task-title').value = task.title;
            document.getElementById('task-desc').value = task.description;
            document.getElementById('task-date').value = task.date;
            document.getElementById('task-time').value = task.time;
            document.getElementById('task-priority').value = task.priority;
            document.getElementById('task-category').value = task.category;
        }
    } else {
        document.getElementById('planner-modal-title').innerText = "Add New Task";
        document.getElementById('planner-task-form').reset();
    }
}

function planner_closeAddTaskModal() {
    document.getElementById('planner-task-modal').style.display = 'none';
}

function planner_saveTask(event) {
    event.preventDefault();
    let id = document.getElementById('task-id').value;
    
    let task = {
        id: id ? id : 'task_' + new Date().getTime(),
        title: document.getElementById('task-title').value,
        description: document.getElementById('task-desc').value,
        date: document.getElementById('task-date').value,
        time: document.getElementById('task-time').value,
        priority: document.getElementById('task-priority').value,
        category: document.getElementById('task-category').value,
        completed: false,
        notified: false
    };

    if (id) {
        let index = plannerTasks.findIndex(t => t.id === id);
        if (index > -1) {
            task.completed = plannerTasks[index].completed;
            task.notified = plannerTasks[index].notified;
            plannerTasks[index] = task;
        }
    } else {
        plannerTasks.push(task);
    }
    
    planner_saveTasksToStorage();
    planner_closeAddTaskModal();
    planner_renderTasks();
}

function planner_toggleTaskCompletion(id) {
    let task = plannerTasks.find(t => t.id === id);
    if (task) {
        task.completed = !task.completed;
        planner_saveTasksToStorage();
        planner_renderTasks();
    }
}

function planner_deleteTask(id) {
    if (confirm("Are you sure you want to delete this task?")) {
        plannerTasks = plannerTasks.filter(t => t.id !== id);
        planner_saveTasksToStorage();
        planner_renderTasks();
    }
}

function planner_getPriorityColor(priority) {
    if (priority === 'High') return '#ef4444';
    if (priority === 'Medium') return '#f59e0b';
    return '#10b981';
}

function planner_createTaskCard(task) {
    let card = document.createElement('div');
    card.className = `planner-card ${task.completed ? 'completed' : ''}`;
    card.style.borderLeft = `4px solid ${planner_getPriorityColor(task.priority)}`;
    card.style.background = 'var(--card-bg)';
    card.style.border = '1px solid var(--card-border)';
    card.style.padding = '15px';
    card.style.borderRadius = '12px';
    card.style.marginBottom = '15px';
    card.style.display = 'flex';
    card.style.flexDirection = 'column';
    card.style.gap = '10px';
    
    let now = new Date();
    let taskDate = new Date(task.date + 'T' + task.time);
    let countdownText = "";
    
    if (!task.completed) {
        let diffMs = taskDate - now;
        if (diffMs > 0) {
            let diffHrs = Math.floor(diffMs / 3600000);
            let diffMins = Math.floor((diffMs % 3600000) / 60000);
            countdownText = `Starts in: ${diffHrs}h ${diffMins}m`;
        } else {
            countdownText = "Due Now!";
        }
    }

    card.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <h3 style="margin: 0; font-size: 1.1rem; color: var(--text-main); ${task.completed ? 'text-decoration: line-through; opacity: 0.6;' : ''}">${task.title}</h3>
            <div style="display: flex; gap: 8px;">
                <span style="font-size: 0.75rem; padding: 4px 8px; border-radius: 6px; background: rgba(255,255,255,0.1); color: var(--text-muted);">${task.category}</span>
                <span style="font-size: 0.75rem; padding: 4px 8px; border-radius: 6px; background: ${planner_getPriorityColor(task.priority)}20; color: ${planner_getPriorityColor(task.priority)};">${task.priority}</span>
            </div>
        </div>
        <p style="margin: 0; font-size: 0.9rem; color: var(--text-muted); ${task.completed ? 'opacity: 0.6;' : ''}">${task.description || ''}</p>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 5px; border-top: 1px solid var(--card-border); padding-top: 10px;">
            <div style="font-size: 0.85rem; color: var(--text-muted); display: flex; gap: 15px;">
                <span>📅 ${task.date}</span>
                <span>⏰ ${task.time}</span>
            </div>
            <div style="color: ${countdownText === 'Due Now!' ? '#ef4444' : 'var(--text-muted)'}; font-size: 0.85rem; font-style: italic; font-weight: 500;">
                ${countdownText}
            </div>
            <div style="display: flex; gap: 10px;">
                <button onclick="planner_toggleTaskCompletion('${task.id}')" title="Complete" style="background: none; border: none; cursor: pointer; font-size: 1.2rem; filter: grayscale(${task.completed ? '0' : '1'});">${task.completed ? '✅' : '⬜'}</button>
                <button onclick="planner_openAddTaskModal('${task.id}')" title="Edit" style="background: none; border: none; cursor: pointer; font-size: 1.1rem;">✏️</button>
                <button onclick="planner_deleteTask('${task.id}')" title="Delete" style="background: none; border: none; cursor: pointer; font-size: 1.1rem;">🗑️</button>
            </div>
        </div>
    `;
    return card;
}

function planner_renderTasks() {
    let todayStr = new Date().toISOString().split('T')[0];
    
    let searchEl = document.getElementById('planner-search');
    let searchQuery = searchEl ? searchEl.value.toLowerCase() : '';
    let categoryEl = document.getElementById('planner-filter-category');
    let filterCategory = categoryEl ? categoryEl.value : 'all';
    let priorityEl = document.getElementById('planner-filter-priority');
    let filterPriority = priorityEl ? priorityEl.value : 'all';
    
    let filteredTasks = plannerTasks.filter(task => {
        let matchesSearch = task.title.toLowerCase().includes(searchQuery) || (task.description && task.description.toLowerCase().includes(searchQuery)) || task.category.toLowerCase().includes(searchQuery);
        let matchesCategory = filterCategory === 'all' || task.category === filterCategory;
        let matchesPriority = filterPriority === 'all' || task.priority === filterPriority;
        return matchesSearch && matchesCategory && matchesPriority;
    });

    let todayContainer = document.getElementById('tasks-today');
    let upcomingContainer = document.getElementById('tasks-upcoming');
    let completedContainer = document.getElementById('tasks-completed');
    
    if(!todayContainer || !upcomingContainer || !completedContainer) return;

    todayContainer.innerHTML = '';
    upcomingContainer.innerHTML = '';
    completedContainer.innerHTML = '';

    let completedCount = 0;
    let todayCount = 0;

    filteredTasks.sort((a, b) => new Date(a.date + 'T' + a.time) - new Date(b.date + 'T' + b.time));

    filteredTasks.forEach(task => {
        let card = planner_createTaskCard(task);
        if (task.completed) {
            completedContainer.appendChild(card);
            if(task.date === todayStr) completedCount++;
        } else {
            if (task.date === todayStr) {
                todayContainer.appendChild(card);
                todayCount++;
            } else if (task.date > todayStr) {
                upcomingContainer.appendChild(card);
            }
        }
    });

    if(todayContainer.children.length === 0) todayContainer.innerHTML = '<p style="color:var(--text-muted); font-style:italic; padding: 10px;">No tasks for today.</p>';
    if(upcomingContainer.children.length === 0) upcomingContainer.innerHTML = '<p style="color:var(--text-muted); font-style:italic; padding: 10px;">No upcoming tasks.</p>';
    if(completedContainer.children.length === 0) completedContainer.innerHTML = '<p style="color:var(--text-muted); font-style:italic; padding: 10px;">No completed tasks yet.</p>';

    // Update Progress Dashboard
    let totalTasksToday = todayCount + completedCount;
    let completionPercent = totalTasksToday === 0 ? 0 : Math.round((completedCount / totalTasksToday) * 100);
    
    let statComp = document.getElementById('planner-stat-completed');
    if(statComp) statComp.innerText = completedCount;
    let statRem = document.getElementById('planner-stat-remaining');
    if(statRem) statRem.innerText = todayCount;
    let statPerc = document.getElementById('planner-stat-percentage');
    if(statPerc) statPerc.innerText = completionPercent + '%';
    let statStrk = document.getElementById('planner-stat-streak');
    let streakVal = completedCount > 0 ? 1 : 0;
    if(statStrk) statStrk.innerText = streakVal;

    // Update sticky sidebar
    let sidebarGoal = document.getElementById('sidebar-goal-text');
    if (sidebarGoal) sidebarGoal.innerText = `${completedCount}/${totalTasksToday}`;
    
    let sidebarStreak = document.getElementById('sidebar-streak-text');
    if (sidebarStreak) sidebarStreak.innerText = `${streakVal} Days`;

    planner_updateDashboardWidget();
}

function planner_updateDashboardWidget() {
    let el = document.getElementById('widget-planner-status');
    if(!el) return;
    
    let todayStr = new Date().toISOString().split('T')[0];
    let pendingToday = plannerTasks.filter(t => t.date === todayStr && !t.completed).length;
    let upcoming = plannerTasks.filter(t => !t.completed).sort((a, b) => new Date(a.date + 'T' + a.time) - new Date(b.date + 'T' + b.time))[0];
    
    let msg = `You have ${pendingToday} pending task(s) today.`;
    
    let sidebarReminder = document.getElementById('sidebar-next-reminder');
    
    if (upcoming) {
        msg += ` Next up: <b>${upcoming.title}</b> at ${upcoming.time}.`;
        if (sidebarReminder) sidebarReminder.innerHTML = `<strong>${upcoming.title}</strong><br/>Today at ${upcoming.time}`;
    } else {
        if (sidebarReminder) sidebarReminder.innerHTML = `No upcoming tasks.`;
    }
    el.innerHTML = msg;
}

function planner_clearFilters() {
    let searchEl = document.getElementById('planner-search');
    let categoryEl = document.getElementById('planner-filter-category');
    let priorityEl = document.getElementById('planner-filter-priority');
    if (searchEl) searchEl.value = '';
    if (categoryEl) categoryEl.value = 'all';
    if (priorityEl) priorityEl.value = 'all';
    planner_renderTasks();
}

function planner_switchView(view) {
    document.querySelectorAll('.planner-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.planner-view').forEach(v => v.style.display = 'none');
    
    let activeTabBtn = null;

    if(view === 'list') {
        document.getElementById('planner-view-list').style.display = 'block';
        document.getElementById('planner-filters-bar').style.display = 'flex';
        activeTabBtn = document.querySelector('.planner-tab[onclick*="list"]');
    } else if (view === 'calendar') {
        document.getElementById('planner-view-calendar').style.display = 'block';
        document.getElementById('planner-filters-bar').style.display = 'none';
        planner_renderCalendar();
        activeTabBtn = document.querySelector('.planner-tab[onclick*="calendar"]');
    } else if (view === 'timetable') {
        document.getElementById('planner-view-timetable').style.display = 'block';
        document.getElementById('planner-filters-bar').style.display = 'none';
        planner_renderTimetable();
        activeTabBtn = document.querySelector('.planner-tab[onclick*="timetable"]');
    }

    if (activeTabBtn) activeTabBtn.classList.add('active');
}

// Timer and Notifications
setInterval(() => {
    let content = document.getElementById('planner_content');
    if (content && content.classList.contains('active') && document.getElementById('planner-view-list').style.display === 'block') {
        planner_renderTasks();
    }

    let now = new Date();
    plannerTasks.forEach((task, idx) => {
        if (!task.completed && !task.notified) {
            let taskTime = new Date(task.date + 'T' + task.time);
            if (taskTime <= now && (now - taskTime) < 60000) {
                plannerTasks[idx].notified = true;
                planner_saveTasksToStorage();
                if (notificationPermission) {
                    new Notification("Study Reminder", {
                        body: `Time to complete: ${task.title}`
                    });
                }
            }
        }
    });
}, 60000);

// Calendar View
function planner_prevMonth() {
    currentMonth--;
    if(currentMonth < 0) { currentMonth = 11; currentYear--; }
    planner_renderCalendar();
}
function planner_nextMonth() {
    currentMonth++;
    if(currentMonth > 11) { currentMonth = 0; currentYear++; }
    planner_renderCalendar();
}
function planner_renderCalendar() {
    let grid = document.getElementById('calendar-grid');
    if(!grid) return;
    
    const monthNames = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
    document.getElementById('calendar-month-year').innerText = `${monthNames[currentMonth]} ${currentYear}`;
    
    let firstDay = new Date(currentYear, currentMonth, 1).getDay();
    let daysInMonth = new Date(currentYear, currentMonth + 1, 0).getDate();
    
    let html = '';
    let days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
    days.forEach(d => html += `<div class="cal-header-day" style="font-weight:bold; color:var(--text-muted); text-align:center; padding:10px 0;">${d}</div>`);
    
    for(let i = 0; i < firstDay; i++) html += `<div class="cal-day empty" style="border: 1px solid var(--card-border); padding: 10px; min-height: 80px;"></div>`;
    
    for(let i = 1; i <= daysInMonth; i++) {
        let dateStr = `${currentYear}-${String(currentMonth + 1).padStart(2, '0')}-${String(i).padStart(2, '0')}`;
        let dayTasks = plannerTasks.filter(t => t.date === dateStr);
        let dots = dayTasks.map(t => `<div style="color:var(--text-main); font-size:0.7rem; background:${planner_getPriorityColor(t.priority)}20; padding:2px 4px; border-radius:4px; margin-top:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${t.completed?'✅ ':''}${t.title}</div>`).join('');
        
        let isToday = (new Date().toISOString().split('T')[0] === dateStr);
        
        html += `<div class="cal-day ${isToday ? 'today' : ''}" style="border: 1px solid var(--card-border); padding: 5px; min-height: 80px; background: ${isToday ? 'rgba(59, 130, 246, 0.1)' : 'transparent'};">
            <span class="day-num" style="display:block; text-align:right; font-size:0.9rem; color:var(--text-muted);">${i}</span>
            <div class="cal-dots" style="display:flex; flex-direction:column; gap:2px; margin-top:5px;">${dots}</div>
        </div>`;
    }
    grid.innerHTML = html;
}

// Timetable View
function planner_renderTimetable() {
    let grid = document.getElementById('timetable-grid');
    if(!grid) return;
    
    let html = '<div style="font-weight:bold; padding:10px; border-bottom:1px solid var(--card-border);">Time</div>';
    let days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
    days.forEach(d => html += `<div style="font-weight:bold; text-align:center; padding:10px; border-bottom:1px solid var(--card-border);">${d}</div>`);
    
    let curr = new Date(); // get current week tasks
    let first = curr.getDate() - curr.getDay();
    
    for(let h = 6; h <= 22; h++) {
        html += `<div style="text-align:right; padding-right:10px; color:var(--text-muted); font-size:0.8rem; padding-top:10px; border-bottom:1px solid var(--card-border);">${h}:00</div>`;
        for(let d = 0; d < 7; d++) {
            let targetDate = new Date(curr.setDate(first + d));
            let dateStr = targetDate.toISOString().split('T')[0];
            
            // find task roughly at this hour
            let hrTasks = plannerTasks.filter(t => t.date === dateStr && parseInt(t.time.split(':')[0]) === h);
            let badges = hrTasks.map(t => `<div style="font-size:0.7rem; background:${planner_getPriorityColor(t.priority)}20; padding:2px 4px; border-radius:4px; margin-bottom:2px; color:var(--text-main);">${t.title}</div>`).join('');
            
            html += `<div style="border-left:1px solid var(--card-border); border-bottom:1px solid var(--card-border); padding:5px;">${badges}</div>`;
        }
        // reset curr
        curr = new Date();
    }
    grid.innerHTML = html;
}

// Export functionality
// Export functionality
function planner_exportCSV() {
    let csvContent = "data:text/csv;charset=utf-8,ID,Title,Description,Date,Time,Priority,Category,Completed\\n";
    plannerTasks.forEach(task => {
        let row = `"${task.id}","${task.title}","${task.description}","${task.date}","${task.time}","${task.priority}","${task.category}","${task.completed}"`;
        csvContent += row + "\\n";
    });
    let encodedUri = encodeURI(csvContent);
    let link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "study_planner_tasks.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function planner_exportPDF() {
    alert("Task List CSV will be downloaded. To generate PDF, open the CSV in Excel/Sheets and Print to PDF. (A pure JS PDF generator can be added later)");
    planner_exportCSV();
}

// Initial Render
window.addEventListener('DOMContentLoaded', () => {
    setTimeout(() => {
        planner_renderTasks();
        planner_updateDashboardWidget();
    }, 100);
});
