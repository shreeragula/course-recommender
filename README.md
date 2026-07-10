# CareerPath AI 🚀
An AI-powered course recommender, career roadmap generator, interactive interview simulator, and study planner built with Flask, SQLite, and Gemini AI.

---

## 🌟 Features

### 1. 🔍 Personalized Course Recommendations
* **Content-Based Filtering**: Leverages Natural Language Processing (NLP) and Cosine Similarity to compare user preferences (interests, skill level, and career goals) with thousands of courses.
* **Smart Matching**: Suggests optimized custom study paths tailored to the user's available weekly hours and skill level (Beginner, Intermediate, Advanced).

### 2. 🚀 AI Career Path Advisor
* **Interactive Roadmap Generation**: Generates customized structured learning roadmaps directly using Gemini AI integration (`ai_advisor.py`).
* **Visual Paths**: Provides logical progression timelines including suggested tech stacks, milestones, and actionable tips for entering any modern tech field.

### 3. 💡 Interview Preparation Console
* **Dynamic Simulations**: Simulates technical, behavioral/HR, or mixed interview sessions based on the user's chosen role, experience, and difficulty level.
* **Dynamic AI Feedback**: Generates context-aware questions, analyzes user input, asks follow-up questions, and compiles a comprehensive evaluation report showing strengths, weaknesses, and expected answer keys.

### 4. 📅 Study Planner & Timetable
* **Stats Dashboard**: Interactive trackers showing Tasks Completed, Tasks Remaining, Current Streak, and Today's Goal completion rate.
* **Multiple Views**: Seamlessly switch between **List View**, **Calendar View**, and an interactive **Weekly Timetable Grid**.
* **Reminders**: Features custom priority labeling (High/Medium/Low), task categories, and local browser notifications for task reminders.

### 5. 🌗 Premium Responsive UI
* A highly optimized UI built with glassmorphic cards, smooth animations, and clean layouts.
* Native **Dark Mode** and **Light Mode** toggles using persistent CSS styling and theme caching.

---

## 🛠️ Technologies Used

### Backend
* **Python (Flask)**: Core server routing and logic.
* **SQLite (with Database.py helper)**: Robust user authentication, encrypted passwords, stored preferences, bookmark logs, and completed interview histories.
* **Gemini Pro (Google Generative AI)**: Powering the Career Roadmap Advisor and the dynamic Interview Simulator.
* **Pandas / NumPy / Scikit-learn**: Sourced data pre-processing, Cosine Similarity matching, and TF-IDF numerical vectorization.

### Frontend
* **HTML5 / CSS3 / JavaScript (ES6)**: Vanilla frontend with responsive layout design.
* **CSS Grid / Flexbox**: Used for dynamic calendar, scheduler, and timetable.
* **Google Fonts**: Plus Jakarta Sans.

---

## 📂 Project Structure

```text
📦 course-recommender
 ┣ 📂 templates
 ┃ ┣ 📜 landing.html             # Landing page for guests
 ┃ ┣ 📜 login.html               # Secure login form
 ┃ ┣ 📜 signup.html              # Secure signup form
 ┃ ┣ 📜 index.html               # Main dashboard (recommender, advisor, bookmarks)
 ┃ ┣ 📜 interview_session.html   # Active mock interview screen
 ┃ ┗ 📜 interview_summary.html   # Final AI feedback summary screen
 ┣ 📂 static
 ┃ ┣ 📜 planner.js               # Study planner interactive operations
 ┃ ┣ 📜 theme.css                # Color schemes for light & dark mode overrides
 ┃ ┣ 📜 theme.js                 # Theme toggler implementation
 ┃ ┗ 📜 bg.jpg                   # Glassmorphic background texture
 ┣ 📜 app.py                     # Flask server routes & application entry point
 ┣ 📜 ml_model.py                # Cosine Similarity course recommendation logic
 ┣ 📜 ai_advisor.py              # Gemini AI integration for roadmap generation
 ┣ 📜 database.py                # SQLite database management helper
 ┣ 📜 interview_ai.py            # Interview session logic & prompts
 ┣ 📜 Coursera.csv               # Dataset containing coursera course metadata
 ┣ 📜 course_embeddings.npy      # Pre-computed vectors for search recommendations
 ┣ 📜 recommender.db             # Local SQLite database file
 ┣ 📜 requirements.txt           # Python application dependencies
 ┗ 📜 README.md                  # Project documentation (this file)
```

---

## 🚀 Installation & Setup

### 1. Clone the repository
```sh
git clone https://github.com/shreeragula/course-recommender.git
cd course-recommender
```

### 2. Set up virtual environment
```sh
python -m venv venv
# On Windows use:
venv\Scripts\activate
# On Linux/macOS use:
source venv/bin/activate
```

### 3. Install dependencies
```sh
pip install -r requirements.txt
```

### 4. Set your API Key
The AI features require a Gemini API Key. Set it in your environment:
```sh
# On Windows (Command Prompt):
set GEMINI_API_KEY="your_api_key_here"
# On Windows (PowerShell):
$env:GEMINI_API_KEY="your_api_key_here"
# On Linux/macOS:
export GEMINI_API_KEY="your_api_key_here"
```

### 5. Start the Application
```sh
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000/` to start using the system.
