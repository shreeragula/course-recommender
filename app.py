import os
import json
import random
from flask import Flask, render_template, request, session, redirect, url_for, flash, g, jsonify
import pandas as pd
from ml_model import recommend_personalized  # Import personalized ML function
from database import (
    init_db,
    register_user,
    authenticate_user,
    update_user_preferences,
    get_user_preferences,
    save_course,
    get_saved_courses,
    delete_saved_course,
    get_db_connection,
    save_completed_interview,
    get_completed_interviews
)

app = Flask(__name__)
app.secret_key = "super_secret_course_recommender_key_9988"

# Initialize SQLite database tables on app boot
init_db()

@app.before_request
def load_logged_in_user():
    user_id = session.get('user_id')
    if user_id is None:
        g.user = None
    else:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, username FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            g.user = {"id": row["id"], "username": row["username"]}
        else:
            g.user = None

from ai_advisor import generate_roadmap  # Import AI advisor functions
from interview_ai import generate_interview_question, evaluate_user_response, generate_followup_question, generate_final_report  # Import dynamic interview tools

@app.route("/", methods=["GET", "POST"])
def home():
    if not g.user:
        return render_template("landing.html")
        
    recommendations = []
    active_tab = request.args.get("tab", "recommender")
    if active_tab not in ["recommender", "advisor", "bookmarks", "interview", "planner"]:
        active_tab = "recommender"
    advisor_data = None
    advisor_form = {
        "career": "",
        "skills": "",
        "experience": "beginner",
        "hours": "5"
    }
    
    # 1. Fetch user's current saved preferences for Course Recommender
    prefs = get_user_preferences(g.user["id"])
    form_data = {
        "interests": prefs["interests"] if prefs else "",
        "career_goal": prefs["career_goal"] if prefs else "",
        "skill_level": prefs["skill_level"] if prefs else "beginner",
        "weekly_hours": str(prefs["weekly_hours"]) if prefs else "5"
    }
    
    # Auto-generate recommendations on landing if interests are already saved
    if request.method == "GET" and form_data["interests"]:
        recommendations = recommend_personalized(
            interests=form_data["interests"],
            career_goal=form_data["career_goal"],
            skill_level=form_data["skill_level"],
            weekly_hours=form_data["weekly_hours"]
        )
        
    if request.method == "POST":
        form_type = request.form.get("form_type", "recommender")
        
        if form_type == "recommender":
            form_data = {
                "interests": request.form.get("interests", ""),
                "career_goal": request.form.get("career_goal", ""),
                "skill_level": request.form.get("skill_level", "beginner"),
                "weekly_hours": request.form.get("weekly_hours", "5")
            }
            # Update user preferences in database
            update_user_preferences(
                user_id=g.user["id"],
                interests=form_data["interests"],
                career_goal=form_data["career_goal"],
                skill_level=form_data["skill_level"],
                weekly_hours=form_data["weekly_hours"]
            )
            recommendations = recommend_personalized(
                interests=form_data["interests"],
                career_goal=form_data["career_goal"],
                skill_level=form_data["skill_level"],
                weekly_hours=form_data["weekly_hours"]
            )
            active_tab = "recommender"
            flash("Preferences updated successfully!")
            
        elif form_type == "advisor":
            advisor_form = {
                "career": request.form.get("advisor_career", ""),
                "skills": request.form.get("advisor_skills", ""),
                "experience": request.form.get("advisor_experience", "beginner"),
                "hours": request.form.get("advisor_hours", "5")
            }
            advisor_data = generate_roadmap(
                advisor_form["career"], 
                advisor_form["skills"],
                advisor_form["experience"],
                advisor_form["hours"]
            )
            active_tab = "advisor"
            
            # Since preferences weren't changed, load current course recommendations silently
            if form_data["interests"]:
                recommendations = recommend_personalized(
                    interests=form_data["interests"],
                    career_goal=form_data["career_goal"],
                    skill_level=form_data["skill_level"],
                    weekly_hours=form_data["weekly_hours"]
                )
            flash("AI Career Roadmap generated!")
        
    # Fetch saved/bookmarked courses
    saved_courses = get_saved_courses(g.user["id"])
    completed_interviews = get_completed_interviews(g.user["id"])
    
    return render_template(
        "index.html", 
        recommendations=recommendations, 
        form_data=form_data, 
        saved_courses=saved_courses,
        active_tab=active_tab,
        advisor_data=advisor_data,
        advisor_form=advisor_form,
        completed_interviews=completed_interviews
    )

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if g.user:
        return redirect(url_for("home"))
        
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        
        if not username or not password:
            flash("Username and password are required.")
        else:
            user_id = register_user(username, password)
            if user_id:
                session["user_id"] = user_id
                flash("Welcome! Your account has been created.")
                return redirect(url_for("home"))
            else:
                flash("Username is already taken. Please choose another.")
                
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if g.user:
        return redirect(url_for("home"))
        
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        
        user = authenticate_user(username, password)
        if user:
            session["user_id"] = user["id"]
            return redirect(url_for("home"))
        else:
            flash("Invalid username or password.")
            
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been signed out.")
    return redirect(url_for("login"))

@app.route("/save_course", methods=["POST"])
def bookmark_course():
    if not g.user:
        return redirect(url_for("login"))
        
    course_name = request.form.get("course_name")
    difficulty = request.form.get("difficulty")
    rating = request.form.get("rating")
    url = request.form.get("url")
    pacing = request.form.get("pacing")
    
    success = save_course(g.user["id"], course_name, difficulty, rating, url, pacing)
    if success:
        flash(f"Bookmarked: {course_name}")
    else:
        flash("Course is already bookmarked!")
        
    return redirect(url_for("home"))

@app.route("/delete_course/<int:course_id>", methods=["POST"])
def delete_course(course_id):
    if not g.user:
        return redirect(url_for("login"))
        
    delete_saved_course(course_id, g.user["id"])
    flash("Course removed from bookmarks.")
    next_tab = request.args.get("tab", "bookmarks")
    return redirect(url_for("home", tab=next_tab))

@app.route("/start_interview", methods=["POST"])
def start_interview():
    if not g.user:
        return redirect(url_for("login"))
        
    prep_role = request.form.get("prep_role", "AI Engineer")
    prep_experience = request.form.get("prep_experience", "Beginner")
    prep_type = request.form.get("prep_type", "Mixed")
    prep_difficulty = request.form.get("prep_difficulty", "Any")
    prep_num_questions = int(request.form.get("prep_num_questions", "5"))
    
    # Fetch user's current interests/skills from database
    prefs = get_user_preferences(g.user["id"])
    current_skills = prefs["interests"] if prefs else ""
    
    # Generate the very first question dynamically
    first_q = generate_interview_question(
        role=prep_role,
        current_skills=current_skills,
        experience_level=prep_experience,
        interview_type=prep_type,
        difficulty=prep_difficulty,
        past_questions=[]
    )
    first_q["id"] = 1
    
    import uuid
    session_id = str(uuid.uuid4())
    
    session["active_interview"] = {
        "session_id": session_id,
        "role": prep_role,
        "experience_level": prep_experience,
        "interview_type": prep_type,
        "difficulty": prep_difficulty,
        "current_skills": current_skills,
        "num_questions": prep_num_questions,
        "questions": [first_q],
        "current_index": 0,
        "answers": {},
        "evaluations": {}
    }
    session.modified = True
    
    return redirect(url_for("interview_session"))

@app.route("/interview/session")
def interview_session():
    if not g.user:
        return redirect(url_for("login"))
        
    active = session.get("active_interview")
    if not active:
        flash("No active interview session found. Please start a session first.")
        return redirect(url_for("home", tab="interview"))
        
    current_index = active["current_index"]
    questions = active["questions"]
    num_questions = active["num_questions"]
    
    if current_index >= num_questions:
        flash("Interview session already finished.")
        return redirect(url_for("home", tab="interview"))
        
    active_q = questions[current_index]
    current_num = current_index + 1
    progress_pct = int(((current_num - 1) / num_questions) * 100)
    is_last = (current_num == num_questions)
    
    return render_template(
        "interview_session.html",
        role=active["role"],
        type=active["interview_type"],
        difficulty=active["difficulty"],
        current_num=current_num,
        total_num=num_questions,
        progress_pct=progress_pct,
        active_q=active_q,
        is_last=is_last
    )

@app.route("/interview/submit_answer", methods=["POST"])
def submit_answer():
    if not g.user:
        return redirect(url_for("login"))
        
    active = session.get("active_interview")
    if not active:
        flash("No active interview session.")
        return redirect(url_for("home", tab="interview"))
        
    answer = request.form.get("answer", "").strip()
    current_index = active["current_index"]
    questions = active["questions"]
    num_questions = active["num_questions"]
    
    # 1. Save answer temporarily in session
    active_q = questions[current_index]
    active["answers"][str(active_q["id"])] = answer
    
    # 2. Call AI answer evaluator dynamically
    eval_res = evaluate_user_response(
        question=active_q["question"],
        expected_answer=active_q.get("expected_answer", ""),
        user_answer=answer
    )
    active["evaluations"][str(active_q["id"])] = eval_res
    
    next_index = current_index + 1
    
    # Check if we need to generate more questions
    if next_index < num_questions:
        # Generate the next question as a follow-up dynamically
        next_q = generate_followup_question(
            role=active["role"],
            previous_question=active_q["question"],
            previous_answer=answer,
            evaluation=eval_res
        )
        next_q["id"] = next_index + 1
        active["questions"].append(next_q)
        active["current_index"] = next_index
        
        session["active_interview"] = active
        session.modified = True
        return redirect(url_for("interview_session"))
    else:
        # Session completed, construct final history details
        history_list = []
        for q in questions:
            ans = active["answers"].get(str(q["id"]), "No response provided.")
            evaluation = active["evaluations"].get(str(q["id"]), {
                "score": 7,
                "strengths": ["Answer submitted."],
                "weaknesses": ["Feedback unavailable."],
                "correct_answer": q.get("expected_answer", ""),
                "tips": "Be concise in live interviews."
            })
            history_list.append({
                "id": q["id"],
                "question": q["question"],
                "expected_answer": q.get("expected_answer", ""),
                "concepts": q.get("concepts", []),
                "answer": ans,
                "evaluation": evaluation
            })
            
        import datetime
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # Generate final report
        final_report = generate_final_report(active["role"], active["experience_level"], history_list)
        
        # Suggest courses based on topics to revise
        topics_str = ", ".join(final_report.get("topics_to_revise", [])) if final_report else ""
        suggested_courses = recommend_personalized(topics_str, active["role"], active["experience_level"], "5")
        
        # Suggest roadmap updates
        roadmap_updates = generate_roadmap(active["role"], topics_str, active["experience_level"], "5")
        
        # Save permanently to the database
        import json
        save_completed_interview(
            user_id=g.user["id"],
            role=active["role"],
            experience_level=active["experience_level"],
            interview_type=active["interview_type"],
            difficulty=active["difficulty"],
            num_questions=num_questions,
            history_json=json.dumps({
                "history": history_list,
                "report": final_report,
                "suggested_courses": suggested_courses,
                "roadmap_updates": roadmap_updates
            })
        )
        
        # Save temporarily inside the session for the summary page
        session["last_summary"] = {
            "role": active["role"],
            "experience_level": active["experience_level"],
            "interview_type": active["interview_type"],
            "difficulty": active["difficulty"],
            "num_questions": num_questions,
            "timestamp": timestamp_str,
            "history": history_list,
            "report": final_report,
            "suggested_courses": suggested_courses,
            "roadmap_updates": roadmap_updates
        }
        
        # Clear the active session
        session.pop("active_interview", None)
        session.modified = True
        
        return redirect(url_for("interview_summary"))

@app.route("/interview/summary")
def interview_summary():
    if not g.user:
        return redirect(url_for("login"))
        
    summary_data = session.get("last_summary")
    if not summary_data:
        flash("No completed interview summary found in the current session.")
        return redirect(url_for("home", tab="interview"))
        
    return render_template(
        "interview_summary.html",
        record=summary_data,
        history=summary_data["history"]
    )

if __name__ == "__main__":
    app.run(debug=True)
