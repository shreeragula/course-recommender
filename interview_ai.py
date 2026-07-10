import os
import json
import random
import urllib.request

def call_gemini_question(role, current_skills, experience_level, interview_type, difficulty, past_questions=None):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    
    past_questions_str = json.dumps(past_questions) if past_questions else "None"
    
    prompt = f"""
    You are an expert technical interviewer.
    Generate ONE interview question for a candidate with the following profile:
    - Target Role: '{role}'
    - Current Skills: '{current_skills}'
    - Experience Level: '{experience_level}'
    - Interview Type: '{interview_type}' (Technical, Behavioral, or Mixed)
    - Target Difficulty: '{difficulty}' (Easy, Medium, Hard, or Any)

    Avoid repeating or asking duplicate questions to the following ones that were already asked in this session:
    {past_questions_str}

    Format your output exactly as a JSON string with the following schema:
    {{
      "question": "The interview question itself",
      "expected_answer": "A summary of the expected answer key points or ideal response",
      "difficulty": "Easy or Medium or Hard matching the target difficulty",
      "concepts": ["Concept 1", "Concept 2"]
    }}

    Return ONLY the raw JSON string. Do not wrap it in markdown codeblocks (e.g. do not use ```json).
    """
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    
    req = urllib.request.Request(
        url, 
        data=json.dumps(payload).encode('utf-8'), 
        headers=headers, 
        method='POST'
    )
    
    try:
        with urllib.request.urlopen(req, timeout=8) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            raw_text = res_data['candidates'][0]['content']['parts'][0]['text'].strip()
            
            # Strip markdown wrapping if model includes it
            if raw_text.startswith("```"):
                lines = raw_text.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_text = "\n".join(lines).strip()
                
            data = json.loads(raw_text)
            
            # Guarantee schema format
            return {
                "question": data.get("question", ""),
                "expected_answer": data.get("expected_answer", "No expected answer key points specified."),
                "difficulty": data.get("difficulty", difficulty if difficulty != "Any" else "Medium"),
                "concepts": data.get("concepts", [role, interview_type])
            }
    except Exception as e:
        print("Gemini Interview API Error:", e)
        return None

def get_local_fallback_question(role, current_skills, experience_level, interview_type, difficulty, past_questions=None):
    questions_file = "interview_questions.json"
    all_questions = []
    if os.path.exists(questions_file):
        with open(questions_file, "r", encoding="utf-8") as f:
            all_questions = json.load(f)
            
    filtered = []
    for q in all_questions:
        # Match Role
        if q["role"].lower() != role.lower():
            continue
        # Match Difficulty
        if difficulty != "Any" and q["difficulty"].lower() != difficulty.lower():
            continue
        # Match Interview Type
        if interview_type != "Mixed" and q["type"].lower() != interview_type.lower():
            continue
        filtered.append(q)
        
    # Fallback to general role if filters are too strict
    if not filtered:
        filtered = [q for q in all_questions if q["role"].lower() == role.lower()]
        
    # Filter out already asked questions to prevent repetitions
    if past_questions:
        unasked = [q for q in filtered if q["question"] not in past_questions]
        if unasked:
            filtered = unasked
            
    if not filtered:
        # If absolutely nothing remains, return a default mock question
        return {
            "question": f"Describe your past experience working with {role} technologies and how you handle project deadlines.",
            "expected_answer": "Discussion of relevant technologies, methodologies, and communication practices.",
            "difficulty": "Medium",
            "concepts": [role, "General Experience"]
        }
        
    selected = random.choice(filtered)
    
    return {
        "question": selected["question"],
        "expected_answer": "Refer to standard documentation or industry guidelines.",
        "difficulty": selected.get("difficulty", "Medium"),
        "concepts": [selected.get("topic", role)]
    }

def generate_interview_question(role, current_skills, experience_level, interview_type, difficulty, past_questions=None):
    # Try calling dynamic Gemini generator first
    result = call_gemini_question(role, current_skills, experience_level, interview_type, difficulty, past_questions)
    if result:
        return result
        
    # Local fallback
    print("Gemini Interview API unavailable. Using local JSON question fallback.")
    return get_local_fallback_question(role, current_skills, experience_level, interview_type, difficulty, past_questions)

def call_gemini_evaluation(question, expected_answer, user_answer):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    
    prompt = f"""
    You are an expert technical interviewer.
    Evaluate the candidate's response to the following interview question.

    - Question: "{question}"
    - Expected Answer Guideline: "{expected_answer}"
    - Candidate's Response Answer: "{user_answer}"

    Analyze the candidate's answer and compare it to the expected answer.
    Provide constructive feedback and assign an evaluation score out of 10.

    Format your output exactly as a JSON string with the following schema:
    {{
      "score": 8, // An integer score from 1 to 10
      "strengths": ["Strength 1 text", "Strength 2 text"], // A list of strengths or positive points in candidate's response
      "weaknesses": ["Weakness 1 text", "Weakness 2 text"], // A list of missing details, errors or weaknesses
      "correct_answer": "Complete detail correct answer guide explaining the conceptual details", // Comprehensive correct answer detail
      "tips": "Tips on how to best answer questions on this topic in a live interview" // Strategic advice
    }}

    Return ONLY the raw JSON string. Do not wrap it in markdown codeblocks (e.g. do not use ```json).
    """
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    
    req = urllib.request.Request(
        url, 
        data=json.dumps(payload).encode('utf-8'), 
        headers=headers, 
        method='POST'
    )
    
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            raw_text = res_data['candidates'][0]['content']['parts'][0]['text'].strip()
            
            # Strip markdown wrapping if model includes it
            if raw_text.startswith("```"):
                lines = raw_text.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_text = "\n".join(lines).strip()
                
            data = json.loads(raw_text)
            
            return {
                "score": int(data.get("score", 7)),
                "strengths": data.get("strengths", ["Answer was successfully submitted."]),
                "weaknesses": data.get("weaknesses", ["Some details could be expanded."]),
                "correct_answer": data.get("correct_answer", expected_answer),
                "tips": data.get("tips", "Be prepared to give real-world examples in your responses.")
            }
    except Exception as e:
        print("Gemini Evaluation API Error:", e)
        return None

def get_local_fallback_evaluation(question, expected_answer, user_answer):
    # Basic keyword matching heuristics
    score = 4
    words = user_answer.lower().split()
    
    if len(words) >= 10:
        score += 1
    if len(words) >= 25:
        score += 2
    if len(words) >= 50:
        score += 1
        
    # Check if candidate mentions key terms from expected answer or question
    expected_words = expected_answer.lower().split()
    matched_keywords = 0
    for w in expected_words:
        if len(w) > 4 and w in user_answer.lower():
            matched_keywords += 1
            if matched_keywords <= 3:
                score += 1
                
    score = min(9, max(1, score)) # cap local score between 1 and 9
    
    strengths = ["Your response directly addresses the question prompt."]
    if len(words) >= 20:
        strengths.append("Answer length indicates decent attempt detail.")
        
    weaknesses = ["Could include more specific architectural details or real-world application examples."]
    if len(words) < 15:
        weaknesses.append("Response is very brief; try to elaborate further on core concepts.")
        
    return {
        "score": score,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "correct_answer": expected_answer,
        "tips": "Always try to structure your answer using the STAR method (Situation, Task, Action, Result) for behavioral questions, or trace details systematically for technical ones."
    }

def evaluate_user_response(question, expected_answer, user_answer):
    # Try calling dynamic Gemini evaluator first
    result = call_gemini_evaluation(question, expected_answer, user_answer)
    if result:
        return result
        
    # Local fallback
    print("Gemini Evaluation API unavailable. Using local rule-based fallback.")
    return get_local_fallback_evaluation(question, expected_answer, user_answer)

def call_gemini_followup_question(role, previous_question, previous_answer, evaluation):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    
    prompt = f"""
    You are an expert technical interviewer.
    Based on the candidate's previous response and its evaluation, generate ONE intelligent follow-up question.
    
    - Target Role: '{role}'
    - Previous Question: '{previous_question}'
    - Candidate's Answer: '{previous_answer}'
    - Evaluation Weaknesses: '{evaluation.get("weaknesses", [])}'
    
    The follow-up should dig deeper into missing concepts, incorrect explanations, or ask for a specific scenario applying the concept.
    Keep the conversation natural, as if in a real technical interview.
    
    Format your output exactly as a JSON string with the following schema:
    {{
      "question": "The follow-up interview question itself",
      "expected_answer": "A summary of the expected answer key points or ideal response",
      "difficulty": "Medium",
      "concepts": ["Concept 1", "Concept 2"]
    }}

    Return ONLY the raw JSON string. Do not wrap it in markdown codeblocks (e.g. do not use ```json).
    """
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    
    req = urllib.request.Request(
        url, 
        data=json.dumps(payload).encode('utf-8'), 
        headers=headers, 
        method='POST'
    )
    
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            raw_text = res_data['candidates'][0]['content']['parts'][0]['text'].strip()
            
            # Strip markdown wrapping if model includes it
            if raw_text.startswith("```"):
                lines = raw_text.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_text = "\n".join(lines).strip()
                
            data = json.loads(raw_text)
            
            return {
                "question": data.get("question", ""),
                "expected_answer": data.get("expected_answer", "No expected answer specified."),
                "difficulty": data.get("difficulty", "Medium"),
                "concepts": data.get("concepts", ["Follow-up"])
            }
    except Exception as e:
        print("Gemini Follow-up API Error:", e)
        return None

def get_local_fallback_followup_question(previous_question):
    return {
        "question": f"Can you elaborate on your previous answer regarding: '{previous_question}'? Can you provide a real-world example?",
        "expected_answer": "The candidate should provide a concrete real-world example applying the concepts they discussed.",
        "difficulty": "Medium",
        "concepts": ["Application", "Real-world Example"]
    }

def generate_followup_question(role, previous_question, previous_answer, evaluation):
    # Try calling dynamic Gemini follow-up generator first
    result = call_gemini_followup_question(role, previous_question, previous_answer, evaluation)
    if result:
        return result
        
    # Local fallback
    print("Gemini Follow-up API unavailable. Using local fallback.")
    return get_local_fallback_followup_question(previous_question)

def call_gemini_final_report(role, experience_level, history_list):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    
    # Compress history for prompt
    history_summary = []
    for h in history_list:
        history_summary.append({
            "question": h["question"],
            "answer": h["answer"],
            "score": h["evaluation"].get("score", 0)
        })
        
    prompt = f"""
    You are an expert technical recruiter and AI Interviewer.
    Generate a final interview report for a '{experience_level}' '{role}' candidate based on their interview history.
    
    Interview History (Questions, Answers, and Scores):
    {json.dumps(history_summary)}
    
    Generate a detailed report containing:
    1. overall_score (1-100)
    2. technical_score (1-100)
    3. hr_score (1-100, behavioral/cultural fit)
    4. coding_score (1-100, syntax/logic)
    5. communication_score (1-100, clarity/conciseness)
    6. strengths (list of 3-5 key strengths)
    7. weak_areas (list of 3-5 weak areas)
    8. topics_to_revise (list of 3-5 specific technical concepts to study)
    9. recommended_resources (list of 2-3 general learning resources like books or websites)

    Format your output exactly as a JSON string with the following schema:
    {{
      "overall_score": 85,
      "technical_score": 82,
      "hr_score": 90,
      "coding_score": 80,
      "communication_score": 88,
      "strengths": ["Strength 1", "Strength 2"],
      "weak_areas": ["Weakness 1", "Weakness 2"],
      "topics_to_revise": ["Topic 1", "Topic 2"],
      "recommended_resources": ["Resource 1", "Resource 2"]
    }}

    Return ONLY the raw JSON string. Do not wrap it in markdown codeblocks.
    """
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    
    req = urllib.request.Request(
        url, 
        data=json.dumps(payload).encode('utf-8'), 
        headers=headers, 
        method='POST'
    )
    
    try:
        with urllib.request.urlopen(req, timeout=12) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            raw_text = res_data['candidates'][0]['content']['parts'][0]['text'].strip()
            
            # Strip markdown wrapping if model includes it
            if raw_text.startswith("```"):
                lines = raw_text.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_text = "\n".join(lines).strip()
                
            return json.loads(raw_text)
    except Exception as e:
        print("Gemini Final Report API Error:", e)
        return None

def get_local_fallback_final_report(history_list):
    # Calculate a rough average score
    avg_score = 0
    if history_list:
        total = sum([h["evaluation"].get("score", 7) for h in history_list])
        avg_score = int((total / len(history_list)) * 10)
    else:
        avg_score = 75
        
    return {
      "overall_score": avg_score,
      "technical_score": max(10, avg_score - 5),
      "hr_score": min(100, avg_score + 5),
      "coding_score": avg_score,
      "communication_score": min(100, avg_score + 10),
      "strengths": ["Good attempt at answering questions", "Completed the interview"],
      "weak_areas": ["Needs more specific real-world examples", "Technical depth could be improved"],
      "topics_to_revise": ["System Design Basics", "Core Programming Concepts", "Behavioral Frameworks"],
      "recommended_resources": ["Official Documentation", "Mock Interview Platforms"]
    }

def generate_final_report(role, experience_level, history_list):
    result = call_gemini_final_report(role, experience_level, history_list)
    if result:
        return result
    print("Gemini Final Report API unavailable. Using local fallback.")
    return get_local_fallback_final_report(history_list)
