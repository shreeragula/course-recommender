import os
import json
import urllib.request
import math

# 1. Predefined Skills Profiles mapping careers to their ordered required skills list
REQUIRED_SKILLS = {
    "AI Engineer": [
        "Python", "Mathematics", "Statistics", "Machine Learning", 
        "Deep Learning", "NLP", "PyTorch", "TensorFlow", "Docker", "AWS", "MLOps", "Deployment"
    ],
    "Data Analyst": [
        "Excel", "SQL", "Python", "Pandas", "NumPy", "Data Visualization", "Power BI", "Tableau", "Portfolio Project"
    ],
    "Data Scientist": [
        "Python", "SQL", "Statistics", "Linear Algebra", "Pandas", "NumPy", 
        "Machine Learning", "Scikit-Learn", "Deep Learning", "Tableau", "Communication"
    ],
    "ML Engineer": [
        "Python", "Software Engineering", "Statistics", "Machine Learning", 
        "PyTorch", "TensorFlow", "Docker", "Kubernetes", "AWS", "MLOps", "Model Serving"
    ],
    "Full Stack Developer": [
        "HTML", "CSS", "JavaScript", "React", "Node.js", "Express", 
        "MongoDB", "SQL", "Git", "Authentication", "Deployment", "Capstone Project"
    ],
    "Backend Developer": [
        "Python", "Java", "SQL", "Node.js", "Express", "PostgreSQL", 
        "MongoDB", "Git", "APIs", "Docker", "System Design"
    ],
    "Cloud Engineer": [
        "Linux", "Networking", "AWS", "Azure", "Terraform", "Docker", "Kubernetes", "Ansible", "CI/CD", "Deployment"
    ],
    "Cyber Security Analyst": [
        "Networking", "Security Protocols", "Linux", "Penetration Testing", 
        "Cryptography", "Firewalls", "Threat Hunting", "Compliance"
    ]
}

# 2. Comprehensive individual skills template database
SKILLS_DB = {
    "Excel": {
        "Title": "Excel Fundamentals",
        "Topics": ["Pivot Tables", "VLOOKUP", "Logical Formulas (IF/AND)"],
        "Difficulty": "Beginner",
        "Project": "Monthly Expense Tracker & Budgeter",
        "Course": "Excel Skills for Business (Coursera)",
        "BaseHours": 10
    },
    "SQL": {
        "Title": "SQL & Database Querying",
        "Topics": ["SELECT Queries", "JOIN operations", "GROUP BY & Aggregations", "Window Functions"],
        "Difficulty": "Beginner",
        "Project": "Library Database Schema Design",
        "Course": "SQL for Data Science (Coursera)",
        "BaseHours": 15
    },
    "Python": {
        "Title": "Python Programming",
        "Topics": ["Variables & Data Types", "Control Flows (Loops/Conditionals)", "Functions & OOP Basics"],
        "Difficulty": "Beginner",
        "Project": "Student Grade Management Tool",
        "Course": "Python for Everybody (Coursera)",
        "BaseHours": 20
    },
    "Pandas": {
        "Title": "Data Wrangling with Pandas",
        "Topics": ["DataFrames & Series", "Indexing & Filtering", "Grouping & Concatenations", "Handling Missing Data"],
        "Difficulty": "Intermediate",
        "Project": "Real Estate Data Cleaning Pipeline",
        "Course": "Data Manipulation with Pandas (Datacamp)",
        "BaseHours": 20
    },
    "NumPy": {
        "Title": "Numerical Computing with NumPy",
        "Topics": ["NDArrays", "Vectorized Array Operations", "Sorting & Slicing"],
        "Difficulty": "Intermediate",
        "Project": "Statistical Operations Array Library",
        "Course": "Introduction to NumPy (Datacamp)",
        "BaseHours": 15
    },
    "Data Visualization": {
        "Title": "Data Storytelling & Visualization",
        "Topics": ["Matplotlib plotting", "Seaborn styling", "Chart customizations"],
        "Difficulty": "Intermediate",
        "Project": "COVID-19 Analysis Plot Dashboard",
        "Course": "Data Visualization with Python (Coursera)",
        "BaseHours": 15
    },
    "Power BI": {
        "Title": "Business Intelligence with Power BI",
        "Topics": ["DAX Modeling", "Interactive Reports", "Relationship Mapping"],
        "Difficulty": "Intermediate",
        "Project": "E-Commerce Interactive Sales Dashboard",
        "Course": "Microsoft Power BI Essentials (Udemy)",
        "BaseHours": 20
    },
    "Tableau": {
        "Title": "Data Presentation with Tableau",
        "Topics": ["Worksheets & Stories", "Interactive Dashboard Filters", "Calculated Fields"],
        "Difficulty": "Intermediate",
        "Project": "Telecom Customer Churn Analytics Dashboard",
        "Course": "Tableau for Data Analysts (Coursera)",
        "BaseHours": 20
    },
    "Portfolio Project": {
        "Title": "Data Analyst Portfolio Project",
        "Topics": ["Data Cleaning", "Data Visualization", "Insights Summarization"],
        "Difficulty": "Advanced",
        "Project": "End-to-End Market Trends Report",
        "Course": "Google Data Analytics Capstone (Coursera)",
        "BaseHours": 30
    },
    "Mathematics": {
        "Title": "Mathematics for Artificial Intelligence",
        "Topics": ["Linear Algebra (Matrices)", "Calculus (Derivatives/Gradients)"],
        "Difficulty": "Intermediate",
        "Project": "Vector Transformations Library",
        "Course": "Mathematics for Machine Learning (Imperial College)",
        "BaseHours": 25
    },
    "Statistics": {
        "Title": "Probability & Statistical Foundations",
        "Topics": ["Probability Distributions", "Hypothesis Testing", "Correlation & Linear Regression"],
        "Difficulty": "Intermediate",
        "Project": "A/B Testing Simulator",
        "Course": "Introduction to Statistics (Stanford)",
        "BaseHours": 20
    },
    "Machine Learning": {
        "Title": "Machine Learning Algorithms",
        "Topics": ["Supervised learning (Regression/Classifiers)", "Unsupervised Clustering", "Scikit-Learn Model Selection"],
        "Difficulty": "Intermediate",
        "Project": "House Price Prediction Flask API",
        "Course": "Machine Learning Specialization (DeepLearning.AI)",
        "BaseHours": 35
    },
    "Deep Learning": {
        "Title": "Neural Networks & Deep Learning",
        "Topics": ["Multi-Layer Perceptrons", "Activation Functions", "Backpropagation optimization"],
        "Difficulty": "Advanced",
        "Project": "MNIST Digit Recognition Neural Network",
        "Course": "Deep Learning Specialization (DeepLearning.AI)",
        "BaseHours": 35
    },
    "NLP": {
        "Title": "Natural Language Processing",
        "Topics": ["Tokenization & Word Embeddings", "RNNs & Transformers"],
        "Difficulty": "Advanced",
        "Project": "SMS Spam/Ham Filtering App",
        "Course": "NLP Specialization (DeepLearning.AI)",
        "BaseHours": 30
    },
    "PyTorch": {
        "Title": "PyTorch for Deep Learning",
        "Topics": ["PyTorch Tensors", "nn.Module structures", "Dataset Loaders"],
        "Difficulty": "Intermediate",
        "Project": "Neural Network Image Classifier (CIFAR)",
        "Course": "PyTorch for Deep Learning (Udemy)",
        "BaseHours": 25
    },
    "TensorFlow": {
        "Title": "TensorFlow Core & Keras",
        "Topics": ["Keras Sequential API", "Functional API", "Keras layers Customizations"],
        "Difficulty": "Intermediate",
        "Project": "Object Detector Script",
        "Course": "TensorFlow Developer Certificate (Coursera)",
        "BaseHours": 25
    },
    "Docker": {
        "Title": "Containerization with Docker",
        "Topics": ["Dockerfiles & Image Building", "Container lifecycles", "Docker Compose"],
        "Difficulty": "Intermediate",
        "Project": "Containerized ML Prediction API",
        "Course": "Docker for Beginners (Udemy)",
        "BaseHours": 15
    },
    "AWS": {
        "Title": "Cloud Infrastructure with AWS",
        "Topics": ["EC2 Virtual Servers", "S3 Storage Buckets", "IAM Roles & Security"],
        "Difficulty": "Intermediate",
        "Project": "Flask App Hosting on AWS EC2",
        "Course": "AWS Certified Cloud Practitioner",
        "BaseHours": 20
    },
    "MLOps": {
        "Title": "MLOps Lifecycle Management",
        "Topics": ["MLflow Model Registries", "Data Version Control (DVC)", "CI/CD Pipelines for ML"],
        "Difficulty": "Advanced",
        "Project": "Continuous Model Validation Pipeline",
        "Course": "Machine Learning Engineering for Production (DeepLearning.AI)",
        "BaseHours": 30
    },
    "Deployment": {
        "Title": "Production Model Serving",
        "Topics": ["FastAPI server setups", "Gunicorn process handlers", "Cloud deployments (Render/Heroku)"],
        "Difficulty": "Advanced",
        "Project": "Production-Grade Prediction Endpoint",
        "Course": "Deploying Machine Learning Models (Udemy)",
        "BaseHours": 20
    },
    "Linear Algebra": {
        "Title": "Linear Algebra & Matrices",
        "Topics": ["Matrix Multiplications", "Eigenvectors & Eigenvalues", "PCA Dimension Reductions"],
        "Difficulty": "Intermediate",
        "Project": "Dimensionality Reduction Script from Scratch",
        "Course": "Linear Algebra (MIT OpenCourseWare)",
        "BaseHours": 20
    },
    "Scikit-Learn": {
        "Title": "Supervised Learning with Scikit-Learn",
        "Topics": ["Pre-processing Pipelines", "Hyperparameter Grid Searches", "Metrics Evaluation"],
        "Difficulty": "Intermediate",
        "Project": "Customer Segmentation Model",
        "Course": "Scikit-Learn Tutorial",
        "BaseHours": 20
    },
    "Communication": {
        "Title": "Technical Reporting & Presentation",
        "Topics": ["Data storytelling principles", "Slide Deck creation", "Executive Summaries"],
        "Difficulty": "Beginner",
        "Project": "Business Analytical Presentation Deck",
        "Course": "Storytelling with Data",
        "BaseHours": 10
    },
    "Software Engineering": {
        "Title": "Software Engineering for ML",
        "Topics": ["Clean Code guidelines", "Modular Object-Oriented design", "Unit Testing (pytest)"],
        "Difficulty": "Intermediate",
        "Project": "Refactored Modular ML Library Package",
        "Course": "Software Engineering for Data Scientists",
        "BaseHours": 25
    },
    "Kubernetes": {
        "Title": "Container Orchestration with Kubernetes",
        "Topics": ["K8s Pods & Services", "Model Deployment configurations", "Load balancing traffic"],
        "Difficulty": "Advanced",
        "Project": "Load-Balanced Prediction Service on K8s",
        "Course": "Certified Kubernetes Administrator (CKA)",
        "BaseHours": 30
    },
    "Model Serving": {
        "Title": "Model Inference Infrastructures",
        "Topics": ["Triton Inference Servers", "gRPC APIs", "High-throughput serving"],
        "Difficulty": "Advanced",
        "Project": "High-Performance Triton Serving Pipeline",
        "Course": "Serving Machine Learning Models",
        "BaseHours": 25
    },
    "HTML": {
        "Title": "HTML5 Web Structuring",
        "Topics": ["Semantic tags", "Forms, Inputs & Validations", "Document Object Models"],
        "Difficulty": "Beginner",
        "Project": "Static Portfolio Page Layout",
        "Course": "HTML Essentials (W3Schools)",
        "BaseHours": 10
    },
    "CSS": {
        "Title": "CSS3 Page Layout & Style",
        "Topics": ["Flexbox layouts", "CSS Grid designs", "Responsive media query viewports"],
        "Difficulty": "Beginner",
        "Project": "Responsive Landing Page Portfolio",
        "Course": "CSS Layouts (Udemy)",
        "BaseHours": 15
    },
    "JavaScript": {
        "Title": "JavaScript Programming",
        "Topics": ["DOM Manipulations", "Asynchronous fetch Requests", "Promise structures"],
        "Difficulty": "Beginner",
        "Project": "Dynamic Weather Dashboard App",
        "Course": "JavaScript - The Complete Guide (Udemy)",
        "BaseHours": 20
    },
    "React": {
        "Title": "Frontend Frameworks (React)",
        "Topics": ["React Components & State", "Hooks (useEffect/useContext)", "State management routing"],
        "Difficulty": "Intermediate",
        "Project": "Task Board Kanban Workspace",
        "Course": "React - The Complete Guide (Udemy)",
        "BaseHours": 30
    },
    "Node.js": {
        "Title": "Backend Servers with Node.js",
        "Topics": ["Event Loop structures", "File System read/write modules", "HTTP Core servers"],
        "Difficulty": "Intermediate",
        "Project": "Automated CLI Log Parser",
        "Course": "Node.js Developer Course (Udemy)",
        "BaseHours": 25
    },
    "Express": {
        "Title": "REST API Development with Express",
        "Topics": ["Express Router paths", "Middleware integrations", "JSON Request parsing"],
        "Difficulty": "Intermediate",
        "Project": "E-Commerce Backend REST API",
        "Course": "Express Essentials",
        "BaseHours": 20
    },
    "MongoDB": {
        "Title": "NoSQL Storage with MongoDB",
        "Topics": ["Document schemas", "CRUD Query functions", "Mongoose DB Connectors"],
        "Difficulty": "Intermediate",
        "Project": "Dynamic Directory Database",
        "Course": "MongoDB Complete Developer Guide",
        "BaseHours": 20
    },
    "Authentication": {
        "Title": "Secure Auth Systems",
        "Topics": ["JSON Web Tokens (JWT)", "Cryptographic Cookie Session guards", "Password encryption (BCrypt)"],
        "Difficulty": "Intermediate",
        "Project": "Secure Auth System Backend",
        "Course": "Web Security Essentials",
        "BaseHours": 20
    },
    "Capstone Project": {
        "Title": "Full-Stack Web App Capstone",
        "Topics": ["System designs", "Client-Server connections", "Live Deployments"],
        "Difficulty": "Advanced",
        "Project": "Interactive Full-Stack Booking Application",
        "Course": "Full Stack Web Developer Capstone",
        "BaseHours": 35
    },
    "Java": {
        "Title": "Java Foundations",
        "Topics": ["Basic variables", "Object-Oriented classes", "Error exceptions handling"],
        "Difficulty": "Beginner",
        "Project": "Local File Organizer Script",
        "Course": "Java Programming (Coursera)",
        "BaseHours": 25
    },
    "PostgreSQL": {
        "Title": "Relational Databases with PostgreSQL",
        "Topics": ["Tables Schema design", "Database indexing configurations", "ACID transactions"],
        "Difficulty": "Intermediate",
        "Project": "Inventory Management DB",
        "Course": "PostgreSQL Bootcamp (Udemy)",
        "BaseHours": 20
    },
    "APIs": {
        "Title": "API Designing",
        "Topics": ["HTTP methods", "Return status codes", "Endpoint designs"],
        "Difficulty": "Beginner",
        "Project": "Public Web Scraper API Server",
        "Course": "REST APIs Basics",
        "BaseHours": 15
    },
    "System Design": {
        "Title": "Distributed Systems Design",
        "Topics": ["In-memory Cache configurations (Redis)", "Load balancer algorithms", "Message brokers (RabbitMQ)"],
        "Difficulty": "Advanced",
        "Project": "High Availability Microservice Architecture Map",
        "Course": "System Design Interview (ByteByteGo)",
        "BaseHours": 30
    },
    "Linux": {
        "Title": "Linux administration",
        "Topics": ["Terminal CLI commands", "File permissions (chmod)", "Bash scripting automation"],
        "Difficulty": "Beginner",
        "Project": "Automated System Log Backup Bash script",
        "Course": "Linux Command Line (Udemy)",
        "BaseHours": 15
    },
    "Networking": {
        "Title": "Computer Networking",
        "Topics": ["TCP/IP protocols", "DNS settings", "IP addressing & subnets"],
        "Difficulty": "Beginner",
        "Project": "Home Network Design Configuration",
        "Course": "Google IT Support Networking",
        "BaseHours": 15
    },
    "Azure": {
        "Title": "Cloud Computing with Azure",
        "Topics": ["Azure VM virtual instances", "Azure Active Directory configs", "Storage Accounts"],
        "Difficulty": "Intermediate",
        "Project": "Load-Balanced Azure App Webserver",
        "Course": "Azure Fundamentals (AZ-900)",
        "BaseHours": 20
    },
    "Terraform": {
        "Title": "Infrastructure as Code with Terraform",
        "Topics": ["Infrastructure Provider setups", "Resource blocks", "Terraform state locks"],
        "Difficulty": "Intermediate",
        "Project": "Multi-resource Cloud Infrastructure deployment script",
        "Course": "Terraform Essentials (HashiCorp)",
        "BaseHours": 25
    },
    "Ansible": {
        "Title": "Configuration Automation with Ansible",
        "Topics": ["Playbook designs", "Ansible inventory variables", "Config modules"],
        "Difficulty": "Intermediate",
        "Project": "Automated Nginx Web Server configuration playbook",
        "Course": "Ansible for Beginners (Udemy)",
        "BaseHours": 20
    },
    "CI/CD": {
        "Title": "CI/CD Automations",
        "Topics": ["GitHub Actions workflows", "Continuous Testing configurations", "Deployment jobs"],
        "Difficulty": "Intermediate",
        "Project": "Continuous Testing Build Script",
        "Course": "CI/CD Pipelines (Udemy)",
        "BaseHours": 20
    },
    "Networks": {
        "Title": "Network Security Audits",
        "Topics": ["Wireshark traffic captures", "Port scanner checks", "VLAN partitioning"],
        "Difficulty": "Beginner",
        "Project": "Vulnerability Scan Audit Report",
        "Course": "Security Network Basics",
        "BaseHours": 15
    },
    "Security Protocols": {
        "Title": "Data Transmissions Security",
        "Topics": ["SSL certifications", "SSH Tunnel creations", "HTTPS server configurations"],
        "Difficulty": "Beginner",
        "Project": "Secure Apache Webserver installation",
        "Course": "Network Security Essentials",
        "BaseHours": 15
    },
    "Penetration Testing": {
        "Title": "Vulnerability Penetration Audits",
        "Topics": ["Metasploit exploits", "Nmap scans", "OWASP top 10 auditing"],
        "Difficulty": "Intermediate",
        "Project": "System Penetration Audit report",
        "Course": "Ethical Hacking (Coursera)",
        "BaseHours": 30
    },
    "Cryptography": {
        "Title": "Practical Cryptography",
        "Topics": ["Symmetric key algorithms (AES)", "Asymmetric keypairs (RSA)", "SHA hashing functions"],
        "Difficulty": "Intermediate",
        "Project": "Local File Encryption CLI App",
        "Course": "Cryptography Essentials",
        "BaseHours": 20
    },
    "Firewalls": {
        "Title": "Firewall configuration rules",
        "Topics": ["UFW configs", "IPTables rules", "Network ACL parameters"],
        "Difficulty": "Intermediate",
        "Project": "Access-Controlled Local Firewall configuration",
        "Course": "Firewall Operations",
        "BaseHours": 15
    },
    "Threat Hunting": {
        "Title": "Incident Analysis",
        "Topics": ["SIEM log captures", "Wireshark inspections", "IDS alert reviews"],
        "Difficulty": "Advanced",
        "Project": "Intruder Attack Log Investigation Report",
        "Course": "Incident Response (Coursera)",
        "BaseHours": 25
    },
    "Compliance": {
        "Title": "Security compliance",
        "Topics": ["ISO 27001 checklists", "SOC2 protocols", "GDPR audit items"],
        "Difficulty": "Intermediate",
        "Project": "Draft Security Compliance Policy Handbook",
        "Course": "Security Audits (Stanford)",
        "BaseHours": 15
    },
    "Git": {
        "Title": "Version Control with Git",
        "Topics": ["Commits, Branching & Pull Requests", "Merge Conflict Resolutions"],
        "Difficulty": "Beginner",
        "Project": "Collaborative GitHub Repository",
        "Course": "Git & GitHub (Coursera)",
        "BaseHours": 15
    }
}

def analyze_skill_gap(career_goal, current_skills):
    # Find matching required skills list
    career_goal_clean = career_goal.strip()
    required = None
    
    # Try finding exact or partial matching career in profiles
    for key, value in REQUIRED_SKILLS.items():
        if key.lower() in career_goal_clean.lower() or career_goal_clean.lower() in key.lower():
            required = value
            career_goal_clean = key
            break
            
    # Default fallback skills profile if no match is found
    if not required:
        required = REQUIRED_SKILLS["AI Engineer"]
        career_goal_clean = "AI Engineer"
        
    # Standardize and clean current skills list
    current_list = [s.strip() for s in current_skills.replace(",", ";").split(";") if s.strip()]
    current_clean = [s.lower() for s in current_list]
    
    known = []
    missing = []
    
    for skill in required:
        # Check if skill matches any user skills (case-insensitive substring check)
        is_known = False
        for user_s in current_clean:
            if user_s == skill.lower() or user_s in skill.lower() or skill.lower() in user_s:
                is_known = True
                break
        
        if is_known:
            known.append(skill)
        else:
            missing.append(skill)
            
    # Compute readiness score (percentage, min 5%)
    readiness = round((len(known) / len(required)) * 100) if required else 5
    readiness = max(5, min(100, readiness))
    
    return {
        "career": career_goal_clean,
        "required": required,
        "known": known,
        "missing": missing,
        "readiness": readiness
    }

def call_gemini_api(career_goal, current_skills, experience_level, learning_time):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
        
    # Analyze gaps to supply detailed parameters to prompt
    analysis = analyze_skill_gap(career_goal, current_skills)
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    
    prompt = f"""
    You are an expert AI Career Advisor.
    The user wants to become a: '{analysis["career"]}'.
    Their current known skills are: {json.dumps(analysis["known"])}.
    Their missing skills to learn are: {json.dumps(analysis["missing"])}.
    Their experience level is: '{experience_level}'.
    They can commit to studying: '{learning_time} hours per week'.

    Generate a highly structured learning roadmap to teach ONLY their missing skills. Do not include or repeat their already known skills in the roadmap.
    Organize the roadmap from beginner topics to advanced.
    Tailor the study duration of each stage dynamically assuming they study '{learning_time} hours per week'.

    Format your output exactly as a JSON string with the following schema:
    {{
      "career": "{analysis["career"]}",
      "estimated_duration": "Total study duration (e.g. 12 Weeks)",
      "career_readiness": {analysis["readiness"]},
      "skill_gap": {{
          "known": {json.dumps(analysis["known"])},
          "missing": {json.dumps(analysis["missing"])}
      }},
      "roadmap": [
          {{
              "stage": 1,
              "title": "Stage title targeting a missing skill",
              "skills": ["Subskill 1", "Subskill 2"],
              "duration": "e.g. 2 Weeks",
              "difficulty": "Beginner or Intermediate or Advanced",
              "project": "Name of one mini-project to build to practice these skills",
              "course": "Recommended course title (e.g. Python for Everybody)"
          }}
      ],
      "advice": "Personalized transition advice. Mention how they should budget their '{learning_time} hours/week' as a '{experience_level}' student."
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
            
            if raw_text.startswith("```"):
                lines = raw_text.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_text = "\n".join(lines).strip()
                
            return json.loads(raw_text)
    except Exception as e:
        print("Gemini API Error:", e)
        return None

def get_local_fallback(career_goal, current_skills, experience_level, learning_time):
    analysis = analyze_skill_gap(career_goal, current_skills)
    
    roadmap = []
    stage_num = 1
    total_weeks = 0
    
    hours_per_week = int(learning_time)
    
    # Process only missing skills
    for skill in analysis["missing"]:
        # Fetch template metadata
        template = SKILLS_DB.get(skill)
        if not template:
            # If no custom template exists, build placeholder
            template = {
                "Title": f"{skill} Advanced Concepts",
                "Topics": [f"Introduction to {skill}", f"Practical {skill} designs", f"{skill} best practices"],
                "Difficulty": "Intermediate",
                "Project": f"Personal {skill} Sandbox App",
                "Course": f"Complete {skill} Bootcamp (Udemy)",
                "BaseHours": 15
            }
            
        # Dynamically compute completion duration based on commitment
        weeks = math.ceil(template["BaseHours"] / hours_per_week)
        weeks = max(1, weeks) # Minimum 1 week
        
        roadmap.append({
            "stage": stage_num,
            "title": template["Title"],
            "skills": template["Topics"],
            "duration": f"{weeks} Weeks" if weeks > 1 else "1 Week",
            "difficulty": template["Difficulty"],
            "project": template["Project"],
            "course": template["Course"]
        })
        
        total_weeks += weeks
        stage_num += 1
        
    duration_str = f"{total_weeks} Weeks" if total_weeks > 1 else "1 Week"
    if not roadmap:
        duration_str = "0 Weeks"
        advice = "Congratulations! You possess all the core required skills for this role. Consider building complex portfolio applications or practicing advanced system design cases."
    else:
        # Build tailored learning advice
        time_text = ""
        if hours_per_week <= 5:
            time_text = f" Since you are balancing a {hours_per_week} hours/week commitment, keep study slots brief but daily (30-45 minutes)."
        else:
            time_text = f" With {hours_per_week} hours/week, structure weekend deep-work sessions to prototype the mini-projects."
            
        exp_text = ""
        if experience_level.lower() == "beginner" or experience_level.lower() == "student":
            exp_text = " Since you are starting out, verify code configurations step-by-step to avoid build failures."
        else:
            exp_text = " As a professional/intermediate, trace dependencies and reuse design paradigms to accelerate conceptual stages."
            
        advice = f"Focus strictly on learning the {len(analysis['missing'])} missing skills identified above. Follow the sequential vertical timeline.{time_text}{exp_text}"
        
    return {
        "career": analysis["career"],
        "estimated_duration": duration_str,
        "career_readiness": analysis["readiness"],
        "skill_gap": {
            "known": analysis["known"],
            "missing": analysis["missing"]
        },
        "roadmap": roadmap,
        "advice": advice
    }

def generate_roadmap(career_goal, current_skills, experience_level, learning_time):
    # Try calling the Gemini API first
    result = call_gemini_api(career_goal, current_skills, experience_level, learning_time)
    if result:
        return result
        
    # Fallback to local database if API key is missing or fails
    print("AI API unavailable. Using intelligent local rule-based advisor.")
    return get_local_fallback(career_goal, current_skills, experience_level, learning_time)
