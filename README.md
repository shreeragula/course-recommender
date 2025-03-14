 

---

### **Altair RapidMiner Course Recommendation System**  

**Description**  
This repository contains a machine-learning-based **Course Recommendation System** that helps users find relevant online courses based on their preferences. The model is built using **Python, Pandas, Scikit-learn, and Natural Language Processing (NLP)** techniques.  

---

## **Features**  
- Uses **cosine similarity** for recommending similar courses.  
- Data preprocessing includes **text cleaning, stemming, and feature extraction**.  
- Flask-based **API** for deploying the recommendation system.  
- Dataset sourced from **Coursera** courses.  

---

## **Installation & Setup**  

1. **Clone the repository**  
   ```sh
   git clone https://github.com/shreeragula/Altair_RapidMiner_Project.git
   cd Altair_RapidMiner_Project
   ```

2. **Create a virtual environment (optional but recommended)**  
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies**  
   ```sh
   pip install -r requirements.txt
   ```

4. **Run the Flask app**  
   ```sh
   python app.py
   ```

5. **Access the API**  
   Open your browser and go to:  
   ```
   http://127.0.0.1:5000/
   ```

---

## **Usage**  

### **Course Recommendation via Python Script**  
You can use the following function from `ml_model.py` to get course recommendations:

```python
from ml_model import recommend_course

recommendations = recommend_course("Machine Learning")
for rec in recommendations:
    print(rec)
```

---

## **Project Structure**  
```
📦 Altair_RapidMiner_Project
 ┣ 📜 app.py            # Flask web API for course recommendations
 ┣ 📜 ml_model.py       # ML model implementation
 ┣ 📜 Coursera.csv      # Dataset with course details
 ┣ 📜 requirements.txt  # List of required Python packages
 ┣ 📜 sorce_code.ipynb  # Jupyter Notebook with model training
 ┗ 📜 README.md         # Project documentation
```

---

## **Technologies Used**  
- **Python (Flask, Pandas, Scikit-learn, NLTK, Matplotlib, Seaborn)**  
- **NLP Techniques (Stemming, Tokenization, TF-IDF Vectorization)**  
- **Machine Learning (Cosine Similarity-based Recommendation System)**  

---

## **Contributing**  
Feel free to contribute by submitting **pull requests** or **raising issues**.  

---

Let me know if you want to modify or add anything! 🚀
