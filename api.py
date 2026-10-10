import os
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# 1. Initialize FastAPI Application

app = FastAPI(
    title="PARAKH AI - ML Career Intelligence Service",
    description="Machine Learning Inference Service for Career Prediction, Clustering, and Skill Gap Analysis",
    version="1.0.0",
)

# Enable CORS for frontend and backend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 2. Configurations & Mappings

# 32 exact skill features fitted during training
SKILL_COLUMNS = [
    "python", "java", "c_cpp", "javascript", "typescript", "html_css",
    "react", "angular_vue", "nodejs", "fastapi_flask", "django", "sql",
    "nosql", "mongodb", "postgresql", "pandas_numpy", "scikit_learn",
    "deep_learning", "pytorch_tensorflow", "nlp", "computer_vision",
    "docker", "kubernetes", "git", "linux", "aws", "azure_gcp",
    "ci_cd", "terraform", "cybersecurity_basics", "penetration_testing",
    "flutter_react_native"
]

# K-Means Cluster Labels (k=2)
CLUSTER_NAMES = {
    0: "Web & Full-Stack Development",
    1: "Data Science, AI & Cloud Engineering"
}

# Career-Skill Requirements Mapping from skill_gap_analysis.ipynb
CAREER_SKILLS = {
    "Frontend Developer": [
        "html_css", "javascript", "react", "git"
    ],
    "Cloud Architect": [
        "python", "aws", "azure_gcp", "linux", "docker"
    ],
    "Data Analyst": [
        "python", "sql", "pandas_numpy"
    ],
    "Mobile App Developer": [
        "java", "flutter_react_native", "git"
    ],
    "Machine Learning Engineer": [
        "python", "pandas_numpy", "scikit_learn", "deep_learning"
    ],
    "AI Engineer": [
        "python", "deep_learning", "pytorch_tensorflow", "nlp", "computer_vision"
    ],
    "DevOps Engineer": [
        "linux", "git", "docker", "kubernetes", "ci_cd", "terraform"
    ],
    "Full Stack Developer": [
        "html_css", "javascript", "react", "nodejs", "sql", "git"
    ],
    "Cybersecurity Analyst": [
        "linux", "cybersecurity_basics", "penetration_testing", "python"
    ],
    "Database Administrator": [
        "sql", "nosql", "mongodb", "postgresql", "linux"
    ],
    "Backend Developer": [
        "python", "sql", "django", "fastapi_flask", "nodejs", "git"
    ],
    "Data Scientist": [
        "python", "sql", "pandas_numpy", "scikit_learn", "deep_learning"
    ]
}


# 3. Load Trained Artifacts

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_artifact(filename: str):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib.load(path)

try:
    scaler = load_artifact("standard_scaler.pkl")
    supervised_model = load_artifact("supervised_model.pkl")
    kmeans_model = load_artifact("kmeans_model.pkl")
    print("All ML artifacts loaded successfully.")
except Exception as e:
    print(f"Warning during artifact loading: {e}")
    scaler = None
    supervised_model = None
    kmeans_model = None


# 4. Request & Response Schemas

# Prediction
class CareerPredictRequest(BaseModel):
    skills: Dict[str, Any]

class CareerPredictResponse(BaseModel):
    career: str
    confidence: float

# Clustering
class ClusterRequest(BaseModel):
    skills: Dict[str, Any]

class ClusterResponse(BaseModel):
    cluster: int
    cluster_name: str

# Skill Gap Analysis
class SkillGapRequest(BaseModel):
    career: str
    skills: Dict[str, Any]

class SkillGapResponse(BaseModel):
    career: str
    match_score: float
    skills_you_have: List[str]
    skills_to_learn: List[str]


# Preprocessing the input 

def prepare_skill_dataframe(skills_dict: Dict[str, Any]) -> pd.DataFrame:
    row = {col: 1 if skills_dict.get(col, 0) in [1, True, "1"] else 0 for col in SKILL_COLUMNS}
    return pd.DataFrame([row], columns=SKILL_COLUMNS)


# Endpoints

@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "service": "PARAKH ML Inference Service",
        "docs_url": "/docs"
    }


@app.post("/predict", response_model=CareerPredictResponse)
def predict_career(payload: CareerPredictRequest):
    """Predicts suitable career category based on 32 student skills."""
    if supervised_model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Supervised model artifacts not loaded.")

    try:
        df_skills = prepare_skill_dataframe(payload.skills)
        scaled_skills = scaler.transform(df_skills)

        predicted_career = supervised_model.predict(scaled_skills)[0]
        probabilities = supervised_model.predict_proba(scaled_skills)[0]
        confidence = float(np.max(probabilities))

        return CareerPredictResponse(
            career=str(predicted_career),
            confidence=round(confidence, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/cluster", response_model=ClusterResponse)
def cluster_student(payload: ClusterRequest):
    """Assigns student to a skill cluster using K-Means (k=2)."""
    if kmeans_model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Clustering artifacts not loaded.")

    try:
        df_skills = prepare_skill_dataframe(payload.skills)
        scaled_skills = scaler.transform(df_skills)

        cluster_id = int(kmeans_model.predict(scaled_skills)[0])
        cluster_name = CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")

        return ClusterResponse(
            cluster=cluster_id,
            cluster_name=cluster_name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering error: {str(e)}")

@app.post("/skill-gap", response_model=SkillGapResponse)
def analyze_skill_gap(payload: SkillGapRequest):
    """
    Compares student skills against requirements of target career from skill_gap_analysis.ipynb.
    Returns match_score, skills_you_have, and skills_to_learn (gaps).
    """
    # Find matching career  key (case-insensitive)
    matched_career = next((c for c in CAREER_SKILLS if c.lower() == payload.career.lower()), None)
    if not matched_career:
        raise HTTPException(
            status_code=404,
            detail=f"Career '{payload.career}' not found. Available: {list(CAREER_SKILLS.keys())}"
        )

    required_skills = CAREER_SKILLS[matched_career]
    user_skills = payload.skills

    skills_have = [
        skill for skill in required_skills
        if user_skills.get(skill, 0) in [1, True, "1"]
    ]
    skills_to_learn = [
        skill for skill in required_skills
        if user_skills.get(skill, 0) not in [1, True, "1"]
    ]

    match_score = round((len(skills_have) / len(required_skills)) * 100, 2) if required_skills else 0.0

    return SkillGapResponse(
        career=matched_career,
        match_score=match_score,
        skills_you_have=skills_have,
        skills_to_learn=skills_to_learn
    )


# 7. Local Entrypoint

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    is_local = "PORT" not in os.environ
    print(f"Starting ML Service on http://127.0.0.1:{port}")
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=is_local)
