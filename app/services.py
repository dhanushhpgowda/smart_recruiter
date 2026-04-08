import fitz
import docx
import json
from groq import Groq
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from config import Config

client = Groq(api_key=Config.GROQ_API_KEY)
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# ─── Extract raw text from PDF or DOCX ───────────────────────────────────────

def extract_text_from_pdf(file_path):
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        text += page.get_text()
    return text.strip()

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text.strip()

def extract_text(file_path):
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    else:
        return ""

# ─── Parse resume text into structured JSON using Groq ───────────────────────

def parse_resume(text):
    prompt = f"""
You are a resume parser. Extract information from the resume below and return ONLY a JSON object with these fields:

{{
    "name": "full name of candidate",
    "email": "email address",
    "phone": "phone number",
    "skills": ["skill1", "skill2", "skill3"],
    "experience": [
        {{
            "company": "company name",
            "role": "job title",
            "years": "duration"
        }}
    ],
    "education": [
        {{
            "degree": "degree name",
            "institution": "university/college name",
            "year": "graduation year"
        }}
    ],
    "total_years_experience": 0,
    "summary": "2 line professional summary"
}}

Return ONLY the JSON. No explanation. No extra text.

Resume:
{text}
"""

    messages = [
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model=Config.GROQ_MODEL,
        messages=messages,
        temperature=0.1
    )

    raw = response.choices[0].message.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw)

# ─── Resume quality score ─────────────────────────────────────────────────────

def score_resume_quality(parsed_data):
    score = 0
    tips = []

    if parsed_data.get("name"):
        score += 10
    if parsed_data.get("email"):
        score += 10
    if parsed_data.get("phone"):
        score += 10

    skills = parsed_data.get("skills", [])
    if len(skills) >= 5:
        score += 20
    elif len(skills) > 0:
        score += 10
        tips.append("Add more skills — aim for at least 5")

    experience = parsed_data.get("experience", [])
    if len(experience) >= 2:
        score += 25
    elif len(experience) == 1:
        score += 15
        tips.append("Add more work experience details")
    else:
        tips.append("No work experience found — add internships or projects")

    education = parsed_data.get("education", [])
    if len(education) >= 1:
        score += 15
    else:
        tips.append("Add your education details")

    if parsed_data.get("summary"):
        score += 10
    else:
        tips.append("Add a professional summary at the top")

    return {
        "score": score,
        "tips": tips
    }

# ─── Connect to Milvus ───────────────────────────────────────────────────────

def connect_milvus():
    connections.connect(
        alias="default",
        host=Config.MILVUS_HOST,
        port=Config.MILVUS_PORT
    )

# ─── Create Milvus collection ────────────────────────────────────────────────

def create_milvus_collection():
    connect_milvus()

    if utility.has_collection("resumes"):
        return Collection("resumes")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="candidate_id", dtype=DataType.INT64),
        FieldSchema(name="job_id", dtype=DataType.INT64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
    ]

    schema = CollectionSchema(fields=fields, description="Resume embeddings")
    collection = Collection(name="resumes", schema=schema)

    collection.create_index(
        field_name="embedding",
        index_params={
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
    )

    return collection

# ─── Embed text ───────────────────────────────────────────────────────────────

def embed_text(text):
    vector = embedding_model.encode(text)
    return vector.tolist()

# ─── Store resume embedding in Milvus ────────────────────────────────────────

def store_embedding(candidate_id, job_id, resume_text):
    collection = create_milvus_collection()
    collection.load()

    vector = embed_text(resume_text)

    data = [
        [candidate_id],
        [candidate_id],
        [job_id],
        [vector]
    ]

    collection.insert(data)
    collection.flush()
    print(f"Embedding stored for candidate {candidate_id}")

    # ─── Search resumes in Milvus ─────────────────────────────────────────────────

def search_resumes(job_description, job_id, top_k=20):
    connect_milvus()
    collection = Collection("resumes")
    collection.load()

    # embed the job description
    jd_vector = embed_text(job_description)

    # search milvus
    results = collection.search(
        data=[jd_vector],
        anns_field="embedding",
        param={
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        },
        limit=top_k,
        expr=f"job_id == {job_id}",
        output_fields=["candidate_id", "job_id"]
    )

    # get candidate ids and scores
    matches = []
    for hit in results[0]:
        matches.append({
            "candidate_id": hit.entity.get("candidate_id"),
            "score": round(hit.score * 100, 2)
        })

    return matches


# ─── Get full candidate details from PostgreSQL ───────────────────────────────

def get_candidates_with_scores(job_description, job_id, top_k=20):
    from app.models import Candidate

    # search milvus first
    matches = search_resumes(job_description, job_id, top_k)

    if not matches:
        return []

    # fetch full candidate data from postgres
    results = []
    for match in matches:
        candidate = Candidate.query.get(match["candidate_id"])
        if candidate:
            results.append({
                "candidate": candidate,
                "score": match["score"],
                "parsed_data": candidate.parsed_data
            })

    # sort by score highest first
    results.sort(key=lambda x: x["score"], reverse=True)

    return results