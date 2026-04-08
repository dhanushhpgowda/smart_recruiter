import os
import uuid
from flask import Blueprint, request, jsonify, render_template, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash
from app import db
from app.models import Job, Candidate, HR
from app.services import extract_text, parse_resume, score_resume_quality, store_embedding
from config import Config

main = Blueprint("main", __name__)

def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS

# ─── Home ─────────────────────────────────────────────────────────────────────

@main.route("/")
def index():
    return redirect(url_for("main.hr_login"))

# ─── HR Login ─────────────────────────────────────────────────────────────────

@main.route("/hr/login", methods=["GET", "POST"])
def hr_login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        hr = HR.query.filter_by(email=email).first()
        if hr and check_password_hash(hr.password, password):
            session["hr_id"] = hr.id
            return redirect(url_for("main.hr_dashboard"))
        return render_template("login.html", error="Invalid email or password")
    return render_template("login.html")

# ─── HR Logout ────────────────────────────────────────────────────────────────

@main.route("/hr/logout")
def hr_logout():
    session.clear()
    return redirect(url_for("main.hr_login"))

# ─── HR Dashboard ─────────────────────────────────────────────────────────────

@main.route("/hr/dashboard")
def hr_dashboard():
    if "hr_id" not in session:
        return redirect(url_for("main.hr_login"))
    hr = HR.query.get(session["hr_id"])
    jobs = Job.query.filter_by(hr_id=hr.id).all()
    total_candidates = sum(len(job.candidates) for job in jobs)
    shortlisted = sum(1 for job in jobs for c in job.candidates if c.status == "shortlisted")
    return render_template("dashboard.html",
        hr=hr,
        jobs=jobs,
        total_jobs=len(jobs),
        total_candidates=total_candidates,
        shortlisted=shortlisted
    )

# ─── Create New Job ───────────────────────────────────────────────────────────

@main.route("/hr/jobs/new", methods=["GET", "POST"])
def new_job():
    if "hr_id" not in session:
        return redirect(url_for("main.hr_login"))
    apply_link = None
    if request.method == "POST":
        title = request.form.get("title")
        description = request.form.get("description")
        job_id = str(uuid.uuid4())[:8]
        job = Job(
            title=title,
            description=description,
            apply_link=job_id,
            hr_id=session["hr_id"]
        )
        db.session.add(job)
        db.session.commit()
        apply_link = f"http://127.0.0.1:5000/apply/{job.id}"
    return render_template("jobs.html", apply_link=apply_link)

# ─── View Candidates for a Job ────────────────────────────────────────────────

@main.route("/hr/jobs/<int:job_id>/candidates")
def job_candidates(job_id):
    if "hr_id" not in session:
        return redirect(url_for("main.hr_login"))
    job = Job.query.get_or_404(job_id)
    candidates = Candidate.query.filter_by(job_id=job_id).all()
    return render_template("candidates.html", job=job, candidates=candidates)

# ─── Run AI Match ─────────────────────────────────────────────────────────────

@main.route("/hr/jobs/<int:job_id>/match")
def match_candidates(job_id):
    if "hr_id" not in session:
        return redirect(url_for("main.hr_login"))
    job = Job.query.get_or_404(job_id)
    from app.agent import run_agent
    results = run_agent(job.description, job.id)
    return render_template("match.html", job=job, results=results)

# ─── Update Candidate Status ──────────────────────────────────────────────────

@main.route("/hr/candidates/<int:candidate_id>/status/<status>")
def update_status(candidate_id, status):
    if "hr_id" not in session:
        return redirect(url_for("main.hr_login"))
    candidate = Candidate.query.get_or_404(candidate_id)
    candidate.status = status
    db.session.commit()
    return redirect(request.referrer or url_for("main.hr_dashboard"))

# ─── Generate Interview Questions ─────────────────────────────────────────────

@main.route("/hr/candidates/<int:candidate_id>/questions")
def interview_questions(candidate_id):
    if "hr_id" not in session:
        return redirect(url_for("main.hr_login"))
    candidate = Candidate.query.get_or_404(candidate_id)
    job_id = request.args.get("job_id")
    job = Job.query.get_or_404(job_id)
    from app.agent import execute_tool
    import json
    result = execute_tool("generate_interview_questions", {
        "candidate_name": candidate.name,
        "candidate_skills": candidate.parsed_data.get("skills", []),
        "job_description": job.description,
        "skill_gaps": []
    })
    questions = json.loads(result)
    return render_template("interview.html", candidate=candidate, questions=questions)

# ─── Candidate Apply Page ─────────────────────────────────────────────────────

@main.route("/apply/<job_id>", methods=["GET", "POST"])
def apply(job_id):
    job = Job.query.get_or_404(job_id)
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        phone = request.form.get("phone")
        file = request.files.get("resume")
        if not file or not allowed_file(file.filename):
            return jsonify({"error": "Please upload a PDF or DOCX file"}), 400
        filename = secure_filename(file.filename)
        file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        file.save(file_path)
        raw_text = extract_text(file_path)
        parsed = parse_resume(raw_text)
        quality = score_resume_quality(parsed)
        candidate = Candidate(
            name=name,
            email=email,
            phone=phone,
            resume_path=file_path,
            resume_text=raw_text,
            parsed_data=parsed,
            job_id=job.id
        )
        db.session.add(candidate)
        db.session.commit()
        store_embedding(candidate.id, job.id, raw_text)
        return jsonify({
            "message": "Application submitted successfully!",
            "quality_score": quality["score"],
            "tips": quality["tips"]
        })
    return render_template("apply.html", job=job)