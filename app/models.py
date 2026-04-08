from app import db, login_manager
from flask_login import UserMixin
from datetime import datetime

@login_manager.user_loader
def load_user(user_id):
    return HR.query.get(int(user_id))

class HR(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(100))
    jobs = db.relationship("Job", backref="hr", lazy=True)

class Job(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    apply_link = db.Column(db.String(100), unique=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    hr_id = db.Column(db.Integer, db.ForeignKey("hr.id"), nullable=False)
    candidates = db.relationship("Candidate", backref="job", lazy=True)

class Candidate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    phone = db.Column(db.String(20))
    resume_path = db.Column(db.String(300))
    parsed_data = db.Column(db.JSON)
    resume_text = db.Column(db.Text)
    match_score = db.Column(db.Float)
    status = db.Column(db.String(50), default="applied")
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)
    job_id = db.Column(db.Integer, db.ForeignKey("job.id"), nullable=False)