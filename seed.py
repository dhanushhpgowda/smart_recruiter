from app import create_app, db
from app.models import Job, HR
from werkzeug.security import generate_password_hash

app = create_app()

with app.app_context():
    db.create_all()

    hr = HR(
        email="hr@test.com",
        password=generate_password_hash("password"),
        name="HR Admin"
    )
    db.session.add(hr)
    db.session.commit()

    job = Job(
        title="Python Developer",
        description="We need a Python developer with Flask and ML experience.",
        apply_link="job-1",
        hr_id=hr.id
    )
    db.session.add(job)
    db.session.commit()

    print("Test job created! Job ID:", job.id)