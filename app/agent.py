import json
from groq import Groq
from config import Config

client = Groq(api_key=Config.GROQ_API_KEY)

# ─── Tool definitions for the agent ──────────────────────────────────────────

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_resumes",
            "description": "Search for relevant resumes in the database based on job description",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_description": {
                        "type": "string",
                        "description": "The job description to search against"
                    },
                    "job_id": {
                        "type": "integer",
                        "description": "The job ID to filter candidates"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of candidates to retrieve, default 20"
                    }
                },
                "required": ["job_description", "job_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_skill_gap",
            "description": "Analyze the skill gap between a candidate and job requirements",
            "parameters": {
                "type": "object",
                "properties": {
                    "candidate_skills": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of candidate skills"
                    },
                    "required_skills": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of required skills from job description"
                    }
                },
                "required": ["candidate_skills", "required_skills"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rank_candidates",
            "description": "Rank candidates based on their fit for the job",
            "parameters": {
                "type": "object",
                "properties": {
                    "candidates": {
                        "type": "array",
                        "description": "List of candidates with their details and scores",
                        "items": {"type": "object"}
                    },
                    "job_description": {
                        "type": "string",
                        "description": "The job description"
                    }
                },
                "required": ["candidates", "job_description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_interview_questions",
            "description": "Generate tailored interview questions for a candidate",
            "parameters": {
                "type": "object",
                "properties": {
                    "candidate_name": {
                        "type": "string",
                        "description": "Name of the candidate"
                    },
                    "candidate_skills": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Candidate skills"
                    },
                    "job_description": {
                        "type": "string",
                        "description": "The job description"
                    },
                    "skill_gaps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Skills the candidate is missing"
                    }
                },
                "required": ["candidate_name", "candidate_skills", "job_description"]
            }
        }
    }
]

# ─── Tool execution functions ─────────────────────────────────────────────────

def execute_tool(tool_name, tool_args, job_id=None):
    from app.services import get_candidates_with_scores

    if tool_name == "search_resumes":
        results = get_candidates_with_scores(
            tool_args["job_description"],
            tool_args.get("job_id", job_id),
            tool_args.get("top_k", 20)
        )
        candidates = []
        for r in results:
            candidates.append({
                "id": r["candidate"].id,
                "name": r["candidate"].name,
                "email": r["candidate"].email,
                "score": r["score"],
                "skills": r["parsed_data"].get("skills", []),
                "experience": r["parsed_data"].get("experience", []),
                "education": r["parsed_data"].get("education", []),
                "total_years_experience": r["parsed_data"].get("total_years_experience", 0),
                "summary": r["parsed_data"].get("summary", "")
            })
        return json.dumps(candidates)

    elif tool_name == "analyze_skill_gap":
        candidate_skills = set(s.lower() for s in tool_args["candidate_skills"])
        required_skills = set(s.lower() for s in tool_args["required_skills"])
        matched = list(candidate_skills & required_skills)
        missing = list(required_skills - candidate_skills)
        bonus = list(candidate_skills - required_skills)
        return json.dumps({
            "matched": matched,
            "missing": missing,
            "bonus": bonus,
            "match_percentage": round(len(matched) / len(required_skills) * 100, 1) if required_skills else 0
        })

    elif tool_name == "rank_candidates":
        candidates = tool_args["candidates"]
        jd = tool_args["job_description"]

        prompt = f"""
You are an expert HR manager. Rank these candidates for this job.

Job Description:
{jd}

Candidates:
{json.dumps(candidates, indent=2)}

Return ONLY a JSON array ranked from best to worst fit. Each item must have:
{{
    "id": candidate id,
    "name": candidate name,
    "email": candidate email,
    "rank": rank number starting from 1,
    "final_score": score out of 100,
    "reason": "2-3 sentence HR-style explanation why this rank",
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"]
}}

Think like an HR manager. Consider skills, experience, and overall fit.
Return ONLY the JSON array. No extra text.
"""
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=Config.GROQ_MODEL,
            messages=messages,
            temperature=0.2
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return raw

    elif tool_name == "generate_interview_questions":
        prompt = f"""
You are an expert HR interviewer. Generate 6 tailored interview questions for this candidate.

Candidate: {tool_args["candidate_name"]}
Skills: {tool_args["candidate_skills"]}
Skill Gaps: {tool_args.get("skill_gaps", [])}
Job Description: {tool_args["job_description"]}

Return ONLY a JSON array of 6 questions. Each item must have:
{{
    "question": "the interview question",
    "type": "technical/behavioral/situational",
    "reason": "why this question is relevant"
}}

Return ONLY the JSON array. No extra text.
"""
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=Config.GROQ_MODEL,
            messages=messages,
            temperature=0.3
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return raw

    return json.dumps({"error": "Unknown tool"})

# ─── Main agent loop ──────────────────────────────────────────────────────────

def run_agent(job_description, job_id):
    print(f"\nAgent starting for job {job_id}...")

    system_prompt = """You are an expert AI recruiter agent. Your job is to:
1. Search for candidates that match the job description
2. Analyze skill gaps for each candidate
3. Rank candidates based on their overall fit
4. Provide clear HR-style reasoning for your decisions

Always use the tools available to you. Think step by step like an experienced HR manager.
Be fair, objective, and focus on skills and experience."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Find and rank the best candidates for this job:\n\n{job_description}\n\nJob ID: {job_id}"}
    ]

    # agent loop
    while True:
        response = client.chat.completions.create(
            model=Config.GROQ_MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.1
        )

        message = response.choices[0].message
        messages.append({"role": "assistant", "content": message.content, "tool_calls": message.tool_calls})

        # if no tool calls agent is done
        if not message.tool_calls:
            print("Agent finished reasoning.")
            break

        # execute each tool call
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            print(f"Agent calling tool: {tool_name}")

            result = execute_tool(tool_name, tool_args, job_id)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

    # get final ranked results from last rank_candidates call
    for msg in reversed(messages):
        if msg.get("role") == "tool":
            try:
                data = json.loads(msg["content"])
                if isinstance(data, list) and len(data) > 0 and "rank" in data[0]:
                    return data
            except:
                continue

    return []