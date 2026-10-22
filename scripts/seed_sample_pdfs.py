"""Generate nine original multi-page sample resumes and a short out-of-domain paragraph."""

from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
RESUME_DIR = ROOT / "data" / "resumes"
OOD_PATH = ROOT / "data" / "ood" / "state_of_the_union.txt"

OOD_TEXT = (
    "This file is an original out-of-domain paragraph used only to prove the resume "
    "loader ignores non-PDF text. It is not a political address, not a resume, and not "
    "copied from any public speech. The ingestion pipeline must leave it out of the "
    "candidate index.\n"
)

PROFILES = [
    {
        "file_name": "01_alex_rivera.pdf",
        "name": "Alex Rivera",
        "role": "Senior Python Backend Engineer",
        "domain": "multi-tenant SaaS platforms",
        "skills": ["Python", "FastAPI", "PostgreSQL", "Redis", "AWS", "Docker", "pytest", "SQL"],
        "tools": ["Linux", "GitHub Actions", "CloudWatch", "SQLAlchemy"],
    },
    {
        "file_name": "02_blair_chen.pdf",
        "name": "Blair Chen",
        "role": "Machine Learning Engineer",
        "domain": "ranking and retrieval systems",
        "skills": ["Python", "PyTorch", "scikit-learn", "embeddings", "AWS", "MLflow", "pandas", "evaluation"],
        "tools": ["SageMaker", "feature stores", "Faiss", "Weights and Biases"],
    },
    {
        "file_name": "03_casey_okonkwo.pdf",
        "name": "Casey Okonkwo",
        "role": "DevOps Engineer",
        "domain": "platform reliability",
        "skills": ["Kubernetes", "Terraform", "AWS", "GitHub Actions", "Prometheus", "Linux", "Docker", "Python"],
        "tools": ["Helm", "Grafana", "IAM", "Terraform Cloud"],
    },
    {
        "file_name": "04_devon_patel.pdf",
        "name": "Devon Patel",
        "role": "Frontend Engineer",
        "domain": "product user interfaces",
        "skills": ["TypeScript", "React", "Next.js", "CSS", "accessibility", "GraphQL", "Jest", "design systems"],
        "tools": ["Figma", "Playwright", "Vite", "Storybook"],
    },
    {
        "file_name": "05_ellis_nguyen.pdf",
        "name": "Ellis Nguyen",
        "role": "Data Scientist",
        "domain": "growth experimentation",
        "skills": ["Python", "statistics", "experimentation", "SQL", "pandas", "scikit-learn", "visualization", "causal inference"],
        "tools": ["dbt", "BigQuery", "Jupyter", "Looker"],
    },
    {
        "file_name": "06_frankie_alvarez.pdf",
        "name": "Frankie Alvarez",
        "role": "Security Engineer",
        "domain": "application security",
        "skills": ["threat modeling", "AWS", "Python", "IAM", "vulnerability management", "logging", "incident response", "secure code review"],
        "tools": ["CloudTrail", "Burp Suite", "Semgrep", "Terraform"],
    },
    {
        "file_name": "07_gray_murphy.pdf",
        "name": "Gray Murphy",
        "role": "Full Stack Engineer",
        "domain": "B2B workflow products",
        "skills": ["TypeScript", "Python", "React", "PostgreSQL", "AWS", "Node.js", "testing", "API design"],
        "tools": ["FastAPI", "Prisma", "Playwright", "Docker"],
    },
    {
        "file_name": "08_harper_singh.pdf",
        "name": "Harper Singh",
        "role": "Data Engineer",
        "domain": "analytical data platforms",
        "skills": ["Spark", "Airflow", "SQL", "dbt", "AWS", "Python", "Kafka", "warehousing"],
        "tools": ["Redshift", "Great Expectations", "Terraform", "dbt Cloud"],
    },
    {
        "file_name": "09_indigo_brooks.pdf",
        "name": "Indigo Brooks",
        "role": "NLP Engineer",
        "domain": "search and retrieval augmented generation",
        "skills": ["Python", "transformers", "embeddings", "information retrieval", "evaluation", "AWS", "FastAPI", "spaCy"],
        "tools": ["sentence-transformers", "Chroma", "Ragas", "Docker"],
    },
]


def build_body(profile: dict) -> str:
    name = profile["name"]
    role = profile["role"]
    skills = profile["skills"]
    tools = profile["tools"]
    domain = profile["domain"]
    paragraphs = [
        f"{name}",
        role,
        (
            f"Professional summary. {name} is a {role} who ships production systems for {domain}. "
            f"Core skills include {', '.join(skills)}. Daily tools include {', '.join(tools)}. "
            f"The work spans design, implementation, testing, and operational handoff."
        ),
    ]
    for skill in skills:
        paragraphs.append(
            f"Selected work on {skill}. {name} designed, built, and operated {skill} capabilities "
            f"for {domain}. Responsibilities included technical design, code review, incident response, "
            f"mentoring, and measurable delivery. The environment used {', '.join(tools)} and required "
            f"clear writing, automated tests, and collaboration with product and security partners. "
            f"{name} documented tradeoffs and kept ownership attached to every long-running work item."
        )
    for year, company in (
        ("2024", "Northwind Labs"),
        ("2022", "Contoso Analytics"),
        ("2019", "Fabrikam Cloud"),
    ):
        paragraphs.append(
            f"Experience. {year} to present at {company} as {role}. "
            f"Delivered projects spanning {', '.join(skills[:4])} for {domain}. "
            f"Impact included faster releases, lower error rates, and clearer handoffs between teams. "
            f"Environment: {', '.join(tools)}."
        )
    paragraphs.append(
        "Education. B.S. Computer Science. This sample resume is original fictional content "
        "for an offline retrieval demonstration and is not a copy of any real curriculum vitae."
    )
    return "\n\n".join(paragraphs)


def write_pdf(path: Path, body: str) -> None:
    pdf = FPDF(format="Letter")
    pdf.set_margins(18, 18, 18)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 6, body)
    pdf.output(path)


def main() -> None:
    RESUME_DIR.mkdir(parents=True, exist_ok=True)
    OOD_PATH.parent.mkdir(parents=True, exist_ok=True)
    OOD_PATH.write_text(OOD_TEXT, encoding="utf-8")
    for profile in PROFILES:
        body = build_body(profile)
        if len(body) < 2500:
            raise RuntimeError(f"{profile['file_name']} is too short to split: {len(body)} chars")
        write_pdf(RESUME_DIR / profile["file_name"], body)
    written = sorted(RESUME_DIR.glob("*.pdf"))
    if len(written) != 9:
        raise RuntimeError(f"expected 9 PDFs, found {len(written)}")
    print(f"wrote {len(written)} resumes to {RESUME_DIR}")
    print(f"wrote {OOD_PATH.name}")


if __name__ == "__main__":
    main()
