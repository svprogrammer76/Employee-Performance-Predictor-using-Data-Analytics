"""Rule-based coaching prompts; not model explanations or causal findings."""


def coaching_prompts(profile: dict) -> list[str]:
    prompts = []
    if float(profile.get("workload_score", 5)) >= 7:
        prompts.append("Discuss workload, priorities, and whether deadlines or resources need adjustment.")
    if float(profile.get("training_hours", 24)) < 12:
        prompts.append("Ask whether targeted learning time or a skills practice opportunity would help.")
    if float(profile.get("skills_score", 65)) < 55:
        prompts.append("Explore the employee's own development goals and useful coaching or mentoring.")
    if float(profile.get("attendance_rate", .94)) < .88:
        prompts.append("Check privately whether support or workplace barriers are affecting attendance; avoid assumptions.")
    if float(profile.get("manager_feedback_score", 3.4)) < 2.8:
        prompts.append("Review feedback for specificity, consistency, and actionable next steps with the employee.")
    if float(profile.get("projects_completed", 2)) < 1:
        prompts.append("Check role expectations, project access, and whether workload allocation is equitable.")
    if not prompts:
        prompts.append("Ask the employee what is helping their work and what support would make the next cycle successful.")
    return prompts
