import logging

logger = logging.getLogger(__name__)


class CompletenessService:
    """
    Evaluates how complete a candidate's self-introduction is against all 9 semantic fields.

    Bug-fixed: previously read from semantic["detected"] (wrong key path — always empty).
    Now correctly reads from semantic["structured"] which is populated by SemanticService.

    Checks all 9 fields:
      - greetings, name, education, skills, strengths,
        areas_of_interest, qualities, experience, career_goals
    """

    # Severity levels: "required" = critical missing, "recommended" = nice to have
    FIELD_RULES = {
        "greetings":         {"label": "Greeting / Opening",      "severity": "recommended"},
        "name":              {"label": "Name introduction",        "severity": "required"},
        "education":         {"label": "Educational background",   "severity": "required"},
        "skills":            {"label": "Technical skills",         "severity": "required"},
        "strengths":         {"label": "Personal strengths",       "severity": "recommended"},
        "areas_of_interest": {"label": "Areas of interest",        "severity": "recommended"},
        "qualities":         {"label": "Personal qualities",       "severity": "recommended"},
        "experience":        {"label": "Work / project experience","severity": "recommended"},
        "career_goals":      {"label": "Career goals / aspirations","severity": "required"},
    }

    def check(self, text: str, semantic: dict) -> list:
        """
        Checks the structured semantic output for missing fields.

        Args:
            text:     Refined transcript text (unused here but kept for API consistency).
            semantic: Full SemanticService result dict with "structured" key.

        Returns:
            List of issue strings, e.g.
            ["[REQUIRED] Name introduction missing",
             "[RECOMMENDED] Personal strengths not mentioned"]
        """
        # ── Fix: read from "structured" not "detected" ───────────────────────────
        structured = semantic.get("structured", {})

        if not structured:
            logger.warning("⚠️ [CompletenessService] structured dict is empty — all fields flagged.")

        issues = []

        for field, rule in self.FIELD_RULES.items():
            value = structured.get(field)

            # Field is missing if: None, empty string, or empty list
            is_missing = (
                value is None
                or value == ""
                or value == []
            )

            if is_missing:
                severity_tag = f"[{rule['severity'].upper()}]"
                issue_msg = f"{severity_tag} {rule['label']} missing"
                issues.append(issue_msg)
                logger.debug(f"📋 [CompletenessService] {issue_msg}")

        logger.info(
            f"📊 [CompletenessService] Check complete: "
            f"{len(issues)} issues found out of {len(self.FIELD_RULES)} fields."
        )
        return issues