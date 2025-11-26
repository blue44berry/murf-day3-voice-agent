import json
import logging
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# -------------------------------------------------------------------
# Basic setup
# -------------------------------------------------------------------

logger = logging.getLogger("agent")
load_dotenv(".env.local")

BASE_DIR = Path(__file__).resolve().parent.parent
SHARED_DATA_DIR = BASE_DIR / "shared-data"
FAQ_PATH = SHARED_DATA_DIR / "day5_company_faq.json"

LEADS_DIR = BASE_DIR / "leads"
LEADS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# Data models
# -------------------------------------------------------------------

@dataclass
class LeadInfo:
    name: Optional[str] = None
    company: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None
    use_case: Optional[str] = None
    team_size: Optional[str] = None
    timeline: Optional[str] = None  # now / soon / later

    def missing_fields(self) -> list[str]:
        missing = []
        if not self.name:
            missing.append("name")
        if not self.company:
            missing.append("company")
        if not self.email:
            missing.append("email")
        if not self.role:
            missing.append("role")
        if not self.use_case:
            missing.append("use_case")
        if not self.team_size:
            missing.append("team_size")
        if not self.timeline:
            missing.append("timeline")
        return missing

    def is_complete(self) -> bool:
        return len(self.missing_fields()) == 0


@dataclass
class SessionState:
    faq_data: dict = field(default_factory=dict)
    lead: LeadInfo = field(default_factory=LeadInfo)


RunCtx = RunContext[SessionState]

# -------------------------------------------------------------------
# Helpers: FAQ loading & search
# -------------------------------------------------------------------

def load_faq_file() -> dict:
    if not FAQ_PATH.exists():
        logger.warning("FAQ file not found at %s", FAQ_PATH)
        return {"company": "Your Product", "tagline": "", "faqs": []}
    try:
        with FAQ_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error("Failed to load FAQ JSON: %s", e)
        return {"company": "Your Product", "tagline": "", "faqs": []}


def find_best_faq_answer(question: str, faq_data: dict) -> str:
    """Very simple keyword-based FAQ search."""
    question_l = question.lower()
    faqs = faq_data.get("faqs", [])
    best = None
    best_score = 0

    for item in faqs:
        q_text = item.get("q", "").lower()
        score = 0
        for word in question_l.split():
            if len(word) > 2 and word in q_text:
                score += 1
        if score > best_score:
            best_score = score
            best = item

    if best and best.get("a"):
        return best["a"]

    return "I'm not fully sure about that. I might need a human teammate to confirm this detail."


def save_lead_to_json(lead: LeadInfo) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = LEADS_DIR / f"lead_{ts}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(lead), f, indent=2, ensure_ascii=False)
    return str(path)


def build_lead_summary(lead: LeadInfo, company_name: str) -> str:
    # fallback values so summary doesn't sound weird if something is missing
    name = lead.name or "there"
    comp = lead.company or "your company"
    role = lead.role or "your role"
    use_case = lead.use_case or "your use case"
    team_size = lead.team_size or "your team size"
    timeline = lead.timeline or "your approximate timeline"

    return (
        f"Thanks {name}! So you are from {comp}, working as {role}. "
        f"You want to use {company_name} for {use_case}, "
        f"with a team size of {team_size}, and your timeline is {timeline}. "
        "I'll share these details with the team."
    )

# -------------------------------------------------------------------
# SDR Agent
# -------------------------------------------------------------------

class SdrAgent(Agent):
    def __init__(self, company_name: str, tagline: str) -> None:
        # Main instructions for the SDR persona
        instructions = f"""
You are a warm, proactive Sales Development Representative (SDR) for {company_name}.
Tagline: {tagline}

Your goals:
- Greet the visitor warmly.
- Ask what brought them here and what they are working on.
- Understand their use case, team size, and rough timeline.
- Answer questions about the product, company, and pricing ONLY using the FAQ content.
- NEVER invent features or pricing that are not clearly in the FAQ.

Lead collection:
- Gradually and naturally collect these fields:
  - name
  - company
  - email
  - role
  - use_case
  - team_size
  - timeline (now / soon / later)
- Whenever the user provides one of these, immediately call the matching tool:
  - set_name, set_company, set_email, set_role, set_use_case, set_team_size, set_timeline.
- If the user seems done (they say things like "that's all", "I'm done", "thanks"), call finalize_lead.
- If finalize_lead tells you there are missing fields, politely ask only for the missing ones.

FAQ usage:
- When the user asks about {company_name}'s product, pricing, or who it is for,
  call the answer_faq tool with their question.
- Then rephrase the returned answer naturally in your own words.

Tone:
- Friendly, concise, and helpful.
- Keep answers short and clear, like a real SDR on a quick intro call.
"""
        super().__init__(instructions=instructions)

    async def on_enter(self) -> None:
        """Initial greeting when the call starts."""
        await self.session.generate_reply(
            instructions="""
Greet the user as an SDR. Briefly say what the company does in one line and ask:
1) What brought them here today?
2) What they are working on or trying to achieve.
"""
        )

    # ----------------- FAQ Tool -----------------

    @function_tool()
    async def answer_faq(self, context: RunCtx, question: str) -> str:
        """
        Use this tool to answer any question about the product, company, or pricing.
        Always pass the user's question here instead of guessing.
        """
        faq_data = context.userdata.faq_data
        answer = find_best_faq_answer(question, faq_data)
        return answer

    # ----------------- Lead Field Tools -----------------

    @function_tool()
    async def set_name(self, context: RunCtx, name: str) -> str:
        """Call this when you learn the lead's name."""
        context.userdata.lead.name = name
        return f"Recorded name as {name}."

    @function_tool()
    async def set_company(self, context: RunCtx, company: str) -> str:
        """Call this when you know the lead's company."""
        context.userdata.lead.company = company
        return f"Recorded company as {company}."

    @function_tool()
    async def set_email(self, context: RunCtx, email: str) -> str:
        """Call this when you know the lead's email address."""
        context.userdata.lead.email = email
        return f"Recorded email as {email}."

    @function_tool()
    async def set_role(self, context: RunCtx, role: str) -> str:
        """Call this when you know the lead's role (e.g. founder, CTO, product manager)."""
        context.userdata.lead.role = role
        return f"Recorded role as {role}."

    @function_tool()
    async def set_use_case(self, context: RunCtx, use_case: str) -> str:
        """Call this when the user explains what they want to use the product for."""
        context.userdata.lead.use_case = use_case
        return f"Recorded use case as: {use_case}"

    @function_tool()
    async def set_team_size(self, context: RunCtx, team_size: str) -> str:
        """Call this when you know the team size (e.g. 1-5, 10, 50+)."""
        context.userdata.lead.team_size = team_size
        return f"Recorded team size as {team_size}."

    @function_tool()
    async def set_timeline(self, context: RunCtx, timeline: str) -> str:
        """Call this when you know the lead's timeline (e.g. 'now', 'soon', 'later')."""
        context.userdata.lead.timeline = timeline
        return f"Recorded timeline as {timeline}."

    # ----------------- Finalization Tool -----------------

    @function_tool()
    async def finalize_lead(self, context: RunCtx) -> str:
        """
        Call this when the user is done with the conversation.
        This will:
        - Check for missing fields.
        - If complete, save the lead to a JSON file and return a summary.
        - If not complete, return which fields are missing so you can ask.
        """
        faq_data = context.userdata.faq_data
        company_name = faq_data.get("company", "our product")
        lead = context.userdata.lead

        missing = lead.missing_fields()
        if missing:
            # Ask the model to request only the missing fields
            missing_str = ", ".join(missing)
            return (
                f"The lead data is NOT complete. Missing fields: {missing_str}. "
                "Please ask the user for ONLY these missing details."
            )

        # All fields present → save to JSON and build a verbal summary
        path = save_lead_to_json(lead)
        summary = build_lead_summary(lead, company_name)
        logger.info("Lead saved to %s", path)

        return (
            f"Lead data is complete and has been saved to: {path}. "
            f"When you speak to the user, say this summary in natural language: {summary}"
        )

# -------------------------------------------------------------------
# LiveKit worker plumbing (same pattern as previous days)
# -------------------------------------------------------------------

def prewarm(proc: JobProcess):
    # Load VAD once and reuse
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Load FAQ once per worker process / session
    faq_data = load_faq_file()
    company_name = faq_data.get("company", "Your Product")
    tagline = faq_data.get("tagline", "")

    session_state = SessionState(faq_data=faq_data)

    # Set up the voice pipeline (Deepgram STT + Gemini LLM + Murf Falcon TTS)
    session = AgentSession[SessionState](
        userdata=session_state,
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=SdrAgent(company_name=company_name, tagline=tagline),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
