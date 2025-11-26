import json
import logging
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import json

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
 

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentTask,
    AgentSession,
    RunContext,
    function_tool,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,

    function_tool,
    RunContext,
    metrics,
    tokenize,
    function_tool,
    RunContext,

)
from livekit.plugins import murf, deepgram, google, silero, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

logger = logging.getLogger("tutor-agent")



WELLNESS_LOG = Path(__file__).parent / "wellness_log.json"

def load_history():
    if not WELLNESS_LOG.exists():
        return []
    try:
        with WELLNESS_LOG.open("r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def save_entry(entry_dict):
    history = load_history()
    history.append(entry_dict)
    with WELLNESS_LOG.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)




@dataclass
class TutorUserData:
    """State shared across all tutor agents in the session."""
    personas: Dict[str, Agent] = field(default_factory=dict)
    tutor_content: List[Dict[str, Any]] = field(default_factory=list)


def load_tutor_content() -> List[Dict[str, Any]]:
    """
    Load concepts from shared-data/day4_tutor_content.json.
    If the file is missing, fall back to a small built-in sample so it still runs.
    """
    content_path = Path(__file__).parent.parent / "shared-data" / "day4_tutor_content.json"

    if not content_path.exists():
        logger.warning("Tutor content file not found at %s, using fallback sample.", content_path)
        return [
            {
                "id": "variables",
                "title": "Variables",
                "summary": "Variables store values so you can reuse them later in your code.",
                "sample_question": "What is a variable and why is it useful?",
            },
            {
                "id": "loops",
                "title": "Loops",
                "summary": "Loops let you repeat an action multiple times without copy-pasting code.",
                "sample_question": "Explain the difference between a for loop and a while loop.",
            },
        ]

    with content_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_concepts_for_prompt(concepts: List[Dict[str, Any]]) -> str:
    """
    Turn JSON concepts into a readable bullet list for the LLM prompt.
    Only uses id, title, summary & sample_question.
    """
    lines: List[str] = []
    for c in concepts:
        cid = c.get("id", "")
        title = c.get("title", "")
        summary = c.get("summary", "")
        sample_q = c.get("sample_question", "")
        lines.append(
            f"- id: {cid}\n"
            f"  title: {title}\n"
            f"  summary: {summary}\n"
            f"  sample_question: {sample_q}\n"
        )
    return "\n".join(lines)


# ---------- Base agent with handoff helper ----------

class BaseTutorAgent(Agent):
    def __init__(self, *, instructions: str, tts: murf.TTS) -> None:
        super().__init__(instructions=instructions, tts=tts)

    async def on_enter(self) -> None:
        # Whenever this agent becomes active, it should talk to the user
        await self.session.generate_reply()

    async def _transfer_to_agent(self, name: str, context: RunContext) -> Agent:
        """Return another agent instance by key from shared userdata.personas."""
        userdata: TutorUserData = context.userdata
        next_agent = userdata.personas.get(name)
        if not next_agent:
            # Just in case config is wrong
            await self.session.say("Sorry, I couldn't switch modes properly. Please try again.")
            return self
        return next_agent


# ---------- LEARN mode agent (Murf Falcon: Matthew) ----------

class LearnAgent(BaseTutorAgent):
    def __init__(self, tutor_content: List[Dict[str, Any]]) -> None:
        concepts_block = format_concepts_for_prompt(tutor_content)

        instructions = f"""
You are the LEARN mode tutor in an Active Recall coaching system.

You are helping the learner understand programming concepts using the content below.

Available concepts (from JSON file):
{concepts_block}

Your behavior:
- First, greet the learner.
- Ask which learning mode they prefer: "learn", "quiz", or "teach_back".
- If they clearly choose **quiz** or **teach_back**, call the appropriate tool to switch modes immediately.
- If they choose **learn**, stay in this agent and proceed.

While in LEARN mode:
- Ask which concept they want to focus on (use the ids or titles).
- Use that concept's summary to explain it in simple language with short, concrete examples.
- Encourage questions and check for understanding.
- If they ask to switch to another mode later, call the appropriate transfer tool.
- Only use concepts that appear in the list above. Do NOT invent new concepts.
"""
        super().__init__(
            instructions=instructions,
            tts=murf.TTS(
                voice="en-US-matthew",   # Murf Falcon Matthew
                style="Conversation",
            ),
        )

    @function_tool()
    async def go_to_quiz_mode(self, context: RunContext) -> Agent:
        """Switch to QUIZ mode when the learner wants to be quizzed."""
        await self.session.say("Great, let's switch to quiz mode and test your understanding.")
        return await self._transfer_to_agent("quiz", context)

    @function_tool()
    async def go_to_teach_back_mode(self, context: RunContext) -> Agent:
        """Switch to TEACH-BACK mode when the learner wants to explain the concept themselves."""
        await self.session.say("Awesome, let's switch to teach-back mode so you can explain it in your own words.")
        return await self._transfer_to_agent("teach_back", context)


# ---------- QUIZ mode agent (Murf Falcon: Alicia) ----------

class QuizAgent(BaseTutorAgent):
    def __init__(self, tutor_content: List[Dict[str, Any]]) -> None:
        concepts_block = format_concepts_for_prompt(tutor_content)

        instructions = f"""
You are the QUIZ mode tutor in an Active Recall coaching system.

Use the concepts and sample questions from the content below to quiz the learner.

Concepts:
{concepts_block}

Behavior:
- Ask the learner which concept they want to be quizzed on.
- Use that concept's sample_question to start, then ask follow-up questions at a similar difficulty level.
- Always wait for the learner's answer before giving feedback.
- Give short, encouraging feedback: what they got right, and one thing to improve.
- If they seem stuck, gently hint using the concept summary (not the exact text).

Mode switching:
- If they say they want to "learn" again, call go_to_learn_mode.
- If they say they want to "teach back" or "explain it themselves", call go_to_teach_back_mode.
"""
        super().__init__(
            instructions=instructions,
            tts=murf.TTS(
                voice="en-US-alicia",   # Murf Falcon Alicia
                style="Conversation",
            ),
        )

    @function_tool()
    async def go_to_learn_mode(self, context: RunContext) -> Agent:
        """Switch back to LEARN mode for more explanations."""
        await self.session.say("No problem, let's go back to learning mode and review the concept together.")
        return await self._transfer_to_agent("learn", context)

    @function_tool()
    async def go_to_teach_back_mode(self, context: RunContext) -> Agent:
        """Switch to TEACH-BACK mode when the learner wants to explain the concept back."""
        await self.session.say("Nice, let's move to teach-back mode so you can explain it to me.")
        return await self._transfer_to_agent("teach_back", context)


# ---------- TEACH-BACK mode agent (Murf Falcon: Ken) ----------

class TeachBackAgent(BaseTutorAgent):
    def __init__(self, tutor_content: List[Dict[str, Any]]) -> None:
        concepts_block = format_concepts_for_prompt(tutor_content)

        instructions = f"""
You are the TEACH-BACK mode tutor in an Active Recall coaching system.

Your job is to get the learner to explain concepts back to you in their own words,
and then give short qualitative feedback.

Content:
{concepts_block}

Behavior:
- Ask which concept they'd like to teach back (variables, loops, etc.).
- Ask them to explain it as if they are teaching a friend.
- Listen to their explanation (you will receive it as text).
- Compare it mentally to the summary for that concept and give brief feedback:
  - What they did well.
  - One or two missing or unclear points.
- Encourage them and suggest if they should review or try a quiz next.

Mode switching:
- If they say they want to "learn" again, call go_to_learn_mode.
- If they say they want to be "quizzed", call go_to_quiz_mode.
"""
        super().__init__(
            instructions=instructions,
            tts=murf.TTS(
                voice="en-US-ken",   # Murf Falcon Ken
                style="Conversation",
            ),
        )

    @function_tool()
    async def go_to_learn_mode(self, context: RunContext) -> Agent:
        """Switch back to LEARN mode for more explanations."""
        await self.session.say("Let's switch back to learn mode so we can review the concept again.")
        return await self._transfer_to_agent("learn", context)

    @function_tool()
    async def go_to_quiz_mode(self, context: RunContext) -> Agent:
        """Switch to QUIZ mode for follow-up questions."""
        await self.session.say("Great, let's switch to quiz mode and test your understanding.")
        return await self._transfer_to_agent("quiz", context)



@dataclass
class WellnessEntry:
    timestamp: str
    mood: str
    energy: str
    goals: list[str] = field(default_factory=list)
    summary: str = ""



class WellnessCheckInTask(AgentTask[dict]):
    def __init__(self, chat_ctx=None):
        super().__init__(
            instructions="""
You are a gentle, supportive daily wellness companion.
Your job is to collect:

1. Mood
2. Energy level
3. 1–3 goals for the day

Ask one thing at a time. Keep it soft and comforting.
Avoid medical language. If unclear, ask kindly.
After collecting all three, call the tools.
""",
            chat_ctx=chat_ctx,
        )
        self.data = {}

    def _check_complete(self):
        if all(k in self.data for k in ["mood", "energy", "goals"]):
            self.complete(self.data)
        else:
            self.session.generate_reply(
                instructions="Continue gently collecting missing mood, energy or goals."
            )

    @function_tool()
    async def set_mood(self, ctx: RunContext, mood: str):
        self.data["mood"] = mood
        self._check_complete()

    @function_tool()
    async def set_energy(self, ctx: RunContext, energy: str):
        self.data["energy"] = energy
        self._check_complete()

    @function_tool()
    async def add_goal(self, ctx: RunContext, goal: str):
        goals = self.data.setdefault("goals", [])
        goals.append(goal)
        self._check_complete()


# -------------------------------------------------------------
# Wellness Agent
# -------------------------------------------------------------

class WellnessAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
You are a warm, reflective wellness companion who talks like a caring daily coach.
Be encouraging, never clinical.
""",
        )

    async def on_enter(self):
        history = load_history()

        # Personalized greeting
        if history:
            last = history[-1]
            greeting = (
                f"Welcome back! Yesterday you said your mood was {last['mood']}. "
                f"How are you feeling today?"
            )
        else:
            greeting = "Hi! I'm here for your daily wellness check-in. How do you feel today?"

        await self.session.say(greeting)

        # Run check-in task
        result = await WellnessCheckInTask(chat_ctx=self.chat_ctx)

        # Save entry
        entry = WellnessEntry(
            timestamp=str(datetime.now()),
            mood=result["mood"],
            energy=result["energy"],
            goals=result["goals"],
            summary="Daily wellness check completed."
        )
        save_entry(asdict(entry))

        # Recap to user
        goals_text = ", ".join(entry.goals)
        recap = (
            f"Thanks for sharing. Mood: {entry.mood}, energy: {entry.energy}. "
            f"Your goals for today are: {goals_text}. "
            f"You're doing great — take it one step at a time."
        )

        await self.session.say(recap)


# -------------------------------------------------------------
# Pipeline and Entry Point
# -------------------------------------------------------------

logger = logging.getLogger("wellness_agent")
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
 def prewarm(proc: JobProcess):
    # Load VAD once and reuse across sessions
    proc.userdata["vad"] = silero.VAD.load()

async def wellness_entrypoint(ctx: JobContext):
    ctx.log_context_fields = { "room": ctx.room.name }


async def entrypoint(ctx: JobContext):
    # Load FAQ once per worker process / session
    faq_data = load_faq_file()
    company_name = faq_data.get("company", "Your Product")
    tagline = faq_data.get("tagline", "")

    session_state = SessionState(faq_data=faq_data)

    # Set up the voice pipeline (Deepgram STT + Gemini LLM + Murf Falcon TTS)
    session = AgentSession[SessionState] (
        userdata=session_state,
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
    # Load content from JSON
    tutor_content = load_tutor_content()

    # Shared userdata across all tutor agents
    userdata = TutorUserData(tutor_content=tutor_content)

    # Create the three personas / modes
    learn_agent = LearnAgent(tutor_content)
    quiz_agent = QuizAgent(tutor_content)
    teach_back_agent = TeachBackAgent(tutor_content)

    userdata.personas = {
        "learn": learn_agent,
        "quiz": quiz_agent,
        "teach_back": teach_back_agent,
      }

    session = AgentSession[TutorUserData](
        userdata=userdata,
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),

    session = AgentSession(
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


    await session.start(
        agent=learn_agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def on_metrics(ev):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        logger.info(f"Usage Summary: {usage_collector.get_summary()}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=SdrAgent(company_name=company_name, tagline=tagline),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        agent=WellnessAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()

async def entrypoint(ctx: JobContext):
    await wellness_entrypoint(ctx)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
