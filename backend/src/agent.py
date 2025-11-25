import logging
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, deepgram, google, silero, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("tutor-agent")

load_dotenv(".env.local")


# ---------- Shared session state ----------

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


# ---------- Worker setup (same pattern as Day 3) ----------

def prewarm(proc: JobProcess):
    # Load VAD once and reuse across sessions
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
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

    # One shared AgentSession with STT + LLM
    session = AgentSession[TutorUserData](
        userdata=userdata,
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Start in LEARN agent, which will greet, ask for mode, and then switch if needed
    await session.start(
        agent=learn_agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
