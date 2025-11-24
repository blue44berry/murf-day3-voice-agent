import logging
import json
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
    metrics,
    tokenize,
)
from livekit.plugins import murf, deepgram, google, silero, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# -------------------------------------------------------------
# Wellness Logging Setup
# -------------------------------------------------------------

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
class WellnessEntry:
    timestamp: str
    mood: str
    energy: str
    goals: list[str] = field(default_factory=list)
    summary: str = ""

# -------------------------------------------------------------
# Wellness Check-in Task
# -------------------------------------------------------------

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

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def wellness_entrypoint(ctx: JobContext):
    ctx.log_context_fields = { "room": ctx.room.name }

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

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def on_metrics(ev):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        logger.info(f"Usage Summary: {usage_collector.get_summary()}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
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
