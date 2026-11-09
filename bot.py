import os
import discord
from dotenv import load_dotenv
from groq import Groq
from flask import Flask
import threading
import asyncio
import logging

# ------------------------------
# Load environment variables
# ------------------------------
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not DISCORD_TOKEN or not GROQ_API_KEY:
    raise RuntimeError("DISCORD_TOKEN or GROQ_API_KEY is missing!")

# ------------------------------
# Logging setup (for Render)
# ------------------------------
logging.basicConfig(level=logging.INFO)

# ------------------------------
# Groq client
# ------------------------------
client_ai = Groq(api_key=GROQ_API_KEY)

MODEL = "openai/gpt-oss-20b"

# ------------------------------
# Tiny Flask server for Render
# ------------------------------
app = Flask("bot")

@app.route("/")
def home():
    return "Bot is alive 💜"

def run_web():
    logging.info("Starting Flask web server on port 10000...")
    app.run(host="0.0.0.0", port=10000)

threading.Thread(target=run_web, daemon=True).start()

# ------------------------------
# Discord client setup
# ------------------------------
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# ------------------------------
# Memory dictionary per user
# ------------------------------
conversation_memory = {}

# ------------------------------
# Discord events
# ------------------------------
@client.event
async def on_ready():
    logging.info(f"Logged in as {client.user}")

def get_response(user_id, prompt):
    """Blocking function to get response from Groq API."""

    if user_id not in conversation_memory:
        conversation_memory[user_id] = []

    messages = [
        {
            "role": "system",
            "content": (
                "You are Crazylearner, a cool, smart, and approachable AI companion. "
                "Speak naturally like a friend who’s into tech, hacking, or learning stuff. "
                "Keep sentences short and casual, using contractions. "
                "Add one emoji naturally. "
                "Avoid hashtags, long motivational speeches, or overly formal phrasing. "
                "Make it feel like chatting with a clever, confident, and slightly edgy friend."
            )
        }
    ] + conversation_memory[user_id] + [{"role": "user", "content": prompt}]

    response = client_ai.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=300,
        temperature=0.6,
    )

    full_reply = response.choices[0].message.content or ""

    # Update memory
    conversation_memory[user_id].append({"role": "user", "content": prompt})
    conversation_memory[user_id].append({"role": "assistant", "content": full_reply})

    # Keep last 8 exchanges per user
    if len(conversation_memory[user_id]) > 16:
        conversation_memory[user_id] = conversation_memory[user_id][-16:]

    return full_reply

@client.event
async def on_message(message):
    # Ignore bot itself
    if message.author == client.user:
        return

    user_id = message.author.id
    prompt = message.content.strip()
    logging.info(f"{message.author} : {prompt}")

    if prompt.lower() == "!reset":
        conversation_memory[user_id] = []
        await message.channel.send("I've cleared our previous chats. Starting fresh! 💜")
        return

    if not prompt:
        return

    try:
        # Run blocking API call in thread to prevent freezing
        reply = await asyncio.to_thread(get_response, user_id, prompt)

        if len(reply) > 1900:
            reply = reply[:1900] + "..."

        await message.channel.send(reply)

    except Exception as e:
        logging.error(f"Error responding to {user_id}: {e}")
        await message.channel.send("Something went wrong while contacting my AI brain. Please try again 💜")

# ------------------------------
# Start Discord bot
# ------------------------------
client.run(DISCORD_TOKEN)
