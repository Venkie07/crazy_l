import os
import discord
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")

# Hugging Face OpenAI-compatible client
client_ai = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

# Discord client setup
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

MODEL = "meta-llama/Llama-3.1-8B-Instruct:novita"

# Memory dictionary to keep conversation per user
conversation_memory = {}  # key = user.id, value = list of messages

@client.event
async def on_ready():
    print(f"Logged in as {client.user}")

def get_response(user_id, prompt):
    """Gets full response from Hugging Face API (no streaming)."""
    
    if user_id not in conversation_memory:
        conversation_memory[user_id] = []

    # Build message history for context
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

    # Send request and get full response
    response = client_ai.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=300,
        temperature=0.6,
    )

    full_reply = response.choices[0].message.content

    # Update memory
    conversation_memory[user_id].append({"role": "user", "content": prompt})
    conversation_memory[user_id].append({"role": "assistant", "content": full_reply})

    # Limit memory to last 8 exchanges (16 messages)
    if len(conversation_memory[user_id]) > 16:
        conversation_memory[user_id] = conversation_memory[user_id][-16:]

    return full_reply

@client.event
async def on_message(message):
    # Ignore bot's own messages
    if message.author == client.user:
        return

    user_id = message.author.id
    prompt = message.content.strip()

    # Optional: Reset command
    if prompt.lower() == "!reset":
        conversation_memory[user_id] = []
        await message.channel.send("I've cleared our previous chats. Starting fresh! 💜")
        return

    if not prompt:
        return

    try:
        reply = get_response(user_id, prompt)
        if len(reply) > 1900:
            reply = reply[:1900] + "..."
        await message.channel.send(reply)
    except Exception as e:
        await message.channel.send(f"Error: {e}")

client.run(DISCORD_TOKEN)
