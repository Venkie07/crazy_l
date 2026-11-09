# 💜 Crazylearner – AI-Powered Discord Chatbot

Crazylearner is an AI-powered Discord chatbot designed to make conversations more engaging, natural, and interactive. Powered by Groq's fast LLM inference, it acts as a friendly AI companion that can chat about technology, programming, learning, and everyday topics.

<a href="https://discord.com/users/1461679876321837157" target="_blank">
  <img src="https://img.shields.io/badge/💜_Try_Now_on_Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Try Now on Discord">
</a>


## ✨ Features

- 🤖 **AI-Powered Conversations:** Uses Groq API with the `openai/gpt-oss-20b` model.
- 💬 **Natural Responses:** Casual, friendly, and conversational communication.
- 🧠 **Conversation Memory:** Maintains separate conversation histories for individual Discord users.
- 🔄 **Reset Memory:** Use `!reset` to clear your conversation history and start fresh.
- ⚡ **Fast AI Responses:** Uses asynchronous execution to avoid blocking Discord event handling.
- 🌐 **Flask Health Endpoint:** Includes a lightweight web server for deployment health checks.
- 🚀 **Render Deployment:** Can be hosted on Render.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Discord.py | Discord bot integration |
| Groq API | AI inference |
| GPT-OSS-20B | Language model |
| Flask | Health-check web server |
| Python-dotenv | Environment variable management |
| Asyncio | Asynchronous execution |

## 📁 Project Structure

```text
Crazylearner/
│
├── bot.py              # Main Discord bot application
├── requirements.txt    # Python dependencies
├── .env                # Environment variables
├── .gitignore          # Git ignored files
└── README.md           # Project documentation
```

## ⚙️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Venkie07/crazy_l.git
cd Crazylearner
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root directory.

```env
DISCORD_TOKEN=your_discord_bot_token
GROQ_API_KEY=your_groq_api_key
```

Get your credentials from:

- Discord Developer Portal: https://discord.com/developers/applications
- Groq Console: https://console.groq.com/keys

**Important:** Never share your API keys or commit your `.env` file to GitHub.

### 5. Configure Discord Bot Intents

1. Open the Discord Developer Portal.
2. Select your bot application.
3. Navigate to **Bot** settings.
4. Enable **Message Content Intent**.
5. Save the changes.
6. Invite your bot to your Discord server with the required permissions to view and send messages.

### 6. Run the Bot

```bash
python bot.py
```

Once connected, the bot will appear online in your Discord server.

## 💬 Bot Commands

| Command | Description |
|---|---|
| `!reset` | Clears your conversation history |
| Normal message | Chat with Crazylearner |

Simply mention or send a message in a channel where the bot can read messages.

## 🧠 Conversation Memory

Crazylearner maintains conversation history separately for each Discord user.

- Each user has an independent conversation history.
- The bot remembers the last 8 exchanges per user.
- Older messages are automatically removed when the history limit is exceeded.
- Using `!reset` clears that user's conversation history.

The current implementation stores memory in a Python dictionary. This means conversation history is temporary and will be lost when the bot restarts.

## 🤖 AI Configuration

Crazylearner uses Groq's OpenAI-compatible chat completion interface through the official Groq Python SDK.

**Current configuration:**

| Setting | Value |
|---|---|
| Provider | Groq |
| Model | `openai/gpt-oss-20b` |
| Max Tokens | 300 |
| Temperature | 0.6 |
| Conversation History | Last 8 exchanges per user |

The AI is configured to behave like a friendly, knowledgeable, slightly edgy companion with an interest in technology, hacking, and learning.

## 📦 Requirements

The `requirements.txt` file should contain:

```text
discord.py
python-dotenv
groq
flask
```

## ☁️ Deployment on Render

Crazylearner includes a Flask health-check endpoint to support deployment on Render.

### Deployment Steps

1. Push your project to GitHub.
2. Log in to https://render.com/.
3. Create a new **Web Service**.
4. Connect your GitHub repository.
5. Configure the following settings:

| Setting | Value |
|---|---|
| Runtime | Python |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `python bot.py` |

6. Add the environment variables:

```env
DISCORD_TOKEN=your_discord_bot_token
GROQ_API_KEY=your_groq_api_key
```

7. Deploy the service.

The Flask server exposes a health endpoint:

```text
/
```

Expected response:

```text
Bot is alive 💜
```

The Flask server listens on port `10000` in the current implementation.

## 🔒 Security

- Store API credentials in environment variables.
- Never upload `.env` to a public repository.
- Add `.env` and `venv/` to `.gitignore`.
- Do not expose your Discord bot token or Groq API key in logs or messages.

## 🚧 Current Limitations

- Conversation memory is stored in RAM and does not persist across restarts.
- The bot currently responds to text messages.
- Conversation history is maintained separately for each Discord user.
- Responses are limited to 1,900 characters before being sent to Discord.
- No database-backed memory or permanent conversation storage is implemented.

## 🔮 Future Improvements

- Persistent conversation memory using SQLite.
- Slash commands for better Discord integration.
- Image and file understanding.
- Customizable AI personalities.
- Improved error handling and API usage monitoring.
- Streaming AI responses.
- Server-specific configuration.

## 📄 License

This project is open-source. You may use, modify, and extend it according to the license you choose for your repository.

---

**Made with 💜 and Python - By [Venkie](https://venkateswaran-k.web.app)**

*Crazylearner – Your friendly AI companion on Discord.*
