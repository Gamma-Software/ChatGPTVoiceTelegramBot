import os
from dotenv import load_dotenv
import openai
from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from elevenlabs import generate
from pydub import AudioSegment


async def on_messages(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """ Fonction de callback pour gérer les messages entrants """
    # Lance ChatGPT en asynchrone pour générer une réponse
    reponse = await conversation.arun(update.message.text)
    # Envoie la réponse
    await update.message.reply_text(reponse)


async def on_voices(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Gere les messages vocaux"""
    audio = await update.message.voice.get_file()
    await audio.download_to_drive("message_vocal.oga")

    # Transcription du message vocal
    with open("message_vocal.oga", "rb") as audio_file:
        transcript = openai.Audio.transcribe(
            "whisper-1",
            audio_file,
            api_key=os.environ["OPENAI_API_KEY"])

    # Lance ChatGPT en asynchrone pour générer une réponse
    reponse = await conversation.arun(transcript["text"])

    # Genere la voix
    audio = generate(
        text=reponse,
        voice="Bella",
        model="eleven_multilingual_v1"
    )

    # Enregistre la voix dans fichier temporaire
    with open("voix_genere.mp3", "wb") as f:
        f.write(audio)

    # Convertis la voix dans un format que Telegram accepte comme message vocal (.ogg)
    AudioSegment.from_file("voix_genere.mp3", "mp3").export(
        "voix_genere.ogg", format="ogg")
    await update.message.reply_voice("voix_genere.ogg")

    # Supprime les fichiers vocaux temporaires
    os.remove("message_vocal.oga")
    os.remove("voix_genere.mp3")
    os.remove("voix_genere.ogg")


def setup_conversation() -> LLMChain:
    # LLM
    llm = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"])

    # Prompt
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "You are a nice chatbot having a conversation with a human. Give simple answers no list or long sentences."
            ),
            # The `variable_name` here is what must align with memory
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
    )

    # Garde en memoire l'historique des messages
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Crée la chaine de conversation basee sur le model ChatGPT
    global conversation
    conversation = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory
    )


def main() -> None:
    # Charge les variables d'environnement
    load_dotenv()

    # Crée la chaine de conversation
    setup_conversation()

    # Crée l'application avec le token Telegram
    application = Application.builder().token(os.environ["TELEGRAM_TOKEN"]).build()

    # Gere les messages entrants (autre que des commandes)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_messages))
    application.add_handler(MessageHandler(filters.VOICE & ~filters.COMMAND, on_voices))

    # Fait tourner le bot à l'infini
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
