import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from easydict import EasyDict
from itertools import zip_longest
import joblib
import random
import argparse

import torch
import sqlite3

# Define your keywords
MAX_RESULTS = 100

# Telegram Bot Token and Chat ID (replace with your actual values)
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN_NOTIF_BOT"]  # Set your Telegram bot token as an environment variable
TELEGRAM_CHAT_ID = int(os.environ["TELEGRAM_BOT_CHAT_ID"]) # Replace with your chat ID

import logging
from datetime import datetime, timedelta, time, UTC
from arxiv_util import *
from hf_util import *
from preference_model import PreferenceModel
from common import *

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, CallbackQueryHandler, CommandHandler

if os.path.exists(global_model_name):
    vectorizer = joblib.load(global_vectorizer_name)
    loaded_model = PreferenceModel(vectorizer.get_feature_names_out().shape[0], 6)
    loaded_model.load_state_dict(torch.load(global_model_name))
    loaded_model.eval()
    print(f"Loaded {global_model_name} and {global_vectorizer_name}")
else:
    loaded_model = None

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

application = None  # Will hold the Telegram application instance

# Open the database
conn = sqlite3.connect(global_dataset_name)
cursor = conn.cursor()

async def fetch_and_send_papers(keywords, backdays, context: ContextTypes.DEFAULT_TYPE, get_hf:bool):
    results = get_arxiv_results(keywords.replace(",", " OR "), MAX_RESULTS)
    if get_hf:
        hf_results = get_hf_results(keywords.replace(",", " OR "), MAX_RESULTS)
        # 去重
        exist_dict = {}
        for idx, result in enumerate(results):
            cur_title_key = re.sub(r'[^a-zA-Z0-9]', '', result.title)
            exist_dict[cur_title_key] = idx
        new_hf_results = []
        for idx, result in enumerate(hf_results):
            cur_title_key = re.sub(r'[^a-zA-Z0-9]', '', result['title'])
            if cur_title_key in exist_dict.keys():
                exist_idx = exist_dict[cur_title_key]
                results[exist_idx].upvotes = result['paper']['upvotes']
            else:
                new_hf_results.append(result)
        # 交替合并两个list
        merged_results = []
        for x, y in zip_longest(results, new_hf_results):
            if x is not None:
                merged_results.append(x)
            if y is not None:
                merged_results.append(y)
        results = merged_results
    now = datetime.now(UTC)
    yesterday = now - timedelta(days=backdays)

    num_sent = 0

    papers_to_send = []
    for idx, result in enumerate(results):
        if type(result) is dict:  # hf result
            result['entry_id'] = f"http://arxiv.org/abs/{result['paper']['id']}"
            result = EasyDict(result)
            iso_time = result['publishedAt']
            submitted_date = datetime.fromisoformat(iso_time.replace('Z', '+00:00'))
            message = get_hf_message(result)
        else:
            submitted_date = result.updated
            # Make the submitted date timezone-aware by assuming UTC
            submitted_date = submitted_date.replace(tzinfo=UTC)
            message = get_arxiv_message(result)

        if submitted_date >= yesterday:
            if loaded_model:
                # Predict the class of the paper
                X = vectorizer.transform([message])
                X_tensor = torch.tensor(X.toarray(), dtype=torch.float32)
                prediction = loaded_model(X_tensor)
                # y_pred = prediction.argmax(dim=1).item()
                # Prepend predicted probabilities of all classes to output text
                y_pred_proba = prediction.softmax(dim=1).detach().cpu()
                y_pred_proba = y_pred_proba[0]

                # Compute an overall rating for the paper. 
                # The rating is a weighted sum of the predicted probabilities of all classes.
                # The weights are [0, 1, 2, ..], i.e. the rating is the sum of the predicted probabilities.
                overall_rating = torch.dot(y_pred_proba, torch.arange(y_pred_proba.shape[0]).float()).item()

                message = f"// {overall_rating} {y_pred_proba}\n{message}"
            else:
                # No model to load yet
                overall_rating = 0
                message = f"// no model yet\n{message}" 

            papers_to_send.append((overall_rating, message, result.entry_id))

    if len(papers_to_send) == 0:
        await context.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="No new papers found.")
        return

    # Sort papers_to_send by overall_rating in descending order
    papers_to_send.sort(key=lambda x: x[0], reverse=True)
    # Select the top 10 papers
    N = 10 if not get_hf else 20
    papers_to_send = papers_to_send[:N]
    try:
        await context.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text='-'*20)
        cur_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        await context.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=cur_time)
    except Exception as e:
        print(e)

    for mes_idx, (overall_rating, message, entry_id) in enumerate(papers_to_send):
        # Provide 5 level of rating for the paper.
        # Provide emoji for each level of rating.
        keys = ["👎", "2️⃣", "3️⃣", "4️⃣", "👍", "️❤️"]
        keyboard = [
            [
                InlineKeyboardButton(emoji, callback_data=f"rating{idx}_{entry_id}") for idx, emoji in enumerate(keys, 1)
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        message = message.split('**')[0] + f"**MessageIdx:** {mes_idx}\n" + "**".join(message.split('**')[1:])
        try:
            await context.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode="Markdown", reply_markup=reply_markup)
        except Exception as e:
            print(e)

async def feedback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    feedback_data = query.data
    feedback_type, entry_id = feedback_data.split('_', 1)

    # Collect feedback (here we just log it)
    logging.info(f"Received feedback: {feedback_type} for paper {entry_id} from user {update.effective_user.id}")

    await query.edit_message_reply_markup(reply_markup=None)
    # await query.message.reply_text(f"Thank you for your feedback: {feedback_type}")
    reply_message = f"Thank you for your feedback: {feedback_type}"
    await context.bot.send_message(chat_id=update.effective_user.id, text=reply_message, reply_to_message_id=query.message.message_id)

async def retrieve_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.callback_query:
        # Handle command case
        if not context.args:
            await update.message.reply_text("Please provide tags to search for. Usage: /get tag1 tag2 tag3")
            return
            
        tags = context.args
    else:
        # Handle callback query case
        query = update.callback_query
        await query.answer()
        data = query.data
        tags = data.split(' ')

    for tag in tags:
        # Retrieve the paper from the database that contains the tags
        cursor.execute('SELECT paper_message_id FROM comments WHERE comment LIKE ?', ('%' + tag + '%',))
        paper_message_ids = cursor.fetchall()

        # Get all papers that contain the tags and return
        papers = []

        for paper_message_id in paper_message_ids:
            # Retrieve the paper from the database
            cursor.execute('SELECT text FROM infos WHERE paper_message_id = ?', paper_message_id)
            for paper in cursor.fetchall():
                # Convert the paper to a string
                papers.append(str(paper[0]))

        # Return the papers
        if update.callback_query:
            await query.message.reply_text(f"For tag {tag}, the papers are the following: \n\n{'\n\n'.join(papers)}", parse_mode="Markdown")
        else:
            await update.message.reply_text(f"For tag {tag}, the papers are the following: \n\n{'\n\n'.join(papers)}", parse_mode="Markdown")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--first_backcheck_day', type=int, default=None)
    parser.add_argument("--keywords", type=str, default="reasoning,planning,preference,optimization,symbolic,grokking")
    parser.add_argument("--get_hf", action='store_false')

    args = parser.parse_args()
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CallbackQueryHandler(feedback_handler))
    application.add_handler(CommandHandler("get", retrieve_handler))
    application.add_handler(CallbackQueryHandler(retrieve_handler, pattern="^get"))
    run_once_fetch_func = lambda context: fetch_and_send_papers(args.keywords, args.first_backcheck_day, context, args.get_hf)
    run_daily_fetch_func = lambda context: fetch_and_send_papers(args.keywords, 2, context, args.get_hf)

    if args.first_backcheck_day is not None:
        application.job_queue.run_once(run_once_fetch_func, when=timedelta(seconds=1))
    application.job_queue.run_daily(run_daily_fetch_func, time(hour=0, minute=30))

    # Run the bot
    application.run_polling()

if __name__ == '__main__':
    main()
