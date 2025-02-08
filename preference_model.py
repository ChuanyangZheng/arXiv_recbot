import re
import sqlite3
from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError
from telethon.tl.functions.messages import GetHistoryRequest
import asyncio
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import joblib
from arxiv_util import *

# Define the model
class PreferenceModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        hidden_dim = 512
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear3 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = self.relu(self.linear(x))
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x

def train_model(data):
    df = pd.DataFrame(data, columns=['text', 'preference'])

    # Display the first few rows
    print("Dataset Preview:")
    print(df.head(), "\n")

    # Features and Labels
    X = df['text']
    y = df['preference']

    # Split the data: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    '''
    # Data augmentation
    for d in data:
        aug_data.append({ "text": d["text"], "preference": min(d["preference"] + 1, 5)})
        aug_data.append({ "text": d["text"], "preference": d["preference"]})
        aug_data.append({ "text": d["text"], "preference": max(d["preference"] - 1, 0)})
    '''

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}\n")

    # Initialize the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer on the training data and transform
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # Transform the testing data
    X_test_tfidf = vectorizer.transform(X_test)

    # Initialize the PyTorch model
    # Convert sparse matrices to tensors
    X_train_tensor = torch.FloatTensor(X_train_tfidf.toarray())
    X_test_tensor = torch.FloatTensor(X_test_tfidf.toarray())
    y_train_tensor = torch.LongTensor(y_train.values)
    y_test_tensor = torch.LongTensor(y_test.values)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # We use the class weights to balance the samples
    class_weights = torch.tensor([(y_train == i).sum().item() for i in range(len(set(y_train)))])
    class_weights = 1.0 / class_weights
    class_weights = class_weights / class_weights.sum()  # Re-normalize

    model = PreferenceModel(X_train_tfidf.shape[1], len(set(y_train)))
    
    def create_smooth_target(y, num_classes=6):
        # Create a zero tensor of shape [batch_size, num_classes]
        batch_size = y.shape[0]
        targets = torch.zeros(batch_size, num_classes)
        
        # Get all labels as tensor
        labels = y.long()
        
        # Set main probability mass (0.6) for current labels
        targets.scatter_(1, labels.unsqueeze(1), 0.6)
        
        # Set 0.2 probability mass for labels-1 where valid
        valid_prev = (labels > 0).nonzero().squeeze()
        if valid_prev.numel() > 0:
            targets[valid_prev, labels[valid_prev]-1] = 0.2
        
        # Set 0.2 probability mass for labels+1 where valid
        valid_next = (labels < (num_classes-1)).nonzero().squeeze()
        if valid_next.numel() > 0:
            targets[valid_next, labels[valid_next]+1] = 0.2
        
        # Normalize each row
        targets = targets / targets.sum(dim=1, keepdim=True)
                
        return targets

    def weighted_kl_loss(pred, target, class_weights):
        # pred shape: [batch_size, num_classes]
        # target shape: [batch_size]
        smooth_targets = create_smooth_target(target)  # [batch_size, num_classes]
        kl_loss = nn.KLDivLoss(reduction='none')(pred.log_softmax(dim=1), smooth_targets)
        # Sum across class dimension
        kl_loss = kl_loss.sum(dim=1)  # [batch_size]
        # Apply class weights
        weight_per_sample = class_weights[target]  # [batch_size]
        weighted_loss = (kl_loss * weight_per_sample).mean()
        return weighted_loss

    criterion = lambda pred, target: weighted_kl_loss(pred, target, class_weights)
    optimizer = optim.Adam(model.parameters(), weight_decay=5e-5)

    # Train the model balancing the samples for each class
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).argmax(dim=1).numpy()
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy * 100:.2f}%\n")
        
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

    # Save the models
    torch.save(model.state_dict(), 'pytorch_preference_model.pt')
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

# Replace these with your own values
api_id = os.getenv('TG_API_ID')
api_hash = os.getenv('TG_API_HASH')

# You can set a session name or use None for an ephemeral session
session_name = "arxiv_retriever"

# The target chat from which to retrieve messages
# target_chat = int(os.environ["TELEGRAM_BOT_CHAT_ID"])
target_chat = os.environ["ArxivMan_bot"]

# Number of messages to retrieve
limit = 10000  

async def main():
    # Initialize the client
    client = TelegramClient(session_name, api_id, api_hash)

    await client.start()
    print("Client Created")

    # If you have 2FA enabled, you'll need to enter your password
    if not await client.is_user_authorized():
        phone = input("Enter your phone number (with country code): ")
        await client.send_code_request(phone)
        code = input("Enter the code you received: ")
        try:
            await client.sign_in(phone, code)
        except SessionPasswordNeededError:
            password = input("Two-Step Verification enabled. Please enter your password: ")
            await client.sign_in(password=password)

    # Get the target entity
    try:
        entity = await client.get_entity(target_chat)
    except ValueError:
        print(f"Could not find the chat: {target_chat}")
        return

    # Use regular expression to extract the label
    matcher = re.compile(r"Thank you for your feedback: (\w+)")
    label2class = { f"rating{i+1}" : i for i in range(6) }
    label2class.update({
        "not": 0,
        "thumb": 4,
        "love": 5,
    })

    # Load the message from the chat history and save it to a sqlite database
    conn = sqlite3.connect('arxiv_preference.db')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY, message_id INTEGER, text TEXT, preference INTEGER)')

    # Iterate over the chat history from the most recent message of the chat
    async for message in client.iter_messages(entity, limit=limit, reverse=False):
        # Check whether the message is a reply
        entry = None
        if message.reply_to_msg_id:
            # Get parent message
            parent_msg_id = message.reply_to_msg_id
            # get the content of the parent message
            parent_msg = await client.get_messages(target_chat, ids=parent_msg_id)

            # Extract label 
            # Example text: Thank you for your feedback: not
            # Use regular expression to extract the label
            m = matcher.search(message.text)
            if m:
                label = label2class.get(m.group(1), None)
                if label is not None:
                    # For parent_msg.txt, if the first line starts with "//", remove it
                    if parent_msg.text.startswith("//"):
                        parent_msg.text = "\n".join(parent_msg.text.split("\n")[1:])
                    entry = {
                        "message_id": message.id,
                        "text": parent_msg.text,
                        "preference": label
                    }
        elif message.sender.is_self and message.text is not None and message.text.startswith("https://arxiv.org"):
            # Put the paper in with default preference 
            results = get_arxiv_results(message.text.split("/")[-1], 1)
            entry = {
                "message_id": message.id,
                "text": get_arxiv_message(results[0]).replace("**", ""),
                "preference": 4
            }

        if entry is not None:
            # If there is no such message, insert it
            # otherwise we break the loop since all previous messages have been sent to the database
            # Check whether the message is already in the database
            cursor.execute('SELECT COUNT(*) FROM messages WHERE message_id = ?', (entry["message_id"],))
            if cursor.fetchone()[0] == 0:
                cursor.execute('INSERT INTO messages (message_id, text, preference) VALUES (?, ?, ?)', (entry["message_id"], entry["text"], entry["preference"]))
                conn.commit()
            else:
                break

    # Get all the messages from the database
    cursor.execute('SELECT text, preference FROM messages')
    data = cursor.fetchall()

    # Train the model
    train_model(data)

    await client.disconnect()

if __name__ == '__main__':
    # Ensure the event loop is properly handled
    asyncio.run(main())
