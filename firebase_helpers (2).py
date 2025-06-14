import firebase_admin
from firebase_admin import credentials, firestore
import json
import os

def initialize_firebase():
    if not firebase_admin._apps:
        service_account_info = json.loads(os.getenv("FIREBASE_CRED_JSON"))
        cred = credentials.Certificate(service_account_info)
        firebase_admin.initialize_app(cred)
    return firestore.client()

def save_user_data(user_id, goal, tasks, rewards):
    db = initialize_firebase()
    db.collection("users").document(user_id).set({
        "goal": goal,
        "tasks": tasks,
        "rewards": rewards,
        "timestamp": firestore.SERVER_TIMESTAMP
    })

def load_user_data(user_id):
    db = initialize_firebase()
    doc = db.collection("users").document(user_id).get()
    if doc.exists:
        return doc.to_dict()
    return None
