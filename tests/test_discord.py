import os, sys
from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.notifications import send_discord_notification

load_dotenv()

webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

metrics = {
    "naive_bayes": {"test_accuracy": 0.92},
    "svm_baseline": {"test_accuracy": 0.90},
    "svm_pca": {"test_accuracy": 0.91}
}

send_discord_notification(webhook_url, success=True, metrics=metrics)
send_discord_notification(webhook_url, success=False, error_msg="This is a test error message")
