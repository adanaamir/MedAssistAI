import requests
from datetime import datetime

def send_discord_notification(webhook_url: str, success: bool, metrics: dict = None, error_msg: str = None):
    try:
        if success:
            embed = {
                "title": "✅ ML Pipeline Success",
                "description": "Medical Assistant model training completed successfully!",
                "color": 3066993,  # Green
                "fields": [
                    {
                        "name": "📊 Model Accuracies",
                        "value": (
                            f"• Naive Bayes: **{metrics['naive_bayes']['test_accuracy']:.4f}**\n"
                            f"• SVM Baseline: **{metrics['svm_baseline']['test_accuracy']:.4f}**\n"
                            f"• SVM + PCA: **{metrics['svm_pca']['test_accuracy']:.4f}**"
                        ),
                        "inline": False
                    },
                    {
                        "name": "🕒 Timestamp",
                        "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "inline": False
                    }
                ],
                "thumbnail": {
                    "url": "https://cdn-icons-png.flaticon.com/512/2833/2833921.png"
                }
            }
        else:
            embed = {
                "title": "❌ ML Pipeline Failed",
                "description": "Medical Assistant model training encountered an error.",
                "color": 15158332,  # Red
                "fields": [
                    {
                        "name": "Error Details",
                        "value": f"```{error_msg[:1000]}```",
                        "inline": False
                    },
                    {
                        "name": "🕒 Timestamp",
                        "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "inline": False
                    }
                ]
            }
        
        payload = {
            "embeds": [embed],
            "username": "ML Pipeline Bot"
        }
        
        response = requests.post(webhook_url, json=payload)
        
        if response.status_code == 204:
            print("✅ Discord notification sent successfully")
        else:
            print(f"⚠️ Discord notification failed: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️ Failed to send Discord notification: {str(e)}")