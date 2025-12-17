#!/usr/bin/env python3
"""
Send Discord notification for CI/CD pipeline status
Usage: python scripts/send_discord_notification.py [success|failure]
"""
import os
import sys
import requests
from datetime import datetime

def send_success_notification():
    webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
    if not webhook_url:
        print("No Discord webhook configured, skipping notification")
        return
    
    branch = os.getenv('GITHUB_REF_NAME', 'unknown')
    commit = os.getenv('GITHUB_SHA', 'unknown')[:7]
    actor = os.getenv('GITHUB_ACTOR', 'unknown')
    
    payload = {
        'embeds': [{
            'title': '✅ ML Pipeline Success',
            'description': 'Medical Assistant CI/CD pipeline completed successfully!',
            'color': 3066993,
            'fields': [
                {'name': '📦 Branch', 'value': branch, 'inline': True},
                {'name': '🔨 Commit', 'value': commit, 'inline': True},
                {'name': '👤 Author', 'value': actor, 'inline': True},
                {'name': '🕐 Timestamp', 'value': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'inline': False}
            ],
            'thumbnail': {'url': 'https://cdn-icons-png.flaticon.com/512/2833/2833921.png'}
        }],
        'username': 'ML Pipeline Bot'
    }
    
    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        if response.status_code == 204:
            print('✅ Discord notification sent successfully!')
        else:
            print(f'⚠️ Discord notification failed: {response.status_code}')
    except Exception as e:
        print(f'⚠️ Failed to send Discord notification: {e}')

def send_failure_notification():
    webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
    if not webhook_url:
        print("No Discord webhook configured, skipping notification")
        return
    
    branch = os.getenv('GITHUB_REF_NAME', 'unknown')
    commit = os.getenv('GITHUB_SHA', 'unknown')[:7]
    actor = os.getenv('GITHUB_ACTOR', 'unknown')
    repo = os.getenv('GITHUB_REPOSITORY', 'unknown')
    run_id = os.getenv('GITHUB_RUN_ID', 'unknown')
    workflow_url = f'https://github.com/{repo}/actions/runs/{run_id}'
    
    payload = {
        'embeds': [{
            'title': '❌ ML Pipeline Failed',
            'description': 'Medical Assistant CI/CD pipeline encountered an error.',
            'color': 15158332,
            'fields': [
                {'name': '📦 Branch', 'value': branch, 'inline': True},
                {'name': '🔨 Commit', 'value': commit, 'inline': True},
                {'name': '👤 Author', 'value': actor, 'inline': True},
                {'name': '🔗 Workflow', 'value': workflow_url, 'inline': False},
                {'name': '🕐 Timestamp', 'value': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'inline': False}
            ]
        }],
        'username': 'ML Pipeline Bot'
    }
    
    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        if response.status_code == 204:
            print('✅ Discord notification sent successfully!')
        else:
            print(f'⚠️ Discord notification failed: {response.status_code}')
    except Exception as e:
        print(f'⚠️ Failed to send Discord notification: {e}')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/send_discord_notification.py [success|failure]")
        sys.exit(1)
    
    status = sys.argv[1].lower()
    
    if status == 'success':
        send_success_notification()
    elif status == 'failure':
        send_failure_notification()
    else:
        print(f"Unknown status: {status}")
        sys.exit(1)