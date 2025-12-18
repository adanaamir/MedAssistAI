import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("Attempting to import app.main...")
try:
    from app import main
    print("Import successful!")
except Exception as e:
    print(f"Import failed: {e}")
    sys.exit(1)
