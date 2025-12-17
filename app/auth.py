import os
from supabase import create_client

_supabase = None

def get_supabase():
    global _supabase

    if os.getenv("DISABLE_SUPABASE") == "1":
        return None

    if _supabase is None:
        _supabase = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_ANON_KEY")
        )

    return _supabase
