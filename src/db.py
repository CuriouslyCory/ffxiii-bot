import sqlite3
import os
import json

DB_PATH = "data/routes.db"

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Routes table
    c.execute('''CREATE TABLE IF NOT EXISTS routes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # Landmarks table (ordered steps in a route)
    c.execute('''CREATE TABLE IF NOT EXISTS landmarks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        route_id INTEGER,
        step_order INTEGER,
        template_name TEXT,
        filename TEXT,
        FOREIGN KEY (route_id) REFERENCES routes (id)
    )''')
    
    # Active Route state table (for auto-resume)
    c.execute('''CREATE TABLE IF NOT EXISTS active_state (
        key TEXT PRIMARY KEY,
        value TEXT
    )''')
    
    conn.commit()
    conn.close()

def save_route(name, landmarks):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO routes (name) VALUES (?)", (name,))
        route_id = c.lastrowid
        
        for idx, lm in enumerate(landmarks):
            c.execute("INSERT INTO landmarks (route_id, step_order, template_name, filename) VALUES (?, ?, ?, ?)",
                      (route_id, idx, lm['name'], lm['filename']))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        print(f"Error: Route '{name}' already exists.")
        return False
    finally:
        conn.close()

def update_route_landmarks(route_id, landmarks):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        # Delete old landmarks
        c.execute("DELETE FROM landmarks WHERE route_id = ?", (route_id,))
        
        # Insert new ones
        for idx, lm in enumerate(landmarks):
            c.execute("INSERT INTO landmarks (route_id, step_order, template_name, filename) VALUES (?, ?, ?, ?)",
                      (route_id, idx, lm['name'], lm['filename']))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error updating route {route_id}: {e}")
        return False
    finally:
        conn.close()

def load_route(route_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("SELECT name FROM routes WHERE id = ?", (route_id,))
    res = c.fetchone()
    if not res: return None
    route_name = res[0]
    
    c.execute("SELECT template_name, filename FROM landmarks WHERE route_id = ? ORDER BY step_order ASC", (route_id,))
    rows = c.fetchall()
    
    landmarks = [{"name": r[0], "filename": r[1]} for r in rows]
    conn.close()
    return {"id": route_id, "name": route_name, "landmarks": landmarks}

def list_routes():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, created_at FROM routes ORDER BY created_at DESC")
    routes = c.fetchall()
    conn.close()
    return routes

def set_active_route(route_id, current_idx):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    state = json.dumps({"route_id": route_id, "current_idx": current_idx})
    c.execute("INSERT OR REPLACE INTO active_state (key, value) VALUES ('current_route', ?)", (state,))
    conn.commit()
    conn.close()

def get_active_route():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT value FROM active_state WHERE key = 'current_route'")
    res = c.fetchone()
    conn.close()
    if res:
        return json.loads(res[0])
    return None

def clear_active_route():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM active_state WHERE key = 'current_route'")
    conn.commit()
    conn.close()
