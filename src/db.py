import sqlite3
import os
import json
import shutil

DB_PATH = "data/routes.db"

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    # Check if we need to migrate/reset
    # Simple approach: If schema is old, delete DB. 
    # For now, per user instruction: "clear the database when we udpate teh schema"
    # We will just force a fresh start if the new tables don't exist or if we want to enforce it.
    # To be safe, let's just drop the tables we know about and recreate.
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Routes table
    c.execute('''CREATE TABLE IF NOT EXISTS routes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # Steps table (ordered steps in a route)
    c.execute('''CREATE TABLE IF NOT EXISTS route_steps (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        route_id INTEGER,
        step_order INTEGER,
        name TEXT,
        FOREIGN KEY (route_id) REFERENCES routes (id)
    )''')
    
    # Images table (multiple images per step)
    c.execute('''CREATE TABLE IF NOT EXISTS step_images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        step_id INTEGER,
        filename TEXT,
        template_name TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (step_id) REFERENCES route_steps (id)
    )''')
    
    # Active Route state table
    c.execute('''CREATE TABLE IF NOT EXISTS active_state (
        key TEXT PRIMARY KEY,
        value TEXT
    )''')
    
    conn.commit()
    conn.close()

def save_route(name, steps_data):
    """
    steps_data: list of dicts. 
    Each dict represents a step: {
        'name': str, 
        'images': [ {'filename': str, 'name': str}, ... ] 
    }
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO routes (name) VALUES (?)", (name,))
        route_id = c.lastrowid
        
        for step_idx, step in enumerate(steps_data):
            # Create Step
            step_name = step.get('name', f"Step_{step_idx}")
            c.execute("INSERT INTO route_steps (route_id, step_order, name) VALUES (?, ?, ?)",
                      (route_id, step_idx, step_name))
            step_id = c.lastrowid
            
            # Insert Images
            for img in step.get('images', []):
                c.execute("INSERT INTO step_images (step_id, filename, template_name) VALUES (?, ?, ?)",
                          (step_id, img['filename'], img['name']))
        
        conn.commit()
        return True
    except sqlite3.IntegrityError as e:
        print(f"Error saving route '{name}': {e}")
        return False
    except Exception as e:
        print(f"Error saving route: {e}")
        return False
    finally:
        conn.close()

def update_route_structure(route_id, steps_data):
    """
    Replaces the entire step/image structure for a route.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        # Get existing step IDs to clean up images? 
        # Easier to just delete all steps for this route (cascade would be nice but manual is safer here)
        
        # 1. Find all steps for this route
        c.execute("SELECT id FROM route_steps WHERE route_id = ?", (route_id,))
        step_ids = [row[0] for row in c.fetchall()]
        
        if step_ids:
            # 2. Delete all images for these steps
            placeholders = ','.join('?' * len(step_ids))
            c.execute(f"DELETE FROM step_images WHERE step_id IN ({placeholders})", step_ids)
            
            # 3. Delete steps
            c.execute("DELETE FROM route_steps WHERE route_id = ?", (route_id,))
        
        # 4. Re-insert steps and images
        for step_idx, step in enumerate(steps_data):
            step_name = step.get('name', f"Step_{step_idx}")
            c.execute("INSERT INTO route_steps (route_id, step_order, name) VALUES (?, ?, ?)",
                      (route_id, step_idx, step_name))
            step_id = c.lastrowid
            
            for img in step.get('images', []):
                c.execute("INSERT INTO step_images (step_id, filename, template_name) VALUES (?, ?, ?)",
                          (step_id, img['filename'], img['name']))
                          
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
    
    # Get Route
    c.execute("SELECT name FROM routes WHERE id = ?", (route_id,))
    res = c.fetchone()
    if not res: return None
    route_name = res[0]
    
    # Get Steps
    c.execute("SELECT id, step_order, name FROM route_steps WHERE route_id = ? ORDER BY step_order ASC", (route_id,))
    step_rows = c.fetchall()
    
    steps = []
    for s_row in step_rows:
        step_id = s_row[0]
        step_order = s_row[1]
        step_name = s_row[2]
        
        # Get Images for this step
        c.execute("SELECT filename, template_name FROM step_images WHERE step_id = ?", (step_id,))
        img_rows = c.fetchall()
        images = [{"filename": r[0], "name": r[1]} for r in img_rows]
        
        steps.append({
            "name": step_name,
            "images": images
        })
        
    conn.close()
    return {"id": route_id, "name": route_name, "landmarks": steps}

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
    try:
        c.execute("SELECT value FROM active_state WHERE key = 'current_route'")
        res = c.fetchone()
        if res:
            return json.loads(res[0])
    except:
        pass
    finally:
        conn.close()
    return None

def clear_active_route():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM active_state WHERE key = 'current_route'")
    conn.commit()
    conn.close()
