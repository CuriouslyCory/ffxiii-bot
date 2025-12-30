import sqlite3
import os
import json
import shutil

DB_PATH = "data/routes.db"

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Routes table
    # Check if 'type' column exists, if not we might need to recreate or add it.
    # Since we can just add it if missing:
    c.execute('''CREATE TABLE IF NOT EXISTS routes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        type TEXT DEFAULT 'LANDMARK', 
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # Check if 'type' column exists (for migration)
    try:
        c.execute("SELECT type FROM routes LIMIT 1")
    except sqlite3.OperationalError:
        print("Migrating routes table: Adding 'type' column.")
        c.execute("ALTER TABLE routes ADD COLUMN type TEXT DEFAULT 'LANDMARK'")

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
    
    # Keylog Events table
    c.execute('''CREATE TABLE IF NOT EXISTS keylog_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        route_id INTEGER,
        event_order INTEGER,
        time_offset REAL,
        event_type TEXT, -- 'down' or 'up'
        key_char TEXT,
        FOREIGN KEY (route_id) REFERENCES routes (id)
    )''')
    
    # Hybrid Nodes table
    c.execute('''CREATE TABLE IF NOT EXISTS hybrid_nodes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        route_id INTEGER,
        node_order INTEGER,
        timestamp REAL,
        minimap_path TEXT,
        main_view_path TEXT,
        input_intent TEXT, -- JSON list of keys
        FOREIGN KEY (route_id) REFERENCES routes (id)
    )''')

    # Active Route state table
    c.execute('''CREATE TABLE IF NOT EXISTS active_state (
        key TEXT PRIMARY KEY,
        value TEXT
    )''')
    
    conn.commit()
    conn.close()

def save_route(name, data, route_type="LANDMARK"):
    """
    Saves a route to the database.
    
    If route_type is "LANDMARK":
        data: list of dicts (steps). 
        Each dict: {'name': str, 'images': [{'filename': str, 'name': str}, ...]}
        
    If route_type is "KEYLOG":
        data: list of dicts (events).
        Each dict: {'time_offset': float, 'event_type': str, 'key': str}
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO routes (name, type) VALUES (?, ?)", (name, route_type))
        route_id = c.lastrowid
        
        if route_type == "LANDMARK":
            steps_data = data
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
        
        elif route_type == "KEYLOG":
            events = data
            for i, event in enumerate(events):
                c.execute("INSERT INTO keylog_events (route_id, event_order, time_offset, event_type, key_char) VALUES (?, ?, ?, ?, ?)",
                          (route_id, i, event['time_offset'], event['event_type'], event['key']))
        
        elif route_type == "HYBRID":
            nodes = data
            for i, node in enumerate(nodes):
                intent_json = json.dumps(node.get('input_intent', []))
                c.execute("INSERT INTO hybrid_nodes (route_id, node_order, timestamp, minimap_path, main_view_path, input_intent) VALUES (?, ?, ?, ?, ?, ?)",
                          (route_id, i, node['timestamp'], node['minimap_path'], node['main_view_path'], intent_json))

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
    Replaces the entire step/image structure for a LANDMARK route.
    Does not currently support updating KEYLOG routes.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        # Check type
        c.execute("SELECT type FROM routes WHERE id = ?", (route_id,))
        res = c.fetchone()
        if not res or res[0] != "LANDMARK":
            print(f"Cannot update structure for route {route_id} (Type: {res[0] if res else 'None'})")
            return False

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
    c.execute("SELECT name, type FROM routes WHERE id = ?", (route_id,))
    res = c.fetchone()
    if not res: return None
    route_name = res[0]
    route_type = res[1] if res[1] else "LANDMARK"
    
    result = {"id": route_id, "name": route_name, "type": route_type}
    
    if route_type == "LANDMARK":
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
        result["landmarks"] = steps
        
    elif route_type == "KEYLOG":
        c.execute("SELECT time_offset, event_type, key_char FROM keylog_events WHERE route_id = ? ORDER BY event_order ASC", (route_id,))
        rows = c.fetchall()
        events = [{"time_offset": r[0], "event_type": r[1], "key": r[2]} for r in rows]
        result["events"] = events

    elif route_type == "HYBRID":
        c.execute("SELECT timestamp, minimap_path, main_view_path, input_intent FROM hybrid_nodes WHERE route_id = ? ORDER BY node_order ASC", (route_id,))
        rows = c.fetchall()
        nodes = []
        for r in rows:
            nodes.append({
                "timestamp": r[0],
                "minimap_path": r[1],
                "main_view_path": r[2],
                "input_intent": json.loads(r[3]) if r[3] else []
            })
        result["nodes"] = nodes

    conn.close()
    return result

def list_routes(route_type=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    query = "SELECT id, name, created_at, type FROM routes"
    params = ()
    if route_type:
        query += " WHERE type = ?"
        params = (route_type,)
    
    query += " ORDER BY created_at DESC"
    
    c.execute(query, params)
    routes = c.fetchall()
    conn.close()
    return routes

def set_active_route(route_id, current_idx):
    # current_idx for LANDMARK is step index (int)
    # current_idx for KEYLOG could be float (time offset) or int (event index)?
    # Let's store it as is, caller handles meaning.
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
