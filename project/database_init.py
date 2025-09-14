import sqlite3


schema = """
PRAGMA foreign_keys = ON;


CREATE TABLE IF NOT EXISTS users (
id INTEGER PRIMARY KEY AUTOINCREMENT,
name TEXT NOT NULL,
email TEXT UNIQUE NOT NULL,
password TEXT NOT NULL,
role TEXT NOT NULL DEFAULT 'user',
created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE IF NOT EXISTS logs (
id INTEGER PRIMARY KEY AUTOINCREMENT,
user_id INTEGER NOT NULL,
action TEXT NOT NULL,
details TEXT,
ip TEXT,
user_agent TEXT,
created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);
"""


if __name__ == "__main__":
    conn = sqlite3.connect("database.db")
    conn.executescript(schema)
    conn.commit()
    conn.close()
