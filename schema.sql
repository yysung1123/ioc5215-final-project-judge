CREATE TABLE submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    status INTEGER DEFAULT 0,
    acc REAL DEFAULT 0,
    time REAL DEFAULT 0,
    error_msg TEXT,
    create_timestamp TIMESTAMP NOT NULL
)