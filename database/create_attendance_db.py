'''
This script is used once to create the SQLite database and its table for storing attendance data.
✅ What it does:
- Creates the attendance.db file (inside database/ folder)
- Creates a table named daily_attendance with the following columns: id, name, date, first_entry, last_exit

Think of this like setting up a digital filing cabinet where we'll store all attendance records.
The database is like a filing cabinet, and the table is like a specific drawer with labeled sections.
'''

import sqlite3  # Library to work with SQLite databases - SQLite is a simple database system
import os  # Library for file and folder operations

# Create database folder if it doesn't exist
'''
This creates a folder called 'database' to store our database file.
If the folder already exists, it won't create a new one (exist_ok=True means "it's okay if it exists").
Think of this as making sure we have a proper place to store our filing cabinet.
'''
os.makedirs('database', exist_ok=True)

# Connect to or create the SQLite database file
'''
This line does two things:
1. If attendance.db doesn't exist, it creates a new database file
2. If attendance.db already exists, it connects to it
Think of this as opening your filing cabinet to work with it.
'''
conn = sqlite3.connect('database/attendance.db')

# Create a cursor object to run SQL commands
'''
A cursor is like a pointer that helps us execute commands in the database.
Think of it as your hand that can write in the filing cabinet.
You need this cursor to create tables, insert data, or read data.
'''
cursor = conn.cursor()

# Create the 'daily_attendance' table to store only first entry and last exit
'''
This creates a table (like a drawer with specific sections) to store attendance data.
The table will have these columns (sections):
- name: The person's name (like "John Smith")
- date: The date in YYYY-MM-DD format (like "2024-12-25")
- first_entry: The time when person first entered (like "09:15:30")
- last_exit: The time when person last left (like "17:45:00")

IF NOT EXISTS means "only create this table if it doesn't already exist"
This prevents errors if we run this script multiple times.
'''
cursor.execute('''
    CREATE TABLE IF NOT EXISTS daily_attendance (
        name TEXT NOT NULL,                    -- Person's name (required field)
        date TEXT NOT NULL,                    -- Date (YYYY-MM-DD format, required field)
        first_entry TEXT,                      -- First entry time (HH:MM:SS format, optional)
        last_exit TEXT                         -- Last exit time (HH:MM:SS format, optional)
    )
''')

# Save and close
'''
conn.commit() saves all changes to the database file (like closing and saving a document)
conn.close() closes the connection to the database (like closing the filing cabinet)
It's important to always close the database connection when you're done.
'''
conn.commit()  # Save all changes to the database file
conn.close()   # Close the database connection

# Print success message to let user know everything worked
print("✅ daily_attendance table created.")

'''
IMPORTANT NOTES:
1. Run this script only once when setting up the system for the first time
2. If you run it again, it won't create duplicate tables (thanks to IF NOT EXISTS)
3. This creates the foundation for storing all attendance records
4. The database file will be created at: database/attendance.db
5. You can view the database using tools like DB Browser for SQLite if needed
'''