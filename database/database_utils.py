# Import necessary libraries for database operations, file handling, and time management
import sqlite3  # For working with SQLite database
import os       # For file and directory operations
import csv      # For reading and writing CSV files
from datetime import datetime  # For working with dates and times

# Define file paths - these are the locations where our database and CSV files are stored
DB_PATH = 'database/attendance.db'  # Path to our SQLite database file
CSV_PATH = 'attendance_log.csv'     # Path to our CSV log file

def log_appearance(name, date_str, time_str=None):
    """
    This function records every single time a person appears in front of the camera.
    Think of it like a detailed log book that writes down every time someone walks by.
    
    Parameters:
    - name: The person's name who appeared
    - date_str: The date when they appeared (format: YYYY-MM-DD)
    - time_str: The exact time they appeared (format: HH:MM:SS)
    """
    # If no time is provided, use the current time
    if time_str is None:
        time_str = datetime.now().strftime('%H:%M:%S')  # Get current time in HH:MM:SS format

    # Create an empty list to store all rows from the CSV file
    rows = []
    found = False  # Flag to track if we found an existing entry for this person and date

    '''
    Read the existing CSV file if it exists.
    This is like opening a notebook to see what's already written.
    '''
    if os.path.exists(CSV_PATH):  # Check if the CSV file exists
        with open(CSV_PATH, 'r', newline='') as f:  # Open file in read mode
            rows = list(csv.reader(f))  # Read all rows into our list

    '''
    If the CSV file is empty or doesn't exist, create a header row.
    This is like writing the column titles at the top of a new notebook page.
    '''
    if not rows:  # If no rows exist (empty file)
        rows.append(["Name", "Date", "Appearances..."])  # Add header row

    '''
    Look through existing rows to find if this person already has an entry for today.
    This is like scanning through the notebook to find if someone's name is already there.
    '''
    for row in rows[1:]:  # Skip the header row (index 0) and check all data rows
        if row[0] == name and row[1] == date_str:  # If we find matching name and date
            row.append(time_str)  # Add the new time to this row
            found = True  # Mark that we found an existing entry
            break  # Stop looking since we found what we needed

    '''
    If no existing entry was found, create a new row for this person and date.
    This is like starting a new line in the notebook for a new person.
    '''
    if not found:  # If we didn't find an existing entry
        rows.append([name, date_str, time_str])  # Create new row with name, date, and time

    '''
    Write all the data back to the CSV file.
    This is like saving our notebook after making changes.
    '''
    with open(CSV_PATH, 'w', newline='') as f:  # Open file in write mode
        csv.writer(f).writerows(rows)  # Write all rows back to the file

    return time_str  # Return the time that was logged

def log_attendance(name, date_str, time_str=None):
    """
    This is the main function that handles attendance logging.
    It does two things:
    1. Records every appearance in CSV (detailed log)
    2. Records only first entry and last exit in database (summary)
    
    Think of it like having both a detailed diary and a summary sheet.
    """
    # If no time is provided, use the current time
    if time_str is None:
        time_str = datetime.now().strftime('%H:%M:%S')  # Get current time
    
    # First, log this appearance in the detailed CSV file
    log_appearance(name, date_str, time_str)

    '''
    Now work with the database to maintain attendance summary.
    The database keeps track of when someone first arrived and when they last left.
    '''
    conn = sqlite3.connect(DB_PATH)  # Connect to the database
    cursor = conn.cursor()  # Create a cursor to execute SQL commands
    
    '''
    Check if this person already has an attendance record for today.
    This is like checking if their name is already in today's attendance sheet.
    '''
    cursor.execute("""
        SELECT * FROM daily_attendance 
        WHERE name = ? AND date = ?
    """, (name, date_str))  # Execute query with name and date as parameters
    
    existing_record = cursor.fetchone()  # Get the first matching record (if any)
    
    '''
    If a record already exists, this means the person has already entered today.
    So we update their exit time (last time they were seen).
    '''
    if existing_record:  # If we found an existing record
        # Update the last_exit time for this person
        cursor.execute("""
            UPDATE daily_attendance 
            SET last_exit = ? 
            WHERE name = ? AND date = ?
        """, (time_str, name, date_str))  # Update with new exit time
        print(f"[DB] Updated EXIT for {name} at {time_str}")  # Print confirmation
        action = "EXIT"  # Mark this as an exit action
    else:
        '''
        If no record exists, this is the first time we're seeing this person today.
        So we create a new entry with their first entry time.
        '''
        cursor.execute("""
            INSERT INTO daily_attendance (name, date, first_entry, last_exit)
            VALUES (?, ?, ?, NULL)
        """, (name, date_str, time_str))  # Insert new record with entry time
        print(f"[DB] Created ENTRY for {name} at {time_str}")  # Print confirmation
        action = "ENTRY"  # Mark this as an entry action
    
    '''
    Save all changes to the database and close the connection.
    This is like saving a document after editing it.
    '''
    conn.commit()  # Save changes to database
    conn.close()   # Close database connection
    
    return action  # Return whether this was an "ENTRY" or "EXIT"

def get_daily_attendance(date_filter=None, name_filter=None):
    """
    This function retrieves attendance records from the database.
    It can filter by date, name, or show all records.
    Think of it like searching through attendance sheets with different criteria.
    """
    conn = sqlite3.connect(DB_PATH)  # Connect to database
    cursor = conn.cursor()  # Create cursor for queries
    
    '''
    Build a flexible query that can be filtered by name or date.
    The "WHERE 1=1" is a trick that makes it easy to add more conditions.
    '''
    query = "SELECT * FROM daily_attendance WHERE 1=1"  # Base query
    params = []  # List to store query parameters
    
    # If name filter is provided, add it to the query
    if name_filter:
        query += " AND name LIKE ?"  # Add name filter condition
        params.append(f"%{name_filter}%")  # Add wildcard search for name
    
    # If date filter is provided, add it to the query
    if date_filter:
        query += " AND date = ?"  # Add date filter condition
        params.append(date_filter)  # Add exact date match
    
    # Order results by date (newest first), then by entry time (latest first)
    query += " ORDER BY date DESC, first_entry DESC"
    
    cursor.execute(query, params)  # Execute the query with parameters
    records = cursor.fetchall()    # Get all matching records
    conn.close()  # Close database connection
    
    return records  # Return the list of records

def sync_csv_to_db():
    """
    This function reads the detailed CSV file and updates the database summary.
    It takes the first and last appearance times from CSV and puts them in the database.
    Think of it like creating a summary report from detailed logs.
    """
    # Check if CSV file exists
    if not os.path.exists(CSV_PATH):
        print("[INFO] No CSV file found to sync")  # Inform user if no CSV exists
        return
    
    conn = sqlite3.connect(DB_PATH)  # Connect to database
    cursor = conn.cursor()  # Create cursor
    
    '''
    Read the CSV file and process each row to extract attendance information.
    '''
    with open(CSV_PATH, 'r') as f:  # Open CSV file
        reader = csv.reader(f)  # Create CSV reader
        rows = list(reader)  # Read all rows into a list
        
        # Check if CSV has valid format (at least 3 columns)
        if not rows or len(rows[0]) < 3:
            print("❌ Invalid CSV format. Header must be: Name, Date, <at least one time>")
            return

        '''
        Process each data row (skip header row at index 0).
        Extract name, date, and all appearance times.
        '''
        for row in rows[1:]:  # Skip header row
            if len(row) < 3:  # Skip rows with insufficient data
                continue
                
            name = row[0].strip()      # Get person's name (remove extra spaces)
            date_str = row[1].strip()  # Get date (remove extra spaces)
            times = [t.strip() for t in row[2:] if t.strip()]  # Get all non-empty times
            
            # Skip if any required data is missing
            if not name or not date_str or not times:
                continue
                
            first_entry = times[0]   # First appearance time
            last_exit = times[-1]    # Last appearance time
            
            '''
            Check if this person already has a record for this date in the database.
            If yes, update it. If no, create a new record.
            '''
            cursor.execute("""
                SELECT COUNT(*) FROM daily_attendance WHERE name=? AND date=?
            """, (name, date_str))  # Count existing records
            
            exists = cursor.fetchone()[0]  # Get the count
            
            if exists:  # If record exists, update it
                cursor.execute("""
                    UPDATE daily_attendance
                    SET first_entry=?, last_exit=?
                    WHERE name=? AND date=?
                """, (first_entry, last_exit, name, date_str))
            else:  # If no record exists, create new one
                cursor.execute("""
                    INSERT INTO daily_attendance (name, date, first_entry, last_exit)
                    VALUES (?, ?, ?, ?)
                """, (name, date_str, first_entry, last_exit))
    
    conn.commit()  # Save all changes to database
    conn.close()   # Close database connection
    print("[INFO] CSV data synced to database")  # Confirm completion

def test_db_connection():
    """
    This function tests if we can connect to the database successfully.
    It's like checking if a door is unlocked before trying to enter.
    """
    try:
        conn = sqlite3.connect(DB_PATH)  # Try to connect to database
        cursor = conn.cursor()  # Create cursor
        cursor.execute("SELECT COUNT(*) FROM daily_attendance")  # Count records
        count = cursor.fetchone()[0]  # Get the count
        conn.close()  # Close connection
        print(f"[✅] Database connection successful. {count} records found.")  # Success message
        return True  # Return success
    except Exception as e:  # If anything goes wrong
        print(f"[❌] Database connection failed: {e}")  # Error message
        return False  # Return failure

'''
This section runs only when the script is executed directly (not imported).
It's like a test area to make sure everything works correctly.
'''
if __name__ == "__main__":
    # Test the database connection first
    test_db_connection()
    
    # Test logging with a sample entry
    test_name = "Test User"  # Sample person name
    test_date = datetime.now().strftime('%Y-%m-%d')  # Today's date
    action = log_attendance(test_name, test_date)  # Log attendance
    print(f"Test completed: {action} logged for {test_name}")  # Show result