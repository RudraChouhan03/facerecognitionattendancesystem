'''
This script synchronizes attendance data from a CSV file to a SQLite database.
It reads the CSV file (like an Excel spreadsheet) and copies the data into a database.

WHAT THIS SCRIPT DOES:
1. Reads attendance data from a CSV file
2. Connects to a SQLite database
3. Updates or inserts attendance records in the database
4. Ensures both CSV and database have the same information

WHY THIS IS NEEDED:
- The CSV file is human-readable but not great for complex queries
- The database is better for searching and generating reports
- This script keeps both files synchronized
'''

import sqlite3  # Import sqlite3 for database operations - this handles the database connection
import csv  # Import csv for reading CSV files - this reads Excel-like files
import os  # Import os for file system operations - this checks if files exist

# Configuration: File paths for CSV and database
CSV_FILE = 'attendance_log.csv'  # Path to the CSV file containing attendance data
DB_PATH = 'database/attendance.db'  # Path to the SQLite database file

'''
Main function that synchronizes CSV data to the database.
This function does all the work of copying data from CSV to database.

HOW IT WORKS:
1. Checks if both CSV file and database exist
2. Reads all data from the CSV file
3. For each person's attendance record, updates the database
4. Handles both new records and updates to existing records
'''
def sync_csv_to_db():
    # Check if CSV file exists
    if not os.path.exists(CSV_FILE):  # If the CSV file doesn't exist
        print(f"❌ CSV file not found: {CSV_FILE}")  # Print error message
        return  # Exit the function

    # Check if database file exists
    if not os.path.exists(DB_PATH):  # If the database file doesn't exist
        print(f"❌ Database file not found: {DB_PATH}")  # Print error message
        return  # Exit the function

    # Connect to the SQLite database
    conn = sqlite3.connect(DB_PATH)  # Create connection to database
    cursor = conn.cursor()  # Create cursor for executing SQL commands

    '''
    Read all data from the CSV file into memory.
    This loads the entire attendance spreadsheet so we can process it.
    '''
    with open(CSV_FILE, 'r') as f:  # Open the CSV file for reading
        reader = csv.reader(f)  # Create CSV reader object
        rows = list(reader)  # Read all rows into a list

    '''
    Validate the CSV format.
    The CSV must have at least 3 columns: Name, Date, and at least one time entry.
    '''
    if not rows or len(rows[0]) < 3:  # If no rows or header has less than 3 columns
        print("❌ Invalid CSV format. Header must be: Name, Date, <at least one time>")  # Print error
        return  # Exit the function

    '''
    Process each row of attendance data (skip the header row).
    For each person's daily attendance, extract their name, date, and entry/exit times.
    '''
    for row in rows[1:]:  # Loop through all rows except the header (first row)
        if len(row) < 3:  # If row doesn't have minimum required columns
            continue  # Skip this row and go to the next one
        
        name = row[0].strip()  # Get person's name and remove extra spaces
        date = row[1].strip()  # Get date and remove extra spaces
        times = [t.strip() for t in row[2:] if t.strip()]  # Get all non-empty time entries
        
        # Skip rows with missing essential data
        if not name or not date or not times:  # If name, date, or times are missing
            continue  # Skip this row and go to the next one
        
        '''
        Extract first entry time and last exit time from all recorded times.
        - first_entry: The first time the person entered that day
        - last_exit: The last time the person left that day
        '''
        first_entry = times[0]  # First time in the list is the first entry
        last_exit = times[-1]  # Last time in the list is the last exit
        
        '''
        Upsert logic: Update if record exists, insert if it doesn't.
        This ensures we don't create duplicate records for the same person and date.
        '''
        # Check if a record already exists for this person and date
        cursor.execute("""
            SELECT COUNT(*) FROM daily_attendance WHERE name=? AND date=?
        """, (name, date))  # Execute SQL query to count existing records
        exists = cursor.fetchone()[0]  # Get the count result
        
        if exists:  # If record already exists in database
            # Update the existing record with new times
            cursor.execute("""
                UPDATE daily_attendance
                SET first_entry=?, last_exit=?
                WHERE name=? AND date=?
            """, (first_entry, last_exit, name, date))  # Update existing record
        else:  # If record doesn't exist in database
            # Insert a new record
            cursor.execute("""
                INSERT INTO daily_attendance (name, date, first_entry, last_exit)
                VALUES (?, ?, ?, ?)
            """, (name, date, first_entry, last_exit))  # Insert new record

    '''
    Save all changes to the database and close the connection.
    This ensures all data is permanently stored in the database.
    '''
    conn.commit()  # Save all changes to the database
    conn.close()  # Close the database connection
    print("✅ Synced CSV data to SQLite database successfully.")  # Print success message

'''
Main execution block - runs when the script is executed directly.
This allows the script to be run from the command line or called from other scripts.
'''
if __name__ == "__main__":  # If this script is run directly (not imported)
    sync_csv_to_db()  # Call the main synchronization function