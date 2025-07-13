'''
This script generates a formatted attendance report from the SQLite database.
It reads attendance data and displays it in a nice table format on the screen.

WHAT THIS SCRIPT DOES:
1. Connects to the attendance database
2. Retrieves all attendance records
3. Displays them in a formatted table with columns for Name, Date, First Entry, and Last Exit
4. Shows when each person first arrived and last left each day

WHY THIS IS USEFUL:
- Provides a quick overview of daily attendance
- Shows arrival and departure times for each person
- Presents data in an easy-to-read table format
- Useful for managers to review attendance patterns
'''

import sqlite3  # Import sqlite3 for database operations - this connects to and reads from the database
from tabulate import tabulate  # Import tabulate for creating formatted tables - this makes data look nice in columns

# Configuration: Database file path
DB_PATH = 'database/attendance.db'  # Path to the SQLite database file - ensure this matches your DB path

'''
Function to fetch attendance records from the database.
This function connects to the database, retrieves all attendance data, and returns it.

HOW IT WORKS:
1. Opens a connection to the SQLite database
2. Executes a query to get all attendance records
3. Sorts the results by date and then by name
4. Returns all the records as a list
'''
def fetch_attendance_records():
    # Connect to the database
    conn = sqlite3.connect(DB_PATH)  # Create connection to the SQLite database
    cursor = conn.cursor()  # Create cursor object to execute SQL commands

    '''
    Execute SQL query to get attendance data.
    This query selects specific columns from the daily_attendance table:
    - name: Person's name
    - date: The date of attendance
    - first_entry: When they first entered that day
    - last_exit: When they last left that day
    Results are sorted by date first, then by name alphabetically.
    '''
    # Fetch data from the updated table: daily_attendance
    cursor.execute("SELECT name, date, first_entry, last_exit FROM daily_attendance ORDER BY date, name")  # Execute SQL query
    records = cursor.fetchall()  # Get all records returned by the query

    # Close connection
    conn.close()  # Close the database connection to free up resources
    return records  # Return the list of attendance records

'''
Main execution block - runs when the script is executed directly.
This section handles displaying the attendance report to the user.
'''
if __name__ == "__main__":  # If this script is run directly (not imported)
    data = fetch_attendance_records()  # Get all attendance records from database

    '''
    Display the attendance data in a formatted table.
    If there's data, create a nice table. If not, show a message.
    '''
    if data:  # If there are attendance records in the database
        headers = ["Name", "Date", "First Entry", "Last Exit"]  # Define column headers for the table
        print("\n📋 Attendance Report (First Entry & Last Exit per Day):\n")  # Print report title
        print(tabulate(data, headers=headers, tablefmt="grid"))  # Display data in a grid table format
    else:  # If no attendance records found
        print("⚠️ No attendance records found in database.")  # Print message indicating no data