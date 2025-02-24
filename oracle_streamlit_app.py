import streamlit as st
import oracledb
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Database configuration with wallet
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
WALLET_LOCATION = os.getenv("WALLET_LOCATION")  # Path to wallet directory
TNS_ADMIN = os.getenv("TNS_ADMIN")  # Path to wallet directory (same as WALLET_LOCATION)
DB_DSN = os.getenv("DB_DSN")  # TNS name from tnsnames.ora
PEM_PASSPHRASE = os.getenv("PEM_PASSPHRASE")  # Add PEM passphrase from .env

def init_connection():
    """Initialize database connection using Oracle Wallet"""
    try:
        # Ensure wallet directory exists
        if not os.path.exists(WALLET_LOCATION):
            raise Exception(f"Wallet directory not found: {WALLET_LOCATION}")
            
        # Check for required wallet files
        required_files = ['tnsnames.ora', 'sqlnet.ora', 'cwallet.sso']
        missing_files = [f for f in required_files 
                        if not os.path.exists(os.path.join(WALLET_LOCATION, f))]
        if missing_files:
            raise Exception(f"Missing wallet files: {', '.join(missing_files)}")
            
        # Set TNS_ADMIN environment variable
        os.environ["TNS_ADMIN"] = WALLET_LOCATION
        
        # Set PEM passphrase in environment
        if PEM_PASSPHRASE:
            os.environ["ORACLEDB_WALLET_PASSWORD"] = PEM_PASSPHRASE
        
        # Create connection with explicit paths
        connection = oracledb.connect(
            config_dir=WALLET_LOCATION,
            user=DB_USER,
            password=DB_PASSWORD,
            dsn=f"{DB_DSN}_high",  # Add service level suffix
            wallet_location=WALLET_LOCATION,
            wallet_password=PEM_PASSPHRASE  # Use PEM passphrase from .env
        )
        
        # Print connection info for debugging
        st.info(f"Connected successfully using DSN: {DB_DSN}_high")
        
        return connection
        
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        st.error("Please make sure your wallet files and .env configuration are correct")
        
        # Show current configuration
        st.code(f"""
Current configuration:
TNS_ADMIN: {os.environ.get('TNS_ADMIN', 'Not set')}
WALLET_LOCATION: {WALLET_LOCATION}
DB_DSN: {DB_DSN}
Wallet files found: {os.listdir(WALLET_LOCATION) if os.path.exists(WALLET_LOCATION) else 'Directory not found'}
        """)
        return None

def run_query(connection, query):
    """Run SQL query and return results as pandas DataFrame"""
    try:
        return pd.read_sql(query, connection)
    except Exception as e:
        st.error(f"Query failed: {str(e)}")
        return None

def main():
    st.title("Oracle Database Query Explorer")
    
    # Initialize connection
    connection = init_connection()
    
    if connection:
        st.success("Connected to Oracle Database!")
        
        # Sample queries
        queries = {
            "List All Tables": """
                SELECT table_name 
                FROM user_tables 
                ORDER BY table_name
            """,
            "Employee Count by Department": """
                SELECT d.department_name, COUNT(e.employee_id) as employee_count
                FROM departments d
                LEFT JOIN employees e ON d.department_id = e.department_id
                GROUP BY d.department_name
                ORDER BY employee_count DESC
            """,
            "Custom Query": "Enter your own query"
        }
        
        # Query selector
        query_type = st.selectbox(
            "Select Query",
            options=list(queries.keys())
        )
        
        if query_type == "Custom Query":
            # Custom query input
            query = st.text_area(
                "Enter your SQL query",
                height=150,
                placeholder="SELECT * FROM your_table"
            )
        else:
            query = queries[query_type]
            st.code(query, language="sql")
        
        # Execute query button
        if st.button("Run Query"):
            if query:
                st.write("Executing query...")
                results = run_query(connection, query)
                
                if results is not None:
                    # Display results
                    st.write("Results:")
                    st.dataframe(
                        results,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Download button
                    csv = results.to_csv(index=False)
                    st.download_button(
                        "Download Results",
                        csv,
                        "query_results.csv",
                        "text/csv",
                        key='download-csv'
                    )
            else:
                st.warning("Please enter a query")
    
    else:
        st.error("Please check your database configuration")

if __name__ == "__main__":
    main() 