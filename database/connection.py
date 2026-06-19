"""
Database connection management for territory optimization system.
"""
import logging
from typing import Optional, Any
import sqlite3
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Manages database connections and operations."""
    
    def __init__(self, db_path: str = "territory_optimizer.db"):
        """
        Initialize database connection manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.connection = None
    
    def connect(self) -> sqlite3.Connection:
        """
        Establish database connection.
        
        Returns:
            Database connection object
        """
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
            logger.info(f"Connected to database: {self.db_path}")
            return self.connection
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Database connection closed")
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.
        
        Yields:
            Database connection object
        """
        conn = self.connect()
        try:
            yield conn
        finally:
            self.disconnect()
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> list:
        """
        Execute SELECT query and return results.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            List of query results
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def execute_update(self, query: str, params: Optional[tuple] = None) -> int:
        """
        Execute INSERT/UPDATE/DELETE query.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Number of affected rows
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                conn.commit()
                return cursor.rowcount
        except Exception as e:
            logger.error(f"Update execution failed: {e}")
            raise
    
    def initialize_schema(self) -> None:
        """Initialize database schema."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create tables if they don't exist
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS optimization_jobs (
                        job_id TEXT PRIMARY KEY,
                        status TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        completed_at TIMESTAMP,
                        parameters TEXT
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS solutions (
                        solution_id TEXT PRIMARY KEY,
                        job_id TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        business_impact TEXT,
                        disruption_metrics TEXT,
                        FOREIGN KEY (job_id) REFERENCES optimization_jobs (job_id)
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS dealer_changes (
                        change_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        solution_id TEXT,
                        dealer_id TEXT,
                        from_ftc_id TEXT,
                        to_ftc_id TEXT,
                        impact_score REAL,
                        FOREIGN KEY (solution_id) REFERENCES solutions (solution_id)
                    )
                """)
                
                conn.commit()
                logger.info("Database schema initialized")
                
        except Exception as e:
            logger.error(f"Schema initialization failed: {e}")
            raise
