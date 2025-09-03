import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import text, bindparam
from typing import List, Dict, Any
from loguru import logger


class Featurizer:
    def __init__(self):
        self.engine = None
    
    async def initialize(self):
        """Initialize database connection"""
        db_url = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:password@db:5432/homedata")
        try:
            self.engine = create_async_engine(db_url)
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
                await conn.execute(text("SELECT * from zipcode_demo LIMIT 1"))
            print("Database connected")
        except Exception as e:
            print(f"Database connection failed: {e}")
    
    async def cleanup(self):
        """Cleanup database connection"""
        if self.engine:
            await self.engine.dispose()
    
    async def enrich(self, features):
        """Add demographic data from database based on zipcode"""
        if not self.engine or 'zipcode' not in features:
            return features
        
        try:
            async with AsyncSession(self.engine) as session:
                result = await session.execute(
                    text("SELECT * FROM zipcode_demo WHERE zipcode = :zipcode"),
                    {"zipcode": int(features['zipcode'])}
                )
                row = result.fetchone()
                if row:
                    # Add all columns except zipcode to features
                    for col, val in zip(result.cursor.description, row):
                        if col[0] != 'zipcode':
                            features[col[0]] = val
        except Exception as e:
            print(f"Failed to enrich: {e}")
        
        return features
    
    def is_ready(self):
        return self.engine is not None