"""
Migration script to add conversation_goal and mcts_statistics columns to conversation_analysis table.

This script adds support for goal-based conversation optimization and MCTS algorithm statistics.
"""

import asyncio
import os
import sys

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.config import app_settings
from app.utils.logger import logger


async def run_migration():
    """Run the migration to add conversation_goal and mcts_statistics columns."""

    DATABASE_URL = f"postgresql+asyncpg://{app_settings.DB_USER}:{app_settings.DB_SECRET}@{app_settings.DB_HOST}:{app_settings.DB_PORT}/{app_settings.DB_NAME}"
    engine = create_async_engine(DATABASE_URL, echo=True)

    async with engine.begin() as conn:
        try:
            # Check if conversation_analysis table exists
            logger.info("Checking if conversation_analysis table exists...")
            result = await conn.execute(
                text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = 'conversation_analysis'
                );
            """)
            )
            table_exists = result.scalar()

            if not table_exists:
                logger.info(
                    "Conversation_analysis table does not exist. Please ensure the database is properly initialized."
                )
                return

            # Check if conversation_goal column exists
            logger.info("Checking if conversation_goal column exists...")
            result = await conn.execute(
                text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'conversation_analysis'
                AND column_name = 'conversation_goal';
            """)
            )
            goal_column_exists = result.fetchone() is not None

            if not goal_column_exists:
                logger.info("Adding conversation_goal column to conversation_analysis table...")
                await conn.execute(
                    text("""
                    ALTER TABLE conversation_analysis
                    ADD COLUMN conversation_goal VARCHAR DEFAULT NULL;
                """)
                )
                logger.info("conversation_goal column added successfully")
            else:
                logger.info("conversation_goal column already exists, skipping...")

            # Check if mcts_statistics column exists
            logger.info("Checking if mcts_statistics column exists...")
            result = await conn.execute(
                text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'conversation_analysis'
                AND column_name = 'mcts_statistics';
            """)
            )
            mcts_stats_column_exists = result.fetchone() is not None

            if not mcts_stats_column_exists:
                logger.info("Adding mcts_statistics column to conversation_analysis table...")
                await conn.execute(
                    text("""
                    ALTER TABLE conversation_analysis
                    ADD COLUMN mcts_statistics JSONB NOT NULL DEFAULT '{}'::jsonb;
                """)
                )
                logger.info("mcts_statistics column added successfully")

                # Update existing rows with default MCTS statistics
                logger.info("Updating existing rows with default MCTS statistics...")
                await conn.execute(
                    text("""
                    UPDATE conversation_analysis
                    SET mcts_statistics = '{
                        "total_iterations": 0,
                        "nodes_created": 0,
                        "nodes_evaluated": 0,
                        "pruned_branches": 0,
                        "parallel_evaluations": 0,
                        "average_depth_explored": 0
                    }'::jsonb
                    WHERE mcts_statistics = '{}'::jsonb;
                """)
                )
                logger.info("Existing rows updated with default MCTS statistics")
            else:
                logger.info("mcts_statistics column already exists, skipping...")

            logger.info("Migration completed successfully!")

        except Exception as e:
            logger.error(f"Migration failed: {str(e)}")
            raise

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(run_migration())
