"""
Migration script to add users table and update foreign key relationships.

This script should be run to update an existing database to support user management.
For new installations, the tables will be created automatically with the correct schema.
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
    """Run the migration to add users table and update relationships."""

    DATABASE_URL = f"postgresql+asyncpg://{app_settings.DB_USER}:{app_settings.DB_SECRET}@{app_settings.DB_HOST}:{app_settings.DB_PORT}/{app_settings.DB_NAME}"
    engine = create_async_engine(DATABASE_URL, echo=True)

    async with engine.begin() as conn:
        try:
            # Check if user table exists
            logger.info("Checking if user table exists...")
            result = await conn.execute(
                text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = 'user'
                );
            """)
            )
            table_exists = result.scalar()

            if table_exists:
                logger.info("User table exists, checking columns...")

                # Check if name column exists
                result = await conn.execute(
                    text("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'user'
                    AND column_name = 'name';
                """)
                )
                name_column_exists = result.fetchone() is not None

                if not name_column_exists:
                    logger.info("Adding name column to user table...")
                    await conn.execute(
                        text("""
                        ALTER TABLE "user"
                        ADD COLUMN name VARCHAR(255) NOT NULL DEFAULT 'Unnamed User';
                    """)
                    )
                    logger.info("Name column added successfully")

                # Check if created_at column exists
                result = await conn.execute(
                    text("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'user'
                    AND column_name = 'created_at';
                """)
                )
                created_at_exists = result.fetchone() is not None

                if not created_at_exists:
                    logger.info("Adding created_at column to user table...")
                    await conn.execute(
                        text("""
                        ALTER TABLE "user"
                        ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
                    """)
                    )
                    logger.info("Created_at column added successfully")
            else:
                # Create users table
                logger.info("Creating users table...")
                await conn.execute(
                    text("""
                    CREATE TABLE "user" (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        name VARCHAR(255) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                )
                logger.info("User table created successfully")

            # Get all unique user_ids from existing chats
            logger.info("Finding existing user_ids in chat table...")
            result = await conn.execute(
                text("""
                SELECT DISTINCT user_id FROM chat;
            """)
            )
            existing_user_ids = [row[0] for row in result.fetchall()]

            if existing_user_ids:
                logger.info(f"Found {len(existing_user_ids)} unique user_ids in existing chats")

                # Create placeholder users for existing chats
                logger.info("Creating placeholder users for existing chats...")
                for user_id in existing_user_ids:
                    # First check if user already exists
                    result = await conn.execute(
                        text("""
                        SELECT id FROM "user" WHERE id = :user_id;
                    """),
                        {"user_id": user_id},
                    )

                    if not result.fetchone():
                        await conn.execute(
                            text("""
                            INSERT INTO "user" (id, name, created_at)
                            VALUES (:user_id, :name, CURRENT_TIMESTAMP);
                        """),
                            {
                                "user_id": user_id,
                                "name": f"Legacy User {str(user_id)[:8]}",
                            },
                        )
                        logger.info(f"Created placeholder user for ID: {user_id}")
                    else:
                        logger.info(f"User already exists for ID: {user_id}")

                logger.info("Placeholder users created successfully")
            else:
                logger.info("No existing chats found, skipping placeholder user creation")

            # Check if foreign key constraint already exists
            logger.info("Checking existing constraints...")
            result = await conn.execute(
                text("""
                SELECT constraint_name
                FROM information_schema.table_constraints
                WHERE table_name = 'chat'
                AND constraint_type = 'FOREIGN KEY'
                AND constraint_name LIKE '%user_id%';
            """)
            )
            existing_constraint = result.fetchone()

            if not existing_constraint:
                # Add foreign key constraint to chat table
                logger.info("Adding foreign key constraint to chat table...")
                await conn.execute(
                    text("""
                    ALTER TABLE chat
                    ADD CONSTRAINT chat_user_id_fkey
                    FOREIGN KEY (user_id)
                    REFERENCES "user"(id)
                    ON DELETE CASCADE;
                """)
                )
                logger.info("Foreign key constraint added successfully")
            else:
                logger.info("Foreign key constraint already exists, skipping...")

            # Update foreign key constraint on chat_message table if needed
            logger.info("Updating chat_message foreign key constraint...")

            # First drop existing constraint if it exists
            await conn.execute(
                text("""
                ALTER TABLE chat_message
                DROP CONSTRAINT IF EXISTS chat_message_chat_id_fkey;
            """)
            )

            # Then add the new constraint with CASCADE
            await conn.execute(
                text("""
                ALTER TABLE chat_message
                ADD CONSTRAINT chat_message_chat_id_fkey
                FOREIGN KEY (chat_id)
                REFERENCES chat(id)
                ON DELETE CASCADE;
            """)
            )

            logger.info("Migration completed successfully!")

        except Exception as e:
            logger.error(f"Migration failed: {str(e)}")
            raise

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(run_migration())
