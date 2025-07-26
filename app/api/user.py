from uuid import UUID

from fastapi import APIRouter, HTTPException, Response, status

from ..db import chat as db
from ..schema.llm.chat import Chat
from ..schema.user import User, UserCreate
from ..utils.logger import logger

router = APIRouter(prefix="/users", tags=["User"])


@router.post("/", response_model=User, status_code=status.HTTP_201_CREATED)
async def create_user(request: UserCreate):
    """
    Creates a new user with the provided name.
    """
    logger.info(
        "Creating new user",
        extra={
            "user_name": request.name,
        },
    )

    try:
        user = await db.create_user(name=request.name)
        logger.info(
            "User created successfully",
            extra={
                "user_id": str(user.id),
                "user_name": user.name,
            },
        )
        return user
    except Exception as e:
        logger.error(
            "Error creating user",
            extra={
                "user_name": request.name,
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=list[User])
async def list_users():
    """
    Retrieves all users in the system.
    """
    logger.info("Listing all users")

    try:
        users = await db.list_users()
        logger.info(
            "Users retrieved successfully",
            extra={
                "user_count": len(users),
            },
        )
        return users
    except Exception as e:
        logger.error(
            "Error listing users",
            extra={
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{user_id}", response_model=User)
async def get_user(user_id: UUID):
    """
    Retrieves a specific user by their ID.
    """
    logger.info(
        "Retrieving user",
        extra={
            "user_id": str(user_id),
        },
    )

    try:
        user = await db.get_user(user_id)
        if not user:
            logger.warning(
                "User not found",
                extra={
                    "user_id": str(user_id),
                },
            )
            raise HTTPException(status_code=404, detail="User not found")

        logger.info(
            "User retrieved successfully",
            extra={
                "user_id": str(user_id),
                "user_name": user.name,
            },
        )
        return user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Error retrieving user",
            extra={
                "user_id": str(user_id),
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: UUID):
    """
    Deletes a user and all their associated chats and messages.
    """
    logger.info(
        "Deleting user",
        extra={
            "user_id": str(user_id),
        },
    )

    try:
        deleted = await db.delete_user(user_id)
        if not deleted:
            logger.warning(
                "User not found for deletion",
                extra={
                    "user_id": str(user_id),
                },
            )
            raise HTTPException(status_code=404, detail="User not found")

        logger.info(
            "User deleted successfully",
            extra={
                "user_id": str(user_id),
            },
        )
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Error deleting user",
            extra={
                "user_id": str(user_id),
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{user_id}/chats", response_model=list[Chat])
async def get_user_chats(user_id: UUID):
    """
    Retrieves all chat sessions for a specific user.
    """
    logger.info(
        "Retrieving chats for user",
        extra={
            "user_id": str(user_id),
        },
    )

    try:
        user = await db.get_user(user_id)
        if not user:
            logger.warning(
                "User not found",
                extra={
                    "user_id": str(user_id),
                },
            )
            raise HTTPException(status_code=404, detail="User not found")

        chats = await db.get_user_chats(user_id)
        logger.info(
            "User chats retrieved successfully",
            extra={
                "user_id": str(user_id),
                "chat_count": len(chats),
            },
        )
        return chats
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Error retrieving user chats",
            extra={
                "user_id": str(user_id),
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))
