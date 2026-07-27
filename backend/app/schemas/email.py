from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel,Field, field_validator


class EmailMessage(BaseModel):
   
    message_id: str = Field(..., alias="MessageID", description="Unique RFC 822 message identifier")
    from_address: str = Field(..., alias="From", description="Sender email address")
    to_address: str = Field(..., alias="To", description="Recipient email address(es)")
    in_reply_to: Optional[str] = Field(default="", alias="InReplyTo", description="Parent message ID")
    subject: str = Field(default="", alias="Subject", description="Email subject line")
    date_str: str = Field(..., alias="Date", description="Raw date string")
    body: str = Field(default="", alias="Body", description="Raw email content")
    thread_key: str = Field(default="", alias="ThreadKey", description="Thread lookup key")
    filename: str = Field(default="", alias="Filename", description="Source filename reference")
    thread_id: str = Field(..., alias="ThreadID", description="Parent thread identifier")
    thread_position: int = Field(default=0, alias="ThreadPosition", description="Sequence order within thread")

    class Config:
        populate_by_name = True


class EmailThread(BaseModel):
    
    thread_id: str
    messages: List[EmailMessage] = Field(default_factory=list)
    @property
    def message_count(self) -> int:
        return len(self.messages)