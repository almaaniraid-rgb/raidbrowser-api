"""
Pydantic Schemas per RaidbrowsAI API
Modelli di validazione per FastAPI endpoints
"""
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, EmailStr, Field, ConfigDict
from uuid import UUID, uuid4


# =============================================================================
# User & Auth Schemas
# =============================================================================

class UserBase(BaseModel):
    """Base user schema"""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    full_name: Optional[str] = Field(None, max_length=100)


class UserCreate(UserBase):
    """Schema per creazione utente"""
    password: str = Field(..., min_length=8, max_length=128)


class UserUpdate(BaseModel):
    """Schema per aggiornamento utente"""
    email: Optional[EmailStr] = None
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    full_name: Optional[str] = Field(None, max_length=100)
    is_active: Optional[bool] = None


class UserInDB(UserBase):
    """Schema utente con campi database"""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    hashed_password: str
    is_active: bool = True
    is_superuser: bool = False
    created_at: datetime
    updated_at: datetime


class UserPublic(UserBase):
    """Schema utente pubblico (senza password)"""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    is_active: bool
    created_at: datetime


class Token(BaseModel):
    """Schema per token JWT"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Dati decodificati dal token"""
    sub: Optional[str] = None  # user id
    username: Optional[str] = None
    scopes: List[str] = []


# =============================================================================
# Chat & Conversation Schemas
# =============================================================================

class MessageRole(str, Literal["system", "user", "assistant", "tool"]):
    """Ruoli messaggi chat"""
    pass


class ChatMessage(BaseModel):
    """Singolo messaggio chat"""
    role: MessageRole
    content: str
    name: Optional[str] = None  # per tool calls
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ChatRequest(BaseModel):
    """Richiesta chat completion"""
    model: Optional[str] = "auto"
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(4096, ge=1, le=32768)
    stream: bool = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[str] = "auto"
    user_id: Optional[UUID] = None
    conversation_id: Optional[UUID] = None


class ChatChoice(BaseModel):
    """Singola scelta nella risposta"""
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatUsage(BaseModel):
    """Token usage"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    """Risposta chat completion (non-streaming)"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: ChatUsage


class ChatStreamChunk(BaseModel):
    """Chunk per streaming response"""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]


class ConversationBase(BaseModel):
    """Base schema conversazione"""
    title: str = Field(..., min_length=1, max_length=200)
    model: Optional[str] = "auto"


class ConversationCreate(ConversationBase):
    """Crea nuova conversazione"""
    pass


class ConversationUpdate(BaseModel):
    """Aggiorna conversazione"""
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    model: Optional[str] = None
    is_archived: Optional[bool] = None


class ConversationInDB(ConversationBase):
    """Conversazione con campi DB"""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    user_id: UUID
    is_archived: bool = False
    created_at: datetime
    updated_at: datetime


class ConversationPublic(ConversationBase):
    """Conversazione pubblica"""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    is_archived: bool
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class MessageInDB(BaseModel):
    """Messaggio salvato nel DB"""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    conversation_id: UUID
    role: MessageRole
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    model: Optional[str] = None
    tokens_used: int = 0
    created_at: datetime


# =============================================================================
# WebSocket Schemas
# =============================================================================

class WSMessageType(str, Literal[
    "chat",
    "chat_stream",
    "tool_start",
    "tool_end",
    "thought",
    "error",
    "ping",
    "pong",
    "auth",
    "auth_ok",
    "auth_fail",
    "conversation_list",
    "conversation_created",
    "conversation_updated",
    "conversation_deleted"
]):
    """Tipi messaggi WebSocket"""
    pass


class WebSocketMessage(BaseModel):
    """Messaggio WebSocket generico"""
    type: WSMessageType
    payload: Dict[str, Any]
    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WSChatRequest(BaseModel):
    """Richiesta chat via WebSocket"""
    model: Optional[str] = "auto"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 4096
    conversation_id: Optional[UUID] = None


class WSChatResponse(BaseModel):
    """Risposta chat via WebSocket"""
    conversation_id: UUID
    message: ChatMessage
    is_final: bool = False
    usage: Optional[ChatUsage] = None


class WSThoughtResponse(BaseModel):
    """Thought streaming"""
    thought: str
    iteration: int


class WSToolStart(BaseModel):
    """Tool execution started"""
    tool_name: str
    arguments: Dict[str, Any]


class WSToolEnd(BaseModel):
    """Tool execution completed"""
    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None


class WSError(BaseModel):
    """Errore WebSocket"""
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


# =============================================================================
# Model & Provider Schemas
# =============================================================================

class ModelInfo(BaseModel):
    """Info modello disponibile"""
    id: str
    object: str = "model"
    created: int
    owned_by: str
    name: str
    context_window: int
    context_length: int
    available: bool
    unavailable_reason: Optional[str] = None
    execution_status: Optional[str] = None
    supported_parameters: List[str] = []


class ModelsResponse(BaseModel):
    """Lista modelli"""
    object: str = "list"
    data: List[ModelInfo]


# =============================================================================
# n8n Integration Schemas
# =============================================================================

class N8NWorkflow(BaseModel):
    """Workflow n8n"""
    id: str
    name: str
    active: bool
    nodes: List[Dict[str, Any]] = []
    connections: Dict[str, Any] = {}
    created_at: datetime
    updated_at: datetime


class N8NExecuteRequest(BaseModel):
    """Richiesta esecuzione workflow"""
    workflow_name: str
    trigger_data: Dict[str, Any]


class N8NExecuteResponse(BaseModel):
    """Risposta esecuzione workflow"""
    execution_id: str
    status: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class N8NWebhookRequest(BaseModel):
    """Richiesta trigger webhook"""
    webhook_id: str
    data: Dict[str, Any]


# =============================================================================
# Health & System Schemas
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response"""
    status: Literal["healthy", "degraded", "unhealthy"]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = "0.1.0"
    services: Dict[str, str] = {}
    uptime_seconds: float


class ServiceStatus(BaseModel):
    """Stato servizio"""
    name: str
    status: Literal["up", "down", "degraded"]
    latency_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


# =============================================================================
# Pagination Schemas
# =============================================================================

class PaginationParams(BaseModel):
    """Parametri paginazione"""
    page: int = Field(1, ge=1)
    page_size: int = Field(20, ge=1, le=100)


class PaginatedResponse(BaseModel):
    """Risposta paginata generica"""
    items: List[Any]
    total: int
    page: int
    page_size: int
    total_pages: int
    
    @classmethod
    def create(cls, items: List[Any], total: int, params: PaginationParams):
        total_pages = (total + params.page_size - 1) // params.page_size
        return cls(
            items=items,
            total=total,
            page=params.page,
            page_size=params.page_size,
            total_pages=total_pages
        )


# =============================================================================
# Error Schemas
# =============================================================================

class ErrorDetail(BaseModel):
    """Dettaglio errore"""
    field: Optional[str] = None
    message: str
    code: str


class ErrorResponse(BaseModel):
    """Risposta errore standard"""
    error: str
    message: str
    details: List[ErrorDetail] = []
    request_id: Optional[str] = None


# =============================================================================
# Vector Memory Schemas
# =============================================================================

class MemorySearchRequest(BaseModel):
    """Richiesta ricerca memoria vettoriale"""
    query: str
    collection: str = "raidbrowsai"
    limit: int = Field(10, ge=1, le=100)
    filter: Optional[Dict[str, Any]] = None


class MemorySearchResult(BaseModel):
    """Risultato ricerca memoria"""
    id: str
    content: str
    metadata: Dict[str, Any] = {}
    score: float


class MemorySearchResponse(BaseModel):
    """Risposta ricerca memoria"""
    results: List[MemorySearchResult]
    query: str
    collection: str


class MemoryAddRequest(BaseModel):
    """Aggiungi alla memoria"""
    content: str
    metadata: Dict[str, Any] = {}
    collection: str = "raidbrowsai"


class MemoryDeleteRequest(BaseModel):
    """Elimina dalla memoria"""
    ids: List[str]
    collection: str = "raidbrowsai"
