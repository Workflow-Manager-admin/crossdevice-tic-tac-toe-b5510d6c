from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
from uuid import uuid4, UUID
from enum import Enum
import datetime

app = FastAPI(
    title="Tic Tac Toe API",
    description="API for a cross-device Tic Tac Toe game. Provides endpoints to start new games, make moves, view game status, and see game history.",
    version="1.0.0",
    openapi_tags=[
        {"name": "Games", "description": "Game creation, move, status & retrieval endpoints"},
        {"name": "History", "description": "Past game result/history endpoints"},
        {"name": "System", "description": "System/health check"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set to specific origins in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------ Data Models ------------

class PlayerEnum(str, Enum):
    X = "X"
    O = "O"

class CellEnum(str, Enum):
    X = "X"
    O = "O"
    EMPTY = " "

class GameStatusEnum(str, Enum):
    IN_PROGRESS = "in_progress"
    X_WON = "X_won"
    O_WON = "O_won"
    DRAW = "draw"

class Move(BaseModel):
    row: int = Field(..., ge=0, le=2, description="Row number (0, 1, or 2)")
    col: int = Field(..., ge=0, le=2, description="Column number (0, 1, or 2)")
    player: PlayerEnum = Field(..., description="Either 'X' or 'O'")

    # Validate that a move is for one of the legit players
    @validator("player")
    def valid_player(cls, v):
        if v not in ("X", "O"):
            raise ValueError("Player must be 'X' or 'O'")
        return v

class GameCreateResponse(BaseModel):
    game_id: UUID
    player_x: str = Field(..., description="Session ID for player X")
    player_o: str = Field(..., description="Session ID for player O (auto-joins on new game)")

class GameState(BaseModel):
    board: List[List[CellEnum]]
    next_turn: PlayerEnum
    status: GameStatusEnum
    winner: Optional[PlayerEnum]
    move_count: int
    history: List["Move"] = []
    started_at: datetime.datetime
    finished_at: Optional[datetime.datetime] = None

    class Config:
        json_encoders = {
            datetime.datetime: lambda v: v.isoformat() if v else None,
        }

class GameRecord(BaseModel):
    game_id: UUID
    status: GameStatusEnum
    winner: Optional[PlayerEnum]
    started_at: datetime.datetime
    finished_at: Optional[datetime.datetime] = None

class MoveRequest(BaseModel):
    row: int = Field(..., ge=0, le=2, description="Row number")
    col: int = Field(..., ge=0, le=2, description="Column number")
    player: PlayerEnum = Field(..., description="Player making the move")

class MoveResponse(BaseModel):
    move_applied: bool
    reason: Optional[str] = ""
    game_state: Optional[GameState] = None

# For forward reference (history in GameState)
GameState.update_forward_refs()


# ------------ In-memory persistence ------------

class TicTacToeManager:
    """
    Central in-memory manager for all games, game history, and session->game mapping.
    """

    def __init__(self):
        self.games: Dict[UUID, GameState] = {}
        self.game_history: List[GameRecord] = []
        self.game_moves: Dict[UUID, List[Move]] = {}
        self.game_players: Dict[UUID, Dict[PlayerEnum, str]] = {}  # game_id -> {X: sid, O: sid}

    # PUBLIC_INTERFACE
    def create_new_game(self, player_x_sid: str) -> (UUID, str):
        """Create new game with X = player_x_sid, O assigned to a new session. Returns game_id and player_o_sid."""
        game_id = uuid4()
        board = [[CellEnum.EMPTY for _ in range(3)] for _ in range(3)]
        now = datetime.datetime.utcnow()
        game_state = GameState(
            board=board,
            next_turn=PlayerEnum.X,
            status=GameStatusEnum.IN_PROGRESS,
            winner=None,
            move_count=0,
            history=[],
            started_at=now,
            finished_at=None,
        )
        self.games[game_id] = game_state
        self.game_moves[game_id] = []
        player_o_sid = str(uuid4())
        self.game_players[game_id] = {PlayerEnum.X: player_x_sid, PlayerEnum.O: player_o_sid}
        return game_id, player_o_sid

    # PUBLIC_INTERFACE
    def join_game(self, game_id: UUID, player: PlayerEnum, session_id: str):
        """Assign session_id to given game and player slot if empty."""
        if game_id not in self.games:
            raise HTTPException(status_code=404, detail="Game ID not found")
        self.game_players[game_id][player] = session_id

    # PUBLIC_INTERFACE
    def make_move(self, game_id: UUID, move: Move, session_id: str) -> MoveResponse:
        """Apply a move to the game, if allowed."""
        # Existence and session/turn checks
        if game_id not in self.games:
            return MoveResponse(move_applied=False, reason="Game not found", game_state=None)
        game = self.games[game_id]
        players = self.game_players[game_id]
        if game.status != GameStatusEnum.IN_PROGRESS:
            return MoveResponse(move_applied=False, reason="Game is finished", game_state=game)
        # Session must match player's session
        if players[move.player] != session_id:
            return MoveResponse(move_applied=False, reason="Wrong session for this player", game_state=game)
        if game.next_turn != move.player:
            return MoveResponse(move_applied=False, reason=f"It is not {move.player}'s turn", game_state=game)
        # Validate move
        r, c = move.row, move.col
        if not (0 <= r <= 2 and 0 <= c <= 2):
            return MoveResponse(move_applied=False, reason="Invalid board position", game_state=game)
        if game.board[r][c] != CellEnum.EMPTY:
            return MoveResponse(move_applied=False, reason="Cell is already occupied", game_state=game)
        # Make move
        game.board[r][c] = CellEnum(move.player)
        game.move_count += 1
        move_copy = Move(row=move.row, col=move.col, player=move.player)
        self.game_moves[game_id].append(move_copy)
        game.history.append(move_copy)
        # Check for winner or draw
        won = check_win(game.board, move.player)
        if won:
            game.status = (
                GameStatusEnum.X_WON if move.player == PlayerEnum.X else GameStatusEnum.O_WON
            )
            game.winner = move.player
            game.finished_at = datetime.datetime.utcnow()
        elif game.move_count == 9:
            game.status = GameStatusEnum.DRAW
            game.winner = None
            game.finished_at = datetime.datetime.utcnow()
        else:
            game.next_turn = PlayerEnum.O if move.player == PlayerEnum.X else PlayerEnum.X
        # Save to games (as update)
        self.games[game_id] = game
        # If now finished, save summary to history
        if game.status != GameStatusEnum.IN_PROGRESS:
            rec = GameRecord(
                game_id=game_id,
                status=game.status,
                winner=game.winner,
                started_at=game.started_at,
                finished_at=game.finished_at
            )
            self.game_history.append(rec)
        return MoveResponse(move_applied=True, reason=None, game_state=game)

    # PUBLIC_INTERFACE
    def get_game(self, game_id: UUID) -> GameState:
        """Get the current state of a game."""
        if game_id not in self.games:
            raise HTTPException(status_code=404, detail="Game ID not found")
        return self.games[game_id]

    # PUBLIC_INTERFACE
    def get_moves(self, game_id: UUID) -> List[Move]:
        """Get the move history for a game."""
        if game_id not in self.games:
            raise HTTPException(status_code=404, detail="Game ID not found")
        return self.game_moves.get(game_id, [])

    # PUBLIC_INTERFACE
    def get_game_history(self) -> List[GameRecord]:
        """Return summary of all finished games."""
        return self.game_history[::-1]  # Most recent first

    # PUBLIC_INTERFACE
    def get_player_role(self, game_id: UUID, session_id: str) -> Optional[PlayerEnum]:
        """Return X or O if this session is mapped to the game, else None."""
        if game_id not in self.game_players:
            return None
        for player, sid in self.game_players[game_id].items():
            if sid == session_id:
                return player
        return None

def check_win(board: List[List[CellEnum]], player: PlayerEnum) -> bool:
    """Return True if player has won on the board."""
    sym = CellEnum(player)
    for i in range(3):
        if all(board[i][j] == sym for j in range(3)):  # Row
            return True
        if all(board[j][i] == sym for j in range(3)):  # Col
            return True
    if all(board[i][i] == sym for i in range(3)):  # Diagonal
        return True
    if all(board[i][2 - i] == sym for i in range(3)):  # Anti-diagonal
        return True
    return False

# Singleton manager instance
ttt_manager = TicTacToeManager()

# ------------ Utility: Session Handling ------------

SESSION_COOKIE_NAME = "tic_tac_toe_session"

# PUBLIC_INTERFACE
def get_session_id(request: Request, response: Response) -> str:
    """
    Get session id (cookie-based). If not present, set a new UUID as the cookie.
    """
    sid = request.cookies.get(SESSION_COOKIE_NAME)
    if not sid:
        sid = str(uuid4())
        # Set session cookie with path, HTTPOnly flag
        response.set_cookie(key=SESSION_COOKIE_NAME, value=sid, httponly=True, samesite="lax")
    return sid

# ------------ API Endpoints ------------

@app.get("/", summary="Health Check", tags=["System"])
def health_check():
    """Root endpoint for verifying service health."""
    return {"message": "Healthy"}

# PUBLIC_INTERFACE
@app.post("/games/", response_model=GameCreateResponse, summary="Start a new game", tags=["Games"])
def start_new_game(response: Response, request: Request):
    """
    Starts a new Tic Tac Toe game. 
    Returns the created game ID and the assigned session IDs for player X (current user) and O (random/auto-join).
    The client must store its session ID (for move requests).
    """
    sid = get_session_id(request, response)
    game_id, player_o_sid = ttt_manager.create_new_game(sid)
    return GameCreateResponse(
        game_id=game_id, player_x=sid, player_o=player_o_sid
    )

# PUBLIC_INTERFACE
@app.post(
    "/games/{game_id}/moves",
    response_model=MoveResponse,
    summary="Make a move in a game",
    tags=["Games"],
)
def make_move(
    game_id: UUID,
    move_req: MoveRequest,
    request: Request,
    response: Response,
):
    """
    Make a move in the specified game. Requires a valid session cookie for the player slot.
    Returns updated game state. Enforces alternating turns, win, and draw rules.
    """
    sid = get_session_id(request, response)
    move = Move(row=move_req.row, col=move_req.col, player=move_req.player)
    resp = ttt_manager.make_move(game_id, move, sid)
    return resp

# PUBLIC_INTERFACE
@app.get(
    "/games/{game_id}/status",
    response_model=GameState,
    summary="Get game status",
    tags=["Games"],
)
def get_game_status(game_id: UUID):
    """Fetch the current state and status of the specified game."""
    return ttt_manager.get_game(game_id)

# PUBLIC_INTERFACE
@app.get(
    "/games/{game_id}/moves",
    response_model=List[Move],
    summary="Retrieve move history for a game",
    tags=["Games"],
)
def get_moves(game_id: UUID):
    """Fetch full move-by-move history for the specified game."""
    return ttt_manager.get_moves(game_id)

# PUBLIC_INTERFACE
@app.get(
    "/games/history",
    response_model=List[GameRecord],
    summary="List result history of all games",
    tags=["History"],
)
def get_history():
    """
    Returns a summary of all finished games (all-time game history, most recent first).
    Each record includes winner, start/end time, and result.
    """
    return ttt_manager.get_game_history()

# PUBLIC_INTERFACE
@app.get(
    "/games/{game_id}/role",
    response_model=Optional[PlayerEnum],
    summary="Get user's role (X or O) in a game",
    tags=["Games"],
)
def get_role(game_id: UUID, request: Request):
    """Returns X/O if caller's session is assigned to this game (else None)."""
    sid = request.cookies.get(SESSION_COOKIE_NAME)
    return ttt_manager.get_player_role(game_id, sid) if sid else None


# Custom OpenAPI with extended doc for WebSocket expansion potential
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Tic Tac Toe API",
        version="1.0.0",
        description=app.description + "\n\n*All endpoints require a 'tic_tac_toe_session' cookie for user session tracking.*",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# FastAPI automatic interactive docs enabled at /docs
