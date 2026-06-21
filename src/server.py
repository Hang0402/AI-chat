"""FastAPI server: REST + WebSocket with streaming chat."""

import asyncio
import json
import os
from pathlib import Path

# Load .env file
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").strip().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _val = _line.partition("=")
            _key = _key.strip()
            _val = _val.strip().strip('"').strip("'")
            os.environ.setdefault(_key, _val)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.engine import ChatEngine
from src.distiller import CharacterDistiller

app = FastAPI(title="Roleplay AI", version="1.0.0")
app.mount("/avatars", StaticFiles(directory="avatars"), name="avatars")

engine = ChatEngine()
distiller = CharacterDistiller()

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/favicon.ico")
async def favicon():
    return FileResponse(STATIC_DIR / "favicon.svg")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/characters")
async def list_characters():
    chars_dir = Path("characters")
    chars_dir.mkdir(exist_ok=True)
    files = list(chars_dir.glob("*.json"))
    result = []
    for f in files:
        try:
            card = json.loads(f.read_text(encoding="utf-8"))
            avatar = card.get("avatar", "")
            result.append({
                "id": f.stem,
                "name": card.get("name", f.stem),
                "personality": card.get("personality", [])[:3],
                "greeting": card.get("greeting", ""),
                "avatar_url": "/" + avatar if avatar else "",
            })
        except Exception:
            pass
    return result


@app.post("/api/character/load/{char_id}")
async def load_character(char_id: str):
    path = Path(f"characters/{char_id}.json")
    if not path.exists():
        return {"error": "Character not found"}
    engine.load_character(path)
    card = engine.card
    avatar = card.get("avatar", "")
    greeting = card.get("greeting", f"你好，我是{card.get('name', '角色')}~")
    return {"ok": True, "name": card.get("name", ""), "greeting": greeting, "avatar_url": "/" + avatar if avatar else ""}


@app.post("/api/character/distill")
async def distill_character(data: dict):
    source_type = data.get("type", "text")
    content = data.get("content", "")
    if source_type == "url":
        card = distiller.distill_from_url(content)
    elif source_type == "search":
        card = distiller.distill_from_search(content)
    else:
        card = distiller.distill_from_text(content)
    if "error" in card:
        return card
    filepath = distiller.save_character(card)
    return {"ok": True, "card": card, "saved_to": str(filepath)}



@app.post("/api/character/avatar/{char_id}")
async def upload_avatar(char_id: str, file: UploadFile = File(...)):
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in (file.filename or "") else "png"
    if ext not in ("png", "jpg", "jpeg", "gif", "webp"):
        return {"error": "Invalid file type"}
    fname = f"{char_id}.{ext}"
    fpath = Path(f"avatars/{fname}")
    fpath.write_bytes(await file.read())
    # Update character card
    card_path = Path(f"characters/{char_id}.json")
    if card_path.exists():
        card = json.loads(card_path.read_text(encoding="utf-8"))
        card["avatar"] = f"avatars/{fname}"
        card_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "avatar_url": f"/{card['avatar']}"}


@app.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            msg_type = msg.get("type", "chat")

            if msg_type == "chat":
                user_msg = msg.get("message", "")
                full = ""
                try:
                    for chunk in engine.chat_stream(user_msg):
                        await ws.send_text(json.dumps({"type": "chunk", "content": chunk}, ensure_ascii=False))
                        await asyncio.sleep(0)
                        full += chunk
                    await ws.send_text(json.dumps({"type": "done", "full": full}, ensure_ascii=False))
                except Exception as e:
                    await ws.send_text(json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False))

            elif msg_type == "load":
                char_id = msg.get("character", "")
                path = Path(f"characters/{char_id}.json")
                if path.exists():
                    engine.load_character(path)
                    card = engine.card
                    await ws.send_text(json.dumps({
                        "type": "loaded",
                        "name": card.get("name", ""),
                        "greeting": card.get("greeting", ""),
                    }, ensure_ascii=False))

            elif msg_type == "reset":
                engine.history = []
                engine.long_term = []
                await ws.send_text(json.dumps({"type": "reset"}, ensure_ascii=False))

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
