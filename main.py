"""Entry point: python main.py [serve|distill|chat]"""

import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py serve | distill | chat")
        print("  serve   - Start web server at http://localhost:8080")
        print("  distill - Distill character from text/url/search")
        print("  chat    - CLI chat with a character")
        return 1

    cmd = sys.argv[1]

    if cmd == "serve":
        import uvicorn
        uvicorn.run("src.server:app", host="0.0.0.0", port=8080, reload=True)

    elif cmd == "distill":
        from src.distiller import CharacterDistiller
        d = CharacterDistiller()
        if len(sys.argv) >= 4 and sys.argv[2] == "--url":
            card = d.distill_from_url(sys.argv[3])
        elif len(sys.argv) >= 4 and sys.argv[2] == "--search":
            card = d.distill_from_search(" ".join(sys.argv[3:]))
        elif len(sys.argv) >= 3 and sys.argv[2] == "--text":
            text = sys.stdin.read() if not sys.stdin.isatty() else input("Paste text: ")
            card = d.distill_from_text(text)
        else:
            print("Usage: python main.py distill --url <URL>")
            print("       python main.py distill --search <query>")
            print("       python main.py distill --text (then paste)")
            return 1
        import json
        print(json.dumps(card, ensure_ascii=False, indent=2))
        filepath = d.save_character(card)
        print(f"\nSaved to: {filepath}")

    elif cmd == "chat":
        from src.engine import ChatEngine
        engine = ChatEngine()
        if len(sys.argv) >= 3:
            engine.load_character(sys.argv[2])
        else:
            engine.load_character("characters/xiaolu.json")
        name = engine.card.get("name", "Character")
        greeting = engine.card.get("greeting", f"Hi, I'm {name}")
        print(f"=== {name} ===\n{greeting}\n")
        while True:
            try:
                msg = input("You: ")
                if msg.lower() in ("quit", "exit", "q"):
                    break
                resp = engine.chat(msg)
                print(f"{name}: {resp}")
            except (KeyboardInterrupt, EOFError):
                break
        print("\nGoodbye!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
