import os
import uvicorn


def main() -> None:
    host = os.environ.get("WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("WEB_PORT", "8000"))

    uvicorn.run("webapp.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()


