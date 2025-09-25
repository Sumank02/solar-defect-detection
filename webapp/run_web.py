import os
import webbrowser
from threading import Timer
import uvicorn


def main() -> None:
    host = os.environ.get("WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("WEB_PORT", "8000"))
    url = f"http://{host}:{port}"

    # Open the default browser shortly after server starts
    # Delay helps ensure the server is listening before navigation
    Timer(0.8, lambda: webbrowser.open_new_tab(url)).start()

    uvicorn.run("webapp.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()


