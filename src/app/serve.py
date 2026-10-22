"""Start the same app locally or on a host that supplies PORT."""
import os
import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), workers=1)
