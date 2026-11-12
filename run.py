"""تشغيل بأمر واحد:  python run.py   (والـ PORT/HOST من environment لو عايز تغيّرهم)"""
import os

import uvicorn

if __name__ == "__main__":
    # worker واحد: الموديلات بتتحمّل في الذاكرة مرة، وQdrant المحلي مبيدعمش أكتر من process
    uvicorn.run("app.main:get_app", factory=True, host=os.getenv("HOST", "127.0.0.1"),
                port=int(os.getenv("PORT", "8000")))
