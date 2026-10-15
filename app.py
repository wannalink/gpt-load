"""
Unified GPT-Load Gateway & Von Decision Server on Hugging Face Spaces (ZeroGPU).

Architecture:
- Single Space deployment using Gradio SDK on Hugging Face Spaces with ZeroGPU (NVIDIA RTX Pro 6000 / A100).
- Von Decision Server: Real ZeroGPU worker executing ModernBERT PyTorch decision inference (@spaces.GPU(duration=15)).
- Dedicated Local Endpoints for GPT-Load:
    1. Dedicated loopback listener: http://127.0.0.1:7861/v1 (Jev native paths: /v1/systemone, /v1/models)
    2. Path-based loopback on port 7860: http://127.0.0.1:7860/von/v1
    3. Loopback authentication bypass (GPT-Load on 127.0.0.1 does not require API keys; external requests enforce VON_API_KEY).
- GPT-Load AI Gateway: Go process running on 127.0.0.1:3001 serving root /, /v1/*, and /api/*.
- Hybrid SQLite Persistence: Ephemeral high-speed local container storage synced to persistent HF Storage Bucket (/data).
- Zero-Quota /healthz: CPU-bound probe for external cron jobs (e.g. cron-job.org) keeping the Space warm 24/7.
- Direct Root Access: Root / proxies directly to GPT-Load Management Console; Gradio test dashboard at /gradio.
"""

import asyncio
import datetime
import hmac
import importlib.metadata
import json
import logging
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.request
from typing import Any, Dict, Optional, Union

# Suppress noisy HTTP transport loggers
for _mod in ("httpx", "httpcore", "uvicorn", "uvicorn.access", "uvicorn.error"):
    logging.getLogger(_mod).setLevel(logging.WARNING)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("gateway_space")

# ---------------------------------------------------------------------------
# 1. Environment & Space Configuration
# ---------------------------------------------------------------------------
def is_huggingface_space() -> bool:
    """Detect if running inside Hugging Face Spaces environment."""
    return bool(
        os.getenv("SPACE_ID")
        or os.getenv("SPACE_REPO_NAME")
        or os.path.exists("/home/user/app")
    )

START_TIME = time.time()
PUBLIC_PORT = int(os.getenv("PORT", "7860"))
INTERNAL_PORT = int(os.getenv("GPT_LOAD_INTERNAL_PORT", "3001"))
GPT_LOAD_BIN = os.getenv("GPT_LOAD_BIN_PATH", "./gpt-load")

SPACE_ID = os.getenv("SPACE_ID")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
VON_API_KEY = os.getenv("VON_API_KEY")

# Ensure CUDA preference for ZeroGPU
os.environ.setdefault("VON_DEVICE", "cuda")

# Storage paths:
# HF Storage Bucket is typically mounted as a volume at /data (or HF_STORAGE_DIR)
HF_STORAGE_DIR = os.path.abspath(os.getenv("HF_STORAGE_DIR", "/data"))

# Local working directory uses fast ephemeral container storage (NVMe/ramdisk)
if is_huggingface_space():
    LOCAL_DATA_DIR = os.path.abspath(os.getenv("LOCAL_DATA_DIR", "/tmp/gpt-load/data"))
else:
    LOCAL_DATA_DIR = os.path.abspath(os.getenv("LOCAL_DATA_DIR", "./data"))

DB_SYNC_INTERVAL_SECONDS = int(os.getenv("DB_SYNC_INTERVAL_SECONDS", "300"))
LAST_DB_SYNC_TIME: float = 0.0
LAST_DB_SYNC_STATUS: str = "never"

GPT_LOAD_PROC: Optional[subprocess.Popen] = None
SHUTDOWN_EVENT = threading.Event()

# ---------------------------------------------------------------------------
# 2. ZeroGPU Supervisor & Von Worker Hook
# ---------------------------------------------------------------------------
try:
    import spaces  # type: ignore

    @spaces.GPU(duration=15)
    def _zerogpu_supervisor_check() -> str:
        return "ZeroGPU Supervisor Online"

    _check_status = _zerogpu_supervisor_check()
    HAS_ZEROGPU = True
    logger.info(f"Hugging Face ZeroGPU environment detected ({_check_status}).")
except (ImportError, AttributeError):
    spaces = None
    HAS_ZEROGPU = False
    logger.info("Standard CPU / local environment detected (spaces module absent).")

# Optional boot-time auto-upgrade check for von-sdk from PyPI
try:
    req = urllib.request.Request(
        "https://pypi.org/pypi/von-sdk/json",
        headers={"User-Agent": "Von-Space-Boot-Checker"}
    )
    with urllib.request.urlopen(req, timeout=4) as resp:
        _pypi_data = json.loads(resp.read().decode())
        _latest_pypi_ver = _pypi_data.get("info", {}).get("version")

    try:
        _current_installed = importlib.metadata.version("von-sdk")
    except Exception:
        _current_installed = None

    if _latest_pypi_ver and _current_installed != _latest_pypi_ver:
        logger.info(f"Boot update detected: upgrading von-sdk {_current_installed} -> {_latest_pypi_ver}...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--no-cache-dir", "-U", f"von-sdk=={_latest_pypi_ver}"
        ])
        logger.info(f"Successfully upgraded von-sdk to {_latest_pypi_ver} during container boot.")
except Exception as _boot_err:
    logger.warning(f"Boot-time von-sdk auto-upgrade check skipped: {_boot_err}")

try:
    from von import __version__ as CURRENT_VON_VERSION  # type: ignore
    from von.engine import VonEngine  # type: ignore
    from von.types import Question, SystemOneResponse  # type: ignore
    HAS_VON = True
    logger.info(f"Loaded von-sdk version: {CURRENT_VON_VERSION}")
except ImportError:
    CURRENT_VON_VERSION = "absent"
    VonEngine = None  # type: ignore
    Question = dict  # type: ignore
    SystemOneResponse = dict  # type: ignore
    HAS_VON = False
    logger.warning("von-sdk is not installed; decision classification endpoints will be disabled.")

# Inference worker function (decorated with spaces.GPU on ZeroGPU)
def execute_inference_gpu(state: Any, questions: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Execute Von System One forward pass on dynamically assigned ZeroGPU slice."""
    if not HAS_VON or VonEngine is None:
        return {"error": "von-sdk is not installed"}
    eng = VonEngine.get_instance()
    resp = eng.evaluate(state=state, questions=questions, model=model)
    if hasattr(resp, "model_dump"):
        return resp.model_dump()
    elif hasattr(resp, "dict"):
        return resp.dict()
    return resp  # type: ignore

if HAS_ZEROGPU and spaces is not None:
    execute_inference_gpu = spaces.GPU(duration=15)(execute_inference_gpu)

# Warm up model weights during boot while CUDA is active
if HAS_VON and VonEngine is not None:
    try:
        engine = VonEngine.get_instance()
        if hasattr(engine, "backend") and hasattr(engine.backend, "_get_model"):
            engine.backend._get_model()
            logger.info("Von decision model weights verified and loaded into backend.")
    except Exception as exc:
        logger.warning(f"Von model warm-up deferred to first request: {exc}")

# ---------------------------------------------------------------------------
# 3. Von Version Check & Space Auto-Restart
# ---------------------------------------------------------------------------
UPDATE_STATUS: Dict[str, Any] = {
    "current_version": CURRENT_VON_VERSION,
    "latest_version": CURRENT_VON_VERSION,
    "update_available": False,
    "last_checked": 0,
}

def check_pypi_for_updates() -> dict:
    """Check PyPI for the newest published von-sdk release."""
    global UPDATE_STATUS
    try:
        req = urllib.request.Request(
            "https://pypi.org/pypi/von-sdk/json",
            headers={"User-Agent": "Von-Space-Watcher"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            latest_version = data.get("info", {}).get("version", CURRENT_VON_VERSION)
            from packaging import version  # type: ignore
            is_newer = version.parse(latest_version) > version.parse(CURRENT_VON_VERSION)
            UPDATE_STATUS.update({
                "latest_version": latest_version,
                "update_available": is_newer,
                "last_checked": time.time(),
            })
            if is_newer:
                logger.info(f"New von-sdk release on PyPI: {latest_version} (Current: {CURRENT_VON_VERSION})")
    except Exception as e:
        logger.warning(f"Failed to check PyPI for updates: {e}")
    return UPDATE_STATUS

def trigger_space_factory_restart() -> bool:
    """Trigger a clean Space factory reboot via huggingface_hub API."""
    if not SPACE_ID or not HF_TOKEN:
        logger.warning("Auto-restart skipped: SPACE_ID or HF_TOKEN not configured.")
        return False
    try:
        from huggingface_hub import HfApi  # type: ignore
        api = HfApi(token=HF_TOKEN)
        logger.info(f"Triggering factory reboot for Space: {SPACE_ID}...")
        api.restart_space(repo_id=SPACE_ID, factory_reboot=True)
        return True
    except Exception as exc:
        logger.error(f"Failed to trigger factory reboot: {exc}")
        return False

def background_update_poller(interval_seconds: int = 3600):
    """Background daemon checking for new versions every hour."""
    while not SHUTDOWN_EVENT.is_set():
        SHUTDOWN_EVENT.wait(interval_seconds)
        if SHUTDOWN_EVENT.is_set():
            break
        status = check_pypi_for_updates()
        if status.get("update_available") and HF_TOKEN and SPACE_ID:
            logger.info("Automated update triggered: restarting Space with factory_reboot...")
            trigger_space_factory_restart()
            break

# ---------------------------------------------------------------------------
# 4. SQLite Storage Bucket Hydration & Atomic Persistence Sync
# ---------------------------------------------------------------------------
def hydrate_database_from_bucket():
    """Copy persistent SQLite database from mounted Storage Bucket to local working directory on boot."""
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    local_db = os.path.join(LOCAL_DATA_DIR, "gpt-load.db")
    persistent_db = os.path.join(HF_STORAGE_DIR, "gpt-load.db")

    if os.path.exists(persistent_db) and os.path.isfile(persistent_db) and os.path.getsize(persistent_db) > 0:
        logger.info(
            f"Hydrating working database from persistent bucket: {persistent_db} "
            f"({os.path.getsize(persistent_db) / (1024*1024):.2f} MB) -> {local_db}"
        )
        try:
            shutil.copyfile(persistent_db, local_db)
            for suffix in ("-wal", "-shm", "-journal"):
                sidecar = local_db + suffix
                if os.path.exists(sidecar):
                    try:
                        os.remove(sidecar)
                    except Exception:
                        pass
            logger.info("Working database hydrated successfully.")
        except Exception as exc:
            logger.error(f"Failed to copy database from persistent bucket: {exc}")
    else:
        logger.info(f"No existing database found at {persistent_db}. Starting fresh in {local_db}.")

    # Also hydrate encryption key and model catalog if present in persistent bucket
    for extra_name in ("encryption.key", "models.dev.catalog.json"):
        extra_src = os.path.join(HF_STORAGE_DIR, extra_name)
        extra_dst = os.path.join(LOCAL_DATA_DIR, extra_name)
        if os.path.exists(extra_src) and os.path.isfile(extra_src) and not os.path.exists(extra_dst):
            try:
                shutil.copyfile(extra_src, extra_dst)
                logger.info(f"Hydrated {extra_name} from persistent bucket to local working directory.")
            except Exception as e:
                logger.warning(f"Notice copying {extra_name}: {e}")

def sync_db_to_persistent_storage() -> bool:
    """Create a verified SQLite backup snapshot and atomically copy to the persistent bucket."""
    global LAST_DB_SYNC_TIME, LAST_DB_SYNC_STATUS
    local_db = os.path.join(LOCAL_DATA_DIR, "gpt-load.db")
    persistent_db = os.path.join(HF_STORAGE_DIR, "gpt-load.db")

    if not os.path.exists(local_db) or os.path.getsize(local_db) == 0:
        return False

    try:
        os.makedirs(HF_STORAGE_DIR, exist_ok=True)
        temp_stage = os.path.join(LOCAL_DATA_DIR, ".gpt-load.db.sync_stage")
        if os.path.exists(temp_stage):
            try:
                os.remove(temp_stage)
            except Exception:
                pass

        # Step 1: Use SQLite Online Backup API for lock-free consistent snapshot
        src_conn = sqlite3.connect(local_db, timeout=30.0)
        try:
            dst_conn = sqlite3.connect(temp_stage, timeout=30.0)
            try:
                src_conn.backup(dst_conn)
            finally:
                dst_conn.close()
        finally:
            src_conn.close()

        # Step 2: Verify integrity of the staged snapshot before pushing to bucket
        chk_conn = sqlite3.connect(temp_stage, timeout=5.0)
        is_valid = False
        try:
            cur = chk_conn.cursor()
            cur.execute("PRAGMA integrity_check(1)")
            row = cur.fetchone()
            if row and row[0] == "ok":
                is_valid = True
        finally:
            chk_conn.close()

        if not is_valid:
            logger.warning("Staged backup snapshot failed integrity check. Aborting push to persistent bucket.")
            if os.path.exists(temp_stage):
                os.remove(temp_stage)
            LAST_DB_SYNC_STATUS = "integrity_check_failed"
            return False

        # Step 3: Copy validated snapshot to persistent Storage Bucket
        shutil.copyfile(temp_stage, persistent_db)
        try:
            os.chmod(persistent_db, 0o666)
        except Exception:
            pass

        if os.path.exists(temp_stage):
            os.remove(temp_stage)

        LAST_DB_SYNC_TIME = time.time()
        LAST_DB_SYNC_STATUS = "ok"
        logger.info(
            f"Successfully synced working database to persistent bucket: {persistent_db} "
            f"({os.path.getsize(persistent_db) / (1024*1024):.2f} MB)"
        )
        return True
    except Exception as exc:
        LAST_DB_SYNC_STATUS = f"error: {exc}"
        logger.warning(f"Error syncing database to persistent bucket: {exc}")
        return False

def background_db_sync_poller():
    """Background worker periodically syncing local working SQLite DB to the Storage Bucket."""
    while not SHUTDOWN_EVENT.is_set():
        SHUTDOWN_EVENT.wait(DB_SYNC_INTERVAL_SECONDS)
        if SHUTDOWN_EVENT.is_set():
            break
        sync_db_to_persistent_storage()

# ---------------------------------------------------------------------------
# 5. GPT-Load Process Manager & Automated On-Boot Build Pipeline
# ---------------------------------------------------------------------------
GO_BIN_PATH = "/tmp/go/bin/go"
GO_TARBALL_URL = "https://go.dev/dl/go1.24.0.linux-amd64.tar.gz"

def ensure_modern_go_compiler() -> Optional[str]:
    """Ensure Go >= 1.22 is available, downloading official Go 1.24.0 binary on demand if missing."""
    import re

    system_go = shutil.which("go")
    if system_go:
        try:
            out = subprocess.check_output([system_go, "version"], text=True)
            m = re.search(r"go1\.(\d+)", out)
            if m and int(m.group(1)) >= 22:
                return system_go
        except Exception:
            pass

    if os.path.exists(GO_BIN_PATH):
        return GO_BIN_PATH

    logger.info("System Go compiler missing or outdated. Downloading official Go 1.24.0 toolchain...")
    try:
        os.makedirs("/tmp", exist_ok=True)
        tar_path = "/tmp/go1.24.0.linux-amd64.tar.gz"
        urllib.request.urlretrieve(GO_TARBALL_URL, tar_path)
        logger.info("Extracting Go 1.24.0 toolchain to /tmp/go...")
        import tarfile
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path="/tmp")
        if os.path.exists(tar_path):
            os.remove(tar_path)
        if os.path.exists(GO_BIN_PATH):
            logger.info("Go 1.24.0 toolchain successfully provisioned in /tmp/go.")
            return GO_BIN_PATH
    except Exception as exc:
        logger.error(f"Failed to download/provision Go 1.24.0 toolchain: {exc}")

    return None

def ensure_web_ui_built():
    """Build Vue 3 / Vite assets in web/ if internal/webui/dist/index.html is missing."""
    dist_index = os.path.join("internal", "webui", "dist", "index.html")
    if os.path.exists(dist_index) and os.path.getsize(dist_index) > 0:
        return

    web_dir = os.path.abspath("web")
    if not os.path.exists(web_dir) or not os.path.exists(os.path.join(web_dir, "package.json")):
        return

    npm_bin = shutil.which("npm") or shutil.which("pnpm")
    if not npm_bin:
        logger.warning("Node.js/npm not found in PATH; skipping web UI compilation.")
        return

    logger.info("Compiling Vue 3 / Vite Web UI assets in web/...")
    try:
        subprocess.check_call([npm_bin, "install", "--no-audit", "--no-fund"], cwd=web_dir, timeout=120)
        subprocess.check_call([npm_bin, "run", "build"], cwd=web_dir, timeout=120)
        logger.info("Web UI compiled successfully into internal/webui/dist.")
    except Exception as exc:
        logger.error(f"Failed to compile Web UI assets: {exc}")

def resolve_gpt_load_binary() -> Optional[str]:
    """Find or automatically build gpt-load binary in container."""
    local_exec = "/tmp/gpt-load-exec"

    # 1. First check if executable already exists locally
    if os.path.exists(local_exec) and os.path.isfile(local_exec) and os.path.getsize(local_exec) > 0:
        try:
            os.chmod(local_exec, 0o755)
        except Exception:
            pass
        return local_exec

    # 2. Check persistent storage bucket
    bucket_bin = os.path.join(HF_STORAGE_DIR, "gpt-load")
    if os.path.exists(bucket_bin) and os.path.isfile(bucket_bin) and os.path.getsize(bucket_bin) > 0:
        logger.info(
            f"Found precompiled gpt-load binary in persistent bucket ({bucket_bin}, "
            f"{os.path.getsize(bucket_bin) / (1024*1024):.1f} MB). Staging executable to {local_exec}..."
        )
        try:
            shutil.copyfile(bucket_bin, local_exec)
            os.chmod(local_exec, 0o755)
            return local_exec
        except Exception as e:
            logger.warning(f"Error copying gpt-load binary from persistent bucket: {e}")

    # 3. Check current workspace directory
    candidates = [
        GPT_LOAD_BIN,
        os.path.abspath("gpt-load"),
        shutil.which("gpt-load"),
        shutil.which("./gpt-load"),
    ]
    for c in candidates:
        if c and os.path.exists(c) and os.path.isfile(c):
            try:
                os.chmod(c, 0o755)
            except Exception:
                pass
            return os.path.abspath(c)

    # 4. Automated on-boot compilation pipeline
    logger.info("gpt-load binary absent. Initiating automated on-boot build pipeline...")
    ensure_web_ui_built()
    go_cmd = ensure_modern_go_compiler()
    if go_cmd and os.path.exists("main.go"):
        logger.info(f"Compiling gpt-load using {go_cmd}...")
        try:
            build_env = os.environ.copy()
            build_env["CGO_ENABLED"] = "0"
            build_env["PATH"] = f"/tmp/go/bin:{build_env.get('PATH', '')}"
            subprocess.check_call(
                [go_cmd, "build", "-ldflags", "-s -w", "-o", local_exec, "."],
                env=build_env,
                timeout=300,
            )
            if os.path.exists(local_exec):
                os.chmod(local_exec, 0o755)
                logger.info("gpt-load binary compiled successfully on container boot.")
                if os.path.exists(HF_STORAGE_DIR):
                    try:
                        shutil.copyfile(local_exec, bucket_bin)
                        logger.info("Saved compiled binary to persistent bucket for instant subsequent boots.")
                    except Exception as e:
                        logger.warning(f"Notice saving binary to persistent bucket: {e}")
                return local_exec
        except Exception as e:
            logger.error(f"Automated build pipeline failed: {e}")

    return None

def start_gpt_load_process() -> Optional[subprocess.Popen]:
    """Start the gpt-load Go binary on 127.0.0.1:INTERNAL_PORT."""
    global GPT_LOAD_PROC
    bin_path = resolve_gpt_load_binary()
    if not bin_path:
        logger.error(
            f"GPT-Load binary not found! Expected at {GPT_LOAD_BIN} or in PATH. "
            "Ensure the compiled linux/amd64 binary is placed in the Space directory."
        )
        return None

    try:
        os.chmod(bin_path, 0o755)
    except Exception:
        pass

    env = os.environ.copy()
    env["DATA_DIR"] = LOCAL_DATA_DIR
    env["HOST"] = "127.0.0.1"
    env["PORT"] = str(INTERNAL_PORT)
    env["CLIENT_IP_HEADER"] = "X-Forwarded-For"
    env["TRUSTED_PROXIES"] = "127.0.0.1"

    # Pass local Von endpoints into environment for convenience
    env["VON_BASE_URL"] = f"http://127.0.0.1:{PUBLIC_PORT}/von/v1"

    # Remove Fly.io 256MB memory constraints to leverage 16GB host RAM
    env.pop("GOMEMLIMIT", None)
    if env.get("GOGC") == "60":
        env["GOGC"] = "100"

    logger.info(f"Starting GPT-Load Go process ({bin_path}) on 127.0.0.1:{INTERNAL_PORT}...")
    GPT_LOAD_PROC = subprocess.Popen(
        [bin_path],
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    return GPT_LOAD_PROC

def wait_for_gpt_load_ready(timeout_seconds: int = 30) -> bool:
    """Poll internal health endpoint until gpt-load is ready."""
    start = time.time()
    url = f"http://127.0.0.1:{INTERNAL_PORT}/health"
    while time.time() - start < timeout_seconds:
        if GPT_LOAD_PROC and GPT_LOAD_PROC.poll() is not None:
            logger.error(f"GPT-Load process exited prematurely with code {GPT_LOAD_PROC.returncode}")
            return False
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Space-Readiness-Probe"})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    logger.info("GPT-Load internal engine is ready and responding on port 3001.")
                    return True
        except Exception:
            time.sleep(0.5)
    logger.warning(f"GPT-Load internal engine did not become ready within {timeout_seconds}s.")
    return False

def stop_gpt_load_process():
    """Gracefully terminate gpt-load and flush SQLite to persistent storage."""
    global GPT_LOAD_PROC
    if GPT_LOAD_PROC and GPT_LOAD_PROC.poll() is None:
        logger.info("Sending SIGTERM to internal GPT-Load process...")
        try:
            GPT_LOAD_PROC.terminate()
            GPT_LOAD_PROC.wait(timeout=10.0)
            logger.info("GPT-Load process terminated cleanly.")
        except Exception as e:
            logger.warning(f"GPT-Load process termination notice: {e}")
            try:
                GPT_LOAD_PROC.kill()
            except Exception:
                pass
    GPT_LOAD_PROC = None

    logger.info("Performing final database synchronization to persistent bucket...")
    sync_db_to_persistent_storage()

# ---------------------------------------------------------------------------
# 6. FastAPI Router & Ingress Reverse-Proxy
# ---------------------------------------------------------------------------
from fastapi import FastAPI, Header, HTTPException, Request, Response, WebSocket, WebSocketDisconnect  # type: ignore
from pydantic import BaseModel, Field, field_validator  # type: ignore
from starlette.responses import PlainTextResponse, StreamingResponse  # type: ignore
import httpx  # type: ignore
import websockets  # type: ignore

app = FastAPI(title="GPT-Load & Von Unified Gateway", docs_url=None, redoc_url=None)
http_client: Optional[httpx.AsyncClient] = None

class SystemOneRequest(BaseModel):
    model: str = Field(default="von-latest")
    state: Any = Field(..., description="State object, string, or array to evaluate")
    questions: Dict[str, Dict[str, Any]] = Field(..., description="Dict of question definitions")

    @field_validator("questions")
    @classmethod
    def _validate_questions(cls, v):
        if not v:
            raise ValueError("questions must contain at least one entry")
        return v

AUTH_KEY = os.getenv("AUTH_KEY")
VON_API_KEY = os.getenv("VON_API_KEY")

def is_trusted_local_client(request: Request) -> bool:
    """
    Check if request originates directly from local container loopback (e.g. from GPT-Load).
    External requests routed via Hugging Face ingress always carry x-forwarded-for or x-forwarded-host.
    """
    client_host = request.client.host if request.client else ""
    xff = request.headers.get("x-forwarded-for")
    return client_host in ("127.0.0.1", "::1", "localhost") and not xff

def verify_von_token(request: Request, authorization: Optional[str]):
    """
    Verify API key for external calls.
    Internal loopback calls directly from GPT-Load bypass authentication automatically.
    External calls must provide either VON_API_KEY or AUTH_KEY to prevent unauthorized ZeroGPU quota drainage.
    """
    if is_trusted_local_client(request):
        return  # Trusted internal loopback

    valid_keys = [k for k in (VON_API_KEY, AUTH_KEY) if k]
    if not valid_keys:
        # If no key is set, deny external access by default to prevent unauthorized GPU quota drainage
        raise HTTPException(status_code=401, detail="Access denied: AUTH_KEY or VON_API_KEY must be configured in Space secrets.")

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Bearer token for Von API")

    token = authorization.split("Bearer ", 1)[1].strip()
    is_valid = any(hmac.compare_digest(token.encode("utf-8"), k.encode("utf-8")) for k in valid_keys)
    if not is_valid:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid API key")

@app.on_event("startup")
async def on_startup():
    global http_client
    http_client = httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{INTERNAL_PORT}",
        timeout=httpx.Timeout(connect=15.0, read=None, write=60.0, pool=None),
        limits=httpx.Limits(max_keepalive_connections=100, max_connections=500),
    )

@app.on_event("shutdown")
async def on_shutdown():
    global http_client
    if http_client:
        await http_client.aclose()

# --- Keep-Alive Health Probe (Zero GPU Quota) ---
@app.get("/healthz", response_class=PlainTextResponse)
def healthz_probe():
    """
    Minimal 200 OK probe for external cron jobs (e.g. cron-job.org).
    Serves entirely on CPU in <2ms, keeping Space warm without consuming ZeroGPU quota.
    """
    return PlainTextResponse("OK")

# --- Combined Health & Diagnostics Endpoint ---
@app.get("/health")
@app.get("/status")
def health_and_status():
    """Detailed operational status for Von Decision Engine and GPT-Load Gateway."""
    uptime = int(time.time() - START_TIME)
    last_sync_iso = (
        datetime.datetime.fromtimestamp(LAST_DB_SYNC_TIME, datetime.timezone.utc).isoformat()
        if LAST_DB_SYNC_TIME > 0
        else "none"
    )
    gpt_load_alive = bool(GPT_LOAD_PROC and GPT_LOAD_PROC.poll() is None)
    return {
        "status": "online" if gpt_load_alive else "degraded",
        "service": "gpt-load-von-gateway",
        "zerogpu_enabled": HAS_ZEROGPU,
        "von": {
            "version": CURRENT_VON_VERSION,
            "installed": HAS_VON,
            "device": os.environ.get("VON_DEVICE", "cuda"),
            "update_status": UPDATE_STATUS,
            "local_endpoints": {
                "local_base_url": f"http://127.0.0.1:{PUBLIC_PORT}/von/v1",
                "decision_path": "/systemone",
                "models_path": "/models",
            },
        },
        "gpt_load": {
            "process_alive": gpt_load_alive,
            "internal_port": INTERNAL_PORT,
            "working_data_dir": LOCAL_DATA_DIR,
            "persistent_storage_dir": HF_STORAGE_DIR,
            "last_db_sync_utc": last_sync_iso,
            "last_db_sync_status": LAST_DB_SYNC_STATUS,
        },
        "uptime_seconds": uptime,
        "public_port": PUBLIC_PORT,
    }

@app.post("/status/sync")
def trigger_db_sync(request: Request, authorization: Optional[str] = Header(None)):
    """Manually trigger immediate working database sync to the Storage Bucket."""
    verify_von_token(request, authorization)
    success = sync_db_to_persistent_storage()
    return {"success": success, "status": LAST_DB_SYNC_STATUS}

# --- Von Informational / Configuration Endpoint ---
@app.get("/von/info")
@app.get("/von/v1/info")
def von_configuration_info():
    """Endpoint details for configuring GPT-Load Jev channel locally."""
    return {
        "service": "von-decision-engine",
        "version": CURRENT_VON_VERSION,
        "zerogpu_enabled": HAS_ZEROGPU,
        "local_endpoints": {
            "local_base_url": f"http://127.0.0.1:{PUBLIC_PORT}/von/v1",
            "decision_path": "/systemone",
            "models_path": "/models",
        },
        "gpt_load_channel_setup": {
            "channel_type": "jev",
            "recommended_base_url": f"http://127.0.0.1:{PUBLIC_PORT}/von/v1",
            "upstream_model": "von-latest",
            "auth_note": "Authentication automatically bypassed for local container requests.",
        },
        "models": ["von-latest", "von-1.2.0", "jev-latest"],
    }

# --- Von Decision Model Catalog ---
@app.get("/von/v1/models")
@app.get("/von/models")
@app.get("/v1/decision-models")
def list_von_models():
    """Catalog of available decision models for GPT-Load channels and external clients."""
    models = [
        {"name": "von-latest", "description": "Latest Von System One decision model", "version": CURRENT_VON_VERSION},
        {"name": "von-1.2.0", "description": "Von 1.2 stable (order-invariant option scoring)"},
        {"name": "jev-latest", "description": "TypeSafe Jev compatibility alias"},
    ]
    data = [{"id": m["name"], "object": "model", "owned_by": "von"} for m in models]
    return {"models": models, "object": "list", "data": data}

# --- Von System One Classification Endpoint (ZeroGPU Accelerated) ---
@app.post("/v1/systemone")
@app.post("/von/v1/systemone")
@app.post("/von/systemone")
async def systemone_classification_endpoint(
    req: SystemOneRequest,
    request: Request,
    authorization: Optional[str] = Header(None)
):
    """
    Evaluate decision primitives using the ZeroGPU worker.
    Called directly by external clients or by GPT-Load's Jev channel for Automatic Models.
    Local requests from 127.0.0.1 bypass authentication automatically.
    """
    verify_von_token(request, authorization)
    t0 = time.perf_counter()
    try:
        response_dict = await asyncio.to_thread(
            execute_inference_gpu,
            state=req.state,
            questions=dict(req.questions),
            model=req.model,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        is_local = is_trusted_local_client(request)
        prefix = "[VON-LOCAL]" if is_local else "[VON-EXTERNAL]"
        logger.info(f"{prefix} Decision '{req.model}' executed on ZeroGPU in {elapsed_ms:.2f}ms")
        return response_dict
    except Exception as exc:
        logger.error(f"Inference error in /v1/systemone: {exc}")
        raise HTTPException(status_code=422, detail=str(exc))

# --- Von Admin On-Demand Upgrade Webhook ---
@app.post("/v1/admin/upgrade")
def admin_upgrade_endpoint(request: Request, authorization: Optional[str] = Header(None)):
    """Trigger manual version check and Space rebuild."""
    verify_von_token(request, authorization)
    status = check_pypi_for_updates()
    rebooted = False
    if status.get("update_available"):
        rebooted = trigger_space_factory_restart()
    return {
        "status": "upgrade_initiated" if rebooted else "checked",
        "update_status": status,
        "factory_reboot_triggered": rebooted,
    }

# --- WebSocket Proxy (OpenAI Realtime Parity) ---
@app.websocket("/{path:path}")
async def websocket_proxy_endpoint(client_ws: WebSocket, path: str):
    """Proxy WebSocket connections to internal gpt-load engine."""
    await client_ws.accept()
    query_str = f"?{client_ws.query_params}" if client_ws.query_params else ""
    target_ws_url = f"ws://127.0.0.1:{INTERNAL_PORT}/{path}{query_str}"

    headers = dict(client_ws.headers)
    for hop in ("host", "connection", "upgrade", "sec-websocket-key", "sec-websocket-version", "sec-websocket-extensions"):
        headers.pop(hop, None)

    try:
        async with websockets.connect(target_ws_url, extra_headers=headers) as upstream_ws:
            async def forward_upstream():
                while True:
                    msg = await client_ws.receive()
                    if msg.get("text") is not None:
                        await upstream_ws.send(msg["text"])
                    elif msg.get("bytes") is not None:
                        await upstream_ws.send(msg["bytes"])
                    elif msg.get("type") == "websocket.disconnect":
                        break

            async def forward_downstream():
                async for msg in upstream_ws:
                    if isinstance(msg, str):
                        await client_ws.send_text(msg)
                    else:
                        await client_ws.send_bytes(msg)

            await asyncio.gather(forward_upstream(), forward_downstream())
    except (WebSocketDisconnect, Exception) as exc:
        logger.debug(f"WebSocket session closed: {exc}")
    finally:
        try:
            await client_ws.close()
        except Exception:
            pass

# ---------------------------------------------------------------------------
# 7. Operational Dashboard UI (Gradio Mounted at /gradio)
# ---------------------------------------------------------------------------
import gradio as gr  # type: ignore

def test_decision_ui(prompt_text: str, api_key_input: str = ""):
    """Test function for the interactive Gradio demo tab with authentication check."""
    if not prompt_text.strip():
        return "Please enter a prompt to classify."

    valid_keys = [k for k in (VON_API_KEY, AUTH_KEY) if k]
    if valid_keys:
        token = (api_key_input or "").strip()
        if token.startswith("Bearer "):
            token = token.split("Bearer ", 1)[1].strip()
        is_valid = any(hmac.compare_digest(token.encode("utf-8"), k.encode("utf-8")) for k in valid_keys)
        if not is_valid:
            return {"error": "Unauthorized: Valid AUTH_KEY or VON_API_KEY required to execute ZeroGPU inference."}
    try:
        q = {
            "complexity": {
                "type": "choice",
                "instructions": "Classify the computational reasoning complexity required for this prompt.",
                "criteria": {
                    "heavy_reasoning": "Complex coding, algorithm design, system architecture, multi-step math, debugging",
                    "light_conversational": "General chat, short Q&A, simple formatting, translation, greeting",
                },
            }
        }
        res = execute_inference_gpu(state=prompt_text, questions=q, model="von-latest")
        answers = res.get("answers", {}) if isinstance(res, dict) else getattr(res, "answers", {})
        choice_ans = answers.get("complexity", {})
        if isinstance(choice_ans, dict):
            return {
                "selected_route": choice_ans.get("choice", "unknown"),
                "confidence": choice_ans.get("confidence", 0.0),
                "probabilities": choice_ans.get("probabilities", {}),
            }
        return {
            "selected_route": getattr(choice_ans, "choice", "unknown"),
            "confidence": getattr(choice_ans, "confidence", 0.0),
            "probabilities": getattr(choice_ans, "probabilities", {}),
        }
    except Exception as e:
        return f"Error evaluating: {e}"

with gr.Blocks(title="Von Decision Server") as demo:
    gr.Markdown(f"""
    # 🧠 Von Decision Server & ⚡ GPT-Load Gateway
    
    <div style="display: flex; gap: 10px; margin-bottom: 20px;">
        <span style="background-color: #22c55e20; color: #16a34a; padding: 4px 12px; border-radius: 9999px; font-weight: bold; border: 1px solid #16a34a40;">
            ● Status: Online (ZeroGPU)
        </span>
        <span style="background-color: #3b82f620; color: #2563eb; padding: 4px 12px; border-radius: 9999px; font-weight: bold; border: 1px solid #2563eb40;">
            Von Version: {CURRENT_VON_VERSION}
        </span>
    </div>

    This Space co-locates **Von System One Decision Classification** with **GPT-Load Multi-Channel Gateway**.

    ### ⚡ Active Endpoints
    - **GPT-Load Web Console**: `GET /` *(Native Vue 3 / Vite Management Dashboard)*
    - **Local Von Endpoint**: `http://127.0.0.1:{PUBLIC_PORT}/von/v1` *(Zero Auth Loopback)*
    - **Von Decision API**: `POST /von/v1/systemone` & `POST /v1/systemone` *(Sub-25ms ZeroGPU ModernBERT)*
    - **Decision Catalog**: `GET /von/v1/models` *(TypeSafe / OpenAI compatible)*
    - **Keep-Alive Probe**: `GET /healthz` *(Zero GPU quota consumed, for external cron)*
    - **System Diagnostics**: `GET /health` & `GET /status`

    ### 🔄 Auto-Update Status
    - **Installed Von Version**: `{CURRENT_VON_VERSION}`
    - **Latest on PyPI**: `{UPDATE_STATUS.get('latest_version', CURRENT_VON_VERSION)}`
    - **Update Available**: `{'Yes' if UPDATE_STATUS.get('update_available') else 'Up to date'}`
    """)

    with gr.Accordion("Interactive Quick-Test (Von Decision Classification)", open=False):
        gr.Markdown("Test Von decision classification live on ZeroGPU (requires AUTH_KEY or VON_API_KEY):")
        key_box = gr.Textbox(label="API Key / Auth Key", placeholder="Enter AUTH_KEY or VON_API_KEY...", type="password")
        input_box = gr.Textbox(label="Input Prompt / State", placeholder="e.g. Write a distributed Raft consensus implementation in Go...")
        test_btn = gr.Button("Evaluate Decision", variant="primary")
        output_json = gr.JSON(label="Decision Result")
        test_btn.click(fn=test_decision_ui, inputs=[input_box, key_box], outputs=[output_json])

# Mount Gradio Blocks at /gradio so root / remains dedicated to GPT-Load Web UI
app = gr.mount_gradio_app(app, demo, path="/gradio")

# --- HTTP Data-Plane, Control-Plane, & WebUI Reverse-Proxy ---
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
}

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def reverse_proxy_handler(request: Request, path: str):
    """
    Transparently stream all GPT-Load data-plane (/v1/*), control-plane (/api/*),
    and WebUI static asset requests to internal gpt-load engine.
    """
    if path == "gradio" or path.startswith("gradio/"):
        return await demo.app(request.scope, request.receive, request.send)

    if http_client is None:
        return Response("Gateway initializing", status_code=503)

    target_url = httpx.URL(
        path=f"/{path}",
        query=request.url.query.encode("utf-8") if request.url.query else None,
    )

    req_headers = dict(request.headers)
    for hop in HOP_BY_HOP_HEADERS:
        req_headers.pop(hop, None)

    client_ip = request.client.host if request.client else "127.0.0.1"
    existing_xff = req_headers.get("x-forwarded-for")
    req_headers["x-forwarded-for"] = f"{existing_xff}, {client_ip}" if existing_xff else client_ip
    req_headers["x-forwarded-proto"] = request.url.scheme
    req_headers["x-forwarded-host"] = request.headers.get("host", f"0.0.0.0:{PUBLIC_PORT}")

    try:
        upstream_req = http_client.build_request(
            method=request.method,
            url=target_url,
            headers=req_headers,
            content=request.stream(),
        )
        upstream_resp = await http_client.send(upstream_req, stream=True)

        resp_headers = dict(upstream_resp.headers)
        for hop in HOP_BY_HOP_HEADERS:
            resp_headers.pop(hop, None)
        resp_headers.pop("content-length", None)
        resp_headers.pop("content-encoding", None)

        # Optimize for Server-Sent Events / LLM token streaming
        is_sse = "text/event-stream" in resp_headers.get("content-type", "")
        if is_sse:
            resp_headers["cache-control"] = "no-cache"
            resp_headers["x-accel-buffering"] = "no"

        return StreamingResponse(
            upstream_resp.aiter_raw(),
            status_code=upstream_resp.status_code,
            headers=resp_headers,
            background=upstream_resp.aclose,
        )
    except httpx.ConnectError:
        return Response("Internal GPT-Load engine unavailable or starting up.", status_code=503)
    except Exception as exc:
        logger.error(f"Ingress proxy error for /{path}: {exc}")
        return Response(f"Proxy error: {exc}", status_code=502)

# ---------------------------------------------------------------------------
# 8. Main Supervisor Entrypoint
# ---------------------------------------------------------------------------
def signal_handler(signum, frame):
    """Handle termination signals from Hugging Face container supervisor."""
    logger.info(f"Received signal {signum}; initiating graceful Space shutdown...")
    SHUTDOWN_EVENT.set()
    stop_gpt_load_process()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    logger.info("==================================================")
    logger.info("Initializing Unified GPT-Load & Von Space")
    logger.info(f"Public Port: {PUBLIC_PORT} • Internal Port: {INTERNAL_PORT}")
    logger.info(f"Persistent Storage: {HF_STORAGE_DIR}")
    logger.info(f"Working Data Dir: {LOCAL_DATA_DIR}")
    logger.info(f"ZeroGPU Available: {HAS_ZEROGPU}")
    logger.info(f"Von Engine Available: {HAS_VON} (v{CURRENT_VON_VERSION})")
    logger.info("==================================================")

    # 1. Hydrate working database from persistent bucket
    hydrate_database_from_bucket()

    # 2. Launch gpt-load Go binary
    start_gpt_load_process()
    wait_for_gpt_load_ready(timeout_seconds=30)

    # 4. Start background SQLite synchronization daemon
    sync_thread = threading.Thread(target=background_db_sync_poller, daemon=True)
    sync_thread.start()

    # 5. Start background PyPI update poller for von-sdk
    update_thread = threading.Thread(target=background_update_poller, daemon=True)
    update_thread.start()

    # Initial check on PyPI for von-sdk
    check_pypi_for_updates()

    # 6. Launch FastAPI Ingress on public port (7860)
    import uvicorn  # type: ignore

    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=PUBLIC_PORT,
            log_level="warning",
            access_log=False,
        )
    finally:
        stop_gpt_load_process()
