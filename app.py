"""
GPT-Load Gateway on Hugging Face Spaces (ZeroGPU / Gradio).

Architecture:
- Single Space deployment using Gradio SDK on Hugging Face Spaces with ZeroGPU.
- GPT-Load AI Gateway: Go process running on 127.0.0.1:3001 serving root /, /v1/*, and /api/*.
- Hybrid SQLite Persistence: Ephemeral high-speed local container storage synced to persistent HF Storage Bucket (/data).
- Zero-Quota /healthz: CPU-bound probe for external cron jobs (e.g. cron-job.org) keeping the Space warm 24/7.
- Direct Root Access: Root / proxies directly to GPT-Load Management Console; Gradio status dashboard at /gradio.
"""

import asyncio
import datetime
import hmac
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

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

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

def detect_space_id() -> Optional[str]:
    """Dynamically determine Hugging Face Space ID without hardcoded fallbacks."""
    for var in ("SPACE_ID", "SPACE_REPO_NAME", "HF_SPACE_ID"):
        val = os.getenv(var)
        if val:
            return val
    try:
        url = subprocess.check_output(["git", "remote", "get-url", "origin"], text=True).strip()
        import re
        m = re.search(r"huggingface\.co/spaces/([^/]+/[^/.]+)", url)
        if m:
            clean = m.group(1)
            if clean.endswith(".git"):
                clean = clean[:-4]
            return clean
    except Exception:
        pass
    return None

SPACE_ID = detect_space_id()
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

# Storage paths:
# HF Storage Bucket is typically mounted as a volume at /data (or HF_STORAGE_DIR)
HF_STORAGE_DIR = os.path.abspath(os.getenv("HF_STORAGE_DIR", "/data"))

# Local working directory uses fast ephemeral container storage (NVMe/ramdisk)
if is_huggingface_space():
    LOCAL_DATA_DIR = os.path.abspath(os.getenv("LOCAL_DATA_DIR", "/tmp/gpt-load/data"))
else:
    LOCAL_DATA_DIR = os.path.abspath(os.getenv("LOCAL_DATA_DIR", "./data"))

# Persistent and local log files alongside gpt-load.db
LOCAL_LOG_FILE = os.path.join(LOCAL_DATA_DIR, "gpt-load.log")
PERSISTENT_LOG_FILE = os.path.join(HF_STORAGE_DIR, "gpt-load.log")

def setup_space_logging():
    """Configure unbuffered logging to console and persistent log file in attached storage."""
    try:
        os.makedirs(HF_STORAGE_DIR, exist_ok=True)
        os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    except Exception:
        pass

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    for log_path in {LOCAL_LOG_FILE, PERSISTENT_LOG_FILE}:
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            f_handler = logging.FileHandler(log_path, encoding="utf-8")
            f_handler.setLevel(logging.INFO)
            f_handler.setFormatter(formatter)
            root_logger.addHandler(f_handler)
        except Exception as exc:
            sys.stderr.write(f"Notice setting up log file {log_path}: {exc}\n")

    for _mod in ("httpx", "httpcore", "uvicorn.access", "uvicorn.error"):
        logging.getLogger(_mod).setLevel(logging.WARNING)

setup_space_logging()
logger = logging.getLogger("gateway_space")

DB_SYNC_INTERVAL_SECONDS = int(os.getenv("DB_SYNC_INTERVAL_SECONDS", "300"))
LAST_DB_SYNC_TIME: float = 0.0
LAST_DB_SYNC_STATUS: str = "never"

GPT_LOAD_PROC: Optional[subprocess.Popen] = None
SHUTDOWN_EVENT = threading.Event()

# ---------------------------------------------------------------------------
# 2. ZeroGPU Supervisor Check
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

# ---------------------------------------------------------------------------
# 3. SQLite Storage Bucket Hydration & Atomic Persistence Sync
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

        # Verify integrity of the hydrated working database
        try:
            test_conn = sqlite3.connect(f"file:{local_db}?mode=ro", uri=True, timeout=10.0)
            res = test_conn.execute("PRAGMA integrity_check(1)").fetchone()
            test_conn.close()
            if not res or res[0] != "ok":
                logger.warning("Working database failed integrity check; re-copying clean snapshot from persistent bucket...")
                shutil.copyfile(persistent_db, local_db)
        except Exception as e:
            logger.warning(f"Database hydration verification notice: {e}")
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

        # Step 1: Use SQLite Online Backup API on a read-only URI connection.
        # mode=ro ensures Python never modifies WAL headers, acquires write locks,
        # or triggers checkpoints, preventing SQLITE_IOERR_SHORT_READ (522) conflicts with Go.
        src_conn = sqlite3.connect(f"file:{local_db}?mode=ro", uri=True, timeout=30.0)
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
        logger.debug(
            f"Successfully synced working database to persistent bucket: {persistent_db} "
            f"({os.path.getsize(persistent_db) / (1024*1024):.2f} MB)"
        )
        return True
    except Exception as exc:
        LAST_DB_SYNC_STATUS = f"error: {exc}"
        logger.warning(f"Error syncing database to persistent bucket: {exc}")
        return False

def sync_logs_to_persistent_storage() -> bool:
    """Flush and synchronize active log file to persistent storage alongside database."""
    for handler in logging.getLogger().handlers:
        try:
            handler.flush()
        except Exception:
            pass

    if not os.path.exists(LOCAL_LOG_FILE) or os.path.getsize(LOCAL_LOG_FILE) == 0:
        return False

    if os.path.abspath(LOCAL_LOG_FILE) == os.path.abspath(PERSISTENT_LOG_FILE):
        return True

    try:
        os.makedirs(HF_STORAGE_DIR, exist_ok=True)
        shutil.copyfile(LOCAL_LOG_FILE, PERSISTENT_LOG_FILE)
        try:
            os.chmod(PERSISTENT_LOG_FILE, 0o666)
        except Exception:
            pass
        return True
    except Exception as exc:
        logger.debug(f"Notice syncing log file to persistent bucket: {exc}")
        return False

def background_db_sync_poller():
    """Background worker periodically syncing local working SQLite DB and logs to the Storage Bucket."""
    while not SHUTDOWN_EVENT.is_set():
        SHUTDOWN_EVENT.wait(DB_SYNC_INTERVAL_SECONDS)
        if SHUTDOWN_EVENT.is_set():
            break
        sync_db_to_persistent_storage()
        sync_logs_to_persistent_storage()

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

    # Automatically generate tiny 1x1 webp placeholder assets if missing so vite build succeeds
    import base64
    tiny_webp_bytes = base64.b64decode('UklGRiQAAABXRUJQVlA4TAgAAAAwAQCdASoBAAEAAQwA/sAANAA=')
    webp_paths = [
        "web/src/frontends/modern/assets/brand/login-mascot-body.webp",
        "web/src/frontends/modern/assets/brand/login-mascot-eye.webp",
        "web/src/frontends/modern/assets/brand/login-mascot-tail.webp",
        "web/src/frontends/classic/assets/channels/cpa.webp",
        "web/src/frontends/classic/assets/channels/jev.webp",
        "web/src/frontends/modern/assets/channels/cpa.webp",
        "web/src/frontends/modern/assets/channels/jev.webp"
    ]
    for wp in webp_paths:
        full_wp = os.path.abspath(wp)
        if not os.path.exists(full_wp):
            try:
                os.makedirs(os.path.dirname(full_wp), exist_ok=True)
                with open(full_wp, "wb") as wf:
                    wf.write(tiny_webp_bytes)
            except Exception:
                pass

    # Ensure dist directory and fallback index.html always exist so go:embed never fails
    dist_dir = os.path.dirname(os.path.join("internal", "webui", "dist", "index.html"))
    os.makedirs(dist_dir, exist_ok=True)
    dist_index = os.path.join(dist_dir, "index.html")
    if not os.path.exists(dist_index) or os.path.getsize(dist_index) == 0:
        try:
            with open(dist_index, "w", encoding="utf-8") as f:
                f.write('<!doctype html><html lang="zh-CN"><head><meta charset="UTF-8"><title>GPT-Load</title></head><body><h1>GPT-Load</h1><p>Web UI fallback</p></body></html>')
        except Exception:
            pass

    web_dir = os.path.abspath("web")
    if not os.path.exists(web_dir) or not os.path.exists(os.path.join(web_dir, "package.json")):
        return

    npm_bin = shutil.which("npm") or shutil.which("pnpm")
    if not npm_bin:
        logger.warning("Node.js/npm not found in PATH; skipping web UI compilation.")
        return

    logger.info("Compiling Vue 3 / Vite Web UI assets in web/...")
    try:
        subprocess.check_call([npm_bin, "install", "-g", "pnpm"], timeout=60)
        subprocess.check_call([npm_bin, "install", "--no-audit", "--no-fund"], cwd=web_dir, timeout=120)
        pnpm_bin = shutil.which("pnpm") or "pnpm"
        subprocess.check_call([pnpm_bin, "run", "build"], cwd=web_dir, timeout=180)
        logger.info("Web UI compiled successfully into internal/webui/dist.")
    except Exception as exc:
        logger.error(f"Failed to compile Web UI assets: {exc}")

def get_current_commit_sha() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return ""

def resolve_gpt_load_binary() -> Optional[str]:
    """Find or automatically build gpt-load binary in container. Force rebuild v2.0.2."""
    local_exec = "/tmp/gpt-load-exec"

    # 1. First check if executable already exists locally
    if os.path.exists(local_exec) and os.path.isfile(local_exec) and os.path.getsize(local_exec) > 0:
        try:
            os.chmod(local_exec, 0o755)
        except Exception:
            pass
        return local_exec

    # 2. Check persistent storage bucket only if commit SHA matches
    sha_file = os.path.join(HF_STORAGE_DIR, ".commit_sha")
    current_sha = get_current_commit_sha()
    stored_sha = ""
    if os.path.exists(sha_file):
        try:
            with open(sha_file, "r") as f:
                stored_sha = f.read().strip()
        except Exception:
            pass

    bucket_bin = os.path.join(HF_STORAGE_DIR, "gpt-load")
    if (
        os.path.exists(bucket_bin)
        and os.path.isfile(bucket_bin)
        and os.path.getsize(bucket_bin) > 0
        and current_sha != ""
        and stored_sha == current_sha
    ):
        logger.info(
            f"Found precompiled gpt-load binary in persistent bucket matching commit {current_sha[:7]} "
            f"({os.path.getsize(bucket_bin) / (1024*1024):.1f} MB). Staging executable to {local_exec}..."
        )
        try:
            shutil.copyfile(bucket_bin, local_exec)
            os.chmod(local_exec, 0o755)
            return local_exec
        except Exception as e:
            logger.warning(f"Error copying gpt-load binary from persistent bucket: {e}")
    elif os.path.exists(bucket_bin):
        logger.info("Persistent bucket binary commit SHA mismatch or absent. Forcing fresh compilation...")

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
                        if current_sha:
                            with open(sha_file, "w") as f:
                                f.write(current_sha)
                        logger.info(f"Saved compiled binary (commit {current_sha[:7]}) to persistent bucket.")
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

    # Remove Fly.io 256MB memory constraints to leverage 16GB host RAM
    env.pop("GOMEMLIMIT", None)
    if env.get("GOGC") == "60":
        env["GOGC"] = "100"

    logger.info(f"Starting GPT-Load Go process ({bin_path}) on 127.0.0.1:{INTERNAL_PORT}...")
    GPT_LOAD_PROC = subprocess.Popen(
        [bin_path],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def pump_go_logs():
        log_files = []
        for path in {LOCAL_LOG_FILE, PERSISTENT_LOG_FILE}:
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                log_files.append(open(path, "a", encoding="utf-8"))
            except Exception:
                pass

        try:
            if GPT_LOAD_PROC and GPT_LOAD_PROC.stdout:
                for line in iter(GPT_LOAD_PROC.stdout.readline, ""):
                    if not line:
                        break
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    for f in log_files:
                        try:
                            f.write(line)
                            f.flush()
                        except Exception:
                            pass
        except Exception:
            pass
        finally:
            for f in log_files:
                try:
                    f.close()
                except Exception:
                    pass

    go_log_thread = threading.Thread(target=pump_go_logs, daemon=True)
    go_log_thread.start()
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

    logger.info("Performing final database and log synchronization to persistent bucket...")
    sync_db_to_persistent_storage()
    sync_logs_to_persistent_storage()

# ---------------------------------------------------------------------------
# 6. FastAPI Router & Ingress Reverse-Proxy
# ---------------------------------------------------------------------------
from fastapi import FastAPI, Header, HTTPException, Request, Response, WebSocket, WebSocketDisconnect  # type: ignore
from starlette.responses import PlainTextResponse, StreamingResponse  # type: ignore
import httpx  # type: ignore
import websockets  # type: ignore

app = FastAPI(title="GPT-Load Gateway", docs_url=None, redoc_url=None)
http_client: Optional[httpx.AsyncClient] = None

AUTH_KEY = os.getenv("AUTH_KEY")

def is_trusted_local_client(request: Request) -> bool:
    """
    Check if request originates directly from local container loopback (e.g. from GPT-Load).
    External requests routed via Hugging Face ingress always carry x-forwarded-for or x-forwarded-host.
    """
    client_host = request.client.host if request.client else ""
    xff = request.headers.get("x-forwarded-for")
    return client_host in ("127.0.0.1", "::1", "localhost") and not xff

def verify_admin_token(request: Request, authorization: Optional[str]):
    """Verify AUTH_KEY Bearer token for external administrative requests."""
    if is_trusted_local_client(request):
        return  # Trusted internal loopback

    if not AUTH_KEY:
        raise HTTPException(status_code=401, detail="Access denied: AUTH_KEY must be configured in Space secrets.")

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Bearer token")

    token = authorization.split("Bearer ", 1)[1].strip()
    if not hmac.compare_digest(token.encode("utf-8"), AUTH_KEY.encode("utf-8")):
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
    """Detailed operational status for GPT-Load Gateway."""
    uptime = int(time.time() - START_TIME)
    last_sync_iso = (
        datetime.datetime.fromtimestamp(LAST_DB_SYNC_TIME, datetime.timezone.utc).isoformat()
        if LAST_DB_SYNC_TIME > 0
        else "none"
    )
    gpt_load_alive = bool(GPT_LOAD_PROC and GPT_LOAD_PROC.poll() is None)
    return {
        "status": "online" if gpt_load_alive else "degraded",
        "service": "gpt-load-gateway",
        "zerogpu_enabled": HAS_ZEROGPU,
        "env_keys": [k for k in os.environ.keys() if "KEY" not in k and "TOKEN" not in k],
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
    verify_admin_token(request, authorization)
    success = sync_db_to_persistent_storage()
    return {"success": success, "status": LAST_DB_SYNC_STATUS}

# --- GitHub Synchronization & Webhook Endpoint ---
def detect_github_repo_url() -> Optional[str]:
    """Dynamically determine GitHub repository URL from environment or Space ID."""
    val = os.getenv("GITHUB_REPO_URL")
    if val:
        return val
    space_id = detect_space_id()
    if space_id:
        return f"https://github.com/{space_id}.git"
    return None

def sync_github_to_hf_space(target_url: Optional[str] = None) -> bool:
    """Fetch latest commit from GitHub and mirror clean tree directly to Hugging Face Space repository."""
    space_id = detect_space_id()
    if not HF_TOKEN or not space_id:
        logger.warning("GitHub auto-sync skipped: HF_TOKEN or SPACE_ID not configured.")
        return False

    repo_url = target_url or detect_github_repo_url()
    if not repo_url:
        logger.warning("GitHub auto-sync skipped: GITHUB_REPO_URL could not be determined.")
        return False

    sha_file = os.path.join(HF_STORAGE_DIR, ".last_synced_github_sha")
    last_synced_sha = ""
    if os.path.exists(sha_file):
        try:
            with open(sha_file, "r", encoding="utf-8") as f:
                last_synced_sha = f.read().strip()
        except Exception:
            pass

    # Step 1: Query remote commit SHA via lightweight git ls-remote (100ms, zero disk/clone overhead)
    latest_gh_sha = ""
    try:
        ls_out = subprocess.check_output(
            ["git", "ls-remote", repo_url, "refs/heads/main"], text=True, timeout=15
        ).strip()
        if ls_out:
            latest_gh_sha = ls_out.split()[0]
    except Exception as e:
        logger.warning(f"git ls-remote check failed for {repo_url}: {e}")

    # If already up-to-date, exit immediately without cloning
    if latest_gh_sha and last_synced_sha and latest_gh_sha == last_synced_sha:
        return False

    # Step 2: Only clone when a new commit is detected
    import tempfile
    tmp_sync_dir = tempfile.mkdtemp(prefix="gh_sync_")
    try:
        logger.info(
            f"New commit detected on GitHub: {latest_gh_sha[:8] if latest_gh_sha else 'unknown'} "
            f"(Previous: {last_synced_sha[:8] if last_synced_sha else 'none'}). Cloning update..."
        )
        subprocess.check_call(["git", "clone", "--depth=1", repo_url, tmp_sync_dir], timeout=60)

        if not latest_gh_sha:
            latest_gh_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=tmp_sync_dir, text=True
            ).strip()

        # Ensure prebuilt webui dist assets are present in sync repo before writing tree
        dist_dir_sync = os.path.join(tmp_sync_dir, "internal", "webui", "dist")
        local_dist_dir = os.path.abspath(os.path.join("internal", "webui", "dist"))
        if os.path.exists(local_dist_dir):
            os.makedirs(dist_dir_sync, exist_ok=True)
            shutil.copytree(local_dist_dir, dist_dir_sync, dirs_exist_ok=True)
            subprocess.run(["git", "add", "-f", "internal/webui/dist"], cwd=tmp_sync_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Strip repository screenshots to satisfy Hugging Face Hub limits
        subprocess.call(
            ["git", "rm", "-r", "--cached", "--ignore-unmatch", "screenshot"],
            cwd=tmp_sync_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        git_env = os.environ.copy()
        git_env["GIT_AUTHOR_NAME"] = "GitHub Sync Bot"
        git_env["GIT_AUTHOR_EMAIL"] = "sync@gpt-load.space"
        git_env["GIT_COMMITTER_NAME"] = "GitHub Sync Bot"
        git_env["GIT_COMMITTER_EMAIL"] = "sync@gpt-load.space"

        commit_msg = subprocess.check_output(
            ["git", "log", "-1", "--format=%s"], cwd=tmp_sync_dir, text=True
        ).strip()
        tree_id = subprocess.check_output(
            ["git", "write-tree"], cwd=tmp_sync_dir, text=True
        ).strip()
        commit_id = subprocess.check_output(
            ["git", "commit-tree", tree_id, "-m", f"Sync from GitHub: {commit_msg}\n\nGitHub-SHA: {latest_gh_sha}"],
            cwd=tmp_sync_dir,
            env=git_env,
            text=True,
        ).strip()

        auth_url = f"https://oauth:{HF_TOKEN}@huggingface.co/spaces/{space_id}.git"
        logger.info(f"Pushing synchronized commit {commit_id[:8]} to Hugging Face Space {space_id}...")
        subprocess.check_call(
            ["git", "push", auth_url, f"{commit_id}:refs/heads/main", "--force", "--no-verify"],
            cwd=tmp_sync_dir,
            timeout=60,
        )

        # Update persistent SHA record to prevent repeat pushes
        try:
            with open(sha_file, "w", encoding="utf-8") as f:
                f.write(latest_gh_sha)
        except Exception as e:
            logger.warning(f"Failed to record last synced SHA: {e}")

        logger.info(f"Successfully synchronized GitHub commit {latest_gh_sha[:8]} to Hugging Face Space.")
        return True
    except Exception as exc:
        logger.error(f"Error syncing GitHub commit to Hugging Face Space: {exc}")
        return False
    finally:
        if os.path.exists(tmp_sync_dir):
            try:
                shutil.rmtree(tmp_sync_dir, ignore_errors=True)
            except Exception:
                pass

def background_github_sync_poller(interval_seconds: int = 600):
    """Background daemon checking GitHub for updates every 10 minutes."""
    while not SHUTDOWN_EVENT.is_set():
        SHUTDOWN_EVENT.wait(interval_seconds)
        if SHUTDOWN_EVENT.is_set():
            break
        sync_github_to_hf_space()

@app.post("/github-sync")
async def github_sync_webhook(request: Request, x_github_event: Optional[str] = Header(None)):
    """
    Receive GitHub push webhooks at https://<space>.hf.space/github-sync.
    Extracts clone URL from payload and mirrors the clean tree to Hugging Face Space.
    """
    if x_github_event and x_github_event != "push":
        return {"status": "ignored", "event": x_github_event}

    target_clone_url = None
    try:
        body = await request.json()
        ref = body.get("ref", "")
        if ref and ref != "refs/heads/main":
            return {"status": "ignored", "ref": ref}
        repo_data = body.get("repository", {})
        target_clone_url = repo_data.get("clone_url") or repo_data.get("html_url")
    except Exception:
        pass

    logger.info("GitHub push webhook received on main branch. Triggering immediate repository sync...")
    synced = await asyncio.to_thread(sync_github_to_hf_space, target_clone_url)
    return {
        "status": "synchronized" if synced else "already_up_to_date_or_failed",
        "synced": synced,
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

with gr.Blocks(title="GPT-Load Gateway") as demo:
    gr.Markdown(f"""
    # ⚡ GPT-Load Gateway

    <div style="display: flex; gap: 10px; margin-bottom: 20px;">
        <span style="background-color: #22c55e20; color: #16a34a; padding: 4px 12px; border-radius: 9999px; font-weight: bold; border: 1px solid #16a34a40;">
            ● Status: Online ({'ZeroGPU' if HAS_ZEROGPU else 'CPU'})
        </span>
    </div>

    This Hugging Face Space hosts the **GPT-Load Multi-Channel AI Gateway** with persistent SQLite storage bucket sync.

    ### ⚡ Active Endpoints
    - **GPT-Load Web Console**: `GET /` *(Native Vue 3 / Vite Management Dashboard)*
    - **Keep-Alive Probe**: `GET /healthz` *(Zero GPU quota consumed, for external cron)*
    - **System Diagnostics**: `GET /health` & `GET /status`
    """)

# Mount Gradio Blocks at /gradio so root / remains dedicated to GPT-Load Web UI
app = gr.mount_gradio_app(app, demo, path="/gradio")

# --- HTTP Data-Plane, Control-Plane, & WebUI Reverse-Proxy ---
STATIC_ASSET_EXTENSIONS = (
    ".js", ".css", ".svg", ".png", ".jpg", ".jpeg",
    ".ico", ".woff", ".woff2", ".ttf", ".map", ".webp", ".json", ".html"
)

def is_internal_asset_path(path: str) -> bool:
    """Return True if path represents static web UI assets or health probes that should not be logged."""
    p = path.strip("/").lower()
    if not p:
        return True  # Root UI HTML
    if p.startswith("assets/") or p.startswith("static/") or p.startswith("favicon") or p.startswith("gradio"):
        return True
    if any(p.endswith(ext) for ext in STATIC_ASSET_EXTENSIONS) and not p.startswith("v1/") and not p.startswith("api/"):
        return True
    if p in ("health", "healthz", "robots.txt", "theme-bootstrap.js"):
        return True
    return False

ACTIVE_CONNECTIONS: dict[str, dict[str, Any]] = {}
ACTIVE_CONN_LOCK = threading.Lock()
_CONN_COUNTER = 0

def register_active_conn(path: str, method: str, client_ip: str) -> Optional[str]:
    if is_internal_asset_path(path):
        return None
    global _CONN_COUNTER
    with ACTIVE_CONN_LOCK:
        _CONN_COUNTER += 1
        conn_id = f"conn_{_CONN_COUNTER}"
        ACTIVE_CONNECTIONS[conn_id] = {
            "path": path,
            "method": method,
            "client_ip": client_ip,
            "start_time": time.time(),
            "status": "waiting_upstream",
        }
        active_count = len(ACTIVE_CONNECTIONS)
    logger.info(f"[INGRESS] Connection opened: {conn_id} {method} /{path} from {client_ip} (active: {active_count})")
    return conn_id

def update_active_conn(conn_id: Optional[str], status: str):
    if not conn_id:
        return
    with ACTIVE_CONN_LOCK:
        if conn_id in ACTIVE_CONNECTIONS:
            ACTIVE_CONNECTIONS[conn_id]["status"] = status

def unregister_active_conn(conn_id: Optional[str], status_desc: str):
    if not conn_id:
        return
    with ACTIVE_CONN_LOCK:
        info = ACTIVE_CONNECTIONS.pop(conn_id, None)
        active_count = len(ACTIVE_CONNECTIONS)
    if info:
        elapsed = time.time() - info["start_time"]
        logger.info(f"[INGRESS] Connection closed: {conn_id} {info['method']} /{info['path']} ({status_desc}) after {elapsed:.2f}s (active remaining: {active_count})")

def background_connection_watchdog():
    """Periodically report active connections in-flight through the ingress proxy."""
    while not SHUTDOWN_EVENT.is_set():
        time.sleep(20)
        with ACTIVE_CONN_LOCK:
            if not ACTIVE_CONNECTIONS:
                continue
            now = time.time()
            summary = [
                f"{cid}: {c['method']} /{c['path']} [status={c['status']}, age={int(now - c['start_time'])}s]"
                for cid, c in list(ACTIVE_CONNECTIONS.items())
            ]
            count = len(summary)
        logger.info(f"[MONITOR] {count} active:\n  " + "\n  ".join(summary))

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

@app.get("/")
@app.get("")
async def root_proxy_handler(request: Request):
    return await reverse_proxy_handler(request, "")

@app.get("/api/system/logs", include_in_schema=False)
async def get_system_logs(lines: int = 500):
    """Return tail lines of the persistent system log file in attached storage."""
    if not os.path.exists(PERSISTENT_LOG_FILE):
        return Response("Log file not found", status_code=404)
    try:
        with open(PERSISTENT_LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            content = f.readlines()
            tail = content[-lines:] if len(content) > lines else content
            return Response("".join(tail), media_type="text/plain")
    except Exception as exc:
        return Response(f"Error reading log file: {exc}", status_code=500)

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"], include_in_schema=False)
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

    conn_id = register_active_conn(path, request.method, client_ip)
    req_cancelled = asyncio.Event()
    upstream_resp = None
    disconnect_task = None

    async def watch_disconnect():
        try:
            while not req_cancelled.is_set():
                if await request.is_disconnected():
                    if conn_id:
                        logger.info(f"[INGRESS] Client disconnected for /{path} ({conn_id}); cancelling upstream request.")
                    req_cancelled.set()
                    break
                await asyncio.sleep(0.5)
        except Exception:
            pass

    try:
        req_body = None
        if request.method in ("POST", "PUT", "PATCH"):
            try:
                read_timeout = float(os.getenv("READ_TIMEOUT", 60))
                req_body = await asyncio.wait_for(request.body(), timeout=read_timeout)
                req_headers["content-length"] = str(len(req_body))
            except asyncio.TimeoutError:
                if conn_id:
                    logger.warning(f"[INGRESS] Client body read timeout for /{path} ({conn_id})")
                unregister_active_conn(conn_id, "body_read_timeout")
                return Response("Request body timeout", status_code=408)

        if await request.is_disconnected():
            if conn_id:
                logger.info(f"[INGRESS] Client disconnected before sending /{path} ({conn_id})")
            req_cancelled.set()
            unregister_active_conn(conn_id, "pre_send_disconnect")
            return Response("Client disconnected", status_code=499)

        disconnect_task = asyncio.create_task(watch_disconnect())

        upstream_req = http_client.build_request(
            method=request.method,
            url=target_url,
            headers=req_headers,
            content=req_body,
        )

        send_task = asyncio.create_task(http_client.send(upstream_req, stream=True))
        disconnect_waiter = asyncio.create_task(req_cancelled.wait())

        done, pending = await asyncio.wait(
            [send_task, disconnect_waiter],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for p in pending:
            p.cancel()

        if send_task in done:
            upstream_resp = send_task.result()
            update_active_conn(conn_id, f"headers_received_{upstream_resp.status_code}")
        else:
            if conn_id:
                logger.info(f"[INGRESS] Client disconnected while waiting for upstream response headers for /{path} ({conn_id})")
            unregister_active_conn(conn_id, "header_wait_disconnect")
            return Response("Client disconnected", status_code=499)

        if req_cancelled.is_set() or await request.is_disconnected():
            if conn_id:
                logger.info(f"[INGRESS] Client disconnected while upstream request was being sent for /{path} ({conn_id})")
            req_cancelled.set()
            await upstream_resp.aclose()
            unregister_active_conn(conn_id, "post_send_disconnect")
            return Response("Client disconnected", status_code=499)

        resp_headers = dict(upstream_resp.headers)
        for hop in HOP_BY_HOP_HEADERS:
            resp_headers.pop(hop, None)
        resp_headers.pop("content-length", None)
        resp_headers.pop("content-encoding", None)

        # Allow Hugging Face Hub to embed the Space UI inside the HF iframe
        resp_headers.pop("x-frame-options", None)
        resp_headers["content-security-policy"] = (
            "frame-ancestors 'self' https://huggingface.co https://*.huggingface.co https://*.hf.space;"
        )

        # Optimize for Server-Sent Events / LLM token streaming
        is_sse = "text/event-stream" in resp_headers.get("content-type", "")
        if is_sse:
            resp_headers["cache-control"] = "no-cache"
            resp_headers["x-accel-buffering"] = "no"

        update_active_conn(conn_id, "streaming" if is_sse else "transferring")

        async def stream_body():
            try:
                async for chunk in upstream_resp.aiter_raw():
                    if req_cancelled.is_set() or await request.is_disconnected():
                        if conn_id:
                            logger.info(f"[INGRESS] Client disconnected during streaming for /{path} ({conn_id}); closing upstream connection.")
                        req_cancelled.set()
                        break
                    yield chunk
            except Exception as e:
                if conn_id:
                    logger.debug(f"[INGRESS] Stream proxy closed for /{path} ({conn_id}): {e}")
            finally:
                req_cancelled.set()
                await upstream_resp.aclose()
                status_note = "client_cancelled" if req_cancelled.is_set() else f"completed_{upstream_resp.status_code}"
                unregister_active_conn(conn_id, status_note)

        return StreamingResponse(
            stream_body(),
            status_code=upstream_resp.status_code,
            headers=resp_headers,
        )
    except httpx.ConnectError:
        req_cancelled.set()
        unregister_active_conn(conn_id, "connect_error")
        return Response("Internal GPT-Load engine unavailable or starting up.", status_code=503)
    except Exception as exc:
        req_cancelled.set()
        if upstream_resp is not None:
            await upstream_resp.aclose()
        if conn_id:
            logger.error(f"[INGRESS] Ingress proxy error for /{path} ({conn_id}): {exc}")
        unregister_active_conn(conn_id, f"error: {exc}")
        return Response(f"Proxy error: {exc}", status_code=502)
    finally:
        if disconnect_task is not None and not disconnect_task.done():
            disconnect_task.cancel()

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
    logger.info("Initializing GPT-Load Gateway Space")
    logger.info(f"Public Port: {PUBLIC_PORT} • Internal Port: {INTERNAL_PORT}")
    logger.info(f"Persistent Storage: {HF_STORAGE_DIR}")
    logger.info(f"Working Data Dir: {LOCAL_DATA_DIR}")
    logger.info(f"ZeroGPU Available: {HAS_ZEROGPU}")
    logger.info("==================================================")

    # 1. Hydrate working database from persistent bucket
    hydrate_database_from_bucket()

    # 2. Launch gpt-load Go binary
    start_gpt_load_process()
    wait_for_gpt_load_ready(timeout_seconds=30)

    # 4. Start background SQLite synchronization daemon
    sync_thread = threading.Thread(target=background_db_sync_poller, daemon=True)
    sync_thread.start()

    # 5. Start background GitHub repository sync daemon
    github_thread = threading.Thread(target=background_github_sync_poller, daemon=True)
    github_thread.start()

    # 6. Start background connection watchdog for ingress proxy
    watchdog_thread = threading.Thread(target=background_connection_watchdog, daemon=True)
    watchdog_thread.start()

    # 7. Launch FastAPI Ingress on public port (7860)
    import uvicorn  # type: ignore

    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=PUBLIC_PORT,
            log_level="info",
            access_log=False,
        )
    finally:
        stop_gpt_load_process()
