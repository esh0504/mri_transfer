#!/bin/bash
# 컨테이너 시작 시 토큰 로그인 + ArtiSynth 준비 상태 확인.
set -e

_setup_tokens() {
    local tokens_file="${TOKENS_FILE:-/workspace/tokens.env}"
    if [ ! -f "$tokens_file" ]; then
        return 0
    fi

    echo "[setup] Loading tokens from $tokens_file"
    set -a
    # shellcheck disable=SC1090
    source "$tokens_file"
    set +a

    local hf_token="${HUGGINGFACE_TOKEN:-${HF_TOKEN:-}}"
    if [ -n "$hf_token" ]; then
        export HF_TOKEN="$hf_token"
        export HUGGING_FACE_HUB_TOKEN="$hf_token"
        mkdir -p "${HOME}/.cache/huggingface"
        printf '%s' "$hf_token" > "${HOME}/.cache/huggingface/token"
        chmod 600 "${HOME}/.cache/huggingface/token"
        if command -v hf >/dev/null 2>&1; then
            hf auth login --token "$hf_token" >/dev/null 2>&1 \
                && echo "[setup] Hugging Face: logged in (hf CLI)" \
                || echo "[setup] Hugging Face: token set (HF_TOKEN)"
        elif python3 -c "import huggingface_hub" 2>/dev/null; then
            HF_TOKEN="$hf_token" python3 -c \
                "import os; from huggingface_hub import login; login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)" \
                >/dev/null 2>&1 \
                && echo "[setup] Hugging Face: logged in (huggingface_hub)" \
                || echo "[setup] Hugging Face: token set (HF_TOKEN)"
        else
            echo "[setup] Hugging Face: token set (HF_TOKEN)"
        fi
    fi

    if [ -n "${GIT_TOKEN:-}" ]; then
        if command -v gh >/dev/null 2>&1; then
            printf '%s' "$GIT_TOKEN" | gh auth login --with-token >/dev/null 2>&1 \
                && echo "[setup] GitHub: logged in (gh CLI)" \
                || echo "[setup] GitHub: gh auth failed (falling back to git credentials)"
        fi
        git config --global credential.helper store
        printf 'https://oauth2:%s@github.com\n' "$GIT_TOKEN" > "${HOME}/.git-credentials"
        chmod 600 "${HOME}/.git-credentials"
        git config --global url."https://oauth2:${GIT_TOKEN}@github.com/".insteadOf "https://github.com/"
        echo "[setup] Git: credentials configured (github.com)"
    fi
}

_setup_tokens

AH="${ARTISYNTH_HOME:-/opt/artisynth/artisynth_core}"
NATIVE="$AH/lib/Linux64"

if [ ! -d "$AH/classes" ]; then
  echo "[setup] ERROR: ArtiSynth classes missing at $AH"
  exit 1
fi

if [ ! -d "$NATIVE" ] || [ -z "$(ls -A "$NATIVE" 2>/dev/null)" ]; then
  echo "[setup] Linux native libs missing -> fetching (one-time fallback)..."
  if ( cd "$AH" && java -cp "lib/vfs2.jar:bin/libraryInstaller.jar" \
         artisynth.core.driver.LibraryInstaller -updateLibs \
         -remoteSource https://www.artisynth.org/files/lib/ ); then
    echo "[setup] native libs ready in $NATIVE"
  else
    echo "[setup] ERROR: native lib fetch failed (need network)"
    exit 1
  fi
else
  echo "[setup] ArtiSynth ready: $AH ($(ls "$NATIVE" | wc -l) native libs)"
fi

exec "$@"
