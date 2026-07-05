# mri_transfer — ArtiSynth + JPype + V2 Python pipeline (all-in-one)
#
#   docker compose build
#   docker compose up -d workspace
#   docker compose exec workspace bash
#
FROM eclipse-temurin:21-jdk-jammy

ARG ARTISYNTH_GIT=https://github.com/artisynth/artisynth_core.git
ARG ARTISYNTH_MODELS_GIT=https://github.com/artisynth/artisynth_models.git
ARG ARTISYNTH_HOME=/opt/artisynth/artisynth_core
ARG ARTISYNTH_MODELS=/opt/artisynth/artisynth_models

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python-is-python3 git make libgomp1 ca-certificates curl \
        xvfb libxrender1 libgl1 libglib2.0-0 libsm6 libxext6 libosmesa6 \
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        -o /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends gh \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt \
    && hf --help >/dev/null 2>&1 || huggingface-cli --help >/dev/null

RUN git clone --depth 1 "${ARTISYNTH_GIT}" "${ARTISYNTH_HOME}" && \
    git clone --depth 1 "${ARTISYNTH_MODELS_GIT}" "${ARTISYNTH_MODELS}" && \
    cp -r "${ARTISYNTH_MODELS}/src/." "${ARTISYNTH_HOME}/src/"

RUN cd "${ARTISYNTH_HOME}" && \
    java -cp "lib/vfs2.jar:bin/libraryInstaller.jar" \
        artisynth.core.driver.LibraryInstaller -updateLibs \
        -remoteSource https://www.artisynth.org/files/lib/ && \
    make

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh \
    && chmod +x /usr/local/bin/docker-entrypoint.sh

ENV PYTHONPATH=/workspace \
    MPLBACKEND=Agg \
    ARTISYNTH_HOME=${ARTISYNTH_HOME} \
    TONGUE_MODEL=artisynth.models.tongue3d.HexTongueDemo \
    JVM_XMX=4g \
    TONGUE_OBJ=${ARTISYNTH_HOME}/src/artisynth/models/tongue3d/geometry/tongue.obj

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["bash"]
