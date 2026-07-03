# mri_transfer — ArtiSynth + JPype + V2 Python pipeline (all-in-one)
#
#   docker compose build
#   docker compose run --rm shell
#   docker compose run --rm pipeline          # stage=all (needs /data mount)
#   docker compose run --rm pipeline stage=fem # fem only (no MRI masks)
#
FROM eclipse-temurin:21-jdk-jammy

ARG ARTISYNTH_GIT=https://github.com/artisynth/artisynth_core.git
ARG ARTISYNTH_MODELS_GIT=https://github.com/artisynth/artisynth_models.git
ARG ARTISYNTH_HOME=/opt/artisynth/artisynth_core
ARG ARTISYNTH_MODELS=/opt/artisynth/artisynth_models

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python-is-python3 git make libgomp1 ca-certificates \
        xvfb libxrender1 libgl1 libglib2.0-0 libsm6 libxext6 libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

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

ENV PYTHONPATH=/app \
    MPLBACKEND=Agg \
    ARTISYNTH_HOME=${ARTISYNTH_HOME} \
    TONGUE_MODEL=artisynth.models.tongue3d.HexTongueDemo \
    JVM_XMX=4g \
    TONGUE_OBJ=${ARTISYNTH_HOME}/src/artisynth/models/tongue3d/geometry/tongue.obj

WORKDIR /app
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["bash"]
