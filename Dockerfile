FROM python:3.9-slim
# install the notebook package
RUN pip install --no-cache --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install torch_geometric && \
    pip install pyg_lib torch_sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
    
# create user with a home directory
ARG NB_USER
ARG NB_UID
ENV USER ${NB_USER}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}
WORKDIR ${HOME}
USER ${USER}
