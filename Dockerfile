FROM python:3.9-slim
USER root

# install the notebook package
RUN pip install --no-cache --upgrade pip && \
    pip install --no-cache  torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache  torch_geometric && \
    pip install --no-cache  pyg_lib torch_sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
