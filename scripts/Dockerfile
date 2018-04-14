FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
ENV CONDA_DIR="/opt/conda"
ENV PATH="$CONDA_DIR/bin:$PATH"
RUN apt-get update && apt-get install -y wget

ARG GIT_USER
ARG GIT_TOKEN

# Install conda
RUN CONDA_VERSION="4.4.10" \
 && CONDA_MD5_CHECKSUM="bec6203dbb2f53011e974e9bf4d46e93" \
 && mkdir -p "$CONDA_DIR" \
 && wget "https://repo.continuum.io/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh" -O miniconda.sh \
 && echo "$CONDA_MD5_CHECKSUM  miniconda.sh" | md5sum -c \
 && bash miniconda.sh -f -b -p "$CONDA_DIR" \
 && echo "export PATH=$CONDA_DIR/bin:\$PATH" > /etc/profile.d/conda.sh \
 && rm miniconda.sh \
 && conda update conda --yes \
 && conda update --all --yes \
 && conda clean --all --yes \
 && conda config --add channels conda-forge

RUN conda install -y pytorch torchvision cuda91 -c pytorch
RUN conda install -y scikit-learn \
                     pandas \
                     numpy \
 && conda clean -y -a

RUN apt-get update && apt-get install -y cifs-utils \
    build-essential libsnappy-dev git nano cmake doxygen graphviz zip htop && \
    yes | pip install python-snappy 'holoviews[recommended]' && \
    yes | pip install git+https://github.com/dask/dask-adlfs

RUN apt-get update && apt-get install -y cifs-utils g++ watch htop \
    vim tree git nano cmake rsync wget zip \
    yes | pip install tensorboardX && \
    yes | pip install -U setuptools && \
    yes | pip install scikit-learn && \
    yes | pip install scikit-optimize && \
    yes | pip install jupyter && \
    yes | pip install cupy && \
    yes | pip install pynvrtc && \
    yes | pip install tqdm && \
    yes | pip install statsmodels

RUN pip install jupyterlab

RUN pip install --upgrade pip

RUN mkdir update

RUN git clone https://${GIT_USER}:${GIT_TOKEN}@github.com/Choteck/DeepLearningTimeSeries && \
    cd /DeepLearningTimeSeries && \
    pip install -e mypackage

RUN mkdir datadrive && \
    cd /datadrive && \
    mkdir wind_power

RUN mkdir workspace
RUN mv /DeepLearningTimeSeries/scripts/* /workspace

WORKDIR /workspace