ARG REGISTRY=quay.io
ARG OWNER=jupyter
ARG BASE_CONTAINER=$REGISTRY/$OWNER/datascience-notebook
FROM $BASE_CONTAINER

USER root

RUN apt-get -y update
RUN apt search gnuplot
RUN apt-get install -y --no-install-recommends octave-dev gnuplot fonts-freefont-otf
RUN pip install nglview

#cmake
RUN apt install -y cmake

#libopenmpi-dev
RUN apt-get -y install libopenmpi-dev

#cuda
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
RUN mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda-repo-ubuntu2204-12-9-local_12.9                                                                                .0-575.51.03-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu2204-12-9-local_12.9.0-575.51.03-1_amd64.deb
RUN cp /var/cuda-repo-ubuntu2204-12-9-local/cuda-*-keyring.gpg /usr/share/keyrings/
RUN apt-get update  -y
RUN apt-get -y install cuda-toolkit-12-9

#sudo
RUN echo "jovyan ALL=(ALL) NOPASSWD:ALL" >>/etc/sudoers
#git-lfs
RUN apt-get install git-lfs

RUN git clone https://github.com/rbi-mtm/ml4mmr
RUN /bin/bash -c "cd ml4mmr/install && source /opt/conda/bin/activate base && source install_environments.sh -p /usr/loc                                                                                al && conda clean --all"
RUN apt autoremove -y
RUN apt  -y clean
RUN conda clean --all
