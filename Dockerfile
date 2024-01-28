FROM nvcr.io/nvidia/pytorch:21.03-py3
RUN pip3 install wrapt --upgrade --ignore-installed

RUN apt-get update

RUN pip install torch==1.8.1
RUN pip install torchaudio==0.8.1
RUN pip install pip --upgrade

ENV PYTHONPATH /workspace/SVSN
WORKDIR /workspace/SVSN

RUN pip3 install wandb --upgrade
RUN wandb login d687755a16832e24059e221a63331c879834bd13

