FROM continuumio/miniconda3

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /home

COPY requirement.txt .

RUN pip install -r requirement.txt

CMD ["/bin/bash"]