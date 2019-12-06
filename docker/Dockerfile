FROM pytorch/pytorch:latest

RUN pip install --upgrade pip

COPY requirements.txt requirements.txt 
RUN pip install --upgrade --no-cache-dir -r requirements.txt && \     
	rm requirements.txt


WORKDIR /home
