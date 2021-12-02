run:
	docker build -t pa2/cs643pa2 .
	docker run --name nicklaboy \
	-v /Users/NickLaBoy/cs643pa2/:/home/jovyan/work \
	-p 8888:8888 \
	pa2/cs643pa2
