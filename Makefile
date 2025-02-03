CC = gcc
CFLAGS = -Wall
LDFLAGS = -lm

all: gnb_predictor

gnb_predictor: gnb_predictor.o
	$(CC) -o gnb_predictor gnb_predictor.o $(LDFLAGS)

gnb_predictor.o: gnb_predictor.c
	$(CC) $(CFLAGS) -c gnb_predictor.c

clean:
	rm -f gnb_predictor gnb_predictor.o
