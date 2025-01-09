CC = gcc
CFLAGS = -Wall -IC:/path/to/vcpkg/installed/x64-windows/include
LDFLAGS = -LC:/path/to/vcpkg/installed/x64-windows/lib -lgsl -lgslcblas -lm

all: gnb_predictor

gnb_predictor: gnb_predictor.o
	$(CC) -o gnb_predictor gnb_predictor.o $(LDFLAGS)

gnb_predictor.o: gnb_predictor.c
	$(CC) $(CFLAGS) -c gnb_predictor.c

clean:
	rm -f gnb_predictor gnb_predictor.o
