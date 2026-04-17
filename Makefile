all: trip

CFLAGS = -m64 -Ofast -fopenmp -march=native -DCACHE_LINESIZE=$(shell getconf LEVEL1_DCACHE_LINESIZE)
LDFLAGS = -ljpeg -lm -lX11 -fopenmp

SRCS = main.c math.c forward.c backward.c model.c utils.c
OBJS = $(SRCS:.c=.o)

trip: $(OBJS)
	gcc $(OBJS) $(LDFLAGS) -o trip

%.o: %.c trip.h
	gcc $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) trip

.PHONY: all clean
