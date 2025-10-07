MPICC=mpicc
CC=gcc
CFLAGS=-O3 -Wall -Wextra -std=c11

all: mpi_primes mpi_cyclic mpi_block mpi_dynamic serial_primes pthreads_block pthreads_primes_cyclic

mpi_primes: mpi_primes.c
	$(MPICC) $(CFLAGS) -o $@ mpi_primes.c -lm

mpi_cyclic: mpi_cyclic.c
	$(MPICC) $(CFLAGS) -o $@ mpi_cyclic.c -lm

mpi_block: mpi_block.c
	$(MPICC) $(CFLAGS) -o $@ mpi_block.c -lm

mpi_dynamic: mpi_dynamic.c
	$(MPICC) $(CFLAGS) -o $@ mpi_dynamic.c -lm

serial_primes: serial_primes.c
	$(CC) $(CFLAGS) -o $@ serial_primes.c -lm

pthreads_block: pthreads_block.c
	$(CC) $(CFLAGS) -o $@ pthreads_block.c -lpthread -lm

pthreads_primes_cyclic: pthreads_primes_cyclic.c
	$(CC) $(CFLAGS) -o $@ pthreads_primes_cyclic.c -lpthread -lm

clean:
	rm -f mpi_primes mpi_cyclic mpi_block mpi_dynamic serial_primes pthreads_block pthreads_primes_cyclic primes.txt cyclic_primes.txt block_primes.txt dynamic_primes.txt serial_primes.txt pthreads_primes.txt pthreads_primes_cyclic.txt pthreads_block.txt

.PHONY: all clean


