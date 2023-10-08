# MAKEFLAGS += -j7

CFLAGS = -Wall -fopenmp
LIBS = -lmpi_cxx -lmpi -lm
CC= g++
FILE = solution.cc
export OMP_NUM_THREADS=2

solution: $(FILE)
	$(CC) $(CFLAGS) $(LIBS) $(FILE) -o solution

run: solution FORCE
	mpirun -np 4 ./solution input.txt

pdf: Report.tex
	tectonic Report.tex

clean:
	rm -f solution *.ans *.diff

FORCE:
