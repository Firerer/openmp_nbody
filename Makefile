# MAKEFLAGS += -j7

CFLAGS = -Wall -fopenmp
SANFLAGS = -fsanitize=object-size  -fsanitize=undefined # -fsanitize=address
LIBS = -lmpi_cxx -lmpi -lm
CC= g++
FILE = solution.cc

solution: $(FILE) FORCE
	$(CC) $(CFLAGS) $(SANFLAGS) $(LIBS) $(FILE) -o solution

run: solution FORCE
	mpirun -np 4 ./solution input.txt

debug: solution FORCE
	mpirun -np 4 ./solution inputbig.txt

pdf: Report.tex
	tectonic Report.tex

clean:
	rm -f solution *.ans *.diff

FORCE:
