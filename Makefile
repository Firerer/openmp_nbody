# MAKEFLAGS += -j7

CFLAGS = -Wall -fopenmp
SANFLAGS = -fsanitize=object-size  -fsanitize=undefined #-fsanitize=address
SANFLAGS = -O3
LIBS = -lmpi_cxx -lmpi -lm
CC= g++
FILE = solution.cc

export OMP_NUM_THREADS=4

solution: $(FILE)
	$(CC) $(CFLAGS) $(SANFLAGS) $(LIBS) $(FILE) -o solution

run: solution
	mpirun -np 4 ./solution input.txt

plot: solution plot.py
	mpirun -np 4 ./solution input.txt && python3 plot.py

debug: solution FORCE
	mpirun -np 4 ./solution inputbig.txt

pdf: Report.tex
	tectonic Report.tex

clean:
	rm -f solution *.ans *.diff

FORCE:


syncto:
	@rsync -avz ./ dlliu3@spartan.hpc.unimelb.edu.au:~/src

syncfrom:
	@rsync -avz dlliu3@spartan.hpc.unimelb.edu.au:~/src ./
