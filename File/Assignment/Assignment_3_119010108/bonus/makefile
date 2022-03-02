Project: main.o user_program.o virtual_memory.o
	nvcc -o Project main.o user_program.o virtual_memory.o
virtual_memory.o: virtual_memory.cu virtual_memory.h
	nvcc -c virtual_memory.cu -rdc=true
user_program.o: virtual_memory.h user_program.cu
	nvcc -c user_program.cu -rdc=true
main.o: main.cu virtual_memory.h
	nvcc -c main.cu -rdc=true
test:
	./Project
clean:
	rm -f Project main.o user_program.o virtual_memory.o