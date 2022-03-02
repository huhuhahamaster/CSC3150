Project: main.o user_program.o file_system.o
	nvcc -o Project main.o user_program.o file_system.o
file_system.o: file_system.cu file_system.h
	nvcc -c file_system.cu -rdc=true
user_program.o: file_system.h user_program.cu
	nvcc -c user_program.cu -rdc=true
main.o: main.cu file_system.h
	nvcc -c main.cu -rdc=true
test:
	./Project
clean:
	rm -f Project main.o user_program.o file_system.o