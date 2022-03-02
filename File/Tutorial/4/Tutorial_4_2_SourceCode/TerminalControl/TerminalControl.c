#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>


int main( int argc, char *argv[] ){

	int isStop = 0;
	
	while(!isStop)
	{	
		printf("Printing withing loop!\n");

		printf("\033[H\033[2J");
		
	}
	
}
