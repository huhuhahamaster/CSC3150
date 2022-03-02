#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <curses.h>
#include <termios.h>
#include <fcntl.h>
//#include <conio.h>



int kbhit(void){
	struct termios oldt, newt;
	int ch;
	int oldf;

	tcgetattr(STDIN_FILENO, &oldt);

	newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);

	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	oldf = fcntl(STDIN_FILENO, F_GETFL, 0);

	fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

	ch = getchar();

	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
	fcntl(STDIN_FILENO, F_SETFL, oldf);

	if(ch != EOF)
	{
		ungetc(ch, stdin);
		return 1;
	}
	return 0;
}


int main( int argc, char *argv[] ){
	
	int isQuit = 0; 

	while (!isQuit){

		if( kbhit() ){

			char dir = getchar() ; 
		
			//printf("\033[H\033[2J");

			if( dir == 'w' || dir == 'W' )
				printf ("UP Hit!\n");
	 
			if( dir == 'a' || dir == 'A' )
				printf ("LEFT Hit!\n");

			if( dir == 'd' || dir == 'D' )	
				printf ("RIGHT Hit!\n");				

			if( dir == 's' || dir == 'S' )
				printf ("DOWN Hit!\n");

			if( dir == 'q' || dir == 'Q' ){	
				printf("Quit!\n");
				isQuit= 1;
			}
		}
	}

	return 0;
	
}
