#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <curses.h>
#include <termios.h>
#include <fcntl.h>
#include <random>


#define ROW 10
#define COLUMN 50 

void initialize_the_log(); 
 
pthread_mutex_t frog_mutex;
pthread_cond_t frog_threshold_cv;
int thread_ids[10] = {0,1,2,3,4,5,6,7,8,9};
int thread_count = 0;

int ISOVER = 0;
int ISQUIT = 0;


struct Node{
	int x , y; 
	Node( int _x , int _y ) : x( _x ) , y( _y ) {}; 
	Node(){} ; 
} frog ; 


char map[ROW+10][COLUMN] ; 

/* initialize the log before the game start */
void initialize_the_log(){
	int min = 0;
	int max = COLUMN-1;
	for (int i = 1; i < ROW; i++){
		int len = rand() % (max-min) + 0;
		int range =  rand() % (15) + 8;

		/* generate the random length of each log */
		for (int j = len; j < len + range; j++){
            map[i][j % (COLUMN - 1)] = '=';
        }
	}
}

/* check the game status after frog move */
int check_status(int x, int y){
    if (y >= COLUMN - 1 || y < 0){
		return -1; 					// -1 represent lose the game
	}
    else if(x == 0){
		return 1; 					// 1 represent win the game
	}
    else if(x == ROW){
		return 0; 					// 0 represent game still continue 
	}
    else{
        if (map[x][y] == '='){
            return 0;
        }
        else return -1;
    }
}

/* Determine a keyboard is hit or not. If yes, return 1. If not, return 0 */
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

/* move the frog after 10 thread logs move and check the game state */
void *frog_move( void * t){

	while (!ISOVER){
		pthread_mutex_lock(&frog_mutex);

		/* wait for every log move */
		while (thread_count != 9){
			pthread_cond_wait(&frog_threshold_cv,&frog_mutex);
		}

		thread_count = 0;

		/* move the frog */
		if (kbhit()){
			char move = getchar();

			switch(move){
				case 'w':
				case 'W':{
					if ( check_status(frog.x-1,frog.y) == 1 || check_status(frog.x-1,frog.y) == -1){
						frog.x--;
						ISOVER =1;
					}
					else{
						if (frog.x == ROW){
							map[frog.x][frog.y] = '|';
						}
						else{
							map[frog.x][frog.y] = '=';
						}
						frog.x--;
						map[frog.x][frog.y] = '0' ;
					}
					break;
				}

				case 's':
				case 'S':{
					if (frog.x != ROW){
						if ( check_status(frog.x+1,frog.y) == 1 || check_status(frog.x+1,frog.y) == -1){
							frog.x++; 
							ISOVER =1;
						}
						else {
							if (frog.x == ROW){
								map[frog.x][frog.y] = '|';
							}
							else{
								map[frog.x][frog.y] = '=';
							}
							frog.x++;
							map[frog.x][frog.y] = '0' ;
						}
					}
					break;
				}

				case 'a':
				case 'A':{
					if ( check_status(frog.x,frog.y-1) == 1 || check_status(frog.x,frog.y-1) == -1){
						frog.y--;
						ISOVER =1;
					} 
					else{
						if (frog.x == ROW){
							map[frog.x][frog.y] = '|';
						}
						else{
							map[frog.x][frog.y] = '=';
						}
						frog.y--;
						map[frog.x][frog.y] = '0' ;
					}
					break;
				}

				case 'd':
				case 'D':{
					if ( check_status(frog.x,frog.y+1) == 1 || check_status(frog.x,frog.y+1) == -1) {
						frog.y++; 
						ISOVER =1;
					}
					else{
						if (frog.x == ROW){
							map[frog.x][frog.y] = '|';
						}
						else{
							map[frog.x][frog.y] = '=';
						}
						frog.y++;
						map[frog.x][frog.y] = '0' ;
					}
					break;
				}

				case 'q':
				case 'Q':{
					ISQUIT = 1;
					ISOVER = 1;
				}
			}
		}

		/* after move the frog and log, update and display the new board */
		puts("\033[H\033[2J");
        for(int i = 0; i <= ROW; ++i){
            puts( map[i] );
        }

		pthread_mutex_unlock(&frog_mutex);
	}
	pthread_exit(NULL);
}


void *logs_move( void *t ){

	/* store the id of each thread */
	int* my_id = (int*)t;

	// printf("*t: %d\n", *my_id);

	while (!ISOVER){
		
		/* set a latency for each log to avoid confliction */
		usleep(100000);
		pthread_mutex_lock(&frog_mutex);

		// move right 
		if ((*my_id + 1) % 2 != 0){
            for (int j = 0; j < COLUMN - 1; j++){
                if (map[*my_id + 1][j] == '=') {
                    map[*my_id + 1][(j + 48) % (COLUMN - 1) ] = '=';
                    map[*my_id + 1][j] = ' ';
                }
                else if(map[*my_id + 1][j] == '0'){
                    map[*my_id + 1][(j + 48) % (COLUMN - 1) ] = '0';
                    map[*my_id + 1][j] = ' ';
                    frog.y --;
                    if (frog.y >= COLUMN - 1 || frog.y < 0) ISOVER = 1;
                }
            }
        }

		// move left
        else{
            for (int j = COLUMN - 2; j > -1; j--){
                if (map[*my_id + 1][j] == '=') {
                    map[*my_id + 1][(j + 1) % (COLUMN - 1)] = '=';
                    map[*my_id + 1][j] = ' ';
                }
                else if(map[*my_id + 1][j] == '0'){
                    map[*my_id + 1][(j + 1) % (COLUMN - 1)] = '0';
                    map[*my_id + 1][j] = ' ';
                    frog.y++;
                    if (frog.y >= COLUMN - 1 || frog.y < 0) ISOVER = 1;
                }
            }
        }

		thread_count += 1;

		/* after 10 thread move, send a signal to move the frog */
        if (thread_count == 9) pthread_cond_signal(&frog_threshold_cv);
        pthread_mutex_unlock(&frog_mutex);
	}
	pthread_exit(NULL);

	
}

int main( int argc, char *argv[] ){

	/* Initialize the river map and frog's starting position */
	memset( map , 0, sizeof( map ) ) ;
	int i , j ; 
	for( i = 1; i < ROW; ++i ){	
		for( j = 0; j < COLUMN - 1; ++j )	
			map[i][j] = ' ' ;  
	}	

	for( j = 0; j < COLUMN - 1; ++j )	
		map[ROW][j] = map[0][j] = '|' ;

	for( j = 0; j < COLUMN - 1; ++j )	
		map[0][j] = map[0][j] = '|' ;

	frog = Node( ROW, (COLUMN-1) / 2 ) ; 
	map[frog.x][frog.y] = '0' ; 

	/* initialize the log */
	initialize_the_log();

	/* Print the map into screen */
	for( i = 0; i <= ROW; ++i)	
		puts( map[i] );


	/*  Create pthreads for wood move and frog control.  */

	pthread_t threads[10];
	pthread_attr_t attr;

	pthread_mutex_init(&frog_mutex,NULL);
	pthread_cond_init(&frog_threshold_cv,NULL);

	pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr,PTHREAD_CREATE_JOINABLE);

	/* create 9 thread, each thread control a log */
	for (int i = 0; i < 9; i++){
		pthread_create(&threads[i],&attr,logs_move,(void*)&thread_ids[i]);
	}

	/* create a thread for frog control */
	pthread_create(&threads[9],&attr,frog_move,(void*)&thread_ids[9]);


	/* join 10 thread for synchronization */
	for (int i=0; i< 10; i++){
        pthread_join(threads[i],NULL);
    }


	/* clear the terminal */
	puts("\033[H\033[2J");

	/*  Display the output for user: win, lose or quit.  */
	if (ISQUIT){
		printf("You Exit The Game.\n");
	}
    else if (check_status(frog.x,frog.y) == 1){
		printf("You Win The Game!\n");
    }
    else{
       printf("You Lose The Game.\n");
    }

	/* destroy the mutex, cv and attr */
	pthread_attr_destroy(&attr);
    pthread_mutex_destroy(&frog_mutex);
    pthread_cond_destroy(&frog_threshold_cv);

	return 0;
	

}
