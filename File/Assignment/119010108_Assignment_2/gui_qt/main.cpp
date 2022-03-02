#include <QApplication>
 #include <QLabel>
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


#include <QApplication>
#include <QMainWindow>
#include <QWidget>
#include <QLabel>
#include <QKeyEvent>
#include<QTime>

#define ROW 10
#define COLUMN 50

void initialize_the_log();
void qrefresh();

pthread_mutex_t frog_mutex;
pthread_cond_t frog_threshold_cv;
int thread_ids[10] = {0,1,2,3,4,5,6,7,8,9};
int thread_count = 0;

int ISOVER = 0;
int ISQUIT = 0;

char map[ROW+10][COLUMN] ;

QWidget * qtmap[ROW+10][COLUMN];


struct Node{
    int x , y;
    Node( int _x , int _y ) : x( _x ) , y( _y ) {};
    Node(){} ;
} frog ;


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

void q_sleep(unsigned int msec){
    QTime reachtime = QTime::currentTime().addMSecs(msec);
    while (QTime::currentTime() < reachtime)
        QCoreApplication::processEvents();
}


void qrefresh(){

    for (int i = 0; i < 11; i++){
        for (int j = 0; j < 49; ++j){

            if (map[i][j] == '|') qtmap[i][j]->setStyleSheet("background-color: white");
            else if (map[i][j] == '0') qtmap[i][j]->setStyleSheet("background-color: rgb(0, 200, 0)");
            else if (map[i][j] == '=') qtmap[i][j]->setStyleSheet("background-color: rgb(150, 75, 0)");
            usleep(1000);
        }
    }


}

void *logs_move( void *t ){

    /* store the id of each thread */
    int* my_id = (int*)t;


    while (!ISOVER){

        /* set a latency for each log to avoid confliction */
        usleep(10000);
        pthread_mutex_lock(&frog_mutex);
        printf("i am : %d\n", *my_id);

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

void *frog_move( void * t){

    while (!ISOVER){
        pthread_mutex_lock(&frog_mutex);

        /* wait for every log move */
        while (thread_count != 9){
            printf("cnt: %d \n",thread_count);
            pthread_cond_wait(&frog_threshold_cv,&frog_mutex);
        }
        printf("i am here \n");
        thread_count = 0;

        qrefresh();
        pthread_mutex_unlock(&frog_mutex);
    }
    pthread_exit(NULL);
}


 int main(int argc, char *argv[]){

     /* Initialize the river map and frog's starting position */
     memset( map , 0, sizeof( map ) ) ;
     memset(qtmap,0,sizeof(qtmap));

     QApplication app(argc, argv);
     QMainWindow window;

     window.setFixedSize(490,110);
     QWidget *widget = new QWidget(&window);
     window.setCentralWidget(widget);
     QPalette pal(widget->palette());
     pal.setColor(QPalette::Background, QColor(193,210,240));
     widget->setAutoFillBackground(true);
     widget->setPalette(pal);
//     widget->show();


     int i , j ;
     for( i = 1; i < ROW; ++i ){
         for( j = 0; j < COLUMN - 1; ++j ){
              map[i][j] = ' ' ;
              qtmap[i][j] = new QWidget(widget);
              qtmap[i][j]->setGeometry(10*j,10*i,10,10);
         }
      }

     for( j = 0; j < COLUMN - 1; ++j ){
         map[ROW][j] = '|' ;
         qtmap[ROW][j] = new QWidget(widget);
         qtmap[ROW][j]->setGeometry(10*j,100,10,10);
     }


     for( j = 0; j < COLUMN - 1; ++j ){
         map[0][j] = '|' ;
         qtmap[0][j] = new QWidget(widget);
         qtmap[0][j]->setGeometry(10*j,0,10,10);
     }


     frog = Node( ROW, (COLUMN-1) / 2 ) ;
     map[frog.x][frog.y] = '0' ;

     qtmap[frog.x][frog.y] =  new QWidget(widget);
     qtmap[frog.x][frog.y]->setStyleSheet("background-color: rgb(0, 200, 0)");
     qtmap[frog.x][frog.y]->setGeometry(frog.y*10,frog.x*10,10,10);

         /* initialize the log */
     initialize_the_log();

     for( i = 0; i <= ROW; ++i)
             puts( map[i] );
     qrefresh();

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

    window.show();

     return app.exec();

 }
