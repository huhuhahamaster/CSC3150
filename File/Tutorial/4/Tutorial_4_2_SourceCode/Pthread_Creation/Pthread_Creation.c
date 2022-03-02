#include <pthread.h>
#include <stdio.h> 
#include <stdlib.h>
#include <unistd.h>

#define NUM_THREAD 5 

void *print_hello(void *threadid){
    long tid;
    tid = (long)threadid;

    printf("Hello world! thread %ld\n", tid);
    pthread_exit(NULL);
}

int main(){

    pthread_t threads[NUM_THREAD];
    int rc;
    long i;

    for(i =0; i<NUM_THREAD; i++){
        printf("In main: create thread %ld\n", i);
        rc = pthread_create(&threads[i], NULL, print_hello, (void*)i);
        if(rc){
            printf("ERROR: return code from pthread_create() is %d", rc);
            exit(1);
        }
    }
    
    
    pthread_exit(NULL);
    return 0;
}
