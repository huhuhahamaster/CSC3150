#include <pthread.h>
#include <stdio.h> 
#include <stdlib.h>
#include <unistd.h>


void *print_hello(void *threadid){
 
    sleep(2);
    printf("Hello world!\n");
    pthread_exit(NULL);
}

int main(){
    pthread_t thread;
    int rc;
    void* i;

    printf("In main: create thread\n");
    rc = pthread_create(&thread, NULL, print_hello, i);
    
    if(rc){
        printf("ERROR: return code from pthread_create() is %d", rc);
        exit(1);
    }
    
    printf("Main thread exits!\n");
    //pthread_exit(NULL);
    
    return 0;
}
