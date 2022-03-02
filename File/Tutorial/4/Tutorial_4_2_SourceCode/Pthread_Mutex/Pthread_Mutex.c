#include <stdio.h>
#include <pthread.h>
#include <unistd.h> 

pthread_mutex_t mutex;

void printer(char *str){
    
    pthread_mutex_lock(&mutex);
    while(*str!='\0'){
        putchar(*str);
        fflush(stdout);
        str++;
        sleep(1);
    }
    printf("\n");
    pthread_mutex_unlock(&mutex);
}

void *thread_fun_1(void *arg){
    char *str = "hello";
    printer(str);
    pthread_exit(NULL);
}

void *thread_fun_2(void *arg){
    char *str = "world";
    printer(str);
    pthread_exit(NULL);
}

int main(void){
    pthread_t tid1, tid2;
    pthread_mutex_init(&mutex, NULL);
    
    pthread_create(&tid1, NULL, thread_fun_1, NULL);
    pthread_create(&tid2, NULL, thread_fun_2, NULL);
    
    pthread_join(tid1, NULL);
    pthread_join(tid2, NULL);
    

    pthread_mutex_destroy(&mutex);
    pthread_exit(NULL);
    
    return 0;
}
