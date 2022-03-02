#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <stdio.h>
#include <stdlib.h>


int main(int argc, char *argv[]){
    
    pid_t pid;
    
    printf("Process start to fork\n");
    pid=fork();
    
    if(pid==-1){
        perror("fork");
        exit(1);
    }
    else{
        
        //Child process
        if(pid==0){
            printf("I'm the Child Process:\n");
            sleep(10);
            printf("\t My pid is:%d.   My ppid is:%d\n", getpid(), getppid());
            exit(0);
        }
        
        //Parent process
        else{
            sleep(3);
            printf("I'm the Parent Process:\n");
            printf("\t My pid is:%d\n", getpid());
            exit(0);
        }
    }
    
    return 0;

}