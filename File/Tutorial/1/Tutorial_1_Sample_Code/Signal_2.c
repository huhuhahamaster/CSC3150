#include <stdio.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/wait.h>


int main(int argc, char *argv[]){

    pid_t pid;
    int status;

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
            printf("I'm raising SIGSTOP signal!\n\n");
            raise(SIGSTOP);
        }
    
        //Parent process
        else{
            waitpid(pid, &status, WUNTRACED);
            printf("Parent process receives the signal\n");
            
            if(WIFEXITED(status)){
                printf("Normal termination with EXIT STATUS = %d\n",WEXITSTATUS(status));
            }
            else if(WIFSIGNALED(status)){
                printf("CHILD EXECUTION FAILED: %d\n", WTERMSIG(status));
            }
            else if(WIFSTOPPED(status)){
                printf("CHILD PROCESS STOPPED: %d\n", WSTOPSIG(status));
            }
            else{
                printf("CHILD PROCESS CONTINUED\n");
            }
            exit(0);
        }
    }

    return 0;
}




