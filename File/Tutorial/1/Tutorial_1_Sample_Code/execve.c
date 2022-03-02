#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/wait.h>


int main (int argc, char *argv[]) {
    
    int state;
    pid_t pid = fork();
    
    if (pid < 0) {
        printf ("Fork error!\n");
    }
    else {
        
        //Child process
        if (pid == 0) {
            
            int i;
            char *arg[argc];
            
            printf("This is child process.\n");
            
            for(i=0;i<argc-1;i++){
                arg[i]=argv[i+1];
            }
            arg[argc-1]=NULL;
            
            
	        printf("Child process id is %d\n", getpid());
            printf("Child process start to execute test program:\n");
            execve(arg[0],arg,NULL);
            
            printf("Continue to run original child process!\n");
            
            perror("execve");
            exit(EXIT_FAILURE);
        }
        
        //Parent process
        else{
            
            wait(&state);
            printf("This is farther process.\n");
            exit(1);
        }
    }
    
    return 0;
}
