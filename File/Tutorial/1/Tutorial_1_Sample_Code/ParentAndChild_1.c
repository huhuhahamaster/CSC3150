#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>


int main(int argc, char *argv[]){
    
    char buf[50] = "Original test strings";    
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
            strcpy(buf, "Test strings are updated by child.");
            printf("I'm the Child Process: %s\n", buf);
            exit(0);
        }
        
        //Parent process
        else{
            sleep(3);
            printf("I'm the Parent Process: %s\n", buf);
            exit(0);
        }
    }
    
    return 0;
}
