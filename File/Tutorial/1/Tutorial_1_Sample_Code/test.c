#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>

int main (void) {

    printf("Test process id is %d\n.",getpid());
    printf("\tTest completed!\n");
    printf("\tExit child process!\n");
    return 0;
}
