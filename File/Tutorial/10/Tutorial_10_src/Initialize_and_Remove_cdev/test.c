#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<fcntl.h>
#include<sys/ioctl.h>


int main()
{
    printf("...............Start...............\n");

    //open my char device:
    int fd = open("/dev/mydev", O_RDWR);
    if(fd == -1) {
        printf("can't open device!\n");
        return -1;
    }
  

    printf("...............End...............\n");

    return 0;
}




