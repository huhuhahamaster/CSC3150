#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<fcntl.h>
#include<sys/ioctl.h>
#include<unistd.h>


void arithmetic(int fd)
{

    int ret;


    /******************Blocking IO******************/
    printf("Blocking I/O\n");
    ret = 1;
    write(fd, &ret, sizeof(ret));	

    //Do not need to synchronize

    printf("Blocking I/O completed!\n");
    /***********************************************/



    /****************Non-Blocking IO****************/
    printf("Non-Blocking I/O\n");
    ret = 0;
    write(fd, &ret, sizeof(ret));

    //Can do something here
    //But cannot confirm computation completed

    printf("Non-Blocking I/O is still executing in kernel!\n");
    /***********************************************/

}

int main()
{
    printf("...............Start...............\n");

    //open my char device:
    int fd = open("/dev/mydev", O_RDWR);
    if(fd == -1) {
        printf("can't open device!\n");
        return -1;
    }


    arithmetic(fd);


    printf("...............End...............\n");

    return 0;
}
