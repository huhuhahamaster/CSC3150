#include<stdlib.h>
#include<stdio.h>
#include<unistd.h>
#include<pthread.h>


int sum;

void * add1(void *cnt)
{
    for(int i=0; i < 5; i++)
    {
        sum += i;
    }
    pthread_exit(NULL);
    return 0;
}
void * add2(void *cnt)
{

    for(int i=5; i<10; i++)
    {
        sum += i;
    }
    pthread_exit(NULL);
    return 0;
}

int main(void)
{
    pthread_t ptid1, ptid2;
    sum=0;
    
    pthread_create(&ptid1, NULL, add1, &sum);
    pthread_create(&ptid2, NULL, add2, &sum);
    
    //pthread_join(ptid1,NULL);
    //pthread_join(ptid2,NULL);
    
    printf("sum %d\n", sum);
    pthread_exit(NULL);

    
    return 0;
}
