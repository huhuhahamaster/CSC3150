#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>

static int *process_tree;
static int *status_tree;
static int *status_count;

static void sig_sigchld_handle(int sig);
void myFork(char **argv, int leftFileNum, int processTree, int start);

int main(int argc, char **argv)
{
    int leftFileNum = argc - 1;
    if (leftFileNum == 0){
        printf("the process tree: %d\n", getpid());
        printf("MyFork process(pid=%d) execute normally\n", getpid());

        return 0;
    }
    process_tree = mmap(NULL, sizeof(int)*(argc), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    status_tree = mmap(NULL, sizeof(int)*(argc), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    status_count = mmap(NULL, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

    *status_count = 0;

    signal(SIGCHLD, sig_sigchld_handle);

    myFork(argv, leftFileNum, 0, 1);
    *(process_tree) = getpid();

    // print the process tree
    int i;

    printf("the process tree: ");
    for (i = 0; i < leftFileNum; ++i){
        printf("%d->", *(process_tree+i));
    }
    printf("%d\n", *(process_tree+i));

    // scan the status for all process
    
    int j;
    int current_pid, father_pid = *(process_tree+i);

    for (j = 0; j < leftFileNum; ++j){
        current_pid = father_pid;
        --i;
        father_pid = *(process_tree+i);

        int status = *(status_tree+j);
        if (WIFEXITED(status)){
            printf("The child process(pid=%d) of parent process(pid=%d) has normal execution\n", current_pid, father_pid);
            printf("Its exit status = %d\n", WEXITSTATUS(status));
        }
        else if (WIFSTOPPED(status)){
            printf("The child process(pid=%d) of parent process(pid=%d) is terminated by signal\n", current_pid, father_pid);
            printf("Its signal number = %d\n", WSTOPSIG(status));
            switch (WSTOPSIG(status))
            {
                case SIGSTOP:
                    printf("child process get SIGSTOP signal\n");
                    break;

                case SIGTSTP:
                    printf("child process get SIGTSTP signal\n");
                    break;

                case SIGTTIN:
                    printf("child process get SIGTTIN signal\n");
                    break;

                case SIGTTOU:
                    printf("child process get SIGTTOU signal\n");
                    break;
                default:
                    break;
            }
            printf("child process stopped\n");
        }
        else if (WIFSIGNALED(status)){
            printf("The child process(pid=%d) of parent process(pid=%d) is terminated by signal\n", current_pid, father_pid);
            printf("Its signal number = %d\n", WTERMSIG(status));
            switch (WTERMSIG(status))
            {
                case SIGABRT:
                    printf("child process get SIGABRT signal\n");
                    printf("child process is abort by abort signal\n");
                    break;

                case SIGALRM:
                    printf("child process get SIGALRM signal\n");
                    printf("child process is abort by alarm signal\n");
                    break;

                case SIGBUS:
                    printf("child process get SIGBUS signal\n");
                    printf("child process is abort by BUS signal\n");
                    break;

                case SIGFPE:
                    printf("child process get SIGFPE signal\n");
                    printf("child process is abort by floating point exception signal\n");
                    break;

                case SIGHUP:
                    printf("child process get SIGHUP signal\n");
                    printf("child process is abort by HangUp signal\n");
                    break;

                case SIGILL:
                    printf("child process get SIGILL signal\n");
                    printf("child process is abort by illegal instruction signal\n");
                    break;

                case SIGINT:
                    printf("child process get SIGINT signal\n");
                    printf("child process is abort by interrupt signal\n");
                    break;
                
                case SIGIO:
                    printf("child process get SIGIO signal\n");
                    printf("child process is abort by asynchronous IO signal\n");
                    break;

                case SIGKILL:
                    printf("child process get SIGKILL signal\n");
                    printf("child process is abort by kill signal\n");
                    break;

                case SIGPIPE:
                    printf("child process get SIGPIPE signal\n");
                    printf("child process is abort by broken pipe signal\n");
                    break;

                case SIGPROF:
                    printf("child process get SIGPROF signal\n");
                    printf("child process is abort by setitimer signal\n");
                    break;

                case SIGQUIT:
                    printf("child process get SIGQUIT signal\n");
                    printf("child process is abort by terminal quit signal\n");
                    break;

                case SIGSEGV:
                    printf("child process get SIGSEGV signal\n");
                    printf("child process is abort by segmentation violation signal\n");
                    break;

                case SIGSYS:
                    printf("child process get SIGSYS signal\n");
                    printf("child process is abort by system call violation signal\n");
                    break;

                case SIGTERM:
                    printf("child process get SIGTERM signal\n");
                    printf("child process is abort by termination signal\n");
                    break;

                case SIGTRAP:
                    printf("child process get SIGTRAP signal\n");
                    printf("child process is abort by trap signal\n");
                    break;

                case SIGVTALRM:
                    printf("child process get SIGVTALRM signal\n");
                    printf("child process is abort by setitimer signal\n");
                    break;

                case SIGXCPU:
                    printf("child process get SIGXCPU signal\n");
                    printf("child process is abort by CPU limit signal\n");
                    break;

                case SIGXFSZ:
                    printf("child process get SIGXFSZ signal\n");
                    printf("child process is abort by file limit signal\n");
                    break;

                default:
                    break;
            }
        }

        printf("\n");
    }

    printf("MyFork process(pid=%d) execute normally\n", getpid());

    return 0;
}

void myFork(char **argv, int leftFileNum, int count, int start)
{
    signal(SIGCHLD, sig_sigchld_handle);

    if (leftFileNum != 0){
        pid_t pid;
        pid = fork();

        if (pid == 0){
            myFork(argv+1, leftFileNum-1, count+1, 0);
        }
        else{
            pause();
            *(process_tree+count+1) = pid;

            if (start == 0){
                execve(argv[0], NULL, NULL);
                exit(EXIT_FAILURE);
            }
            else{
                return ;
            }
        }
    }
    else{
        execve(argv[0], NULL, NULL);
        exit(EXIT_FAILURE);
    }
}

static void sig_sigchld_handle(int sig)
{
    static int i = 0;
    int status;
	waitpid(-1, &status, 0 | WUNTRACED);

    *(status_tree+*(status_count)) = status;
    *(status_count) = *(status_count) + 1;
}